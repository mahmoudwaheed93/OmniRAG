"""Reranker interface, adapters, and factory.

Reranking is the highest-leverage retrieval upgrade after hybrid search: the
vector+BM25 stage optimises recall over a wide candidate set, and the reranker
optimises precision over the handful of passages actually shown to the LLM.
"""

from __future__ import annotations

import json
import math
import re
import threading
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from backend.models import (
    ProviderError,
    RerankError,
    Role,
)
from backend.providers.llm import (
    BaseLLMProvider,
    LLMMessage,
    llm_operation,
    post_json,
    retry_call,
)
from backend.utils import (
    get_logger,
    normalize_for_search,
    tokenize,
    truncate,
)

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Base interface
# --------------------------------------------------------------------------- #
@dataclass
class RerankCandidate:
    """A passage to score. ``ref`` links back to the originating chunk."""

    ref: str
    text: str


@dataclass
class RerankScore:
    ref: str
    score: float
    rank: int = 0


class BaseReranker(ABC):
    name: str = "base"
    is_model_based: bool = True

    def __init__(self, *, model: str = "", top_n: int = 8):
        self.model = model
        self.top_n = top_n

    @abstractmethod
    def rerank(
        self, query: str, candidates: Sequence[RerankCandidate], *, top_n: int | None = None
    ) -> List[RerankScore]:
        pass

    def describe(self) -> dict:
        return {"provider": self.name, "model": self.model, "model_based": self.is_model_based}


# --------------------------------------------------------------------------- #
# Heuristic reranker
# --------------------------------------------------------------------------- #
_NUMERIC = set("0123456789٠١٢٣٤٥٦٧٨٩")


class HeuristicReranker(BaseReranker):
    name = "heuristic"
    is_model_based = False

    def __init__(self, *, top_n: int = 8):
        super().__init__(model="lexical-heuristic", top_n=top_n)

    def rerank(
        self, query: str, candidates: Sequence[RerankCandidate], *, top_n: int | None = None
    ) -> List[RerankScore]:
        if not candidates:
            return []

        query_terms = [t for t in tokenize(query) if len(t) > 1]
        if not query_terms:
            return [
                RerankScore(ref=c.ref, score=1.0 / (rank + 1), rank=rank)
                for rank, c in enumerate(candidates[: top_n or self.top_n])
            ]

        doc_tokens = [tokenize(c.text) for c in candidates]
        idf = self._idf(query_terms, doc_tokens)
        phrase = normalize_for_search(query)

        scored: List[tuple[float, str]] = []
        for candidate, tokens in zip(candidates, doc_tokens):
            scored.append(
                (self._score(query_terms, tokens, candidate, phrase, idf), candidate.ref)
            )

        scored.sort(key=lambda item: item[0], reverse=True)
        limit = min(top_n or self.top_n, len(scored))
        return [
            RerankScore(ref=ref, score=score, rank=rank)
            for rank, (score, ref) in enumerate(scored[:limit])
        ]

    @staticmethod
    def _idf(query_terms: Sequence[str], doc_tokens: Sequence[List[str]]) -> Dict[str, float]:
        total = max(1, len(doc_tokens))
        sets = [set(tokens) for tokens in doc_tokens]
        idf: Dict[str, float] = {}
        for term in set(query_terms):
            hits = sum(1 for token_set in sets if term in token_set)
            idf[term] = math.log(1.0 + (total - hits + 0.5) / (hits + 0.5))
        return idf

    def _score(
        self,
        query_terms: Sequence[str],
        tokens: Sequence[str],
        candidate: RerankCandidate,
        phrase: str,
        idf: Dict[str, float],
    ) -> float:
        if not tokens:
            return 0.0

        counts = Counter(tokens)
        length_norm = 1.0 / (1.0 + math.log(1.0 + len(tokens) / 120.0))

        overlap = 0.0
        matched = 0
        for term in set(query_terms):
            count = counts.get(term, 0)
            if count:
                matched += 1
                overlap += idf.get(term, 1.0) * (1.0 + math.log(count))

        if matched == 0:
            return 0.0

        coverage = matched / len(set(query_terms))
        score = overlap * length_norm * (0.5 + 0.5 * coverage)

        text_normalized = normalize_for_search(candidate.text)
        if len(phrase) > 8 and phrase in text_normalized:
            score *= 1.6

        score *= 1.0 + 0.3 * self._proximity(query_terms, tokens)

        query_numbers = {t for t in query_terms if _is_numeric(t)}
        if query_numbers:
            hits = len(query_numbers & set(tokens))
            score *= 1.0 + 0.4 * (hits / len(query_numbers))

        return score

    @staticmethod
    def _proximity(query_terms: Sequence[str], tokens: Sequence[str]) -> float:
        wanted = set(query_terms)
        positions = [i for i, token in enumerate(tokens) if token in wanted]
        if len(positions) < 2:
            return 0.0
        span = positions[-1] - positions[0] + 1
        return min(1.0, len(positions) / max(1, span))


def _is_numeric(token: str) -> bool:
    return any(ch in _NUMERIC for ch in token)


# --------------------------------------------------------------------------- #
# Cohere reranker
# --------------------------------------------------------------------------- #
class CohereReranker(BaseReranker):
    name = "cohere"

    DEFAULT_BASE_URL = "https://api.cohere.com/v1"
    DEFAULT_MODEL = "rerank-multilingual-v3.0"
    MAX_DOC_CHARS = 4000

    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = "",
        top_n: int = 8,
        timeout_s: float = 30.0,
    ):
        super().__init__(model=model or self.DEFAULT_MODEL, top_n=top_n)
        self.api_key = api_key
        self.base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.timeout_s = timeout_s

    def rerank(
        self, query: str, candidates: Sequence[RerankCandidate], *, top_n: int | None = None
    ) -> List[RerankScore]:
        if not candidates:
            return []

        limit = min(top_n or self.top_n, len(candidates))
        payload: Dict[str, Any] = {
            "model": self.model,
            "query": query,
            "documents": [c.text[:self.MAX_DOC_CHARS] for c in candidates],
            "top_n": limit,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        body = retry_call(
            lambda: post_json(
                f"{self.base_url}/rerank",
                payload,
                headers=headers,
                timeout_s=self.timeout_s,
                provider="rerank",
            ),
            attempts=2,
            operation=f"cohere/rerank ({self.model})",
        )

        results = body.get("results")
        if not isinstance(results, list):
            raise RerankError(
                f"Unexpected Cohere rerank payload: {str(body)[:200]}", provider=self.name
            )

        scores: List[RerankScore] = []
        for rank, item in enumerate(results):
            index = item.get("index")
            if not isinstance(index, int) or not 0 <= index < len(candidates):
                continue
            scores.append(
                RerankScore(
                    ref=candidates[index].ref,
                    score=float(item.get("relevance_score", 0.0)),
                    rank=rank,
                )
            )
        return scores


# --------------------------------------------------------------------------- #
# Jina reranker
# --------------------------------------------------------------------------- #
class JinaReranker(BaseReranker):
    name = "jina"

    DEFAULT_BASE_URL = "https://api.jina.ai/v1"
    DEFAULT_MODEL = "jina-reranker-v2-base-multilingual"
    MAX_DOC_CHARS = 4000

    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = "",
        top_n: int = 8,
        timeout_s: float = 30.0,
    ):
        super().__init__(model=model or self.DEFAULT_MODEL, top_n=top_n)
        self.api_key = api_key
        self.base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.timeout_s = timeout_s

    def rerank(
        self, query: str, candidates: Sequence[RerankCandidate], *, top_n: int | None = None
    ) -> List[RerankScore]:
        if not candidates:
            return []

        limit = min(top_n or self.top_n, len(candidates))
        payload: Dict[str, Any] = {
            "model": self.model,
            "query": query,
            "documents": [c.text[:self.MAX_DOC_CHARS] for c in candidates],
            "top_n": limit,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        body = retry_call(
            lambda: post_json(
                f"{self.base_url}/rerank",
                payload,
                headers=headers,
                timeout_s=self.timeout_s,
                provider="rerank",
            ),
            attempts=2,
            operation=f"jina/rerank ({self.model})",
        )

        results = body.get("results")
        if not isinstance(results, list):
            raise RerankError(
                f"Unexpected Jina rerank payload: {str(body)[:200]}", provider=self.name
            )

        scores: List[RerankScore] = []
        for rank, item in enumerate(results):
            index = item.get("index")
            if not isinstance(index, int) or not 0 <= index < len(candidates):
                continue
            scores.append(
                RerankScore(
                    ref=candidates[index].ref,
                    score=float(item.get("relevance_score", 0.0)),
                    rank=rank,
                )
            )
        return scores


# --------------------------------------------------------------------------- #
# LLM-as-reranker
# --------------------------------------------------------------------------- #
MAX_DOC_CHARS_LLM = 900
MAX_CANDIDATES = 30

SYSTEM_PROMPT = """You score how well each numbered passage answers a user's question.

Rules:
- Judge only relevance to the question, never writing quality or length.
- A passage describing a chart, diagram or table is relevant if the question asks about that visual.
- Questions and passages may be in Arabic, English, or both. Judge across languages.
- Respond with a single JSON object: {"scores": [{"id": <int>, "score": <0.0-1.0>}, ...]}
- Include every passage id exactly once. No prose, no markdown."""


class LLMReranker(BaseReranker):
    name = "llm"

    def __init__(
        self,
        llm: BaseLLMProvider,
        *,
        top_n: int = 8,
        max_candidates: int = MAX_CANDIDATES,
        max_output_tokens: int = 256,
    ):
        super().__init__(model=getattr(llm, "model", "llm"), top_n=top_n)
        self.llm = llm
        self.max_candidates = max_candidates
        self.max_output_tokens = max_output_tokens

    def rerank(
        self, query: str, candidates: Sequence[RerankCandidate], *, top_n: int | None = None
    ) -> List[RerankScore]:
        if not candidates:
            return []

        window = list(candidates[: self.max_candidates])
        prompt_parts = [f"Question: {query}", "", "Passages:"]
        for index, candidate in enumerate(window):
            prompt_parts.append(f"[{index}] {truncate(candidate.text, MAX_DOC_CHARS_LLM)}")
        prompt = "\n".join(prompt_parts)

        try:
            with llm_operation("rerank"):
                response = self.llm.complete(
                    [LLMMessage(role=Role.USER, text=prompt)],
                    system=SYSTEM_PROMPT,
                    temperature=0.0,
                    max_output_tokens=self.max_output_tokens,
                    json_mode=True,
                )
        except ProviderError as exc:
            raise RerankError(
                f"LLM reranking failed: {exc}", provider=self.name, retryable=False
            ) from exc

        scores = _parse_scores(response.text, len(window))
        if not scores:
            raise RerankError("LLM reranker returned no usable scores", provider=self.name)

        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        limit = min(top_n or self.top_n, len(ordered))
        return [
            RerankScore(ref=window[index].ref, score=score, rank=rank)
            for rank, (index, score) in enumerate(ordered[:limit])
        ]


def _parse_scores(text: str, count: int) -> Dict[int, float]:
    """Tolerant JSON extraction."""
    payload = text.strip()
    if not payload:
        return {}

    match = re.search(r"\{.*\}", payload, re.DOTALL)
    if match:
        payload = match.group(0)

    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return {}

    items = data.get("scores") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return {}

    out: Dict[int, float] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("id"))
            score = float(item.get("score", 0.0))
        except (TypeError, ValueError):
            continue
        if 0 <= index < count:
            out[index] = max(0.0, min(1.0, score))
    return out


# --------------------------------------------------------------------------- #
# Factory and cache
# --------------------------------------------------------------------------- #
_rerank_cache: Dict[str, BaseReranker] = {}
_rerank_lock = threading.Lock()

SUPPORTED_PROVIDERS = ("auto", "cohere", "jina", "llm", "heuristic", "none")


def build_reranker(
    cfg: Any, *, top_n: int = 8, settings: Any = None
) -> BaseReranker:
    """Build the best reranker available for the current configuration."""
    provider = (cfg.provider or "auto").lower()

    if not cfg.enabled or provider == "none":
        return HeuristicReranker(top_n=top_n)

    if provider == "heuristic":
        return HeuristicReranker(top_n=top_n)

    if provider in ("cohere", "jina") and not cfg.api_key:
        logger.warning(
            "RERANK_PROVIDER=%s but no RERANK_API_KEY is set — using the heuristic reranker",
            provider,
        )
        return HeuristicReranker(top_n=top_n)

    if provider == "cohere":
        return CohereReranker(
            api_key=cfg.api_key, model=cfg.model, base_url=cfg.base_url,
            top_n=top_n, timeout_s=cfg.timeout_s,
        )
    if provider == "jina":
        return JinaReranker(
            api_key=cfg.api_key, model=cfg.model, base_url=cfg.base_url,
            top_n=top_n, timeout_s=cfg.timeout_s,
        )

    if provider == "llm":
        llm = _try_llm(settings)
        if llm is not None:
            return LLMReranker(
                llm,
                top_n=top_n,
                max_output_tokens=(settings or _get_settings()).retrieval.rerank_max_output_tokens,
            )
        return HeuristicReranker(top_n=top_n)

    # --- auto ---------------------------------------------------------- #
    if cfg.api_key:
        model = (cfg.model or "").lower()
        if "jina" in model:
            return JinaReranker(
                api_key=cfg.api_key, model=cfg.model, base_url=cfg.base_url,
                top_n=top_n, timeout_s=cfg.timeout_s,
            )
        return CohereReranker(
            api_key=cfg.api_key, model=cfg.model, base_url=cfg.base_url,
            top_n=top_n, timeout_s=cfg.timeout_s,
        )

    llm = _try_llm(settings)
    if llm is not None:
        return LLMReranker(
            llm,
            top_n=top_n,
            max_output_tokens=(settings or _get_settings()).retrieval.rerank_max_output_tokens,
        )

    return HeuristicReranker(top_n=top_n)


def _get_settings():
    from backend.config import get_settings
    return get_settings()


def _try_llm(settings: Any = None):
    """Return the LLM router, or ``None`` when no usable LLM is configured."""
    try:
        from backend.providers.llm import get_llm_provider

        resolved = settings or _get_settings()
        if not resolved.llm.is_configured:
            return None
        if all(e.provider == "mock" for e in resolved.llm.configured_endpoints):
            return None
        return get_llm_provider(resolved)
    except Exception as exc:
        logger.info("LLM reranker unavailable (%s) — falling back to heuristic", exc)
        return None


def get_reranker(settings: Any = None) -> BaseReranker:
    resolved = settings or _get_settings()
    cfg = resolved.rerank
    key = f"{cfg.provider}|{cfg.model}|{len(cfg.api_key)}|{cfg.enabled}|{resolved.retrieval.rerank_top_k}"
    with _rerank_lock:
        reranker = _rerank_cache.get(key)
        if reranker is None:
            reranker = build_reranker(
                cfg, top_n=resolved.retrieval.rerank_top_k, settings=resolved
            )
            _rerank_cache[key] = reranker
            logger.info("Reranker ready: %s (%s)", reranker.name, reranker.model)
        return reranker


def reset_rerank_cache() -> None:
    with _rerank_lock:
        _rerank_cache.clear()


__all__ = [
    "RerankCandidate",
    "RerankScore",
    "BaseReranker",
    "HeuristicReranker",
    "CohereReranker",
    "JinaReranker",
    "LLMReranker",
    "SUPPORTED_PROVIDERS",
    "build_reranker",
    "get_reranker",
    "reset_rerank_cache",
]

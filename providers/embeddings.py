"""Embedding provider interface, adapters, and factory.

Deliberately independent of the LLM failover chain: an LLM outage must never
invalidate vectors already written to Qdrant, and swapping the answering model
must not force a re-index.
"""

from __future__ import annotations

import hashlib
import math
import threading
from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

from backend.models import (
    ConfigurationError,
    EmbeddingError,
    MissingCredentialError,
    ProviderAuthError,
    ProviderBadRequestError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    RateLimitError,
)
from backend.providers.llm import post_json, retry_call
from backend.utils import (
    get_logger,
    normalize_for_search,
    tokenize,
)

logger = get_logger(__name__)

Vector = List[float]


# --------------------------------------------------------------------------- #
# Base interface
# --------------------------------------------------------------------------- #
class BaseEmbeddingProvider(ABC):
    """Contract for every embedding backend."""

    name: str = "base"
    dimensions: int = 0
    supports_task_type: bool = False

    def __init__(self, *, model: str, batch_size: int = 64, max_chars: int = 8000):
        self.model = model
        self.batch_size = max(1, batch_size)
        self.max_chars = max_chars

    @abstractmethod
    def embed_batch(self, texts: Sequence[str], *, is_query: bool = False) -> List[Vector]:
        pass

    def embed_documents(self, texts: Sequence[str]) -> List[Vector]:
        return self._embed_all(texts, is_query=False)

    def embed_query(self, text: str) -> Vector:
        vectors = self._embed_all([text], is_query=True)
        return vectors[0] if vectors else []

    def _embed_all(self, texts: Sequence[str], *, is_query: bool) -> List[Vector]:
        prepared = [self._prepare(t) for t in texts]
        out: List[Vector] = []
        for start in range(0, len(prepared), self.batch_size):
            batch = prepared[start : start + self.batch_size]
            vectors = self.embed_batch(batch, is_query=is_query)
            if len(vectors) != len(batch):
                raise EmbeddingError(
                    f"{self.name} returned {len(vectors)} vectors for {len(batch)} inputs",
                    provider=self.name,
                )
            out.extend(vectors)
        if out and not self.dimensions:
            self.dimensions = len(out[0])
        return out

    def _prepare(self, text: str) -> str:
        cleaned = (text or "").strip()
        if not cleaned:
            return " "
        return cleaned[: self.max_chars]

    def describe(self) -> Dict[str, Any]:
        return {"provider": self.name, "model": self.model, "dimensions": self.dimensions}


# --------------------------------------------------------------------------- #
# Hashing adapter (offline fallback)
# --------------------------------------------------------------------------- #
DEFAULT_DIMENSIONS = 1024
_NGRAM_SIZE = 4


class HashingEmbeddings(BaseEmbeddingProvider):
    name = "hash"

    def __init__(
        self,
        *,
        model: str = "hash-1024",
        dimensions: int = DEFAULT_DIMENSIONS,
        batch_size: int = 256,
        max_chars: int = 8000,
        use_char_ngrams: bool = True,
        announce: bool = True,
        **_ignored: object,
    ):
        super().__init__(model=model, batch_size=batch_size, max_chars=max_chars)
        self.dimensions = dimensions or DEFAULT_DIMENSIONS
        self.use_char_ngrams = use_char_ngrams
        if announce:
            logger.warning(
                "Using the offline `hash` embedding provider — retrieval is lexical "
                "only and cross-lingual search will not work. Configure a real "
                "embedding provider for production."
            )

    def embed_batch(self, texts: Sequence[str], *, is_query: bool = False) -> List[Vector]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> Vector:
        vector = [0.0] * self.dimensions
        counts: Counter[str] = Counter(tokenize(text))

        if self.use_char_ngrams:
            normalized = normalize_for_search(text)
            for word in normalized.split():
                if len(word) <= _NGRAM_SIZE:
                    continue
                padded = f"^{word}$"
                for i in range(len(padded) - _NGRAM_SIZE + 1):
                    counts[f"#{padded[i : i + _NGRAM_SIZE]}"] += 1

        if not counts:
            return vector

        for term, count in counts.items():
            index, sign = self._hash(term)
            vector[index] += sign * (1.0 + math.log(count))

        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0.0:
            return vector
        return [v / norm for v in vector]

    def _hash(self, term: str) -> tuple[int, float]:
        digest = hashlib.blake2b(term.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        return value % self.dimensions, 1.0 if (value >> 63) & 1 else -1.0


# --------------------------------------------------------------------------- #
# OpenAI-compatible adapter
# --------------------------------------------------------------------------- #
class OpenAICompatibleEmbeddings(BaseEmbeddingProvider):
    name = "openai"

    DEFAULT_BASE_URL = "https://api.openai.com/v1"

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "text-embedding-3-large",
        base_url: str = "",
        dimensions: int = 0,
        batch_size: int = 64,
        timeout_s: float = 60.0,
        max_chars: int = 8000,
    ):
        super().__init__(model=model, batch_size=batch_size, max_chars=max_chars)
        self.api_key = api_key
        self.base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.timeout_s = timeout_s
        self.dimensions = dimensions

    def embed_batch(self, texts: Sequence[str], *, is_query: bool = False) -> List[Vector]:
        if not texts:
            return []

        payload: Dict[str, Any] = {"model": self.model, "input": list(texts)}
        if self.dimensions:
            payload["dimensions"] = self.dimensions

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        body = retry_call(
            lambda: post_json(
                f"{self.base_url}/embeddings",
                payload,
                headers=headers,
                timeout_s=self.timeout_s,
                provider="embeddings",
            ),
            attempts=3,
            operation=f"embeddings ({self.model})",
        )

        data = body.get("data")
        if not isinstance(data, list):
            raise EmbeddingError(
                f"Unexpected embeddings payload: {str(body)[:200]}", provider=self.name
            )

        ordered = sorted(data, key=lambda item: item.get("index", 0))
        vectors = [list(item.get("embedding") or []) for item in ordered]
        if any(not v for v in vectors):
            raise EmbeddingError("Embedding API returned an empty vector", provider=self.name)
        return vectors


# --------------------------------------------------------------------------- #
# Gemini adapter
# --------------------------------------------------------------------------- #
class GeminiEmbeddings(BaseEmbeddingProvider):
    name = "gemini"
    supports_task_type = True

    DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
    DEFAULT_MODEL = "gemini-embedding-001"
    MAX_BATCH = 100

    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = "",
        dimensions: int = 0,
        batch_size: int = 64,
        timeout_s: float = 60.0,
        max_chars: int = 8000,
    ):
        super().__init__(
            model=model or self.DEFAULT_MODEL,
            batch_size=min(max(1, batch_size), self.MAX_BATCH),
            max_chars=max_chars,
        )
        self.api_key = api_key
        self.base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.timeout_s = timeout_s
        self.dimensions = dimensions

    @property
    def _model_path(self) -> str:
        return self.model if self.model.startswith("models/") else f"models/{self.model}"

    def embed_batch(self, texts: Sequence[str], *, is_query: bool = False) -> List[Vector]:
        if not texts:
            return []

        task_type = "RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT"
        requests: List[Dict[str, Any]] = []
        for text in texts:
            request: Dict[str, Any] = {
                "model": self._model_path,
                "content": {"parts": [{"text": text}]},
                "taskType": task_type,
            }
            if self.dimensions:
                request["outputDimensionality"] = self.dimensions
            requests.append(request)

        url = f"{self.base_url}/{self._model_path}:batchEmbedContents"
        headers = {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}

        body = retry_call(
            lambda: post_json(
                url,
                {"requests": requests},
                headers=headers,
                timeout_s=self.timeout_s,
                provider="embeddings",
            ),
            attempts=3,
            operation=f"gemini/batchEmbedContents ({self.model})",
        )

        embeddings = body.get("embeddings")
        if not isinstance(embeddings, list):
            raise EmbeddingError(
                f"Unexpected Gemini embeddings payload: {str(body)[:200]}", provider=self.name
            )
        vectors = [list(item.get("values") or []) for item in embeddings]
        if any(not v for v in vectors):
            raise EmbeddingError("Gemini returned an empty vector", provider=self.name)
        return vectors


# --------------------------------------------------------------------------- #
# Cohere adapter
# --------------------------------------------------------------------------- #
class CohereEmbeddings(BaseEmbeddingProvider):
    name = "cohere"
    supports_task_type = True

    DEFAULT_BASE_URL = "https://api.cohere.com/v1"
    DEFAULT_MODEL = "embed-multilingual-v3.0"
    MAX_BATCH = 96

    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = "",
        dimensions: int = 0,
        batch_size: int = 64,
        timeout_s: float = 60.0,
        max_chars: int = 8000,
    ):
        super().__init__(
            model=model or self.DEFAULT_MODEL,
            batch_size=min(max(1, batch_size), self.MAX_BATCH),
            max_chars=max_chars,
        )
        self.api_key = api_key
        self.base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.timeout_s = timeout_s
        self.dimensions = dimensions

    def embed_batch(self, texts: Sequence[str], *, is_query: bool = False) -> List[Vector]:
        if not texts:
            return []

        payload: Dict[str, Any] = {
            "model": self.model,
            "texts": list(texts),
            "input_type": "search_query" if is_query else "search_document",
            "embedding_types": ["float"],
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        body = retry_call(
            lambda: post_json(
                f"{self.base_url}/embed",
                payload,
                headers=headers,
                timeout_s=self.timeout_s,
                provider="embeddings",
            ),
            attempts=3,
            operation=f"cohere/embed ({self.model})",
        )

        embeddings = body.get("embeddings")
        if isinstance(embeddings, dict):
            embeddings = embeddings.get("float")
        if not isinstance(embeddings, list) or not embeddings:
            raise EmbeddingError(
                f"Unexpected Cohere embeddings payload: {str(body)[:200]}", provider=self.name
            )
        return [list(vector) for vector in embeddings]


# --------------------------------------------------------------------------- #
# Resilient wrapper
# --------------------------------------------------------------------------- #
FALLBACK_ERRORS = (
    RateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    ProviderAuthError,
    ProviderBadRequestError,
)


class ResilientEmbeddings(BaseEmbeddingProvider):
    """Use a real provider until a known availability failure, then hash."""

    def __init__(
        self,
        primary: BaseEmbeddingProvider,
        fallback: HashingEmbeddings | None = None,
    ) -> None:
        super().__init__(
            model=primary.model,
            batch_size=primary.batch_size,
            max_chars=primary.max_chars,
        )
        self.primary = primary
        self.fallback = fallback or HashingEmbeddings(
            max_chars=primary.max_chars, announce=False
        )
        self.name = primary.name
        self.dimensions = primary.dimensions
        self.supports_task_type = primary.supports_task_type
        self.fallback_reason = ""
        self._fallback_active = False
        self._lock = threading.RLock()

    @property
    def fallback_active(self) -> bool:
        return self._fallback_active

    def _activate_fallback(self, exc: BaseException) -> None:
        with self._lock:
            if self._fallback_active:
                return
            self._fallback_active = True
            self.fallback_reason = type(exc).__name__
            self.name = self.fallback.name
            self.model = self.fallback.model
            self.dimensions = self.fallback.dimensions
            self.batch_size = self.fallback.batch_size
            self.supports_task_type = self.fallback.supports_task_type
            logger.warning(
                "Embedding provider %s unavailable (%s); using offline hash embeddings",
                self.primary.name,
                self.fallback_reason,
            )

    def embed_batch(self, texts: Sequence[str], *, is_query: bool = False) -> List[Vector]:
        if self._fallback_active:
            return self.fallback.embed_batch(texts, is_query=is_query)
        try:
            vectors = self.primary.embed_batch(texts, is_query=is_query)
            self.dimensions = self.primary.dimensions or (len(vectors[0]) if vectors else 0)
            return vectors
        except FALLBACK_ERRORS as exc:
            self._activate_fallback(exc)
            return self.fallback.embed_batch(texts, is_query=is_query)

    def embed_documents(self, texts: Sequence[str]) -> List[Vector]:
        if self._fallback_active:
            return self.fallback.embed_documents(texts)
        try:
            vectors = self.primary.embed_documents(texts)
            self.dimensions = self.primary.dimensions
            return vectors
        except FALLBACK_ERRORS as exc:
            self._activate_fallback(exc)
            return self.fallback.embed_documents(texts)

    def embed_query(self, text: str) -> Vector:
        if self._fallback_active:
            return self.fallback.embed_query(text)
        try:
            vector = self.primary.embed_query(text)
            self.dimensions = self.primary.dimensions
            return vector
        except FALLBACK_ERRORS as exc:
            self._activate_fallback(exc)
            return self.fallback.embed_query(text)


# --------------------------------------------------------------------------- #
# Factory and cache
# --------------------------------------------------------------------------- #
_embedding_cache: Dict[str, BaseEmbeddingProvider] = {}
_embedding_lock = threading.Lock()

SUPPORTED_PROVIDERS = ("openai", "openai_compatible", "gemini", "cohere", "jina", "hash", "mock")


def build_embedding_provider(cfg: Any) -> BaseEmbeddingProvider:
    """Instantiate the configured embedding backend (no caching)."""
    provider = (cfg.provider or "openai").lower()

    if provider in ("hash", "mock"):
        return HashingEmbeddings(
            model=cfg.model or "hash-1024",
            dimensions=cfg.dimensions or 1024,
            batch_size=cfg.batch_size,
            max_chars=cfg.max_chars_per_input,
        )

    if not cfg.api_key:
        raise MissingCredentialError("EMBEDDING_API_KEY", "document indexing")

    common = dict(
        api_key=cfg.api_key,
        model=cfg.model,
        base_url=cfg.base_url,
        dimensions=cfg.dimensions,
        batch_size=cfg.batch_size,
        timeout_s=cfg.timeout_s,
        max_chars=cfg.max_chars_per_input,
    )

    if provider in ("openai", "openai_compatible"):
        primary = OpenAICompatibleEmbeddings(**common)
        return ResilientEmbeddings(primary)
    if provider == "gemini":
        return ResilientEmbeddings(GeminiEmbeddings(**common))
    if provider == "cohere":
        return ResilientEmbeddings(CohereEmbeddings(**common))
    if provider == "jina":
        common["base_url"] = cfg.base_url or "https://api.jina.ai/v1"
        return ResilientEmbeddings(OpenAICompatibleEmbeddings(**common))

    raise ConfigurationError(
        f"Unknown EMBEDDING_PROVIDER={provider!r}",
        user_message=(
            f"`EMBEDDING_PROVIDER={provider}` is not supported. "
            f"Choose one of: {', '.join(SUPPORTED_PROVIDERS)}."
        ),
    )


def get_embedding_provider(settings: Any = None) -> BaseEmbeddingProvider:
    from backend.config import get_settings
    cfg = (settings or get_settings()).embedding
    key = f"{cfg.provider}|{cfg.model}|{cfg.dimensions}|{len(cfg.api_key)}|{cfg.base_url}"
    with _embedding_lock:
        provider = _embedding_cache.get(key)
        if provider is None:
            provider = build_embedding_provider(cfg)
            _embedding_cache[key] = provider
            logger.info("Embedding provider ready: %s (%s)", provider.name, provider.model)
        return provider


def reset_embedding_cache() -> None:
    with _embedding_lock:
        _embedding_cache.clear()


__all__ = [
    "Vector",
    "BaseEmbeddingProvider",
    "HashingEmbeddings",
    "OpenAICompatibleEmbeddings",
    "GeminiEmbeddings",
    "CohereEmbeddings",
    "ResilientEmbeddings",
    "FALLBACK_ERRORS",
    "SUPPORTED_PROVIDERS",
    "build_embedding_provider",
    "get_embedding_provider",
    "reset_embedding_cache",
]

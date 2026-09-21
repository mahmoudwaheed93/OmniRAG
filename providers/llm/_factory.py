"""Factory, cache, and provider chain construction."""

from __future__ import annotations

import threading
from hashlib import sha256
from typing import Any, Dict, List

from backend.config import get_settings
from backend.models import (
    ConfigurationError,
    MissingCredentialError,
)
from backend.utils import get_logger

from backend.providers.llm._types import LLMMessage
from backend.providers.llm._base import BaseLLMProvider
from backend.providers.llm._openai import OpenAICompatibleLLM
from backend.providers.llm._gemini import GeminiLLM
from backend.providers.llm._groq import GroqLLM
from backend.providers.llm._openrouter import OpenRouterLLM
from backend.providers.llm._anthropic import AnthropicLLM
from backend.providers.llm._mock import MockLLM
from backend.providers.llm._fallback import FallbackLLMProvider

logger = get_logger(__name__)

_factory_cache: Dict[str, BaseLLMProvider] = {}
_factory_lock = threading.Lock()

SUPPORTED_PROVIDERS = (
    "gemini",
    "groq",
    "openrouter",
    "openai",
    "openai_compatible",
    "anthropic",
    "mock",
)


def _factory_signature(cfg: Any) -> str:
    parts = [
        cfg.chain_label,
        str(cfg.enable_fallback),
        str(cfg.temperature),
        str(cfg.max_output_tokens),
        str(cfg.openrouter_free_fallback),
        str(cfg.rate_limit_cooldown_seconds),
        str(cfg.hard_quota_cooldown_seconds),
        str(cfg.groq_max_rate_limit_wait_seconds),
        str(cfg.groq_tpm_limit),
        str(cfg.groq_estimated_image_tokens),
        str(cfg.groq_focused_vision_max_output_tokens),
    ]
    parts.extend(
        ":".join(
            (
                endpoint.provider,
                sha256(endpoint.api_key.encode("utf-8")).hexdigest()
                if endpoint.api_key
                else "",
                endpoint.model,
                endpoint.vision_model,
                endpoint.base_url,
                str(endpoint.supports_images),
            )
        )
        for endpoint in cfg.endpoints
    )
    return "|".join(parts)


def build_endpoint_provider(
    endpoint: Any, cfg: Any
) -> BaseLLMProvider:
    """Instantiate the adapter for a single vendor endpoint."""
    provider = (endpoint.provider or "gemini").lower()

    if provider == "mock":
        return MockLLM(model=endpoint.model or "mock-llm", temperature=cfg.temperature)

    if not endpoint.api_key:
        raise MissingCredentialError(
            f"{provider.upper()}_API_KEY", f"the {provider} language model"
        )

    common = dict(
        api_key=endpoint.api_key,
        model=endpoint.model,
        base_url=endpoint.base_url,
        temperature=cfg.temperature,
        max_output_tokens=cfg.max_output_tokens,
        timeout_s=cfg.timeout_s,
        vision_model=endpoint.effective_vision_model,
        retry_attempts=cfg.retry_attempts,
    )

    if provider == "gemini":
        return GeminiLLM(**common)
    if provider == "groq":
        return GroqLLM(
            supports_images_override=endpoint.supports_images,
            max_rate_limit_wait_seconds=cfg.groq_max_rate_limit_wait_seconds,
            tpm_limit=cfg.groq_tpm_limit,
            estimated_image_tokens=cfg.groq_estimated_image_tokens,
            focused_vision_max_output_tokens=(
                cfg.groq_focused_vision_max_output_tokens
            ),
            **common,
        )
    if provider == "openrouter":
        return OpenRouterLLM(
            supports_images_override=endpoint.supports_images,
            free_fallback=cfg.openrouter_free_fallback,
            **common,
        )
    if provider in ("openai", "openai_compatible"):
        return OpenAICompatibleLLM(**common)
    if provider == "anthropic":
        common.pop("retry_attempts", None)
        return AnthropicLLM(**common)

    raise ConfigurationError(
        f"Unknown LLM provider {provider!r}",
        user_message=(
            f"`{provider}` is not a supported LLM provider. "
            f"Choose one of: {', '.join(SUPPORTED_PROVIDERS)}."
        ),
    )


def build_llm_provider(cfg: Any) -> BaseLLMProvider:
    """Build the full provider chain (uncached)."""
    endpoints = cfg.configured_endpoints
    if not endpoints:
        raise MissingCredentialError(
            "GEMINI_API_KEY, GROQ_API_KEY, or OPENROUTER_API_KEY",
            "answer generation",
        )

    providers: List[BaseLLMProvider] = []
    errors: List[str] = []
    for endpoint in endpoints:
        try:
            providers.append(build_endpoint_provider(endpoint, cfg))
        except ConfigurationError as exc:
            errors.append(f"{endpoint.provider}: {exc.detail or exc}")
            logger.warning("Skipping provider %s: %s", endpoint.provider, exc.detail or exc)

    if not providers:
        raise ConfigurationError(
            "; ".join(errors) or "no usable LLM provider",
            user_message=(
                "None of the configured LLM providers could be initialised. "
                + (errors[0] if errors else "")
            ),
        )

    router = FallbackLLMProvider(
        providers,
        enable_fallback=cfg.enable_fallback,
        rate_limit_cooldown_seconds=cfg.rate_limit_cooldown_seconds,
        hard_quota_cooldown_seconds=cfg.hard_quota_cooldown_seconds,
    )
    logger.info(
        "LLM chain ready: %s (fallback %s)",
        " → ".join(f"{p.name}:{p.model}" for p in router.chain),
        "enabled" if router.enable_fallback else "disabled",
    )
    return router


def get_llm_provider(settings: Any = None) -> BaseLLMProvider:
    """Cached accessor used across the app."""
    cfg = (settings or get_settings()).llm
    key = _factory_signature(cfg)
    with _factory_lock:
        provider = _factory_cache.get(key)
        if provider is None:
            provider = build_llm_provider(cfg)
            _factory_cache[key] = provider
        return provider


def reset_llm_cache() -> None:
    with _factory_lock:
        _factory_cache.clear()


def model_supports_images(model: str) -> bool:
    m = (model or "").strip().lower()
    if not m:
        return False
    if m in GroqLLM.VISION_MODELS:
        return True
    if m == OpenRouterLLM.FREE_ROUTER_MODEL:
        return True
    if any(marker in m for marker in OpenRouterLLM.VISION_MODEL_MARKERS):
        if not any(marker in m for marker in OpenRouterLLM.TEXT_ONLY_MARKERS):
            return True
    return False

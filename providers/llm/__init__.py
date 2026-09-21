"""LLM provider interface, adapters, routing, and HTTP plumbing.

This package consolidates the entire LLM provider layer into separate modules,
including the base interface, all vendor adapters, the fallback router,
factory/cache, error classification, and shared HTTP utilities.
"""

from backend.providers.llm._types import (
    current_llm_operation,
    current_generation_id,
    current_llm_session_id,
    llm_operation,
    generation_context,
    llm_session,
    FailureClass,
    ImagePart,
    LLMMessage,
    LLMResponse,
    LLMRequestRequirements,
)

from backend.providers.llm._http import (
    get_client,
    close_clients,
    post_json,
    attach_http_context,
    is_quota_exhausted,
)

from backend.providers.llm._classify import (
    classify,
    should_failover,
    should_retry_same_provider,
    describe,
)

from backend.providers.llm._base import BaseLLMProvider

from backend.providers.llm._openai import OpenAICompatibleLLM
from backend.providers.llm._gemini import GeminiLLM
from backend.providers.llm._groq import GroqLLM
from backend.providers.llm._openrouter import OpenRouterLLM
from backend.providers.llm._anthropic import AnthropicLLM
from backend.providers.llm._mock import MockLLM

from backend.providers.llm._fallback import (
    ProviderAttempt,
    RouterStats,
    FallbackLLMProvider,
)

from backend.providers.llm._factory import (
    SUPPORTED_PROVIDERS,
    build_endpoint_provider,
    build_llm_provider,
    get_llm_provider,
    reset_llm_cache,
    model_supports_images,
)

from backend.utils import retry_call  # re-exported for backward compat

__all__ = [
    # Context
    "current_llm_operation",
    "current_generation_id",
    "current_llm_session_id",
    "llm_operation",
    "generation_context",
    "llm_session",
    # HTTP
    "get_client",
    "close_clients",
    "post_json",
    "attach_http_context",
    "is_quota_exhausted",
    # Error classification
    "FailureClass",
    "classify",
    "should_failover",
    "should_retry_same_provider",
    "describe",
    # Base interface
    "ImagePart",
    "LLMMessage",
    "LLMResponse",
    "LLMRequestRequirements",
    "BaseLLMProvider",
    # Adapters
    "OpenAICompatibleLLM",
    "GeminiLLM",
    "GroqLLM",
    "OpenRouterLLM",
    "AnthropicLLM",
    "MockLLM",
    # Router
    "ProviderAttempt",
    "RouterStats",
    "FallbackLLMProvider",
    # Factory
    "SUPPORTED_PROVIDERS",
    "build_endpoint_provider",
    "build_llm_provider",
    "get_llm_provider",
    "reset_llm_cache",
    "model_supports_images",
]

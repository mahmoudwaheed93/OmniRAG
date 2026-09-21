"""Base LLM provider contract and shared text-extraction helpers."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple

from backend.models import (
    ProviderAuthError,
    ProviderBadRequestError,
    ProviderPaymentRequiredError,
    ProviderPolicyError,
    ProviderTokenBudgetExceededError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    RateLimitError,
)

from backend.providers.llm._types import (
    LLMMessage,
    LLMRequestRequirements,
    LLMResponse,
    Role,
    _LEADING_THINK_RE,
    _REASONING_PART_TYPES,
    _VISIBLE_TEXT_PART_TYPES,
)
from backend.providers.llm._http import attach_http_context, is_quota_exhausted


class BaseLLMProvider(ABC):
    """Contract every LLM adapter implements."""

    name: str = "base"
    supports_vision: bool = False

    def __init__(self, *, model: str, temperature: float = 0.1, max_output_tokens: int = 1400):
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    def supports_images(self, model: Optional[str] = None) -> bool:
        return self.supports_vision

    def model_for_request(
        self, messages: Sequence[LLMMessage], model: Optional[str] = None
    ) -> str:
        return model or self.model

    @abstractmethod
    def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        model: Optional[str] = None,
        json_mode: bool = False,
        requirements: Optional[LLMRequestRequirements] = None,
    ) -> LLMResponse:
        pass

    def complete_text(self, prompt: str, *, system: Optional[str] = None, **kwargs) -> str:
        response = self.complete([LLMMessage(role=Role.USER, text=prompt)], system=system, **kwargs)
        return response.text

    def health(self) -> bool:
        return True

    def describe(self) -> Dict[str, Any]:
        return {
            "provider": self.name,
            "model": self.model,
            "vision": self.supports_vision,
            "images": self.supports_images(),
        }


# --------------------------------------------------------------------------- #
# Text extraction helpers
# --------------------------------------------------------------------------- #
def _visible_message_text(message: Dict[str, Any]) -> tuple[str, int]:
    reasoning_chars = _nested_text_length(message.get("reasoning"))
    content = message.get("content")
    if isinstance(content, str):
        return content, reasoning_chars
    if not isinstance(content, list):
        return "", reasoning_chars

    visible: List[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        part_type = str(part.get("type") or "").strip().lower()
        part_text = part.get("text")
        if part_type in _REASONING_PART_TYPES:
            reasoning_chars += _nested_text_length(part_text)
        elif part_type in _VISIBLE_TEXT_PART_TYPES and isinstance(part_text, str):
            visible.append(part_text)
    return "".join(visible), reasoning_chars


def _without_leading_thinking(text: str) -> tuple[str, int]:
    remaining = text or ""
    suppressed = 0
    while True:
        match = _LEADING_THINK_RE.match(remaining)
        if not match:
            break
        suppressed += len(match.group(1))
        remaining = remaining[match.end():]
    if re.match(r"^\s*<think(?:\s[^>]*)?>", remaining, flags=re.IGNORECASE):
        return "", suppressed + len(remaining)
    return remaining.strip(), suppressed


def _nested_text_length(value: Any) -> int:
    if isinstance(value, str):
        return len(value)
    if isinstance(value, list):
        return sum(_nested_text_length(item) for item in value)
    if isinstance(value, dict):
        return sum(
            _nested_text_length(value.get(key))
            for key in ("text", "content", "reasoning")
            if key in value
        )
    return 0


def raise_gateway_error(error: Dict[str, Any], *, provider: str) -> None:
    """Translate an error object returned inside a 200 response."""
    message = str(error.get("message", ""))[:300]
    code = error.get("code")
    try:
        status = int(code)
    except (TypeError, ValueError):
        status = 0

    if status == 413 and any(
        marker in message.lower()
        for marker in ("token", "tpm", "request too large", "rate_limit_exceeded")
    ):
        raise attach_http_context(
            ProviderTokenBudgetExceededError(
                f"{provider} token budget exceeded: {message}",
                provider=provider,
            ),
            status,
            message,
        )
    if status == 429 or "rate limit" in message.lower():
        raise attach_http_context(RateLimitError(
            f"{provider} rate limited: {message}",
            provider=provider,
            quota_exhausted=is_quota_exhausted(message),
        ), status or 429, message)
    if status in (408, 504) or "timeout" in message.lower():
        raise attach_http_context(
            ProviderTimeoutError(f"{provider} timeout: {message}", provider=provider),
            status,
            message,
        )
    if status >= 500:
        raise attach_http_context(ProviderUnavailableError(
            f"{provider} upstream error {status}: {message}", provider=provider
        ), status, message)
    if status in (401, 403):
        raise attach_http_context(ProviderAuthError(
            f"{provider} auth error: {message}",
            provider=provider,
            user_message=f"The {provider} API rejected your credentials.",
        ), status, message)
    if status == 402:
        raise attach_http_context(ProviderPaymentRequiredError(
            f"{provider} payment required: {message}",
            provider=provider,
            user_message="The configured OpenRouter route requires credits.",
        ), status, message)
    raise attach_http_context(ProviderBadRequestError(
        f"{provider} returned an error: {message}", provider=provider
    ), status, message)




"""Error classification helpers for failover routing."""

from __future__ import annotations

from typing import Tuple

from backend.models import (
    AllProvidersFailedError,
    ProviderAuthError,
    ProviderBadRequestError,
    ProviderCapabilityError,
    ProviderError,
    ProviderPaymentRequiredError,
    ProviderPolicyError,
    ProviderTokenBudgetExceededError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    RateLimitError,
)

from backend.providers.llm._types import FailureClass


RECOVERABLE_TYPES: Tuple[type, ...] = (
    RateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    ProviderTokenBudgetExceededError,
)


def classify(exc: BaseException) -> FailureClass:
    """Map an exception to its FailureClass. Never raises."""
    if isinstance(exc, ProviderPolicyError):
        return FailureClass.POLICY
    if isinstance(exc, ProviderCapabilityError):
        return FailureClass.CAPABILITY
    if isinstance(exc, ProviderAuthError):
        return FailureClass.AUTH
    if isinstance(exc, ProviderPaymentRequiredError):
        return FailureClass.RECOVERABLE
    if isinstance(exc, ProviderTokenBudgetExceededError):
        return FailureClass.RECOVERABLE
    if isinstance(exc, ProviderBadRequestError):
        return FailureClass.BAD_REQUEST
    if isinstance(exc, AllProvidersFailedError):
        return FailureClass.BAD_REQUEST
    if isinstance(exc, RECOVERABLE_TYPES):
        return FailureClass.RECOVERABLE
    if isinstance(exc, ProviderError):
        return FailureClass.RECOVERABLE if exc.retryable else FailureClass.BAD_REQUEST
    return FailureClass.BUG


def should_failover(exc: BaseException) -> bool:
    """True only when trying a different provider could plausibly succeed."""
    return classify(exc) is FailureClass.RECOVERABLE


def should_retry_same_provider(exc: BaseException) -> bool:
    """True when re-issuing the same request to the same provider may work."""
    if isinstance(exc, RateLimitError) and exc.quota_exhausted:
        return False
    if isinstance(exc, ProviderPaymentRequiredError):
        return False
    if isinstance(exc, ProviderTokenBudgetExceededError):
        return False
    return classify(exc) is FailureClass.RECOVERABLE


def describe(exc: BaseException) -> str:
    """Short, log-safe description."""
    provider = getattr(exc, "provider", "") or "provider"
    return f"{provider}/{type(exc).__name__}[{classify(exc).value}]"

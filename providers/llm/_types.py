"""Context variables, enums, dataclasses, and shared constants for the LLM layer."""

from __future__ import annotations

import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Tuple

from backend.models import Role


# --------------------------------------------------------------------------- #
# Context variables for diagnostics
# --------------------------------------------------------------------------- #
_operation: ContextVar[str] = ContextVar("omnirag_llm_operation", default="unspecified")
_generation_id: ContextVar[str] = ContextVar("omnirag_generation_id", default="")
_session_id: ContextVar[str] = ContextVar("omnirag_llm_session_id", default="")


def current_llm_operation() -> str:
    return _operation.get()


def current_generation_id() -> str:
    return _generation_id.get()


def current_llm_session_id() -> str:
    return _session_id.get()


@contextmanager
def llm_operation(name: str) -> Iterator[None]:
    token = _operation.set(name or "unspecified")
    try:
        yield
    finally:
        _operation.reset(token)


@contextmanager
def generation_context(generation_id: str) -> Iterator[None]:
    token = _generation_id.set(generation_id or "")
    try:
        yield
    finally:
        _generation_id.reset(token)


@contextmanager
def llm_session(session_id: str) -> Iterator[None]:
    token = _session_id.set(session_id or "")
    try:
        yield
    finally:
        _session_id.reset(token)


# --------------------------------------------------------------------------- #
# Enums / dataclasses
# --------------------------------------------------------------------------- #
class FailureClass(str, Enum):
    RECOVERABLE = "recoverable"
    AUTH = "auth"
    BAD_REQUEST = "bad_request"
    PAYMENT = "payment_required"
    POLICY = "policy"
    CAPABILITY = "capability"
    BUG = "bug"


@dataclass
class ImagePart:
    """An image attached to a message."""

    data: bytes
    media_type: str = "image/png"
    label: str = ""


@dataclass
class LLMMessage:
    role: Role = Role.USER
    text: str = ""
    images: List[ImagePart] = field(default_factory=list)

    @property
    def has_images(self) -> bool:
        return bool(self.images)


@dataclass
class LLMResponse:
    text: str
    model: str = ""
    finish_reason: str = ""
    usage: Dict[str, Any] = field(default_factory=dict)
    raw: Optional[Dict[str, Any]] = None
    provider: str = ""
    fallback_used: bool = False
    attempts: List[str] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def provider_label(self) -> str:
        if self.provider and self.model:
            return f"{self.provider} · {self.model}"
        return self.provider or self.model or "unknown"


@dataclass(frozen=True)
class LLMRequestRequirements:
    """Capabilities a routed operation must preserve end to end."""

    requires_text: bool = True
    requires_images: bool = False
    requires_structured_output: bool = False
    operation: str = "unspecified"


# --------------------------------------------------------------------------- #
# Text-part type constants
# --------------------------------------------------------------------------- #
_TEXT_ONLY_HINTS = ("embedding", "whisper", "tts", "moderation", "instruct")
_REASONING_PART_TYPES = frozenset({"analysis", "reasoning", "thinking"})
_VISIBLE_TEXT_PART_TYPES = frozenset({"", "text", "output_text"})
_LEADING_THINK_RE = re.compile(
    r"^\s*<think(?:\s[^>]*)?>(.*?)</think>\s*",
    flags=re.IGNORECASE | re.DOTALL,
)

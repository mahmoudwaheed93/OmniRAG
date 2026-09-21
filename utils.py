"""Utility functions shared across the OmniRAG pipeline.

Includes logging, hashing, language detection, text normalization, image helpers,
retry logic, and user-facing message sanitization.
"""

from __future__ import annotations

import base64
import functools
import hashlib
import io
import logging
import os
import re
import sys
import time
import unicodedata
import uuid
from collections import Counter
from typing import Any, Callable, Iterable, List, Mapping, Optional, Sequence, Tuple, Type, TypeVar

from backend.models import (
    Language,
    ProviderError,
    ProviderCapabilityError,
    ProviderPolicyError,
    ProviderTimeoutError,
    RateLimitError,
)


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
_CONFIGURED = False
_LOGGER_NAME = "omnirag"

_SECRET_PATTERNS = [
    re.compile(r"(sk-[A-Za-z0-9_\-]{8,})"),
    re.compile(r"(sk-ant-[A-Za-z0-9_\-]{8,})"),
    re.compile(r"(AIza[0-9A-Za-z_\-]{20,})"),
    re.compile(r"((?i:api[_-]?key)\"?\s*[:=]\s*\"?)([^\s\"',}]{6,})"),
    re.compile(r"(Bearer\s+)([A-Za-z0-9._\-]{10,})"),
]


class RedactingFilter(logging.Filter):
    """Masks anything that looks like a credential in the formatted message."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # pragma: no cover
            return True
        redacted = redact(message)
        if redacted != message:
            record.msg = redacted
            record.args = ()
        return True


def redact(text: str) -> str:
    """Replace credential-looking substrings with ``***``."""
    out = text
    for pattern in _SECRET_PATTERNS:
        if pattern.groups >= 2:
            out = pattern.sub(lambda m: f"{m.group(1)}***", out)
        else:
            out = pattern.sub("***", out)
    return out


def configure_logging(level: Optional[str] = None, *, force: bool = False) -> None:
    """Configure the ``omnirag`` logger tree exactly once."""
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    resolved = (level or os.environ.get("LOG_LEVEL") or "INFO").upper()
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(getattr(logging, resolved, logging.INFO))
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    handler.addFilter(RedactingFilter())
    logger.addHandler(handler)

    for noisy in ("httpx", "httpcore", "urllib3", "qdrant_client", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str = _LOGGER_NAME) -> logging.Logger:
    """Return a namespaced logger, configuring the tree on first use."""
    configure_logging()
    if name == _LOGGER_NAME or name.startswith(f"{_LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_LOGGER_NAME}.{name.split('.')[-1]}")


logger = get_logger(__name__)


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    """Structured-ish one-line event log."""
    parts = " ".join(f"{k}={v!r}" for k, v in fields.items() if v is not None)
    logger.info("%s %s", event, parts)


# --------------------------------------------------------------------------- #
# Hashing
# --------------------------------------------------------------------------- #
_UNSAFE = re.compile(r"[^\w\s.\-()؀-ۿ]", re.UNICODE)
_WS = re.compile(r"\s+")
_NAMESPACE = uuid.UUID("6f5c1c8e-0d9e-4f0a-9d3d-2f3f6f9a1b77")


def content_hash(data: bytes) -> str:
    """SHA-256 of raw bytes — the deduplication key."""
    return hashlib.sha256(data).hexdigest()


def short_hash(data: bytes | str, length: int = 12) -> str:
    payload = data.encode("utf-8") if isinstance(data, str) else data
    return hashlib.sha256(payload).hexdigest()[:length]


def text_hash(text: str) -> str:
    """Hash of normalised text — used to dedupe identical chunks/captions."""
    normalized = _WS.sub(" ", unicodedata.normalize("NFKC", text)).strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def stable_id(*parts: str) -> str:
    """Deterministic UUID5 hex from the given parts."""
    name = "|".join(str(p) for p in parts)
    return uuid.uuid5(_NAMESPACE, name).hex


def stable_uuid(*parts: str) -> str:
    """Same as :func:`stable_id` but in canonical UUID form."""
    return str(uuid.uuid5(_NAMESPACE, "|".join(str(p) for p in parts)))


def sanitize_filename(filename: str, *, max_length: int = 120) -> str:
    """Make an uploaded filename safe to use as a display name and path part."""
    if not filename:
        return "untitled"

    name = unicodedata.normalize("NFKC", filename)
    name = name.replace("\\", "/").split("/")[-1]
    name = "".join(ch for ch in name if ch.isprintable())
    name = _UNSAFE.sub("_", name)
    name = _WS.sub(" ", name).strip(" .")

    if not name:
        return "untitled"

    root, ext = os.path.splitext(name)
    ext = ext[:12]
    root = root[: max(1, max_length - len(ext))] or "untitled"

    if root.upper() in {
        "CON", "PRN", "AUX", "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }:
        root = f"_{root}"
    return f"{root}{ext}"


def file_extension(filename: str) -> str:
    """Lower-case extension without the dot."""
    return os.path.splitext(filename)[1].lower().lstrip(".")


# --------------------------------------------------------------------------- #
# Language detection
# --------------------------------------------------------------------------- #
_ARABIC_RANGES = (
    (0x0600, 0x06FF),
    (0x0750, 0x077F),
    (0x08A0, 0x08FF),
    (0xFB50, 0xFDFF),
    (0xFE70, 0xFEFF),
)
_LATIN_RE = re.compile(r"[A-Za-z]")
_ARABIC_DIACRITICS = re.compile(r"[ً-ٰٟۖ-ۭ]")
_TATWEEL = "ـ"
_MIXED_THRESHOLD = 0.20


def is_arabic_char(ch: str) -> bool:
    code = ord(ch)
    return any(start <= code <= end for start, end in _ARABIC_RANGES)


def script_ratios(text: str) -> tuple[float, float]:
    """Return ``(arabic_ratio, latin_ratio)`` over alphabetic characters only."""
    arabic = latin = 0
    for ch in text:
        if is_arabic_char(ch):
            arabic += 1
        elif _LATIN_RE.match(ch):
            latin += 1
    total = arabic + latin
    if total == 0:
        return 0.0, 0.0
    return arabic / total, latin / total


def detect_language(text: str) -> Language:
    """Classify text into Arabic / English / mixed / unknown."""
    if not text or not text.strip():
        return Language.UNKNOWN
    arabic, latin = script_ratios(text)
    if arabic == 0.0 and latin == 0.0:
        return Language.UNKNOWN
    if arabic >= _MIXED_THRESHOLD and latin >= _MIXED_THRESHOLD:
        return Language.MIXED
    return Language.ARABIC if arabic > latin else Language.ENGLISH


def detect_languages(texts: Iterable[str]) -> Language:
    """Aggregate detection over many strings (document-level language)."""
    joined = " ".join(t for t in texts if t)[:20000]
    return detect_language(joined)


def normalize_arabic(text: str) -> str:
    """Light Arabic normalisation for *search keys only*."""
    if not text:
        return ""
    out = unicodedata.normalize("NFKC", text)
    out = _ARABIC_DIACRITICS.sub("", out)
    out = out.replace(_TATWEEL, "")
    out = re.sub("[آأإٱ]", "ا", out)
    out = out.replace("ى", "ي")
    out = out.replace("ة", "ه")
    out = out.replace("ؤ", "و").replace("ئ", "ي")
    return out


def normalize_for_search(text: str) -> str:
    """Full normalisation used by the keyword index and query pipeline."""
    return re.sub(r"\s+", " ", normalize_arabic(text).lower()).strip()


def contains_arabic(text: str) -> bool:
    return any(is_arabic_char(ch) for ch in text)


def language_name(language: Language) -> str:
    return {
        Language.ARABIC: "Arabic",
        Language.ENGLISH: "English",
        Language.MIXED: "mixed Arabic/English",
        Language.UNKNOWN: "the user's language",
    }[language]


def is_rtl(text: str) -> bool:
    """Whether a string should be rendered right-to-left."""
    arabic, latin = script_ratios(text)
    return arabic > latin


# --------------------------------------------------------------------------- #
# Text normalization
# --------------------------------------------------------------------------- #
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MULTI_SPACE = re.compile(r"[ \t ]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_SOFT_BREAK = re.compile(r"(?<![.!?:;،؛])\n(?![\n\-\*•\d])")
_PAGE_NUMBER_LINE = re.compile(r"^\s*(?:page\s*)?[-–—]?\s*\d{1,4}\s*[-–—]?\s*$", re.I)
_BULLET = re.compile(r"^\s*[•●▪·\-\*]\s+")
_TOKEN = re.compile(r"[\w؀-ۿ]+", re.UNICODE)

_CHARS_PER_TOKEN = 3.2


def clean_text(text: str, *, join_soft_breaks: bool = True) -> str:
    """Normalise extracted text without altering its meaning."""
    if not text:
        return ""
    out = unicodedata.normalize("NFKC", text)
    out = _CONTROL.sub("", out)
    out = out.replace("\r\n", "\n").replace("\r", "\n")
    out = _HYPHEN_BREAK.sub(r"\1\2", out)
    if join_soft_breaks:
        out = _SOFT_BREAK.sub(" ", out)
    out = _MULTI_SPACE.sub(" ", out)
    out = _MULTI_NEWLINE.sub("\n\n", out)
    return "\n".join(line.strip() for line in out.split("\n")).strip()


def strip_page_artifacts(text: str) -> str:
    """Drop stand-alone page-number lines left over from PDF extraction."""
    kept = [ln for ln in text.split("\n") if not _PAGE_NUMBER_LINE.match(ln)]
    return "\n".join(kept).strip()


def detect_repeated_lines(
    page_texts: Sequence[str], *, min_pages: int = 3, ratio: float = 0.6, edge_lines: int = 3
) -> set[str]:
    """Find running headers/footers shared by most pages."""
    if len(page_texts) < min_pages:
        return set()

    counter: Counter[str] = Counter()
    for text in page_texts:
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        candidates = lines[:edge_lines] + lines[-edge_lines:]
        for line in set(candidates):
            if 3 <= len(line) <= 120:
                counter[line] += 1

    threshold = max(min_pages, int(len(page_texts) * ratio))
    return {line for line, count in counter.items() if count >= threshold}


def remove_lines(text: str, blocked: set[str]) -> str:
    if not blocked:
        return text
    kept = [ln for ln in text.split("\n") if ln.strip() not in blocked]
    return "\n".join(kept).strip()


def normalize_bullets(text: str) -> str:
    return "\n".join(_BULLET.sub("- ", ln) for ln in text.split("\n"))


def tokenize(text: str) -> List[str]:
    """Language-aware tokens for BM25."""
    return _TOKEN.findall(normalize_for_search(text))


def estimate_tokens(text: str) -> int:
    """Cheap token estimate used for context budgeting."""
    if not text:
        return 0
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


def truncate(text: str, max_chars: int, *, suffix: str = "…") -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    boundary = max(cut.rfind(" "), cut.rfind("\n"))
    if boundary > max_chars * 0.6:
        cut = cut[:boundary]
    return cut.rstrip() + suffix


def snippet(text: str, max_chars: int = 320) -> str:
    return truncate(" ".join(text.split()), max_chars)


def split_sentences(text: str) -> List[str]:
    """Sentence split supporting Arabic punctuation and ellipses."""
    parts = re.split(r"(?<=[.!?؟])\s+|\n{2,}", text)
    return [p.strip() for p in parts if p and p.strip()]


def split_paragraphs(text: str) -> List[str]:
    parts = re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if p and p.strip()]


def is_meaningful(text: str, *, min_chars: int = 12, min_tokens: int = 2) -> bool:
    """Whether a fragment is worth indexing."""
    stripped = text.strip()
    if len(stripped) < min_chars:
        return False
    tokens = _TOKEN.findall(stripped)
    if len(tokens) < min_tokens:
        return False
    alnum = sum(1 for ch in stripped if ch.isalnum())
    return alnum / max(1, len(stripped)) >= 0.35


def dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for item in items:
        key = normalize_for_search(item)
        if key and key not in seen:
            seen.add(key)
            out.append(item)
    return out


# --------------------------------------------------------------------------- #
# Image helpers
# --------------------------------------------------------------------------- #
try:  # pragma: no cover
    from PIL import Image, ImageStat

    PIL_AVAILABLE = True
except Exception:  # pragma: no cover
    Image = None  # type: ignore[assignment]
    ImageStat = None  # type: ignore[assignment]
    PIL_AVAILABLE = False


def open_image(data: bytes) -> Optional["Image.Image"]:
    if not PIL_AVAILABLE or not data:
        return None
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
        return image
    except Exception as exc:
        logger.debug("Could not open image (%d bytes): %s", len(data), exc)
        return None


def image_size(data: bytes) -> Tuple[int, int]:
    image = open_image(data)
    return (image.width, image.height) if image is not None else (0, 0)


def normalize_image(
    data: bytes,
    *,
    max_edge: int = 1400,
    jpeg_quality: int = 82,
    force_format: Optional[str] = None,
) -> Tuple[bytes, str]:
    """Downscale and re-encode an image for API transport."""
    image = open_image(data)
    if image is None:
        return data, "image/png"

    try:
        has_alpha = image.mode in ("RGBA", "LA", "P") and "transparency" in image.info
        fmt = force_format or ("PNG" if has_alpha else "JPEG")

        if max(image.width, image.height) > max_edge:
            ratio = max_edge / float(max(image.width, image.height))
            new_size = (max(1, int(image.width * ratio)), max(1, int(image.height * ratio)))
            image = image.resize(new_size, Image.LANCZOS)

        if fmt == "JPEG" and image.mode not in ("RGB", "L"):
            background = Image.new("RGB", image.size, (255, 255, 255))
            if image.mode in ("RGBA", "LA"):
                background.paste(image, mask=image.split()[-1])
                image = background
            else:
                image = image.convert("RGB")

        buffer = io.BytesIO()
        if fmt == "JPEG":
            image.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
            return buffer.getvalue(), "image/jpeg"
        image.save(buffer, format="PNG", optimize=True)
        return buffer.getvalue(), "image/png"
    except Exception as exc:
        logger.debug("Image normalisation failed: %s", exc)
        return data, "image/png"


def to_data_url(data: bytes, media_type: str = "image/png") -> str:
    return f"data:{media_type};base64,{base64.b64encode(data).decode('ascii')}"


def to_base64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def is_probably_blank(data: bytes, *, std_threshold: float = 4.0) -> bool:
    """Detect near-uniform images."""
    image = open_image(data)
    if image is None or ImageStat is None:
        return False
    try:
        grayscale = image.convert("L")
        if grayscale.width < 4 or grayscale.height < 4:
            return True
        stat = ImageStat.Stat(grayscale)
        return float(stat.stddev[0]) < std_threshold
    except Exception:
        return False


def is_probably_decorative(
    data: bytes, *, min_pixels: int = 110 * 110, max_aspect: float = 12.0
) -> bool:
    """Heuristic filter for logos, bullets, rules and other non-informative art."""
    width, height = image_size(data)
    if width == 0 or height == 0:
        return True
    if width * height < min_pixels:
        return True
    aspect = max(width / height, height / width)
    if aspect > max_aspect:
        return True
    return is_probably_blank(data)


def crop(data: bytes, box: Tuple[float, float, float, float], *, padding: int = 6) -> Optional[bytes]:
    """Crop a region (pixel coordinates) with a little padding."""
    image = open_image(data)
    if image is None:
        return None
    try:
        x0, y0, x1, y1 = box
        region = (
            max(0, int(x0) - padding),
            max(0, int(y0) - padding),
            min(image.width, int(x1) + padding),
            min(image.height, int(y1) + padding),
        )
        if region[2] <= region[0] or region[3] <= region[1]:
            return None
        buffer = io.BytesIO()
        image.crop(region).save(buffer, format="PNG", optimize=True)
        return buffer.getvalue()
    except Exception as exc:
        logger.debug("Crop failed: %s", exc)
        return None


def ensure_min_size(data: bytes, min_edge: int = 320) -> bytes:
    """Upscale very small crops so OCR/vision models can read them."""
    image = open_image(data)
    if image is None:
        return data
    if max(image.width, image.height) >= min_edge:
        return data
    try:
        ratio = min_edge / float(max(1, max(image.width, image.height)))
        resized = image.resize(
            (max(1, int(image.width * ratio)), max(1, int(image.height * ratio))),
            Image.LANCZOS,
        )
        buffer = io.BytesIO()
        resized.convert("RGB").save(buffer, format="PNG")
        return buffer.getvalue()
    except Exception:
        return data


# --------------------------------------------------------------------------- #
# Retry logic
# --------------------------------------------------------------------------- #
T = TypeVar("T")

DEFAULT_RETRYABLE: tuple[Type[BaseException], ...] = (
    RateLimitError,
    ProviderTimeoutError,
)


def is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, RateLimitError):
        return not exc.quota_exhausted
    if isinstance(exc, DEFAULT_RETRYABLE):
        return True
    if isinstance(exc, ProviderError):
        return exc.retryable
    return False


def backoff_delay(attempt: int, base: float, cap: float) -> float:
    """Exponential backoff with full jitter (attempt is 1-based)."""
    import random
    raw = min(cap, base * (2 ** (attempt - 1)))
    return random.uniform(base * 0.5, raw)


def retry_call(
    func: Callable[[], T],
    *,
    attempts: int = 3,
    base_delay: float = 0.8,
    max_delay: float = 8.0,
    retry_on: Iterable[Type[BaseException]] | None = None,
    operation: str = "provider call",
    sleep: Callable[[float], None] | None = None,
    skip_if_retry_after_exceeds_max: bool = False,
) -> T:
    """Call ``func`` retrying transient failures."""
    extra = tuple(retry_on or ())
    sleep_fn = sleep or time.sleep
    last: BaseException | None = None

    for attempt in range(1, max(1, attempts) + 1):
        try:
            return func()
        except BaseException as exc:  # noqa: BLE001
            retryable = is_retryable(exc) or isinstance(exc, extra)
            if not retryable or attempt >= attempts:
                raise
            last = exc
            retry_after = getattr(exc, "retry_after", None)
            if (
                skip_if_retry_after_exceeds_max
                and retry_after is not None
                and float(retry_after) > max_delay
            ):
                raise
            delay = retry_after or backoff_delay(
                attempt, base_delay, max_delay
            )
            delay = min(float(delay), max_delay)
            logger.warning(
                "%s failed (attempt %d/%d): %s — retrying in %.1fs",
                operation,
                attempt,
                attempts,
                type(exc).__name__,
                delay,
            )
            sleep_fn(float(delay))

    assert last is not None  # pragma: no cover
    raise last


def with_retry(
    *,
    attempts: int = 3,
    base_delay: float = 0.8,
    max_delay: float = 8.0,
    operation: str = "",
):
    """Decorator form of :func:`retry_call`."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            return retry_call(
                lambda: func(*args, **kwargs),
                attempts=attempts,
                base_delay=base_delay,
                max_delay=max_delay,
                operation=operation or func.__name__,
            )

        return wrapper

    return decorator


# --------------------------------------------------------------------------- #
# User-facing messages
# --------------------------------------------------------------------------- #
SERVICE_UNAVAILABLE = "The AI service is temporarily unavailable. Please try again shortly."
SERVICE_NOT_CONFIGURED = (
    "The AI service is not configured. Please contact the app administrator."
)

_CONFIG_MARKERS = (
    "api_key", "api key", "credential", "not configured", "streamlit secrets",
)
_PROVIDER_MARKERS = (
    "gemini", "groq", "openrouter", "provider", "fallback", "failover", "endpoint",
    "rate limit", "rate-limit", "quota", "credits", "payment required",
    "embedding", "reranker", "vector store", "qdrant", "faiss", "chroma",
    "http 4", "http 5", " 402", " 429", " 500", " 502", " 503", " 504",
)
_EXCEPTION_NAME = re.compile(r"\b[A-Z][A-Za-z]+(?:Error|Exception)\b")
_RAW_ERROR_BODY = re.compile(r"[\{\[]\s*[\"']?(?:error|status|code)[\"']?\s*:", re.I)
_HTTP_STATUS = re.compile(
    r"\b(?:http(?:\s+status)?|status(?:\s+code)?)[\s:=_-]*[45]\d{2}\b",
    re.I,
)


def provider_error_message(exc: BaseException, *, debug: bool = False) -> str:
    """Return a neutral provider failure unless diagnostics were requested."""
    if debug:
        return str(getattr(exc, "user_message", "") or exc)
    if isinstance(exc, ProviderPolicyError):
        return "This request could not be completed because of safety restrictions."
    if isinstance(exc, ProviderCapabilityError):
        return (
            "The AI service cannot process the required visual evidence right now. "
            "Please try again shortly."
        )
    return SERVICE_UNAVAILABLE


def public_error_text(message: str, *, debug: bool = False) -> str:
    """Sanitize technical errors already present in UI session state."""
    text = str(message or "").strip()
    if debug or not text:
        return text
    lowered = text.lower()
    if any(marker in lowered for marker in _CONFIG_MARKERS):
        return SERVICE_NOT_CONFIGURED
    if (
        any(marker in lowered for marker in _PROVIDER_MARKERS)
        or _EXCEPTION_NAME.search(text)
        or _RAW_ERROR_BODY.search(text)
        or _HTTP_STATUS.search(text)
    ):
        return SERVICE_UNAVAILABLE
    return text


def public_generation_warning(message: str, *, debug: bool = False) -> Optional[str]:
    """Keep actionable answer notes while suppressing implementation details."""
    text = str(message or "").strip()
    if debug:
        return text or None
    lowered = text.lower()
    if "citation" in lowered or "source" in lowered:
        if "did not cite" in lowered:
            return "This answer did not include source references. Please verify it carefully."
        return "Some invalid source references were removed from the answer."
    if "output limit" in lowered or "continued" in lowered:
        return "The answer reached its length limit; the available response is shown."
    if any(word in lowered for word in ("visual", "image", "chart", "diagram")):
        return "Some visual evidence could not be analysed for this answer."
    return None


def public_processing_note(message: str, *, debug: bool = False) -> str:
    """Sanitize ingestion notes while retaining useful degradation context."""
    text = str(message or "").strip()
    if debug or not text:
        return text
    lowered = text.lower()
    if (
        any(marker in lowered for marker in _PROVIDER_MARKERS + _CONFIG_MARKERS)
        or "embedding" in lowered
        or "image-capable model" in lowered
        or _EXCEPTION_NAME.search(text)
        or _RAW_ERROR_BODY.search(text)
    ):
        return (
            "Some advanced document analysis was unavailable. "
            "The remaining document content is still usable."
        )
    return text


__all__ = [
    # Logging
    "configure_logging",
    "get_logger",
    "log_event",
    "redact",
    # Hashing
    "content_hash",
    "short_hash",
    "text_hash",
    "stable_id",
    "stable_uuid",
    "sanitize_filename",
    "file_extension",
    # Language
    "is_arabic_char",
    "script_ratios",
    "detect_language",
    "detect_languages",
    "normalize_arabic",
    "normalize_for_search",
    "contains_arabic",
    "language_name",
    "is_rtl",
    # Text
    "clean_text",
    "strip_page_artifacts",
    "detect_repeated_lines",
    "remove_lines",
    "normalize_bullets",
    "tokenize",
    "estimate_tokens",
    "truncate",
    "snippet",
    "split_sentences",
    "split_paragraphs",
    "is_meaningful",
    "dedupe_preserve_order",
    # Images
    "PIL_AVAILABLE",
    "open_image",
    "image_size",
    "normalize_image",
    "to_data_url",
    "to_base64",
    "is_probably_blank",
    "is_probably_decorative",
    "crop",
    "ensure_min_size",
    # Retry
    "is_retryable",
    "backoff_delay",
    "retry_call",
    "with_retry",
    # User messages
    "SERVICE_UNAVAILABLE",
    "SERVICE_NOT_CONFIGURED",
    "provider_error_message",
    "public_error_text",
    "public_generation_warning",
    "public_processing_note",
]

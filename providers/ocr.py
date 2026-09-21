"""OCR provider interface and implementations.

OCR is never allowed to abort ingestion: a failed page yields an empty
:class:`OCRResult` carrying the reason, and the document continues. Confidence
is always reported so downstream consumers (and the user) can see how much to
trust a recognised passage.

``OCR_PROVIDER=auto`` (default) resolves to:

1. local Tesseract when it is genuinely installed and usable — cheapest;
2. the vision LLM when an image-capable model is configured — works on
   Streamlit Cloud with no system packages;
3. :class:`NullOCRProvider`, which explains the gap instead of failing silently.
"""

from __future__ import annotations

import io
import json
import re
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from backend.models import (
    Language,
    ProviderError,
    Role,
)
from backend.providers.llm import (
    BaseLLMProvider,
    ImagePart,
    LLMMessage,
)
from backend.utils import (
    detect_language,
    ensure_min_size,
    get_logger,
    normalize_image,
)

logger = get_logger(__name__)

try:  # pragma: no cover - optional dependency
    import pytesseract
    from PIL import Image

    PYTESSERACT_AVAILABLE = True
except Exception:  # pragma: no cover
    pytesseract = None  # type: ignore[assignment]
    Image = None  # type: ignore[assignment]
    PYTESSERACT_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Base interface
# --------------------------------------------------------------------------- #
@dataclass
class OCRResult:
    """Recognised text plus an honest confidence signal."""

    text: str = ""
    confidence: Optional[float] = None
    language: Language = Language.UNKNOWN
    engine: str = ""
    uncertain: bool = False
    error: Optional[str] = None
    word_confidences: List[float] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.text.strip()) and self.error is None

    @classmethod
    def empty(cls, engine: str, error: Optional[str] = None) -> "OCRResult":
        return cls(engine=engine, error=error, uncertain=True)

    def finalize(self, *, min_confidence: float = 0.35) -> "OCRResult":
        if self.text and self.language == Language.UNKNOWN:
            self.language = detect_language(self.text)
        if self.confidence is not None and self.confidence < min_confidence:
            self.uncertain = True
        return self


class BaseOCRProvider(ABC):
    """Contract for every OCR backend."""

    name: str = "base"
    supports_handwriting: bool = False
    supports_arabic: bool = True

    def __init__(self, *, languages: str = "ara+eng", min_confidence: float = 0.35):
        self.languages = languages
        self.min_confidence = min_confidence

    @abstractmethod
    def recognize(self, image: bytes, *, hint: str = "") -> OCRResult:
        """Extract text from an image. Must not raise — return an empty result."""

    def is_available(self) -> bool:
        return True

    def describe(self) -> dict:
        return {
            "provider": self.name,
            "languages": self.languages,
            "handwriting": self.supports_handwriting,
            "available": self.is_available(),
        }


class NullOCRProvider(BaseOCRProvider):
    """No-op backend used when OCR is disabled or nothing is configured."""

    name = "none"

    def recognize(self, image: bytes, *, hint: str = "") -> OCRResult:
        return OCRResult.empty(
            self.name,
            error=(
                "No OCR provider is configured. Set an LLM API key to enable "
                "vision-based OCR, or install pytesseract for local OCR."
            ),
        )

    def is_available(self) -> bool:
        return False


# --------------------------------------------------------------------------- #
# Tesseract
# --------------------------------------------------------------------------- #
class TesseractOCRProvider(BaseOCRProvider):
    name = "tesseract"
    supports_handwriting = False
    supports_arabic = True

    def __init__(
        self,
        *,
        languages: str = "ara+eng",
        min_confidence: float = 0.35,
        tesseract_cmd: str = "",
        psm: int = 3,
    ):
        super().__init__(languages=languages, min_confidence=min_confidence)
        self.psm = psm
        if tesseract_cmd and PYTESSERACT_AVAILABLE:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self._available: Optional[bool] = None

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        if not PYTESSERACT_AVAILABLE:
            self._available = False
            return False
        try:
            pytesseract.get_tesseract_version()
            self._available = True
        except Exception as exc:
            logger.info("Tesseract binary unavailable: %s", exc)
            self._available = False
        return self._available

    def recognize(self, image: bytes, *, hint: str = "") -> OCRResult:
        if not image:
            return OCRResult.empty(self.name, error="empty image")
        if not self.is_available():
            return OCRResult.empty(self.name, error="Tesseract is not installed on this host.")

        try:
            pil_image = Image.open(io.BytesIO(image))
            pil_image.load()
            config = f"--psm {self.psm}"
            data = pytesseract.image_to_data(
                pil_image,
                lang=self.languages,
                config=config,
                output_type=pytesseract.Output.DICT,
            )
        except Exception as exc:
            logger.warning("Tesseract OCR failed: %s", exc)
            return OCRResult.empty(self.name, error=f"OCR failed: {type(exc).__name__}")

        words, confidences = [], []
        for text, raw_conf in zip(data.get("text", []), data.get("conf", [])):
            token = (text or "").strip()
            if not token:
                continue
            try:
                confidence = float(raw_conf)
            except (TypeError, ValueError):
                confidence = -1.0
            if confidence < 0:
                continue
            words.append(token)
            confidences.append(confidence / 100.0)

        if not words:
            return OCRResult.empty(self.name, error="no text detected")

        mean_confidence = sum(confidences) / len(confidences)
        return OCRResult(
            text=" ".join(words),
            confidence=mean_confidence,
            engine=self.name,
            word_confidences=confidences,
        ).finalize(min_confidence=self.min_confidence)


# --------------------------------------------------------------------------- #
# Vision LLM OCR
# --------------------------------------------------------------------------- #
VISION_SYSTEM_PROMPT = """You are a precise OCR engine for Arabic and English documents.

Transcribe ALL visible text from the image, exactly as written.

Rules:
- Preserve the original language and script. Never translate.
- Preserve reading order and line breaks. For Arabic, transcribe right-to-left text correctly.
- Preserve numbers, dates and units exactly as printed. Never normalise or recompute them.
- If a word or region is illegible, write [?] instead of guessing.
- Do not describe the image, do not add commentary, do not summarise.
- If there is no text at all, return an empty string for "text".

Return a single JSON object:
{"text": "<verbatim transcription>", "confidence": <0.0-1.0>, "has_handwriting": <true|false>, "notes": "<short note or empty>"}

"confidence" is your honest self-assessment of transcription accuracy."""

HANDWRITING_HINT = """This image is expected to contain handwriting.
Handwriting recognition is unreliable — be conservative:
- transcribe only what you can actually read;
- use [?] for uncertain words;
- lower "confidence" accordingly (handwriting rarely deserves > 0.8)."""

MAX_IMAGE_EDGE = 1600


class VisionOCRProvider(BaseOCRProvider):
    """OCR backed by whichever multimodal LLM chain is configured."""

    name = "vision"
    supports_handwriting = True
    supports_arabic = True

    def __init__(
        self,
        llm: BaseLLMProvider,
        *,
        languages: str = "ara+eng",
        min_confidence: float = 0.35,
        max_output_tokens: int = 2600,
    ):
        super().__init__(languages=languages, min_confidence=min_confidence)
        self.llm = llm
        self.max_output_tokens = max_output_tokens

    def is_available(self) -> bool:
        return self.llm is not None and self.llm.supports_images()

    def recognize(self, image: bytes, *, hint: str = "") -> OCRResult:
        if not image:
            return OCRResult.empty(self.name, error="empty image")
        if not self.is_available():
            return OCRResult.empty(
                self.name,
                error="The configured model cannot read images, so OCR is unavailable.",
            )

        prepared, media_type = normalize_image(ensure_min_size(image), max_edge=MAX_IMAGE_EDGE)
        system = VISION_SYSTEM_PROMPT
        if hint == "handwriting":
            system = f"{VISION_SYSTEM_PROMPT}\n\n{HANDWRITING_HINT}"

        try:
            response = self.llm.complete(
                [
                    LLMMessage(
                        role=Role.USER,
                        text="Transcribe every piece of text visible in this image.",
                        images=[ImagePart(data=prepared, media_type=media_type)],
                    )
                ],
                system=system,
                temperature=0.0,
                max_output_tokens=self.max_output_tokens,
                json_mode=True,
            )
        except ProviderError as exc:
            logger.warning("Vision OCR unavailable: %s", exc)
            return OCRResult.empty(self.name, error=exc.user_message)
        except Exception as exc:
            logger.exception("Vision OCR failed unexpectedly")
            return OCRResult.empty(self.name, error=f"OCR failed: {type(exc).__name__}")

        return self._parse(response.text)

    def _parse(self, raw: str) -> OCRResult:
        data = _loads(raw)
        if data is None:
            text = raw.strip()
            if not text:
                return OCRResult.empty(self.name, error="empty OCR response")
            return OCRResult(
                text=text, confidence=None, engine=self.name, uncertain=True
            ).finalize(min_confidence=self.min_confidence)

        text = str(data.get("text", "")).strip()
        confidence = _as_float(data.get("confidence"))
        handwriting = bool(data.get("has_handwriting", False))

        result = OCRResult(
            text=text,
            confidence=confidence,
            engine=self.name,
            uncertain=handwriting or (confidence is not None and confidence < self.min_confidence),
        )
        if not text:
            result.error = str(data.get("notes") or "") or None
        return result.finalize(min_confidence=self.min_confidence)


def _loads(raw: str) -> Optional[dict]:
    payload = (raw or "").strip()
    if not payload:
        return None
    if payload.startswith("```"):
        payload = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", payload).strip()
    try:
        data = json.loads(payload)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", payload, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None


def _as_float(value: object) -> Optional[float]:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Factory and cache
# --------------------------------------------------------------------------- #
_cache: Dict[str, BaseOCRProvider] = {}
_lock = threading.Lock()

SUPPORTED_PROVIDERS = ("auto", "vision", "tesseract", "none")


def _get_settings():
    from backend.config import get_settings
    return get_settings()


def build_ocr_provider(settings: Optional["AppSettings"] = None) -> BaseOCRProvider:
    resolved = settings or _get_settings()
    cfg = resolved.ocr
    provider = (cfg.provider or "auto").lower()

    if provider == "none":
        return NullOCRProvider(languages=cfg.languages, min_confidence=cfg.min_confidence)

    if provider in ("tesseract", "local"):
        tesseract = TesseractOCRProvider(
            languages=cfg.languages,
            min_confidence=cfg.min_confidence,
            tesseract_cmd=cfg.tesseract_cmd,
        )
        if tesseract.is_available():
            return tesseract
        logger.warning("OCR_PROVIDER=tesseract but it is unavailable — trying vision OCR")
        return _vision_or_null(resolved)

    if provider == "vision":
        return _vision_or_null(resolved)

    # --- auto ---------------------------------------------------------- #
    tesseract = TesseractOCRProvider(
        languages=cfg.languages,
        min_confidence=cfg.min_confidence,
        tesseract_cmd=cfg.tesseract_cmd,
    )
    if tesseract.is_available():
        logger.info("OCR: using local Tesseract (%s)", cfg.languages)
        return tesseract
    return _vision_or_null(resolved)


def _vision_or_null(settings) -> BaseOCRProvider:
    if not settings.llm.is_configured:
        return NullOCRProvider(
            languages=settings.ocr.languages, min_confidence=settings.ocr.min_confidence
        )
    try:
        from backend.providers.llm import get_llm_provider

        llm = get_llm_provider(settings)
    except Exception as exc:
        logger.warning("Vision OCR unavailable: %s", exc)
        return NullOCRProvider(
            languages=settings.ocr.languages, min_confidence=settings.ocr.min_confidence
        )

    provider = VisionOCRProvider(
        llm,
        languages=settings.ocr.languages,
        min_confidence=settings.ocr.min_confidence,
        max_output_tokens=settings.ocr.max_output_tokens,
    )
    if not provider.is_available():
        logger.warning(
            "Configured model cannot read images — OCR of scanned pages is disabled"
        )
        return NullOCRProvider(
            languages=settings.ocr.languages, min_confidence=settings.ocr.min_confidence
        )
    return provider


def get_ocr_provider(settings: Optional["AppSettings"] = None) -> BaseOCRProvider:
    resolved = settings or _get_settings()
    cfg = resolved.ocr
    key = f"{cfg.provider}|{cfg.languages}|{resolved.llm.chain_label}"
    with _lock:
        provider = _cache.get(key)
        if provider is None:
            provider = build_ocr_provider(resolved)
            _cache[key] = provider
            logger.info("OCR provider ready: %s", provider.name)
        return provider


def reset_ocr_cache() -> None:
    with _lock:
        _cache.clear()


__all__ = [
    "OCRResult",
    "BaseOCRProvider",
    "NullOCRProvider",
    "TesseractOCRProvider",
    "VisionOCRProvider",
    "SUPPORTED_PROVIDERS",
    "build_ocr_provider",
    "get_ocr_provider",
    "reset_ocr_cache",
]

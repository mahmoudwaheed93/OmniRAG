"""Intelligence layer: OCR orchestration, visual understanding, handwriting, tables, layout.

This module sits between the raw providers and the ingestion pipeline, adding
the policies the providers themselves should not own:

* **content-hash caching** — the same page image is never OCR'd twice;
* **confidence policy** — results below the threshold are marked uncertain
  rather than silently trusted;
* **failure containment** — a failed page returns an empty result with a reason,
  never an exception.

Visual understanding produces faithful, searchable *semantic* descriptions while
the original image stays in the file store. Handwriting extraction is deliberately
honest about its unreliability. Tables are stored in multiple forms so different
question types can be answered from them. Layout analysis is deterministic — no
model calls.
"""

from __future__ import annotations

import json
import re
import statistics
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from backend.models import (
    BlockType,
    BoundingBox,
    Language,
    ProviderError,
    SourceKind,
    TableData,
    VisualRef,
)
from backend.utils import (
    clean_text,
    detect_language,
    ensure_min_size,
    is_meaningful,
    is_probably_blank,
    is_probably_decorative,
    is_rtl,
    normalize_image,
    short_hash,
    truncate,
)
from backend.providers.llm import (
    BaseLLMProvider,
    ImagePart,
    LLMMessage,
    llm_operation,
)
from backend.utils import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# OCR Engine
# --------------------------------------------------------------------------- #
from backend.providers.ocr import BaseOCRProvider, OCRResult  # noqa: E402


@dataclass
class OCRStats:
    calls: int = 0
    cache_hits: int = 0
    failures: int = 0
    low_confidence: int = 0


class OCREngine:
    """Caching, policy-aware wrapper around a :class:`BaseOCRProvider`."""

    def __init__(self, provider: BaseOCRProvider, *, min_confidence: float = 0.35):
        self.provider = provider
        self.min_confidence = min_confidence
        self._cache: Dict[str, OCRResult] = {}
        self._lock = threading.Lock()
        self.stats = OCRStats()

    @property
    def available(self) -> bool:
        return self.provider.is_available()

    @property
    def name(self) -> str:
        return self.provider.name

    def recognize(self, image: bytes, *, hint: str = "") -> OCRResult:
        if not image:
            return OCRResult.empty(self.provider.name, error="empty image")

        key = f"{short_hash(image, 24)}|{hint}"
        with self._lock:
            cached = self._cache.get(key)
        if cached is not None:
            self.stats.cache_hits += 1
            return cached

        self.stats.calls += 1
        result = self.provider.recognize(image, hint=hint)

        if result.text:
            result.text = clean_text(result.text)
            if not is_meaningful(result.text, min_chars=3, min_tokens=1):
                result.text = ""
                result.error = result.error or "no meaningful text detected"

        result.finalize(min_confidence=self.min_confidence)

        if not result.ok:
            self.stats.failures += 1
        elif result.uncertain:
            self.stats.low_confidence += 1

        with self._lock:
            self._cache[key] = result
        return result

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()


def build_ocr_engine(settings=None) -> OCREngine:
    from backend.config import get_settings
    from backend.providers.ocr import get_ocr_provider

    resolved = settings or get_settings()
    return OCREngine(
        get_ocr_provider(resolved), min_confidence=resolved.ocr.min_confidence
    )


# --------------------------------------------------------------------------- #
# Vision Analyzer
# --------------------------------------------------------------------------- #
VISION_CACHE_VERSION = "vision-cache-v3"
_shared_cache: Dict[str, "VisualAnalysis"] = {}
_shared_cache_lock = threading.Lock()

VISION_SYSTEM_PROMPT = """You analyse a visual element from a document so it can be found by search and reasoned about later.

Classify the visual as exactly one of:
"chart", "diagram", "table", "screenshot", "photo", "handwriting", "text", "decorative".

Then describe it FAITHFULLY. Never invent details that are not visible.

For a CHART, report: chart type; title; axis labels and units; legend entries; the
data series and their approximate values (read them off the axes — say "approximately");
the overall trend; notable comparisons, maxima, minima and anomalies; and the
conclusion a reader would draw.

For a DIAGRAM / FLOWCHART / TECHNICAL DRAWING, report: every component and its label;
how components connect; arrow directions; the sequence or process flow, step by step;
any hierarchy or grouping; and annotations, dimensions or callouts.

For a TABLE, report its title, column headers, row labels, and the values, preserving
the numeric relationships.

For a SCREENSHOT or PHOTO, describe factually what is shown and any text visible in it.

For HANDWRITING, transcribe what you can actually read and mark unclear words with [?].

Rules:
- Preserve every number, unit, date and proper name EXACTLY as shown. Do not round or recompute.
- Keep the original language of any text. If the visual contains Arabic, keep the Arabic.
- If something is unreadable or ambiguous, say so explicitly. Never guess.
- Write plain prose and short labelled lines. No markdown headings.

Return a single JSON object:
{
  "type": "<one of the categories above>",
  "title": "<title of the visual, or empty>",
  "description": "<the faithful description described above>",
  "text": "<verbatim text visible inside the visual, or empty>",
  "entities": ["<key labels, components, series or column names>"],
  "data_points": ["<'label: value' pairs you can actually read, or empty list>"],
  "confidence": <0.0-1.0>,
  "unreadable": <true|false>
}"""

VISION_USER_PROMPT = "Analyse this visual element from a document."

_TYPE_TO_BLOCK = {
    "chart": BlockType.CHART,
    "diagram": BlockType.DIAGRAM,
    "table": BlockType.TABLE,
    "handwriting": BlockType.HANDWRITING,
    "screenshot": BlockType.IMAGE,
    "photo": BlockType.IMAGE,
    "text": BlockType.IMAGE,
    "decorative": BlockType.IMAGE,
}


@dataclass
class VisualAnalysis:
    """Structured result of analysing one visual."""

    block_type: BlockType = BlockType.IMAGE
    title: str = ""
    description: str = ""
    text: str = ""
    entities: List[str] = field(default_factory=list)
    data_points: List[str] = field(default_factory=list)
    confidence: Optional[float] = None
    unreadable: bool = False
    language: Language = Language.UNKNOWN
    decorative: bool = False
    error: Optional[str] = None
    provider: str = ""
    model: str = ""

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.searchable_text)

    @property
    def searchable_text(self) -> str:
        parts: List[str] = []
        if self.title:
            parts.append(self.title)
        if self.description:
            parts.append(self.description)
        if self.text:
            parts.append(self.text)
        if self.entities:
            parts.append("Labels: " + ", ".join(self.entities[:40]))
        if self.data_points:
            parts.append("Values: " + "; ".join(self.data_points[:40]))
        return "\n".join(p for p in parts if p.strip()).strip()

    @classmethod
    def failed(cls, error: str) -> "VisualAnalysis":
        return cls(error=error, unreadable=True)

    @classmethod
    def skipped_decorative(cls) -> "VisualAnalysis":
        return cls(decorative=True, error="skipped: decorative or blank image")


class VisionAnalyzer:
    """Analyses visuals through the configured multimodal LLM chain."""

    def __init__(
        self,
        llm: Optional[BaseLLMProvider],
        *,
        max_image_edge: int = 1400,
        jpeg_quality: int = 82,
        min_image_pixels: int = 110 * 110,
        max_output_tokens: int = 1200,
        enabled: bool = True,
    ):
        self.llm = llm
        self.max_image_edge = max_image_edge
        self.jpeg_quality = jpeg_quality
        self.min_image_pixels = min_image_pixels
        self.max_output_tokens = max_output_tokens
        self.enabled = enabled
        self._cache = _shared_cache
        self._lock = _shared_cache_lock
        self.calls = 0
        self.cache_hits = 0

    @property
    def available(self) -> bool:
        return bool(self.enabled and self.llm is not None and self.llm.supports_images())

    def analyze(
        self,
        image: bytes,
        *,
        context: str = "",
        expect: Optional[BlockType] = None,
        skip_decorative_check: bool = False,
        document_id: str = "",
        document_hash: str = "",
        page_number: Optional[int] = None,
    ) -> VisualAnalysis:
        if not image:
            return VisualAnalysis.failed("empty image")

        if not skip_decorative_check and is_probably_decorative(
            image, min_pixels=self.min_image_pixels
        ):
            return VisualAnalysis.skipped_decorative()
        if skip_decorative_check and is_probably_blank(image):
            return VisualAnalysis.skipped_decorative()

        if not self.available:
            return VisualAnalysis.failed(
                "Visual understanding is unavailable: no image-capable model is configured."
            )

        neutral_key = short_hash(
            "|".join([
                VISION_CACHE_VERSION,
                document_hash or document_id,
                str(page_number or ""),
                short_hash(image, 32),
                expect.value if expect else "auto",
            ]),
            32,
        )
        with self._lock:
            cached = self._cache.get(neutral_key)
        if cached is not None:
            self.cache_hits += 1
            return cached

        prepared, media_type = normalize_image(
            image, max_edge=self.max_image_edge, jpeg_quality=self.jpeg_quality
        )
        prompt = VISION_USER_PROMPT
        if context:
            prompt = f"{VISION_USER_PROMPT}\n\nSurrounding document context (for naming only, do not copy):\n{context[:600]}"
        if expect is not None:
            prompt += f"\n\nThis element was extracted as a {expect.value}."

        try:
            self.calls += 1
            with llm_operation("vision_analysis"):
                response = self.llm.complete(
                    [
                        LLMMessage(
                            text=prompt,
                            images=[ImagePart(data=prepared, media_type=media_type)],
                        )
                    ],
                    system=VISION_SYSTEM_PROMPT,
                    temperature=0.0,
                    max_output_tokens=self.max_output_tokens,
                    json_mode=True,
                )
        except ProviderError as exc:
            logger.warning("Visual analysis failed: %s", exc)
            return VisualAnalysis.failed(exc.user_message)
        except Exception as exc:
            logger.exception("Unexpected visual-analysis failure")
            return VisualAnalysis.failed(f"visual analysis failed: {type(exc).__name__}")

        analysis = _parse_vision(response.text, expect=expect)
        if analysis.ok:
            analysis.provider = response.provider
            analysis.model = response.model
            provider_key = short_hash(
                f"{neutral_key}|{analysis.provider}|{analysis.model}|{VISION_CACHE_VERSION}",
                32,
            )
            with self._lock:
                self._cache[neutral_key] = analysis
                self._cache[provider_key] = analysis
        return analysis

    def stats(self) -> Dict[str, int]:
        return {"calls": self.calls, "cache_hits": self.cache_hits, "cached": len(self._cache)}

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()


def _parse_vision(raw: str, *, expect: Optional[BlockType] = None) -> VisualAnalysis:
    data = _loads_json(raw)
    if data is None:
        text = (raw or "").strip()
        if not text:
            return VisualAnalysis.failed("empty visual-analysis response")
        return VisualAnalysis(
            block_type=expect or BlockType.IMAGE,
            description=text,
            language=detect_language(text),
        )

    category = str(data.get("type", "")).strip().lower()
    block_type = _TYPE_TO_BLOCK.get(category, expect or BlockType.IMAGE)

    description = str(data.get("description", "")).strip()
    text = str(data.get("text", "")).strip()
    analysis = VisualAnalysis(
        block_type=block_type,
        title=str(data.get("title", "")).strip(),
        description=description,
        text=text,
        entities=_as_str_list(data.get("entities")),
        data_points=_as_str_list(data.get("data_points")),
        confidence=_as_float(data.get("confidence")),
        unreadable=bool(data.get("unreadable", False)),
        decorative=category == "decorative",
        language=detect_language(f"{description}\n{text}"),
    )
    if not analysis.searchable_text:
        analysis.error = "visual analysis produced no usable description"
    return analysis


def _loads_json(raw: str) -> Optional[dict]:
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


def _as_str_list(value: object, limit: int = 60) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for item in value[:limit]:
        text = str(item).strip()
        if text:
            out.append(text[:200])
    return out


def _as_float(value: object) -> Optional[float]:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return None


def build_vision_analyzer(settings=None) -> VisionAnalyzer:
    from backend.config import get_settings

    resolved = settings or get_settings()
    cfg = resolved.vision

    llm: Optional[BaseLLMProvider] = None
    if cfg.enabled and resolved.llm.is_configured:
        try:
            from backend.providers.llm import get_llm_provider

            llm = get_llm_provider(resolved)
        except Exception as exc:
            logger.warning("Vision analyser has no LLM available: %s", exc)

    return VisionAnalyzer(
        llm,
        max_image_edge=cfg.max_image_edge,
        jpeg_quality=cfg.jpeg_quality,
        min_image_pixels=cfg.min_image_pixels,
        max_output_tokens=cfg.analysis_max_output_tokens,
        enabled=cfg.enabled,
    )


# --------------------------------------------------------------------------- #
# Handwriting Extraction
# --------------------------------------------------------------------------- #
HANDWRITING_SUSPICION_THRESHOLD = 0.55
UNREADABLE_MARKER_RATIO = 0.25


@dataclass
class HandwritingResult:
    text: str = ""
    confidence: Optional[float] = None
    language: Language = Language.UNKNOWN
    uncertain: bool = True
    engine: str = ""
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return bool(self.text.strip()) and self.error is None

    @property
    def block_type(self) -> BlockType:
        return BlockType.HANDWRITING

    @property
    def source_kind(self) -> SourceKind:
        return SourceKind.OCR


class HandwritingExtractor:
    """Best-effort handwriting reader built on the OCR and vision engines."""

    def __init__(
        self,
        ocr: Optional[OCREngine] = None,
        vision: Optional[VisionAnalyzer] = None,
    ):
        self.ocr = ocr
        self.vision = vision

    @property
    def available(self) -> bool:
        if self.vision is not None and self.vision.available:
            return True
        return bool(
            self.ocr is not None
            and self.ocr.available
            and self.ocr.provider.supports_handwriting
        )

    def looks_handwritten(self, analysis: Optional[VisualAnalysis], ocr_confidence: Optional[float]) -> bool:
        if analysis is not None and analysis.block_type == BlockType.HANDWRITING:
            return True
        if ocr_confidence is not None and ocr_confidence < HANDWRITING_SUSPICION_THRESHOLD:
            return True
        return False

    def extract(self, image: bytes) -> HandwritingResult:
        if not image:
            return HandwritingResult(error="empty image")

        if self.ocr is not None and self.ocr.available and self.ocr.provider.supports_handwriting:
            result = self.ocr.recognize(image, hint="handwriting")
            if result.ok:
                return _from_ocr(result)
            error = result.error
        else:
            error = "no handwriting-capable OCR provider configured"

        if self.vision is not None and self.vision.available:
            analysis = self.vision.analyze(image, expect=BlockType.HANDWRITING)
            if analysis.ok:
                text = analysis.text or analysis.description
                return HandwritingResult(
                    text=text,
                    confidence=analysis.confidence,
                    language=detect_language(text),
                    uncertain=True,
                    engine="vision",
                )
            error = analysis.error or error

        return HandwritingResult(error=error or "handwriting extraction unavailable")


def _from_ocr(result) -> HandwritingResult:
    text = result.text
    uncertain = True
    if _unreadable_ratio(text) > UNREADABLE_MARKER_RATIO:
        logger.info("Handwriting transcription is mostly unreadable — flagging")
    return HandwritingResult(
        text=text,
        confidence=result.confidence,
        language=result.language if result.language != Language.UNKNOWN else detect_language(text),
        uncertain=uncertain,
        engine=result.engine,
    )


def _unreadable_ratio(text: str) -> float:
    if not text:
        return 1.0
    words = text.split()
    if not words:
        return 1.0
    return sum(1 for w in words if "[?]" in w) / len(words)


# --------------------------------------------------------------------------- #
# Table Extraction
# --------------------------------------------------------------------------- #
_NUMBER_RE = re.compile(r"^[\s\(\)\-+]*[\d٠-٩][\d٠-٩.,\s%٪]*\)?$")
_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
MAX_SUMMARY_COLUMNS = 12
MAX_MARKDOWN_ROWS = 200


def build_table(
    rows: Sequence[Sequence[Any]],
    *,
    has_header: bool = True,
    caption: str = "",
) -> Optional[TableData]:
    cleaned = _clean_rows(rows)
    if not cleaned:
        return None
    if len(cleaned) < 2 and len(cleaned[0]) < 2:
        return None

    header_looks_real = has_header and _looks_like_header(cleaned[0])
    table = TableData.from_rows(cleaned, has_header=header_looks_real)
    if table.n_cols == 0:
        return None

    if len(table.rows) > MAX_MARKDOWN_ROWS:
        from backend.models import _rows_to_markdown

        table.markdown = _rows_to_markdown(
            table.header, table.rows[:MAX_MARKDOWN_ROWS], table.n_cols
        ) + f"\n\n_({len(table.rows) - MAX_MARKDOWN_ROWS} further rows not shown)_"

    table.summary = summarize_table(table, caption=caption)
    return table


def _clean_rows(rows: Sequence[Sequence[Any]]) -> List[List[str]]:
    out: List[List[str]] = []
    for row in rows:
        if row is None:
            continue
        cells = [clean_text(str(cell)) if cell is not None else "" for cell in row]
        cells = [c.replace("\n", " ").strip() for c in cells]
        if any(cells):
            out.append(cells)
    return out


def _looks_like_header(row: Sequence[str]) -> bool:
    values = [c for c in row if c.strip()]
    if not values:
        return False
    numeric = sum(1 for c in values if is_numeric_cell(c))
    return numeric <= len(values) * 0.4


def is_numeric_cell(value: str) -> bool:
    text = value.strip().translate(_ARABIC_DIGITS)
    if not text:
        return False
    return bool(_NUMBER_RE.match(text))


def parse_number(value: str) -> Optional[float]:
    text = value.strip().translate(_ARABIC_DIGITS)
    if not text:
        return None
    negative = text.startswith("(") and text.endswith(")")
    text = text.strip("()").replace(",", "").replace("٪", "").replace("%", "").strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return -number if negative else number


def summarize_table(table: TableData, *, caption: str = "") -> str:
    parts: List[str] = []
    if caption:
        parts.append(f"Table: {truncate(clean_text(caption), 160)}")

    header = table.header or []
    if header:
        shown = [h for h in header if h][:MAX_SUMMARY_COLUMNS]
        suffix = "" if len(header) <= MAX_SUMMARY_COLUMNS else f" (+{len(header) - MAX_SUMMARY_COLUMNS} more)"
        parts.append(f"Columns: {', '.join(shown)}{suffix}.")

    parts.append(f"{table.n_rows} data rows across {table.n_cols} columns.")

    if table.rows:
        labels = [row[0] for row in table.rows[:15] if row and row[0] and not is_numeric_cell(row[0])]
        if labels:
            parts.append(f"Row labels include: {', '.join(labels[:15])}.")

    numeric_columns = _numeric_column_ranges(table)
    if numeric_columns:
        parts.append("Numeric ranges: " + "; ".join(numeric_columns) + ".")

    return " ".join(parts)


def _numeric_column_ranges(table: TableData, limit: int = 6) -> List[str]:
    if not table.rows:
        return []
    out: List[str] = []
    for index in range(min(table.n_cols, 20)):
        values: List[float] = []
        for row in table.rows:
            if index >= len(row):
                continue
            number = parse_number(row[index])
            if number is not None:
                values.append(number)
        if len(values) < 2:
            continue
        name = (
            table.header[index]
            if table.header and index < len(table.header) and table.header[index]
            else f"column {index + 1}"
        )
        out.append(f"{name}: {_fmt(min(values))} to {_fmt(max(values))}")
        if len(out) >= limit:
            break
    return out


def _fmt(value: float) -> str:
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return f"{value:g}"


def table_to_text(table: TableData, *, caption: str = "") -> str:
    parts = []
    if caption:
        parts.append(clean_text(caption))
    if table.summary:
        parts.append(table.summary)
    if table.markdown:
        parts.append(table.markdown)
    return "\n\n".join(p for p in parts if p).strip()


# --------------------------------------------------------------------------- #
# Layout Analysis
# --------------------------------------------------------------------------- #
MIN_CHARS_FOR_DIGITAL_PAGE = 90
SCAN_IMAGE_COVERAGE = 0.55
HEADING_SIZE_RATIO = 1.15


@dataclass
class TextSpan:
    text: str
    size: float = 0.0
    bold: bool = False
    bbox: Optional[BoundingBox] = None
    font: str = ""


@dataclass
class PageLayoutInfo:
    is_scanned: bool = False
    char_count: int = 0
    image_coverage: float = 0.0
    reason: str = ""


def analyze_page_layout(
    *,
    text: str,
    page_area: float,
    image_area: float,
    has_text_layer: bool,
) -> PageLayoutInfo:
    char_count = len(text.strip())
    coverage = (image_area / page_area) if page_area > 0 else 0.0

    if not has_text_layer or char_count == 0:
        return PageLayoutInfo(
            is_scanned=True, char_count=char_count, image_coverage=coverage,
            reason="no text layer",
        )

    if char_count < MIN_CHARS_FOR_DIGITAL_PAGE and coverage >= SCAN_IMAGE_COVERAGE:
        return PageLayoutInfo(
            is_scanned=True, char_count=char_count, image_coverage=coverage,
            reason="sparse text over a full-page image",
        )

    if not is_meaningful(text, min_chars=MIN_CHARS_FOR_DIGITAL_PAGE // 3, min_tokens=3):
        return PageLayoutInfo(
            is_scanned=True, char_count=char_count, image_coverage=coverage,
            reason="text layer contains no meaningful words",
        )

    return PageLayoutInfo(
        is_scanned=False, char_count=char_count, image_coverage=coverage,
        reason="digital text layer",
    )


_NUMBERED_HEADING = re.compile(r"^\s*(?:\d+(?:\.\d+)*|[IVXLC]+\.|[A-Z]\.)\s+\S")
_ALL_CAPS = re.compile(r"^[^a-z]{4,}$")


def body_font_size(spans: Sequence[TextSpan]) -> float:
    sizes: List[float] = []
    for span in spans:
        if span.size > 0 and span.text.strip():
            sizes.extend([span.size] * max(1, len(span.text) // 20))
    return statistics.median(sizes) if sizes else 0.0


def is_heading_span(span: TextSpan, baseline: float) -> bool:
    text = span.text.strip()
    if not text or len(text) > 160:
        return False
    if text.endswith((".", "،", "؛", ";")) and len(text) > 60:
        return False

    if baseline > 0 and span.size >= baseline * HEADING_SIZE_RATIO:
        return True
    if span.bold and len(text) <= 100 and baseline > 0 and span.size >= baseline * 0.98:
        return True
    return False


def looks_like_heading(line: str) -> bool:
    text = line.strip()
    if not text or len(text) > 150:
        return False
    if text.startswith("#"):
        return True
    if _NUMBERED_HEADING.match(text) and len(text) < 100:
        return True
    if _ALL_CAPS.match(text) and 4 <= len(text) <= 90:
        return True
    if text.endswith(":") and len(text) <= 80:
        return True
    return False


def heading_level(line: str) -> int:
    text = line.strip()
    if text.startswith("#"):
        return min(6, len(text) - len(text.lstrip("#")))
    match = _NUMBERED_HEADING.match(text)
    if match:
        return min(6, 1 + match.group(0).strip().count("."))
    return 2


def sort_reading_order(
    items: Sequence[Tuple[BoundingBox, object]],
    *,
    page_width: float = 0.0,
    rtl: bool = False,
    column_tolerance: float = 0.12,
) -> List[object]:
    if not items:
        return []

    entries = [(bbox, payload) for bbox, payload in items if bbox is not None]
    if not entries:
        return [payload for _, payload in items]

    columns = _detect_columns(entries, page_width, column_tolerance)
    if columns <= 1:
        return [
            payload
            for _, payload in sorted(
                entries,
                key=lambda e: (round(e[0].y0, 1), -e[0].x0 if rtl else e[0].x0),
            )
        ]

    width = page_width or max(bbox.x1 for bbox, _ in entries)
    band = width / columns

    def key(entry):
        bbox = entry[0]
        column = min(columns - 1, int(bbox.x0 / band)) if band > 0 else 0
        if rtl:
            column = columns - 1 - column
        return (column, round(bbox.y0, 1), bbox.x0)

    return [payload for _, payload in sorted(entries, key=key)]


def _detect_columns(
    entries: Sequence[Tuple[BoundingBox, object]], page_width: float, tolerance: float
) -> int:
    if len(entries) < 6:
        return 1
    width = page_width or max(bbox.x1 for bbox, _ in entries)
    if width <= 0:
        return 1

    midpoints = sorted((bbox.x0 + bbox.x1) / 2 / width for bbox, _ in entries)
    left = [m for m in midpoints if m < 0.5]
    right = [m for m in midpoints if m >= 0.5]
    if len(left) < 3 or len(right) < 3:
        return 1

    gap = min(right) - max(left)
    return 2 if gap > tolerance else 1


def detect_rtl_page(text: str) -> bool:
    return is_rtl(text)


__all__ = [
    "OCREngine",
    "OCRResult",
    "OCRStats",
    "VisualAnalysis",
    "VisionAnalyzer",
    "HandwritingExtractor",
    "HandwritingResult",
    "TextSpan",
    "PageLayoutInfo",
    "build_ocr_engine",
    "build_vision_analyzer",
    "build_table",
    "is_numeric_cell",
    "parse_number",
    "summarize_table",
    "table_to_text",
    "analyze_page_layout",
    "body_font_size",
    "detect_rtl_page",
    "heading_level",
    "is_heading_span",
    "looks_like_heading",
    "sort_reading_order",
]

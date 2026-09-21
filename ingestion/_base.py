from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional, Protocol, Tuple

from backend.models import (
    BlockType,
    BoundingBox,
    ContentBlock,
    Document,
    FileType,
    IngestionStatus,
    PipelineStage,
    SourceKind,
    VisualRef,
)
from backend.utils import (
    clean_text,
    detect_language,
    get_logger,
    is_meaningful,
    is_probably_blank,
    is_probably_decorative,
    normalize_image,
    stable_id,
)
from backend.config import AppSettings

logger = get_logger(__name__)


class ProgressCallback(Protocol):
    def __call__(self, stage: PipelineStage, progress: float, message: str = "") -> None:
        ...


def _noop_progress(stage: PipelineStage, progress: float, message: str = "") -> None:
    return None


@dataclass
class ProcessingContext:
    """Everything a processor needs, injected rather than imported."""

    session_id: str
    document_id: str
    filename: str
    settings: AppSettings
    file_store: Any  # FileStore
    ocr: Any = None  # Optional[OCREngine]
    vision: Any = None  # Optional[VisionAnalyzer]
    handwriting: Any = None  # Optional[HandwritingExtractor]
    progress: ProgressCallback = _noop_progress
    warnings: List[str] = field(default_factory=list)
    visual_budget: int = 40
    _order: int = 0

    def next_order(self) -> int:
        self._order += 1
        return self._order

    def warn(self, message: str) -> None:
        if message and message not in self.warnings:
            self.warnings.append(message)
            logger.info("[%s] %s", self.filename, message)

    def consume_visual_budget(self) -> bool:
        if self.visual_budget <= 0:
            return False
        self.visual_budget -= 1
        return True

    @property
    def vision_available(self) -> bool:
        return self.vision is not None and self.vision.available

    @property
    def ocr_available(self) -> bool:
        return self.ocr is not None and self.ocr.available


class BaseDocumentProcessor(ABC):
    """Contract for every file-type processor."""

    extensions: Tuple[str, ...] = ()
    file_type: FileType = FileType.UNKNOWN
    display_name: str = "Document"

    @abstractmethod
    def parse(self, data: bytes, ctx: ProcessingContext) -> Document:
        """Parse raw bytes into a canonical :class:`Document`."""

    def supports(self, extension: str) -> bool:
        return extension.lower().lstrip(".") in self.extensions

    def new_document(self, data: bytes, ctx: ProcessingContext) -> Document:
        return Document(
            document_id=ctx.document_id,
            session_id=ctx.session_id,
            filename=ctx.filename,
            file_type=self.file_type,
            size_bytes=len(data),
        )

    def block_id(self, ctx: ProcessingContext, page_number: int, index: int, kind: str) -> str:
        return stable_id(ctx.document_id, str(page_number), kind, str(index))

    def make_text_block(
        self,
        ctx: ProcessingContext,
        *,
        page_number: int,
        text: str,
        index: int,
        block_type: BlockType = BlockType.TEXT,
        source_kind: SourceKind = SourceKind.DIGITAL,
        section: Optional[str] = None,
        bbox: Optional[BoundingBox] = None,
        confidence: Optional[float] = None,
        uncertain: bool = False,
        visual: Optional[VisualRef] = None,
    ) -> Optional[ContentBlock]:
        cleaned = clean_text(text)
        if not cleaned:
            return None
        if block_type in (BlockType.TEXT, BlockType.OCR_TEXT) and not is_meaningful(cleaned):
            return None

        return ContentBlock(
            block_id=self.block_id(ctx, page_number, index, block_type.value),
            document_id=ctx.document_id,
            session_id=ctx.session_id,
            page_number=page_number,
            block_type=block_type,
            source_kind=source_kind,
            text=cleaned,
            parent_section=section,
            bbox=bbox,
            language=detect_language(cleaned),
            confidence=confidence,
            uncertain=uncertain,
            visual=visual,
            order=ctx.next_order(),
        )

    def store_visual(
        self,
        ctx: ProcessingContext,
        image: bytes,
        *,
        origin: str = "embedded",
        page_number: Optional[int] = None,
    ) -> Optional[VisualRef]:
        if not image:
            return None
        try:
            normalized, media_type = normalize_image(
                image,
                max_edge=ctx.settings.vision.max_image_edge,
                jpeg_quality=ctx.settings.vision.jpeg_quality,
            )
            asset = ctx.file_store.put(ctx.session_id, normalized, media_type=media_type)
        except Exception as exc:
            logger.warning("Could not store visual for %s: %s", ctx.filename, exc)
            return None

        from backend.utils import image_size

        width, height = image_size(normalized)
        return VisualRef(
            asset_id=asset.asset_id,
            media_type=media_type,
            width=width or None,
            height=height or None,
            origin=origin,
            page_number=page_number,
        )

    def process_visual(
        self,
        ctx: ProcessingContext,
        image: bytes,
        *,
        page_number: int,
        index: int,
        context_text: str = "",
        origin: str = "embedded",
        expect: Optional[BlockType] = None,
        skip_decorative_check: bool = False,
    ) -> Optional[ContentBlock]:
        if not image:
            return None

        if not skip_decorative_check and is_probably_decorative(
            image, min_pixels=ctx.settings.vision.min_image_pixels
        ):
            return None
        if skip_decorative_check and is_probably_blank(image):
            return None

        analysis = None
        lazy_analysis = ctx.settings.vision.lazy_analysis
        if ctx.vision_available and not lazy_analysis:
            if ctx.consume_visual_budget():
                analysis = ctx.vision.analyze(
                    image,
                    context=context_text,
                    expect=expect,
                    skip_decorative_check=skip_decorative_check,
                    document_id=ctx.document_id,
                    page_number=page_number,
                )
            else:
                ctx.warn(
                    "Visual analysis budget reached for this document — some "
                    "images were indexed without a description."
                )
        elif not lazy_analysis and ctx.vision is not None and not ctx.vision.available:
            ctx.warn(
                "Visual understanding is unavailable (no image-capable model "
                "configured), so charts and diagrams were not described."
            )

        visual = self.store_visual(ctx, image, origin=origin, page_number=page_number)

        if analysis is None or not analysis.ok:
            if analysis is not None and analysis.decorative:
                return None
            if visual is None:
                return None
            placeholder = f"Image on page {page_number} of {ctx.filename}."
            if analysis is not None and analysis.error:
                placeholder += f" (Not analysed: {analysis.error})"
            return ContentBlock(
                block_id=self.block_id(ctx, page_number, index, "image"),
                document_id=ctx.document_id,
                session_id=ctx.session_id,
                page_number=page_number,
                block_type=BlockType.IMAGE,
                source_kind=SourceKind.VISION,
                visual_description=placeholder,
                visual=visual,
                uncertain=True,
                order=ctx.next_order(),
                metadata={"visual_analysis_pending": lazy_analysis},
            )

        block_type = expect or analysis.block_type
        if analysis.block_type in (BlockType.CHART, BlockType.DIAGRAM, BlockType.HANDWRITING):
            block_type = analysis.block_type

        return ContentBlock(
            block_id=self.block_id(ctx, page_number, index, block_type.value),
            document_id=ctx.document_id,
            session_id=ctx.session_id,
            page_number=page_number,
            block_type=block_type,
            source_kind=SourceKind.VISION,
            text=analysis.text,
            visual_description=analysis.searchable_text,
            visual=visual,
            language=analysis.language,
            confidence=analysis.confidence,
            uncertain=analysis.unreadable or block_type == BlockType.HANDWRITING,
            order=ctx.next_order(),
            metadata={"visual_title": analysis.title} if analysis.title else {},
        )

    def finalize(self, document: Document, ctx: ProcessingContext) -> Document:
        document.pages = [p for p in document.pages if p.blocks or p.page_image]
        document.page_count = len(document.pages)
        document.warnings = list(ctx.warnings)
        document.language = detect_language(
            "\n".join(b.search_text for b in document.blocks[:200])
        )
        document.status = IngestionStatus.PARSING
        return document


def elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0

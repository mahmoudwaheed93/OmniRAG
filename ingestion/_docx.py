from __future__ import annotations

import io
from typing import Any, Iterator, List, Optional, Tuple

from backend.models import (
    BlockType,
    ContentBlock,
    CorruptedDocumentError,
    Document,
    EmptyDocumentError,
    FileType,
    Page,
    PipelineStage,
    SourceKind,
    TableData,
)
from backend.utils import (
    clean_text,
    get_logger,
    is_probably_decorative,
)

from ._base import BaseDocumentProcessor, ProcessingContext

logger = get_logger(__name__)

# DOCX-specific constants
SECTION_CHAR_LIMIT = 6000
MIN_IMAGE_BYTES = 2048

try:
    import docx  # python-docx
    from docx.document import Document as DocxDocument
    from docx.oxml.table import CT_Tbl
    from docx.oxml.text.paragraph import CT_P
    from docx.table import Table as DocxTable
    from docx.text.paragraph import Paragraph as DocxParagraph
    PYTHON_DOCX_AVAILABLE = True
except Exception:
    docx = None  # type: ignore[assignment]
    PYTHON_DOCX_AVAILABLE = False


class WordProcessor(BaseDocumentProcessor):
    extensions = ("docx",)
    file_type = FileType.DOCX
    display_name = "Word document"

    def parse(self, data: bytes, ctx: ProcessingContext) -> Document:
        from backend.intelligence import looks_like_heading, build_table, table_to_text

        if not PYTHON_DOCX_AVAILABLE:
            raise CorruptedDocumentError(ctx.filename, "python-docx is not installed")

        ctx.progress(PipelineStage.PARSING, 0.05, "Opening document")
        try:
            source = docx.Document(io.BytesIO(data))
        except Exception as exc:
            raise CorruptedDocumentError(ctx.filename, str(exc)) from exc

        document = self.new_document(data, ctx)
        document.metadata.update(_core_properties(source))

        section_number = 1
        section_title: Optional[str] = None
        section_chars = 0
        page = _new_docx_page(ctx, section_number, section_title)
        index = 0

        items = list(_iter_body(source))
        total = max(1, len(items))

        for position, item in enumerate(items):
            if position % 20 == 0:
                ctx.progress(
                    PipelineStage.EXTRACTING_TEXT,
                    0.1 + 0.5 * (position / total),
                    f"Element {position + 1} of {total}",
                )

            if isinstance(item, DocxParagraph):
                text = clean_text(item.text)
                if not text:
                    continue

                style = (item.style.name if item.style is not None else "") or ""
                is_heading = style.lower().startswith("heading") or style.lower() == "title"
                if not is_heading and looks_like_heading(text) and len(text) < 120:
                    is_heading = style.lower() in ("subtitle",) or text.isupper()

                if is_heading:
                    if page.blocks and section_chars > 0:
                        document.pages.append(page)
                        section_number += 1
                        page = _new_docx_page(ctx, section_number, text)
                        section_chars = 0
                    section_title = text
                    page.label = f"Section {section_number}"
                    page.metadata["title"] = text

                block = self.make_text_block(
                    ctx,
                    page_number=page.page_number,
                    text=text,
                    index=index,
                    block_type=BlockType.HEADING if is_heading else BlockType.TEXT,
                    section=None if is_heading else section_title,
                )
                index += 1
                if block is not None:
                    page.blocks.append(block)
                    section_chars += len(text)

            elif isinstance(item, DocxTable):
                block = _docx_table_block(self, item, ctx, page.page_number, index, section_title)
                index += 1
                if block is not None:
                    page.blocks.append(block)
                    section_chars += 400

            if section_chars >= SECTION_CHAR_LIMIT:
                document.pages.append(page)
                section_number += 1
                page = _new_docx_page(ctx, section_number, section_title)
                section_chars = 0

        if page.blocks:
            document.pages.append(page)

        _process_docx_images(self, source, document, ctx)

        document = self.finalize(document, ctx)
        if not document.blocks:
            raise EmptyDocumentError(ctx.filename)
        return document


def _new_docx_page(ctx: ProcessingContext, number: int, title: Optional[str]) -> Page:
    page = Page(
        document_id=ctx.document_id,
        session_id=ctx.session_id,
        page_number=number,
        label=f"Section {number}",
    )
    if title:
        page.metadata["title"] = title
    return page


def _docx_table_block(
    processor: BaseDocumentProcessor,
    table: Any,
    ctx: ProcessingContext,
    page_number: int,
    index: int,
    section: Optional[str],
) -> Optional[ContentBlock]:
    from backend.intelligence import build_table, table_to_text

    try:
        rows = [[cell.text for cell in row.cells] for row in table.rows]
    except Exception as exc:
        logger.debug("Could not read a DOCX table: %s", exc)
        return None

    data = build_table(rows, has_header=True)
    if data is None:
        return None

    return ContentBlock(
        block_id=processor.block_id(ctx, page_number, index, "table"),
        document_id=ctx.document_id,
        session_id=ctx.session_id,
        page_number=page_number,
        block_type=BlockType.TABLE,
        source_kind=SourceKind.STRUCTURED,
        text=table_to_text(data),
        table=data,
        parent_section=section,
        order=ctx.next_order(),
    )


def _process_docx_images(processor: BaseDocumentProcessor, source: Any, document: Document, ctx: ProcessingContext) -> None:
    if not ctx.settings.vision.enabled:
        return
    try:
        parts = [
            part for part in source.part.package.parts
            if str(getattr(part, "content_type", "")).startswith("image/")
        ]
    except Exception:
        return
    if not parts:
        return

    target = document.pages[0] if document.pages else _new_docx_page(ctx, 1, None)
    if not document.pages:
        document.pages.append(target)

    context_text = " ".join(b.text for b in target.blocks if b.text)[:500]
    ctx.progress(PipelineStage.ANALYZING_VISUALS, 0.65, f"{len(parts)} embedded images")

    for index, part in enumerate(parts[:20]):
        try:
            image_bytes = part.blob
        except Exception:
            continue
        if not image_bytes or len(image_bytes) < MIN_IMAGE_BYTES:
            continue
        if is_probably_decorative(image_bytes, min_pixels=ctx.settings.vision.min_image_pixels):
            continue

        block = processor.process_visual(
            ctx, image_bytes,
            page_number=target.page_number,
            index=200 + index,
            context_text=context_text,
            origin="embedded",
        )
        if block is not None:
            target.blocks.append(block)


def _iter_body(source: "DocxDocument") -> Iterator[Any]:
    body = source.element.body
    for child in body.iterchildren():
        if isinstance(child, CT_P):
            yield DocxParagraph(child, source)
        elif isinstance(child, CT_Tbl):
            yield DocxTable(child, source)


def _core_properties(source: Any) -> dict:
    try:
        props = source.core_properties
    except Exception:
        return {}
    out = {}
    for name in ("title", "author", "subject", "category", "comments"):
        value = getattr(props, name, None)
        if value:
            out[name] = str(value)[:300]
    return out

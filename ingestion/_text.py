from __future__ import annotations

import re
from typing import List, Optional, Tuple

from backend.models import (
    BlockType,
    ContentBlock,
    Document,
    EmptyDocumentError,
    FileType,
    Page,
    PipelineStage,
    SourceKind,
)
from backend.utils import (
    clean_text,
    get_logger,
)

from ._base import BaseDocumentProcessor, ProcessingContext

logger = get_logger(__name__)

# Text-specific patterns and constants
ATX_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")
SETEXT_UNDERLINE = re.compile(r"^(=+|-{2,})\s*$")
FENCE = re.compile(r"^\s*(```|~~~)")
TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
TABLE_SEPARATOR = re.compile(r"^\s*\|[\s:\-\|]+\|\s*$")
PAGE_CHAR_LIMIT = 4500
_ENCODINGS = ("utf-8-sig", "utf-8", "utf-16", "cp1256", "iso-8859-6", "cp1252", "latin-1")


class TextProcessor(BaseDocumentProcessor):
    extensions = ("txt", "md", "markdown", "text")
    file_type = FileType.TXT
    display_name = "Text document"

    def parse(self, data: bytes, ctx: ProcessingContext) -> Document:
        from backend.intelligence import build_table, table_to_text

        if not data.strip():
            raise EmptyDocumentError(ctx.filename)

        ctx.progress(PipelineStage.PARSING, 0.1, "Decoding text")
        text, encoding = decode_text(data)
        if not text.strip():
            raise EmptyDocumentError(ctx.filename)

        is_markdown = ctx.filename.lower().endswith((".md", ".markdown"))
        document = self.new_document(data, ctx)
        document.file_type = FileType.MARKDOWN if is_markdown else FileType.TXT
        document.metadata["encoding"] = encoding
        if encoding not in ("utf-8", "utf-8-sig"):
            ctx.warn(f"File was decoded as {encoding} (not UTF-8); check for garbled characters.")

        ctx.progress(PipelineStage.EXTRACTING_TEXT, 0.4, "Splitting sections")
        elements = _parse_elements(text) if is_markdown else _parse_plain(text)

        page_number = 1
        page = _new_text_page(ctx, page_number)
        section: Optional[str] = None
        chars = 0
        index = 0

        for kind, payload in elements:
            if kind == "heading":
                level, heading_text = payload
                if chars >= PAGE_CHAR_LIMIT and page.blocks:
                    document.pages.append(page)
                    page_number += 1
                    page = _new_text_page(ctx, page_number)
                    chars = 0
                section = heading_text
                block = self.make_text_block(
                    ctx, page_number=page_number, text=heading_text, index=index, block_type=BlockType.HEADING,
                )
                if block is not None:
                    block.metadata["level"] = level
                    page.blocks.append(block)
                    chars += len(heading_text)

            elif kind == "table":
                data_table = build_table(payload, has_header=True)
                if data_table is not None:
                    page.blocks.append(
                        ContentBlock(
                            block_id=self.block_id(ctx, page_number, index, "table"),
                            document_id=ctx.document_id,
                            session_id=ctx.session_id,
                            page_number=page_number,
                            block_type=BlockType.TABLE,
                            source_kind=SourceKind.STRUCTURED,
                            text=table_to_text(data_table),
                            table=data_table,
                            parent_section=section,
                            order=ctx.next_order(),
                        )
                    )
                    chars += 400

            else:
                block = self.make_text_block(
                    ctx, page_number=page_number, text=payload, index=index, section=section,
                )
                if block is not None:
                    page.blocks.append(block)
                    chars += len(payload)

            index += 1
            if chars >= PAGE_CHAR_LIMIT * 1.6 and page.blocks:
                document.pages.append(page)
                page_number += 1
                page = _new_text_page(ctx, page_number)
                chars = 0

        if page.blocks:
            document.pages.append(page)

        document = self.finalize(document, ctx)
        if not document.blocks:
            raise EmptyDocumentError(ctx.filename)

        if len(document.pages) == 1:
            document.pages[0].label = "Document"
        return document


def _new_text_page(ctx: ProcessingContext, number: int) -> Page:
    return Page(
        document_id=ctx.document_id,
        session_id=ctx.session_id,
        page_number=number,
        label=f"Part {number}",
    )


def decode_text(data: bytes) -> Tuple[str, str]:
    for encoding in _ENCODINGS:
        try:
            decoded = data.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
        if decoded.count("\ufffd") > max(3, len(decoded) // 500):
            continue
        return decoded, encoding
    return data.decode("utf-8", errors="replace"), "utf-8 (with replacements)"


def _parse_plain(text: str) -> List[Tuple[str, object]]:
    return [("paragraph", p) for p in split_paragraphs(clean_text(text, join_soft_breaks=False))]


def _parse_elements(text: str) -> List[Tuple[str, object]]:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    elements: List[Tuple[str, object]] = []
    buffer: List[str] = []
    table_buffer: List[str] = []
    in_fence = False
    fence_marker = ""

    def flush_paragraph() -> None:
        if buffer:
            content = clean_text("\n".join(buffer), join_soft_breaks=False)
            for paragraph in split_paragraphs(content):
                elements.append(("paragraph", paragraph))
            buffer.clear()

    def flush_table() -> None:
        if table_buffer:
            rows = _markdown_table_rows(table_buffer)
            if rows:
                elements.append(("table", rows))
            else:
                elements.append(("paragraph", "\n".join(table_buffer)))
            table_buffer.clear()

    index = 0
    while index < len(lines):
        line = lines[index]

        fence = FENCE.match(line)
        if fence:
            if in_fence and line.strip().startswith(fence_marker):
                buffer.append(line)
                elements.append(("paragraph", "\n".join(buffer)))
                buffer.clear()
                in_fence = False
            elif not in_fence:
                flush_paragraph()
                flush_table()
                in_fence = True
                fence_marker = fence.group(1)
                buffer.append(line)
            index += 1
            continue

        if in_fence:
            buffer.append(line)
            index += 1
            continue

        if TABLE_ROW.match(line):
            flush_paragraph()
            table_buffer.append(line)
            index += 1
            continue
        flush_table()

        heading = ATX_HEADING.match(line)
        if heading:
            flush_paragraph()
            elements.append(("heading", (len(heading.group(1)), heading.group(2).strip())))
            index += 1
            continue

        if (
            line.strip()
            and index + 1 < len(lines)
            and SETEXT_UNDERLINE.match(lines[index + 1])
            and len(line.strip()) < 120
        ):
            flush_paragraph()
            level = 1 if lines[index + 1].strip().startswith("=") else 2
            elements.append(("heading", (level, line.strip())))
            index += 2
            continue

        buffer.append(line)
        index += 1

    flush_paragraph()
    flush_table()
    return elements


def _markdown_table_rows(lines: List[str]) -> List[List[str]]:
    rows: List[List[str]] = []
    for line in lines:
        if TABLE_SEPARATOR.match(line):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if any(cells):
            rows.append(cells)
    return rows if len(rows) >= 2 else []


# Re-export text helpers needed by other modules
from backend.utils import (  # noqa: E402
    split_paragraphs,
    image_size,
)

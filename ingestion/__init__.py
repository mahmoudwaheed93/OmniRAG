"""Ingestion pipeline: file-type routing, processors, and shared machinery.

Every processor turns raw bytes into the same canonical
:class:`~backend.models.Document`, so downstream stages never branch on
file type. The base class owns the policies that must be identical across
formats: the visual-analysis budget, storing originals in the file store,
per-block error containment, and stable id generation.

Adding a format means writing one processor and registering it in the router —
nothing else in the pipeline changes.

Validation happens *before* any parsing: extension allow-list, size limit,
emptiness, and content-sniffing to catch a file whose extension lies about its
type. Uploaded files are only ever read as data — nothing is executed.
"""

from ._base import (
    BaseDocumentProcessor,
    ProcessingContext,
    ProgressCallback,
    elapsed_ms,
)
from ._pdf import (
    PDFProcessor,
    _extract_medical_metadata,
)
from ._docx import WordProcessor
from ._pptx import PowerPointProcessor
from ._image import ImageProcessor
from ._text import (
    TextProcessor,
    decode_text,
)
from ._router import (
    DocumentRouter,
    FileValidation,
    PROCESSOR_CLASSES,
    get_router,
    reset_router,
)

# Re-export text helpers needed by other modules
from backend.utils import (  # noqa: E402
    detect_repeated_lines,
    remove_lines,
    split_paragraphs,
    image_size,
)

__all__ = [
    "BaseDocumentProcessor",
    "ProcessingContext",
    "ProgressCallback",
    "PDFProcessor",
    "WordProcessor",
    "PowerPointProcessor",
    "ImageProcessor",
    "TextProcessor",
    "DocumentRouter",
    "FileValidation",
    "PROCESSOR_CLASSES",
    "elapsed_ms",
    "decode_text",
    "get_router",
    "reset_router",
    "_extract_medical_metadata",
    "detect_repeated_lines",
    "remove_lines",
    "split_paragraphs",
    "image_size",
]

"""Service layer: engine, ingestion, chat and conversation history.

This module owns the composition root and the orchestration logic that sits
between the UI and the RAG/ingestion pipelines.
"""

from __future__ import annotations

import os
import subprocess
import time
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from backend.config import AppSettings, get_settings
from backend.models import (
    BlockType,
    ChatMessage,
    Document,
    DocumentSummary,
    EmbeddingError,
    FileType,
    IngestionError,
    IngestionResult,
    IngestionStatus,
    Language,
    MissingCredentialError,
    OmniRAGError,
    PipelineStage,
    ProviderCapabilityError,
    ProviderError,
    Role,
    VectorStoreError,
)
from backend.intelligence import (
    HandwritingExtractor,
    OCREngine,
    VisionAnalyzer,
    build_ocr_engine,
    build_vision_analyzer,
)
from backend.providers.embeddings import BaseEmbeddingProvider
from backend.providers.llm import BaseLLMProvider
from backend.rag.chunking import Chunker
from backend.rag.embeddings import EmbeddingPipeline
from backend.rag.vector_store import BaseVectorStore, build_vector_store
from backend.rag.generation import AnswerGenerator, GenerationRequest
from backend.rag.retrieval import RetrievalRequest, Retriever
from backend.ingestion import ProcessingContext
from backend.storage import (
    FileStore,
    get_file_store,
    get_registry,
    require_session_id,
    new_session_id,
    DocumentRegistry,
)
from backend.utils import (
    configure_logging,
    content_hash,
    estimate_tokens,
    get_logger,
    sanitize_filename,
    stable_id,
)
from backend.utils import (
    SERVICE_NOT_CONFIGURED,
    provider_error_message,
    public_error_text,
    public_generation_warning,
)

logger = get_logger(__name__)

ProgressFn = Callable[[PipelineStage, float, str], None]


def _noop(stage: PipelineStage, progress: float, message: str = "") -> None:
    return None


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
@dataclass
class EngineStatus:
    ready: bool
    issues: List[str]
    warnings: List[str]
    llm_chain: str
    llm_available: bool
    embedding_provider: str
    embedding_model: str
    vector_store: str
    reranker: str
    ocr_provider: str
    vision_available: bool


class OmniRAGEngine:
    """Holds every long-lived component; safe to share across sessions."""

    def __init__(self, settings: Optional[AppSettings] = None):
        self.settings = settings or get_settings()
        self.revision = _runtime_revision()
        configure_logging(self.settings.log_level)
        logger.info(
            "Runtime revision=%s LLM configuration chain=%s LLM_MAX_OUTPUT_TOKENS=%d "
            "LLM_EXHAUSTIVE_MAX_OUTPUT_TOKENS=%d",
            self.revision,
            self.settings.llm.chain_label,
            self.settings.llm.max_output_tokens,
            self.settings.llm.exhaustive_max_output_tokens,
        )

        self._lock = threading.RLock()
        self._llm: Optional[BaseLLMProvider] = None
        self._embeddings: Optional[BaseEmbeddingProvider] = None
        self._vector_store: Optional[BaseVectorStore] = None
        self._ocr: Optional[OCREngine] = None
        self._vision: Optional[VisionAnalyzer] = None
        self._handwriting: Optional[HandwritingExtractor] = None
        self._chunker: Optional[Chunker] = None
        self._file_store: Optional[FileStore] = None
        self._registry: Optional[DocumentRegistry] = None

    @property
    def file_store(self) -> FileStore:
        with self._lock:
            if self._file_store is None:
                self._file_store = get_file_store(self.settings.workspace_dir or None)
            return self._file_store

    @property
    def registry(self) -> DocumentRegistry:
        with self._lock:
            if self._registry is None:
                self._registry = get_registry()
            return self._registry

    @property
    def chunker(self) -> Chunker:
        with self._lock:
            if self._chunker is None:
                self._chunker = Chunker(self.settings.chunking)
            return self._chunker

    @property
    def vector_store(self) -> BaseVectorStore:
        with self._lock:
            if self._vector_store is None:
                self._vector_store = build_vector_store(self.settings.vector_store)
            return self._vector_store

    @property
    def llm(self) -> Optional[BaseLLMProvider]:
        with self._lock:
            if self._llm is None and self.settings.llm.is_configured:
                try:
                    from backend.providers.llm import get_llm_provider
                    self._llm = get_llm_provider(self.settings)
                except Exception as exc:
                    logger.warning("LLM unavailable: %s", exc)
            return self._llm

    @property
    def embeddings(self) -> BaseEmbeddingProvider:
        with self._lock:
            if self._embeddings is None:
                from backend.providers.embeddings import get_embedding_provider
                self._embeddings = get_embedding_provider(self.settings)
            return self._embeddings

    @property
    def embedding_pipeline(self) -> EmbeddingPipeline:
        return EmbeddingPipeline(self.embeddings)

    @property
    def ocr(self) -> OCREngine:
        with self._lock:
            if self._ocr is None:
                self._ocr = build_ocr_engine(self.settings)
            return self._ocr

    @property
    def vision(self) -> VisionAnalyzer:
        with self._lock:
            if self._vision is None:
                self._vision = build_vision_analyzer(self.settings)
            return self._vision

    @property
    def handwriting(self) -> HandwritingExtractor:
        with self._lock:
            if self._handwriting is None:
                self._handwriting = HandwritingExtractor(self.ocr, self.vision)
            return self._handwriting

    def status(self) -> EngineStatus:
        try:
            reranker_name = self._reranker_name()
        except Exception:
            reranker_name = "heuristic"
        try:
            ocr_name = self.ocr.name
        except Exception:
            ocr_name = "none"
        try:
            vision_available = self.vision.available
        except Exception:
            vision_available = False
        return EngineStatus(
            ready=self.settings.is_ready,
            issues=self.settings.validation_issues(),
            warnings=self.settings.warnings(),
            llm_chain=self.settings.llm.chain_label,
            llm_available=self.llm is not None,
            embedding_provider=self.settings.embedding.provider,
            embedding_model=self.settings.embedding.model,
            vector_store=self.vector_store.name,
            reranker=reranker_name,
            ocr_provider=ocr_name,
            vision_available=vision_available,
        )

    def _reranker_name(self) -> str:
        from backend.providers.rerank import get_reranker
        return get_reranker(self.settings).name

    def provider_stats(self) -> Dict[str, Any]:
        llm = self._llm
        stats = getattr(llm, "stats", None)
        if stats is None:
            return {}
        return {
            "calls": stats.calls,
            "failovers": stats.failovers,
            "by_provider": dict(stats.by_provider),
            "last_provider": stats.last_provider,
            "last_model": stats.last_model,
            "last_attempts": list(stats.last_attempts),
        }

    def clear_session(self, session_id: str) -> Dict[str, int]:
        from backend.rag.retrieval import get_bm25_cache
        removed_vectors = 0
        try:
            removed_vectors = self.vector_store.delete_session(session_id)
        except Exception as exc:
            logger.warning("Could not clear vectors for the session: %s", exc)
        removed_files = 0
        try:
            removed_files = self.file_store.delete_session(session_id)
        except Exception as exc:
            logger.warning("Could not clear files for the session: %s", exc)
        document_ids = self.registry.clear(session_id)
        get_bm25_cache().invalidate(session_id)
        logger.info(
            "Cleared session: %d documents, %d vectors, %d assets",
            len(document_ids), removed_vectors, removed_files,
        )
        return {"documents": len(document_ids), "vectors": removed_vectors, "assets": removed_files}

    def cleanup_expired_sessions(self) -> int:
        expired = self.registry.expired_sessions(self.settings.session_ttl_minutes)
        for session_id in expired:
            try:
                self.clear_session(session_id)
                self.registry.drop_session(session_id)
            except Exception as exc:
                logger.warning("Cleanup failed for an expired session: %s", exc)
        return len(expired)


_engine: Optional[OmniRAGEngine] = None
_engine_lock = threading.Lock()


def _runtime_revision() -> str:
    for name in ("STREAMLIT_GIT_COMMIT", "GIT_COMMIT", "SOURCE_VERSION"):
        value = os.environ.get(name, "").strip()
        if value:
            return value[:12]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"],
            stderr=subprocess.DEVNULL, text=True, timeout=2,
        ).strip()
    except Exception:
        return "unknown"


def get_engine(settings: Optional[AppSettings] = None) -> OmniRAGEngine:
    global _engine
    resolved = settings or get_settings()
    with _engine_lock:
        if _engine is None or _engine.settings != resolved:
            if _engine is not None:
                logger.warning("Runtime configuration changed; rebuilding cached engine and providers")
            _engine = OmniRAGEngine(resolved)
        return _engine


def reset_engine() -> None:
    global _engine
    with _engine_lock:
        _engine = None


# --------------------------------------------------------------------------- #
# Ingestion service
# --------------------------------------------------------------------------- #
@dataclass
class ReindexReport:
    reindexed: List[str] = None  # type: ignore[assignment]
    missing_source: List[str] = None  # type: ignore[assignment]
    failed: List[tuple] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.reindexed = self.reindexed or []
        self.missing_source = self.missing_source or []
        self.failed = self.failed or []

    @property
    def total(self) -> int:
        return len(self.reindexed) + len(self.missing_source) + len(self.failed)

    @property
    def ok(self) -> bool:
        return not self.failed and not self.missing_source


@dataclass
class UploadedFile:
    name: str
    data: bytes

    @property
    def size(self) -> int:
        return len(self.data)


class IngestionService:
    """Runs the full ingestion pipeline for one session."""

    def __init__(self, engine: OmniRAGEngine, *, router=None):
        self.engine = engine
        from backend.ingestion import get_router
        self.router = router or get_router()

    @property
    def settings(self) -> AppSettings:
        return self.engine.settings

    def ingest(
        self, session_id: str, upload: UploadedFile, *, progress: ProgressFn = _noop, force: bool = False,
    ) -> IngestionResult:
        session_id = require_session_id(session_id)
        started = time.perf_counter()
        safe_name = sanitize_filename(upload.name)
        result = IngestionResult(filename=safe_name, status=IngestionStatus.PENDING)

        try:
            progress(PipelineStage.UPLOADING, 0.02, "Validating file")
            validation = self.router.validate(upload.name, upload.data, settings=self.settings)
            processor = self.router.route(validation.safe_filename)

            digest = content_hash(upload.data)
            existing = self.engine.registry.find_by_hash(session_id, digest)
            if existing is not None and not force:
                logger.info("Skipping duplicate upload: %s", safe_name)
                result.status = IngestionStatus.DUPLICATE
                result.document_id = existing.document_id
                result.duplicate_of = existing.filename
                result.summary = existing
                result.chunk_count = existing.chunk_count
                result.page_count = existing.page_count
                result.duration_s = time.perf_counter() - started
                progress(PipelineStage.READY, 1.0, "Already indexed")
                return result

            document_id = stable_id(session_id, digest)
            summary = DocumentSummary(
                document_id=document_id, session_id=session_id,
                filename=validation.safe_filename, file_type=validation.file_type,
                size_bytes=validation.size_bytes, content_hash=digest,
                status=IngestionStatus.PARSING,
            )
            self.engine.registry.add(summary)
            result.document_id = document_id

            ctx = ProcessingContext(
                session_id=session_id, document_id=document_id,
                filename=validation.safe_filename, settings=self.settings,
                file_store=self.engine.file_store, ocr=self.engine.ocr,
                vision=self.engine.vision, handwriting=self.engine.handwriting,
                progress=progress,
                visual_budget=self.settings.vision.max_images_per_document,
            )

            try:
                asset = self.engine.file_store.put(
                    session_id, upload.data, media_type=_media_type(validation.file_type)
                )
                source_asset_id = asset.asset_id
            except Exception:
                source_asset_id = None

            progress(PipelineStage.PARSING, 0.05, f"Reading {processor.display_name}")
            document = processor.parse(upload.data, ctx)
            document.content_hash = digest
            document.source_asset_id = source_asset_id
            document.size_bytes = validation.size_bytes

            summary = _summarize(document, summary)
            summary.source_asset_id = source_asset_id
            summary.status = IngestionStatus.CHUNKING
            self.engine.registry.update(summary)

            progress(PipelineStage.CHUNKING, 0.72, "Generating chunks")
            chunks = self.engine.chunker.chunk_document(document)
            if not chunks:
                raise IngestionError(
                    f"{safe_name} produced no chunks",
                    user_message=f"**{safe_name}** contained no indexable content after processing.",
                )

            progress(PipelineStage.EMBEDDING, 0.80, f"Embedding {len(chunks)} chunks")
            summary.status = IngestionStatus.EMBEDDING
            self.engine.registry.update(summary)

            embedded = self.engine.embedding_pipeline.embed_chunks(chunks)
            embeddings = self.engine.embeddings
            if getattr(embeddings, "fallback_active", False):
                ctx.warn(
                    "The configured embedding service was unavailable, so this "
                    "document uses offline hash embeddings. Search remains "
                    "available, but semantic and cross-lingual quality is reduced."
                )
            if not embedded.chunks:
                raise EmbeddingError(
                    "No chunk could be embedded",
                    user_message="Embeddings could not be created for this document. Check the embedding provider configuration.",
                )
            if embedded.failed:
                ctx.warn(f"{len(embedded.failed)} chunk(s) could not be embedded and are not searchable.")

            progress(PipelineStage.INDEXING, 0.92, "Indexing")
            summary.status = IngestionStatus.INDEXING
            self.engine.registry.update(summary)

            store = self.engine.vector_store
            store.ensure_collection(embedded.dimensions)
            written = store.upsert(session_id, embedded.chunks, embedded.vectors)
            from backend.rag.retrieval import get_bm25_cache
            get_bm25_cache().invalidate(session_id)

            summary.chunk_count = written
            summary.warnings = list(ctx.warnings)
            summary.status = IngestionStatus.READY
            self.engine.registry.update(summary)

            result.status = IngestionStatus.READY
            result.summary = summary
            result.chunk_count = written
            result.page_count = summary.page_count
            result.warnings = list(ctx.warnings)
            progress(PipelineStage.READY, 1.0, "Ready")

            logger.info(
                "Indexed %s: %d pages, %d chunks, %d visual blocks",
                safe_name, summary.page_count, written, summary.visual_block_count,
            )

        except OmniRAGError as exc:
            logger.warning("Ingestion failed for %s: %s", safe_name, exc.detail or exc)
            result.status = IngestionStatus.FAILED
            result.error = exc.user_message
            self._mark_failed(session_id, result.document_id, exc.user_message)
        except Exception as exc:
            logger.exception("Unexpected ingestion failure for %s", safe_name)
            message = f"**{safe_name}** could not be processed ({type(exc).__name__}). Please check the file and try again."
            result.status = IngestionStatus.FAILED
            result.error = message
            self._mark_failed(session_id, result.document_id, message)

        result.duration_s = time.perf_counter() - started
        return result

    def ingest_many(
        self, session_id: str, uploads: Sequence[UploadedFile],
        *, progress: Optional[Callable[[int, int, str, PipelineStage, float], None]] = None,
    ) -> List[IngestionResult]:
        results: List[IngestionResult] = []
        total = len(uploads)
        for position, upload in enumerate(uploads):
            def file_progress(
                stage: PipelineStage, value: float, message: str = "",
                _position: int = position, _name: str = upload.name,
            ) -> None:
                if progress is not None:
                    progress(_position, total, _name, stage, value)
            results.append(self.ingest(session_id, upload, progress=file_progress))
        return results

    def reindex(self, session_id: str, *, progress=None) -> "ReindexReport":
        session_id = require_session_id(session_id)
        report = ReindexReport()
        summaries = self.engine.registry.list(session_id)
        total = len(summaries)
        for position, summary in enumerate(summaries):
            if progress is not None:
                progress(position, total, summary.filename)
            data = None
            if summary.source_asset_id:
                try:
                    data = self.engine.file_store.get(summary.source_asset_id)
                except Exception as exc:
                    logger.warning("Could not read stored source bytes: %s", exc)
            if not data:
                report.missing_source.append(summary.filename)
                continue
            try:
                self.engine.vector_store.delete_document(session_id, summary.document_id)
            except Exception as exc:
                logger.warning("Could not clear old vectors for re-index: %s", exc)
            self.engine.registry.remove(session_id, summary.document_id)
            result = self.ingest(session_id, UploadedFile(name=summary.filename, data=data), force=True)
            if result.status == IngestionStatus.READY:
                report.reindexed.append(summary.filename)
            else:
                report.failed.append((summary.filename, result.error or "failed"))
        from backend.rag.retrieval import get_bm25_cache
        get_bm25_cache().invalidate(session_id)
        if progress is not None:
            progress(total, total, "")
        logger.info(
            "Re-index complete: %d rebuilt, %d missing source, %d failed",
            len(report.reindexed), len(report.missing_source), len(report.failed),
        )
        return report

    def remove_document(self, session_id: str, document_id: str) -> bool:
        session_id = require_session_id(session_id)
        try:
            self.engine.vector_store.delete_document(session_id, document_id)
        except VectorStoreError as exc:
            logger.warning("Could not delete vectors: %s", exc.detail)
            return False
        self.engine.registry.remove(session_id, document_id)
        from backend.rag.retrieval import get_bm25_cache
        get_bm25_cache().invalidate(session_id)
        return True

    def clear_session(self, session_id: str) -> dict:
        return self.engine.clear_session(require_session_id(session_id))

    def _mark_failed(self, session_id: str, document_id: Optional[str], message: str) -> None:
        if not document_id:
            return
        summary = self.engine.registry.get(session_id, document_id)
        if summary is None:
            return
        summary.status = IngestionStatus.FAILED
        summary.error = message
        summary.content_hash = ""
        self.engine.registry.update(summary)


def _summarize(document: Document, summary: DocumentSummary) -> DocumentSummary:
    blocks = document.blocks
    summary.page_count = document.page_count
    summary.block_count = len(blocks)
    summary.visual_block_count = sum(1 for b in blocks if b.has_visual)
    summary.table_count = sum(1 for b in blocks if b.block_type == BlockType.TABLE)
    summary.language = document.language
    summary.file_type = document.file_type
    summary.warnings = list(document.warnings)
    medical_md = getattr(document, "_medical_metadata", None)
    if medical_md:
        summary.medical_metadata = medical_md
    return summary


def _media_type(file_type: FileType) -> str:
    return {
        FileType.PDF: "application/pdf",
        FileType.DOCX: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        FileType.PPTX: "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        FileType.IMAGE: "image/png",
        FileType.TXT: "text/plain",
        FileType.MARKDOWN: "text/markdown",
    }.get(file_type, "application/octet-stream")


# --------------------------------------------------------------------------- #
# Chat service
# --------------------------------------------------------------------------- #
@dataclass
class ChatRequest:
    question: str
    session_id: str
    document_ids: Optional[Sequence[str]] = None
    history: Sequence[ChatMessage] = field(default_factory=list)
    user_message_id: Optional[str] = None
    generation_id: str = ""


class ChatService:
    """Answers questions over a session's indexed documents."""

    def __init__(self, engine: OmniRAGEngine):
        self.engine = engine

    @property
    def settings(self) -> AppSettings:
        return self.engine.settings

    def answer(self, request: ChatRequest) -> ChatMessage:
        session_id = require_session_id(request.session_id)
        from backend.providers.llm import llm_session
        with llm_session(session_id):
            return self._answer(request, session_id)

    def _answer(self, request: ChatRequest, session_id: str) -> ChatMessage:
        question = (request.question or "").strip()
        started = time.perf_counter()

        if not question:
            return _error_message("Please enter a question.")
        if not self.settings.llm.is_configured:
            return _error_message(SERVICE_NOT_CONFIGURED)

        try:
            retrieval = self._retrieve(request, session_id)
        except OmniRAGError as exc:
            logger.warning("Retrieval failed: %s", exc.detail or exc)
            return _error_message(exc.user_message)
        except Exception as exc:
            logger.exception("Unexpected retrieval failure")
            return _error_message("Search over your documents failed. Please try again.")

        try:
            generation_started = time.perf_counter()
            generator = self._generator()
            from backend.rag.retrieval import parse_query
            plan = parse_query(question, request.history)
            result = generator.generate(
                GenerationRequest(
                    question=question, retrieval=retrieval,
                    session_id=session_id, history=request.history,
                    plan=plan, answer_language=plan.answer_language,
                    generation_id=request.generation_id,
                )
            )
            generation_ms = (time.perf_counter() - generation_started) * 1000
        except MissingCredentialError as exc:
            logger.warning("Generation credentials unavailable: %s", exc.detail or exc)
            return _error_message(
                exc.user_message if self.settings.debug_generation else SERVICE_NOT_CONFIGURED
            )
        except ProviderCapabilityError as exc:
            logger.warning("Generation capability unavailable: %s", exc.detail or exc)
            return _error_message(provider_error_message(exc, debug=self.settings.debug_generation))
        except ProviderError as exc:
            logger.warning("Generation failed: %s", exc.detail or exc)
            return _error_message(
                provider_error_message(exc, debug=self.settings.debug_generation),
                retrieval=retrieval,
            )
        except OmniRAGError as exc:
            return _error_message(exc.user_message, retrieval=retrieval)
        except Exception as exc:
            logger.exception("Unexpected generation failure")
            return _error_message("The answer could not be generated. Please try again.", retrieval=retrieval)

        elapsed = time.perf_counter() - started
        message = ChatMessage(
            role=Role.ASSISTANT, content=result.answer,
            citations=result.citations, retrieval=retrieval,
            used_documents=sorted({c.document_id for c in result.citations}),
            debug={
                "model": result.model, "provider": self._last_provider(),
                "images_sent": result.used_images, "contexts": len(retrieval.results),
                "strategy": retrieval.strategy, "reranked": retrieval.reranked,
                "insufficient_evidence": result.insufficient_evidence,
                "elapsed_s": round(elapsed, 2), "timings_ms": retrieval.timings_ms,
                "warnings": result.warnings, "usage": result.usage,
                "provider_attempts": self._last_attempts(),
                "query_scope": retrieval.query_scope,
                "pages_covered": retrieval.unique_pages,
                "total_pages": retrieval.total_pages,
                "candidate_count": retrieval.candidate_count,
                "structured_matches": retrieval.structured_matches,
                "completeness_pass": retrieval.completeness_pass,
                "generation_ms": round(generation_ms, 1),
                "requested_max_output_tokens": (
                    max(self.settings.llm.max_output_tokens, self.settings.llm.exhaustive_max_output_tokens)
                    if retrieval.query_scope != "FOCUSED"
                    else self.settings.llm.max_output_tokens
                ),
                "finish_reason": result.finish_reason,
                "continued": result.continued,
                "returned_chars": len(result.answer),
                "returned_token_estimate": estimate_tokens(result.answer),
                "generation_id": result.generation_id,
                **dict(result.generation_debug or {}),
            },
            reply_to_message_id=request.user_message_id,
        )
        logger.info(
            "Stored assistant message query_scope=%s provider=%s model=%s "
            "finish_reason=%s chat_service_chars=%d stored_token_estimate=%d "
            "continued=%s generation_id=%s message_id=%s",
            retrieval.query_scope, message.debug.get("provider", ""),
            message.debug.get("model", ""), result.finish_reason or "unspecified",
            len(message.content), estimate_tokens(message.content),
            result.continued, result.generation_id, message.message_id,
        )
        return message

    def _retrieve(self, request: ChatRequest, session_id: str):
        from backend.rag.retrieval import Retriever
        retriever = Retriever(
            vector_store=self.engine.vector_store, embeddings=self.engine.embeddings,
            reranker=self._reranker(),
            llm=self.engine.llm if self.settings.retrieval.query_rewrite else None,
            settings=self.settings,
        )
        return retriever.retrieve(
            RetrievalRequest(
                query=request.question, session_id=session_id,
                document_ids=list(request.document_ids) if request.document_ids else None,
                history=request.history,
            )
        )

    def _reranker(self):
        try:
            from backend.providers.rerank import get_reranker
            return get_reranker(self.settings)
        except Exception as exc:
            logger.info("Reranking unavailable: %s", exc)
            return None

    def _generator(self) -> AnswerGenerator:
        llm = self.engine.llm
        if llm is None:
            raise MissingCredentialError("GEMINI_API_KEY, GROQ_API_KEY, or OPENROUTER_API_KEY", "answer generation")
        return AnswerGenerator(llm, file_store=self.engine.file_store, vision=self.engine.vision, settings=self.settings)

    def _last_provider(self) -> str:
        stats = getattr(self.engine.llm, "stats", None)
        if stats is None:
            return getattr(self.engine.llm, "name", "")
        return stats.last_provider or getattr(self.engine.llm, "name", "")

    def _last_attempts(self) -> List[str]:
        stats = getattr(self.engine.llm, "stats", None)
        return list(stats.last_attempts) if stats is not None else []

    def suggested_prompts(self, session_id: str) -> List[str]:
        documents = self.engine.registry.ready_documents(session_id)
        if not documents:
            return []
        prompts: List[str] = ["Summarize these documents."]
        if len(documents) > 1:
            prompts.append(f"Compare {documents[0].filename} with {documents[1].filename}.")
        if any(d.visual_block_count for d in documents):
            prompts.append("Explain the charts and diagrams in these documents.")
        if any(d.table_count for d in documents):
            prompts.append("What do the tables show?")
        if any(d.page_count and d.page_count > 3 for d in documents):
            prompts.append("Explain page 3.")
        if any(d.language in (Language.ARABIC, Language.MIXED) for d in documents):
            prompts.append("لخّص هذه المستندات بالعربية.")
        else:
            prompts.append("Answer in Arabic: what are the key findings?")
        return prompts[:6]


def _error_message(text: str, retrieval=None) -> ChatMessage:
    return ChatMessage(role=Role.ASSISTANT, content=text, error=text, retrieval=retrieval)


# --------------------------------------------------------------------------- #
# Chat history
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RegenerationPlan:
    user_index: int
    prompt: str
    history: List[ChatMessage]
    user_message_id: str


def plan_regeneration(
    messages: Sequence[ChatMessage], message_id: str, *, edited_text: str | None = None
) -> RegenerationPlan:
    target_index = next(
        (index for index, message in enumerate(messages) if message.message_id == message_id), -1,
    )
    if target_index < 0:
        raise ValueError("message no longer exists")
    target = messages[target_index]
    if target.role == Role.USER:
        user_index = target_index
    elif target.role == Role.ASSISTANT:
        user_index = _preceding_user_index(messages, target_index, target.reply_to_message_id)
    else:
        raise ValueError("only user and assistant messages can be regenerated")
    user = messages[user_index]
    prompt = user.content if edited_text is None else edited_text.strip()
    if not prompt:
        raise ValueError("edited prompt cannot be empty")
    return RegenerationPlan(user_index=user_index, prompt=prompt, history=list(messages[:user_index]), user_message_id=user.message_id)


def apply_regeneration(
    messages: Sequence[ChatMessage], plan: RegenerationPlan, answer: ChatMessage,
) -> List[ChatMessage]:
    original = messages[plan.user_index]
    user = original.model_copy(update={"content": plan.prompt})
    linked_answer = answer.model_copy(update={"reply_to_message_id": user.message_id})
    return [*plan.history, user, linked_answer]


def _preceding_user_index(messages: Sequence[ChatMessage], assistant_index: int, reply_to: str | None) -> int:
    if reply_to:
        linked = next(
            (index for index in range(assistant_index - 1, -1, -1)
             if messages[index].message_id == reply_to and messages[index].role == Role.USER), -1,
        )
        if linked >= 0:
            return linked
    for index in range(assistant_index - 1, -1, -1):
        if messages[index].role == Role.USER:
            return index
    raise ValueError("assistant message has no preceding user prompt")


__all__ = [
    "EngineStatus", "OmniRAGEngine", "get_engine", "reset_engine",
    "ReindexReport", "UploadedFile", "IngestionService", "ProgressFn",
    "ChatRequest", "ChatService",
    "RegenerationPlan", "apply_regeneration", "plan_regeneration",
]

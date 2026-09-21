"""Storage layer: binary asset store and document registry.

Streamlit Community Cloud gives you an *ephemeral* filesystem: anything written
is lost on restart and is not shared between replicas. The engine therefore
never assumes a durable path — it talks to a :class:`FileStore` interface and
the default implementation keeps assets in a per-process temporary workspace
with an in-memory index.

Isolation model: every uploaded document, every vector, and every query is
tagged with a ``session_id``. The vector store *refuses* to search without one,
so a bug elsewhere cannot leak one visitor's documents into another's answers.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional

from backend.models import (
    DocumentSummary,
    IngestionStatus,
    SessionInfo,
    SessionIsolationError,
)
from backend.utils import (
    get_logger,
    short_hash,
    stable_id,
)

logger = get_logger(__name__)

SESSION_PREFIX = "s_"


# --------------------------------------------------------------------------- #
# Session helpers
# --------------------------------------------------------------------------- #
def new_session_id() -> str:
    """Create an unguessable session namespace id."""
    return f"{SESSION_PREFIX}{uuid.uuid4().hex}"


def is_valid_session_id(session_id: Optional[str]) -> bool:
    return bool(session_id) and isinstance(session_id, str) and len(session_id) >= 8


def require_session_id(session_id: Optional[str]) -> str:
    """Guard used at every boundary that touches user data."""
    if not is_valid_session_id(session_id):
        raise SessionIsolationError(
            f"Operation attempted without a valid session id (got {session_id!r})"
        )
    return str(session_id)


# --------------------------------------------------------------------------- #
# File store
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StoredAsset:
    asset_id: str
    session_id: str
    media_type: str
    size_bytes: int
    path: Optional[str] = None


class FileStore(ABC):
    """Content-addressed blob store scoped by session."""

    @abstractmethod
    def put(
        self,
        session_id: str,
        data: bytes,
        *,
        media_type: str = "application/octet-stream",
        suffix: str = "",
    ) -> StoredAsset:
        """Store bytes and return a handle. Identical bytes reuse the same id."""

    @abstractmethod
    def get(self, asset_id: str) -> Optional[bytes]:
        """Return the bytes for ``asset_id`` or ``None`` when absent/expired."""

    @abstractmethod
    def delete_session(self, session_id: str) -> int:
        """Remove every asset of a session; returns how many were deleted."""

    def exists(self, asset_id: str) -> bool:
        return self.get(asset_id) is not None


class LocalFileStore(FileStore):
    """Temp-directory backed store, safe for Streamlit's ephemeral disk."""

    def __init__(self, root: Optional[str] = None, *, max_bytes: int = 900 * 1024 * 1024):
        self._root = root or os.path.join(tempfile.gettempdir(), "omnirag_workspace")
        os.makedirs(self._root, exist_ok=True)
        self._lock = threading.RLock()
        self._index: Dict[str, StoredAsset] = {}
        self._memory: Dict[str, bytes] = {}
        self._max_bytes = max_bytes
        self._total_bytes = 0
        logger.info("Local file store initialised at %s", self._root)

    @property
    def root(self) -> str:
        return self._root

    def _session_dir(self, session_id: str) -> str:
        path = os.path.join(self._root, short_hash(session_id, 16))
        os.makedirs(path, exist_ok=True)
        return path

    def put(
        self,
        session_id: str,
        data: bytes,
        *,
        media_type: str = "application/octet-stream",
        suffix: str = "",
    ) -> StoredAsset:
        asset_id = stable_id(session_id, short_hash(data, 32), media_type)
        with self._lock:
            existing = self._index.get(asset_id)
            if existing is not None and existing.path and os.path.exists(existing.path):
                return existing

            if self._total_bytes + len(data) > self._max_bytes:
                self._evict_unsafe(len(data))

            filename = f"{asset_id}{suffix or _suffix_for(media_type)}"
            path = os.path.join(self._session_dir(session_id), filename)
            try:
                with open(path, "wb") as handle:
                    handle.write(data)
            except OSError as exc:
                logger.warning("Falling back to memory for asset %s: %s", asset_id, exc)
                path = None
                self._memory[asset_id] = data

            asset = StoredAsset(
                asset_id=asset_id,
                session_id=session_id,
                media_type=media_type,
                size_bytes=len(data),
                path=path,
            )
            self._index[asset_id] = asset
            self._total_bytes += len(data)
            return asset

    def get(self, asset_id: str) -> Optional[bytes]:
        with self._lock:
            asset = self._index.get(asset_id)
            if asset is None:
                return self._memory.get(asset_id)
            if asset.path and os.path.exists(asset.path):
                try:
                    with open(asset.path, "rb") as handle:
                        return handle.read()
                except OSError as exc:
                    logger.warning("Could not read asset %s: %s", asset_id, exc)
                    return None
            return self._memory.get(asset_id)

    def delete_session(self, session_id: str) -> int:
        with self._lock:
            removed = 0
            for asset_id, asset in list(self._index.items()):
                if asset.session_id != session_id:
                    continue
                self._index.pop(asset_id, None)
                self._memory.pop(asset_id, None)
                self._total_bytes = max(0, self._total_bytes - asset.size_bytes)
                removed += 1
            directory = os.path.join(self._root, short_hash(session_id, 16))
            shutil.rmtree(directory, ignore_errors=True)
            logger.info("Cleared %d assets for session %s", removed, short_hash(session_id, 8))
            return removed

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {"assets": len(self._index), "bytes": self._total_bytes}

    def _evict_unsafe(self, needed: int) -> None:
        freed = 0
        for asset_id, asset in list(self._index.items()):
            if freed >= needed:
                break
            self._index.pop(asset_id, None)
            self._memory.pop(asset_id, None)
            if asset.path and os.path.exists(asset.path):
                try:
                    os.remove(asset.path)
                except OSError:
                    pass
            freed += asset.size_bytes
            self._total_bytes = max(0, self._total_bytes - asset.size_bytes)
        if freed:
            logger.info("Evicted %d bytes from the file store", freed)


class MemoryFileStore(FileStore):
    """Pure in-memory store — used by tests and read-only filesystems."""

    def __init__(self) -> None:
        self._data: Dict[str, bytes] = {}
        self._meta: Dict[str, StoredAsset] = {}
        self._lock = threading.RLock()

    def put(
        self,
        session_id: str,
        data: bytes,
        *,
        media_type: str = "application/octet-stream",
        suffix: str = "",
    ) -> StoredAsset:
        asset_id = stable_id(session_id, short_hash(data, 32), media_type)
        asset = StoredAsset(
            asset_id=asset_id,
            session_id=session_id,
            media_type=media_type,
            size_bytes=len(data),
        )
        with self._lock:
            self._data[asset_id] = data
            self._meta[asset_id] = asset
        return asset

    def get(self, asset_id: str) -> Optional[bytes]:
        with self._lock:
            return self._data.get(asset_id)

    def delete_session(self, session_id: str) -> int:
        with self._lock:
            targets = [a for a, m in self._meta.items() if m.session_id == session_id]
            for asset_id in targets:
                self._data.pop(asset_id, None)
                self._meta.pop(asset_id, None)
            return len(targets)


def _suffix_for(media_type: str) -> str:
    return {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/webp": ".webp",
        "application/pdf": ".pdf",
        "text/plain": ".txt",
    }.get(media_type, ".bin")


_default_store: Optional[FileStore] = None
_default_lock = threading.Lock()


def get_file_store(root: Optional[str] = None) -> FileStore:
    """Process-wide default store (lazily created)."""
    global _default_store
    with _default_lock:
        if _default_store is None:
            try:
                _default_store = LocalFileStore(root)
            except OSError as exc:
                logger.warning("Local file store unavailable (%s); using memory", exc)
                _default_store = MemoryFileStore()
        return _default_store


def set_file_store(store: FileStore) -> None:
    """Override the default store (tests, alternative backends)."""
    global _default_store
    with _default_lock:
        _default_store = store


# --------------------------------------------------------------------------- #
# Document registry
# --------------------------------------------------------------------------- #
class DocumentRegistry:
    """In-process catalogue of the documents belonging to each session."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._sessions: Dict[str, SessionInfo] = {}
        self._documents: Dict[str, Dict[str, DocumentSummary]] = {}
        self._hashes: Dict[str, Dict[str, str]] = {}

    def touch(self, session_id: str) -> SessionInfo:
        session_id = require_session_id(session_id)
        with self._lock:
            info = self._sessions.get(session_id)
            if info is None:
                info = SessionInfo(session_id=session_id)
                self._sessions[session_id] = info
                self._documents[session_id] = {}
                self._hashes[session_id] = {}
                logger.info("New session namespace %s", short_hash(session_id, 8))
            else:
                info.last_active = datetime.now(timezone.utc)
            return info

    def sessions(self) -> List[SessionInfo]:
        with self._lock:
            return list(self._sessions.values())

    def add(self, summary: DocumentSummary) -> DocumentSummary:
        session_id = require_session_id(summary.session_id)
        self.touch(session_id)
        with self._lock:
            self._documents[session_id][summary.document_id] = summary
            if summary.content_hash:
                self._hashes[session_id][summary.content_hash] = summary.document_id
            self._recount(session_id)
            return summary

    def update(self, summary: DocumentSummary) -> DocumentSummary:
        return self.add(summary)

    def get(self, session_id: str, document_id: str) -> Optional[DocumentSummary]:
        session_id = require_session_id(session_id)
        with self._lock:
            return self._documents.get(session_id, {}).get(document_id)

    def list(self, session_id: str) -> List[DocumentSummary]:
        session_id = require_session_id(session_id)
        with self._lock:
            docs = list(self._documents.get(session_id, {}).values())
        return sorted(docs, key=lambda d: d.created_at)

    def ready_documents(self, session_id: str) -> List[DocumentSummary]:
        return [d for d in self.list(session_id) if d.status == IngestionStatus.READY]

    def find_by_hash(self, session_id: str, content_hash: str) -> Optional[DocumentSummary]:
        session_id = require_session_id(session_id)
        with self._lock:
            document_id = self._hashes.get(session_id, {}).get(content_hash)
            if not document_id:
                return None
            return self._documents.get(session_id, {}).get(document_id)

    def remove(self, session_id: str, document_id: str) -> Optional[DocumentSummary]:
        session_id = require_session_id(session_id)
        with self._lock:
            summary = self._documents.get(session_id, {}).pop(document_id, None)
            if summary and summary.content_hash:
                self._hashes.get(session_id, {}).pop(summary.content_hash, None)
            self._recount(session_id)
            return summary

    def clear(self, session_id: str) -> List[str]:
        session_id = require_session_id(session_id)
        with self._lock:
            document_ids = list(self._documents.get(session_id, {}).keys())
            self._documents[session_id] = {}
            self._hashes[session_id] = {}
            self._recount(session_id)
            return document_ids

    def drop_session(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
            self._documents.pop(session_id, None)
            self._hashes.pop(session_id, None)

    def expired_sessions(self, ttl_minutes: int) -> List[str]:
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=max(1, ttl_minutes))
        with self._lock:
            return [
                sid for sid, info in self._sessions.items() if info.last_active < cutoff
            ]

    def _recount(self, session_id: str) -> None:
        info = self._sessions.get(session_id)
        if info is None:
            return
        docs = self._documents.get(session_id, {}).values()
        info.document_count = len(docs)
        info.chunk_count = sum(d.chunk_count for d in docs)
        info.last_active = datetime.now(timezone.utc)


_registry: Optional[DocumentRegistry] = None
_registry_lock = threading.Lock()


def get_registry() -> DocumentRegistry:
    """Process-wide registry singleton."""
    global _registry
    with _registry_lock:
        if _registry is None:
            _registry = DocumentRegistry()
        return _registry


def reset_registry() -> None:
    global _registry
    with _registry_lock:
        _registry = DocumentRegistry()


def document_ids(summaries: Iterable[DocumentSummary]) -> List[str]:
    return [s.document_id for s in summaries]


__all__ = [
    "new_session_id",
    "is_valid_session_id",
    "require_session_id",
    "StoredAsset",
    "FileStore",
    "LocalFileStore",
    "MemoryFileStore",
    "get_file_store",
    "set_file_store",
    "DocumentRegistry",
    "get_registry",
    "reset_registry",
    "document_ids",
]

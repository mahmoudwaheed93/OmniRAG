"""HTTP client for the OmniRAG FastAPI backend.

Used by Streamlit to communicate with FastAPI via HTTP instead of
direct service imports. Supports automatic fallback to direct mode
if the FastAPI server is not reachable.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import httpx

from backend.utils import get_logger

logger = get_logger(__name__)

_DEFAULT_BASE_URL = "http://127.0.0.1:8000"
_TIMEOUT_S = 120.0


def _get_base_url() -> str:
    return os.environ.get("OMNIRAG_API_URL", _DEFAULT_BASE_URL)


def _get_client() -> httpx.Client:
    return httpx.Client(base_url=_get_base_url(), timeout=_TIMEOUT_S)


def _request(
    method: str,
    path: str,
    *,
    json: Any = None,
    files: Any = None,
    data: Any = None,
    params: Any = None,
) -> httpx.Response:
    """Make an HTTP request to the FastAPI backend."""
    with _get_client() as client:
        return client.request(method, path, json=json, files=files, data=data, params=params)


def is_api_available() -> bool:
    """Check if the FastAPI backend is reachable."""
    try:
        resp = _request("GET", "/api/v1/health")
        return resp.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException, Exception):
        return False


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #

def get_health() -> Dict[str, Any]:
    resp = _request("GET", "/api/v1/health")
    resp.raise_for_status()
    return resp.json()


def get_status() -> Dict[str, Any]:
    resp = _request("GET", "/api/v1/status")
    resp.raise_for_status()
    return resp.json()


# --------------------------------------------------------------------------- #
# Chat
# --------------------------------------------------------------------------- #

def create_session() -> str:
    resp = _request("POST", "/api/v1/chat/session")
    resp.raise_for_status()
    return resp.json()["session_id"]


def chat(
    question: str,
    *,
    session_id: str = "",
    document_ids: Optional[List[str]] = None,
    history: Optional[List[Dict[str, str]]] = None,
    answer_language: Optional[str] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"question": question}
    if session_id:
        payload["session_id"] = session_id
    if document_ids:
        payload["document_ids"] = document_ids
    if history:
        payload["history"] = history
    if answer_language:
        payload["answer_language"] = answer_language

    resp = _request("POST", "/api/v1/chat", json=payload)
    resp.raise_for_status()
    return resp.json()


def get_suggestions(session_id: str = "") -> List[str]:
    params: Dict[str, str] = {}
    if session_id:
        params["session_id"] = session_id
    resp = _request("GET", "/api/v1/chat/suggestions", params=params)
    resp.raise_for_status()
    return resp.json()


# --------------------------------------------------------------------------- #
# Documents
# --------------------------------------------------------------------------- #

def list_documents(session_id: str = "") -> Dict[str, Any]:
    params: Dict[str, str] = {}
    if session_id:
        params["session_id"] = session_id
    resp = _request("GET", "/api/v1/documents", params=params)
    resp.raise_for_status()
    return resp.json()


def upload_document(
    filename: str,
    content: bytes,
    *,
    session_id: str = "",
    content_type: str = "application/octet-stream",
) -> Dict[str, Any]:
    params: Dict[str, str] = {}
    if session_id:
        params["session_id"] = session_id

    resp = _request(
        "POST",
        "/api/v1/documents/upload",
        files={"file": (filename, content, content_type)},
        params=params,
    )
    resp.raise_for_status()
    return resp.json()


def delete_document(document_id: str, *, session_id: str = "") -> Dict[str, Any]:
    params: Dict[str, str] = {}
    if session_id:
        params["session_id"] = session_id
    resp = _request("DELETE", f"/api/v1/documents/{document_id}", params=params)
    resp.raise_for_status()
    return resp.json()

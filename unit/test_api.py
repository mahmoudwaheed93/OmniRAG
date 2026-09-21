"""FastAPI API route tests."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.api.app import app


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


# --------------------------------------------------------------------------- #
# Root / Health
# --------------------------------------------------------------------------- #
class TestRoot:
    def test_root_returns_api_info(self, client):
        r = client.get("/")
        assert r.status_code == 200
        body = r.json()
        assert body["name"] == "OmniRAG Medical RAG API"
        assert body["medical_safety"] is True
        assert "disclaimer" in body

    def test_root_includes_request_id(self, client):
        r = client.get("/")
        assert "X-Request-ID" in r.headers
        assert "X-Response-Time" in r.headers


class TestHealth:
    def test_health_returns_200(self, client):
        r = client.get("/api/v1/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["medical_safety"] is True
        assert isinstance(body["llm_configured"], bool)

    def test_health_has_request_id(self, client):
        r = client.get("/api/v1/health")
        assert "X-Request-ID" in r.headers


class TestStatus:
    def test_status_returns_200(self, client):
        r = client.get("/api/v1/status")
        assert r.status_code == 200
        body = r.json()
        assert "ready" in body
        assert "issues" in body
        assert "vector_store" in body


# --------------------------------------------------------------------------- #
# Chat
# --------------------------------------------------------------------------- #
class TestChatSession:
    def test_create_session(self, client):
        r = client.post("/api/v1/chat/session")
        assert r.status_code == 200
        body = r.json()
        assert "session_id" in body
        assert len(body["session_id"]) > 0


class TestChat:
    def test_chat_without_config_returns_error(self, client):
        r = client.post("/api/v1/chat", json={"question": "hello"})
        assert r.status_code == 200
        body = r.json()
        assert body["answer"]
        assert "not configured" in body["answer"].lower() or "not available" in body["answer"].lower()

    def test_chat_empty_question_rejected(self, client):
        r = client.post("/api/v1/chat", json={"question": ""})
        assert r.status_code == 422

    def test_chat_has_disclaimer(self, client):
        r = client.post("/api/v1/chat", json={"question": "test"})
        body = r.json()
        assert "disclaimer" in body
        assert "educational" in body["disclaimer"].lower()


class TestSuggestions:
    def test_suggestions_return_list(self, client):
        r = client.get("/api/v1/chat/suggestions")
        assert r.status_code == 200
        body = r.json()
        assert isinstance(body, list)


# --------------------------------------------------------------------------- #
# Documents
# --------------------------------------------------------------------------- #
class TestDocuments:
    def test_list_documents_empty(self, client):
        r = client.get("/api/v1/documents")
        assert r.status_code == 200
        body = r.json()
        assert body["documents"] == []
        assert body["total"] == 0

    def test_upload_no_file_returns_422(self, client):
        r = client.post("/api/v1/documents/upload")
        assert r.status_code == 422


# --------------------------------------------------------------------------- #
# Error handling
# --------------------------------------------------------------------------- #
class TestErrorHandling:
    def test_404_returns_json(self, client):
        r = client.get("/api/v1/nonexistent")
        assert r.status_code == 404

    def test_method_not_allowed(self, client):
        r = client.put("/api/v1/health")
        assert r.status_code == 405

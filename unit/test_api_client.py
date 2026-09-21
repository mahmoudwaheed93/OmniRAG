"""API client tests."""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from frontend.api_client import (
    is_api_available,
    get_health,
    create_session,
    chat,
    get_suggestions,
    list_documents,
)


class TestAPIAvailability:
    @patch("frontend.api_client._request")
    def test_available_when_health_ok(self, mock_req):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_req.return_value = mock_resp
        assert is_api_available() is True

    @patch("frontend.api_client._request", side_effect=ConnectionError)
    def test_unavailable_on_connection_error(self, mock_req):
        assert is_api_available() is False


class TestUIAPIIntegration:
    def test_use_api_default_false(self):
        from frontend import ui
        ui._API_MODE_CHECKED = False
        ui._API_MODE = False
        import os
        os.environ.pop("OMNIRAG_API_MODE", None)
        assert ui._use_api() is False

    @patch("frontend.api_client._request")
    def test_chat_service_returns_api_adapter_when_mode_on(self, mock_req):
        from frontend import ui
        ui._API_MODE_CHECKED = False
        ui._API_MODE = False
        import os
        os.environ["OMNIRAG_API_MODE"] = "1"

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_req.return_value = mock_resp

        svc = ui.chat_service()
        assert isinstance(svc, ui._APIChatService)

        os.environ.pop("OMNIRAG_API_MODE", None)
        ui._API_MODE_CHECKED = False
        ui._API_MODE = False

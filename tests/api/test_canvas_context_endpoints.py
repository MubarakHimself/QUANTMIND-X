"""
Tests for Canvas Context API Endpoints

Tests model structures and endpoint responses for canvas template loading.
"""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import FastAPI


class TestCanvasContextModels:
    """Test CanvasContext request/response models."""

    def test_canvas_context_request_required_fields(self):
        from src.api.canvas_context_endpoints import CanvasContextRequest

        req = CanvasContextRequest(canvas="risk")
        assert req.canvas == "risk"
        assert req.session_id is None
        assert req.include_memory_identifiers is True

    def test_canvas_context_request_with_session(self):
        from src.api.canvas_context_endpoints import CanvasContextRequest

        req = CanvasContextRequest(
            canvas="live_trading",
            session_id="sess-123",
            include_memory_identifiers=False,
        )
        assert req.canvas == "live_trading"
        assert req.session_id == "sess-123"
        assert req.include_memory_identifiers is False

    def test_canvas_context_response_structure(self):
        from src.api.canvas_context_endpoints import CanvasContextResponse

        resp = CanvasContextResponse(
            canvas="risk",
            template={"key": "value"},
            memory_identifiers=["node-1", "node-2"],
            session_id="sess-abc",
            loaded_at="2026-03-21T10:00:00+00:00",
        )
        assert resp.canvas == "risk"
        assert len(resp.memory_identifiers) == 2
        assert resp.session_id == "sess-abc"

    def test_canvas_context_response_defaults(self):
        from src.api.canvas_context_endpoints import CanvasContextResponse

        resp = CanvasContextResponse(
            canvas="portfolio",
            template={},
            loaded_at="2026-03-21T10:00:00+00:00",
        )
        assert resp.memory_identifiers == []
        assert resp.session_id is None


class TestCanvasContextEndpoints:
    """Test canvas context API endpoints."""

    def setup_method(self):
        from fastapi import FastAPI
        from src.api.canvas_context_endpoints import router

        self.app = FastAPI()
        self.app.include_router(router)
        self.client = TestClient(self.app)

    def test_health_endpoint_returns_healthy(self):
        response = self.client.get("/api/canvas-context/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "supported_canvases" in data

    def test_list_canvases_returns_list(self):
        with patch("src.api.canvas_context_endpoints.get_canvas_list") as mock_list:
            mock_list.return_value = ["risk", "live_trading", "portfolio"]
            response = self.client.get("/api/canvas-context/canvases")

        assert response.status_code == 200
        data = response.json()
        assert "canvases" in data
        assert isinstance(data["canvases"], list)

    def test_list_templates_returns_template_list(self):
        mock_template = MagicMock()
        mock_template.canvas_display_name = "Risk Canvas"
        mock_template.canvas_icon = "shield"
        mock_template.department_head = "risk_head"

        with patch("src.api.canvas_context_endpoints.get_all_templates") as mock_all:
            mock_all.return_value = {"risk": mock_template}
            response = self.client.get("/api/canvas-context/templates")

        assert response.status_code == 200
        data = response.json()
        assert "templates" in data
        assert len(data["templates"]) == 1
        assert data["templates"][0]["canvas"] == "risk"

    def test_get_template_not_found_returns_404(self):
        with patch("src.api.canvas_context_endpoints.load_template") as mock_load:
            mock_load.side_effect = FileNotFoundError("Template not found")
            response = self.client.get("/api/canvas-context/template/nonexistent")

        assert response.status_code == 404

    def test_get_template_invalid_canvas_returns_400(self):
        with patch("src.api.canvas_context_endpoints.load_template") as mock_load:
            mock_load.side_effect = ValueError("Invalid canvas name")
            response = self.client.get("/api/canvas-context/template/invalid$$canvas")

        assert response.status_code == 400

    def test_get_template_accepts_frontend_live_trading_alias(self):
        with patch("src.api.canvas_context_endpoints.load_template") as mock_load:
            mock_template = MagicMock()
            mock_template.model_dump.return_value = {"canvas": "live_trading"}
            mock_load.return_value = mock_template

            response = self.client.get("/api/canvas-context/template/live-trading")

        assert response.status_code == 200
        mock_load.assert_called_once_with("live-trading")

    def test_load_canvas_context_not_found_returns_404(self):
        with patch("src.api.canvas_context_endpoints.load_template") as mock_load:
            mock_load.side_effect = FileNotFoundError("Not found")
            response = self.client.post(
                "/api/canvas-context/load",
                json={"canvas": "nonexistent_canvas"},
            )
        assert response.status_code == 404

    def test_load_canvas_context_accepts_frontend_shared_assets_alias(self):
        with patch("src.api.canvas_context_endpoints.load_template") as mock_load:
            mock_template = MagicMock()
            mock_template.max_identifiers = 50
            mock_template.model_dump.return_value = {"canvas": "shared_assets"}
            mock_load.return_value = mock_template

            response = self.client.post(
                "/api/canvas-context/load",
                json={"canvas": "shared-assets", "include_memory_identifiers": False},
            )

        assert response.status_code == 200
        assert response.json()["canvas"] == "shared-assets"
        mock_load.assert_called_once_with("shared-assets")

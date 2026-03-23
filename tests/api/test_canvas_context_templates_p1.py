"""P1 Tests: CanvasContextTemplate loading per canvas."""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import FastAPI


class TestCanvasTemplatePerCanvas:
    """P1: Test template loading for each canvas type."""

    def setup_method(self):
        from src.api.canvas_context_endpoints import router
        self.app = FastAPI()
        self.app.include_router(router)
        self.client = TestClient(self.app)

    def test_load_live_trading_canvas_context(self):
        """[P1] Loading live_trading canvas should return relevant template."""
        with patch("src.api.canvas_context_endpoints.load_template") as mock_load:
            mock_template = MagicMock()
            mock_template.canvas_display_name = "Live Trading"
            mock_template.canvas_icon = "activity"
            mock_template.department_head = "execution_head"
            mock_load.return_value = mock_template

            response = self.client.post(
                "/api/canvas-context/load",
                json={"canvas": "live_trading", "session_id": "test-session"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["canvas"] == "live_trading"
            assert "template" in data

    def test_load_risk_canvas_context(self):
        """[P1] Loading risk canvas should return relevant template."""
        with patch("src.api.canvas_context_endpoints.load_template") as mock_load:
            mock_template = MagicMock()
            mock_template.canvas_display_name = "Risk Canvas"
            mock_template.canvas_icon = "shield"
            mock_template.department_head = "risk_head"
            mock_load.return_value = mock_template

            response = self.client.post(
                "/api/canvas-context/load",
                json={"canvas": "risk", "session_id": "test-session"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["canvas"] == "risk"

    def test_load_portfolio_canvas_context(self):
        """[P1] Loading portfolio canvas should return relevant template."""
        with patch("src.api.canvas_context_endpoints.load_template") as mock_load:
            mock_template = MagicMock()
            mock_template.canvas_display_name = "Portfolio"
            mock_template.canvas_icon = "briefcase"
            mock_template.department_head = "portfolio_head"
            mock_load.return_value = mock_template

            response = self.client.post(
                "/api/canvas-context/load",
                json={"canvas": "portfolio", "session_id": "test-session"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["canvas"] == "portfolio"

    def test_load_research_canvas_context(self):
        """[P1] Loading research canvas should return relevant template."""
        with patch("src.api.canvas_context_endpoints.load_template") as mock_load:
            mock_template = MagicMock()
            mock_template.canvas_display_name = "Research"
            mock_template.canvas_icon = "search"
            mock_template.department_head = "research_head"
            mock_load.return_value = mock_template

            response = self.client.post(
                "/api/canvas-context/load",
                json={"canvas": "research", "session_id": "test-session"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["canvas"] == "research"

    def test_load_workshop_canvas_context(self):
        """[P1] Loading workshop canvas should return relevant template."""
        with patch("src.api.canvas_context_endpoints.load_template") as mock_load:
            mock_template = MagicMock()
            mock_template.canvas_display_name = "Workshop"
            mock_template.canvas_icon = "code"
            mock_template.department_head = "development_head"
            mock_load.return_value = mock_template

            response = self.client.post(
                "/api/canvas-context/load",
                json={"canvas": "workshop", "session_id": "test-session"},
            )

            assert response.status_code == 200


class TestCanvasContextTokenBudget:
    """P1: Test token budget enforcement in canvas context."""

    def setup_method(self):
        from src.api.canvas_context_endpoints import router
        self.app = FastAPI()
        self.app.include_router(router)
        self.client = TestClient(self.app)

    def test_context_includes_memory_identifiers_by_default(self):
        """[P1] Canvas context should include memory_identifiers by default."""
        with patch("src.api.canvas_context_endpoints.load_template") as mock_load, \
             patch("src.api.canvas_context_endpoints.get_memory_nodes_for_canvas") as mock_mem:

            mock_template = MagicMock()
            mock_load.return_value = mock_template
            mock_mem.return_value = ["node-1", "node-2", "node-3"]

            response = self.client.post(
                "/api/canvas-context/load",
                json={"canvas": "risk"},
            )

            assert response.status_code == 200
            data = response.json()
            assert "memory_identifiers" in data
            assert len(data["memory_identifiers"]) == 3

    def test_context_excludes_memory_identifiers_when_requested(self):
        """[P1] Canvas context should exclude memory_identifiers when include_memory_identifiers=False."""
        with patch("src.api.canvas_context_endpoints.load_template") as mock_load:

            mock_template = MagicMock()
            mock_load.return_value = mock_template

            response = self.client.post(
                "/api/canvas-context/load",
                json={
                    "canvas": "risk",
                    "include_memory_identifiers": False,
                },
            )

            assert response.status_code == 200
            data = response.json()
            # Should return empty memory_identifiers
            assert data.get("memory_identifiers", []) == []

    def test_context_load_includes_session_id(self):
        """[P1] Canvas context should pass session_id for session isolation."""
        with patch("src.api.canvas_context_endpoints.load_template") as mock_load, \
             patch("src.api.canvas_context_endpoints.get_memory_nodes_for_canvas") as mock_mem:

            mock_template = MagicMock()
            mock_load.return_value = mock_template
            mock_mem.return_value = []

            response = self.client.post(
                "/api/canvas-context/load",
                json={
                    "canvas": "live_trading",
                    "session_id": "session-abc-123",
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data.get("session_id") == "session-abc-123"

"""
Tests for Node Update API Endpoints

Tests deployment window logic and node update API structure.
"""
import pytest
from unittest.mock import patch
from datetime import datetime, timezone
from fastapi.testclient import TestClient
from fastapi import FastAPI


class TestDeployWindowLogic:
    """Test deploy window validation logic."""

    def test_weekday_is_invalid(self):
        from src.api.node_update_endpoints import is_valid_deploy_window

        # Monday = 0, invalid
        monday = datetime(2026, 3, 16, 12, 0, tzinfo=timezone.utc)
        with patch("src.api.node_update_endpoints.datetime") as mock_dt:
            mock_dt.now.return_value = monday
            result = is_valid_deploy_window()
        assert result is False

    def test_friday_before_22_is_invalid(self):
        from src.api.node_update_endpoints import is_valid_deploy_window

        friday_21 = datetime(2026, 3, 20, 21, 0, tzinfo=timezone.utc)  # Friday 21:00
        with patch("src.api.node_update_endpoints.datetime") as mock_dt:
            mock_dt.now.return_value = friday_21
            result = is_valid_deploy_window()
        assert result is False

    def test_saturday_is_valid(self):
        from src.api.node_update_endpoints import is_valid_deploy_window

        saturday = datetime(2026, 3, 21, 12, 0, tzinfo=timezone.utc)  # Saturday noon
        with patch("src.api.node_update_endpoints.datetime") as mock_dt:
            mock_dt.now.return_value = saturday
            result = is_valid_deploy_window()
        assert result is True

    def test_sunday_before_22_is_valid(self):
        from src.api.node_update_endpoints import is_valid_deploy_window

        sunday_20 = datetime(2026, 3, 22, 20, 0, tzinfo=timezone.utc)  # Sunday 20:00
        with patch("src.api.node_update_endpoints.datetime") as mock_dt:
            mock_dt.now.return_value = sunday_20
            result = is_valid_deploy_window()
        assert result is True

    def test_sunday_after_22_is_invalid(self):
        from src.api.node_update_endpoints import is_valid_deploy_window

        sunday_23 = datetime(2026, 3, 22, 23, 0, tzinfo=timezone.utc)  # Sunday 23:00
        with patch("src.api.node_update_endpoints.datetime") as mock_dt:
            mock_dt.now.return_value = sunday_23
            result = is_valid_deploy_window()
        assert result is False


class TestNodeUpdateModels:
    """Test node update request/response models."""

    def test_update_request_defaults(self):
        from src.api.node_update_endpoints import UpdateRequest

        req = UpdateRequest()
        assert req.version is None
        assert req.nodes is None

    def test_node_update_response_structure(self):
        from src.api.node_update_endpoints import NodeUpdateResponse, UpdateStatus

        resp = NodeUpdateResponse(
            node="contabo",
            status=UpdateStatus.completed,
            message="Updated to 1.2.0",
            version="1.2.0",
        )
        assert resp.node == "contabo"
        assert resp.status == UpdateStatus.completed

    def test_sequential_update_response_structure(self):
        from src.api.node_update_endpoints import (
            SequentialUpdateResponse,
            UpdateStatus,
            NodeUpdateResponse,
        )
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        resp = SequentialUpdateResponse(
            run_id="node-update-20260321",
            status=UpdateStatus.completed,
            message="Updated 2 nodes",
            nodes=[
                NodeUpdateResponse(node="contabo", status=UpdateStatus.completed, message="OK"),
                NodeUpdateResponse(node="cloudzy", status=UpdateStatus.completed, message="OK"),
            ],
            start_time=now,
        )
        assert len(resp.nodes) == 2
        assert resp.rolled_back_nodes == []

    def test_deploy_window_status_structure(self):
        from src.api.node_update_endpoints import DeployWindowStatus

        status = DeployWindowStatus(
            valid=True,
            current_time="2026-03-21T12:00:00+00:00",
            window_start="Friday 22:00 UTC",
            window_end="Sunday 22:00 UTC",
            message="Deploy window is open.",
        )
        assert status.valid is True
        assert "Friday" in status.window_start

    def test_get_current_version_uses_quantmind_data_dir(self, monkeypatch, tmp_path):
        from src.api.node_update_endpoints import get_current_version

        monkeypatch.setenv("QUANTMIND_DATA_DIR", str(tmp_path))
        (tmp_path / "version.txt").write_text("2.4.1")

        assert get_current_version() == "2.4.1"

    def test_get_node_url_prefers_public_or_internal_api_base(self, monkeypatch):
        from src.api.node_update_endpoints import get_node_url

        monkeypatch.setenv("API_BASE_URL", "https://app.quantmindx.com")
        monkeypatch.setenv("INTERNAL_API_BASE_URL", "https://internal.quantmindx.local")
        monkeypatch.delenv("DESKTOP_URL", raising=False)

        assert get_node_url("desktop") == "https://app.quantmindx.com"


class TestNodeUpdateEndpoints:
    """Test node update REST endpoints."""

    def setup_method(self):
        from fastapi import FastAPI
        from src.api.node_update_endpoints import router

        self.app = FastAPI()
        self.app.include_router(router)
        self.client = TestClient(self.app)

    def test_deploy_window_endpoint_returns_status(self):
        response = self.client.get("/api/node-update/deploy-window")
        assert response.status_code == 200
        data = response.json()
        assert "valid" in data
        assert "current_time" in data
        assert "message" in data

    def test_get_node_statuses(self):
        response = self.client.get("/api/node-update/nodes")
        assert response.status_code == 200
        data = response.json()
        assert "nodes" in data
        assert "timestamp" in data
        assert len(data["nodes"]) == 3  # contabo, cloudzy, desktop

    def test_update_outside_window_returns_403(self):
        # Patch deploy window to be invalid
        with patch("src.api.node_update_endpoints.is_valid_deploy_window") as mock_valid:
            mock_valid.return_value = False
            response = self.client.post(
                "/api/node-update/update",
                json={},
            )
        assert response.status_code == 403

    def test_get_update_status_placeholder(self):
        response = self.client.get("/api/node-update/status/some-run-id")
        assert response.status_code == 200
        data = response.json()
        assert data["run_id"] == "some-run-id"

    def test_update_persists_version_under_quantmind_data_dir(self, monkeypatch, tmp_path):
        monkeypatch.setenv("QUANTMIND_DATA_DIR", str(tmp_path))

        with patch("src.api.node_update_endpoints.is_valid_deploy_window", return_value=True), \
             patch("src.api.node_update_endpoints.check_node_health", return_value={"status": "healthy"}), \
             patch("src.api.node_update_endpoints.get_node_url", return_value="https://node.quantmindx.test"):
            response = self.client.post(
                "/api/node-update/update",
                json={"version": "3.0.0", "nodes": ["contabo"]},
            )

        assert response.status_code == 200
        assert (tmp_path / "versions" / "contabo.txt").read_text() == "3.0.0"

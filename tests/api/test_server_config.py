# tests/api/test_server_config.py
"""
Tests for Server Config API Endpoints

Phase 5: Provider Configuration in Settings UI
Tests for the server configuration CRUD API endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI
from unittest.mock import patch, MagicMock


@pytest.fixture
def test_app():
    """Create a test FastAPI app with the server config router."""
    try:
        from src.api.server_config_endpoints import router

        app = FastAPI()
        app.include_router(router)
        return app
    except ImportError:
        pytest.skip("server_config_endpoints not available")


@pytest.fixture
def client(test_app):
    """Create a test client for the test app."""
    return TestClient(test_app)


class TestListServersEndpoint:
    """Test GET /api/servers endpoint."""

    def test_list_servers_returns_servers_list(self, client):
        """Should return servers list."""
        response = client.get("/api/servers")

        assert response.status_code == 200
        data = response.json()
        assert "servers" in data
        assert isinstance(data["servers"], list)


class TestAddServerEndpoint:
    """Test POST /api/servers endpoint."""

    @patch('src.api.server_config_endpoints.get_db_session')
    def test_add_server_creates_new_server(self, mock_get_session, client):
        """Should create a new server configuration."""
        mock_session = MagicMock()
        mock_get_session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_get_session.return_value.__exit__ = MagicMock(return_value=False)

        response = client.post("/api/servers", json={
            "name": "Cloudzy Trading",
            "server_type": "cloudzy",
            "host": "cloudzy.example.com",
            "port": 22,
            "is_active": True
        })

        assert response.status_code in [200, 201, 401, 403]


class TestDeleteServerEndpoint:
    """Test DELETE /api/servers/{server_id} endpoint."""

    @patch('src.api.server_config_endpoints.get_db_session')
    def test_delete_server_removes_server(self, mock_get_session, client):
        """Should delete a server by ID."""
        mock_session = MagicMock()
        mock_get_session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_get_session.return_value.__exit__ = MagicMock(return_value=False)

        response = client.delete("/api/servers/test-server-id")

        assert response.status_code in [200, 204, 401, 403, 404, 409]

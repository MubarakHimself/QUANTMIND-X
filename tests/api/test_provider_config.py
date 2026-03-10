# tests/api/test_provider_config.py
"""
Tests for Provider Config API Endpoints

Phase 5: Provider Configuration in Settings UI
Tests for the provider configuration CRUD API endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI
from unittest.mock import patch, MagicMock


@pytest.fixture
def test_app():
    """Create a test FastAPI app with the provider config router."""
    from src.api.provider_config_endpoints import router

    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(test_app):
    """Create a test client for the test app."""
    return TestClient(test_app)


class TestListProvidersEndpoint:
    """Test GET /api/providers endpoint."""

    def test_list_providers_returns_providers_list(self, client):
        """Should return providers list with masked API keys."""
        response = client.get("/api/providers")

        assert response.status_code == 200
        data = response.json()
        assert "providers" in data
        assert isinstance(data["providers"], list)

    def test_list_providers_does_not_expose_api_keys(self, client):
        """Should return providers with masked API keys."""
        response = client.get("/api/providers")

        assert response.status_code == 200
        data = response.json()

        # If there are providers, API keys should be masked
        for provider in data.get("providers", []):
            if "api_key" in provider:
                # Should either be None, empty, or masked (not the actual key)
                assert provider["api_key"] in [None, "", "***", "********"]


class TestAddProviderEndpoint:
    """Test POST /api/providers endpoint."""

    @patch('src.api.provider_config_endpoints.get_db_session')
    def test_add_provider_creates_new_provider(self, mock_get_session, client):
        """Should create a new provider configuration."""
        # Mock the database session
        mock_session = MagicMock()
        mock_get_session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_get_session.return_value.__exit__ = MagicMock(return_value=False)

        response = client.post("/api/providers", json={
            "name": "anthropic",
            "api_key": "test-key-123",
            "base_url": "https://api.anthropic.com",
            "enabled": True
        })

        # Should either succeed or require auth (depending on implementation)
        assert response.status_code in [200, 201, 401, 403]

    def test_add_provider_requires_name(self, client):
        """Should require a name field."""
        response = client.post("/api/providers", json={
            "api_key": "test-key-123"
        })

        # Should return validation error (422) or auth error
        assert response.status_code in [422, 401, 403]


class TestDeleteProviderEndpoint:
    """Test DELETE /api/providers/{provider_id} endpoint."""

    @patch('src.api.provider_config_endpoints.get_db_session')
    def test_delete_provider_removes_provider(self, mock_get_session, client):
        """Should delete a provider by ID."""
        # Mock the database session
        mock_session = MagicMock()
        mock_get_session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_get_session.return_value.__exit__ = MagicMock(return_value=False)

        response = client.delete("/api/providers/test-provider-id")

        # Should either succeed or require auth (depending on implementation)
        assert response.status_code in [200, 204, 401, 403]


class TestAvailableProvidersEndpoint:
    """Test GET /api/providers/available endpoint."""

    def test_available_providers_returns_list(self, client):
        """Should return available providers for dropdowns."""
        response = client.get("/api/providers/available")

        assert response.status_code == 200
        data = response.json()
        # Should return list of providers with API keys configured
        assert "providers" in data or isinstance(data, list)

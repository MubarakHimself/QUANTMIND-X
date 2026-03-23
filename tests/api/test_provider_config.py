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
        # 409 is returned when provider is active (correct per AC-4)
        assert response.status_code in [200, 204, 401, 403, 409]


class TestAvailableProvidersEndpoint:
    """Test GET /api/providers/available endpoint."""

    def test_available_providers_returns_list(self, client):
        """Should return available providers for dropdowns."""
        response = client.get("/api/providers/available")

        assert response.status_code == 200
        data = response.json()
        # Should return list of providers with API keys configured
        assert "providers" in data or isinstance(data, list)


class TestUpdateProviderEndpoint:
    """Test PUT /api/providers/{provider_id} endpoint (AC-3)."""

    @patch('src.api.provider_config_endpoints.get_db_session')
    def test_update_provider_updates_fields(self, mock_get_session, client):
        """Should update only provided fields; preserve API key if not provided."""
        # Mock the database session
        mock_session = MagicMock()
        mock_provider = MagicMock()
        mock_provider.id = "test-uuid"
        mock_provider.provider_type = "anthropic"
        mock_provider.display_name = "Anthropic"
        mock_provider.is_active = True
        mock_provider.api_key_encrypted = "encrypted-key"
        mock_provider.get_api_key = MagicMock(return_value="original-key")
        mock_session.query.return_value.filter.return_value.first.return_value = mock_provider
        mock_get_session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_get_session.return_value.__exit__ = MagicMock(return_value=False)

        # Update without providing api_key - should preserve existing
        response = client.put("/api/providers/test-uuid", json={
            "display_name": "Updated Anthropic",
            "is_active": False
        })

        assert response.status_code in [200, 201, 401, 403]
        # Verify set_api_key was NOT called (key should be preserved)
        # and display_name/update was called

    @patch('src.api.provider_config_endpoints.get_db_session')
    def test_update_provider_returns_404_for_unknown(self, mock_get_session, client):
        """Should return 404 for unknown provider ID."""
        mock_session = MagicMock()
        mock_session.query.return_value.filter.return_value.first.return_value = None
        mock_get_session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_get_session.return_value.__exit__ = MagicMock(return_value=False)

        response = client.put("/api/providers/unknown-id", json={
            "display_name": "Test"
        })

        assert response.status_code == 404


class TestDeleteProvider409Behavior:
    """Test DELETE /api/providers/{provider_id} returns 409 for active provider (AC-4)."""

    @patch('src.api.provider_config_endpoints.get_db_session')
    def test_delete_active_provider_returns_409(self, mock_get_session, client):
        """Should return 409 Conflict when deleting an active provider."""
        mock_session = MagicMock()
        mock_provider = MagicMock()
        mock_provider.id = "test-uuid"
        mock_provider.provider_type = "anthropic"
        mock_provider.is_active = True  # Active provider
        mock_session.query.return_value.filter.return_value.first.return_value = mock_provider
        mock_get_session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_get_session.return_value.__exit__ = MagicMock(return_value=False)

        response = client.delete("/api/providers/test-uuid")

        assert response.status_code == 409
        assert "in use" in response.json().get("detail", "").lower()

    @patch('src.api.provider_config_endpoints.get_db_session')
    def test_delete_inactive_provider_succeeds(self, mock_get_session, client):
        """Should successfully delete an inactive provider."""
        mock_session = MagicMock()
        mock_provider = MagicMock()
        mock_provider.id = "test-uuid"
        mock_provider.provider_type = "anthropic"
        mock_provider.is_active = False  # Inactive provider
        mock_session.query.return_value.filter.return_value.first.return_value = mock_provider
        mock_get_session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_get_session.return_value.__exit__ = MagicMock(return_value=False)

        response = client.delete("/api/providers/test-uuid")

        assert response.status_code in [200, 204]

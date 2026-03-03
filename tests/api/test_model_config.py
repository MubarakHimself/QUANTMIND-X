# tests/api/test_model_config.py
"""
Tests for Model Config API Endpoints

Phase 5: Model Provider Hierarchy in UI
Tests for the hierarchical model list API that provides provider groups with their models.
"""
import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI
from unittest.mock import patch


@pytest.fixture
def test_app():
    """Create a test FastAPI app with the model config router."""
    from src.api.model_config_endpoints import router

    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(test_app):
    """Create a test client for the test app."""
    return TestClient(test_app)


class TestAvailableModelsEndpoint:
    """Test GET /api/agent-config/available-models endpoint."""

    @patch('src.api.model_config_endpoints.has_api_key')
    def test_returns_hierarchical_structure_with_providers_list(self, mock_has_api_key, client):
        """Should return a hierarchical structure with providers array containing id, name, models."""
        # Mock all providers as available
        mock_has_api_key.return_value = True

        response = client.get("/api/agent-config/available-models")

        assert response.status_code == 200
        data = response.json()

        # Should have 'providers' key at root level
        assert "providers" in data
        assert isinstance(data["providers"], list)

        # Each provider should have id, name, and models
        for provider in data["providers"]:
            assert "id" in provider
            assert "name" in provider
            assert "models" in provider
            assert isinstance(provider["models"], list)

    @patch('src.api.model_config_endpoints.has_api_key')
    def test_includes_openai_provider(self, mock_has_api_key, client):
        """Should include OpenAI provider with GPT models."""
        mock_has_api_key.return_value = True

        response = client.get("/api/agent-config/available-models")
        data = response.json()

        provider_ids = [p["id"] for p in data["providers"]]
        assert "openai" in provider_ids

        openai_provider = next(p for p in data["providers"] if p["id"] == "openai")
        assert "gpt-4o" in openai_provider["models"]
        assert "gpt-4o-mini" in openai_provider["models"]

    @patch('src.api.model_config_endpoints.has_api_key')
    def test_includes_anthropic_provider(self, mock_has_api_key, client):
        """Should include Anthropic provider with Claude models."""
        mock_has_api_key.return_value = True

        response = client.get("/api/agent-config/available-models")
        data = response.json()

        provider_ids = [p["id"] for p in data["providers"]]
        assert "anthropic" in provider_ids

        anthropic_provider = next(p for p in data["providers"] if p["id"] == "anthropic")
        assert "claude-opus-4-20250514" in anthropic_provider["models"]
        assert "claude-sonnet-4-20250514" in anthropic_provider["models"]
        assert "claude-haiku-3-20240307" in anthropic_provider["models"]

    @patch('src.api.model_config_endpoints.has_api_key')
    def test_includes_google_provider(self, mock_has_api_key, client):
        """Should include Google provider with Gemini models."""
        mock_has_api_key.return_value = True

        response = client.get("/api/agent-config/available-models")
        data = response.json()

        provider_ids = [p["id"] for p in data["providers"]]
        assert "google" in provider_ids

        google_provider = next(p for p in data["providers"] if p["id"] == "google")
        assert "gemini-2.0-flash" in google_provider["models"]
        assert "gemini-1.5-pro" in google_provider["models"]

    @patch('src.api.model_config_endpoints.has_api_key')
    def test_includes_deepseek_provider(self, mock_has_api_key, client):
        """Should include DeepSeek provider with DeepSeek models."""
        mock_has_api_key.return_value = True

        response = client.get("/api/agent-config/available-models")
        data = response.json()

        provider_ids = [p["id"] for p in data["providers"]]
        assert "deepseek" in provider_ids

        deepseek_provider = next(p for p in data["providers"] if p["id"] == "deepseek")
        assert "deepseek-chat" in deepseek_provider["models"]
        assert "deepseek-coder" in deepseek_provider["models"]

    @patch('src.api.model_config_endpoints.has_api_key')
    def test_provider_display_names_are_human_readable(self, mock_has_api_key, client):
        """Should have human-readable display names for providers."""
        mock_has_api_key.return_value = True

        response = client.get("/api/agent-config/available-models")
        data = response.json()

        for provider in data["providers"]:
            # Display names should not be empty and should be meaningful
            assert provider["name"]
            assert len(provider["name"]) > 0


class TestListAgentModelsEndpoint:
    """Test GET /api/agent-config/models endpoint."""

    def test_returns_agent_model_config(self, client):
        """Should return model configuration for all agents."""
        response = client.get("/api/agent-config/models")

        assert response.status_code == 200
        data = response.json()

        # Should contain default agent configs
        assert isinstance(data, dict)
        assert "copilot" in data
        assert "floor_manager" in data

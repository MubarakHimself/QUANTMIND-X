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
from src.agents.providers.router import RuntimeLLMConfig


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
    def test_returns_provider_mapping_with_models(self, mock_has_api_key, client):
        """Should return a providers mapping keyed by provider id."""
        # Mock all providers as available
        mock_has_api_key.return_value = True

        response = client.get("/api/agent-config/available-models")

        assert response.status_code == 200
        data = response.json()

        # Should have 'providers' key at root level
        assert "providers" in data
        assert isinstance(data["providers"], dict)

        # Each provider should have availability + models
        for provider_info in data["providers"].values():
            assert "available" in provider_info
            assert "models" in provider_info
            assert isinstance(provider_info["models"], list)

    @patch('src.api.model_config_endpoints.has_api_key')
    def test_includes_openai_provider(self, mock_has_api_key, client):
        """Should include OpenAI provider with GPT models."""
        mock_has_api_key.return_value = True

        response = client.get("/api/agent-config/available-models")
        data = response.json()

        assert "openai" in data["providers"]

        openai_provider = data["providers"]["openai"]
        model_ids = [m["id"] for m in openai_provider["models"]]
        assert "gpt-4o" in model_ids
        assert "gpt-4o-mini" in model_ids

    @patch('src.api.model_config_endpoints.has_api_key')
    def test_includes_anthropic_provider(self, mock_has_api_key, client):
        """Should include Anthropic provider with Claude models."""
        mock_has_api_key.return_value = True

        response = client.get("/api/agent-config/available-models")
        data = response.json()

        assert "anthropic" in data["providers"]

        anthropic_provider = data["providers"]["anthropic"]
        model_ids = [m["id"] for m in anthropic_provider["models"]]
        assert "claude-opus-4-20250514" in model_ids
        assert "claude-sonnet-4-20250514" in model_ids
        assert "claude-haiku-3-20240307" in model_ids

    @patch('src.api.model_config_endpoints.has_api_key')
    def test_includes_minimax_provider(self, mock_has_api_key, client):
        """Should include MiniMax provider with configured models."""
        mock_has_api_key.return_value = True

        response = client.get("/api/agent-config/available-models")
        data = response.json()

        assert "minimax" in data["providers"]

        minimax_provider = data["providers"]["minimax"]
        model_ids = [m["id"] for m in minimax_provider["models"]]
        assert "MiniMax-M2.7" in model_ids
        assert "MiniMax-M2.5" in model_ids

    @patch('src.api.model_config_endpoints.has_api_key')
    def test_includes_zhipu_provider(self, mock_has_api_key, client):
        """Should include Zhipu provider with GLM models."""
        mock_has_api_key.return_value = True

        response = client.get("/api/agent-config/available-models")
        data = response.json()

        assert "zhipu" in data["providers"]

        zhipu_provider = data["providers"]["zhipu"]
        model_ids = [m["id"] for m in zhipu_provider["models"]]
        assert "glm-4-plus" in model_ids
        assert "glm-4" in model_ids

    @patch('src.api.model_config_endpoints.has_api_key')
    def test_provider_entries_include_availability_flags(self, mock_has_api_key, client):
        """Each provider entry should expose availability metadata."""
        mock_has_api_key.return_value = True

        response = client.get("/api/agent-config/available-models")
        data = response.json()

        for provider in data["providers"].values():
            assert isinstance(provider["available"], bool)


class TestListAgentModelsEndpoint:
    """Test GET /api/agent-config/models endpoint."""

    def test_returns_agent_model_config(self, client):
        """Should return model configuration for all agents."""
        response = client.get("/api/agent-config/models")

        assert response.status_code == 200
        data = response.json()

        # Should contain default agent configs
        assert isinstance(data, dict)
        assert "floor_manager" in data
        assert "research" in data


class TestRuntimeResolvedDefaults:
    """Provider-neutral defaults should come from runtime config, not hardcoded vendors."""

    @patch("src.api.model_config_endpoints._save_models")
    @patch("src.api.model_config_endpoints.get_router")
    def test_patch_uses_runtime_resolved_defaults_when_provider_missing(self, mock_get_router, _mock_save_models, client):
        from src.api import model_config_endpoints as module

        router = mock_get_router.return_value
        router.resolve_runtime_config.return_value = RuntimeLLMConfig(
            provider_type="openai",
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            model="gpt-4o",
            source="provider_config",
            display_name="OpenAI",
        )

        original_agent_models = dict(module._agent_models)
        module._agent_models = {
            "floor_manager": {"provider": "", "model": ""},
            "research": {"provider": "", "model": ""},
            "development": {"provider": "", "model": ""},
            "trading": {"provider": "", "model": ""},
            "risk": {"provider": "", "model": ""},
            "portfolio": {"provider": "", "model": ""},
        }
        try:
            response = client.patch("/api/agent-config/floor_manager/model", json={"model": "", "provider": ""})
            assert response.status_code == 200
            payload = response.json()
            assert payload["provider"] == "openai"
            assert payload["model"] == "gpt-4o"
        finally:
            module._agent_models = original_agent_models

    @patch("src.api.model_config_endpoints.get_router")
    def test_models_endpoint_uses_runtime_resolved_defaults_when_unconfigured(self, mock_get_router, client):
        from src.api import model_config_endpoints as module

        router = mock_get_router.return_value
        router.resolve_runtime_config.return_value = RuntimeLLMConfig(
            provider_type="anthropic",
            api_key="test-key",
            base_url="https://api.anthropic.com/v1",
            model="claude-sonnet-4-20250514",
            source="provider_config",
            display_name="Anthropic",
        )

        original_agent_models = dict(module._agent_models)
        module._agent_models = {
            "floor_manager": {"provider": "", "model": ""},
            "research": {"provider": "", "model": ""},
            "development": {"provider": "", "model": ""},
            "trading": {"provider": "", "model": ""},
            "risk": {"provider": "", "model": ""},
            "portfolio": {"provider": "", "model": ""},
        }
        try:
            response = client.get("/api/agent-config/models")
            assert response.status_code == 200
            payload = response.json()
            assert payload["floor_manager"]["provider"] == "anthropic"
            assert payload["floor_manager"]["model"] == "claude-sonnet-4-20250514"
        finally:
            module._agent_models = original_agent_models

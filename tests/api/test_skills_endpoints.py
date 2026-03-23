"""
Tests for Skills API Endpoints (Story 7.4 - Skill Catalogue)

Tests skill registry request/response models and endpoint behavior.
"""
import pytest
import tempfile
import os
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import FastAPI


def _make_client():
    from src.api.skills_endpoints import router

    app = FastAPI()
    app.include_router(router, prefix="/api")
    return TestClient(app)


class TestSkillModels:
    """Test skill request/response model structures."""

    def test_skill_create_request_required_fields(self):
        from src.api.skills_endpoints import SkillCreateRequest

        req = SkillCreateRequest(
            name="run_backtest",
            description="Run a backtest for a strategy",
        )
        assert req.name == "run_backtest"
        assert req.category == "general"
        assert req.departments == []
        assert req.version == "1.0.0"

    def test_skill_authoring_request_structure(self):
        from src.api.skills_endpoints import SkillAuthoringRequest

        req = SkillAuthoringRequest(
            name="generate_trd",
            description="Generate a Trading Requirements Document",
            inputs={"hypothesis": "string"},
            outputs={"trd_id": "string"},
            sop_steps=["Receive hypothesis", "Validate inputs", "Generate TRD"],
            departments=["research"],
        )
        assert req.name == "generate_trd"
        assert len(req.sop_steps) == 3
        assert "research" in req.departments

    def test_skill_execute_request_defaults(self):
        from src.api.skills_endpoints import SkillExecuteRequest

        req = SkillExecuteRequest()
        assert req.parameters == {}

    def test_skill_response_required_fields(self):
        from src.api.skills_endpoints import SkillResponse

        resp = SkillResponse(
            name="run_backtest",
            description="Run a backtest",
            slash_command="/run-backtest",
            version="1.0.0",
            usage_count=5,
        )
        assert resp.slash_command == "/run-backtest"
        assert resp.usage_count == 5


class TestFormatDictAsYaml:
    """Test the YAML formatting helper."""

    def test_simple_dict(self):
        from src.api.skills_endpoints import _format_dict_as_yaml

        result = _format_dict_as_yaml({"key": "value"})
        assert "key: value" in result

    def test_nested_dict(self):
        from src.api.skills_endpoints import _format_dict_as_yaml

        result = _format_dict_as_yaml({"section": {"subkey": "subval"}})
        assert "section:" in result
        assert "subkey: subval" in result

    def test_list_values(self):
        from src.api.skills_endpoints import _format_dict_as_yaml

        result = _format_dict_as_yaml({"items": ["a", "b", "c"]})
        assert "items:" in result
        assert "- a" in result


class TestSkillsEndpoints:
    """Test skills API endpoints."""

    def test_list_skills_endpoint(self):
        client = _make_client()
        mock_manager = MagicMock()
        mock_manager.list_skills.return_value = ["run_backtest", "generate_hypothesis"]
        mock_manager.get_skill_info.side_effect = lambda name: {
            "name": name,
            "description": f"{name} description",
            "slash_command": f"/{name.replace('_', '-')}",
            "version": "1.0.0",
            "usage_count": 0,
        }

        with patch("src.api.skills_endpoints.get_skill_manager", return_value=mock_manager):
            response = client.get("/api/skills")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2

    def test_list_skills_by_department(self):
        client = _make_client()
        mock_manager = MagicMock()
        mock_manager.get_skills_by_department.return_value = ["research_skill"]
        mock_manager.get_skill_info.return_value = {
            "name": "research_skill",
            "description": "A research skill",
            "slash_command": "/research-skill",
            "version": "1.0.0",
            "usage_count": 0,
        }

        with patch("src.api.skills_endpoints.get_skill_manager", return_value=mock_manager):
            response = client.get("/api/skills", params={"department": "research"})

        assert response.status_code == 200
        mock_manager.get_skills_by_department.assert_called_once_with("research")

    def test_get_skill_not_found_returns_404(self):
        client = _make_client()
        mock_manager = MagicMock()
        mock_manager.get_skill_info.side_effect = Exception("Not found")

        with patch("src.api.skills_endpoints.get_skill_manager", return_value=mock_manager):
            response = client.get("/api/skills/nonexistent_skill")

        assert response.status_code == 404

    def test_create_skill_returns_metadata(self):
        client = _make_client()
        mock_manager = MagicMock()
        # Simulate skill not existing (get_skill_info raises on first call)
        mock_manager.get_skill_info.side_effect = Exception("Not found")

        with patch("src.api.skills_endpoints.get_skill_manager", return_value=mock_manager):
            response = client.post(
                "/api/skills",
                json={
                    "name": "new_skill",
                    "description": "A brand new skill",
                },
            )

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "new_skill"
        assert data["slash_command"] == "/new-skill"

    def test_skill_forge_authoring_creates_file(self):
        client = _make_client()

        with tempfile.TemporaryDirectory() as tmpdir:
            skills_dir = os.path.join(tmpdir, "shared_assets", "skills")

            with patch("src.api.skills_endpoints.SKILLS_DIR", skills_dir):
                response = client.post(
                    "/api/skills/authoring",
                    json={
                        "name": "detect_regime",
                        "description": "Detect market regime using HMM",
                        "sop_steps": ["Load data", "Run HMM", "Return regime label"],
                        "departments": ["research", "risk"],
                    },
                )

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "detect_regime"
        assert data["status"] == "created"
        assert data["slash_command"] == "/detect-regime"

    def test_execute_skill_success(self):
        client = _make_client()
        mock_manager = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.data = {"output": "ok"}
        mock_result.error = None
        mock_result.execution_time_ms = 15
        mock_manager.execute.return_value = mock_result

        with patch("src.api.skills_endpoints.get_skill_manager", return_value=mock_manager):
            response = client.post(
                "/api/skills/run_backtest/execute",
                json={"parameters": {"strategy_id": "strat-001"}},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_execute_skill_failure_returns_500(self):
        client = _make_client()
        mock_manager = MagicMock()
        mock_manager.execute.side_effect = Exception("Skill execution error")

        with patch("src.api.skills_endpoints.get_skill_manager", return_value=mock_manager):
            response = client.post(
                "/api/skills/broken_skill/execute",
                json={"parameters": {}},
            )

        assert response.status_code == 500

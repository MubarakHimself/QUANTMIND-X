"""
Tests for Alpha Forge Template API Endpoints

Tests template library management for strategy fast-track deployment.
"""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import FastAPI


def _make_client():
    from src.api.alpha_forge_templates import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


class TestTemplateModels:
    """Test template request/response model structures."""

    def test_template_create_request_defaults(self):
        from src.api.alpha_forge_templates import TemplateCreateRequest

        req = TemplateCreateRequest(name="High Impact News Template")
        assert req.name == "High Impact News Template"
        assert req.is_islamic_compliant is True
        assert req.avg_deployment_time == 11
        assert req.lot_sizing_multiplier == 0.5

    def test_template_response_structure(self):
        from src.api.alpha_forge_templates import TemplateResponse

        resp = TemplateResponse(
            id="tmpl-001",
            name="News Breakout Template",
            strategy_type="news_event_breakout",
            applicable_events=["HIGH_IMPACT_NEWS"],
            applicable_symbols=["EURUSD", "GBPUSD"],
            risk_profile="conservative",
            avg_deployment_time=11,
            parameters={"stop_loss_pips": 20},
            lot_sizing_multiplier=0.5,
            auto_expiry_hours=24,
            is_islamic_compliant=True,
            created_at="2026-03-21T10:00:00",
            updated_at="2026-03-21T10:00:00",
            author="system",
            is_active=True,
        )
        assert resp.id == "tmpl-001"
        assert "HIGH_IMPACT_NEWS" in resp.applicable_events

    def test_template_match_request_defaults(self):
        from src.api.alpha_forge_templates import TemplateMatchRequest

        req = TemplateMatchRequest(event_type="HIGH_IMPACT_NEWS")
        assert req.event_type == "HIGH_IMPACT_NEWS"
        assert req.affected_symbols == []
        assert req.impact_tier == "HIGH"

    def test_template_match_response_structure(self):
        from src.api.alpha_forge_templates import TemplateMatchResponse

        resp = TemplateMatchResponse(
            matches=[{"template_id": "tmpl-001", "score": 0.95}],
            total=1,
            event_type="HIGH_IMPACT_NEWS",
            impact_tier="HIGH",
        )
        assert resp.total == 1
        assert resp.event_type == "HIGH_IMPACT_NEWS"

    def test_fast_track_deploy_request(self):
        from src.api.alpha_forge_templates import FastTrackDeployRequest

        req = FastTrackDeployRequest(template_id="tmpl-001", symbol="EURUSD")
        assert req.template_id == "tmpl-001"
        assert req.symbol == "EURUSD"
        assert req.strategy_name is None


class TestAlphaForgeTemplateEndpoints:
    """Test alpha forge template REST endpoints."""

    def _mock_template(self, template_id="tmpl-001"):
        mock_t = MagicMock()
        mock_t.id = template_id
        mock_t.name = "Test Template"
        mock_t.is_active = True
        mock_t.avg_deployment_time = 11
        mock_t.to_dict.return_value = {
            "id": template_id,
            "name": "Test Template",
            "strategy_type": "news_event_breakout",
            "applicable_events": ["HIGH_IMPACT_NEWS"],
            "applicable_symbols": ["EURUSD"],
            "risk_profile": "conservative",
            "avg_deployment_time": 11,
            "parameters": {},
            "lot_sizing_multiplier": 0.5,
            "auto_expiry_hours": 24,
            "is_islamic_compliant": True,
            "created_at": "2026-03-21T10:00:00",
            "updated_at": "2026-03-21T10:00:00",
            "author": "system",
            "is_active": True,
        }
        return mock_t

    def test_get_templates_returns_list(self):
        client = _make_client()
        mock_t = self._mock_template()

        with patch("src.api.alpha_forge_templates.get_template_storage") as mock_storage_fn:
            mock_storage = mock_storage_fn.return_value
            mock_storage.seed_default_templates.return_value = [mock_t]

            response = client.get("/api/alpha-forge/templates")

        assert response.status_code == 200
        data = response.json()
        assert "templates" in data
        assert data["total"] == 1

    def test_get_templates_active_filter(self):
        client = _make_client()
        active_t = self._mock_template("active-tmpl")
        inactive_t = self._mock_template("inactive-tmpl")
        inactive_t.is_active = False

        with patch("src.api.alpha_forge_templates.get_template_storage") as mock_storage_fn:
            mock_storage = mock_storage_fn.return_value
            mock_storage.seed_default_templates.return_value = [active_t, inactive_t]

            response = client.get("/api/alpha-forge/templates", params={"active_only": True})

        data = response.json()
        assert data["total"] == 1

    def test_get_template_not_found_returns_404(self):
        client = _make_client()

        with patch("src.api.alpha_forge_templates.get_template_storage") as mock_storage_fn:
            mock_storage = mock_storage_fn.return_value
            mock_storage.get.return_value = None

            response = client.get("/api/alpha-forge/templates/nonexistent-id")

        assert response.status_code == 404

    def test_create_template_returns_201(self):
        client = _make_client()
        mock_t = self._mock_template("new-tmpl")

        with patch("src.api.alpha_forge_templates.get_template_storage") as mock_storage_fn:
            mock_storage = mock_storage_fn.return_value
            mock_storage.save.return_value = mock_t

            response = client.post(
                "/api/alpha-forge/templates",
                json={"name": "New Test Template"},
            )

        assert response.status_code == 201

    def test_delete_template_not_found_returns_404(self):
        client = _make_client()

        with patch("src.api.alpha_forge_templates.get_template_storage") as mock_storage_fn:
            mock_storage = mock_storage_fn.return_value
            mock_storage.delete.return_value = False

            response = client.delete("/api/alpha-forge/templates/nonexistent-id")

        assert response.status_code == 404

    def test_delete_template_success(self):
        client = _make_client()

        with patch("src.api.alpha_forge_templates.get_template_storage") as mock_storage_fn:
            mock_storage = mock_storage_fn.return_value
            mock_storage.delete.return_value = True

            response = client.delete("/api/alpha-forge/templates/tmpl-001")

        assert response.status_code == 200
        assert response.json()["status"] == "deleted"

    def test_match_templates_endpoint(self):
        client = _make_client()
        mock_match = MagicMock()
        mock_match.to_dict.return_value = {"template_id": "tmpl-001", "score": 0.95}

        with patch("src.api.alpha_forge_templates.get_template_matcher") as mock_matcher_fn:
            mock_matcher = mock_matcher_fn.return_value
            mock_matcher.get_top_matches.return_value = [mock_match]

            response = client.post(
                "/api/alpha-forge/templates/match",
                json={"event_type": "HIGH_IMPACT_NEWS", "affected_symbols": ["EURUSD"]},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["event_type"] == "HIGH_IMPACT_NEWS"
        assert data["total"] == 1

    def test_simulate_template_match(self):
        client = _make_client()

        with patch("src.api.alpha_forge_templates.get_template_matcher") as mock_matcher_fn:
            mock_matcher = mock_matcher_fn.return_value
            mock_matcher.get_top_matches.return_value = []

            response = client.get(
                "/api/alpha-forge/templates/match/simulate",
                params={"event_type": "CENTRAL_BANK", "symbols": "EURUSD,GBPUSD"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["event_type"] == "CENTRAL_BANK"

    def test_fast_track_deploy_template_not_found(self):
        client = _make_client()

        with patch("src.api.alpha_forge_templates.get_template_storage") as mock_storage_fn:
            mock_storage = mock_storage_fn.return_value
            mock_storage.get.return_value = None

            response = client.post(
                "/api/alpha-forge/fast-track/deploy",
                json={"template_id": "nonexistent", "symbol": "EURUSD"},
            )

        assert response.status_code == 404

    def test_fast_track_deploy_inactive_template_400(self):
        client = _make_client()
        mock_t = self._mock_template("tmpl-inactive")
        mock_t.is_active = False

        with patch("src.api.alpha_forge_templates.get_template_storage") as mock_storage_fn:
            mock_storage = mock_storage_fn.return_value
            mock_storage.get.return_value = mock_t

            response = client.post(
                "/api/alpha-forge/fast-track/deploy",
                json={"template_id": "tmpl-inactive", "symbol": "EURUSD"},
            )

        assert response.status_code == 400

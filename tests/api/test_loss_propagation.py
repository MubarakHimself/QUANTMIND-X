"""
Tests for Cross-Strategy Loss Propagation API

Tests model structures, service logic, and REST endpoints for FR76.
"""
import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import FastAPI


class TestLossPropagationModels:
    """Test loss propagation model structures."""

    def test_loss_cap_breach_event_defaults(self):
        from src.api.loss_propagation import LossCapBreachEvent

        event = LossCapBreachEvent(
            strategy_id="STRAT_A",
            breach_time="2026-03-21T10:00:00Z",
            loss_amount=500.0,
            daily_loss_cap=500.0,
        )
        assert event.strategy_id == "STRAT_A"
        assert event.correlation_threshold == 0.5

    def test_affected_strategy_structure(self):
        from src.api.loss_propagation import AffectedStrategy

        affected = AffectedStrategy(
            strategy_id="STRAT_B",
            original_kelly=0.10,
            adjusted_kelly=0.075,
            correlation=0.7,
            reason="Correlation 0.70 >= 0.5",
        )
        assert affected.adjusted_kelly == 0.075
        assert affected.correlation == 0.7

    def test_loss_propagation_response_structure(self):
        from src.api.loss_propagation import LossPropagationResponse, AffectedStrategy

        affected = AffectedStrategy(
            strategy_id="STRAT_B",
            original_kelly=0.10,
            adjusted_kelly=0.075,
            correlation=0.6,
            reason="Test",
        )
        resp = LossPropagationResponse(
            success=True,
            source_strategy="STRAT_A",
            breach_time="2026-03-21T10:00:00Z",
            affected_strategies=[affected],
            event_id="FR76-20260321-100000",
            message="1 strategies adjusted.",
        )
        assert resp.success is True
        assert len(resp.affected_strategies) == 1
        assert resp.event_id.startswith("FR76-")


class TestLossPropagationService:
    """Test the LossPropagationService logic."""

    @pytest.mark.asyncio
    async def test_adjust_kelly_fraction(self):
        from src.api.loss_propagation import LossPropagationService

        svc = LossPropagationService()
        adjusted = await svc.adjust_kelly_fraction("STRAT_B", 0.10, adjustment_factor=0.75)
        assert adjusted == pytest.approx(0.075)

    @pytest.mark.asyncio
    async def test_get_correlated_strategies_above_threshold(self):
        from src.api.loss_propagation import LossPropagationService
        import numpy as np

        svc = LossPropagationService()
        corr_matrix = {
            "STRAT_A": np.array([1.0, 0.7, 0.3]),
            "STRAT_B": np.array([0.7, 1.0, 0.2]),
            "STRAT_C": np.array([0.3, 0.2, 1.0]),
        }
        result = await svc.get_correlated_strategies("STRAT_A", corr_matrix, threshold=0.5)
        strategy_ids = [r["strategy_id"] for r in result]
        assert "STRAT_B" in strategy_ids
        assert "STRAT_C" not in strategy_ids

    @pytest.mark.asyncio
    async def test_unknown_strategy_returns_empty(self):
        from src.api.loss_propagation import LossPropagationService
        import numpy as np

        svc = LossPropagationService()
        result = await svc.get_correlated_strategies("UNKNOWN", {}, threshold=0.5)
        assert result == []

    @pytest.mark.asyncio
    async def test_trigger_loss_propagation_returns_response(self):
        from src.api.loss_propagation import LossPropagationService, LossCapBreachEvent

        svc = LossPropagationService()
        breach = LossCapBreachEvent(
            strategy_id="STRAT_A",
            breach_time="2026-03-21T10:00:00Z",
            loss_amount=500.0,
            daily_loss_cap=500.0,
            correlation_threshold=0.5,
        )
        result = await svc.trigger_loss_propagation(breach)
        assert result.success is True
        assert result.source_strategy == "STRAT_A"
        assert result.event_id.startswith("FR76-")


class TestLossPropagationEndpoints:
    """Test loss propagation REST endpoints."""

    def setup_method(self):
        from fastapi import FastAPI
        from src.api.loss_propagation import router

        self.app = FastAPI()
        self.app.include_router(router)
        self.client = TestClient(self.app)

    def test_trigger_loss_propagation_endpoint(self):
        response = self.client.post(
            "/api/risk/loss-propagation/trigger",
            json={
                "strategy_id": "STRAT_A",
                "breach_time": "2026-03-21T10:00:00Z",
                "loss_amount": 500.0,
                "daily_loss_cap": 500.0,
                "correlation_threshold": 0.5,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["source_strategy"] == "STRAT_A"
        assert "affected_strategies" in data

    def test_get_propagation_history_returns_list(self):
        response = self.client.get("/api/risk/loss-propagation/history")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_get_strategy_kelly_status(self):
        response = self.client.get("/api/risk/loss-propagation/status/STRAT_A")
        assert response.status_code == 200
        data = response.json()
        assert "strategy_id" in data
        assert "base_kelly" in data
        assert "current_kelly" in data

    def test_reset_kelly_adjustment(self):
        response = self.client.post("/api/risk/loss-propagation/reset/STRAT_B")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["strategy_id"] == "STRAT_B"

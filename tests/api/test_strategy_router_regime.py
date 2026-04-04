# tests/api/test_strategy_router_regime.py
"""
Tests for Strategy Router & Regime State API Endpoints (Story 4.3)

Tests for:
- GET /api/risk/regime - Regime classification endpoint
- GET /api/risk/router/state - Strategy router state endpoint
- GET /api/risk/physics - Physics sensor outputs endpoint
"""

import pytest
import asyncio
from datetime import datetime, timezone
from pydantic import ValidationError


class TestRegimeResponseModel:
    """Test validation for RegimeResponse model."""

    def test_regime_response_valid_full_model(self):
        """Should create valid RegimeResponse with all fields."""
        from src.api.risk_endpoints import RegimeResponse

        response = RegimeResponse(
            regime="TREND",
            confidence_pct=85.0,
            transition_at_utc=datetime.now(timezone.utc),
            previous_regime="RANGE",
            active_strategy_count=3,
            paused_strategy_count=1
        )

        assert response.regime == "TREND"
        assert response.confidence_pct == 85.0
        assert response.previous_regime == "RANGE"
        assert response.active_strategy_count == 3
        assert response.paused_strategy_count == 1

    def test_regime_response_confidence_upper_bound(self):
        """Should accept confidence_pct at 100."""
        from src.api.risk_endpoints import RegimeResponse

        response = RegimeResponse(
            regime="CHAOS",
            confidence_pct=100.0,
            transition_at_utc=datetime.now(timezone.utc),
            previous_regime=None,
            active_strategy_count=0,
            paused_strategy_count=4
        )

        assert response.confidence_pct == 100.0

    def test_regime_response_confidence_lower_bound(self):
        """Should accept confidence_pct at 0."""
        from src.api.risk_endpoints import RegimeResponse

        response = RegimeResponse(
            regime="UNKNOWN",
            confidence_pct=0.0,
            transition_at_utc=datetime.now(timezone.utc),
            previous_regime=None,
            active_strategy_count=0,
            paused_strategy_count=0
        )

        assert response.confidence_pct == 0.0

    def test_regime_response_confidence_out_of_range_rejected(self):
        """Should reject confidence_pct > 100."""
        from src.api.risk_endpoints import RegimeResponse

        with pytest.raises(ValidationError):
            RegimeResponse(
                regime="TREND",
                confidence_pct=150.0,
                transition_at_utc=datetime.now(timezone.utc),
                previous_regime=None,
                active_strategy_count=0,
                paused_strategy_count=0
            )

    def test_regime_response_negative_count_rejected(self):
        """Should reject negative strategy counts."""
        from src.api.risk_endpoints import RegimeResponse

        with pytest.raises(ValidationError):
            RegimeResponse(
                regime="TREND",
                confidence_pct=50.0,
                transition_at_utc=datetime.now(timezone.utc),
                previous_regime=None,
                active_strategy_count=-1,
                paused_strategy_count=0
            )

    def test_regime_types_valid(self):
        """Should accept all valid regime types."""
        from src.api.risk_endpoints import RegimeResponse

        for regime in ["TREND", "RANGE", "BREAKOUT", "CHAOS", "UNKNOWN"]:
            response = RegimeResponse(
                regime=regime,
                confidence_pct=50.0,
                transition_at_utc=datetime.now(timezone.utc),
                previous_regime=None,
                active_strategy_count=0,
                paused_strategy_count=0
            )
            assert response.regime == regime


class TestStrategyStateItemModel:
    """Test validation for StrategyStateItem model."""

    def test_strategy_state_active(self):
        """Should create valid active strategy state."""
        from src.api.risk_endpoints import StrategyStateItem

        item = StrategyStateItem(
            strategy_id="trend-follower-001",
            status="active",
            pause_reason=None,
            eligible_regimes=["TREND", "BREAKOUT"]
        )

        assert item.strategy_id == "trend-follower-001"
        assert item.status == "active"
        assert item.pause_reason is None
        assert item.eligible_regimes == ["TREND", "BREAKOUT"]

    def test_strategy_state_paused(self):
        """Should create valid paused strategy state."""
        from src.api.risk_endpoints import StrategyStateItem

        item = StrategyStateItem(
            strategy_id="range-trader-002",
            status="paused",
            pause_reason="regime_mismatch",
            eligible_regimes=["RANGE"]
        )

        assert item.strategy_id == "range-trader-002"
        assert item.status == "paused"
        assert item.pause_reason == "regime_mismatch"

    def test_strategy_state_quarantine(self):
        """Should create valid quarantine strategy state."""
        from src.api.risk_endpoints import StrategyStateItem

        item = StrategyStateItem(
            strategy_id="volatility-adaptor-004",
            status="quarantine",
            pause_reason="risk_breach",
            eligible_regimes=["TREND", "RANGE", "BREAKOUT", "CHAOS"]
        )

        assert item.status == "quarantine"
        assert item.pause_reason == "risk_breach"

    def test_strategy_state_default_eligible_regimes(self):
        """Should have empty list as default for eligible_regimes."""
        from src.api.risk_endpoints import StrategyStateItem

        item = StrategyStateItem(
            strategy_id="test-strategy",
            status="active",
            pause_reason=None
        )

        assert item.eligible_regimes == []


class TestRouterStateEndpointBehavior:
    """Test live router-state fallback behavior."""

    def test_router_state_returns_empty_when_router_unavailable(self, monkeypatch):
        """Should return an honest empty list instead of demo strategy rows."""
        from src.api import risk_endpoints

        monkeypatch.setattr(risk_endpoints, "ROUTER_AVAILABLE", False)

        response = asyncio.run(risk_endpoints.get_router_state())

        assert response.strategies == []


class TestRouterMarketEndpointBehavior:
    """Test honest unavailable responses for router market state."""

    def test_market_state_returns_unavailable_payload_when_router_missing(self, monkeypatch):
        from src.api import router_endpoints

        monkeypatch.setattr(router_endpoints, "_strategy_router", None)

        response = asyncio.run(router_endpoints.get_market_state())

        assert response["regime"] is None
        assert response["symbols"] == []
        assert response["unavailable_reason"] == "Router not configured"

    def test_market_state_returns_unavailable_payload_when_mt5_missing(self, monkeypatch):
        from src.api import router_endpoints
        from src.data.brokers import mt5_socket_adapter

        class RouterStub:
            def get_status(self):
                return {
                    "sentinel": {
                        "regime_quality": 0.42,
                        "current_regime": "UNKNOWN",
                        "chaos": 0.0,
                        "volatility": "UNKNOWN",
                    }
                }

        monkeypatch.setattr(router_endpoints, "_strategy_router", RouterStub())
        monkeypatch.setattr(mt5_socket_adapter, "get_mt5_adapter", lambda: None, raising=False)

        response = asyncio.run(router_endpoints.get_market_state())

        assert response["regime"]["quality"] == 0.42
        assert response["symbols"] == []
        assert "MT5 adapter not connected" in response["unavailable_reason"]

    def test_router_state_returns_empty_when_router_has_no_registered_bots(self, monkeypatch):
        """Should not synthesize strategies when the router is configured but empty."""
        from src.api import risk_endpoints
        from src.api import router_endpoints

        class EmptyRouter:
            registered_bots = {}
            governor = None

        monkeypatch.setattr(risk_endpoints, "ROUTER_AVAILABLE", True)
        monkeypatch.setattr(router_endpoints, "_strategy_router", EmptyRouter())

        response = asyncio.run(risk_endpoints.get_router_state())

        assert response.strategies == []


class TestPhysicsOutputModels:
    """Test validation for physics output models."""

    def test_ising_output_normal(self):
        """Should create valid Ising output with normal alert."""
        from src.api.risk_endpoints import PhysicsIsingOutput

        output = PhysicsIsingOutput(
            magnetization=0.85,
            correlation_matrix={"EURUSD": 0.7},
            alert="normal"
        )

        assert output.magnetization == 0.85
        assert output.alert == "normal"

    def test_ising_output_magnetization_range(self):
        """Should accept magnetization values from -1 to 1."""
        from src.api.risk_endpoints import PhysicsIsingOutput

        # Test positive
        output = PhysicsIsingOutput(magnetization=1.0, alert="normal")
        assert output.magnetization == 1.0

        # Test negative
        output = PhysicsIsingOutput(magnetization=-1.0, alert="normal")
        assert output.magnetization == -1.0

    def test_lyapunov_output(self):
        """Should create valid Lyapunov output."""
        from src.api.risk_endpoints import PhysicsLyapunovOutput

        output = PhysicsLyapunovOutput(
            exponent_value=0.15,
            divergence_rate=0.02,
            alert="normal"
        )

        assert output.exponent_value == 0.15
        assert output.divergence_rate == 0.02
        assert output.alert == "normal"

    def test_lyapunov_output_warning(self):
        """Should create valid Lyapunov output with warning."""
        from src.api.risk_endpoints import PhysicsLyapunovOutput

        output = PhysicsLyapunovOutput(
            exponent_value=0.35,
            divergence_rate=None,
            alert="warning"
        )

        assert output.alert == "warning"
        assert output.divergence_rate is None

    def test_hmm_output(self):
        """Should create valid HMM output."""
        from src.api.risk_endpoints import PhysicsHMMOutput

        output = PhysicsHMMOutput(
            current_state="TREND",
            transition_probabilities={"TREND": 0.7, "RANGE": 0.3},
            alert="warning"  # HMM always warning in shadow mode
        )

        assert output.current_state == "TREND"
        assert output.transition_probabilities == {"TREND": 0.7, "RANGE": 0.3}
        assert output.alert == "warning"

    def test_hmm_output_shadow_mode_always_warning(self):
        """HMM should be in shadow mode - alert should be warning by default."""
        from src.api.risk_endpoints import PhysicsHMMOutput

        output = PhysicsHMMOutput(
            current_state=None,
            transition_probabilities=None,
            alert="warning"
        )

        assert output.alert == "warning"


class TestPhysicsResponseModel:
    """Test validation for complete PhysicsResponse model."""

    def test_physics_response_complete(self):
        """Should create complete physics response."""
        from src.api.risk_endpoints import PhysicsResponse, PhysicsIsingOutput, PhysicsLyapunovOutput, PhysicsHMMOutput

        from src.api.risk_endpoints import PhysicsKellyOutput
        response = PhysicsResponse(
            ising=PhysicsIsingOutput(magnetization=0.5, correlation_matrix=None, alert="warning"),
            lyapunov=PhysicsLyapunovOutput(exponent_value=0.25, divergence_rate=0.05, alert="normal"),
            hmm=PhysicsHMMOutput(current_state="RANGE", transition_probabilities={"RANGE": 0.8}, alert="warning"),
            kelly=PhysicsKellyOutput(fraction=0.5, multiplier=1.0, house_of_money=False, kelly_fraction_setting=0.5)
        )

        assert response.ising.alert == "warning"
        assert response.lyapunov.alert == "normal"
        assert response.hmm.alert == "warning"


class TestAlertStates:
    """Test alert state values."""

    def test_alert_states_valid(self):
        """Should accept all valid alert states."""
        from src.api.risk_endpoints import AlertState, PhysicsIsingOutput

        for alert in ["normal", "warning", "critical"]:
            output = PhysicsIsingOutput(magnetization=0.5, alert=alert)
            assert output.alert == alert


class TestPauseReasons:
    """Test pause reason values."""

    def test_pause_reasons_valid(self):
        """Should accept all valid pause reasons."""
        from src.api.risk_endpoints import PauseReason

        valid_reasons = [
            "calendar_rule",
            "risk_breach",
            "manual",
            "regime_mismatch"
        ]

        for reason in valid_reasons:
            assert reason in valid_reasons

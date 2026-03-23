"""
Tests for A/B Race Board API Endpoints

Tests model structures and endpoint logic for strategy variant comparison.
"""
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient
from fastapi import FastAPI


class TestVariantMetricsModel:
    """Test suite for VariantMetrics model."""

    def test_variant_metrics_required_fields(self):
        from src.api.ab_race_endpoints import VariantMetrics

        m = VariantMetrics(pnl=100.0, trade_count=50, drawdown=5.0, sharpe=1.5)
        assert m.pnl == 100.0
        assert m.trade_count == 50
        assert m.drawdown == 5.0
        assert m.sharpe == 1.5

    def test_variant_metrics_optional_defaults(self):
        from src.api.ab_race_endpoints import VariantMetrics

        m = VariantMetrics(pnl=0.0, trade_count=0, drawdown=0.0, sharpe=0.0)
        assert m.win_rate == 0
        assert m.avg_profit == 0
        assert m.avg_loss == 0
        assert m.profit_factor == 0
        assert m.max_consecutive_wins == 0
        assert m.max_consecutive_losses == 0


class TestStatisticalSignificanceModel:
    """Test suite for StatisticalSignificance model."""

    def test_significant_result(self):
        from src.api.ab_race_endpoints import StatisticalSignificance

        sig = StatisticalSignificance(
            p_value=0.02,
            is_significant=True,
            winner="A",
            confidence_level=98.0,
            sample_size_a=100,
            sample_size_b=120,
        )
        assert sig.is_significant is True
        assert sig.winner == "A"
        assert sig.confidence_level == 98.0

    def test_non_significant_result(self):
        from src.api.ab_race_endpoints import StatisticalSignificance

        sig = StatisticalSignificance(
            p_value=0.45,
            is_significant=False,
            winner=None,
            confidence_level=55.0,
            sample_size_a=30,
            sample_size_b=35,
        )
        assert sig.is_significant is False
        assert sig.winner is None


class TestABComparisonResponseModel:
    """Test suite for ABComparisonResponse model."""

    def test_ab_comparison_response_structure(self):
        from src.api.ab_race_endpoints import (
            ABComparisonResponse,
            VariantMetrics,
        )
        from datetime import datetime

        metrics = VariantMetrics(pnl=500.0, trade_count=100, drawdown=5.0, sharpe=2.0)
        response = ABComparisonResponse(
            strategy_id="strat-001",
            variant_a="vanilla",
            variant_b="spiced",
            metrics_a=metrics,
            metrics_b=metrics,
            statistical_significance=None,
            timestamp=datetime.utcnow().isoformat(),
        )
        assert response.strategy_id == "strat-001"
        assert response.variant_a == "vanilla"
        assert response.variant_b == "spiced"
        assert response.statistical_significance is None


class TestCalculateStatisticalSignificance:
    """Test the statistical significance calculation function."""

    def test_insufficient_data_returns_none(self):
        from src.api.ab_race_endpoints import calculate_statistical_significance

        # Less than 50 trades in each group → None
        result = calculate_statistical_significance([1.0] * 30, [2.0] * 40)
        assert result is None

    def test_sufficient_data_returns_result(self):
        from src.api.ab_race_endpoints import calculate_statistical_significance

        trades_a = [10.0] * 60 + [-5.0] * 40
        trades_b = [8.0] * 55 + [-4.0] * 45
        result = calculate_statistical_significance(trades_a, trades_b)
        # With 100 trades each, should return a result
        assert result is not None
        assert 0.0 <= result.p_value <= 1.0
        assert isinstance(result.is_significant, bool)
        assert result.sample_size_a == 100
        assert result.sample_size_b == 100

    def test_identical_distributions_not_significant(self):
        from src.api.ab_race_endpoints import calculate_statistical_significance
        import random
        random.seed(42)

        # Identical distributions should not be significant
        base = [random.gauss(10, 2) for _ in range(100)]
        result = calculate_statistical_significance(base[:], base[:])
        assert result is not None
        assert result.p_value > 0.05 or result.is_significant is False


class TestABRaceEndpoints:
    """Test A/B race REST endpoints using TestClient."""

    def setup_method(self):
        from fastapi import FastAPI
        from src.api.ab_race_endpoints import router

        self.app = FastAPI()
        self.app.include_router(router)
        self.client = TestClient(self.app)

    def test_get_variant_metrics_endpoint(self):
        response = self.client.get(
            "/api/strategies/variants/strat-001/metrics",
            params={"variant_id": "vanilla"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "pnl" in data
        assert "trade_count" in data
        assert "sharpe" in data

    def test_compare_variants_endpoint(self):
        response = self.client.get(
            "/api/strategies/variants/strat-001/compare",
            params={"variant_a": "vanilla", "variant_b": "spiced"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["strategy_id"] == "strat-001"
        assert data["variant_a"] == "vanilla"
        assert data["variant_b"] == "spiced"
        assert "metrics_a" in data
        assert "metrics_b" in data
        assert "timestamp" in data

    def test_compare_variants_returns_iso_timestamp(self):
        response = self.client.get(
            "/api/strategies/variants/test-strategy/compare",
            params={"variant_a": "v1", "variant_b": "v2"},
        )
        assert response.status_code == 200
        data = response.json()
        # Timestamp should be a valid ISO string
        assert "T" in data["timestamp"]

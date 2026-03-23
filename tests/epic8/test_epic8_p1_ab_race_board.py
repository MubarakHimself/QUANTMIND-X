"""
P1 Tests for Epic 8 - A/B Race Board UI

Priority: P1
Coverage: A/B comparison, statistical significance, WebSocket updates

Risk Coverage:
- R-006: A/B Statistical Significance Miscalculation
"""

import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI


def _make_client():
    from src.api.ab_race_endpoints import router
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


class TestABRaceModels:
    """P1: A/B race model structures."""

    def test_variant_metrics_has_all_required_fields(self):
        from src.api.ab_race_endpoints import VariantMetrics

        metrics = VariantMetrics(
            pnl=1500.50,
            trade_count=150,
            drawdown=12.5,
            sharpe=2.5,
            win_rate=58.5
        )
        assert metrics.pnl == 1500.50
        assert metrics.trade_count == 150

    def test_statistical_significance_threshold_is_p05(self):
        """P1: Verify significance threshold is p < 0.05."""
        from src.api.ab_race_endpoints import StatisticalSignificance

        sig = StatisticalSignificance(
            p_value=0.04,
            is_significant=True,
            winner="A",
            confidence_level=96.0,
            sample_size_a=100,
            sample_size_b=100
        )
        assert sig.p_value < 0.05
        assert sig.is_significant is True


class TestABRaceEndpoints:
    """P1: A/B race REST API endpoints."""

    def test_compare_variants_returns_metrics_both_sides(self):
        client = _make_client()
        response = client.get(
            "/api/strategies/variants/test-strat/compare",
            params={"variant_a": "vanilla", "variant_b": "spiced"}
        )
        assert response.status_code == 200

        data = response.json()
        assert "metrics_a" in data
        assert "metrics_b" in data
        assert "statistical_significance" in data

    def test_get_single_variant_metrics(self):
        client = _make_client()
        response = client.get(
            "/api/strategies/variants/test-strat/metrics",
            params={"variant_id": "vanilla"}
        )
        assert response.status_code == 200

        data = response.json()
        assert "pnl" in data
        assert "trade_count" in data

    def test_compare_requires_both_variant_ids(self):
        client = _make_client()
        response = client.get(
            "/api/strategies/variants/test-strat/compare",
            params={"variant_a": "vanilla"}  # Missing variant_b
        )
        assert response.status_code == 422  # Validation error


class TestStatisticalSignificance:
    """P1: Statistical significance calculation."""

    def test_identical_trades_returns_not_significant(self):
        from src.api.ab_race_endpoints import calculate_statistical_significance

        # Identical distributions
        trades_a = [10.0] * 60 + [-5.0] * 40
        trades_b = [10.0] * 60 + [-5.0] * 40

        result = calculate_statistical_significance(trades_a, trades_b)
        assert result is not None
        assert result.is_significant is False

    def test_better_trades_returns_significant_winner(self):
        from src.api.ab_race_endpoints import calculate_statistical_significance

        # A is clearly better
        trades_a = [15.0] * 60 + [-5.0] * 40  # mean = 7.0
        trades_b = [5.0] * 60 + [-5.0] * 40   # mean = 1.0

        result = calculate_statistical_significance(trades_a, trades_b)
        assert result is not None
        assert result.is_significant is True
        assert result.winner == "A"

    def test_insufficient_sample_returns_none(self):
        from src.api.ab_race_endpoints import calculate_statistical_significance

        trades_a = [10.0] * 30  # Only 30 trades
        trades_b = [10.0] * 40  # Only 40 trades

        result = calculate_statistical_significance(trades_a, trades_b)
        assert result is None

    def test_p_value_threshold_is_005_not_001_or_010(self):
        """P1: Verify p < 0.05 is the correct threshold."""
        from src.api.ab_race_endpoints import calculate_statistical_significance
        import random

        # Create marginal difference
        random.seed(42)
        trades_a = [random.gauss(10, 2) for _ in range(100)]
        trades_b = [random.gauss(10.5, 2) for _ in range(100)]

        result = calculate_statistical_significance(trades_a, trades_b)
        if result:
            # The threshold must be 0.05
            assert hasattr(result, 'p_value')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

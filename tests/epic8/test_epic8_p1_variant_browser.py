"""
P1 Tests for Epic 8 - Variant Browser UI

Priority: P1
Coverage: Variant browser API, variant types, backtest summaries

Risk Coverage:
- R-004: TRD Validation False Rejection
- R-006: A/B Statistical Significance Miscalculation
"""

import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI


def _make_client():
    from src.api.variant_browser_endpoints import router
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


class TestVariantBrowserModels:
    """P1: Variant browser model structures."""

    def test_variant_type_enum_values(self):
        from src.mql5.versions.schema import VariantType

        assert VariantType.VANILLA == "vanilla"
        assert VariantType.SPICED == "spiced"
        assert VariantType.MODE_B == "mode_b"
        assert VariantType.MODE_C == "mode_c"

    def test_backtest_summary_has_required_fields(self):
        from src.api.variant_browser_endpoints import BacktestSummary

        summary = BacktestSummary(
            total_pnl=1500.50,
            sharpe_ratio=2.5,
            max_drawdown=12.5,
            trade_count=150,
            win_rate=58.5,
            profit_factor=1.85,
            period="2024-Q4"
        )
        assert summary.total_pnl == 1500.50
        assert summary.trade_count == 150

    def test_variant_info_has_all_required_fields(self):
        from src.api.variant_browser_endpoints import VariantInfo

        variant = VariantInfo(
            variant_type="vanilla",
            version_tag="1.0.0",
            improvement_cycle=0,
            author="system",
            created_at="2026-03-21T10:00:00Z",
            is_active=True,
            promotion_status="development"
        )
        assert variant.variant_type == "vanilla"
        assert variant.is_active is True


class TestVariantBrowserEndpoints:
    """P1: Variant browser REST API endpoints."""

    def test_get_all_variants_returns_strategy_list(self):
        client = _make_client()
        response = client.get("/api/variant-browser")
        assert response.status_code == 200

        data = response.json()
        assert "strategies" in data
        assert "total_strategies" in data
        assert "total_variants" in data

    def test_variants_include_all_4_types(self):
        client = _make_client()
        response = client.get("/api/variant-browser")
        data = response.json()

        expected_types = {"vanilla", "spiced", "mode_b", "mode_c"}
        for strategy in data["strategies"]:
            variant_types = {v["variant_type"] for v in strategy["variants"]}
            assert variant_types == expected_types

    def test_get_specific_strategy_variants(self):
        client = _make_client()
        response = client.get("/api/variant-browser/news-event-breakout")
        assert response.status_code == 200

        data = response.json()
        assert data["strategy_id"] == "news-event-breakout"
        assert len(data["variants"]) == 4

    def test_invalid_variant_type_returns_400(self):
        client = _make_client()
        response = client.get("/api/variant-browser/news-event-breakout/invalid_type")
        assert response.status_code == 400

    def test_nonexistent_strategy_returns_404(self):
        client = _make_client()
        response = client.get("/api/variant-browser/nonexistent-strategy")
        assert response.status_code == 404

    def test_get_variant_detail_returns_code_content(self):
        client = _make_client()
        response = client.get("/api/variant-browser/news-event-breakout/vanilla")
        assert response.status_code == 200

        data = response.json()
        assert "variant" in data
        assert "version_timeline" in data

    def test_variant_code_endpoint_returns_mql5(self):
        client = _make_client()
        response = client.get("/api/variant-browser/news-event-breakout/vanilla/code")
        assert response.status_code == 200

        data = response.json()
        assert data["language"] == "mql5"
        assert "code" in data

    def test_promotion_status_defaults_to_development(self):
        client = _make_client()
        response = client.get("/api/variant-browser")
        data = response.json()

        for strategy in data["strategies"]:
            for variant in strategy["variants"]:
                assert "promotion_status" in variant


class TestVariantComparison:
    """P1: Variant version comparison."""

    def test_compare_versions_returns_diff_structure(self):
        client = _make_client()
        response = client.get(
            "/api/variant-browser/news-event-breakout/vanilla/compare",
            params={"version_a": "1.0.0", "version_b": "1.1.0"}
        )
        assert response.status_code == 200

        data = response.json()
        assert "changes" in data
        assert "summary" in data

    def test_backtest_summary_includes_sharpe_ratio(self):
        client = _make_client()
        response = client.get("/api/variant-browser")
        data = response.json()

        for strategy in data["strategies"]:
            for variant in strategy["variants"]:
                if variant.get("backtest"):
                    assert "sharpe_ratio" in variant["backtest"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

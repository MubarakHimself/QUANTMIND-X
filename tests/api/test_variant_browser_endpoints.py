"""
Tests for Variant Browser API Endpoints

Tests variant browser model structures and endpoint responses.
"""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import FastAPI


def _make_client():
    from src.api.variant_browser_endpoints import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


class TestVariantBrowserModels:
    """Test variant browser Pydantic model structures."""

    def test_backtest_summary_defaults(self):
        from src.api.variant_browser_endpoints import BacktestSummary

        bt = BacktestSummary()
        assert bt.total_pnl == 0.0
        assert bt.trade_count == 0
        assert bt.win_rate == 0.0

    def test_backtest_summary_with_values(self):
        from src.api.variant_browser_endpoints import BacktestSummary

        bt = BacktestSummary(
            total_pnl=1500.0,
            sharpe_ratio=2.1,
            max_drawdown=8.5,
            trade_count=150,
            win_rate=58.0,
            profit_factor=1.8,
            period="2024-Q4",
        )
        assert bt.total_pnl == 1500.0
        assert bt.period == "2024-Q4"

    def test_variant_info_required_fields(self):
        from src.api.variant_browser_endpoints import VariantInfo

        vi = VariantInfo(variant_type="vanilla", version_tag="1.0.0")
        assert vi.variant_type == "vanilla"
        assert vi.version_tag == "1.0.0"
        assert vi.is_active is False
        assert vi.backtest is None

    def test_strategy_variants_structure(self):
        from src.api.variant_browser_endpoints import StrategyVariants, VariantInfo

        sv = StrategyVariants(
            strategy_id="trend-follower",
            strategy_name="Trend Follower",
            variants=[
                VariantInfo(variant_type="vanilla", version_tag="1.0.0"),
                VariantInfo(variant_type="spiced", version_tag="1.1.0"),
            ],
            variant_counts={"vanilla": 1, "spiced": 1},
        )
        assert sv.strategy_id == "trend-follower"
        assert len(sv.variants) == 2

    def test_variant_browser_response_defaults(self):
        from src.api.variant_browser_endpoints import VariantBrowserResponse

        resp = VariantBrowserResponse()
        assert resp.strategies == []
        assert resp.total_strategies == 0
        assert resp.total_variants == 0

    def test_variant_detail_response_structure(self):
        from src.api.variant_browser_endpoints import VariantDetailResponse, VariantInfo

        resp = VariantDetailResponse(
            strategy_id="news-event",
            strategy_name="News Event Breakout",
            variant=VariantInfo(variant_type="vanilla", version_tag="1.0.0"),
            version_timeline=[],
        )
        assert resp.strategy_id == "news-event"
        assert resp.code_content is None


class TestGetPromotionStatus:
    """Test promotion status logic."""

    def test_development_status_cycle_0(self):
        from src.api.variant_browser_endpoints import _get_promotion_status

        assert _get_promotion_status(0) == "development"

    def test_paper_trading_status_cycle_1(self):
        from src.api.variant_browser_endpoints import _get_promotion_status

        assert _get_promotion_status(1) == "paper_trading"

    def test_sit_status_cycle_2(self):
        from src.api.variant_browser_endpoints import _get_promotion_status

        assert _get_promotion_status(2) == "sit"

    def test_live_status_cycle_3_or_more(self):
        from src.api.variant_browser_endpoints import _get_promotion_status

        assert _get_promotion_status(3) == "live"
        assert _get_promotion_status(5) == "live"


class TestVariantBrowserEndpoints:
    """Test variant browser REST endpoints."""

    def test_get_all_variants_returns_strategies(self):
        client = _make_client()
        response = client.get("/api/variant-browser")
        assert response.status_code == 200
        data = response.json()
        assert "strategies" in data
        assert "total_strategies" in data
        assert data["total_strategies"] > 0

    def test_get_all_variants_has_variant_types(self):
        client = _make_client()
        response = client.get("/api/variant-browser")
        data = response.json()
        for strategy in data["strategies"]:
            assert "variants" in strategy
            variant_types = {v["variant_type"] for v in strategy["variants"]}
            assert "vanilla" in variant_types

    def test_filter_by_strategy_id(self):
        client = _make_client()
        response = client.get(
            "/api/variant-browser",
            params={"strategy_id": "news-event-breakout"},
        )
        assert response.status_code == 200
        data = response.json()
        for s in data["strategies"]:
            assert s["strategy_id"] == "news-event-breakout"

    def test_get_specific_strategy_variants(self):
        client = _make_client()
        response = client.get("/api/variant-browser/news-event-breakout")
        assert response.status_code == 200
        data = response.json()
        assert data["strategy_id"] == "news-event-breakout"

    def test_get_nonexistent_strategy_returns_404(self):
        client = _make_client()
        response = client.get("/api/variant-browser/nonexistent-strategy-xyz")
        assert response.status_code == 404

    def test_get_variant_detail_vanilla(self):
        client = _make_client()
        response = client.get("/api/variant-browser/news-event-breakout/vanilla")
        assert response.status_code == 200
        data = response.json()
        assert data["variant"]["variant_type"] == "vanilla"

    def test_get_variant_detail_invalid_type_returns_400(self):
        client = _make_client()
        response = client.get("/api/variant-browser/news-event-breakout/invalid_type")
        assert response.status_code == 400

    def test_get_variant_code_returns_mql5(self):
        client = _make_client()
        response = client.get("/api/variant-browser/news-event-breakout/vanilla/code")
        assert response.status_code == 200
        data = response.json()
        assert data["language"] == "mql5"
        assert "code" in data

    def test_compare_variants_invalid_type_400(self):
        client = _make_client()
        response = client.get(
            "/api/variant-browser/news-event-breakout/bad_type/compare",
            params={"version_a": "1.0.0", "version_b": "1.1.0"},
        )
        assert response.status_code == 400

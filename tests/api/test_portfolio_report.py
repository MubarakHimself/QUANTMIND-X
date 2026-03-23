"""
Tests for Portfolio Report API Endpoints

Story: 7-8-risk-trading-portfolio-department-real-implementations
Tests AC #3: GET /api/portfolio/report
"""
import pytest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient


# Create a mock portfolio head instance
_mock_portfolio_instance = MagicMock()
_mock_portfolio_instance.generate_portfolio_report.return_value = {
    "total_equity": 85000.0,
    "pnl_attribution": {
        "by_strategy": [
            {"strategy": "TrendFollower_v2.1", "pnl": 2450.0, "percentage": 51.5},
            {"strategy": "RangeTrader_v1.5", "pnl": 820.0, "percentage": 17.2},
            {"strategy": "BreakoutScaler_v3.0", "pnl": 1530.0, "percentage": 32.1},
            {"strategy": "ScalperPro_v1.0", "pnl": -320.0, "percentage": -6.7},
        ],
        "by_broker": [
            {"broker": "ICMarkets", "pnl": 3100.0, "percentage": 65.1},
            {"broker": "OANDA", "pnl": 1280.0, "percentage": 26.9},
            {"broker": "Pepperstone", "pnl": 100.0, "percentage": 2.1},
        ],
    },
    "drawdown_by_account": [
        {"account_id": "acc_main", "drawdown_pct": 8.3},
        {"account_id": "acc_backup", "drawdown_pct": 5.1},
        {"account_id": "acc_paper", "drawdown_pct": 2.4},
    ],
    "generated_at": "2026-03-19T12:00:00+00:00",
}
_mock_portfolio_instance.get_total_equity.return_value = {
    "total_equity": 85000.0,
    "accounts": [
        {"account_id": "acc_main", "balance": 50000.0},
        {"account_id": "acc_backup", "balance": 25000.0},
        {"account_id": "acc_paper", "balance": 10000.0},
    ],
    "account_count": 3,
}
_mock_portfolio_instance.get_strategy_pnl.return_value = {
    "period": "all",
    "total_pnl": 4760.0,
    "by_strategy": [
        {"strategy": "TrendFollower_v2.1", "pnl": 2450.0, "percentage": 51.5},
        {"strategy": "RangeTrader_v1.5", "pnl": 820.0, "percentage": 17.2},
        {"strategy": "BreakoutScaler_v3.0", "pnl": 1530.0, "percentage": 32.1},
        {"strategy": "ScalperPro_v1.0", "pnl": -320.0, "percentage": -6.7},
    ],
}
_mock_portfolio_instance.get_broker_pnl.return_value = {
    "period": "all",
    "total_pnl": 4760.0,
    "by_broker": [
        {"broker": "ICMarkets", "pnl": 3100.0, "percentage": 65.1},
        {"broker": "OANDA", "pnl": 1280.0, "percentage": 26.9},
        {"broker": "Pepperstone", "pnl": 100.0, "percentage": 2.1},
    ],
}
_mock_portfolio_instance.get_account_drawdowns.return_value = {
    "by_account": [
        {"account_id": "acc_main", "drawdown_pct": 8.3},
        {"account_id": "acc_backup", "drawdown_pct": 5.1},
        {"account_id": "acc_paper", "drawdown_pct": 2.4},
    ],
}


@pytest.fixture
def client():
    """Create test client with mocked portfolio head."""
    with patch("src.api.portfolio_endpoints.get_portfolio_head", return_value=_mock_portfolio_instance):
        from src.api.portfolio_endpoints import router
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)

        with TestClient(app) as test_client:
            yield test_client


class TestPortfolioReportEndpoint:
    """Test suite for portfolio report endpoints."""

    def test_get_portfolio_report(self, client):
        """Test GET /api/portfolio/report returns correct structure (AC #3)."""
        response = client.get("/api/portfolio/report")

        assert response.status_code == 200

        data = response.json()

        # AC #3: Returns total equity, P&L attribution per strategy,
        # P&L attribution per broker, drawdown per account
        assert "total_equity" in data
        assert "pnl_attribution" in data
        assert "drawdown_by_account" in data

    def test_portfolio_report_total_equity(self, client):
        """Test total equity is returned."""
        response = client.get("/api/portfolio/report")
        data = response.json()

        assert data["total_equity"] == 85000.0

    def test_portfolio_report_pnl_attribution_structure(self, client):
        """Test P&L attribution structure."""
        response = client.get("/api/portfolio/report")
        data = response.json()

        pnl = data["pnl_attribution"]
        assert "by_strategy" in pnl
        assert "by_broker" in pnl

    def test_portfolio_report_by_strategy(self, client):
        """Test P&L attribution by strategy."""
        response = client.get("/api/portfolio/report")
        data = response.json()

        strategies = data["pnl_attribution"]["by_strategy"]
        assert len(strategies) > 0

        # Check first strategy has required fields
        assert "strategy" in strategies[0]
        assert "pnl" in strategies[0]
        assert "percentage" in strategies[0]

    def test_portfolio_report_by_broker(self, client):
        """Test P&L attribution by broker."""
        response = client.get("/api/portfolio/report")
        data = response.json()

        brokers = data["pnl_attribution"]["by_broker"]
        assert len(brokers) > 0

        # Check first broker has required fields
        assert "broker" in brokers[0]
        assert "pnl" in brokers[0]
        assert "percentage" in brokers[0]

    def test_portfolio_report_drawdown_by_account(self, client):
        """Test drawdown by account."""
        response = client.get("/api/portfolio/report")
        data = response.json()

        drawdowns = data["drawdown_by_account"]
        assert len(drawdowns) > 0

        dd = drawdowns[0]
        assert "account_id" in dd
        assert "drawdown_pct" in dd


class TestEquityEndpoint:
    """Test suite for equity endpoint."""

    def test_get_total_equity(self, client):
        """Test GET /api/portfolio/equity."""
        response = client.get("/api/portfolio/equity")

        assert response.status_code == 200

        data = response.json()
        assert "total_equity" in data
        assert "account_count" in data


class TestStrategyPnLEndpoint:
    """Test suite for strategy P&L endpoint."""

    def test_get_strategy_pnl(self, client):
        """Test GET /api/portfolio/pnl/strategy."""
        response = client.get("/api/portfolio/pnl/strategy")

        assert response.status_code == 200

        data = response.json()
        assert "period" in data
        assert "total_pnl" in data
        assert "by_strategy" in data


class TestBrokerPnLEndpoint:
    """Test suite for broker P&L endpoint."""

    def test_get_broker_pnl(self, client):
        """Test GET /api/portfolio/pnl/broker."""
        response = client.get("/api/portfolio/pnl/broker")

        assert response.status_code == 200

        data = response.json()
        assert "period" in data
        assert "total_pnl" in data
        assert "by_broker" in data


class TestDrawdownsEndpoint:
    """Test suite for drawdowns endpoint."""

    def test_get_account_drawdowns(self, client):
        """Test GET /api/portfolio/drawdowns."""
        response = client.get("/api/portfolio/drawdowns")

        assert response.status_code == 200

        data = response.json()
        assert "by_account" in data

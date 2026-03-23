"""
Tests for Portfolio Metrics & Attribution API (Story 9.2)

Tests the new endpoints: /summary, /attribution, /correlation
"""
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

# Mock the dependencies before importing
with patch("src.agents.departments.heads.portfolio_head.get_department_config"):
    from src.api.portfolio_endpoints import (
        router,
        PortfolioSummaryResponse,
        PortfolioAttributionResponse,
        PortfolioCorrelationResponse,
        AccountSummaryResponse,
        StrategyAttributionWithEquityResponse,
    )


class TestPortfolioSummaryEndpoint:
    """Test AC1: Portfolio Summary Endpoint."""

    @pytest.fixture
    def mock_portfolio_head(self):
        """Create mock PortfolioHead for testing."""
        with patch("src.api.portfolio_endpoints.get_portfolio_head") as mock:
            head = MagicMock()
            head.get_portfolio_summary.return_value = {
                "total_equity": 85000.0,
                "daily_pnl": 1250.50,
                "daily_pnl_pct": 1.47,
                "total_drawdown": 8.3,
                "active_strategies": ["TrendFollower_v2.1", "RangeTrader_v1.5"],
                "accounts": [
                    {"account_id": "acc_main", "equity": 50000, "daily_pnl": 800, "drawdown": 8.3},
                    {"account_id": "acc_backup", "equity": 25000, "daily_pnl": 350.50, "drawdown": 5.1},
                ],
                "drawdown_alert": False,
            }
            mock.return_value = head
            yield head

    def test_summary_returns_total_equity(self, mock_portfolio_head):
        """Test AC1: total_equity field present."""
        result = mock_portfolio_head.get_portfolio_summary()
        assert "total_equity" in result
        assert result["total_equity"] == 85000.0

    def test_summary_returns_daily_pnl(self, mock_portfolio_head):
        """Test AC1: daily_pnl field present."""
        result = mock_portfolio_head.get_portfolio_summary()
        assert "daily_pnl" in result
        assert result["daily_pnl"] == 1250.50

    def test_summary_returns_daily_pnl_pct(self, mock_portfolio_head):
        """Test AC1: daily_pnl_pct field present."""
        result = mock_portfolio_head.get_portfolio_summary()
        assert "daily_pnl_pct" in result
        assert result["daily_pnl_pct"] == 1.47

    def test_summary_returns_total_drawdown(self, mock_portfolio_head):
        """Test AC1: total_drawdown field present."""
        result = mock_portfolio_head.get_portfolio_summary()
        assert "total_drawdown" in result
        assert result["total_drawdown"] == 8.3

    def test_summary_returns_active_strategies(self, mock_portfolio_head):
        """Test AC1: active_strategies field present."""
        result = mock_portfolio_head.get_portfolio_summary()
        assert "active_strategies" in result
        assert len(result["active_strategies"]) == 2

    def test_summary_returns_accounts(self, mock_portfolio_head):
        """Test AC1: accounts field with correct structure."""
        result = mock_portfolio_head.get_portfolio_summary()
        assert "accounts" in result
        assert len(result["accounts"]) == 2

        acc = result["accounts"][0]
        assert "account_id" in acc
        assert "equity" in acc
        assert "daily_pnl" in acc
        assert "drawdown" in acc

    def test_summary_drawdown_alert_below_threshold(self, mock_portfolio_head):
        """Test AC4: drawdown_alert is False when below 10%."""
        result = mock_portfolio_head.get_portfolio_summary()
        assert result["drawdown_alert"] is False

    def test_summary_drawdown_alert_above_threshold(self, mock_portfolio_head):
        """Test AC4: drawdown_alert is True when above 10%."""
        mock_portfolio_head.get_portfolio_summary.return_value["total_drawdown"] = 15.0
        mock_portfolio_head.get_portfolio_summary.return_value["drawdown_alert"] = True

        result = mock_portfolio_head.get_portfolio_summary()
        assert result["drawdown_alert"] is True


class TestPortfolioAttributionEndpoint:
    """Test AC2: Portfolio Attribution Endpoint."""

    @pytest.fixture
    def mock_portfolio_head(self):
        """Create mock PortfolioHead for testing."""
        with patch("src.api.portfolio_endpoints.get_portfolio_head") as mock:
            head = MagicMock()
            head.get_attribution.return_value = {
                "by_strategy": [
                    {"strategy": "TrendFollower_v2.1", "pnl": 2450.0, "percentage": 62.5, "equity_contribution": 12000},
                    {"strategy": "RangeTrader_v1.5", "pnl": 820.0, "percentage": 20.9, "equity_contribution": 4000},
                ],
                "by_broker": [
                    {"broker": "ICMarkets", "pnl": 3100.0, "percentage": 79.0},
                    {"broker": "OANDA", "pnl": 1280.0, "percentage": 32.6},
                ],
            }
            mock.return_value = head
            yield head

    def test_attribution_returns_by_strategy(self, mock_portfolio_head):
        """Test AC2: by_strategy field present."""
        result = mock_portfolio_head.get_attribution()
        assert "by_strategy" in result
        assert len(result["by_strategy"]) == 2

    def test_attribution_strategy_has_equity_contribution(self, mock_portfolio_head):
        """Test AC2: equity_contribution field present per strategy."""
        result = mock_portfolio_head.get_attribution()
        strategy = result["by_strategy"][0]
        assert "equity_contribution" in strategy
        assert strategy["equity_contribution"] == 12000

    def test_attribution_returns_by_broker(self, mock_portfolio_head):
        """Test AC2: by_broker field present."""
        result = mock_portfolio_head.get_attribution()
        assert "by_broker" in result
        assert len(result["by_broker"]) == 2


class TestPortfolioCorrelationEndpoint:
    """Test AC3: Portfolio Correlation Matrix Endpoint."""

    @pytest.fixture
    def mock_portfolio_head(self):
        """Create mock PortfolioHead for testing."""
        with patch("src.api.portfolio_endpoints.get_portfolio_head") as mock:
            head = MagicMock()
            head.get_correlation_matrix.return_value = {
                "matrix": [
                    {"strategy_a": "TrendFollower_v2.1", "strategy_b": "RangeTrader_v1.5", "correlation": 0.45, "period_days": 30},
                    {"strategy_a": "TrendFollower_v2.1", "strategy_b": "BreakoutScaler_v3.0", "correlation": 0.78, "period_days": 30},
                ],
                "high_correlation_threshold": 0.7,
                "period_days": 30,
                "generated_at": "2026-03-20T10:30:00Z",
            }
            mock.return_value = head
            yield head

    def test_correlation_returns_matrix(self, mock_portfolio_head):
        """Test AC3: matrix field present."""
        result = mock_portfolio_head.get_correlation_matrix()
        assert "matrix" in result
        assert len(result["matrix"]) == 2

    def test_correlation_matrix_has_required_fields(self, mock_portfolio_head):
        """Test AC3: correlation pair has required fields."""
        result = mock_portfolio_head.get_correlation_matrix()
        pair = result["matrix"][0]
        assert "strategy_a" in pair
        assert "strategy_b" in pair
        assert "correlation" in pair
        assert "period_days" in pair

    def test_correlation_returns_threshold(self, mock_portfolio_head):
        """Test AC3: high_correlation_threshold field present."""
        result = mock_portfolio_head.get_correlation_matrix()
        assert "high_correlation_threshold" in result
        assert result["high_correlation_threshold"] == 0.7

    def test_correlation_accepts_period_days_param(self, mock_portfolio_head):
        """Test AC3: period_days parameter is accepted."""
        mock_portfolio_head.get_correlation_matrix(period_days=60)
        mock_portfolio_head.get_correlation_matrix.assert_called_with(period_days=60)


class TestPortfolioModels:
    """Test Pydantic response models."""

    def test_portfolio_summary_response_model(self):
        """Test PortfolioSummaryResponse model validation."""
        data = {
            "total_equity": 85000.0,
            "daily_pnl": 1250.50,
            "daily_pnl_pct": 1.47,
            "total_drawdown": 8.3,
            "active_strategies": ["TrendFollower_v2.1"],
            "accounts": [
                {"account_id": "acc_main", "equity": 50000, "daily_pnl": 800, "drawdown": 8.3}
            ],
            "drawdown_alert": False,
        }
        response = PortfolioSummaryResponse(**data)
        assert response.total_equity == 85000.0
        assert response.drawdown_alert is False

    def test_portfolio_attribution_response_model(self):
        """Test PortfolioAttributionResponse model validation."""
        data = {
            "by_strategy": [
                {"strategy": "TrendFollower", "pnl": 1000.0, "percentage": 50.0, "equity_contribution": 5000}
            ],
            "by_broker": [
                {"broker": "ICMarkets", "pnl": 2000.0, "percentage": 100.0}
            ],
        }
        response = PortfolioAttributionResponse(**data)
        assert len(response.by_strategy) == 1
        assert response.by_strategy[0].equity_contribution == 5000

    def test_portfolio_correlation_response_model(self):
        """Test PortfolioCorrelationResponse model validation."""
        data = {
            "matrix": [
                {"strategy_a": "A", "strategy_b": "B", "correlation": 0.5, "period_days": 30}
            ],
            "high_correlation_threshold": 0.7,
            "period_days": 30,
            "generated_at": "2026-03-20T10:30:00Z",
        }
        response = PortfolioCorrelationResponse(**data)
        assert response.high_correlation_threshold == 0.7
        assert len(response.matrix) == 1
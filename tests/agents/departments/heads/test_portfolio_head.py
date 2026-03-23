"""
Tests for Portfolio Department Head - Report Generation

Story: 7-8-risk-trading-portfolio-department-real-implementations
Tests AC #3: Portfolio Department report API
"""
import pytest
from unittest.mock import MagicMock, patch

from src.agents.departments.heads.portfolio_head import (
    PortfolioHead,
    StrategyAttribution,
    BrokerAttribution,
    AccountDrawdown,
    PortfolioReport,
)


class TestPortfolioHead:
    """Test suite for PortfolioHead."""

    @pytest.fixture
    def portfolio_head(self):
        """Create a PortfolioHead instance."""
        with patch("src.agents.departments.heads.portfolio_head.get_department_config"), \
             patch("src.agents.departments.heads.base.DepartmentHead._init_spawner"):
            return PortfolioHead()

    def test_initialization(self, portfolio_head):
        """Test PortfolioHead initializes correctly."""
        assert portfolio_head is not None

    def test_get_tools(self, portfolio_head):
        """Test tool definitions include portfolio report."""
        tools = portfolio_head.get_tools()
        tool_names = [t["name"] for t in tools]

        assert "generate_portfolio_report" in tool_names
        assert "get_total_equity" in tool_names
        assert "get_strategy_pnl" in tool_names
        assert "get_broker_pnl" in tool_names
        assert "get_account_drawdowns" in tool_names

    def test_generate_portfolio_report_returns_structure(self, portfolio_head):
        """Test report returns expected structure (AC #3)."""
        result = portfolio_head.generate_portfolio_report()

        # AC #3: Returns total equity, P&L attribution per strategy, per broker, drawdown per account
        assert "total_equity" in result
        assert "pnl_attribution" in result
        assert "drawdown_by_account" in result
        assert "generated_at" in result

    def test_portfolio_report_total_equity(self, portfolio_head):
        """Test total equity is calculated."""
        result = portfolio_head.generate_portfolio_report()

        assert isinstance(result["total_equity"], (int, float))
        assert result["total_equity"] > 0

    def test_portfolio_report_pnl_attribution(self, portfolio_head):
        """Test P&L attribution is included."""
        result = portfolio_head.generate_portfolio_report()

        pnl = result["pnl_attribution"]
        assert "by_strategy" in pnl
        assert "by_broker" in pnl

        # Check by_strategy has entries
        assert len(pnl["by_strategy"]) > 0

        # Check by_broker has entries
        assert len(pnl["by_broker"]) > 0

    def test_portfolio_report_drawdown_by_account(self, portfolio_head):
        """Test drawdown by account is included."""
        result = portfolio_head.generate_portfolio_report()

        assert len(result["drawdown_by_account"]) > 0
        dd = result["drawdown_by_account"][0]
        assert "account_id" in dd
        assert "drawdown_pct" in dd

    def test_get_total_equity(self, portfolio_head):
        """Test getting total equity."""
        result = portfolio_head.get_total_equity()

        assert "total_equity" in result
        assert "accounts" in result
        assert "account_count" in result
        assert result["account_count"] > 0

    def test_get_strategy_pnl(self, portfolio_head):
        """Test getting strategy P&L attribution."""
        result = portfolio_head.get_strategy_pnl()

        assert "period" in result
        assert "total_pnl" in result
        assert "by_strategy" in result
        assert len(result["by_strategy"]) > 0

    def test_strategy_pnl_includes_percentage(self, portfolio_head):
        """Test strategy P&L includes percentage attribution."""
        result = portfolio_head.get_strategy_pnl()

        strategy = result["by_strategy"][0]
        assert "pnl" in strategy
        assert "percentage" in strategy
        assert isinstance(strategy["percentage"], float)

    def test_get_broker_pnl(self, portfolio_head):
        """Test getting broker P&L attribution."""
        result = portfolio_head.get_broker_pnl()

        assert "period" in result
        assert "total_pnl" in result
        assert "by_broker" in result
        assert len(result["by_broker"]) > 0

    def test_broker_pnl_includes_percentage(self, portfolio_head):
        """Test broker P&L includes percentage attribution."""
        result = portfolio_head.get_broker_pnl()

        broker = result["by_broker"][0]
        assert "pnl" in broker
        assert "percentage" in broker

    def test_get_account_drawdowns(self, portfolio_head):
        """Test getting account drawdowns."""
        result = portfolio_head.get_account_drawdowns()

        assert "by_account" in result
        assert len(result["by_account"]) > 0

        account = result["by_account"][0]
        assert "account_id" in account
        assert "drawdown_pct" in account

    def test_optimize_allocation(self, portfolio_head):
        """Test portfolio optimization."""
        result = portfolio_head.optimize_allocation(
            assets=["EURUSD", "GBPUSD", "USDJPY"],
            target_return=0.10,
            max_risk=0.20,
        )

        assert "allocation" in result
        assert len(result["allocation"]) == 3

    def test_rebalance_portfolio(self, portfolio_head):
        """Test portfolio rebalancing."""
        target = {"EURUSD": 0.5, "GBPUSD": 0.3, "USDJPY": 0.2}
        result = portfolio_head.rebalance_portfolio(target_allocation=target)

        assert "rebalance_needed" in result
        assert "actions" in result

    def test_track_performance(self, portfolio_head):
        """Test performance tracking."""
        result = portfolio_head.track_performance(period="month", benchmark="SPY")

        assert "period" in result
        assert "benchmark" in result
        assert "portfolio_return" in result
        assert "alpha" in result
        assert "sharpe" in result


class TestStrategyAttribution:
    """Test StrategyAttribution dataclass."""

    def test_dataclass_creation(self):
        """Test creating StrategyAttribution."""
        attr = StrategyAttribution(
            strategy="TrendFollower",
            pnl=1000.0,
            percentage=45.5,
        )

        assert attr.strategy == "TrendFollower"
        assert attr.pnl == 1000.0
        assert attr.percentage == 45.5


class TestBrokerAttribution:
    """Test BrokerAttribution dataclass."""

    def test_dataclass_creation(self):
        """Test creating BrokerAttribution."""
        attr = BrokerAttribution(
            broker="ICMarkets",
            pnl=2500.0,
            percentage=65.0,
        )

        assert attr.broker == "ICMarkets"
        assert attr.pnl == 2500.0
        assert attr.percentage == 65.0


class TestAccountDrawdown:
    """Test AccountDrawdown dataclass."""

    def test_dataclass_creation(self):
        """Test creating AccountDrawdown."""
        dd = AccountDrawdown(
            account_id="acc_001",
            drawdown_pct=8.5,
        )

        assert dd.account_id == "acc_001"
        assert dd.drawdown_pct == 8.5

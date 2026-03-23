"""
Tests for Trading Department Head - Paper Trading Monitoring

Story: 7-8-risk-trading-portfolio-department-real-implementations
Tests AC #2: Trading Department paper trade monitoring
"""
import pytest
from unittest.mock import MagicMock, patch

from src.agents.departments.heads.execution_head import (
    TradingHead,
    TradeSide,
    TradeStatus,
    RegimeType,
    PaperTradingMetrics,
)


class TestTradingHead:
    """Test suite for TradingHead."""

    @pytest.fixture
    def trading_head(self):
        """Create a TradingHead instance."""
        with patch("src.agents.departments.heads.execution_head.get_department_config"), \
             patch("src.agents.departments.heads.base.DepartmentHead._init_spawner"):
            return TradingHead()

    def test_initialization(self, trading_head):
        """Test TradingHead initializes correctly."""
        assert trading_head._update_interval == 60
        assert trading_head.COPILOT_UPDATES_TOPIC == "copilot:updates"

    def test_get_tools(self, trading_head):
        """Test tool definitions include paper trading monitoring."""
        tools = trading_head.get_tools()
        tool_names = [t["name"] for t in tools]

        assert "monitor_paper_trading" in tool_names
        assert "get_paper_trading_status" in tool_names
        assert "get_trade_history" in tool_names
        assert "stop_monitoring" in tool_names
        assert "route_order" in tool_names

    def test_monitor_paper_trading_starts_monitoring(self, trading_head):
        """Test starting paper trading monitoring."""
        result = trading_head.monitor_paper_trading(
            agent_id="paper_001",
            strategy_name="TrendFollower",
        )

        assert result["status"] == "monitoring_started"
        assert result["agent_id"] == "paper_001"
        assert result["strategy_name"] == "TrendFollower"
        assert result["update_interval_seconds"] == 60

    def test_monitor_paper_trading_already_monitoring(self, trading_head):
        """Test starting monitoring when already monitoring."""
        trading_head.monitor_paper_trading(
            agent_id="paper_001",
            strategy_name="TrendFollower",
        )

        result = trading_head.monitor_paper_trading(
            agent_id="paper_001",
            strategy_name="TrendFollower",
        )

        assert result["status"] == "already_monitoring"

    def test_get_paper_trading_status(self, trading_head):
        """Test getting paper trading status."""
        result = trading_head.get_paper_trading_status("paper_001")

        assert "agent_id" in result
        assert result["agent_id"] == "paper_001"
        assert "status" in result
        assert "metrics" in result
        assert "regime" in result

    def test_paper_trading_status_metrics(self, trading_head):
        """Test paper trading metrics are included."""
        result = trading_head.get_paper_trading_status("paper_001")

        m = result["metrics"]
        assert "total_trades" in m
        assert "winning_trades" in m
        assert "losing_trades" in m
        assert "win_rate" in m
        assert "total_pnl" in m
        assert "max_drawdown" in m
        assert "regime_correlation" in m

    def test_paper_trading_status_regime(self, trading_head):
        """Test regime is included in status."""
        result = trading_head.get_paper_trading_status("paper_001")

        assert result["regime"] in [r.value for r in RegimeType]

    def test_get_trade_history(self, trading_head):
        """Test getting trade history."""
        result = trading_head.get_trade_history("paper_001", limit=10)

        assert "agent_id" in result
        assert result["agent_id"] == "paper_001"
        assert "total_trades" in result
        assert "trades" in result
        assert len(result["trades"]) <= 10

    def test_trade_history_includes_pnl(self, trading_head):
        """Test trade history includes P&L tracking."""
        result = trading_head.get_trade_history("paper_001")

        if result["trades"]:
            trade = result["trades"][0]
            assert "pnl_value" in trade
            assert "pnl_pct" in trade

    def test_stop_monitoring_not_monitoring(self, trading_head):
        """Test stopping when not monitoring."""
        result = trading_head.stop_monitoring("paper_001")

        assert result["status"] == "not_monitoring"

    def test_route_order(self, trading_head):
        """Test order routing."""
        result = trading_head.route_order(
            symbol="EURUSD",
            side="BUY",
            quantity=10000,
            order_type="MARKET",
        )

        assert "order_id" in result
        assert result["symbol"] == "EURUSD"
        assert result["side"] == "BUY"
        assert result["quantity"] == 10000
        assert result["order_type"] == "MARKET"
        assert result["status"] == "routed"

    def test_track_fill(self, trading_head):
        """Test fill tracking."""
        result = trading_head.track_fill("ord_001")

        assert result["order_id"] == "ord_001"
        assert "status" in result
        assert "fill_price" in result

    def test_monitor_slippage(self, trading_head):
        """Test slippage monitoring."""
        result = trading_head.monitor_slippage("EURUSD", period=60)

        assert result["symbol"] == "EURUSD"
        assert result["period_minutes"] == 60
        assert "avg_slippage_bps" in result
        assert "max_slippage_bps" in result


class TestPaperTradingMetrics:
    """Test PaperTradingMetrics dataclass."""

    def test_dataclass_creation(self):
        """Test creating PaperTradingMetrics."""
        metrics = PaperTradingMetrics(
            total_trades=100,
            winning_trades=60,
            losing_trades=40,
            win_rate=60.0,
            total_pnl=5.5,
            avg_pnl=0.055,
            max_drawdown=8.2,
            avg_hold_time_minutes=45.0,
            regime_correlation={"TREND": 0.75, "RANGE": 0.25},
        )

        assert metrics.total_trades == 100
        assert metrics.winning_trades == 60
        assert metrics.losing_trades == 40
        assert metrics.win_rate == 60.0
        assert metrics.total_pnl == 5.5
        assert metrics.max_drawdown == 8.2

    def test_default_values(self):
        """Test default values."""
        metrics = PaperTradingMetrics()

        assert metrics.total_trades == 0
        assert metrics.winning_trades == 0
        assert metrics.losing_trades == 0
        assert metrics.win_rate == 0.0


class TestRegimeType:
    """Test RegimeType enum."""

    def test_regime_types(self):
        """Test all regime types are defined."""
        assert RegimeType.TREND.value == "TREND"
        assert RegimeType.RANGE.value == "RANGE"
        assert RegimeType.BREAKOUT.value == "BREAKOUT"
        assert RegimeType.CHAOS.value == "CHAOS"

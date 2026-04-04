"""Unit tests for EA Management MCP Tools."""

import pytest
from unittest.mock import patch, MagicMock

from fastmcp import Client

# Import the main module to access mcp
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp_mt5.main import mcp


@pytest.mark.unit
class TestListInstalledEAs:
    """Test list_installed_eas tool."""

    @patch("mcp_mt5.ea_manager.get_ea_manager")
    def test_list_installed_eas_success(self, mock_get_manager):
        """Test successful listing of installed EAs."""
        mock_manager = MagicMock()
        mock_ea1 = MagicMock()
        mock_ea1.model_dump.return_value = {
            "name": "MyScalper",
            "path": "MyScalper.ex5",
            "file_size": 102400,
            "modified_time": "2026-01-29T10:00:00Z",
            "is_compiled": True
        }
        mock_ea2 = MagicMock()
        mock_ea2.model_dump.return_value = {
            "name": "TrendFollower",
            "path": "Strategies/TrendFollower.ex5",
            "file_size": 204800,
            "modified_time": "2026-01-28T15:30:00Z",
            "is_compiled": True
        }
        mock_manager.list_installed_eas.return_value = [mock_ea1, mock_ea2]
        mock_get_manager.return_value = mock_manager

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool("list_installed_eas", {})
                return result

        result = run_test()
        assert len(result.data) == 2
        assert result.data[0]["name"] == "MyScalper"
        mock_manager.list_installed_eas.assert_called_once()

    @patch("mcp_mt5.ea_manager.get_ea_manager")
    def test_list_installed_eas_empty(self, mock_get_manager):
        """Test list_installed_eas when no EAs are installed."""
        mock_manager = MagicMock()
        mock_manager.list_installed_eas.return_value = []
        mock_get_manager.return_value = mock_manager

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool("list_installed_eas", {})
                return result

        result = run_test()
        assert result.data == []


@pytest.mark.unit
class TestGetEAInfo:
    """Test get_ea_info tool."""

    @patch("mcp_mt5.ea_manager.get_ea_manager")
    def test_get_ea_info_success(self, mock_get_manager):
        """Test getting EA info for existing EA."""
        mock_manager = MagicMock()
        mock_ea = MagicMock()
        mock_ea.model_dump.return_value = {
            "name": "MyScalper",
            "path": "MyScalper.ex5",
            "file_size": 102400,
            "modified_time": "2026-01-29T10:00:00Z",
            "is_compiled": True
        }
        mock_manager.get_ea_info.return_value = mock_ea
        mock_get_manager.return_value = mock_manager

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool(
                    "get_ea_info",
                    {"ea_name": "MyScalper"}
                )
                return result

        result = run_test()
        assert result.data["name"] == "MyScalper"
        assert result.data["is_compiled"] is True
        mock_manager.get_ea_info.assert_called_once_with("MyScalper")

    @patch("mcp_mt5.ea_manager.get_ea_manager")
    def test_get_ea_info_not_found(self, mock_get_manager):
        """Test get_ea_info returns None for non-existent EA."""
        mock_manager = MagicMock()
        mock_manager.get_ea_info.return_value = None
        mock_get_manager.return_value = mock_manager

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool(
                    "get_ea_info",
                    {"ea_name": "NonExistentEA"}
                )
                return result

        result = run_test()
        assert result.data is None


@pytest.mark.unit
class TestGetEAStatus:
    """Test get_ea_status tool."""

    @patch("mcp_mt5.ea_manager.get_ea_manager")
    def test_get_ea_status_active(self, mock_get_manager):
        """Test getting status of an active EA."""
        mock_manager = MagicMock()
        mock_status = MagicMock()
        mock_status.model_dump.return_value = {
            "magic_number": 123456,
            "is_active": True,
            "last_trade_time": "2026-01-29T14:30:00Z",
            "total_trades": 45,
            "open_positions": 2,
            "symbols_traded": ["EURUSD", "GBPUSD"]
        }
        mock_manager.get_ea_status.return_value = mock_status
        mock_get_manager.return_value = mock_manager

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool(
                    "get_ea_status",
                    {"magic_number": 123456, "days": 30}
                )
                return result

        result = run_test()
        assert result.data["magic_number"] == 123456
        assert result.data["is_active"] is True
        assert result.data["total_trades"] == 45
        mock_manager.get_ea_status.assert_called_once_with(123456, 30)

    @patch("mcp_mt5.ea_manager.get_ea_manager")
    def test_get_ea_status_inactive(self, mock_get_manager):
        """Test getting status of an inactive EA."""
        mock_manager = MagicMock()
        mock_status = MagicMock()
        mock_status.model_dump.return_value = {
            "magic_number": 999999,
            "is_active": False,
            "last_trade_time": None,
            "total_trades": 0,
            "open_positions": 0,
            "symbols_traded": []
        }
        mock_manager.get_ea_status.return_value = mock_status
        mock_get_manager.return_value = mock_manager

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool(
                    "get_ea_status",
                    {"magic_number": 999999}
                )
                return result

        result = run_test()
        assert result.data["is_active"] is False
        assert result.data["total_trades"] == 0

    @patch("mcp_mt5.ea_manager.get_ea_manager")
    def test_get_ea_status_default_days(self, mock_get_manager):
        """Test get_ea_status uses default 30 days."""
        mock_manager = MagicMock()
        mock_status = MagicMock()
        mock_status.model_dump.return_value = {
            "magic_number": 123456,
            "is_active": False,
            "last_trade_time": None,
            "total_trades": 0,
            "open_positions": 0,
            "symbols_traded": []
        }
        mock_manager.get_ea_status.return_value = mock_status
        mock_get_manager.return_value = mock_manager

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool(
                    "get_ea_status",
                    {"magic_number": 123456}
                )
                return result

        result = run_test()
        mock_manager.get_ea_status.assert_called_once_with(123456, 30)


@pytest.mark.unit
class TestGetEAPerformance:
    """Test get_ea_performance tool."""

    @patch("mcp_mt5.ea_manager.get_ea_manager")
    def test_get_ea_performance_success(self, mock_get_manager):
        """Test getting EA performance metrics."""
        mock_manager = MagicMock()
        mock_perf = MagicMock()
        mock_perf.model_dump.return_value = {
            "magic_number": 123456,
            "period_days": 30,
            "total_trades": 50,
            "winning_trades": 33,
            "losing_trades": 17,
            "win_rate": 0.66,
            "total_profit": 1500.0,
            "total_loss": -500.0,
            "net_profit": 1000.0,
            "profit_factor": 3.0,
            "average_win": 45.45,
            "average_loss": -29.41,
            "largest_win": 200.0,
            "largest_loss": -100.0,
            "max_consecutive_wins": 8,
            "max_consecutive_losses": 3
        }
        mock_manager.get_ea_performance.return_value = mock_perf
        mock_get_manager.return_value = mock_manager

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool(
                    "get_ea_performance",
                    {"magic_number": 123456, "days": 30}
                )
                return result

        result = run_test()
        assert result.data["total_trades"] == 50
        assert result.data["win_rate"] == 0.66
        assert result.data["profit_factor"] == 3.0
        mock_manager.get_ea_performance.assert_called_once_with(123456, 30)

    @patch("mcp_mt5.ea_manager.get_ea_manager")
    def test_get_ea_performance_no_trades(self, mock_get_manager):
        """Test getting EA performance when no trades exist."""
        mock_manager = MagicMock()
        mock_perf = MagicMock()
        mock_perf.model_dump.return_value = {
            "magic_number": 123456,
            "total_trades": 0,
            "win_rate": 0.0,
            "net_profit": 0.0
        }
        mock_manager.get_ea_performance.return_value = mock_perf
        mock_get_manager.return_value = mock_manager

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool(
                    "get_ea_performance",
                    {"magic_number": 123456}
                )
                return result

        result = run_test()
        assert result.data["total_trades"] == 0


@pytest.mark.unit
class TestCreateEATemplate:
    """Test create_ea_template tool."""

    @patch("mcp_mt5.ea_manager.get_ea_manager")
    def test_create_ea_template_success(self, mock_get_manager):
        """Test successful EA template creation."""
        mock_manager = MagicMock()
        mock_manager.create_ea_template.return_value = "/path/to/templates/QuantMindX_MyScalper_EURUSD_123456.tpl"
        mock_get_manager.return_value = mock_manager

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool(
                    "create_ea_template",
                    {
                        "ea_name": "MyScalper",
                        "symbol": "EURUSD",
                        "timeframe": 60,
                        "magic_number": 123456,
                        "inputs": {"TakeProfit": 50, "StopLoss": 30}
                    }
                )
                return result

        result = run_test()
        assert result.data["success"] is True
        assert "template_path" in result.data
        assert "instructions" in result.data
        mock_manager.create_ea_template.assert_called_once()

    @patch("mcp_mt5.ea_manager.get_ea_manager")
    def test_create_ea_template_without_inputs(self, mock_get_manager):
        """Test create_ea_template without optional inputs."""
        mock_manager = MagicMock()
        mock_manager.create_ea_template.return_value = "/path/to/template.tpl"
        mock_get_manager.return_value = mock_manager

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool(
                    "create_ea_template",
                    {
                        "ea_name": "TrendFollower",
                        "symbol": "GBPUSD",
                        "timeframe": 240,
                        "magic_number": 654321
                    }
                )
                return result

        result = run_test()
        assert result.data["success"] is True

    @patch("mcp_mt5.ea_manager.get_ea_manager")
    def test_create_ea_template_failure(self, mock_get_manager):
        """Test create_ea_template handles errors."""
        mock_manager = MagicMock()
        mock_manager.create_ea_template.side_effect = Exception("Failed to write template")
        mock_get_manager.return_value = mock_manager

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool(
                    "create_ea_template",
                    {
                        "ea_name": "BadEA",
                        "symbol": "EURUSD",
                        "timeframe": 60,
                        "magic_number": 111111
                    }
                )
                return result

        result = run_test()
        assert result.data["success"] is False
        assert "Failed to write template" in result.data["error"]


@pytest.mark.unit
class TestStopEAByMagic:
    """Test stop_ea_by_magic tool."""

    @patch("mcp_mt5.ea_manager.get_ea_manager")
    def test_stop_ea_by_magic_success(self, mock_get_manager):
        """Test successful EA emergency stop."""
        mock_manager = MagicMock()
        mock_manager.stop_ea_by_magic.return_value = {
            "magic_number": 123456,
            "positions_closed": 3,
            "orders_cancelled": 1,
            "errors": []
        }
        mock_get_manager.return_value = mock_manager

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool(
                    "stop_ea_by_magic",
                    {"magic_number": 123456}
                )
                return result

        result = run_test()
        assert result.data["magic_number"] == 123456
        assert result.data["positions_closed"] == 3
        assert result.data["orders_cancelled"] == 1
        mock_manager.stop_ea_by_magic.assert_called_once_with(123456)

    @patch("mcp_mt5.ea_manager.get_ea_manager")
    def test_stop_ea_by_magic_no_open_positions(self, mock_get_manager):
        """Test stop_ea_by_magic when EA has no open positions."""
        mock_manager = MagicMock()
        mock_manager.stop_ea_by_magic.return_value = {
            "magic_number": 999999,
            "positions_closed": 0,
            "orders_cancelled": 0,
            "errors": []
        }
        mock_get_manager.return_value = mock_manager

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool(
                    "stop_ea_by_magic",
                    {"magic_number": 999999}
                )
                return result

        result = run_test()
        assert result.data["positions_closed"] == 0
        assert result.data["orders_cancelled"] == 0

    @patch("mcp_mt5.ea_manager.get_ea_manager")
    def test_stop_ea_by_magic_with_errors(self, mock_get_manager):
        """Test stop_ea_by_magic handles partial failures."""
        mock_manager = MagicMock()
        mock_manager.stop_ea_by_magic.return_value = {
            "magic_number": 123456,
            "positions_closed": 2,
            "orders_cancelled": 0,
            "errors": ["Failed to close position 789: market closed"]
        }
        mock_get_manager.return_value = mock_manager

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool(
                    "stop_ea_by_magic",
                    {"magic_number": 123456}
                )
                return result

        result = run_test()
        assert len(result.data["errors"]) == 1


@pytest.mark.unit
class TestGetDailyPnL:
    """Test get_daily_pnl tool."""

    @patch("mcp_mt5.ea_tools.mt5")
    def test_get_daily_pnl_success(self, mock_mt5):
        """Test successful daily PnL retrieval."""
        # Mock deals
        mock_deal1 = MagicMock()
        mock_deal1.time = 1706524800  # 2026-01-29
        mock_deal1.profit = 100.0
        mock_deal1.commission = -2.0
        mock_deal1.swap = 0.0

        mock_deal2 = MagicMock()
        mock_deal2.time = 1706524800  # Same day
        mock_deal2.profit = -50.0
        mock_deal2.commission = -1.0
        mock_deal2.swap = -5.0

        mock_deal3 = MagicMock()
        mock_deal3.time = 1706438400  # 2026-01-28
        mock_deal3.profit = 75.0
        mock_deal3.commission = -1.5
        mock_deal3.swap = 0.0

        mock_mt5.history_deals_get.return_value = [mock_deal1, mock_deal2, mock_deal3]

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool("get_daily_pnl", {"days": 7})
                return result

        result = run_test()
        assert len(result.data) == 2  # Two different dates

    @patch("mcp_mt5.ea_tools.mt5")
    def test_get_daily_pnl_no_deals(self, mock_mt5):
        """Test get_daily_pnl when no deals exist."""
        mock_mt5.history_deals_get.return_value = []

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool("get_daily_pnl", {"days": 7})
                return result

        result = run_test()
        assert result.data == []

    @patch("mcp_mt5.ea_tools.mt5")
    def test_get_daily_pnl_default_days(self, mock_mt5):
        """Test get_daily_pnl uses default 7 days."""
        mock_mt5.history_deals_get.return_value = []

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool("get_daily_pnl", {})
                return result

        result = run_test()
        mock_mt5.history_deals_get.assert_called_once()


@pytest.mark.unit
class TestEAEnums:
    """Test EA management enums."""

    def test_ea_status_values(self):
        """Test EAStatus enum has expected values."""
        from mcp_mt5.ea_manager import EAStatus

        assert EAStatus.RUNNING.value == "running"
        assert EAStatus.STOPPED.value == "stopped"
        assert EAStatus.ERROR.value == "error"

    def test_ea_info_model(self):
        """Test EAInfo model can be instantiated."""
        from mcp_mt5.ea_manager import EAInfo

        ea_info = EAInfo(
            name="TestEA",
            path="TestEA.ex5",
            file_size=102400,
            modified_time="2026-01-29T10:00:00Z",
            is_compiled=True
        )
        assert ea_info.name == "TestEA"
        assert ea_info.is_compiled is True

    def test_ea_performance_model(self):
        """Test EAPerformance model can be instantiated."""
        from mcp_mt5.ea_manager import EAPerformance

        perf = EAPerformance(
            magic_number=123456,
            period_days=30,
            total_trades=10,
            winning_trades=7,
            losing_trades=3,
            win_rate=0.7,
            total_profit=500.0,
            total_loss=150.0,
            net_profit=350.0,
            profit_factor=3.33
        )
        assert perf.magic_number == 123456
        assert perf.win_rate == 0.7

"""Unit tests for Trade Journal MCP Tools."""

import pytest
from unittest.mock import patch, MagicMock

from fastmcp import Client

# Import the main module to access mcp
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp_mt5.main import mcp


@pytest.mark.unit
class TestSyncJournalWithMT5:
    """Test sync_journal_with_mt5 tool."""

    @patch("mcp_mt5.journal.get_trade_journal")
    def test_sync_journal_with_mt5_success(self, mock_get_journal):
        """Test successful journal sync from MT5."""
        mock_journal = MagicMock()
        mock_journal.sync_from_mt5.return_value = {"synced": 5, "errors": []}
        mock_get_journal.return_value = mock_journal

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool("sync_journal_with_mt5", {"days": 30})
                return result

        result = run_test()
        assert result.data["synced"] == 5
        mock_journal.sync_from_mt5.assert_called_once_with(days=30)

    @patch("mcp_mt5.journal.get_trade_journal")
    def test_sync_journal_with_default_days(self, mock_get_journal):
        """Test sync_journal_with_mt5 with default 30 days."""
        mock_journal = MagicMock()
        mock_journal.sync_from_mt5.return_value = {"synced": 10, "errors": []}
        mock_get_journal.return_value = mock_journal

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool("sync_journal_with_mt5", {})
                return result

        result = run_test()
        mock_journal.sync_from_mt5.assert_called_once_with(days=30)

    @patch("mcp_mt5.journal.get_trade_journal")
    def test_sync_journal_with_errors(self, mock_get_journal):
        """Test sync_journal_with_mt5 handles sync errors."""
        mock_journal = MagicMock()
        mock_journal.sync_from_mt5.return_value = {
            "synced": 3,
            "errors": ["Failed to fetch deal 12345", "Invalid timestamp"]
        }
        mock_get_journal.return_value = mock_journal

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool("sync_journal_with_mt5", {"days": 7})
                return result

        result = run_test()
        assert result.data["synced"] == 3
        assert len(result.data["errors"]) == 2

    @patch("mcp_mt5.journal.get_trade_journal")
    def test_sync_journal_failure(self, mock_get_journal):
        """Test sync_journal_with_mt5 handles exceptions."""
        mock_journal = MagicMock()
        mock_journal.sync_from_mt5.side_effect = Exception("MT5 not connected")
        mock_get_journal.return_value = mock_journal

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool("sync_journal_with_mt5", {"days": 30})
                return result

        result = run_test()
        assert result.data["synced"] == 0
        assert "MT5 not connected" in result.data["errors"][0]


@pytest.mark.unit
class TestAnnotateTrade:
    """Test annotate_trade tool."""

    @patch("mcp_mt5.journal.get_trade_journal")
    def test_annotate_trade_success(self, mock_get_journal):
        """Test successful trade annotation."""
        mock_journal = MagicMock()
        mock_journal.annotate_trade.return_value = True
        mock_get_journal.return_value = mock_journal

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool(
                    "annotate_trade",
                    {
                        "ticket": 12345,
                        "notes": "Good entry on breakout",
                        "setup_type": "breakout",
                        "rating": 4,
                        "tags": ["news", "impulsive"],
                        "lessons": "Wait for retest next time"
                    }
                )
                return result

        result = run_test()
        assert result.data["success"] is True
        assert result.data["ticket"] == 12345
        mock_journal.annotate_trade.assert_called_once()

    @patch("mcp_mt5.journal.get_trade_journal")
    def test_annotate_trade_with_minimal_fields(self, mock_get_journal):
        """Test annotate_trade with only required fields."""
        mock_journal = MagicMock()
        mock_journal.annotate_trade.return_value = True
        mock_get_journal.return_value = mock_journal

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool(
                    "annotate_trade",
                    {"ticket": 67890, "notes": "Quick scalp"}
                )
                return result

        result = run_test()
        assert result.data["success"] is True

    @patch("mcp_mt5.journal.get_trade_journal")
    def test_annotate_trade_failure(self, mock_get_journal):
        """Test annotate_trade handles errors."""
        mock_journal = MagicMock()
        mock_journal.annotate_trade.side_effect = Exception("Trade not found")
        mock_get_journal.return_value = mock_journal

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool(
                    "annotate_trade",
                    {"ticket": 99999, "notes": "Test"}
                )
                return result

        result = run_test()
        assert result.data["success"] is False
        assert "Trade not found" in result.data["error"]

    @patch("mcp_mt5.journal.get_trade_journal")
    def test_annotate_trade_rating_validation(self, mock_get_journal):
        """Test annotate_trade accepts valid ratings."""
        mock_journal = MagicMock()
        mock_journal.annotate_trade.return_value = True
        mock_get_journal.return_value = mock_journal

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool(
                    "annotate_trade",
                    {"ticket": 12345, "rating": 5}
                )
                return result

        result = run_test()
        assert result.data["success"] is True


@pytest.mark.unit
class TestGetJournalStats:
    """Test get_journal_stats tool."""

    @patch("mcp_mt5.journal.get_trade_journal")
    def test_get_journal_stats_success(self, mock_get_journal):
        """Test successful journal stats retrieval."""
        mock_journal = MagicMock()
        mock_journal.get_performance_stats.return_value = {
            "win_rate": 65.5,
            "profit_factor": 1.8,
            "net_profit": 1250.50,
            "total_trades": 50,
            "average_win": 45.0,
            "average_loss": -25.0,
            "best_day": {"date": "2026-01-25", "profit": 350.0},
            "worst_day": {"date": "2026-01-20", "profit": -150.0}
        }
        mock_get_journal.return_value = mock_journal

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool("get_journal_stats", {"days": 30})
                return result

        result = run_test()
        assert result.data["win_rate"] == 65.5
        assert result.data["profit_factor"] == 1.8
        assert result.data["total_trades"] == 50
        mock_journal.get_performance_stats.assert_called_once_with(days=30)

    @patch("mcp_mt5.journal.get_trade_journal")
    def test_get_journal_stats_default_days(self, mock_get_journal):
        """Test get_journal_stats with default 30 days."""
        mock_journal = MagicMock()
        mock_journal.get_performance_stats.return_value = {"total_trades": 0}
        mock_get_journal.return_value = mock_journal

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool("get_journal_stats", {})
                return result

        result = run_test()
        mock_journal.get_performance_stats.assert_called_once_with(days=30)

    @patch("mcp_mt5.journal.get_trade_journal")
    def test_get_journal_stats_empty_journal(self, mock_get_journal):
        """Test get_journal_stats when no trades exist."""
        mock_journal = MagicMock()
        mock_journal.get_performance_stats.return_value = {
            "win_rate": 0,
            "total_trades": 0
        }
        mock_get_journal.return_value = mock_journal

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool("get_journal_stats", {"days": 7})
                return result

        result = run_test()
        assert result.data["total_trades"] == 0

    @patch("mcp_mt5.journal.get_trade_journal")
    def test_get_journal_stats_error(self, mock_get_journal):
        """Test get_journal_stats handles errors."""
        mock_journal = MagicMock()
        mock_journal.get_performance_stats.side_effect = Exception("Database error")
        mock_get_journal.return_value = mock_journal

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool("get_journal_stats", {})
                return result

        result = run_test()
        assert "error" in result.data


@pytest.mark.unit
class TestSearchJournal:
    """Test search_journal tool."""

    @patch("mcp_mt5.journal.get_trade_journal")
    def test_search_journal_by_symbol(self, mock_get_journal):
        """Test search_journal filters by symbol."""
        mock_journal = MagicMock()
        mock_journal.get_trades.return_value = [
            {"ticket": 123, "symbol": "EURUSD", "profit": 50.0},
            {"ticket": 456, "symbol": "EURUSD", "profit": -20.0}
        ]
        mock_get_journal.return_value = mock_journal

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool(
                    "search_journal",
                    {"symbol": "EURUSD"}
                )
                return result

        result = run_test()
        assert len(result.data) == 2
        mock_journal.get_trades.assert_called_once()

    @patch("mcp_mt5.journal.get_trade_journal")
    def test_search_journal_by_setup_type(self, mock_get_journal):
        """Test search_journal filters by setup type."""
        mock_journal = MagicMock()
        mock_journal.get_trades.return_value = [
            {"ticket": 123, "setup_type": "breakout"}
        ]
        mock_get_journal.return_value = mock_journal

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool(
                    "search_journal",
                    {"setup_type": "breakout"}
                )
                return result

        result = run_test()
        assert len(result.data) == 1
        assert result.data[0]["setup_type"] == "breakout"

    @patch("mcp_mt5.journal.get_trade_journal")
    def test_search_journal_by_status(self, mock_get_journal):
        """Test search_journal filters by status."""
        mock_journal = MagicMock()
        mock_journal.get_trades.return_value = [
            {"ticket": 123, "status": "closed"}
        ]
        mock_get_journal.return_value = mock_journal

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool(
                    "search_journal",
                    {"status": "closed"}
                )
                return result

        result = run_test()
        mock_journal.get_trades.assert_called_once_with(
            symbol=None, setup_type=None, status="closed", limit=50
        )

    @patch("mcp_mt5.journal.get_trade_journal")
    def test_search_journal_with_limit(self, mock_get_journal):
        """Test search_journal respects limit parameter."""
        mock_journal = MagicMock()
        mock_journal.get_trades.return_value = [
            {"ticket": i} for i in range(10)
        ]
        mock_get_journal.return_value = mock_journal

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool(
                    "search_journal",
                    {"limit": 10}
                )
                return result

        result = run_test()
        assert len(result.data) == 10

    @patch("mcp_mt5.journal.get_trade_journal")
    def test_search_journal_empty_results(self, mock_get_journal):
        """Test search_journal returns empty list when no matches."""
        mock_journal = MagicMock()
        mock_journal.get_trades.return_value = []
        mock_get_journal.return_value = mock_journal

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool(
                    "search_journal",
                    {"symbol": "NONEXISTENT"}
                )
                return result

        result = run_test()
        assert result.data == []

    @patch("mcp_mt5.journal.get_trade_journal")
    def test_search_journal_error(self, mock_get_journal):
        """Test search_journal handles errors."""
        mock_journal = MagicMock()
        mock_journal.get_trades.side_effect = Exception("Database locked")
        mock_get_journal.return_value = mock_journal

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool("search_journal", {})
                return result

        result = run_test()
        assert "error" in result.data[0]


@pytest.mark.unit
class TestExportJournal:
    """Test export_journal tool."""

    @patch("mcp_mt5.journal.get_trade_journal")
    @patch("os.path.expanduser")
    @patch("builtins.open", create=True)
    def test_export_journal_csv(self, mock_open, mock_expanduser, mock_get_journal):
        """Test export_journal exports to CSV format."""
        mock_journal = MagicMock()
        mock_journal.export_csv.return_value = "/path/to/export.csv"
        mock_get_journal.return_value = mock_journal
        mock_expanduser.return_value = "/home/user"

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool(
                    "export_journal",
                    {"format": "csv", "days": 30}
                )
                return result

        result = run_test()
        assert result.data["success"] is True
        assert result.data["format"] == "csv"
        mock_journal.export_csv.assert_called_once()

    @patch("mcp_mt5.journal.get_trade_journal")
    @patch("os.path.expanduser")
    @patch("builtins.open", create=True)
    def test_export_journal_json(self, mock_open, mock_expanduser, mock_get_journal):
        """Test export_journal exports to JSON format."""
        mock_journal = MagicMock()
        mock_journal.export_json.return_value = "/path/to/export.json"
        mock_get_journal.return_value = mock_journal
        mock_expanduser.return_value = "/home/user"

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool(
                    "export_journal",
                    {"format": "json", "days": 7}
                )
                return result

        result = run_test()
        assert result.data["success"] is True
        assert result.data["format"] == "json"
        mock_journal.export_json.assert_called_once()

    @patch("mcp_mt5.journal.get_trade_journal")
    @patch("os.path.expanduser")
    @patch("builtins.open", create=True)
    def test_export_journal_with_custom_filename(self, mock_open, mock_expanduser, mock_get_journal):
        """Test export_journal uses custom filename when provided."""
        mock_journal = MagicMock()
        mock_journal.export_csv.return_value = "/path/to/my_custom_export.csv"
        mock_get_journal.return_value = mock_journal
        mock_expanduser.return_value = "/home/user"

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool(
                    "export_journal",
                    {"format": "csv", "filename": "my_custom_export.csv"}
                )
                return result

        result = run_test()
        assert result.data["success"] is True

    @patch("mcp_mt5.journal.get_trade_journal")
    @patch("os.path.expanduser")
    @patch("builtins.open", create=True)
    def test_export_journal_default_format(self, mock_open, mock_expanduser, mock_get_journal):
        """Test export_journal defaults to CSV format."""
        mock_journal = MagicMock()
        mock_journal.export_csv.return_value = "/path/to/export.csv"
        mock_get_journal.return_value = mock_journal
        mock_expanduser.return_value = "/home/user"

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool("export_journal", {})
                return result

        result = run_test()
        mock_journal.export_csv.assert_called_once()

    @patch("mcp_mt5.journal.get_trade_journal")
    def test_export_journal_failure(self, mock_get_journal):
        """Test export_journal handles errors."""
        mock_journal = MagicMock()
        mock_journal.export_csv.side_effect = Exception("Permission denied")
        mock_get_journal.return_value = mock_journal

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool("export_journal", {"format": "csv"})
                return result

        result = run_test()
        assert result.data["success"] is False
        assert "Permission denied" in result.data["error"]


@pytest.mark.unit
class TestJournalEnums:
    """Test trade journal enums."""

    def test_trade_status_values(self):
        """Test TradeStatus enum has expected values."""
        from mcp_mt5.journal import TradeStatus

        assert TradeStatus.OPEN.value == "open"
        assert TradeStatus.CLOSED.value == "closed"
        assert TradeStatus.PENDING.value == "pending"

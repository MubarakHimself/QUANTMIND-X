"""Unit tests for Alert Service MCP Tools."""

import pytest
from unittest.mock import patch, MagicMock

from fastmcp import Client

# Import the main module to access mcp
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp_mt5.main import mcp


@pytest.mark.unit
class TestConfigureEmailAlerts:
    """Test configure_email_alerts tool."""

    @patch("mcp_mt5.alert_service.get_alert_service")
    def test_configure_email_alerts_success(self, mock_get_service):
        """Test successful email alert configuration."""
        mock_service = MagicMock()
        mock_get_service.return_value = mock_service

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool(
                    "configure_email_alerts",
                    {
                        "smtp_server": "smtp.gmail.com",
                        "smtp_port": 587,
                        "username": "test@gmail.com",
                        "password": "app_password",
                        "to_addresses": ["alerts@example.com"],
                    }
                )
                return result

        result = run_test()
        assert result.data["success"] is True
        assert result.data["smtp_server"] == "smtp.gmail.com"
        mock_service.configure_email.assert_called_once()

    @patch("mcp_mt5.alert_service.get_alert_service")
    def test_configure_email_alerts_with_defaults(self, mock_get_service):
        """Test configure_email_alerts with default SMTP settings."""
        mock_service = MagicMock()
        mock_get_service.return_value = mock_service

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool(
                    "configure_email_alerts",
                    {"username": "test@gmail.com", "password": "password"}
                )
                return result

        result = run_test()
        assert result.data["success"] is True

    @patch("mcp_mt5.alert_service.get_alert_service")
    def test_configure_email_alerts_failure(self, mock_get_service):
        """Test configure_email_alerts handles errors."""
        mock_service = MagicMock()
        mock_service.configure_email.side_effect = Exception("SMTP connection failed")
        mock_get_service.return_value = mock_service

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool(
                    "configure_email_alerts",
                    {"username": "test@gmail.com", "password": "password"}
                )
                return result

        result = run_test()
        assert result.data["success"] is False
        assert "SMTP connection failed" in result.data["error"]


@pytest.mark.unit
class TestTestEmailAlert:
    """Test test_email_alert tool."""

    @patch("mcp_mt5.alert_service.get_alert_service")
    def test_test_email_alert_success(self, mock_get_service):
        """Test successful test email sending."""
        mock_service = MagicMock()
        mock_service.test_email.return_value = {"success": True, "recipients": ["test@example.com"]}
        mock_get_service.return_value = mock_service

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool("test_email_alert", {})
                return result

        result = run_test()
        assert result.data["success"] is True
        mock_service.test_email.assert_called_once()

    @patch("mcp_mt5.alert_service.get_alert_service")
    def test_test_email_alert_failure(self, mock_get_service):
        """Test test_email_alert handles errors."""
        mock_service = MagicMock()
        mock_service.test_email.side_effect = Exception("Email send failed")
        mock_get_service.return_value = mock_service

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool("test_email_alert", {})
                return result

        result = run_test()
        assert result.data["success"] is False


@pytest.mark.unit
class TestSendTradingAlert:
    """Test send_trading_alert tool."""

    @patch("mcp_mt5.alert_service.get_alert_service")
    def test_send_trading_alert_info(self, mock_get_service):
        """Test sending info severity trading alert."""
        mock_service = MagicMock()
        mock_alert = MagicMock()
        mock_alert.sent_email = True
        mock_alert.timestamp = "2026-01-29T10:00:00Z"
        mock_alert.severity.value = "info"
        mock_alert.category.value = "trade"
        mock_service.send_alert_sync.return_value = mock_alert
        mock_get_service.return_value = mock_service

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool(
                    "send_trading_alert",
                    {
                        "title": "Position Opened",
                        "message": "Buy 0.1 EURUSD",
                        "severity": "info",
                        "category": "trade",
                    }
                )
                return result

        result = run_test()
        assert result.data["success"] is True
        assert result.data["sent_email"] is True
        mock_service.send_alert_sync.assert_called_once()

    @patch("mcp_mt5.alert_service.get_alert_service")
    def test_send_trading_alert_warning(self, mock_get_service):
        """Test sending warning severity trading alert."""
        mock_service = MagicMock()
        mock_alert = MagicMock()
        mock_alert.sent_email = True
        mock_alert.timestamp = "2026-01-29T10:00:00Z"
        mock_alert.severity.value = "warning"
        mock_alert.category.value = "risk"
        mock_service.send_alert_sync.return_value = mock_alert
        mock_get_service.return_value = mock_service

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool(
                    "send_trading_alert",
                    {
                        "title": "Drawdown Warning",
                        "message": "Daily drawdown at 3.5%",
                        "severity": "warning",
                        "category": "risk",
                    }
                )
                return result

        result = run_test()
        assert result.data["success"] is True
        assert result.data["severity"] == "warning"

    @patch("mcp_mt5.alert_service.get_alert_service")
    def test_send_trading_alert_with_data(self, mock_get_service):
        """Test sending alert with additional data."""
        mock_service = MagicMock()
        mock_alert = MagicMock()
        mock_alert.sent_email = True
        mock_alert.timestamp = "2026-01-29T10:00:00Z"
        mock_alert.severity.value = "info"
        mock_alert.category.value = "trade"
        mock_service.send_alert_sync.return_value = mock_alert
        mock_get_service.return_value = mock_service

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool(
                    "send_trading_alert",
                    {
                        "title": "Trade Executed",
                        "message": "EURUSD buy order filled",
                        "data": {"ticket": 12345, "volume": 0.1, "price": 1.0850},
                    }
                )
                return result

        result = run_test()
        assert result.data["success"] is True

    @patch("mcp_mt5.alert_service.get_alert_service")
    def test_send_trading_alert_failure(self, mock_get_service):
        """Test send_trading_alert handles errors."""
        mock_service = MagicMock()
        mock_service.send_alert_sync.side_effect = Exception("Alert service unavailable")
        mock_get_service.return_value = mock_service

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool(
                    "send_trading_alert",
                    {"title": "Test", "message": "Test alert"}
                )
                return result

        result = run_test()
        assert result.data["success"] is False


@pytest.mark.unit
class TestGetAlertHistory:
    """Test get_alert_history tool."""

    @patch("mcp_mt5.alert_service.get_alert_service")
    def test_get_alert_history_default_limit(self, mock_get_service):
        """Test get_alert_history with default limit."""
        mock_service = MagicMock()
        mock_service.get_history.return_value = [
            {"title": "Alert 1", "severity": "info"},
            {"title": "Alert 2", "severity": "warning"},
        ]
        mock_get_service.return_value = mock_service

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool("get_alert_history", {})
                return result

        result = run_test()
        assert len(result.data) == 2
        mock_service.get_history.assert_called_once_with(
            limit=50, severity=None, category=None
        )

    @patch("mcp_mt5.alert_service.get_alert_service")
    def test_get_alert_history_with_filters(self, mock_get_service):
        """Test get_alert_history with severity and category filters."""
        mock_service = MagicMock()
        mock_service.get_history.return_value = [
            {"title": "Critical Alert", "severity": "critical", "category": "risk"}
        ]
        mock_get_service.return_value = mock_service

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool(
                    "get_alert_history",
                    {"limit": 10, "severity": "critical", "category": "risk"}
                )
                return result

        result = run_test()
        assert len(result.data) == 1
        mock_service.get_history.assert_called_once_with(
            limit=10,
            severity="critical",
            category="risk"
        )

    @patch("mcp_mt5.alert_service.get_alert_service")
    def test_get_alert_history_empty(self, mock_get_service):
        """Test get_alert_history when no alerts exist."""
        mock_service = MagicMock()
        mock_service.get_history.return_value = []
        mock_get_service.return_value = mock_service

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool("get_alert_history", {})
                return result

        result = run_test()
        assert result.data == []

    @patch("mcp_mt5.alert_service.get_alert_service")
    def test_get_alert_history_error(self, mock_get_service):
        """Test get_alert_history handles errors."""
        mock_service = MagicMock()
        mock_service.get_history.side_effect = Exception("Database error")
        mock_get_service.return_value = mock_service

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool("get_alert_history", {})
                return result

        result = run_test()
        assert "error" in result.data[0]


@pytest.mark.unit
class TestGetAlertConfig:
    """Test get_alert_config tool."""

    @patch("mcp_mt5.alert_service.get_alert_service")
    def test_get_alert_config_success(self, mock_get_service):
        """Test get_alert_config returns configuration."""
        mock_service = MagicMock()
        mock_config = MagicMock()
        mock_config.email.enabled = True
        mock_config.email.smtp_server = "smtp.gmail.com"
        mock_config.email.from_address = "alerts@gmail.com"
        mock_config.email.to_addresses = ["recipient@example.com"]
        mock_config.rate_limit_per_minute = 10
        mock_config.rate_limit_per_hour = 100
        mock_config.min_severity.value = "info"
        mock_config.respect_quiet_hours = True
        mock_config.quiet_hours_start = "22:00"
        mock_config.quiet_hours_end = "08:00"
        mock_service.config = mock_config
        mock_get_service.return_value = mock_service

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool("get_alert_config", {})
                return result

        result = run_test()
        assert result.data["email_enabled"] is True
        assert result.data["smtp_server"] == "smtp.gmail.com"
        assert result.data["rate_limit_per_minute"] == 10

    @patch("mcp_mt5.alert_service.get_alert_service")
    def test_get_alert_config_error(self, mock_get_service):
        """Test get_alert_config handles errors."""
        mock_service = MagicMock()
        mock_service.config = None
        mock_get_service.return_value = mock_service

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool("get_alert_config", {})
                return result

        result = run_test()
        assert "error" in result.data


@pytest.mark.unit
class TestAlertEnums:
    """Test alert severity and category enums."""

    def test_alert_severity_values(self):
        """Test AlertSeverity enum has expected values."""
        from mcp_mt5.alert_service import AlertSeverity

        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.ERROR.value == "error"
        assert AlertSeverity.CRITICAL.value == "critical"

    def test_alert_category_values(self):
        """Test AlertCategory enum has expected values."""
        from mcp_mt5.alert_service import AlertCategory

        assert AlertCategory.TRADE.value == "trade"
        assert AlertCategory.RISK.value == "risk"
        assert AlertCategory.SYSTEM.value == "system"
        assert AlertCategory.PERFORMANCE.value == "performance"
        assert AlertCategory.CONNECTION.value == "connection"

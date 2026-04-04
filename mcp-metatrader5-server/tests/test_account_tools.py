"""Unit tests for Account Management MCP Tools (Vault Operations)."""

import pytest
from unittest.mock import patch, MagicMock

from fastmcp import Client

# Import the main module to access mcp
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp_mt5.main import mcp


@pytest.mark.unit
class TestUnlockVault:
    """Test unlock_vault tool."""

    @patch("mcp_mt5.account_manager.get_account_manager")
    def test_unlock_vault_success(self, mock_get_manager):
        """Test successful vault unlock."""
        mock_manager = MagicMock()
        mock_manager.unlock.return_value = True
        mock_account1 = MagicMock()
        mock_account1.to_safe_dict.return_value = {"login": 123456, "server": "TestServer"}
        mock_account2 = MagicMock()
        mock_account2.to_safe_dict.return_value = {"login": 654321, "server": "TestServer2"}
        mock_manager.list_accounts.return_value = [mock_account1, mock_account2]
        mock_get_manager.return_value = mock_manager

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool(
                    "unlock_vault",
                    {"master_password": "correct_password"}
                )
                return result

        result = run_test()
        assert result.data["success"] is True
        assert result.data["accounts_loaded"] == 2

    @patch("mcp_mt5.account_manager.get_account_manager")
    def test_unlock_vault_invalid_password(self, mock_get_manager):
        """Test vault unlock with invalid password."""
        mock_manager = MagicMock()
        mock_manager.unlock.return_value = False
        mock_get_manager.return_value = mock_manager

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool(
                    "unlock_vault",
                    {"master_password": "wrong_password"}
                )
                return result

        result = run_test()
        assert result.data["success"] is False
        assert "Invalid master password" in result.data["error"]

    @patch("mcp_mt5.account_manager.get_account_manager")
    def test_unlock_vault_with_custom_config_path(self, mock_get_manager):
        """Test vault unlock with custom config path."""
        mock_manager = MagicMock()
        mock_manager.unlock.return_value = True
        mock_manager.list_accounts.return_value = []
        mock_get_manager.return_value = mock_manager

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool(
                    "unlock_vault",
                    {"master_password": "password", "config_path": "/custom/path.json"}
                )
                return result

        result = run_test()
        assert result.data["success"] is True
        mock_get_manager.assert_called_with("/custom/path.json")

    @patch("mcp_mt5.account_manager.get_account_manager")
    def test_unlock_vault_exception(self, mock_get_manager):
        """Test vault unlock handles exceptions."""
        mock_manager = MagicMock()
        mock_manager.unlock.side_effect = Exception("Corrupted vault file")
        mock_get_manager.return_value = mock_manager

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool(
                    "unlock_vault",
                    {"master_password": "password"}
                )
                return result

        result = run_test()
        assert result.data["success"] is False
        assert "Corrupted vault file" in result.data["error"]


@pytest.mark.unit
class TestLockVault:
    """Test lock_vault tool."""

    @patch("mcp_mt5.account_manager.get_account_manager")
    def test_lock_vault_success(self, mock_get_manager):
        """Test successful vault lock."""
        mock_manager = MagicMock()
        mock_get_manager.return_value = mock_manager

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool("lock_vault", {})
                return result

        result = run_test()
        assert result.data["success"] is True
        mock_manager.lock.assert_called_once()

    @patch("mcp_mt5.account_manager.get_account_manager")
    def test_lock_vault_exception(self, mock_get_manager):
        """Test vault lock handles errors."""
        mock_manager = MagicMock()
        mock_manager.lock.side_effect = Exception("Lock operation failed")
        mock_get_manager.return_value = mock_manager

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool("lock_vault", {})
                return result

        result = run_test()
        assert result.data["success"] is False


@pytest.mark.unit
class TestListAccounts:
    """Test list_accounts tool."""

    @patch("mcp_mt5.account_manager.get_account_manager")
    def test_list_accounts_success(self, mock_get_manager):
        """Test successful account listing."""
        mock_manager = MagicMock()
        mock_account1 = MagicMock()
        mock_account1.to_safe_dict.return_value = {
            "login": 12345678,
            "server": "Exness-MT5Real",
            "broker": "exness",
            "account_type": "live",
            "nickname": "Main Account",
            "is_active": True,
            "last_used": "2026-01-29T10:00:00Z"
        }
        mock_account2 = MagicMock()
        mock_account2.to_safe_dict.return_value = {
            "login": 87654321,
            "server": "MetaQuotes-Demo",
            "broker": "generic",
            "account_type": "demo",
            "nickname": "Demo Account",
            "is_active": True,
            "last_used": "2026-01-28T15:30:00Z"
        }
        mock_manager.list_accounts.return_value = [mock_account1, mock_account2]
        mock_get_manager.return_value = mock_manager

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool("list_accounts", {})
                return result

        result = run_test()
        assert len(result.data) == 2
        assert result.data[0]["login"] == 12345678
        assert result.data[1]["login"] == 87654321

    @patch("mcp_mt5.account_manager.get_account_manager")
    def test_list_accounts_empty(self, mock_get_manager):
        """Test list_accounts when no accounts configured."""
        mock_manager = MagicMock()
        mock_manager.list_accounts.return_value = []
        mock_get_manager.return_value = mock_manager

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool("list_accounts", {})
                return result

        result = run_test()
        assert result.data == []

    @patch("mcp_mt5.account_manager.get_account_manager")
    def test_list_accounts_vault_locked(self, mock_get_manager):
        """Test list_accounts when vault is locked."""
        mock_manager = MagicMock()
        mock_manager.list_accounts.side_effect = ValueError("Vault is locked")
        mock_get_manager.return_value = mock_manager

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool("list_accounts", {})
                return result

        result = run_test()
        assert "error" in result.data[0]


@pytest.mark.unit
class TestAddAccount:
    """Test add_account tool."""

    @patch("mcp_mt5.account_manager.get_account_manager")
    def test_add_account_success(self, mock_get_manager):
        """Test successful account addition."""
        mock_manager = MagicMock()
        mock_account = MagicMock()
        mock_account.to_safe_dict.return_value = {
            "login": 12345678,
            "server": "Exness-MT5Real",
            "broker": "exness",
            "account_type": "live",
            "nickname": "Main Exness"
        }
        mock_manager.add_account.return_value = mock_account
        mock_get_manager.return_value = mock_manager

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool(
                    "add_account",
                    {
                        "login": 12345678,
                        "password": "secure_password",
                        "server": "Exness-MT5Real",
                        "broker": "exness",
                        "account_type": "live",
                        "nickname": "Main Exness"
                    }
                )
                return result

        result = run_test()
        assert result.data["success"] is True
        assert result.data["account"]["login"] == 12345678

    @patch("mcp_mt5.account_manager.get_account_manager")
    def test_add_account_with_defaults(self, mock_get_manager):
        """Test add_account with default broker and account_type."""
        mock_manager = MagicMock()
        mock_account = MagicMock()
        mock_account.to_safe_dict.return_value = {
            "login": 11111111,
            "server": "TestServer",
            "broker": "generic",
            "account_type": "demo"
        }
        mock_manager.add_account.return_value = mock_account
        mock_get_manager.return_value = mock_manager

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool(
                    "add_account",
                    {
                        "login": 11111111,
                        "password": "password",
                        "server": "TestServer"
                    }
                )
                return result

        result = run_test()
        assert result.data["success"] is True
        mock_manager.add_account.assert_called_once()

    @patch("mcp_mt5.account_manager.get_account_manager")
    def test_add_account_failure(self, mock_get_manager):
        """Test add_account handles errors."""
        mock_manager = MagicMock()
        mock_manager.add_account.side_effect = Exception("Account already exists")
        mock_get_manager.return_value = mock_manager

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool(
                    "add_account",
                    {
                        "login": 12345678,
                        "password": "password",
                        "server": "TestServer"
                    }
                )
                return result

        result = run_test()
        assert result.data["success"] is False
        assert "Account already exists" in result.data["error"]


@pytest.mark.unit
class TestRemoveAccount:
    """Test remove_account tool."""

    @patch("mcp_mt5.account_manager.get_account_manager")
    def test_remove_account_success(self, mock_get_manager):
        """Test successful account removal."""
        mock_manager = MagicMock()
        mock_manager.remove_account.return_value = True
        mock_get_manager.return_value = mock_manager

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool(
                    "remove_account",
                    {"login": 12345678}
                )
                return result

        result = run_test()
        assert result.data["success"] is True
        assert "removed" in result.data["message"]

    @patch("mcp_mt5.account_manager.get_account_manager")
    def test_remove_account_not_found(self, mock_get_manager):
        """Test remove_account when account doesn't exist."""
        mock_manager = MagicMock()
        mock_manager.remove_account.return_value = False
        mock_get_manager.return_value = mock_manager

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool(
                    "remove_account",
                    {"login": 99999999}
                )
                return result

        result = run_test()
        assert result.data["success"] is False
        assert "not found" in result.data["message"]

    @patch("mcp_mt5.account_manager.get_account_manager")
    def test_remove_account_exception(self, mock_get_manager):
        """Test remove_account handles errors."""
        mock_manager = MagicMock()
        mock_manager.remove_account.side_effect = Exception("Vault is locked")
        mock_get_manager.return_value = mock_manager

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool(
                    "remove_account",
                    {"login": 12345678}
                )
                return result

        result = run_test()
        assert result.data["success"] is False


@pytest.mark.unit
class TestSwitchAccount:
    """Test switch_account tool."""

    @patch("mcp_mt5.account_manager.get_account_manager")
    def test_switch_account_success(self, mock_get_manager):
        """Test successful account switch."""
        mock_manager = MagicMock()
        mock_manager.switch_account.return_value = {
            "success": True,
            "login": 12345678,
            "server": "Exness-MT5Real",
            "balance": 10000.00,
            "equity": 10150.50,
            "currency": "USD"
        }
        mock_get_manager.return_value = mock_manager

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool(
                    "switch_account",
                    {"login": 12345678}
                )
                return result

        result = run_test()
        assert result.data["success"] is True
        assert result.data["login"] == 12345678
        assert result.data["balance"] == 10000.00

    @patch("mcp_mt5.account_manager.get_account_manager")
    def test_switch_account_failure(self, mock_get_manager):
        """Test switch_account handles errors."""
        mock_manager = MagicMock()
        mock_manager.switch_account.side_effect = Exception("MT5 not initialized")
        mock_get_manager.return_value = mock_manager

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool(
                    "switch_account",
                    {"login": 12345678}
                )
                return result

        result = run_test()
        assert result.data["success"] is False
        assert "MT5 not initialized" in result.data["error"]


@pytest.mark.unit
class TestGetConnectionStatus:
    """Test get_connection_status tool."""

    @patch("mcp_mt5.account_manager.get_account_manager")
    def test_get_connection_status_connected(self, mock_get_manager):
        """Test get_connection_status when connected."""
        mock_manager = MagicMock()
        mock_manager.get_connection_status.return_value = {
            "connected": True,
            "current_login": 12345678,
            "current_server": "Exness-MT5Real",
            "connected_at": "2026-01-29T10:00:00Z",
            "balance": 10000.00,
            "equity": 10150.50,
            "margin": 500.00,
            "free_margin": 9650.50,
            "profit": 150.50,
            "currency": "USD",
            "last_error": None
        }
        mock_get_manager.return_value = mock_manager

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool("get_connection_status", {})
                return result

        result = run_test()
        assert result.data["connected"] is True
        assert result.data["current_login"] == 12345678
        assert result.data["balance"] == 10000.00

    @patch("mcp_mt5.account_manager.get_account_manager")
    def test_get_connection_status_disconnected(self, mock_get_manager):
        """Test get_connection_status when disconnected."""
        mock_manager = MagicMock()
        mock_manager.get_connection_status.return_value = {
            "connected": False,
            "last_error": "MT5 terminal not running"
        }
        mock_get_manager.return_value = mock_manager

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool("get_connection_status", {})
                return result

        result = run_test()
        assert result.data["connected"] is False

    @patch("mcp_mt5.account_manager.get_account_manager")
    def test_get_connection_status_exception(self, mock_get_manager):
        """Test get_connection_status handles errors."""
        mock_manager = MagicMock()
        mock_manager.get_connection_status.side_effect = Exception("Manager not initialized")
        mock_get_manager.return_value = mock_manager

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool("get_connection_status", {})
                return result

        result = run_test()
        assert result.data["connected"] is False
        assert "Manager not initialized" in result.data["error"]


@pytest.mark.unit
class TestDisconnectAccount:
    """Test disconnect_account tool."""

    @patch("mcp_mt5.account_manager.get_account_manager")
    def test_disconnect_account_success(self, mock_get_manager):
        """Test successful account disconnect."""
        mock_manager = MagicMock()
        mock_manager.disconnect.return_value = True
        mock_get_manager.return_value = mock_manager

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool("disconnect_account", {})
                return result

        result = run_test()
        assert result.data["success"] is True
        mock_manager.disconnect.assert_called_once()

    @patch("mcp_mt5.account_manager.get_account_manager")
    def test_disconnect_account_failure(self, mock_get_manager):
        """Test disconnect_account handles errors."""
        mock_manager = MagicMock()
        mock_manager.disconnect.side_effect = Exception("Not connected")
        mock_get_manager.return_value = mock_manager

        async def run_test():
            async with Client(mcp) as client:
                result = await client.call_tool("disconnect_account", {})
                return result

        result = run_test()
        assert result.data["success"] is False
        assert "Not connected" in result.data["error"]


@pytest.mark.unit
class TestAccountManagerEnums:
    """Test account manager enums and models."""

    def test_account_type_values(self):
        """Test AccountType enum has expected values."""
        from mcp_mt5.account_manager import AccountType

        assert AccountType.LIVE.value == "live"
        assert AccountType.DEMO.value == "demo"
        assert AccountType.CONTEST.value == "contest"

    def test_broker_enum_values(self):
        """Test Broker enum has expected values."""
        from mcp_mt5.account_manager import Broker

        assert Broker.EXNESS.value == "exness"
        assert Broker.FTMO.value == "ftmo"
        assert Broker.THE5ERS.value == "the5ers"
        assert Broker.MFF.value == "mff"
        assert Broker.GENERIC.value == "generic"

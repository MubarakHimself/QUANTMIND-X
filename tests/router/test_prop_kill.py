"""
Tests for Prop Firm Emergency Kill Functionality

Tests for emergency_kill_prop_firm() and get_accounts_by_book() methods.

**Validates: Task 7 - Strategy Intelligence - Prop Firm Emergency Kill**
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

from src.router.progressive_kill_switch import (
    ProgressiveKillSwitch, ProgressiveConfig,
    get_progressive_kill_switch, reset_progressive_kill_switch
)
from src.router.alert_manager import (
    AlertManager, AlertLevel,
    get_alert_manager, reset_alert_manager
)
from src.router.bot_manifest import AccountBook


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def reset_all_singletons():
    """Reset all singleton instances before each test."""
    reset_progressive_kill_switch()
    reset_alert_manager()
    yield
    reset_progressive_kill_switch()
    reset_alert_manager()


@pytest.fixture
def progressive_kill():
    """Create a ProgressiveKillSwitch for testing."""
    config = ProgressiveConfig(enabled=True)
    alert_manager = AlertManager()

    # Mock the kill_switch
    mock_kill_switch = Mock()
    mock_kill_switch.trigger = AsyncMock(return_value=None)

    pks = ProgressiveKillSwitch(
        config=config,
        kill_switch=mock_kill_switch,
        alert_service=None
    )
    pks.alert_manager = alert_manager
    return pks


@pytest.fixture
def alert_manager():
    """Create a fresh AlertManager for testing."""
    return AlertManager()


# =============================================================================
# Account Book Registration Tests
# =============================================================================

class TestAccountBookRegistration:
    """Tests for account book registration functionality."""

    def test_register_account_book_personal(self, progressive_kill):
        """Test registering a personal account."""
        progressive_kill.register_account_book("account_001", AccountBook.PERSONAL)

        accounts = progressive_kill.get_accounts_by_book(AccountBook.PERSONAL)
        assert "account_001" in accounts
        assert len(accounts) == 1

    def test_register_account_book_prop_firm(self, progressive_kill):
        """Test registering a prop firm account."""
        progressive_kill.register_account_book("prop_account_001", AccountBook.PROP_FIRM)

        accounts = progressive_kill.get_accounts_by_book(AccountBook.PROP_FIRM)
        assert "prop_account_001" in accounts
        assert len(accounts) == 1

    def test_register_multiple_accounts_different_books(self, progressive_kill):
        """Test registering multiple accounts with different book types."""
        progressive_kill.register_account_book("personal_001", AccountBook.PERSONAL)
        progressive_kill.register_account_book("personal_002", AccountBook.PERSONAL)
        progressive_kill.register_account_book("prop_001", AccountBook.PROP_FIRM)
        progressive_kill.register_account_book("prop_002", AccountBook.PROP_FIRM)

        personal_accounts = progressive_kill.get_accounts_by_book(AccountBook.PERSONAL)
        prop_accounts = progressive_kill.get_accounts_by_book(AccountBook.PROP_FIRM)

        assert len(personal_accounts) == 2
        assert len(prop_accounts) == 2
        assert "personal_001" in personal_accounts
        assert "personal_002" in personal_accounts
        assert "prop_001" in prop_accounts
        assert "prop_002" in prop_accounts

    def test_get_accounts_by_book_empty(self, progressive_kill):
        """Test getting accounts when none registered."""
        accounts = progressive_kill.get_accounts_by_book(AccountBook.PROP_FIRM)
        assert accounts == []


# =============================================================================
# Emergency Kill Prop Firm Tests
# =============================================================================

class TestEmergencyKillPropFirm:
    """Tests for emergency_kill_prop_firm() functionality."""

    @pytest.mark.asyncio
    async def test_emergency_kill_no_accounts(self, progressive_kill):
        """Test emergency kill with no registered prop firm accounts."""
        result = await progressive_kill.emergency_kill_prop_firm(
            reason="Test emergency kill"
        )

        assert result["success"] is True
        assert result["killed_count"] == 0
        assert result["accounts"] == []
        assert "No prop firm accounts" in result["message"]

    @pytest.mark.asyncio
    async def test_emergency_kill_all_prop_firms(self, progressive_kill):
        """Test emergency kill for all prop firm accounts."""
        # Register accounts
        progressive_kill.register_account_book("prop_001", AccountBook.PROP_FIRM)
        progressive_kill.register_account_book("prop_002", AccountBook.PROP_FIRM)
        progressive_kill.register_account_book("personal_001", AccountBook.PERSONAL)

        result = await progressive_kill.emergency_kill_prop_firm(
            reason="Test emergency kill"
        )

        assert result["success"] is True
        assert result["killed_count"] == 2
        assert "prop_001" in result["accounts"]
        assert "prop_002" in result["accounts"]
        assert "personal_001" not in result["accounts"]
        assert result["prop_firm_name"] is None

        # Verify kill_switch.trigger was called for each prop firm account
        assert progressive_kill.kill_switch.trigger.call_count == 2

    @pytest.mark.asyncio
    async def test_emergency_kill_specific_firm_name(self, progressive_kill):
        """Test emergency kill with specific prop firm name."""
        progressive_kill.register_account_book("prop_001", AccountBook.PROP_FIRM)
        progressive_kill.register_account_book("prop_002", AccountBook.PROP_FIRM)

        result = await progressive_kill.emergency_kill_prop_firm(
            prop_firm_name="FirmA",
            reason="Risk limit exceeded"
        )

        assert result["success"] is True
        assert result["killed_count"] == 2
        assert result["prop_firm_name"] == "FirmA"
        assert result["reason"] == "Risk limit exceeded"

    @pytest.mark.asyncio
    async def test_emergency_kill_only_personal_accounts(self, progressive_kill):
        """Test that emergency kill doesn't affect personal accounts."""
        progressive_kill.register_account_book("personal_001", AccountBook.PERSONAL)
        progressive_kill.register_account_book("personal_002", AccountBook.PERSONAL)

        result = await progressive_kill.emergency_kill_prop_firm(
            reason="Test"
        )

        assert result["killed_count"] == 0
        assert progressive_kill.kill_switch.trigger.call_count == 0

    @pytest.mark.asyncio
    async def test_emergency_kill_raises_alert(self, progressive_kill):
        """Test that emergency kill raises appropriate alert."""
        progressive_kill.register_account_book("prop_001", AccountBook.PROP_FIRM)

        await progressive_kill.emergency_kill_prop_firm(reason="Test alert")

        # Check that an alert was raised
        alerts = progressive_kill.alert_manager.active_alerts
        assert len(alerts) == 1
        assert alerts[0].source == "emergency_kill"
        assert alerts[0].tier == 5
        assert "prop firm accounts" in alerts[0].message.lower()


# =============================================================================
# Integration Tests
# =============================================================================

class TestPropKillIntegration:
    """Integration tests for prop firm kill functionality."""

    @pytest.mark.asyncio
    async def test_full_kill_workflow(self, progressive_kill):
        """Test complete workflow: register accounts -> kill -> verify."""
        # Register accounts
        progressive_kill.register_account_book("prop_acc_1", AccountBook.PROP_FIRM)
        progressive_kill.register_account_book("prop_acc_2", AccountBook.PROP_FIRM)
        progressive_kill.register_account_book("personal_acc", AccountBook.PERSONAL)

        # Verify registration
        prop_accounts = progressive_kill.get_accounts_by_book(AccountBook.PROP_FIRM)
        personal_accounts = progressive_kill.get_accounts_by_book(AccountBook.PERSONAL)

        assert len(prop_accounts) == 2
        assert len(personal_accounts) == 1

        # Execute emergency kill
        result = await progressive_kill.emergency_kill_prop_firm(
            reason="Daily risk limit exceeded"
        )

        # Verify results
        assert result["killed_count"] == 2
        assert "prop_acc_1" in result["accounts"]
        assert "prop_acc_2" in result["accounts"]
        assert "personal_acc" not in result["accounts"]

    @pytest.mark.asyncio
    async def test_kill_switch_called_with_correct_params(self, progressive_kill):
        """Test that kill_switch.trigger is called with correct parameters."""
        progressive_kill.register_account_book("prop_001", AccountBook.PROP_FIRM)

        await progressive_kill.emergency_kill_prop_firm(
            prop_firm_name="TestFirm",
            reason="Emergency stop"
        )

        # Verify trigger was called
        progressive_kill.kill_switch.trigger.assert_called_once()
        call_args = progressive_kill.kill_switch.trigger.call_args

        # Check kwargs
        assert call_args.kwargs["triggered_by"] == "emergency_prop_firm_kill"
        assert "Emergency stop" in call_args.kwargs["message"]
        assert call_args.kwargs["account_id"] == "prop_001"


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestPropKillEdgeCases:
    """Edge case tests for prop firm kill functionality."""

    def test_register_same_account_twice(self, progressive_kill):
        """Test registering same account twice."""
        progressive_kill.register_account_book("prop_001", AccountBook.PROP_FIRM)
        progressive_kill.register_account_book("prop_001", AccountBook.PERSONAL)

        accounts = progressive_kill.get_accounts_by_book(AccountBook.PROP_FIRM)
        personal = progressive_kill.get_accounts_by_book(AccountBook.PERSONAL)

        # Last registration wins
        assert "prop_001" in personal
        assert "prop_001" not in accounts

    @pytest.mark.asyncio
    async def test_kill_with_kill_switch_exception(self, progressive_kill):
        """Test handling when kill_switch.trigger raises exception."""
        progressive_kill.register_account_book("prop_001", AccountBook.PROP_FIRM)

        # Make trigger raise an exception
        progressive_kill.kill_switch.trigger = AsyncMock(
            side_effect=Exception("Kill failed")
        )

        # Should still return success with partial results
        result = await progressive_kill.emergency_kill_prop_firm(reason="Test")

        assert result["success"] is True
        assert result["killed_count"] == 0  # Failed to kill

    @pytest.mark.asyncio
    async def test_empty_account_id(self, progressive_kill):
        """Test with empty account ID registration."""
        progressive_kill.register_account_book("", AccountBook.PROP_FIRM)

        accounts = progressive_kill.get_accounts_by_book(AccountBook.PROP_FIRM)
        assert "" in accounts

        result = await progressive_kill.emergency_kill_prop_firm(reason="Test")
        assert result["killed_count"] == 1

"""Tests for max drawdown enforcement."""
import pytest
from unittest.mock import MagicMock, patch
from src.router.bot_manifest import BotManifest, AccountBook, StrategyType, TradeFrequency
from src.risk.prop_firm_overlay import PropFirmRiskOverlay


def test_trade_blocked_when_drawdown_exceeds_limit():
    """Trade should be blocked when current drawdown >= max_drawdown for account type."""
    # Setup: Create overlay with FTMO account (3% max drawdown)
    overlay = PropFirmRiskOverlay(firm_name="FTMO")

    # Simulate current drawdown at 4% (exceeds FTMO's 3% limit)
    current_drawdown = 0.04

    # Execute: Check if trade should be blocked
    should_block, reason = overlay.should_block_trade(
        account_book="prop_firm",
        prop_firm_name="FTMO",
        current_drawdown=current_drawdown
    )

    # Assert: Trade should be blocked
    assert should_block is True
    assert "max" in reason.lower() and "drawdown" in reason.lower()
    assert "exceeds" in reason.lower()


def test_trade_allowed_when_drawdown_within_limit():
    """Trade should be allowed when current drawdown < max_drawdown for account type."""
    # Setup: Create overlay with FTMO account (3% max drawdown)
    overlay = PropFirmRiskOverlay(firm_name="FTMO")

    # Simulate current drawdown at 2% (within FTMO's 3% limit)
    current_drawdown = 0.02

    # Execute: Check if trade should be blocked
    should_block, reason = overlay.should_block_trade(
        account_book="prop_firm",
        prop_firm_name="FTMO",
        current_drawdown=current_drawdown
    )

    # Assert: Trade should be allowed
    assert should_block is False
    assert reason == ""


def test_personal_account_drawdown_limit():
    """Personal accounts should use 5% drawdown limit."""
    # Setup: Personal account (5% default limit)
    overlay = PropFirmRiskOverlay(firm_name=None)

    # Current drawdown at 6% (exceeds personal 5% limit)
    current_drawdown = 0.06

    # Execute: Check if trade should be blocked
    should_block, reason = overlay.should_block_trade(
        account_book="personal",
        prop_firm_name=None,
        current_drawdown=current_drawdown
    )

    # Assert: Trade should be blocked (exceeds 5%)
    assert should_block is True
    assert "max" in reason.lower() and "drawdown" in reason.lower()


def test_personal_account_drawdown_within_limit():
    """Personal accounts should allow trade when within 5% limit."""
    # Setup: Personal account (5% default limit)
    overlay = PropFirmRiskOverlay(firm_name=None)

    # Current drawdown at 4% (within personal 5% limit)
    current_drawdown = 0.04

    # Execute: Check if trade should be blocked
    should_block, reason = overlay.should_block_trade(
        account_book="personal",
        prop_firm_name=None,
        current_drawdown=current_drawdown
    )

    # Assert: Trade should be allowed
    assert should_block is False
    assert reason == ""


def test_prop_firm_topstep_drawdown_limit():
    """Topstep prop firm has 4% max drawdown."""
    # Setup: Topstep account (4% max drawdown)
    overlay = PropFirmRiskOverlay(firm_name="Topstep")

    # Current drawdown at 5% (exceeds Topstep's 4% limit)
    current_drawdown = 0.05

    # Execute: Check if trade should be blocked
    should_block, reason = overlay.should_block_trade(
        account_book="prop_firm",
        prop_firm_name="Topstep",
        current_drawdown=current_drawdown
    )

    # Assert: Trade should be blocked
    assert should_block is True


def test_drawdown_at_exact_limit():
    """Trade should be blocked when drawdown equals exactly the limit."""
    # Setup: FTMO account (3% max drawdown)
    overlay = PropFirmRiskOverlay(firm_name="FTMO")

    # Current drawdown at exactly 3%
    current_drawdown = 0.03

    # Execute: Check if trade should be blocked
    should_block, reason = overlay.should_block_trade(
        account_book="prop_firm",
        prop_firm_name="FTMO",
        current_drawdown=current_drawdown
    )

    # Assert: Trade should be blocked (>= limit)
    assert should_block is True

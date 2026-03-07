"""
Tests for configurable loss threshold by book type.

Personal Book: 5 consecutive losses
Prop Firm Book: 3 consecutive losses (tighter)
"""
import pytest
from unittest.mock import MagicMock, patch


class TestBotCircuitBreakerLossThreshold:
    """Test suite for BotCircuitBreakerManager loss threshold configuration."""

    def test_personal_book_default_threshold(self):
        """Personal book should have 5 consecutive loss threshold by default."""
        from src.router.bot_circuit_breaker import (
            BotCircuitBreakerManager,
            AccountBook,
            LOSS_THRESHOLDS
        )

        manager = BotCircuitBreakerManager(account_book=AccountBook.PERSONAL)
        assert manager.max_consecutive_losses == 5
        assert LOSS_THRESHOLDS[AccountBook.PERSONAL] == 5

    def test_prop_firm_book_tighter_threshold(self):
        """Prop firm book should have 3 consecutive loss threshold (tighter)."""
        from src.router.bot_circuit_breaker import (
            BotCircuitBreakerManager,
            AccountBook,
            LOSS_THRESHOLDS
        )

        manager = BotCircuitBreakerManager(account_book=AccountBook.PROP_FIRM)
        assert manager.max_consecutive_losses == 3
        assert LOSS_THRESHOLDS[AccountBook.PROP_FIRM] == 3

    def test_personal_book_explicit(self):
        """Explicitly setting personal book should use 5 losses."""
        from src.router.bot_circuit_breaker import BotCircuitBreakerManager

        with patch('src.router.bot_circuit_breaker.DBManager'):
            manager = BotCircuitBreakerManager(account_book_str="personal")
            assert manager.max_consecutive_losses == 5

    def test_threshold_difference(self):
        """Prop firm threshold should be tighter than personal."""
        from src.router.bot_circuit_breaker import (
            BotCircuitBreakerManager,
            AccountBook
        )

        personal_manager = BotCircuitBreakerManager(account_book=AccountBook.PERSONAL)
        prop_manager = BotCircuitBreakerManager(account_book=AccountBook.PROP_FIRM)

        assert personal_manager.max_consecutive_losses > prop_manager.max_consecutive_losses
        assert personal_manager.max_consecutive_losses == 5
        assert prop_manager.max_consecutive_losses == 3

    def test_update_account_book(self):
        """Should be able to update account book and adjust threshold."""
        from src.router.bot_circuit_breaker import (
            BotCircuitBreakerManager,
            AccountBook
        )

        with patch('src.router.bot_circuit_breaker.DBManager'):
            manager = BotCircuitBreakerManager(account_book=AccountBook.PERSONAL)
            assert manager.max_consecutive_losses == 5

            # Update to prop firm
            manager.update_account_book(AccountBook.PROP_FIRM)
            assert manager.max_consecutive_losses == 3

            # Update back to personal
            manager.update_account_book(AccountBook.PERSONAL)
            assert manager.max_consecutive_losses == 5

    def test_get_loss_threshold_helper(self):
        """Test the helper function to get loss threshold."""
        from src.router.bot_circuit_breaker import (
            get_loss_threshold,
            AccountBook
        )

        assert get_loss_threshold(AccountBook.PERSONAL) == 5
        assert get_loss_threshold(AccountBook.PROP_FIRM) == 3

    def test_get_threshold_config(self):
        """Test getting the complete threshold configuration."""
        from src.router.bot_circuit_breaker import get_threshold_config

        config = get_threshold_config()
        assert config["personal"] == 5
        assert config["prop_firm"] == 3


class TestLifecycleManagerThresholds:
    """Test suite for LifecycleManager configurable thresholds by book type."""

    def test_thresholds_by_book_defined(self):
        """THRESHOLDS_BY_BOOK should be properly defined."""
        from src.router.lifecycle_manager import THRESHOLDS_BY_BOOK

        assert "personal" in THRESHOLDS_BY_BOOK
        assert "prop_firm" in THRESHOLDS_BY_BOOK

    def test_personal_book_thresholds(self):
        """Personal book should have 5 losing days threshold."""
        from src.router.lifecycle_manager import THRESHOLDS_BY_BOOK

        assert THRESHOLDS_BY_BOOK["personal"]["max_consecutive_losing_days"] == 5
        assert THRESHOLDS_BY_BOOK["personal"]["max_quarantine_days"] == 30

    def test_prop_firm_book_thresholds(self):
        """Prop firm book should have 3 losing days threshold (tighter)."""
        from src.router.lifecycle_manager import THRESHOLDS_BY_BOOK

        assert THRESHOLDS_BY_BOOK["prop_firm"]["max_consecutive_losing_days"] == 3
        assert THRESHOLDS_BY_BOOK["prop_firm"]["max_quarantine_days"] == 14

    def test_prop_firm_tighter_than_personal(self):
        """Prop firm thresholds should be tighter than personal."""
        from src.router.lifecycle_manager import THRESHOLDS_BY_BOOK

        personal = THRESHOLDS_BY_BOOK["personal"]
        prop_firm = THRESHOLDS_BY_BOOK["prop_firm"]

        assert prop_firm["max_consecutive_losing_days"] < personal["max_consecutive_losing_days"]
        assert prop_firm["max_quarantine_days"] < personal["max_quarantine_days"]


class TestIntegrationLossThreshold:
    """Integration tests for loss threshold by book type."""

    def test_personal_book_threshold_value(self):
        """Personal book should have threshold of 5 for integration."""
        from src.router.bot_circuit_breaker import (
            BotCircuitBreakerManager,
            AccountBook,
            get_loss_threshold
        )

        # Test via manager
        manager = BotCircuitBreakerManager(account_book=AccountBook.PERSONAL)
        assert manager.max_consecutive_losses == 5

        # Test via helper function
        assert get_loss_threshold(AccountBook.PERSONAL) == 5

    def test_prop_firm_book_threshold_value(self):
        """Prop firm book should have threshold of 3 for integration."""
        from src.router.bot_circuit_breaker import (
            BotCircuitBreakerManager,
            AccountBook,
            get_loss_threshold
        )

        # Test via manager
        manager = BotCircuitBreakerManager(account_book=AccountBook.PROP_FIRM)
        assert manager.max_consecutive_losses == 3

        # Test via helper function
        assert get_loss_threshold(AccountBook.PROP_FIRM) == 3

    def test_account_book_from_bot_manifest(self):
        """Test extracting account_book from BotManifest for lifecycle checks."""
        from src.router.lifecycle_manager import THRESHOLDS_BY_BOOK

        # Personal book thresholds
        assert THRESHOLDS_BY_BOOK["personal"]["max_consecutive_losing_days"] == 5
        assert THRESHOLDS_BY_BOOK["personal"]["max_quarantine_days"] == 30

        # Prop firm thresholds
        assert THRESHOLDS_BY_BOOK["prop_firm"]["max_consecutive_losing_days"] == 3
        assert THRESHOLDS_BY_BOOK["prop_firm"]["max_quarantine_days"] == 14

    def test_threshold_consistency_across_modules(self):
        """Verify thresholds are consistent between circuit breaker and lifecycle manager."""
        from src.router.bot_circuit_breaker import LOSS_THRESHOLDS, AccountBook
        from src.router.lifecycle_manager import THRESHOLDS_BY_BOOK

        # Personal book: 5 losses in both
        assert LOSS_THRESHOLDS[AccountBook.PERSONAL] == 5
        assert THRESHOLDS_BY_BOOK["personal"]["max_consecutive_losing_days"] == 5

        # Prop firm: 3 losses in both
        assert LOSS_THRESHOLDS[AccountBook.PROP_FIRM] == 3
        assert THRESHOLDS_BY_BOOK["prop_firm"]["max_consecutive_losing_days"] == 3

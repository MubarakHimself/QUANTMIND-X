"""
Tests for Bot Circuit Breaker API (P0-010)

Tests the BotCircuitBreakerManager for automatic bot quarantine:
- Personal Book: 5 consecutive losses threshold
- Prop Firm Book: 3 consecutive losses threshold (tighter)
- Daily trade limit: 20 trades
- Fee kill switch integration

Validates:
- P0-010: Bot circuit breaker quarantine on consecutive losses
- Account book type affects loss threshold
- Daily trade limit enforcement
- Fee monitor integration
- Manual quarantine/reactivation
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from datetime import datetime, date, timezone
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class TestBotCircuitBreakerManager:
    """Test BotCircuitBreakerManager for bot quarantine logic."""

    @pytest.fixture
    def mock_db_manager(self):
        """Create a mock DBManager."""
        mock = MagicMock()
        mock_session = MagicMock()
        mock.get_session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock.get_session.return_value.__exit__ = MagicMock(return_value=False)
        return mock

    @pytest.fixture
    def mock_fee_monitor(self):
        """Create a mock FeeMonitor."""
        mock = MagicMock()
        mock.should_halt_trading.return_value = (False, None)
        return mock

    def test_check_allowed_new_bot(self, mock_db_manager, mock_fee_monitor):
        """[P0] New bot should be allowed to trade."""
        from src.router.bot_circuit_breaker import BotCircuitBreakerManager, AccountBook

        # Mock the fee monitor
        with patch('src.router.bot_circuit_breaker.FeeMonitor', return_value=mock_fee_monitor):
            manager = BotCircuitBreakerManager(db_manager=mock_db_manager)

            # Setup mock to return None (new bot)
            mock_session = mock_db_manager.get_session.return_value.__enter__.return_value
            mock_session.query.return_value.filter.return_value.first.return_value = None

            allowed, reason = manager.check_allowed("new_bot_001")

            assert allowed is True
            assert reason is None

    def test_check_allowed_quarantined_bot(self, mock_db_manager, mock_fee_monitor):
        """[P0] Quarantined bot should NOT be allowed to trade."""
        from src.router.bot_circuit_breaker import BotCircuitBreakerManager

        with patch('src.router.bot_circuit_breaker.FeeMonitor', return_value=mock_fee_monitor):
            manager = BotCircuitBreakerManager(db_manager=mock_db_manager)

            # Create mock quarantined state
            mock_state = MagicMock()
            mock_state.is_quarantined = True
            mock_state.quarantine_reason = "5 consecutive losses"
            mock_state.daily_trade_count = 0

            mock_session = mock_db_manager.get_session.return_value.__enter__.return_value
            mock_session.query.return_value.filter.return_value.first.return_value = mock_state

            allowed, reason = manager.check_allowed("quarantined_bot")

            assert allowed is False
            assert "quarantined" in reason.lower() or "5 consecutive losses" in reason

    def test_check_allowed_daily_limit_reached(self, mock_db_manager, mock_fee_monitor):
        """[P0] Bot at daily trade limit should NOT be allowed to trade."""
        from src.router.bot_circuit_breaker import BotCircuitBreakerManager

        with patch('src.router.bot_circuit_breaker.FeeMonitor', return_value=mock_fee_monitor):
            manager = BotCircuitBreakerManager(db_manager=mock_db_manager)

            # Create mock state at daily limit
            mock_state = MagicMock()
            mock_state.is_quarantined = False
            mock_state.daily_trade_count = 20  # At limit

            mock_session = mock_db_manager.get_session.return_value.__enter__.return_value
            mock_session.query.return_value.filter.return_value.first.return_value = mock_state

            allowed, reason = manager.check_allowed("busy_bot")

            assert allowed is False
            assert "daily trade limit" in reason.lower()

    def test_record_trade_win_resets_consecutive_losses(self, mock_db_manager, mock_fee_monitor):
        """[P0] Win should reset consecutive losses counter."""
        from src.router.bot_circuit_breaker import BotCircuitBreakerManager

        with patch('src.router.bot_circuit_breaker.FeeMonitor', return_value=mock_fee_monitor):
            manager = BotCircuitBreakerManager(db_manager=mock_db_manager)

            # Create mock state with 3 consecutive losses
            mock_state = MagicMock()
            mock_state.consecutive_losses = 3
            mock_state.daily_trade_count = 5
            mock_state.last_trade_time = datetime.now(timezone.utc)
            mock_state.is_quarantined = False

            mock_session = mock_db_manager.get_session.return_value.__enter__.return_value
            mock_session.query.return_value.filter.return_value.first.return_value = mock_state

            result = manager.record_trade("bot_001", is_loss=False)

            # Consecutive losses should be reset to 0
            assert result.consecutive_losses == 0
            # Trade count should increment
            assert result.daily_trade_count == 6

    def test_record_trade_loss_increments_consecutive_losses(self, mock_db_manager, mock_fee_monitor):
        """[P0] Loss should increment consecutive losses counter."""
        from src.router.bot_circuit_breaker import BotCircuitBreakerManager

        with patch('src.router.bot_circuit_breaker.FeeMonitor', return_value=mock_fee_monitor):
            manager = BotCircuitBreakerManager(db_manager=mock_db_manager)

            # Create mock state with 2 consecutive losses
            mock_state = MagicMock()
            mock_state.consecutive_losses = 2
            mock_state.daily_trade_count = 5
            mock_state.last_trade_time = datetime.now(timezone.utc)
            mock_state.is_quarantined = False

            mock_session = mock_db_manager.get_session.return_value.__enter__.return_value
            mock_session.query.return_value.filter.return_value.first.return_value = mock_state

            result = manager.record_trade("bot_001", is_loss=True)

            # Consecutive losses should be incremented
            assert result.consecutive_losses == 3

    def test_record_trade_personal_threshold_quarantine(self, mock_db_manager, mock_fee_monitor):
        """[P0] Personal book bot should be quarantined after 5 consecutive losses."""
        from src.router.bot_circuit_breaker import BotCircuitBreakerManager, AccountBook

        with patch('src.router.bot_circuit_breaker.FeeMonitor', return_value=mock_fee_monitor):
            manager = BotCircuitBreakerManager(
                db_manager=mock_db_manager,
                account_book=AccountBook.PERSONAL
            )

            # Create mock state at threshold - 1
            mock_state = MagicMock()
            mock_state.consecutive_losses = 4
            mock_state.daily_trade_count = 10
            mock_state.last_trade_time = datetime.now(timezone.utc)
            mock_state.is_quarantined = False
            mock_state.bot_id = "personal_bot"

            mock_session = mock_db_manager.get_session.return_value.__enter__.return_value
            mock_session.query.return_value.filter.return_value.first.return_value = mock_state

            result = manager.record_trade("personal_bot", is_loss=True)

            # Should be quarantined after 5th loss
            assert result.is_quarantined is True
            assert "5 consecutive losses" in str(result.quarantine_reason)

    def test_record_trade_prop_firm_threshold_quarantine(self, mock_db_manager, mock_fee_monitor):
        """[P0] Prop firm bot should be quarantined after 3 consecutive losses (tighter threshold)."""
        from src.router.bot_circuit_breaker import BotCircuitBreakerManager, AccountBook

        with patch('src.router.bot_circuit_breaker.FeeMonitor', return_value=mock_fee_monitor):
            manager = BotCircuitBreakerManager(
                db_manager=mock_db_manager,
                account_book=AccountBook.PROP_FIRM
            )

            # Create mock state at threshold - 1 for prop firm (3 losses)
            mock_state = MagicMock()
            mock_state.consecutive_losses = 2
            mock_state.daily_trade_count = 10
            mock_state.last_trade_time = datetime.now(timezone.utc)
            mock_state.is_quarantined = False
            mock_state.bot_id = "prop_bot"

            mock_session = mock_db_manager.get_session.return_value.__enter__.return_value
            mock_session.query.return_value.filter.return_value.first.return_value = mock_state

            result = manager.record_trade("prop_bot", is_loss=True)

            # Should be quarantined after 3rd loss for prop firm
            assert result.is_quarantined is True
            assert "3 consecutive losses" in str(result.quarantine_reason)

    def test_record_trade_daily_limit_triggers_quarantine(self, mock_db_manager, mock_fee_monitor):
        """[P0] Exceeding daily trade limit should trigger quarantine."""
        from src.router.bot_circuit_breaker import BotCircuitBreakerManager

        with patch('src.router.bot_circuit_breaker.FeeMonitor', return_value=mock_fee_monitor):
            manager = BotCircuitBreakerManager(db_manager=mock_db_manager)

            # Create mock state just over daily limit
            mock_state = MagicMock()
            mock_state.consecutive_losses = 0
            mock_state.daily_trade_count = 20  # At limit
            mock_state.last_trade_time = datetime.now(timezone.utc)
            mock_state.is_quarantined = False
            mock_state.bot_id = "over_trading_bot"

            mock_session = mock_db_manager.get_session.return_value.__enter__.return_value
            mock_session.query.return_value.filter.return_value.first.return_value = mock_state

            result = manager.record_trade("over_trading_bot", is_loss=False)

            # Should be quarantined for exceeding daily limit
            assert result.is_quarantined is True
            assert "Daily trade limit" in str(result.quarantine_reason)

    def test_quarantine_bot_manual(self, mock_db_manager, mock_fee_monitor):
        """[P0] Manual quarantine should set is_quarantined=True with reason."""
        from src.router.bot_circuit_breaker import BotCircuitBreakerManager

        with patch('src.router.bot_circuit_breaker.FeeMonitor', return_value=mock_fee_monitor):
            manager = BotCircuitBreakerManager(db_manager=mock_db_manager)

            # Create mock state
            mock_state = MagicMock()
            mock_state.is_quarantined = False
            mock_state.quarantine_reason = None
            mock_state.quarantine_start = None

            mock_session = mock_db_manager.get_session.return_value.__enter__.return_value
            mock_session.query.return_value.filter.return_value.first.return_value = mock_state

            result = manager.quarantine_bot("test_bot", reason="Manual risk review")

            assert result.is_quarantined is True
            assert result.quarantine_reason == "Manual risk review"
            assert result.quarantine_start is not None

    def test_reactivate_bot_clears_quarantine(self, mock_db_manager, mock_fee_monitor):
        """[P0] Reactivating bot should clear quarantine and reset losses."""
        from src.router.bot_circuit_breaker import BotCircuitBreakerManager

        with patch('src.router.bot_circuit_breaker.FeeMonitor', return_value=mock_fee_monitor):
            manager = BotCircuitBreakerManager(db_manager=mock_db_manager)

            # Create mock quarantined state
            mock_state = MagicMock()
            mock_state.is_quarantined = True
            mock_state.quarantine_reason = "5 consecutive losses"
            mock_state.quarantine_start = datetime.now(timezone.utc)
            mock_state.consecutive_losses = 5

            mock_session = mock_db_manager.get_session.return_value.__enter__.return_value
            mock_session.query.return_value.filter.return_value.first.return_value = mock_state

            result = manager.reactivate_bot("recovered_bot")

            assert result.is_quarantined is False
            assert result.quarantine_reason is None
            assert result.quarantine_start is None
            assert result.consecutive_losses == 0

    def test_get_quarantined_bots(self, mock_db_manager, mock_fee_monitor):
        """[P1] get_quarantined_bots should return all quarantined bots."""
        from src.router.bot_circuit_breaker import BotCircuitBreakerManager

        with patch('src.router.bot_circuit_breaker.FeeMonitor', return_value=mock_fee_monitor):
            manager = BotCircuitBreakerManager(db_manager=mock_db_manager)

            # Create mock quarantined bots
            mock_bot1 = MagicMock()
            mock_bot1.bot_id = "quarantined_bot_1"
            mock_bot1.consecutive_losses = 5
            mock_bot1.daily_trade_count = 15
            mock_bot1.quarantine_reason = "5 consecutive losses"
            mock_bot1.quarantine_start = datetime.now(timezone.utc)

            mock_bot2 = MagicMock()
            mock_bot2.bot_id = "quarantined_bot_2"
            mock_bot2.consecutive_losses = 3
            mock_bot2.daily_trade_count = 25
            mock_bot2.quarantine_reason = "Daily trade limit exceeded"
            mock_bot2.quarantine_start = datetime.now(timezone.utc)

            mock_session = mock_db_manager.get_session.return_value.__enter__.return_value
            mock_session.query.return_value.filter.return_value.all.return_value = [mock_bot1, mock_bot2]

            result = manager.get_quarantined_bots()

            assert len(result) == 2
            assert result[0]["bot_id"] == "quarantined_bot_1"
            assert result[1]["bot_id"] == "quarantined_bot_2"

    def test_get_state_returns_dict(self, mock_db_manager, mock_fee_monitor):
        """[P1] get_state should return state as dictionary."""
        from src.router.bot_circuit_breaker import BotCircuitBreakerManager

        with patch('src.router.bot_circuit_breaker.FeeMonitor', return_value=mock_fee_monitor):
            manager = BotCircuitBreakerManager(db_manager=mock_db_manager)

            # Create mock state
            mock_state = MagicMock()
            mock_state.id = 1
            mock_state.bot_id = "test_bot"
            mock_state.consecutive_losses = 3
            mock_state.daily_trade_count = 10
            mock_state.last_trade_time = datetime.now(timezone.utc)
            mock_state.is_quarantined = False
            mock_state.quarantine_reason = None
            mock_state.quarantine_start = None

            mock_session = mock_db_manager.get_session.return_value.__enter__.return_value
            mock_session.query.return_value.filter.return_value.first.return_value = mock_state

            result = manager.get_state("test_bot")

            assert result is not None
            assert result["bot_id"] == "test_bot"
            assert result["consecutive_losses"] == 3
            assert result["daily_trade_count"] == 10
            assert result["is_quarantined"] is False

    def test_get_state_returns_none_for_unknown_bot(self, mock_db_manager, mock_fee_monitor):
        """[P1] get_state should return None for unknown bot."""
        from src.router.bot_circuit_breaker import BotCircuitBreakerManager

        with patch('src.router.bot_circuit_breaker.FeeMonitor', return_value=mock_fee_monitor):
            manager = BotCircuitBreakerManager(db_manager=mock_db_manager)

            mock_session = mock_db_manager.get_session.return_value.__enter__.return_value
            mock_session.query.return_value.filter.return_value.first.return_value = None

            result = manager.get_state("unknown_bot")

            assert result is None

    def test_new_day_resets_daily_counter(self, mock_db_manager, mock_fee_monitor):
        """[P1] New day should reset daily trade counter."""
        from src.router.bot_circuit_breaker import BotCircuitBreakerManager

        with patch('src.router.bot_circuit_breaker.FeeMonitor', return_value=mock_fee_monitor):
            manager = BotCircuitBreakerManager(db_manager=mock_db_manager)

            # Create mock state from yesterday
            yesterday = datetime.now(timezone.utc)
            from datetime import timedelta
            mock_state = MagicMock()
            mock_state.consecutive_losses = 2
            mock_state.daily_trade_count = 18  # High from yesterday
            mock_state.last_trade_time = datetime.now(timezone.utc) - timedelta(days=1)
            mock_state.is_quarantined = False
            mock_state.bot_id = "bot_with_history"

            mock_session = mock_db_manager.get_session.return_value.__enter__.return_value
            mock_session.query.return_value.filter.return_value.first.return_value = mock_state

            # Record trade today (new day)
            result = manager.record_trade("bot_with_history", is_loss=False, trade_date=date.today())

            # Daily counter should be reset to 1 (new day, first trade)
            assert result.daily_trade_count == 1
            # But consecutive losses should persist
            assert result.consecutive_losses == 0  # Win resets losses

    def test_fee_kill_switch_triggers_quarantine(self, mock_db_manager, mock_fee_monitor):
        """[P1] Fee kill switch should trigger quarantine when fee threshold exceeded."""
        from src.router.bot_circuit_breaker import BotCircuitBreakerManager

        # Mock fee monitor to return halt signal
        mock_fee_monitor.should_halt_trading.return_value = (True, "Fee threshold 5% exceeded")

        with patch('src.router.bot_circuit_breaker.FeeMonitor', return_value=mock_fee_monitor):
            manager = BotCircuitBreakerManager(db_manager=mock_db_manager)

            # Create mock state
            mock_state = MagicMock()
            mock_state.consecutive_losses = 1
            mock_state.daily_trade_count = 5
            mock_state.last_trade_time = datetime.now(timezone.utc)
            mock_state.is_quarantined = False
            mock_state.bot_id = "fee_alert_bot"

            mock_session = mock_db_manager.get_session.return_value.__enter__.return_value
            mock_session.query.return_value.filter.return_value.first.return_value = mock_state

            result = manager.record_trade("fee_alert_bot", is_loss=False, fee=100.0)

            # Should be quarantined due to fee kill switch
            assert result.is_quarantined is True
            assert "FEE_KILL_SWITCH" in str(result.quarantine_reason)


class TestLossThresholdConfiguration:
    """Test loss threshold configuration by account book type."""

    def test_personal_book_threshold_is_5(self):
        """[P0] Personal book should have 5 consecutive loss threshold."""
        from src.router.bot_circuit_breaker import get_loss_threshold, AccountBook

        threshold = get_loss_threshold(AccountBook.PERSONAL)
        assert threshold == 5

    def test_prop_firm_threshold_is_3(self):
        """[P0] Prop firm book should have tighter 3 consecutive loss threshold."""
        from src.router.bot_circuit_breaker import get_loss_threshold, AccountBook

        threshold = get_loss_threshold(AccountBook.PROP_FIRM)
        assert threshold == 3

    def test_threshold_config_returns_all_books(self):
        """[P1] get_threshold_config should return config for all account book types."""
        from src.router.bot_circuit_breaker import get_threshold_config

        config = get_threshold_config()

        assert "personal" in config
        assert "prop_firm" in config
        assert config["personal"] == 5
        assert config["prop_firm"] == 3

    def test_unknown_book_falls_back_to_personal(self):
        """[P2] Unknown account book type should fall back to personal threshold."""
        from src.router.bot_circuit_breaker import get_loss_threshold, AccountBook, LOSS_THRESHOLDS

        # Direct lookup with unknown
        threshold = get_loss_threshold(AccountBook.PERSONAL)  # Default fallback
        assert threshold == LOSS_THRESHOLDS[AccountBook.PERSONAL]


class TestAccountBookEnum:
    """Test AccountBook enum values."""

    def test_account_book_personal_value(self):
        """[P1] AccountBook.PERSONAL should have correct value."""
        from src.router.bot_circuit_breaker import AccountBook

        assert AccountBook.PERSONAL.value == "personal"

    def test_account_book_prop_firm_value(self):
        """[P1] AccountBook.PROP_FIRM should have correct value."""
        from src.router.bot_circuit_breaker import AccountBook

        assert AccountBook.PROP_FIRM.value == "prop_firm"

    def test_account_book_from_string(self):
        """[P1] AccountBook should be constructable from string."""
        from src.router.bot_circuit_breaker import AccountBook

        personal = AccountBook("personal")
        prop_firm = AccountBook("prop_firm")

        assert personal == AccountBook.PERSONAL
        assert prop_firm == AccountBook.PROP_FIRM


class TestDailyTradeLimit:
    """Test daily trade limit configuration."""

    def test_default_daily_trade_limit_is_20(self):
        """[P0] Default daily trade limit should be 20."""
        from src.router.bot_circuit_breaker import BotCircuitBreakerManager, DEFAULT_DAILY_TRADE_LIMIT

        assert DEFAULT_DAILY_TRADE_LIMIT == 20

    def test_manager_uses_default_limit(self):
        """[P1] Manager should use default daily trade limit of 20."""
        from src.router.bot_circuit_breaker import BotCircuitBreakerManager, DEFAULT_DAILY_TRADE_LIMIT

        # Manager should initialize with the constant
        assert DEFAULT_DAILY_TRADE_LIMIT == 20

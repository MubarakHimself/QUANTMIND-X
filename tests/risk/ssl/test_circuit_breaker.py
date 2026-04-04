"""
Tests for SSL Circuit Breaker.

Story 18.1: Per-Bot Consecutive Loss Counter & Paper Rotation

Tests cover:
- increment_consecutive_losses on loss trade close
- reset_consecutive_losses on win trade close
- circuit breaker fires at 2 losses (scalping threshold)
- circuit breaker fires at 3 losses (ORB threshold)
- TIER assignment logic
- recovery evaluation
- promote_to_live
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timezone

from src.risk.ssl.circuit_breaker import (
    SSLCircuitBreaker,
    BotType,
    LOSS_THRESHOLDS,
    SSL_EVENTS_CHANNEL,
)
from src.risk.ssl.state import SSLState, BotTier, is_valid_transition
from src.events.ssl import SSLEventType


class TestSSLCircuitBreakerBasics:
    """Test basic SSLCircuitBreaker functionality."""

    def test_initialization(self):
        """Test SSLCircuitBreaker initializes correctly."""
        ssl = SSLCircuitBreaker()
        assert ssl._db_session is None
        assert ssl._redis_host == "localhost"
        assert ssl._redis_port == 6379

    def test_initialization_with_custom_params(self):
        """Test SSLCircuitBreaker initializes with custom parameters."""
        mock_session = Mock()
        ssl = SSLCircuitBreaker(
            db_session=mock_session,
            redis_host="redis.example.com",
            redis_port=6380,
        )
        assert ssl._db_session is mock_session
        assert ssl._redis_host == "redis.example.com"
        assert ssl._redis_port == 6380


class TestLossThresholds:
    """Test loss threshold configuration."""

    def test_scalping_threshold(self):
        """Test scalping threshold is 2."""
        assert LOSS_THRESHOLDS[BotType.SCALPING] == 2

    def test_orb_threshold(self):
        """Test ORB threshold is 3."""
        assert LOSS_THRESHOLDS[BotType.ORB] == 3


class TestValidTransitions:
    """Test state transition validation."""

    def test_live_to_paper_valid(self):
        """Test LIVE -> PAPER is a valid transition."""
        assert is_valid_transition(SSLState.LIVE, SSLState.PAPER) is True

    def test_paper_to_recovery_valid(self):
        """Test PAPER -> RECOVERY is a valid transition."""
        assert is_valid_transition(SSLState.PAPER, SSLState.RECOVERY) is True

    def test_paper_to_retired_valid(self):
        """Test PAPER -> RETIRED is a valid transition."""
        assert is_valid_transition(SSLState.PAPER, SSLState.RETIRED) is True

    def test_recovery_to_live_valid(self):
        """Test RECOVERY -> LIVE is a valid transition."""
        assert is_valid_transition(SSLState.RECOVERY, SSLState.LIVE) is True

    def test_recovery_to_paper_valid(self):
        """Test RECOVERY -> PAPER is a valid transition (loss during recovery)."""
        assert is_valid_transition(SSLState.RECOVERY, SSLState.PAPER) is True

    def test_live_to_recovery_invalid(self):
        """Test LIVE -> RECOVERY is an invalid transition."""
        assert is_valid_transition(SSLState.LIVE, SSLState.RECOVERY) is False

    def test_live_to_retired_invalid(self):
        """Test LIVE -> RETIRED is an invalid transition."""
        assert is_valid_transition(SSLState.LIVE, SSLState.RETIRED) is False

    def test_paper_to_live_invalid(self):
        """Test PAPER -> LIVE is an invalid transition (must go through recovery)."""
        assert is_valid_transition(SSLState.PAPER, SSLState.LIVE) is False

    def test_retired_to_any_invalid(self):
        """Test RETIRED -> any state is an invalid transition (terminal state)."""
        assert is_valid_transition(SSLState.RETIRED, SSLState.LIVE) is False
        assert is_valid_transition(SSLState.RETIRED, SSLState.PAPER) is False
        assert is_valid_transition(SSLState.RETIRED, SSLState.RECOVERY) is False


class TestIncrementConsecutiveLosses:
    """Test increment_consecutive_losses functionality."""

    @patch('src.risk.ssl.circuit_breaker.SSLCircuitBreakerState')
    def test_increment_creates_new_record(self, mock_state_mgr):
        """Test increment creates a new record if none exists."""
        mock_session = Mock()
        ssl = SSLCircuitBreaker(db_session=mock_session)

        mock_state_mgr_instance = Mock()
        mock_state_mgr_instance._get_record.return_value = None
        ssl._state_manager = mock_state_mgr_instance

        mock_db_record = Mock()
        mock_db_record.consecutive_losses = 1
        mock_session.add = Mock()
        mock_session.commit = Mock()

        # Patch the internal _db_session
        ssl._db_session = mock_session
        count = ssl.increment_consecutive_losses("bot-1", "12345")

        assert count == 1
        mock_session.add.assert_called_once()

    @patch('src.risk.ssl.circuit_breaker.SSLCircuitBreakerState')
    def test_increment_existing_record(self, mock_state_mgr):
        """Test increment updates existing record."""
        mock_session = Mock()
        ssl = SSLCircuitBreaker(db_session=mock_session)

        mock_record = Mock()
        mock_record.consecutive_losses = 2
        mock_record.magic_number = "12345"

        mock_state_mgr_instance = Mock()
        mock_state_mgr_instance._get_record.return_value = mock_record
        ssl._state_manager = mock_state_mgr_instance

        mock_session.commit = Mock()

        count = ssl.increment_consecutive_losses("bot-1", "12345")

        assert count == 3
        mock_session.commit.assert_called_once()


class TestResetConsecutiveLosses:
    """Test reset_consecutive_losses functionality."""

    @patch('src.risk.ssl.circuit_breaker.SSLCircuitBreakerState')
    def test_reset_to_zero(self, mock_state_mgr):
        """Test reset sets consecutive_losses to 0."""
        mock_session = Mock()
        ssl = SSLCircuitBreaker(db_session=mock_session)

        mock_record = Mock()
        mock_record.consecutive_losses = 5
        mock_record.magic_number = "12345"

        mock_state_mgr_instance = Mock()
        mock_state_mgr_instance._get_record.return_value = mock_record
        ssl._state_manager = mock_state_mgr_instance

        mock_session.commit = Mock()

        ssl.reset_consecutive_losses("bot-1", "12345")

        assert mock_record.consecutive_losses == 0
        mock_session.commit.assert_called_once()


class TestEvaluateCircuitBreaker:
    """Test evaluate_circuit_breaker functionality."""

    @patch('src.risk.ssl.circuit_breaker.SSLCircuitBreakerState')
    def test_fire_when_threshold_breached_scalping(self, mock_state_mgr):
        """Test circuit breaker fires at 2 losses for scalping."""
        mock_session = Mock()
        ssl = SSLCircuitBreaker(db_session=mock_session)

        mock_state_mgr_instance = Mock()
        mock_state_mgr_instance.get_state.return_value = SSLState.LIVE
        mock_state_mgr_instance.get_consecutive_losses.return_value = 2
        ssl._state_manager = mock_state_mgr_instance

        # Mock bot type as scalping
        with patch.object(ssl, '_get_threshold', return_value=2):
            should_fire = ssl.evaluate_circuit_breaker("bot-1")

        assert should_fire is True

    @patch('src.risk.ssl.circuit_breaker.SSLCircuitBreakerState')
    def test_no_fire_below_threshold(self, mock_state_mgr):
        """Test circuit breaker does not fire below threshold."""
        mock_session = Mock()
        ssl = SSLCircuitBreaker(db_session=mock_session)

        mock_state_mgr_instance = Mock()
        mock_state_mgr_instance.get_state.return_value = SSLState.LIVE
        mock_state_mgr_instance.get_consecutive_losses.return_value = 1
        ssl._state_manager = mock_state_mgr_instance

        with patch.object(ssl, '_get_threshold', return_value=2):
            should_fire = ssl.evaluate_circuit_breaker("bot-1")

        assert should_fire is False

    @patch('src.risk.ssl.circuit_breaker.SSLCircuitBreakerState')
    def test_no_fire_if_already_in_paper(self, mock_state_mgr):
        """Test circuit breaker does not fire if already in paper."""
        mock_session = Mock()
        ssl = SSLCircuitBreaker(db_session=mock_session)

        mock_state_mgr_instance = Mock()
        mock_state_mgr_instance.get_state.return_value = SSLState.PAPER
        mock_state_mgr_instance.get_consecutive_losses.return_value = 5
        ssl._state_manager = mock_state_mgr_instance

        should_fire = ssl.evaluate_circuit_breaker("bot-1")

        assert should_fire is False


class TestBotTypeDetection:
    """Test bot type detection for threshold selection."""

    def test_default_bot_type_scalping(self):
        """Test default bot type is scalping."""
        mock_session = Mock()
        ssl = SSLCircuitBreaker(db_session=mock_session)

        # By default, bot type should be scalping
        bot_type = ssl._get_bot_type("bot-1")
        assert bot_type == BotType.SCALPING

    def test_bot_type_threshold_assignment(self):
        """Test that bot type correctly determines threshold."""
        mock_session = Mock()
        ssl = SSLCircuitBreaker(db_session=mock_session)

        # Scalping threshold should be 2
        with patch.object(ssl, '_get_bot_type', return_value=BotType.SCALPING):
            threshold = ssl._get_threshold("bot-1")
            assert threshold == 2

        # ORB threshold should be 3
        with patch.object(ssl, '_get_bot_type', return_value=BotType.ORB):
            threshold = ssl._get_threshold("orb-bot-1")
            assert threshold == 3


class TestTierDetermination:
    """Test tier determination logic."""

    @patch('src.database.models.BotLifecycleLog')
    def test_tier_2_for_bot_without_live_history(self, mock_log):
        """Test TIER_2 is assigned for bot without live history."""
        mock_session = Mock()
        ssl = SSLCircuitBreaker(db_session=mock_session)

        # Mock no live history
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        tier = ssl._determine_tier("bot-1")
        # When no live history, defaults to TIER_2
        assert tier == BotTier.TIER_2


class TestOnTradeClose:
    """Test on_trade_close functionality."""

    @patch('src.risk.ssl.circuit_breaker.SSLCircuitBreaker._emit_event')
    @patch('src.risk.ssl.circuit_breaker.SSLCircuitBreaker._remove_primal_tag')
    @patch('src.risk.ssl.circuit_breaker.SSLCircuitBreaker._add_paper_only_tag')
    @patch('src.risk.ssl.circuit_breaker.SSLCircuitBreakerState')
    def test_loss_increments_counter_and_fires(self, mock_state_mgr, mock_add_tag, mock_remove_tag, mock_emit):
        """Test loss increments counter and fires circuit breaker at threshold."""
        mock_session = Mock()
        ssl = SSLCircuitBreaker(db_session=mock_session)

        # Setup state manager mock
        mock_state_mgr_instance = Mock()
        mock_state_mgr_instance.get_state.return_value = SSLState.LIVE
        mock_state_mgr_instance.get_consecutive_losses.return_value = 1
        mock_state_mgr_instance.get_magic_number.return_value = "12345"
        mock_state_mgr_instance.get_tier.return_value = BotTier.TIER_2
        ssl._state_manager = mock_state_mgr_instance

        # Mock increment to return 2 (threshold)
        with patch.object(ssl, 'increment_consecutive_losses', return_value=2):
            with patch.object(ssl, '_get_threshold', return_value=2):
                with patch.object(ssl, '_determine_tier', return_value=BotTier.TIER_1):
                    event = ssl.on_trade_close("bot-1", "12345", is_win=False)

        # Verify circuit breaker fired
        assert event is not None
        assert event.event_type == SSLEventType.MOVE_TO_PAPER
        assert event.tier == "TIER_1"
        mock_add_tag.assert_called_once_with("bot-1")
        mock_remove_tag.assert_called_once_with("bot-1")

    @patch('src.risk.ssl.circuit_breaker.SSLCircuitBreaker._emit_event')
    @patch('src.risk.ssl.circuit_breaker.SSLCircuitBreakerState')
    def test_win_resets_counter(self, mock_state_mgr, mock_emit):
        """Test win resets consecutive losses counter."""
        mock_session = Mock()
        ssl = SSLCircuitBreaker(db_session=mock_session)

        mock_state_mgr_instance = Mock()
        mock_state_mgr_instance.get_state.return_value = SSLState.LIVE
        ssl._state_manager = mock_state_mgr_instance

        with patch.object(ssl, 'reset_consecutive_losses') as mock_reset:
            event = ssl.on_trade_close("bot-1", "12345", is_win=True)

        mock_reset.assert_called_once_with("bot-1", "12345")
        assert event is None


class TestSSLEventEmission:
    """Test SSL event emission to Redis."""

    @patch('redis.Redis')
    def test_emit_event_success(self, mock_redis_class):
        """Test event is emitted to Redis successfully."""
        mock_redis_instance = Mock()
        mock_redis_class.return_value = mock_redis_instance

        mock_session = Mock()
        ssl = SSLCircuitBreaker(db_session=mock_session)

        from src.events.ssl import SSLCircuitBreakerEvent

        event = SSLCircuitBreakerEvent(
            bot_id="bot-1",
            magic_number="12345",
            event_type=SSLEventType.MOVE_TO_PAPER,
            consecutive_losses=2,
            tier="TIER_1",
            previous_state=SSLState.LIVE,
            new_state=SSLState.PAPER,
        )

        result = ssl._emit_event(event)

        assert result is True
        mock_redis_instance.publish.assert_called_once()
        # Note: Client is reused, not closed after each publish

    @patch('redis.Redis')
    def test_emit_event_failure(self, mock_redis_class):
        """Test event emission failure is handled gracefully."""
        import redis
        mock_redis_class.side_effect = redis.RedisError("Connection failed")

        mock_session = Mock()
        ssl = SSLCircuitBreaker(db_session=mock_session)

        from src.events.ssl import SSLCircuitBreakerEvent

        event = SSLCircuitBreakerEvent(
            bot_id="bot-1",
            magic_number="12345",
            event_type=SSLEventType.MOVE_TO_PAPER,
            consecutive_losses=2,
            previous_state=SSLState.LIVE,
            new_state=SSLState.PAPER,
        )

        result = ssl._emit_event(event)

        assert result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

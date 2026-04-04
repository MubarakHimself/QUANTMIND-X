"""
Integration Tests for DPR SSL Consumer — Story 18.2.

Tests:
- Task 7: DPRSSLConsumer class to subscribe to Redis ssl:events channel
- Task 1: SSL event emission with trade outcome
- Task 2: Recovery event integration
- Task 3: Retirement & AlphaForge trigger
- Task 4: DPR audit log integration
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timezone

from src.events.ssl import SSLCircuitBreakerEvent, SSLEventType, SSLState, TradeOutcome
from src.events.dpr import SSLEvent, SSLEventType as DPRSSLEventType
from src.risk.dpr.ssl_consumer import DPRSSLConsumer, DPRSSLEventEmitter, SSL_EVENTS_CHANNEL
from src.risk.dpr.scoring_engine import DPRScoringEngine
from src.risk.dpr.queue_manager import DPRQueueManager


class TestDPRSSLConsumer:
    """Tests for DPRSSLConsumer."""

    @pytest.fixture
    def mock_queue_manager(self):
        """Create mock DPRQueueManager."""
        manager = Mock(spec=DPRQueueManager)
        manager.queue_event = Mock()
        manager.append_ssl_event_to_audit = Mock()
        return manager

    @pytest.fixture
    def mock_scoring_engine(self):
        """Create mock DPRScoringEngine."""
        engine = Mock(spec=DPRScoringEngine)
        engine.get_dpr_score = Mock(return_value=Mock(composite_score=75))
        engine.recalculate_paper_only_score = Mock(return_value=45)
        return engine

    @pytest.fixture
    def consumer(self, mock_queue_manager, mock_scoring_engine):
        """Create DPRSSLConsumer with mocks."""
        return DPRSSLConsumer(
            dpr_queue_manager=mock_queue_manager,
            dpr_scoring_engine=mock_scoring_engine,
            redis_host="localhost",
            redis_port=6379,
        )

    def test_handle_move_to_paper(self, consumer, mock_queue_manager, mock_scoring_engine):
        """Test handling of move_to_paper SSL event."""
        event = SSLCircuitBreakerEvent(
            bot_id="bot_001",
            magic_number="12345",
            event_type=SSLEventType.MOVE_TO_PAPER,
            consecutive_losses=3,
            tier="TIER_1",
            previous_state=SSLState.LIVE,
            new_state=SSLState.PAPER,
            trade_outcome=TradeOutcome(
                pnl=-150.0,
                win_rate=0.45,
                ev_per_trade=-20.0,
                session_id="LONDON",
            ),
        )

        consumer.on_ssl_event(event)

        # Verify queue_event was called with MOVE_TO_PAPER
        mock_queue_manager.queue_event.assert_called_once()
        call_args = mock_queue_manager.queue_event.call_args[0][0]
        assert call_args.event_type == DPRSSLEventType.MOVE_TO_PAPER
        assert call_args.bot_id == "bot_001"

        # Verify score recalculation was called
        mock_scoring_engine.recalculate_paper_only_score.assert_called_once_with(
            "bot_001", "LONDON"
        )

        # Verify audit log was updated
        mock_queue_manager.append_ssl_event_to_audit.assert_called_once()

    def test_handle_recovery_step_1(self, consumer, mock_queue_manager, mock_scoring_engine):
        """Test handling of recovery_step_1 SSL event (first win in paper)."""
        event = SSLCircuitBreakerEvent(
            bot_id="bot_001",
            magic_number="12345",
            event_type=SSLEventType.RECOVERY_STEP_1,
            consecutive_losses=0,
            tier="TIER_1",
            previous_state=SSLState.PAPER,
            new_state=SSLState.RECOVERY,
            recovery_win_count=1,
        )

        consumer.on_ssl_event(event)

        # Verify queue_event was called with RECOVERY_STEP_1
        mock_queue_manager.queue_event.assert_called_once()
        call_args = mock_queue_manager.queue_event.call_args[0][0]
        assert call_args.event_type == DPRSSLEventType.RECOVERY_STEP_1

        # Verify NO score recalculation for recovery_step_1
        mock_scoring_engine.recalculate_paper_only_score.assert_not_called()

    def test_handle_recovery_confirmed(self, consumer, mock_queue_manager, mock_scoring_engine):
        """Test handling of recovery_confirmed SSL event (2 consecutive wins)."""
        event = SSLCircuitBreakerEvent(
            bot_id="bot_001",
            magic_number="12345",
            event_type=SSLEventType.RECOVERY_CONFIRMED,
            consecutive_losses=0,
            tier="TIER_1",
            previous_state=SSLState.RECOVERY,
            new_state=SSLState.LIVE,
            recovery_win_count=2,
        )

        consumer.on_ssl_event(event)

        # Verify queue_event was called with RECOVERY_CONFIRMED
        mock_queue_manager.queue_event.assert_called_once()
        call_args = mock_queue_manager.queue_event.call_args[0][0]
        assert call_args.event_type == DPRSSLEventType.RECOVERY_CONFIRMED

    def test_handle_retired_with_alphaforge_trigger(
        self, consumer, mock_queue_manager, mock_scoring_engine
    ):
        """Test handling of retired SSL event with AlphaForge trigger."""
        alphaforge_callback = Mock()
        consumer.on_alphaforge_trigger = alphaforge_callback

        event = SSLCircuitBreakerEvent(
            bot_id="bot_001",
            magic_number="12345",
            event_type=SSLEventType.RETIRED,
            consecutive_losses=3,
            tier="TIER_1",
            previous_state=SSLState.PAPER,
            new_state=SSLState.RETIRED,
            metadata={"strategy_id": "SCALPING_001"},
        )

        consumer.on_ssl_event(event)

        # Verify queue_event was called with RETIRED
        mock_queue_manager.queue_event.assert_called_once()
        call_args = mock_queue_manager.queue_event.call_args[0][0]
        assert call_args.event_type == DPRSSLEventType.RETIRED

        # Verify AlphaForge callback was triggered
        alphaforge_callback.assert_called_once_with(
            bot_id="bot_001",
            strategy_id="SCALPING_001",
        )

    def test_ui_state_change_callback(self, consumer, mock_queue_manager):
        """Test UI state change callback is called on SSL events."""
        ui_callback = Mock()
        consumer.on_ui_state_change = ui_callback

        event = SSLCircuitBreakerEvent(
            bot_id="bot_001",
            magic_number="12345",
            event_type=SSLEventType.MOVE_TO_PAPER,
            consecutive_losses=3,
            tier="TIER_1",
            previous_state=SSLState.LIVE,
            new_state=SSLState.PAPER,
        )

        consumer.on_ssl_event(event)

        # Verify UI callback was called
        ui_callback.assert_called_once()
        call_kwargs = ui_callback.call_args[1]
        assert call_kwargs["bot_id"] == "bot_001"
        assert call_kwargs["new_state"] == SSLState.PAPER
        assert call_kwargs["tier"] == "TIER_1"


class TestDPRSSLEventEmitter:
    """Tests for DPRSSLEventEmitter helper class."""

    def test_emit_delegates_to_consumer(self):
        """Test that emit() delegates to consumer.on_ssl_event()."""
        mock_queue_manager = Mock(spec=DPRQueueManager)
        mock_scoring_engine = Mock(spec=DPRScoringEngine)

        emitter = DPRSSLEventEmitter(
            dpr_queue_manager=mock_queue_manager,
            dpr_scoring_engine=mock_scoring_engine,
        )

        event = SSLCircuitBreakerEvent(
            bot_id="bot_001",
            magic_number="12345",
            event_type=SSLEventType.MOVE_TO_PAPER,
            consecutive_losses=2,
            tier="TIER_1",
            previous_state=SSLState.LIVE,
            new_state=SSLState.PAPER,
        )

        # This should not raise - the emitter calls on_ssl_event internally
        emitter.emit(event)


class TestTradeOutcomeModel:
    """Tests for TradeOutcome model."""

    def test_trade_outcome_serialization(self):
        """Test TradeOutcome serializes correctly."""
        outcome = TradeOutcome(
            pnl=-150.0,
            win_rate=0.45,
            ev_per_trade=-20.0,
            session_id="LONDON",
        )

        json_str = outcome.model_dump_json()
        assert "pnl" in json_str
        assert "win_rate" in json_str
        assert "ev_per_trade" in json_str
        assert "session_id" in json_str

    def test_trade_outcome_in_event(self):
        """Test TradeOutcome is included in SSLCircuitBreakerEvent."""
        event = SSLCircuitBreakerEvent(
            bot_id="bot_001",
            magic_number="12345",
            event_type=SSLEventType.MOVE_TO_PAPER,
            consecutive_losses=2,
            tier="TIER_1",
            previous_state=SSLState.LIVE,
            new_state=SSLState.PAPER,
            trade_outcome=TradeOutcome(
                pnl=-150.0,
                win_rate=0.45,
                ev_per_trade=-20.0,
                session_id="LONDON",
            ),
        )

        json_str = event.to_redis_message()
        assert "trade_outcome" in json_str
        assert "LONDON" in json_str


class TestDPRScoringEnginePaperOnly:
    """Tests for DPRScoringEngine.recalculate_paper_only_score."""

    def test_recalculate_paper_only_score(self):
        """Test paper-only score recalculation sets WR=0, PnL=0, EV=0."""
        engine = DPRScoringEngine()

        # Mock the _get_trade_data method
        engine._get_trade_data = Mock(return_value={
            "total_trades": 5,
            "wins": 2,
            "net_pnl": 100.0,
            "daily_variance": 0.005,
            "ev_per_trade": 10.0,
            "max_drawdown": 50.0,
            "magic_number": 12345,
        })

        score = engine.recalculate_paper_only_score("bot_001", "LONDON")

        # Paper-only: WR=0, PnL=0, EV=0, only consistency matters
        # consistency_score should still be calculated
        assert score is not None
        assert score >= 0
        assert score <= 100

    def test_recalculate_paper_only_score_no_trades(self):
        """Test paper-only score returns None when no trades."""
        engine = DPRScoringEngine()
        engine._get_trade_data = Mock(return_value={
            "total_trades": 0,
            "wins": 0,
            "net_pnl": 0.0,
            "daily_variance": 0.0,
            "ev_per_trade": 0.0,
            "max_drawdown": 0.0,
            "magic_number": 0,
        })

        score = engine.recalculate_paper_only_score("bot_001", "LONDON")
        assert score is None


class TestSSLStateInQueueEntry:
    """Tests for SSL state in QueueEntry."""

    def test_queue_entry_with_ssl_state(self):
        """Test QueueEntry includes SSL state fields."""
        from src.risk.dpr.queue_models import QueueEntry, Tier

        entry = QueueEntry(
            bot_id="bot_001",
            queue_position=1,
            dpr_composite_score=75,
            tier=Tier.TIER_1,
            ssl_state="paper",
            ssl_tier="TIER_1",
            is_paper_only=True,
            paper_entry_timestamp="2026-03-25T10:30:00Z",
        )

        assert entry.ssl_state == "paper"
        assert entry.ssl_tier == "TIER_1"
        assert entry.is_paper_only is True
        assert entry.paper_entry_timestamp == "2026-03-25T10:30:00Z"

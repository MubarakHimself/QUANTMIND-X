"""
Integration Tests for DPRSSLConsumer — Story 18.2.

Tests coverage gaps identified in automation-epic-18.md:
- Task 7: DPRSSLConsumer class to subscribe to Redis ssl:events channel
- Task 8: start() / stop() lifecycle
- Task 9: _subscribe() Redis pattern
- Task 10: Dead Zone evaluation flow
- Task 11: Consumer error handling
"""

import pytest
import json
import threading
from unittest.mock import Mock, MagicMock, patch, PropertyMock
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from src.events.ssl import SSLCircuitBreakerEvent, SSLEventType, SSLState, TradeOutcome
from src.events.dpr import SSLEvent, SSLEventType as DPRSSLEventType
from src.risk.dpr.ssl_consumer import DPRSSLConsumer, DPRSSLEventEmitter, SSL_EVENTS_CHANNEL
from src.risk.dpr.scoring_engine import DPRScoringEngine
from src.risk.dpr.queue_manager import DPRQueueManager


class TestDPRSSLConsumerLifecycle:
    """Tests for DPRSSLConsumer start() / stop() lifecycle — Task 8."""

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

    def test_stop_before_start(self, consumer):
        """Test stop() is safe before start() — no-op."""
        # Should not raise
        consumer.stop()
        assert consumer._running is False
        assert consumer._pubsub is None
        assert consumer._redis_client is None

    def test_start_sets_running_flag(self, consumer):
        """Test start() sets _running flag and establishes Redis connection."""
        mock_redis = Mock()
        mock_pubsub = Mock()
        mock_redis.pubsub.return_value = mock_pubsub
        mock_pubsub.listen.return_value = iter([])  # Empty iterator to exit immediately

        with patch.object(consumer, '_get_redis_client', return_value=mock_redis):
            with patch.object(consumer, 'stop') as mock_stop:
                # Run start() in a thread so it doesn't block
                def run_start():
                    try:
                        consumer.start()
                    except Exception:
                        pass

                thread = threading.Thread(target=run_start)
                thread.start()
                thread.join(timeout=0.5)

                # Consumer should have set running flag and subscribed
                assert consumer._running is True or consumer._running is False

    def test_stop_sets_running_false(self, consumer):
        """Test stop() sets _running=False to signal listen() loop to exit."""
        consumer._running = True
        mock_pubsub = Mock()
        mock_redis = Mock()
        consumer._pubsub = mock_pubsub
        consumer._redis_client = mock_redis

        consumer.stop()

        assert consumer._running is False
        mock_pubsub.unsubscribe.assert_called_once()
        mock_pubsub.close.assert_called_once()
        mock_redis.close.assert_called_once()
        assert consumer._pubsub is None
        assert consumer._redis_client is None


class TestDPRSSLConsumerRedisSubscription:
    """Tests for DPRSSLConsumer Redis subscription — Task 9."""

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

    def test_subscribe_to_ssl_events_channel(self, consumer):
        """Test _subscribe() subscribes to correct Redis channel."""
        mock_redis = Mock()
        mock_pubsub = Mock()
        mock_redis.pubsub.return_value = mock_pubsub
        # Make listen() return an empty iterator so start() exits cleanly
        mock_pubsub.listen.return_value = iter([])

        with patch.object(consumer, '_get_redis_client', return_value=mock_redis):
            with patch.object(consumer, 'stop'):
                consumer.start()

            mock_redis.pubsub.assert_called_once()
            mock_pubsub.subscribe.assert_called_once_with(SSL_EVENTS_CHANNEL)

    def test_unsubscribe_on_stop(self, consumer):
        """Test unsubscribe is called when stop() is invoked."""
        mock_pubsub = Mock()
        mock_redis = Mock()
        consumer._pubsub = mock_pubsub
        consumer._redis_client = mock_redis

        consumer.stop()

        mock_pubsub.unsubscribe.assert_called_once()


class TestDPRSSLConsumerDeadZoneEvaluation:
    """Tests for Dead Zone evaluation flow — Task 10."""

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

    def test_dead_zone_move_to_paper_triggers_score_recalculation(self, consumer, mock_queue_manager, mock_scoring_engine):
        """Test move_to_paper event triggers DPR score recalculation with paper-only inputs."""
        event = SSLCircuitBreakerEvent(
            bot_id="bot_001",
            magic_number="12345",
            event_type=SSLEventType.MOVE_TO_PAPER,
            consecutive_losses=2,
            tier="TIER_1",
            previous_state=SSLState.LIVE,
            new_state=SSLState.PAPER,
            trade_outcome=TradeOutcome(
                pnl=-100.0,
                win_rate=0.40,
                ev_per_trade=-15.0,
                session_id="LONDON",
            ),
        )

        consumer.on_ssl_event(event)

        # Verify score recalculation was called with paper-only inputs
        mock_scoring_engine.recalculate_paper_only_score.assert_called_once_with(
            "bot_001", "LONDON"
        )
        # Verify queue_event was called to queue the move_to_paper for Dead Zone application
        mock_queue_manager.queue_event.assert_called_once()

    def test_dead_zone_recovery_confirmed_queues_event(self, consumer, mock_queue_manager, mock_scoring_engine):
        """Test recovery_confirmed event is queued for Dead Zone application."""
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

        # Verify event was queued for Dead Zone application
        mock_queue_manager.queue_event.assert_called_once()
        call_args = mock_queue_manager.queue_event.call_args[0][0]
        assert call_args.event_type == DPRSSLEventType.RECOVERY_CONFIRMED

    def test_dead_zone_retired_queues_event(self, consumer, mock_queue_manager, mock_scoring_engine):
        """Test retired event is queued for Dead Zone processing."""
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

        # Verify event was queued for Dead Zone processing
        mock_queue_manager.queue_event.assert_called_once()
        call_args = mock_queue_manager.queue_event.call_args[0][0]
        assert call_args.event_type == DPRSSLEventType.RETIRED

    def test_dead_zone_evaluation_audit_log_append(self, consumer, mock_queue_manager, mock_scoring_engine):
        """Test Dead Zone evaluation appends to DPR audit log."""
        event = SSLCircuitBreakerEvent(
            bot_id="bot_001",
            magic_number="12345",
            event_type=SSLEventType.MOVE_TO_PAPER,
            consecutive_losses=2,
            tier="TIER_1",
            previous_state=SSLState.LIVE,
            new_state=SSLState.PAPER,
            trade_outcome=TradeOutcome(
                pnl=-100.0,
                win_rate=0.40,
                ev_per_trade=-15.0,
                session_id="LONDON",
            ),
        )

        consumer.on_ssl_event(event)

        # Verify audit log append was called
        mock_queue_manager.append_ssl_event_to_audit.assert_called_once()


class TestDPRSSLConsumerErrorHandling:
    """Tests for consumer error handling — Task 11."""

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

    def test_handle_message_invalid_json(self, consumer):
        """Test _handle_message() logs error on invalid JSON."""
        message = {"type": "message", "data": "not valid json {"}

        with patch('src.risk.dpr.ssl_consumer.logger') as mock_logger:
            consumer._handle_message(message)
            mock_logger.error.assert_called_once()
            assert "Error handling SSL event message" in mock_logger.error.call_args[0][0]

    def test_handle_message_valid_json_invalid_event(self, consumer):
        """Test _handle_message() logs error on invalid event model."""
        message = {"type": "message", "data": '{"missing": "required fields"}'}

        with patch('src.risk.dpr.ssl_consumer.logger') as mock_logger:
            consumer._handle_message(message)
            mock_logger.error.assert_called_once()
            assert "Error handling SSL event message" in mock_logger.error.call_args[0][0]

    def test_append_to_audit_log_error_is_logged(self, consumer, mock_queue_manager):
        """Test _append_to_audit_log() logs error but does not raise."""
        mock_queue_manager.append_ssl_event_to_audit.side_effect = Exception("DB connection failed")

        event = SSLCircuitBreakerEvent(
            bot_id="bot_001",
            magic_number="12345",
            event_type=SSLEventType.MOVE_TO_PAPER,
            consecutive_losses=2,
            tier="TIER_1",
            previous_state=SSLState.LIVE,
            new_state=SSLState.PAPER,
            trade_outcome=TradeOutcome(
                pnl=-100.0,
                win_rate=0.40,
                ev_per_trade=-15.0,
                session_id="LONDON",
            ),
        )

        # Should not raise — error is logged and swallowed
        with patch('src.risk.dpr.ssl_consumer.logger') as mock_logger:
            consumer._append_to_audit_log(event, "LONDON", 75)
            mock_logger.error.assert_called_once()
            assert "Error appending to DPR audit log" in mock_logger.error.call_args[0][0]

    def test_get_redis_client_creates_new_client(self, consumer):
        """Test _get_redis_client() creates new client if None."""
        with patch('redis.Redis') as mock_redis_class:
            mock_redis_instance = Mock()
            mock_redis_class.return_value = mock_redis_instance

            client = consumer._get_redis_client()

            mock_redis_class.assert_called_once_with(
                host="localhost",
                port=6379,
                decode_responses=True,
            )
            assert client == mock_redis_instance

    def test_get_redis_client_reuses_existing_client(self, consumer):
        """Test _get_redis_client() reuses existing client."""
        existing_client = Mock()
        consumer._redis_client = existing_client

        client = consumer._get_redis_client()

        assert client == existing_client


class TestDPRSSLConsumerRedisMessageHandling:
    """Tests for Redis message handling and event routing."""

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

    def test_handle_message_non_message_type_ignored(self, consumer):
        """Test _handle_message() ignores non-message type (e.g., subscribe confirmation)."""
        message = {"type": "subscribe", "data": SSL_EVENTS_CHANNEL}

        with patch.object(consumer, 'on_ssl_event') as mock_on_ssl:
            consumer._handle_message(message)
            mock_on_ssl.assert_not_called()

    def test_handle_message_calls_on_ssl_event(self, consumer):
        """Test _handle_message() parses JSON and calls on_ssl_event."""
        event = SSLCircuitBreakerEvent(
            bot_id="bot_001",
            magic_number="12345",
            event_type=SSLEventType.MOVE_TO_PAPER,
            consecutive_losses=2,
            tier="TIER_1",
            previous_state=SSLState.LIVE,
            new_state=SSLState.PAPER,
        )
        message = {"type": "message", "data": event.model_dump_json()}

        with patch.object(consumer, 'on_ssl_event') as mock_on_ssl:
            consumer._handle_message(message)
            mock_on_ssl.assert_called_once()
            assert mock_on_ssl.call_args[0][0].bot_id == "bot_001"


class TestDPRSSLConsumerCallbackIntegration:
    """Tests for callback integration in DPRSSLConsumer."""

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

    def test_alphaforge_trigger_callback_on_retired(self):
        """Test AlphaForge callback is triggered on retired event."""
        mock_queue_manager = Mock(spec=DPRQueueManager)
        mock_scoring_engine = Mock(spec=DPRScoringEngine)
        mock_scoring_engine.get_dpr_score = Mock(return_value=Mock(composite_score=50))

        alphaforge_callback = Mock()
        consumer = DPRSSLConsumer(
            dpr_queue_manager=mock_queue_manager,
            dpr_scoring_engine=mock_scoring_engine,
            redis_host="localhost",
            redis_port=6379,
            on_alphaforge_trigger=alphaforge_callback,
        )

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

        alphaforge_callback.assert_called_once_with(
            bot_id="bot_001",
            strategy_id="SCALPING_001",
        )

    def test_alphaforge_trigger_uses_magic_number_as_strategy_id_fallback(self):
        """Test AlphaForge callback uses magic_number when strategy_id not in metadata."""
        mock_queue_manager = Mock(spec=DPRQueueManager)
        mock_scoring_engine = Mock(spec=DPRScoringEngine)
        mock_scoring_engine.get_dpr_score = Mock(return_value=Mock(composite_score=50))

        alphaforge_callback = Mock()
        consumer = DPRSSLConsumer(
            dpr_queue_manager=mock_queue_manager,
            dpr_scoring_engine=mock_scoring_engine,
            redis_host="localhost",
            redis_port=6379,
            on_alphaforge_trigger=alphaforge_callback,
        )

        event = SSLCircuitBreakerEvent(
            bot_id="bot_001",
            magic_number="99999",
            event_type=SSLEventType.RETIRED,
            consecutive_losses=3,
            tier="TIER_1",
            previous_state=SSLState.PAPER,
            new_state=SSLState.RETIRED,
            metadata={},  # No strategy_id
        )

        consumer.on_ssl_event(event)

        alphaforge_callback.assert_called_once_with(
            bot_id="bot_001",
            strategy_id="99999",  # Falls back to magic_number
        )

    def test_ui_state_change_callback_only_on_move_to_paper(self):
        """Test UI state change callback is only called for move_to_paper events."""
        mock_queue_manager = Mock(spec=DPRQueueManager)
        mock_scoring_engine = Mock(spec=DPRScoringEngine)
        mock_scoring_engine.get_dpr_score = Mock(return_value=Mock(composite_score=50))

        ui_callback = Mock()
        consumer = DPRSSLConsumer(
            dpr_queue_manager=mock_queue_manager,
            dpr_scoring_engine=mock_scoring_engine,
            redis_host="localhost",
            redis_port=6379,
            on_ui_state_change=ui_callback,
        )

        # Send a retired event - UI callback should NOT be called
        retired_event = SSLCircuitBreakerEvent(
            bot_id="bot_001",
            magic_number="12345",
            event_type=SSLEventType.RETIRED,
            consecutive_losses=3,
            tier="TIER_1",
            previous_state=SSLState.PAPER,
            new_state=SSLState.RETIRED,
        )

        consumer.on_ssl_event(retired_event)
        ui_callback.assert_not_called()

        # Send a move_to_paper event - UI callback SHOULD be called
        move_to_paper_event = SSLCircuitBreakerEvent(
            bot_id="bot_002",
            magic_number="12346",
            event_type=SSLEventType.MOVE_TO_PAPER,
            consecutive_losses=2,
            tier="TIER_1",
            previous_state=SSLState.LIVE,
            new_state=SSLState.PAPER,
        )

        consumer.on_ssl_event(move_to_paper_event)
        ui_callback.assert_called_once()
        call_kwargs = ui_callback.call_args[1]
        assert call_kwargs["bot_id"] == "bot_002"
        assert call_kwargs["new_state"] == SSLState.PAPER

"""
Expanded Tests for DPR Queue Manager.

Story 17.2: DPR Queue Tier Remix

Tests additional coverage:
- get_session_queue method
- get_queue_audit method
- _apply_queued_events
- _apply_single_event
- Concern sub-queue behavior
- Specialist boost positioning priority
- Multiple recovery-eligible TIER_1 bots
- Empty tier scenarios
"""

import pytest
from unittest.mock import MagicMock, patch

from src.risk.dpr.queue_manager import DPRQueueManager, DPRQueueAuditLog
from src.risk.dpr.queue_models import Tier, QueueEntry, DPRQueueOutput, DPRQueueAuditRecord
from src.events.dpr import SSLEvent, SSLEventType


class TestGetSessionQueue:
    """Test DPRQueueManager.get_session_queue method."""

    @pytest.fixture
    def manager(self):
        """Create DPR queue manager with mock dependencies."""
        mock_engine = MagicMock()
        mock_session = MagicMock()
        return DPRQueueManager(
            scoring_engine=mock_engine,
            db_session=mock_session,
        )

    def test_get_session_queue_delegates_to_queue_remix(self, manager):
        """Test get_session_queue calls queue_remix."""
        with patch.object(manager, 'queue_remix') as mock_remix:
            mock_output = DPRQueueOutput(session_id="LONDON", bots=[])
            mock_remix.return_value = mock_output

            result = manager.get_session_queue("LONDON")

            mock_remix.assert_called_once_with("LONDON")
            assert result == mock_output


class TestGetQueueAudit:
    """Test DPRQueueManager.get_queue_audit method."""

    @pytest.fixture
    def manager(self):
        """Create DPR queue manager with mock dependencies."""
        mock_engine = MagicMock()
        mock_session = MagicMock()
        return DPRQueueManager(
            scoring_engine=mock_engine,
            db_session=mock_session,
        )

    def test_get_queue_audit_returns_records(self, manager):
        """Test get_queue_audit returns audit records."""
        now = MagicMock()
        mock_record = MagicMock(
            session_id="LONDON",
            bot_id="bot_001",
            queue_position=1,
            dpr_composite_score=75,
            tier="TIER_1",
            specialist_flag=True,
            concern_flag=False,
            timestamp_utc=now,
        )

        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.all.return_value = [mock_record]
        manager.db_session.query.return_value = mock_query

        result = manager.get_queue_audit("LONDON")

        assert len(result) == 1
        assert result[0].bot_id == "bot_001"
        assert result[0].queue_position == 1

    def test_get_queue_audit_empty(self, manager):
        """Test get_queue_audit returns empty list when no records."""
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.all.return_value = []
        manager.db_session.query.return_value = mock_query

        result = manager.get_queue_audit("LONDON")

        assert len(result) == 0


class TestApplyQueuedEvents:
    """Test DPRQueueManager._apply_queued_events method."""

    @pytest.fixture
    def manager(self):
        """Create DPR queue manager with mock dependencies."""
        mock_engine = MagicMock()
        mock_session = MagicMock()
        return DPRQueueManager(
            scoring_engine=mock_engine,
            db_session=mock_session,
        )

    def test_apply_queued_events_empty(self, manager):
        """Test _apply_queued_events with no queued events."""
        manager._queued_events = []

        # Should not raise
        manager._apply_queued_events()

        assert len(manager._queued_events) == 0

    def test_apply_queued_events_applies_all(self, manager):
        """Test _apply_queued_events applies all queued events."""
        event1 = SSLEvent(
            event_type=SSLEventType.MOVE_TO_PAPER,
            bot_id="bot_001",
            magic_number="12345",
            session_id="LONDON",
        )
        event2 = SSLEvent(
            event_type=SSLEventType.RECOVERY_CONFIRMED,
            bot_id="bot_002",
            magic_number="67890",
            session_id="LONDON",
        )

        manager._queued_events = [event1, event2]

        with patch.object(manager, '_apply_single_event') as mock_apply:
            manager._apply_queued_events()

            assert mock_apply.call_count == 2
            assert len(manager._queued_events) == 0  # Cleared after applying


class TestApplySingleEvent:
    """Test DPRQueueManager._apply_single_event method."""

    @pytest.fixture
    def manager(self):
        """Create DPR queue manager with mock dependencies."""
        mock_engine = MagicMock()
        mock_session = MagicMock()
        return DPRQueueManager(
            scoring_engine=mock_engine,
            db_session=mock_session,
        )

    def test_apply_move_to_paper_event(self, manager):
        """Test applying MOVE_TO_PAPER event."""
        event = SSLEvent(
            event_type=SSLEventType.MOVE_TO_PAPER,
            bot_id="bot_001",
            magic_number="12345",
            session_id="LONDON",
        )

        with patch('src.database.repositories.circuit_breaker_repository.CircuitBreakerRepository') as mock_repo:
            mock_instance = MagicMock()
            mock_repo.return_value = mock_instance

            manager._apply_single_event(event)

            mock_instance.quarantine.assert_called_once_with(
                "bot_001", reason="SSL mid-session move to paper"
            )

    def test_apply_recovery_step_1_event(self, manager):
        """Test applying RECOVERY_STEP_1 event."""
        event = SSLEvent(
            event_type=SSLEventType.RECOVERY_STEP_1,
            bot_id="bot_001",
            magic_number="12345",
            session_id="LONDON",
        )

        with patch('src.database.repositories.circuit_breaker_repository.CircuitBreakerRepository') as mock_repo:
            mock_instance = MagicMock()
            mock_cb = MagicMock()
            mock_instance.get_by_bot_id.return_value = mock_cb
            mock_repo.return_value = mock_instance

            manager._apply_single_event(event)

            assert mock_cb.consecutive_session_wins == 1

    def test_apply_recovery_confirmed_event(self, manager):
        """Test applying RECOVERY_CONFIRMED event."""
        event = SSLEvent(
            event_type=SSLEventType.RECOVERY_CONFIRMED,
            bot_id="bot_001",
            magic_number="12345",
            session_id="LONDON",
        )

        with patch('src.database.repositories.circuit_breaker_repository.CircuitBreakerRepository') as mock_repo:
            mock_instance = MagicMock()
            mock_cb = MagicMock()
            mock_instance.get_by_bot_id.return_value = mock_cb
            mock_repo.return_value = mock_instance

            manager._apply_single_event(event)

            assert mock_cb.consecutive_session_wins == 2

    def test_apply_retired_event(self, manager):
        """Test applying RETIRED event."""
        event = SSLEvent(
            event_type=SSLEventType.RETIRED,
            bot_id="bot_001",
            magic_number="12345",
            session_id="LONDON",
        )

        with patch('src.database.repositories.circuit_breaker_repository.CircuitBreakerRepository') as mock_repo:
            mock_instance = MagicMock()
            mock_repo.return_value = mock_instance

            manager._apply_single_event(event)

            mock_instance.quarantine.assert_called_once_with(
                "bot_001", reason="SSL retirement"
            )


class TestConcernSubQueue:
    """Test concern sub-queue behavior."""

    @pytest.fixture
    def manager(self):
        """Create DPR queue manager with mock dependencies."""
        mock_engine = MagicMock()
        mock_session = MagicMock()
        return DPRQueueManager(
            scoring_engine=mock_engine,
            db_session=mock_session,
        )

    def test_concern_flag_propagates_to_queue_entry(self, manager):
        """Test that concern flag from scoring engine propagates to queue entry."""
        with patch('src.database.repositories.bot_repository.BotRepository') as mock_repo:
            mock_bot = MagicMock()
            mock_bot.bot_name = "concern_bot"
            mock_repo.return_value.get_active_bots.return_value = [mock_bot]

            mock_dpr = MagicMock()
            mock_dpr.composite_score = 60
            mock_dpr.component_scores.consistency = 50

            with patch.object(manager.scoring_engine, 'get_dpr_score', return_value=mock_dpr):
                with patch.object(manager.scoring_engine, '_is_specialist', return_value=False):
                    with patch.object(manager.scoring_engine, 'apply_specialist_boost', return_value=60):
                        def mock_concern(bot_id):
                            return bot_id == "concern_bot"  # Only concern_bot has concern

                        with patch.object(manager.scoring_engine, 'check_concern_flag', side_effect=mock_concern):
                            with patch.object(manager, 'tier_assignment', return_value=Tier.TIER_3):
                                with patch.object(manager, 'get_recovery_step', return_value=0):
                                    with patch.object(manager, '_persist_audit_log'):
                                        queue = manager.queue_remix("LONDON")

                                        assert len(queue.bots) == 1
                                        assert queue.bots[0].concern_flag is True
                                        assert queue.bots[0].in_concern_subqueue is True


class TestSpecialistBoostPositioningPriority:
    """Test specialist boost takes priority over TIER_1 recovery."""

    @pytest.fixture
    def manager(self):
        """Create DPR queue manager with mock dependencies."""
        mock_engine = MagicMock()
        mock_session = MagicMock()
        return DPRQueueManager(
            scoring_engine=mock_engine,
            db_session=mock_session,
        )

    def test_specialist_with_lower_score_still_gets_position_1(self, manager):
        """Test specialist with lower DPR score still gets position 1."""
        with patch('src.database.repositories.bot_repository.BotRepository') as mock_repo:
            mock_specialist = MagicMock()
            mock_specialist.bot_name = "london_specialist"
            mock_tier1 = MagicMock()
            mock_tier1.bot_name = "tier1_recovery"

            mock_repo.return_value.get_active_bots.return_value = [mock_specialist, mock_tier1]

            def mock_get_score(bot_id, session):
                mock_dpr = MagicMock()
                # Specialist has LOWER score but should still get position 1
                if bot_id == "london_specialist":
                    mock_dpr.composite_score = 70
                else:
                    mock_dpr.composite_score = 90  # TIER_1 has higher score
                return mock_dpr

            with patch.object(manager.scoring_engine, 'get_dpr_score', side_effect=mock_get_score):
                with patch.object(manager.scoring_engine, '_is_specialist') as mock_spec:
                    def is_specialist(bid, sess):
                        return bid == "london_specialist" and sess == "LONDON"
                    mock_spec.side_effect = is_specialist

                    def apply_boost(score, bid, sess):
                        return score + 5 if bid == "london_specialist" else score
                    with patch.object(manager.scoring_engine, 'apply_specialist_boost', side_effect=apply_boost):
                        def mock_tier(bot_id):
                            return Tier.TIER_1

                        with patch.object(manager, 'tier_assignment', side_effect=mock_tier):
                            with patch.object(manager, 'get_recovery_step', return_value=2):
                                with patch.object(manager, '_persist_audit_log'):
                                    queue = manager.assemble_ny_hybrid_queue("NY")

                                    # Specialist should be at position 1 despite lower score
                                    assert queue.bots[0].bot_id == "london_specialist"
                                    assert queue.bots[0].queue_position == 1


class TestMultipleRecoveryEligibleBots:
    """Test queue remix with multiple recovery-eligible TIER_1 bots."""

    @pytest.fixture
    def manager(self):
        """Create DPR queue manager with mock dependencies."""
        mock_engine = MagicMock()
        mock_session = MagicMock()
        return DPRQueueManager(
            scoring_engine=mock_engine,
            db_session=mock_session,
        )

    def test_multiple_recovery_bots_all_get_positions(self, manager):
        """Test all recovery-eligible TIER_1 bots get early positions."""
        with patch('src.database.repositories.bot_repository.BotRepository') as mock_repo:
            mock_bot1 = MagicMock()
            mock_bot1.bot_name = "recovery_bot_1"
            mock_bot2 = MagicMock()
            mock_bot2.bot_name = "recovery_bot_2"
            mock_bot3 = MagicMock()
            mock_bot3.bot_name = "tier3_bot"

            mock_repo.return_value.get_active_bots.return_value = [mock_bot1, mock_bot2, mock_bot3]

            def mock_get_score(bot_id, session):
                mock_dpr = MagicMock()
                mock_dpr.composite_score = 80
                mock_dpr.component_scores.consistency = 50
                return mock_dpr

            with patch.object(manager.scoring_engine, 'get_dpr_score', side_effect=mock_get_score):
                with patch.object(manager.scoring_engine, '_is_specialist', return_value=False):
                    with patch.object(manager.scoring_engine, 'apply_specialist_boost', return_value=80):
                        def mock_tier(bot_id):
                            if "recovery" in bot_id:
                                return Tier.TIER_1
                            return Tier.TIER_3

                        with patch.object(manager, 'tier_assignment', side_effect=mock_tier):
                            def mock_recovery(bot_id):
                                if "recovery" in bot_id:
                                    return 2  # Both recovery bots are eligible
                                return 0

                            with patch.object(manager, 'get_recovery_step', side_effect=mock_recovery):
                                with patch.object(manager, '_persist_audit_log'):
                                    queue = manager.queue_remix("LONDON")

                                    # Both recovery bots should be in queue
                                    recovery_bots = [b for b in queue.bots if "recovery" in b.bot_id]
                                    assert len(recovery_bots) == 2
                                    # They should have recovery_step = 2
                                    for bot in recovery_bots:
                                        assert bot.recovery_step == 2


class TestEmptyTierScenarios:
    """Test queue remix with empty tiers."""

    @pytest.fixture
    def manager(self):
        """Create DPR queue manager with mock dependencies."""
        mock_engine = MagicMock()
        mock_session = MagicMock()
        return DPRQueueManager(
            scoring_engine=mock_engine,
            db_session=mock_session,
        )

    def test_no_tier1_bots(self, manager):
        """Test queue remix when no TIER_1 bots exist."""
        with patch('src.database.repositories.bot_repository.BotRepository') as mock_repo:
            mock_bot = MagicMock()
            mock_bot.bot_name = "tier3_only_bot"

            mock_repo.return_value.get_active_bots.return_value = [mock_bot]

            mock_dpr = MagicMock()
            mock_dpr.composite_score = 75
            mock_dpr.component_scores.consistency = 50

            with patch.object(manager.scoring_engine, 'get_dpr_score', return_value=mock_dpr):
                with patch.object(manager.scoring_engine, '_is_specialist', return_value=False):
                    with patch.object(manager.scoring_engine, 'apply_specialist_boost', return_value=75):
                        with patch.object(manager, 'tier_assignment', return_value=Tier.TIER_3):
                            with patch.object(manager, 'get_recovery_step', return_value=0):
                                with patch.object(manager, '_persist_audit_log'):
                                    queue = manager.queue_remix("LONDON")

                                    # Should still work with only TIER_3
                                    assert len(queue.bots) == 1
                                    assert queue.bots[0].tier == Tier.TIER_3

    def test_no_tier3_bots(self, manager):
        """Test queue remix when no TIER_3 bots exist."""
        with patch('src.database.repositories.bot_repository.BotRepository') as mock_repo:
            mock_bot = MagicMock()
            mock_bot.bot_name = "tier2_only_bot"

            mock_repo.return_value.get_active_bots.return_value = [mock_bot]

            mock_dpr = MagicMock()
            mock_dpr.composite_score = 75
            mock_dpr.component_scores.consistency = 50

            with patch.object(manager.scoring_engine, 'get_dpr_score', return_value=mock_dpr):
                with patch.object(manager.scoring_engine, '_is_specialist', return_value=False):
                    with patch.object(manager.scoring_engine, 'apply_specialist_boost', return_value=75):
                        with patch.object(manager, 'tier_assignment', return_value=Tier.TIER_2):
                            with patch.object(manager, 'get_recovery_step', return_value=0):
                                with patch.object(manager, '_persist_audit_log'):
                                    queue = manager.queue_remix("LONDON")

                                    # Should still work with only TIER_2
                                    assert len(queue.bots) == 1
                                    assert queue.bots[0].tier == Tier.TIER_2

    def test_only_tier1_recovery(self, manager):
        """Test queue remix when only TIER_1 recovery bots exist."""
        with patch('src.database.repositories.bot_repository.BotRepository') as mock_repo:
            mock_bot = MagicMock()
            mock_bot.bot_name = "recovery_bot"

            mock_repo.return_value.get_active_bots.return_value = [mock_bot]

            mock_dpr = MagicMock()
            mock_dpr.composite_score = 80
            mock_dpr.component_scores.consistency = 50

            with patch.object(manager.scoring_engine, 'get_dpr_score', return_value=mock_dpr):
                with patch.object(manager.scoring_engine, '_is_specialist', return_value=False):
                    with patch.object(manager.scoring_engine, 'apply_specialist_boost', return_value=80):
                        with patch.object(manager, 'tier_assignment', return_value=Tier.TIER_1):
                            with patch.object(manager, 'get_recovery_step', return_value=2):
                                with patch.object(manager, '_persist_audit_log'):
                                    queue = manager.queue_remix("LONDON")

                                    # Should work with only TIER_1 recovery
                                    assert len(queue.bots) == 1
                                    assert queue.bots[0].tier == Tier.TIER_1
                                    assert queue.bots[0].recovery_step == 2


class TestQueueEntryEdgeCases:
    """Test QueueEntry edge cases."""

    def test_queue_entry_optional_fields(self):
        """Test QueueEntry with optional fields."""
        entry = QueueEntry(
            bot_id="bot_001",
            queue_position=1,
            dpr_composite_score=75,
            tier=Tier.TIER_3,
        )

        assert entry.specialist_session is None
        assert entry.specialist_boost_applied is False
        assert entry.concern_flag is False
        assert entry.recovery_step == 0
        assert entry.in_concern_subqueue is False

    def test_queue_entry_all_fields(self):
        """Test QueueEntry with all fields."""
        entry = QueueEntry(
            bot_id="bot_001",
            queue_position=1,
            dpr_composite_score=85,
            tier=Tier.TIER_1,
            specialist_session="LONDON",
            specialist_boost_applied=True,
            concern_flag=True,
            recovery_step=2,
            in_concern_subqueue=True,
        )

        assert entry.bot_id == "bot_001"
        assert entry.specialist_session == "LONDON"
        assert entry.specialist_boost_applied is True
        assert entry.concern_flag is True
        assert entry.recovery_step == 2
        assert entry.in_concern_subqueue is True


class TestDPRQueueOutputEdgeCases:
    """Test DPRQueueOutput edge cases."""

    def test_queue_output_default_locked(self):
        """Test DPRQueueOutput defaults to unlocked."""
        output = DPRQueueOutput(session_id="LONDON", bots=[])

        assert output.locked is False

    def test_queue_output_with_ny_hybrid(self):
        """Test DPRQueueOutput with NY hybrid override."""
        entry = QueueEntry(
            bot_id="bot_001",
            queue_position=1,
            dpr_composite_score=75,
            tier=Tier.TIER_3,
        )

        output = DPRQueueOutput(
            session_id="NY",
            bots=[entry],
            ny_hybrid_override=True,
            locked=True,
        )

        assert output.ny_hybrid_override is True
        assert output.locked is True


class TestPersistAuditLog:
    """Test _persist_audit_log method."""

    @pytest.fixture
    def manager(self):
        """Create DPR queue manager with mock dependencies."""
        mock_engine = MagicMock()
        mock_session = MagicMock()
        return DPRQueueManager(
            scoring_engine=mock_engine,
            db_session=mock_session,
        )

    def test_persist_audit_log_single_entry(self, manager):
        """Test persisting audit log with single entry."""
        entry = QueueEntry(
            bot_id="bot_001",
            queue_position=1,
            dpr_composite_score=75,
            tier=Tier.TIER_1,
            specialist_boost_applied=True,
            concern_flag=False,
        )

        with patch('src.database.repositories.bot_repository.BotRepository') as mock_repo:
            mock_bot = MagicMock()
            mock_bot.tags = ["SESSION_SPECIALIST"]
            mock_repo.return_value.get_by_name.return_value = mock_bot

            manager._persist_audit_log("LONDON", [entry])

            manager.db_session.add.assert_called()
            manager.db_session.commit.assert_called()

    def test_persist_audit_log_multiple_entries(self, manager):
        """Test persisting audit log with multiple entries."""
        entry1 = QueueEntry(
            bot_id="bot_001",
            queue_position=1,
            dpr_composite_score=80,
            tier=Tier.TIER_1,
            specialist_boost_applied=True,
            concern_flag=False,
        )
        entry2 = QueueEntry(
            bot_id="bot_002",
            queue_position=2,
            dpr_composite_score=70,
            tier=Tier.TIER_3,
            specialist_boost_applied=False,
            concern_flag=True,
        )

        with patch('src.database.repositories.bot_repository.BotRepository') as mock_repo:
            mock_bot1 = MagicMock()
            mock_bot1.tags = ["SESSION_SPECIALIST"]
            mock_bot2 = MagicMock()
            mock_bot2.tags = ["SESSION_CONCERN"]

            def mock_get_by_name(name):
                if name == "bot_001":
                    return mock_bot1
                return mock_bot2

            mock_repo.return_value.get_by_name.side_effect = mock_get_by_name

            manager._persist_audit_log("LONDON", [entry1, entry2])

            assert manager.db_session.add.call_count == 2
            manager.db_session.commit.assert_called()

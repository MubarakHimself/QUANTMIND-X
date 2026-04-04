"""
Tests for DPR Queue Manager.

Story 17.2: DPR Queue Tier Remix

Tests:
- Tier assignment for TIER_1, TIER_2, TIER_3 bots
- Queue remix with T1/T3/T2 interleaving
- Recovery eligible detection (2 consecutive wins)
- Specialist boost positioning
- Concern sub-queue flagging
- Queue lock mechanism
- SSL event queuing and Dead Zone application
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from datetime import datetime, timezone

from src.risk.dpr.queue_manager import DPRQueueManager, DPRQueueAuditLog
from src.risk.dpr.queue_models import Tier, QueueEntry, DPRQueueOutput
from src.events.dpr import SSLEvent, SSLEventType
from src.risk.dpr.scoring_engine import DPRScoringEngine


class TestTierAssignment:
    """Test tier assignment for bots."""

    @pytest.fixture
    def manager(self):
        """Create DPR queue manager with mock dependencies."""
        mock_engine = MagicMock(spec=DPRScoringEngine)
        mock_session = MagicMock()
        return DPRQueueManager(
            scoring_engine=mock_engine,
            db_session=mock_session,
        )

    def test_tier_assignment_no_circuit_breaker(self, manager):
        """Test tier assignment defaults to TIER_3 when no circuit breaker record."""
        with patch('src.database.repositories.circuit_breaker_repository.CircuitBreakerRepository') as mock_repo:
            mock_repo.return_value.get_by_bot_id.return_value = None
            tier = manager.tier_assignment("bot_001")
            assert tier == Tier.TIER_3

    def test_tier_assignment_quarantined_with_recovery(self, manager):
        """Test tier assignment TIER_1 for quarantined bot with recovery wins."""
        with patch('src.database.repositories.circuit_breaker_repository.CircuitBreakerRepository') as mock_repo:
            mock_cb = MagicMock()
            mock_cb.is_quarantined = True
            mock_repo.return_value.get_by_bot_id.return_value = mock_cb

            with patch.object(manager, 'get_recovery_step', return_value=2):
                tier = manager.tier_assignment("bot_001")
                assert tier == Tier.TIER_1

    def test_tier_assignment_quarantined_no_recovery(self, manager):
        """Test tier assignment TIER_3 for quarantined bot without recovery wins."""
        with patch('src.database.repositories.circuit_breaker_repository.CircuitBreakerRepository') as mock_repo:
            mock_cb = MagicMock()
            mock_cb.is_quarantined = True
            mock_cb.quarantine_start = None
            mock_repo.return_value.get_by_bot_id.return_value = mock_cb

            with patch.object(manager, 'get_recovery_step', return_value=0):
                tier = manager.tier_assignment("bot_001")
                assert tier == Tier.TIER_3

    def test_tier_assignment_active_bot(self, manager):
        """Test tier assignment TIER_3 for active non-quarantined bot."""
        with patch('src.database.repositories.circuit_breaker_repository.CircuitBreakerRepository') as mock_repo:
            mock_cb = MagicMock()
            mock_cb.is_quarantined = False
            mock_cb.mode = None  # Not DEMO mode
            mock_repo.return_value.get_by_bot_id.return_value = mock_cb

            tier = manager.tier_assignment("bot_001")
            assert tier == Tier.TIER_3

    def test_tier_assignment_demo_2_week_paper_is_tier_2(self, manager):
        """Test tier assignment TIER_2 for fresh AlphaForge candidate (DEMO mode, 2+ weeks paper)."""
        from datetime import timedelta
        from src.database.models.base import TradingMode

        with patch('src.database.repositories.circuit_breaker_repository.CircuitBreakerRepository') as mock_repo:
            mock_cb = MagicMock()
            mock_cb.is_quarantined = False
            mock_cb.mode = TradingMode.DEMO
            mock_cb.created_at = datetime.now(timezone.utc) - timedelta(days=20)  # 20 days in paper
            mock_repo.return_value.get_by_bot_id.return_value = mock_cb

            tier = manager.tier_assignment("bot_001")
            assert tier == Tier.TIER_2, f"Expected TIER_2 for DEMO bot with 20 days paper, got {tier}"

    def test_tier_assignment_demo_less_than_2_weeks_is_tier_3(self, manager):
        """Test tier assignment TIER_3 for DEMO bot with less than 2 weeks paper."""
        from datetime import timedelta
        from src.database.models.base import TradingMode

        with patch('src.database.repositories.circuit_breaker_repository.CircuitBreakerRepository') as mock_repo:
            mock_cb = MagicMock()
            mock_cb.is_quarantined = False
            mock_cb.mode = TradingMode.DEMO
            mock_cb.created_at = datetime.now(timezone.utc) - timedelta(days=7)  # Only 7 days in paper
            mock_repo.return_value.get_by_bot_id.return_value = mock_cb

            tier = manager.tier_assignment("bot_001")
            assert tier == Tier.TIER_3, f"Expected TIER_3 for DEMO bot with only 7 days paper, got {tier}"


class TestRecoveryStep:
    """Test recovery step detection."""

    @pytest.fixture
    def manager(self):
        """Create DPR queue manager with mock dependencies."""
        mock_engine = MagicMock(spec=DPRScoringEngine)
        mock_session = MagicMock()
        return DPRQueueManager(
            scoring_engine=mock_engine,
            db_session=mock_session,
        )

    def test_recovery_step_not_quarantined(self, manager):
        """Test recovery step 0 for non-quarantined bot."""
        with patch('src.database.repositories.circuit_breaker_repository.CircuitBreakerRepository') as mock_repo:
            mock_cb = MagicMock()
            mock_cb.is_quarantined = False
            mock_repo.return_value.get_by_bot_id.return_value = mock_cb

            step = manager.get_recovery_step("bot_001")
            assert step == 0

    def test_recovery_step_no_circuit_breaker(self, manager):
        """Test recovery step 0 when no circuit breaker record."""
        with patch('src.database.repositories.circuit_breaker_repository.CircuitBreakerRepository') as mock_repo:
            mock_repo.return_value.get_by_bot_id.return_value = None

            step = manager.get_recovery_step("bot_001")
            assert step == 0

    def test_recovery_step_first_win(self, manager):
        """Test recovery step 1 for bot with first consecutive win."""
        with patch('src.database.repositories.circuit_breaker_repository.CircuitBreakerRepository') as mock_repo:
            mock_cb = MagicMock()
            mock_cb.is_quarantined = True
            mock_cb.consecutive_session_wins = 1
            mock_repo.return_value.get_by_bot_id.return_value = mock_cb

            step = manager.get_recovery_step("bot_001")
            assert step == 1

    def test_recovery_step_eligible(self, manager):
        """Test recovery step 2 for bot with 2 consecutive wins."""
        with patch('src.database.repositories.circuit_breaker_repository.CircuitBreakerRepository') as mock_repo:
            mock_cb = MagicMock()
            mock_cb.is_quarantined = True
            mock_cb.consecutive_session_wins = 2
            mock_repo.return_value.get_by_bot_id.return_value = mock_cb

            step = manager.get_recovery_step("bot_001")
            assert step == 2


class TestRecoveryEligibleBots:
    """Test recovery eligible bot detection."""

    @pytest.fixture
    def manager(self):
        """Create DPR queue manager with mock dependencies."""
        mock_engine = MagicMock(spec=DPRScoringEngine)
        mock_session = MagicMock()
        return DPRQueueManager(
            scoring_engine=mock_engine,
            db_session=mock_session,
        )

    def test_get_recovery_eligible_bots(self, manager):
        """Test detection of recovery eligible bots."""
        with patch('src.database.repositories.circuit_breaker_repository.CircuitBreakerRepository') as mock_repo:
            mock_cb1 = MagicMock()
            mock_cb1.bot_id = "bot_001"
            mock_cb2 = MagicMock()
            mock_cb2.bot_id = "bot_002"

            mock_repo.return_value.get_quarantined.return_value = [mock_cb1, mock_cb2]

            with patch.object(manager, 'get_recovery_step') as mock_step:
                mock_step.side_effect = lambda bid: 2 if bid == "bot_001" else 0
                eligible = manager.get_recovery_eligible_bots()
                assert eligible == ["bot_001"]


class TestTierSort:
    """Test within-tier sorting by DPR score."""

    @pytest.fixture
    def manager(self):
        """Create DPR queue manager with mock dependencies."""
        mock_engine = MagicMock(spec=DPRScoringEngine)
        mock_session = MagicMock()
        return DPRQueueManager(
            scoring_engine=mock_engine,
            db_session=mock_session,
        )

    def test_tier_sort_descending(self, manager):
        """Test bots are sorted by score descending."""
        bots = ["bot_001", "bot_002", "bot_003"]
        scores = {
            "bot_001": 50,
            "bot_002": 80,
            "bot_003": 65,
        }

        sorted_bots = manager.tier_sort(bots, scores)
        assert sorted_bots == ["bot_002", "bot_003", "bot_001"]

    def test_tier_sort_empty_list(self, manager):
        """Test sorting empty list returns empty list."""
        sorted_bots = manager.tier_sort([], {})
        assert sorted_bots == []

    def test_tier_sort_single_bot(self, manager):
        """Test sorting single bot returns same bot."""
        sorted_bots = manager.tier_sort(["bot_001"], {"bot_001": 75})
        assert sorted_bots == ["bot_001"]

    def test_tier_sort_equal_scores(self, manager):
        """Test tie-breaking by bot_id when scores are equal."""
        bots = ["bot_001", "bot_002", "bot_003"]
        scores = {
            "bot_001": 75,
            "bot_002": 75,
            "bot_003": 75,
        }

        sorted_bots = manager.tier_sort(bots, scores)
        assert sorted_bots == ["bot_001", "bot_002", "bot_003"]


class TestQueueRemix:
    """Test queue remix with T1/T3/T2 interleaving."""

    @pytest.fixture
    def manager(self):
        """Create DPR queue manager with mock dependencies."""
        mock_engine = MagicMock(spec=DPRScoringEngine)
        mock_session = MagicMock()
        return DPRQueueManager(
            scoring_engine=mock_engine,
            db_session=mock_session,
        )

    def test_queue_remix_empty(self, manager):
        """Test queue remix with no active bots returns empty queue."""
        with patch('src.database.repositories.bot_repository.BotRepository') as mock_repo:
            mock_repo.return_value.get_active_bots.return_value = []

            queue = manager.queue_remix("LONDON")

            assert queue.session_id == "LONDON"
            assert len(queue.bots) == 0

    def test_queue_remix_single_bot(self, manager):
        """Test queue remix with single bot."""
        with patch('src.database.repositories.bot_repository.BotRepository') as mock_repo:
            mock_bot = MagicMock()
            mock_bot.bot_name = "bot_001"
            mock_repo.return_value.get_active_bots.return_value = [mock_bot]

            with patch.object(manager.scoring_engine, 'get_dpr_score') as mock_score:
                mock_dpr = MagicMock()
                mock_dpr.composite_score = 75
                mock_dpr.component_scores.consistency = 50
                mock_score.return_value = mock_dpr

                with patch.object(manager.scoring_engine, '_is_specialist', return_value=False):
                    with patch.object(manager.scoring_engine, 'apply_specialist_boost', return_value=75):
                        with patch.object(manager, 'tier_assignment', return_value=Tier.TIER_3):
                            with patch.object(manager, 'get_recovery_step', return_value=0):
                                queue = manager.queue_remix("LONDON")

                                assert len(queue.bots) == 1
                                assert queue.bots[0].bot_id == "bot_001"
                                assert queue.bots[0].queue_position == 1

    def test_queue_remix_tier_interleaving(self, manager):
        """Test T1/T3/T2 interleaving in queue remix."""
        with patch('src.database.repositories.bot_repository.BotRepository') as mock_repo:
            mock_bot1 = MagicMock()
            mock_bot1.bot_name = "t1_bot"
            mock_bot2 = MagicMock()
            mock_bot2.bot_name = "t3_bot"
            mock_bot3 = MagicMock()
            mock_bot3.bot_name = "t2_bot"

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
                            if bot_id == "t1_bot":
                                return Tier.TIER_1
                            elif bot_id == "t3_bot":
                                return Tier.TIER_3
                            return Tier.TIER_2

                        with patch.object(manager, 'tier_assignment', side_effect=mock_tier):
                            with patch.object(manager, 'get_recovery_step', return_value=0):
                                with patch.object(manager, '_persist_audit_log'):
                                    queue = manager.queue_remix("LONDON")

                                    assert len(queue.bots) == 3
                                    # TIER_3 (40% of 3 = 1 bot) comes before TIER_2
                                    t3_pos = next(i for i, b in enumerate(queue.bots) if b.bot_id == "t3_bot")
                                    t2_pos = next(i for i, b in enumerate(queue.bots) if b.bot_id == "t2_bot")
                                    assert t3_pos < t2_pos


class TestQueueLock:
    """Test queue lock mechanism."""

    @pytest.fixture
    def manager(self):
        """Create DPR queue manager with mock dependencies."""
        mock_engine = MagicMock(spec=DPRScoringEngine)
        mock_session = MagicMock()
        return DPRQueueManager(
            scoring_engine=mock_engine,
            db_session=mock_session,
        )

    def test_queue_not_locked_initially(self, manager):
        """Test queue is not locked initially."""
        assert manager.queue_locked("LONDON") is False

    def test_lock_queue(self, manager):
        """Test locking a queue."""
        manager.lock_queue("LONDON")
        assert manager.queue_locked("LONDON") is True

    def test_lock_different_sessions(self, manager):
        """Test locking one session doesn't affect others."""
        manager.lock_queue("LONDON")
        assert manager.queue_locked("NY") is False


class TestSSLEventQueuing:
    """Test SSL event queuing and Dead Zone application."""

    @pytest.fixture
    def manager(self):
        """Create DPR queue manager with mock dependencies."""
        mock_engine = MagicMock(spec=DPRScoringEngine)
        mock_session = MagicMock()
        return DPRQueueManager(
            scoring_engine=mock_engine,
            db_session=mock_session,
        )

    def test_queue_event_when_locked(self, manager):
        """Test queuing SSL event when session is locked."""
        manager.lock_queue("LONDON")

        event = SSLEvent(
            event_type=SSLEventType.MOVE_TO_PAPER,
            bot_id="bot_001",
            magic_number="12345",
            session_id="LONDON",
        )

        manager.queue_event(event)
        assert len(manager._queued_events) == 1

    def test_queue_event_when_not_locked(self, manager):
        """Test SSL event is not queued when session is not locked."""
        event = SSLEvent(
            event_type=SSLEventType.MOVE_TO_PAPER,
            bot_id="bot_001",
            magic_number="12345",
            session_id="LONDON",
        )

        manager.queue_event(event)
        assert len(manager._queued_events) == 0


class TestQueueEntry:
    """Test QueueEntry model."""

    def test_queue_entry_creation(self):
        """Test creating a queue entry."""
        entry = QueueEntry(
            bot_id="bot_001",
            queue_position=1,
            dpr_composite_score=85,
            tier=Tier.TIER_1,
            specialist_session="LONDON",
            specialist_boost_applied=True,
            concern_flag=False,
            recovery_step=2,
            in_concern_subqueue=False,
        )

        assert entry.bot_id == "bot_001"
        assert entry.queue_position == 1
        assert entry.dpr_composite_score == 85
        assert entry.tier == Tier.TIER_1
        assert entry.specialist_session == "LONDON"
        assert entry.recovery_step == 2

    def test_queue_entry_str(self):
        """Test queue entry string representation."""
        entry = QueueEntry(
            bot_id="bot_001",
            queue_position=1,
            dpr_composite_score=85,
            tier=Tier.TIER_1,
        )

        s = str(entry)
        assert "bot_001" in s
        assert "TIER_1" in s
        assert "85" in s


class TestDPRQueueOutput:
    """Test DPRQueueOutput model."""

    def test_queue_output_creation(self):
        """Test creating a queue output."""
        entry = QueueEntry(
            bot_id="bot_001",
            queue_position=1,
            dpr_composite_score=85,
            tier=Tier.TIER_1,
        )

        output = DPRQueueOutput(
            session_id="LONDON",
            bots=[entry],
            ny_hybrid_override=False,
        )

        assert output.session_id == "LONDON"
        assert len(output.bots) == 1
        assert output.ny_hybrid_override is False

    def test_queue_output_str(self):
        """Test queue output string representation."""
        output = DPRQueueOutput(session_id="LONDON", bots=[])
        s = str(output)
        assert "LONDON" in s


class TestSSLEvent:
    """Test SSLEvent model."""

    def test_ssl_event_creation(self):
        """Test creating an SSL event."""
        event = SSLEvent(
            event_type=SSLEventType.MOVE_TO_PAPER,
            bot_id="bot_001",
            magic_number="12345",
            session_id="LONDON",
        )

        assert event.event_type == SSLEventType.MOVE_TO_PAPER
        assert event.bot_id == "bot_001"

    def test_ssl_event_str(self):
        """Test SSL event string representation."""
        event = SSLEvent(
            event_type=SSLEventType.RECOVERY_CONFIRMED,
            bot_id="bot_001",
            magic_number="12345",
            session_id="LONDON",
        )

        s = str(event)
        assert "RECOVERY_CONFIRMED" in s
        assert "bot_001" in s

"""
Tests for DPR Event Models.

Story 17.1: DPR Composite Score Calculation

Tests:
- DPRComponentScores composite calculation
- DPRScoreEvent audit log creation
- DPRConcernEvent concern flag detection
- DPR_WEIGHTS constant
- SSLEventType enum
- DPRQueueOutput model
- DPRQueueAuditRecord model
"""

import pytest
from datetime import datetime, timezone

from src.events.dpr import (
    DPRComponentScores,
    DPRScoreEvent,
    DPRConcernEvent,
    DPR_WEIGHTS,
    SSLEventType,
    SSLEvent,
)
from src.risk.dpr.queue_models import (
    DPRQueueOutput,
    DPRQueueAuditRecord,
    QueueEntry,
    Tier,
)


class TestDPRComponentScores:
    """Test DPRComponentScores model."""

    def test_composite_score_zero(self):
        """Test composite score with all zeros."""
        components = DPRComponentScores(
            win_rate=0.0,
            pnl=0.0,
            consistency=0.0,
            ev_per_trade=0.0,
        )
        assert components.composite_score() == 0

    def test_composite_score_hundred(self):
        """Test composite score with all 100s."""
        components = DPRComponentScores(
            win_rate=100.0,
            pnl=100.0,
            consistency=100.0,
            ev_per_trade=100.0,
        )
        assert components.composite_score() == 100

    def test_composite_score_mixed(self):
        """Test composite score with mixed values."""
        components = DPRComponentScores(
            win_rate=100.0,  # 25% weight
            pnl=0.0,        # 30% weight
            consistency=100.0,  # 20% weight
            ev_per_trade=0.0,  # 25% weight
        )
        # 100*0.25 + 0*0.30 + 100*0.20 + 0*0.25 = 25 + 0 + 20 + 0 = 45
        assert components.composite_score() == 45

    def test_default_weights(self):
        """Test default weights are set correctly."""
        components = DPRComponentScores(
            win_rate=50.0,
            pnl=50.0,
            consistency=50.0,
            ev_per_trade=50.0,
        )
        assert components.weights == (0.25, 0.30, 0.20, 0.25)
        assert sum(components.weights) == 1.0

    def test_str_representation(self):
        """Test string representation."""
        components = DPRComponentScores(
            win_rate=75.0,
            pnl=80.0,
            consistency=85.0,
            ev_per_trade=90.0,
        )
        s = str(components)
        assert "WR=75.0" in s
        assert "PnL=80.0" in s
        assert "Cons=85.0" in s
        assert "EV=90.0" in s


class TestDPRScoreEvent:
    """Test DPRScoreEvent model."""

    def test_create_score_event(self):
        """Test DPRScoreEvent creation."""
        components = DPRComponentScores(
            win_rate=80.0,
            pnl=70.0,
            consistency=90.0,
            ev_per_trade=60.0,
        )
        event = DPRScoreEvent(
            bot_id="bot_001",
            session_id="LONDON",
            component_scores=components,
            composite_score=74,
        )

        assert event.bot_id == "bot_001"
        assert event.session_id == "LONDON"
        assert event.composite_score == 74
        assert event.is_tied is False
        assert event.specialist_boost_applied is False
        assert event.session_concern_flag is False

    def test_score_event_with_tie(self):
        """Test DPRScoreEvent with tie."""
        components = DPRComponentScores(
            win_rate=65.0,
            pnl=65.0,
            consistency=65.0,
            ev_per_trade=65.0,
        )
        event = DPRScoreEvent(
            bot_id="bot_001",
            session_id="LONDON",
            component_scores=components,
            composite_score=65,
            is_tied=True,
            tie_break_winner="bot_002",
        )

        assert event.is_tied is True
        assert event.tie_break_winner == "bot_002"

    def test_score_event_with_specialist_boost(self):
        """Test DPRScoreEvent with specialist boost applied."""
        components = DPRComponentScores(
            win_rate=80.0,
            pnl=70.0,
            consistency=90.0,
            ev_per_trade=60.0,
        )
        event = DPRScoreEvent(
            bot_id="bot_001",
            session_id="LONDON",
            component_scores=components,
            composite_score=79,  # 74 + 5 boost
            specialist_boost_applied=True,
        )

        assert event.specialist_boost_applied is True

    def test_score_event_with_concern_flag(self):
        """Test DPRScoreEvent with SESSION_CONCERN flag."""
        components = DPRComponentScores(
            win_rate=40.0,
            pnl=30.0,
            consistency=50.0,
            ev_per_trade=35.0,
        )
        event = DPRScoreEvent(
            bot_id="bot_001",
            session_id="LONDON",
            component_scores=components,
            composite_score=38,
            session_concern_flag=True,
        )

        assert event.session_concern_flag is True

    def test_score_event_timestamp(self):
        """Test DPRScoreEvent timestamp is set automatically."""
        components = DPRComponentScores(
            win_rate=50.0,
            pnl=50.0,
            consistency=50.0,
            ev_per_trade=50.0,
        )
        event = DPRScoreEvent(
            bot_id="bot_001",
            session_id="LONDON",
            component_scores=components,
            composite_score=50,
        )

        assert event.timestamp_utc is not None
        assert isinstance(event.timestamp_utc, datetime)

    def test_score_event_metadata(self):
        """Test DPRScoreEvent with metadata."""
        components = DPRComponentScores(
            win_rate=80.0,
            pnl=70.0,
            consistency=90.0,
            ev_per_trade=60.0,
        )
        event = DPRScoreEvent(
            bot_id="bot_001",
            session_id="LONDON",
            component_scores=components,
            composite_score=74,
            metadata={
                "trade_count": 25,
                "magic_number": 12345,
                "max_drawdown": 5.2,
            },
        )

        assert event.metadata["trade_count"] == 25
        assert event.metadata["magic_number"] == 12345
        assert event.metadata["max_drawdown"] == 5.2

    def test_score_event_str(self):
        """Test DPRScoreEvent string representation."""
        components = DPRComponentScores(
            win_rate=80.0,
            pnl=70.0,
            consistency=90.0,
            ev_per_trade=60.0,
        )
        event = DPRScoreEvent(
            bot_id="bot_001",
            session_id="LONDON",
            component_scores=components,
            composite_score=74,
        )

        s = str(event)
        assert "bot_001" in s
        assert "LONDON" in s
        assert "74" in s


class TestDPRConcernEvent:
    """Test DPRConcernEvent model."""

    def test_create_concern_event(self):
        """
        Test DPRConcernEvent creation.

        AC #4: Given a bot's DPR score drops >20 points week-over-week,
        When the fortnight accumulation completes,
        Then a SESSION_CONCERN flag is set on that bot.
        """
        event = DPRConcernEvent(
            bot_id="bot_001",
            previous_score=80,
            current_score=55,
            score_delta=-25,
        )

        assert event.bot_id == "bot_001"
        assert event.previous_score == 80
        assert event.current_score == 55
        assert event.score_delta == -25
        assert event.threshold == -20

    def test_concern_event_default_threshold(self):
        """Test DPRConcernEvent default threshold."""
        event = DPRConcernEvent(
            bot_id="bot_001",
            previous_score=80,
            current_score=55,
            score_delta=-25,
        )

        assert event.threshold == -20

    def test_concern_event_custom_threshold(self):
        """Test DPRConcernEvent with custom threshold."""
        event = DPRConcernEvent(
            bot_id="bot_001",
            previous_score=80,
            current_score=55,
            score_delta=-30,
            threshold=-30,
        )

        assert event.threshold == -30

    def test_concern_event_positive_delta(self):
        """Test DPRConcernEvent with positive delta (no concern)."""
        event = DPRConcernEvent(
            bot_id="bot_001",
            previous_score=50,
            current_score=75,
            score_delta=25,
        )

        assert event.score_delta == 25

    def test_concern_event_str(self):
        """Test DPRConcernEvent string representation."""
        event = DPRConcernEvent(
            bot_id="bot_001",
            previous_score=80,
            current_score=55,
            score_delta=-25,
        )

        s = str(event)
        assert "bot_001" in s
        assert "-25" in s
        assert "80" in s
        assert "55" in s


class TestDPRWeights:
    """Test DPR_WEIGHTS constant."""

    def test_weights_sum_to_one(self):
        """Test DPR_WEIGHTS sum to 1.0."""
        assert sum(DPR_WEIGHTS) == 1.0

    def test_weights_correct_values(self):
        """Test DPR_WEIGHTS have correct values."""
        # WR (25%), PnL (30%), consistency (20%), EV (25%)
        assert DPR_WEIGHTS == (0.25, 0.30, 0.20, 0.25)

    def test_weights_tuple_length(self):
        """Test DPR_WEIGHTS is a 4-element tuple."""
        assert len(DPR_WEIGHTS) == 4


class TestSSLEventType:
    """Test SSLEventType enum."""

    def test_ssl_event_types_exist(self):
        """Test all SSL event types are defined."""
        assert SSLEventType.MOVE_TO_PAPER.value == "MOVE_TO_PAPER"
        assert SSLEventType.RECOVERY_STEP_1.value == "RECOVERY_STEP_1"
        assert SSLEventType.RECOVERY_CONFIRMED.value == "RECOVERY_CONFIRMED"
        assert SSLEventType.RETIRED.value == "RETIRED"

    def test_ssl_event_type_count(self):
        """Test all 4 SSL event types exist."""
        assert len(SSLEventType) == 4


class TestSSLEventModel:
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
        assert event.magic_number == "12345"
        assert event.session_id == "LONDON"

    def test_ssl_event_timestamp_auto_set(self):
        """Test SSL event timestamp is auto-set."""
        event = SSLEvent(
            event_type=SSLEventType.RECOVERY_CONFIRMED,
            bot_id="bot_001",
            magic_number="12345",
            session_id="LONDON",
        )

        assert event.timestamp_utc is not None
        assert isinstance(event.timestamp_utc, datetime)

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
        assert "LONDON" in s

    def test_ssl_event_all_types(self):
        """Test creating SSL events for all event types."""
        for event_type in SSLEventType:
            event = SSLEvent(
                event_type=event_type,
                bot_id="bot_001",
                magic_number="12345",
                session_id="LONDON",
            )
            assert event.event_type == event_type


class TestDPRQueueOutput:
    """Test DPRQueueOutput model."""

    def test_queue_output_creation(self):
        """Test creating a queue output."""
        output = DPRQueueOutput(session_id="LONDON")

        assert output.session_id == "LONDON"
        assert output.locked is False
        assert output.ny_hybrid_override is False
        assert output.bots == []

    def test_queue_output_with_bots(self):
        """Test queue output with queue entries."""
        entry = QueueEntry(
            bot_id="bot_001",
            queue_position=1,
            dpr_composite_score=85,
            tier=Tier.TIER_1,
        )
        output = DPRQueueOutput(
            session_id="LONDON",
            bots=[entry],
        )

        assert len(output.bots) == 1
        assert output.bots[0].bot_id == "bot_001"

    def test_queue_output_timestamp_auto_set(self):
        """Test queue output timestamp is auto-set."""
        output = DPRQueueOutput(session_id="LONDON")

        assert output.queue_timestamp is not None
        assert isinstance(output.queue_timestamp, datetime)

    def test_queue_output_ny_hybrid(self):
        """Test queue output with NY hybrid override."""
        output = DPRQueueOutput(
            session_id="NY",
            ny_hybrid_override=True,
        )

        assert output.ny_hybrid_override is True

    def test_queue_output_locked(self):
        """Test queue output locked state."""
        output = DPRQueueOutput(
            session_id="LONDON",
            locked=True,
        )

        assert output.locked is True

    def test_queue_output_str(self):
        """Test queue output string representation."""
        output = DPRQueueOutput(session_id="LONDON")

        s = str(output)
        assert "LONDON" in s
        assert "locked=False" in s
        assert "ny_hybrid=False" in s


class TestDPRQueueAuditRecord:
    """Test DPRQueueAuditRecord model."""

    def test_audit_record_creation(self):
        """Test creating an audit record."""
        record = DPRQueueAuditRecord(
            session_id="LONDON",
            bot_id="bot_001",
            queue_position=1,
            dpr_composite_score=85,
            tier="TIER_1",
        )

        assert record.session_id == "LONDON"
        assert record.bot_id == "bot_001"
        assert record.queue_position == 1
        assert record.dpr_composite_score == 85
        assert record.tier == "TIER_1"

    def test_audit_record_defaults(self):
        """Test audit record default values."""
        record = DPRQueueAuditRecord(
            session_id="LONDON",
            bot_id="bot_001",
            queue_position=1,
            dpr_composite_score=75,
            tier="TIER_3",
        )

        assert record.specialist_flag is False
        assert record.concern_flag is False

    def test_audit_record_with_flags(self):
        """Test audit record with specialist and concern flags."""
        record = DPRQueueAuditRecord(
            session_id="LONDON",
            bot_id="bot_001",
            queue_position=1,
            dpr_composite_score=85,
            tier="TIER_1",
            specialist_flag=True,
            concern_flag=True,
        )

        assert record.specialist_flag is True
        assert record.concern_flag is True

    def test_audit_record_timestamp_auto_set(self):
        """Test audit record timestamp is auto-set."""
        record = DPRQueueAuditRecord(
            session_id="LONDON",
            bot_id="bot_001",
            queue_position=1,
            dpr_composite_score=75,
            tier="TIER_3",
        )

        assert record.timestamp_utc is not None
        assert isinstance(record.timestamp_utc, datetime)

    def test_audit_record_str(self):
        """Test audit record string representation."""
        record = DPRQueueAuditRecord(
            session_id="LONDON",
            bot_id="bot_001",
            queue_position=1,
            dpr_composite_score=85,
            tier="TIER_1",
        )

        s = str(record)
        assert "bot_001" in s
        assert "LONDON" in s
        assert "TIER_1" in s

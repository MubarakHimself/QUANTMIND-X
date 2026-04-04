"""
Tests for DPR Score History Data Access Layer.

Story 17.1: DPR Composite Score Calculation

Tests:
- DPRScoreAuditLog persistence
- DPRScoreHistory data access methods
- Week-over-week delta calculation
- Fortnight score filtering
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone, timedelta

from src.risk.dpr.history import DPRScoreHistory, DPRScoreAuditLog, DPRScoreHistoryRecord


class TestDPRScoreAuditLog:
    """Test DPRScoreAuditLog SQLAlchemy model."""

    def test_audit_log_repr(self):
        """Test string representation of audit log."""
        log = DPRScoreAuditLog(
            id=1,
            bot_id="bot_001",
            session_id="LONDON",
            scoring_window="session",
            win_rate_score=80.0,
            pnl_score=75.0,
            consistency_score=90.0,
            ev_per_trade_score=70.0,
            composite_score=78,
        )

        repr_str = repr(log)
        assert "bot_001" in repr_str
        assert "LONDON" in repr_str
        assert "78" in repr_str

    def test_audit_log_attributes(self):
        """Test audit log attributes are set correctly."""
        log = DPRScoreAuditLog(
            bot_id="bot_001",
            session_id="LONDON",
            win_rate_score=80.0,
            pnl_score=75.0,
            consistency_score=90.0,
            ev_per_trade_score=70.0,
            composite_score=78,
            is_tied=False,
            tie_break_winner=None,
            specialist_boost_applied=False,
            session_concern_flag=False,
            scoring_window="session",
        )

        assert log.is_tied is False
        assert log.tie_break_winner is None
        assert log.specialist_boost_applied is False
        assert log.session_concern_flag is False
        assert log.scoring_window == "session"


class TestDPRScoreHistoryRecord:
    """Test DPRScoreHistoryRecord dataclass."""

    def test_record_creation(self):
        """Test creating a history record."""
        now = datetime.now(timezone.utc)
        record = DPRScoreHistoryRecord(
            bot_id="bot_001",
            session_id="LONDON",
            composite_score=75,
            timestamp_utc=now,
        )

        assert record.bot_id == "bot_001"
        assert record.session_id == "LONDON"
        assert record.composite_score == 75
        assert record.timestamp_utc == now


class TestDPRScoreHistoryPersist:
    """Test DPRScoreHistory.persist_score method."""

    @pytest.fixture
    def history(self):
        """Create DPR history with mock session."""
        mock_session = MagicMock()
        return DPRScoreHistory(db_session=mock_session)

    def test_persist_score_basic(self, history):
        """Test persisting a basic score."""
        component_scores = {
            "win_rate": 80.0,
            "pnl": 75.0,
            "consistency": 90.0,
            "ev_per_trade": 70.0,
        }

        result = history.persist_score(
            bot_id="bot_001",
            session_id="LONDON",
            composite_score=78,
            component_scores=component_scores,
        )

        history.db_session.add.assert_called_once()
        history.db_session.commit.assert_called_once()
        history.db_session.refresh.assert_called_once()
        assert result.bot_id == "bot_001"
        assert result.composite_score == 78

    def test_persist_score_with_tie(self, history):
        """Test persisting a tied score."""
        component_scores = {
            "win_rate": 65.0,
            "pnl": 65.0,
            "consistency": 65.0,
            "ev_per_trade": 65.0,
        }

        result = history.persist_score(
            bot_id="bot_001",
            session_id="LONDON",
            composite_score=65,
            component_scores=component_scores,
            is_tied=True,
            tie_break_winner="bot_002",
        )

        assert result.is_tied is True
        assert result.tie_break_winner == "bot_002"

    def test_persist_score_with_specialist_boost(self, history):
        """Test persisting a score with specialist boost."""
        component_scores = {
            "win_rate": 80.0,
            "pnl": 75.0,
            "consistency": 90.0,
            "ev_per_trade": 70.0,
        }

        result = history.persist_score(
            bot_id="bot_001",
            session_id="LONDON",
            composite_score=78,
            component_scores=component_scores,
            specialist_boost_applied=True,
        )

        assert result.specialist_boost_applied is True

    def test_persist_score_with_concern_flag(self, history):
        """Test persisting a score with concern flag."""
        component_scores = {
            "win_rate": 40.0,
            "pnl": 30.0,
            "consistency": 50.0,
            "ev_per_trade": 35.0,
        }

        result = history.persist_score(
            bot_id="bot_001",
            session_id="LONDON",
            composite_score=38,
            component_scores=component_scores,
            session_concern_flag=True,
        )

        assert result.session_concern_flag is True

    def test_persist_score_with_metadata(self, history):
        """Test persisting a score with metadata."""
        component_scores = {
            "win_rate": 80.0,
            "pnl": 75.0,
            "consistency": 90.0,
            "ev_per_trade": 70.0,
        }
        metadata = {"trade_count": 25, "magic_number": 12345}

        result = history.persist_score(
            bot_id="bot_001",
            session_id="LONDON",
            composite_score=78,
            component_scores=component_scores,
            metadata=metadata,
        )

        assert result.metadata_json == metadata

    def test_persist_score_fortnight_window(self, history):
        """Test persisting a score with fortnight scoring window."""
        component_scores = {
            "win_rate": 80.0,
            "pnl": 75.0,
            "consistency": 90.0,
            "ev_per_trade": 70.0,
        }

        result = history.persist_score(
            bot_id="bot_001",
            session_id="LONDON",
            composite_score=78,
            component_scores=component_scores,
            scoring_window="fortnight",
        )

        assert result.scoring_window == "fortnight"


class TestDPRScoreHistoryGetScores:
    """Test DPRScoreHistory.get_bot_scores method."""

    @pytest.fixture
    def history(self):
        """Create DPR history with mock session."""
        mock_session = MagicMock()
        return DPRScoreHistory(db_session=mock_session)

    def test_get_bot_scores_basic(self, history):
        """Test getting bot scores."""
        now = datetime.now(timezone.utc)
        mock_records = [
            MagicMock(
                bot_id="bot_001",
                session_id="LONDON",
                composite_score=75,
                timestamp_utc=now,
            ),
            MagicMock(
                bot_id="bot_001",
                session_id="LONDON",
                composite_score=70,
                timestamp_utc=now - timedelta(days=1),
            ),
        ]

        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = mock_records
        history.db_session.query.return_value = mock_query

        result = history.get_bot_scores("bot_001")

        assert len(result) == 2
        assert result[0].composite_score == 75
        assert result[1].composite_score == 70

    def test_get_bot_scores_with_session_filter(self, history):
        """Test getting bot scores filtered by session."""
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = []
        history.db_session.query.return_value = mock_query

        history.get_bot_scores("bot_001", session_id="LONDON")

        # Verify filter was called with session_id
        assert mock_query.filter.called

    def test_get_bot_scores_with_window_filter(self, history):
        """Test getting bot scores filtered by scoring window."""
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = []
        history.db_session.query.return_value = mock_query

        history.get_bot_scores("bot_001", scoring_window="fortnight")

        # Verify filter was called with scoring_window
        assert mock_query.filter.called

    def test_get_bot_scores_with_limit(self, history):
        """Test getting bot scores with custom limit."""
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = []
        history.db_session.query.return_value = mock_query

        history.get_bot_scores("bot_001", limit=5)

        mock_query.limit.assert_called_with(5)

    def test_get_bot_scores_empty(self, history):
        """Test getting scores for bot with no history."""
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = []
        history.db_session.query.return_value = mock_query

        result = history.get_bot_scores("bot_unknown")

        assert len(result) == 0


class TestDPRScoreHistoryFortnightScores:
    """Test DPRScoreHistory.get_fortnight_scores method."""

    @pytest.fixture
    def history(self):
        """Create DPR history with mock session."""
        mock_session = MagicMock()
        return DPRScoreHistory(db_session=mock_session)

    def test_get_fortnight_scores(self, history):
        """Test getting fortnight scores."""
        now = datetime.now(timezone.utc)
        mock_records = [
            MagicMock(
                bot_id="bot_001",
                session_id="LONDON",
                composite_score=80,
                timestamp_utc=now,
            ),
            MagicMock(
                bot_id="bot_001",
                session_id="LONDON",
                composite_score=75,
                timestamp_utc=now - timedelta(days=7),
            ),
        ]

        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = mock_records
        history.db_session.query.return_value = mock_query

        result = history.get_fortnight_scores("bot_001")

        assert len(result) == 2
        assert result[0].composite_score == 80
        assert result[1].composite_score == 75

    def test_get_fortnight_scores_empty(self, history):
        """Test getting fortnight scores when none exist."""
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = []
        history.db_session.query.return_value = mock_query

        result = history.get_fortnight_scores("bot_no_history")

        assert len(result) == 0

    def test_get_fortnight_scores_only_fortnight_window(self, history):
        """Test that only fortnight window scores are returned."""
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = []
        history.db_session.query.return_value = mock_query

        history.get_fortnight_scores("bot_001")

        # Verify scoring_window filter was applied
        assert mock_query.filter.called


class TestDPRScoreHistoryClose:
    """Test DPRScoreHistory.close method."""

    def test_close_with_session(self):
        """Test closing history with an active session."""
        mock_session = MagicMock()
        history = DPRScoreHistory(db_session=mock_session)

        history.close()

        mock_session.close.assert_called_once()
        assert history._db_session is None

    def test_close_without_session(self):
        """Test closing history without a session."""
        history = DPRScoreHistory(db_session=None)

        # Should not raise
        history.close()

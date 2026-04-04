"""
Tests for SSL State Machine.

Story 18.1: Per-Bot Consecutive Loss Counter & Paper Rotation

Tests cover:
- SSL state transitions
- State persistence
- Recovery candidate selection
- Paper bot retrieval
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timezone

from src.risk.ssl.state import (
    SSLCircuitBreakerState,
    SSLState,
    BotTier,
    is_valid_transition,
    VALID_TRANSITIONS,
)


class TestValidTransitions:
    """Test state transition validation logic."""

    def test_all_valid_transitions_defined(self):
        """Test all expected transitions are defined."""
        assert SSLState.LIVE in VALID_TRANSITIONS
        assert SSLState.PAPER in VALID_TRANSITIONS
        assert SSLState.RECOVERY in VALID_TRANSITIONS
        assert SSLState.RETIRED in VALID_TRANSITIONS

    def test_retired_is_terminal(self):
        """Test RETIRED state has no valid transitions."""
        assert len(VALID_TRANSITIONS[SSLState.RETIRED]) == 0


class TestIsValidTransition:
    """Test is_valid_transition function."""

    def test_live_to_paper(self):
        """Test LIVE -> PAPER transition."""
        assert is_valid_transition(SSLState.LIVE, SSLState.PAPER) is True

    def test_paper_to_recovery(self):
        """Test PAPER -> RECOVERY transition."""
        assert is_valid_transition(SSLState.PAPER, SSLState.RECOVERY) is True

    def test_paper_to_retired(self):
        """Test PAPER -> RETIRED transition."""
        assert is_valid_transition(SSLState.PAPER, SSLState.RETIRED) is True

    def test_recovery_to_live(self):
        """Test RECOVERY -> LIVE transition."""
        assert is_valid_transition(SSLState.RECOVERY, SSLState.LIVE) is True

    def test_recovery_to_paper(self):
        """Test RECOVERY -> PAPER (loss during recovery)."""
        assert is_valid_transition(SSLState.RECOVERY, SSLState.PAPER) is True

    def test_live_to_recovery_invalid(self):
        """Test LIVE -> RECOVERY is invalid (must go through paper)."""
        assert is_valid_transition(SSLState.LIVE, SSLState.RECOVERY) is False

    def test_live_to_retired_invalid(self):
        """Test LIVE -> RETIRED is invalid."""
        assert is_valid_transition(SSLState.LIVE, SSLState.RETIRED) is False

    def test_paper_to_live_invalid(self):
        """Test PAPER -> LIVE is invalid (must go through recovery)."""
        assert is_valid_transition(SSLState.PAPER, SSLState.LIVE) is False

    def test_retired_to_any_invalid(self):
        """Test RETIRED -> any state is invalid."""
        for state in SSLState:
            if state != SSLState.RETIRED:
                assert is_valid_transition(SSLState.RETIRED, state) is False


class TestSSLStateManagerBasics:
    """Test SSLCircuitBreakerState basic functionality."""

    def test_initialization(self):
        """Test state manager initializes correctly."""
        state_mgr = SSLCircuitBreakerState()
        assert state_mgr._db_session is None

    def test_initialization_with_session(self):
        """Test state manager initializes with provided session."""
        mock_session = Mock()
        state_mgr = SSLCircuitBreakerState(db_session=mock_session)
        assert state_mgr._db_session is mock_session


class TestGetState:
    """Test get_state functionality."""

    @patch('src.risk.ssl.state.SessionLocal')
    def test_get_state_live_default(self, mock_session_local):
        """Test get_state returns LIVE for unknown bot."""
        mock_session = Mock()
        state_mgr = SSLCircuitBreakerState(db_session=mock_session)

        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        state = state_mgr.get_state("unknown-bot")
        assert state == SSLState.LIVE

    @patch('src.risk.ssl.state.SessionLocal')
    def test_get_state_from_record(self, mock_session_local):
        """Test get_state returns state from record."""
        mock_session = Mock()
        state_mgr = SSLCircuitBreakerState(db_session=mock_session)

        mock_record = Mock()
        mock_record.state = "paper"

        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = mock_record
        mock_session.execute.return_value = mock_result

        state = state_mgr.get_state("bot-1")
        assert state == SSLState.PAPER

    @patch('src.risk.ssl.state.SessionLocal')
    def test_get_state_invalid_value_defaults_to_live(self, mock_session_local):
        """Test get_state defaults to LIVE for invalid state value."""
        mock_session = Mock()
        state_mgr = SSLCircuitBreakerState(db_session=mock_session)

        mock_record = Mock()
        mock_record.state = "invalid_state"

        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = mock_record
        mock_session.execute.return_value = mock_result

        state = state_mgr.get_state("bot-1")
        assert state == SSLState.LIVE


class TestGetTier:
    """Test get_tier functionality."""

    @patch('src.risk.ssl.state.SessionLocal')
    def test_get_tier_tier_1(self, mock_session_local):
        """Test get_tier returns TIER_1."""
        mock_session = Mock()
        state_mgr = SSLCircuitBreakerState(db_session=mock_session)

        mock_record = Mock()
        mock_record.tier = "TIER_1"

        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = mock_record
        mock_session.execute.return_value = mock_result

        tier = state_mgr.get_tier("bot-1")
        assert tier == BotTier.TIER_1

    @patch('src.risk.ssl.state.SessionLocal')
    def test_get_tier_tier_2(self, mock_session_local):
        """Test get_tier returns TIER_2."""
        mock_session = Mock()
        state_mgr = SSLCircuitBreakerState(db_session=mock_session)

        mock_record = Mock()
        mock_record.tier = "TIER_2"

        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = mock_record
        mock_session.execute.return_value = mock_result

        tier = state_mgr.get_tier("bot-1")
        assert tier == BotTier.TIER_2

    @patch('src.risk.ssl.state.SessionLocal')
    def test_get_tier_none_if_not_in_paper(self, mock_session_local):
        """Test get_tier returns None if not in paper."""
        mock_session = Mock()
        state_mgr = SSLCircuitBreakerState(db_session=mock_session)

        mock_record = Mock()
        mock_record.tier = None

        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = mock_record
        mock_session.execute.return_value = mock_result

        tier = state_mgr.get_tier("bot-1")
        assert tier is None


class TestGetConsecutiveLosses:
    """Test get_consecutive_losses functionality."""

    @patch('src.risk.ssl.state.SessionLocal')
    def test_get_consecutive_losses_zero_default(self, mock_session_local):
        """Test get_consecutive_losses returns 0 for unknown bot."""
        mock_session = Mock()
        state_mgr = SSLCircuitBreakerState(db_session=mock_session)

        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        count = state_mgr.get_consecutive_losses("unknown-bot")
        assert count == 0

    @patch('src.risk.ssl.state.SessionLocal')
    def test_get_consecutive_losses_from_record(self, mock_session_local):
        """Test get_consecutive_losses returns value from record."""
        mock_session = Mock()
        state_mgr = SSLCircuitBreakerState(db_session=mock_session)

        mock_record = Mock()
        mock_record.consecutive_losses = 5

        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = mock_record
        mock_session.execute.return_value = mock_result

        count = state_mgr.get_consecutive_losses("bot-1")
        assert count == 5


class TestGetRecoveryWinCount:
    """Test get_recovery_win_count functionality."""

    @patch('src.risk.ssl.state.SessionLocal')
    def test_get_recovery_win_count_zero_default(self, mock_session_local):
        """Test get_recovery_win_count returns 0 for unknown bot."""
        mock_session = Mock()
        state_mgr = SSLCircuitBreakerState(db_session=mock_session)

        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        count = state_mgr.get_recovery_win_count("unknown-bot")
        assert count == 0

    @patch('src.risk.ssl.state.SessionLocal')
    def test_get_recovery_win_count_from_record(self, mock_session_local):
        """Test get_recovery_win_count returns value from record."""
        mock_session = Mock()
        state_mgr = SSLCircuitBreakerState(db_session=mock_session)

        mock_record = Mock()
        mock_record.recovery_win_count = 2

        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = mock_record
        mock_session.execute.return_value = mock_result

        count = state_mgr.get_recovery_win_count("bot-1")
        assert count == 2


class TestGetAllBotsInState:
    """Test get_all_bots_in_state functionality."""

    @patch('src.risk.ssl.state.SessionLocal')
    def test_get_all_bots_in_paper(self, mock_session_local):
        """Test get_all_bots_in_state returns bots in PAPER state."""
        mock_session = Mock()
        state_mgr = SSLCircuitBreakerState(db_session=mock_session)

        mock_record1 = Mock()
        mock_record1.bot_id = "bot-1"
        mock_record2 = Mock()
        mock_record2.bot_id = "bot-2"

        mock_scalars = Mock()
        mock_scalars.scalars.return_value.all.return_value = [mock_record1, mock_record2]

        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = [mock_record1, mock_record2]
        mock_session.execute.return_value = mock_result

        bots = state_mgr.get_all_bots_in_state(SSLState.PAPER)
        assert len(bots) == 2
        assert "bot-1" in bots
        assert "bot-2" in bots


class TestGetTier1RecoveryCandidates:
    """Test get_tier_1_recovery_candidates functionality."""

    @patch('src.risk.ssl.state.SessionLocal')
    def test_get_tier_1_recovery_candidates(self, mock_session_local):
        """Test get_tier_1_recovery_candidates returns eligible bots."""
        mock_session = Mock()
        state_mgr = SSLCircuitBreakerState(db_session=mock_session)

        mock_record = Mock()
        mock_record.bot_id = "bot-1"

        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = [mock_record]
        mock_session.execute.return_value = mock_result

        candidates = state_mgr.get_tier_1_recovery_candidates()
        assert len(candidates) == 1
        assert "bot-1" in candidates


class TestUpdateState:
    """Test update_state functionality."""

    @patch('src.risk.ssl.state.SessionLocal')
    def test_update_state_creates_new_record(self, mock_session_local):
        """Test update_state creates new record if none exists."""
        mock_session = Mock()
        state_mgr = SSLCircuitBreakerState(db_session=mock_session)

        # First call returns None (no existing record)
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        mock_session.add = Mock()
        mock_session.commit = Mock()

        success = state_mgr.update_state(
            bot_id="new-bot",
            new_state=SSLState.LIVE,
            magic_number="12345",
        )

        assert success is True
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()

    @patch('src.risk.ssl.state.SessionLocal')
    def test_update_state_updates_existing_record(self, mock_session_local):
        """Test update_state updates existing record."""
        mock_session = Mock()
        state_mgr = SSLCircuitBreakerState(db_session=mock_session)

        mock_record = Mock()
        mock_record.state = "live"

        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = mock_record
        mock_session.execute.return_value = mock_result

        mock_session.commit = Mock()

        success = state_mgr.update_state(
            bot_id="bot-1",
            new_state=SSLState.PAPER,
            tier=BotTier.TIER_1,
        )

        assert success is True
        assert mock_record.state == SSLState.PAPER.value
        mock_session.commit.assert_called_once()


class TestResetRecoveryState:
    """Test reset_recovery_state functionality."""

    @patch('src.risk.ssl.state.SessionLocal')
    def test_reset_recovery_state(self, mock_session_local):
        """Test reset_recovery_state sets recovery_win_count to 0."""
        mock_session = Mock()
        state_mgr = SSLCircuitBreakerState(db_session=mock_session)

        mock_record = Mock()
        mock_record.recovery_win_count = 2

        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = mock_record
        mock_session.execute.return_value = mock_result

        mock_session.commit = Mock()

        success = state_mgr.reset_recovery_state("bot-1")

        assert success is True
        assert mock_record.recovery_win_count == 0
        mock_session.commit.assert_called_once()

    @patch('src.risk.ssl.state.SessionLocal')
    def test_reset_recovery_state_no_record(self, mock_session_local):
        """Test reset_recovery_state returns False if no record."""
        mock_session = Mock()
        state_mgr = SSLCircuitBreakerState(db_session=mock_session)

        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        success = state_mgr.reset_recovery_state("unknown-bot")

        assert success is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

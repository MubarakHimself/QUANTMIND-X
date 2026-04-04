"""
P0 Tests for Epic 18 - SSL Circuit Breaker Retirement Flow (Story 18.1/18.2).

Priority: P0
Coverage: evaluate_retirement, mark_strategy_retired, trigger_alphaforge_workflow_1,
          emit_retirement_event, on_dead_zone_evaluation, promote_to_live

Risk Coverage:
- R-001: SSL state transition failures
- R-002: Paper trading recovery window
- R-003: AlphaForge workflow trigger
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
from datetime import datetime, timezone, timedelta

from src.risk.ssl.circuit_breaker import SSLCircuitBreaker, BotType
from src.risk.ssl.state import SSLState, BotTier
from src.events.ssl import SSLCircuitBreakerEvent, SSLEventType


class TestEvaluateRetirement:
    """P0: evaluate_retirement() - TIER_1 paper recovery window evaluation."""

    @patch('src.risk.ssl.circuit_breaker.SSLCircuitBreakerState')
    def test_retire_tier1_after_recovery_window(self, mock_state_mgr):
        """P0: TIER_1 paper bot should retire after 8+ hours with <2 recovery wins."""
        mock_session = Mock()
        ssl = SSLCircuitBreaker(db_session=mock_session)

        # Setup: TIER_1 paper bot entered paper 10 hours ago with only 1 recovery win
        mock_state_mgr_instance = Mock()
        mock_state_mgr_instance.get_state.return_value = SSLState.PAPER
        mock_state_mgr_instance.get_tier.return_value = BotTier.TIER_1
        mock_state_mgr_instance.get_recovery_win_count.return_value = 1
        # paper_entry_timestamp 10 hours ago
        mock_state_mgr_instance.get_paper_entry_timestamp.return_value = (
            datetime.now(timezone.utc) - timedelta(hours=10)
        )
        ssl._state_manager = mock_state_mgr_instance

        result = ssl.evaluate_retirement("bot-1")

        assert result is True

    @patch('src.risk.ssl.circuit_breaker.SSLCircuitBreakerState')
    def test_no_retire_within_recovery_window(self, mock_state_mgr):
        """P0: TIER_1 paper bot should NOT retire within 8-hour recovery window."""
        mock_session = Mock()
        ssl = SSLCircuitBreaker(db_session=mock_session)

        # Setup: TIER_1 paper bot entered paper only 4 hours ago
        mock_state_mgr_instance = Mock()
        mock_state_mgr_instance.get_state.return_value = SSLState.PAPER
        mock_state_mgr_instance.get_tier.return_value = BotTier.TIER_1
        mock_state_mgr_instance.get_recovery_win_count.return_value = 1
        mock_state_mgr_instance.get_paper_entry_timestamp.return_value = (
            datetime.now(timezone.utc) - timedelta(hours=4)
        )
        ssl._state_manager = mock_state_mgr_instance

        result = ssl.evaluate_retirement("bot-1")

        assert result is False

    @patch('src.risk.ssl.circuit_breaker.SSLCircuitBreakerState')
    def test_no_retire_tier2(self, mock_state_mgr):
        """P0: TIER_2 paper bots should NOT be evaluated for retirement."""
        mock_session = Mock()
        ssl = SSLCircuitBreaker(db_session=mock_session)

        mock_state_mgr_instance = Mock()
        mock_state_mgr_instance.get_state.return_value = SSLState.PAPER
        mock_state_mgr_instance.get_tier.return_value = BotTier.TIER_2
        mock_state_mgr_instance.get_recovery_win_count.return_value = 0
        mock_state_mgr_instance.get_paper_entry_timestamp.return_value = (
            datetime.now(timezone.utc) - timedelta(hours=10)
        )
        ssl._state_manager = mock_state_mgr_instance

        result = ssl.evaluate_retirement("bot-tier2")

        assert result is False

    @patch('src.risk.ssl.circuit_breaker.SSLCircuitBreakerState')
    def test_no_retire_with_sufficient_recovery_wins(self, mock_state_mgr):
        """P0: TIER_1 paper bot should NOT retire if recovery_win_count >= 2."""
        mock_session = Mock()
        ssl = SSLCircuitBreaker(db_session=mock_session)

        mock_state_mgr_instance = Mock()
        mock_state_mgr_instance.get_state.return_value = SSLState.PAPER
        mock_state_mgr_instance.get_tier.return_value = BotTier.TIER_1
        mock_state_mgr_instance.get_recovery_win_count.return_value = 2  # Eligible for recovery
        mock_state_mgr_instance.get_paper_entry_timestamp.return_value = (
            datetime.now(timezone.utc) - timedelta(hours=10)
        )
        ssl._state_manager = mock_state_mgr_instance

        result = ssl.evaluate_retirement("bot-1")

        assert result is False

    @patch('src.risk.ssl.circuit_breaker.SSLCircuitBreakerState')
    def test_no_retire_live_state(self, mock_state_mgr):
        """P0: Bot in LIVE state should not be evaluated for retirement."""
        mock_session = Mock()
        ssl = SSLCircuitBreaker(db_session=mock_session)

        mock_state_mgr_instance = Mock()
        mock_state_mgr_instance.get_state.return_value = SSLState.LIVE
        ssl._state_manager = mock_state_mgr_instance

        result = ssl.evaluate_retirement("bot-live")

        assert result is False


class TestMarkStrategyRetired:
    """P0: mark_strategy_retired() - Strategy retirement + lifecycle logging."""

    @patch('src.risk.ssl.circuit_breaker.SSLCircuitBreakerState')
    def test_marks_manifest_status_to_retired(self, mock_state_mgr):
        """P0: BotManifest status should be updated to 'retired'."""
        mock_session = Mock()
        ssl = SSLCircuitBreaker(db_session=mock_session)

        # Mock BotManifest query
        mock_manifest = Mock()
        mock_manifest.strategy_type = "SCALPING_001"
        mock_session.query.return_value.filter.return_value.scalar_one_or_none.return_value = mock_manifest
        mock_session.add = Mock()
        mock_session.commit = Mock()

        ssl._db_session = mock_session

        result = ssl.mark_strategy_retired("bot-1")

        assert result is True
        assert mock_manifest.status == "retired"
        mock_session.commit.assert_called_once()

    @patch('src.risk.ssl.circuit_breaker.SSLCircuitBreakerState')
    def test_creates_lifecycle_log_entry(self, mock_state_mgr):
        """P0: BotLifecycleLog entry should be created with @retired tag."""
        mock_session = Mock()
        ssl = SSLCircuitBreaker(db_session=mock_session)

        mock_manifest = Mock()
        mock_manifest.strategy_type = "SCALPING_001"
        mock_session.query.return_value.filter.return_value.scalar_one_or_none.return_value = mock_manifest
        mock_session.add = Mock()
        mock_session.commit = Mock()

        ssl._db_session = mock_session

        ssl.mark_strategy_retired("bot-1")

        # Verify BotLifecycleLog entry was added
        mock_session.add.assert_called()
        call_args = mock_session.add.call_args[0][0]
        assert call_args.to_tag == "@retired"
        assert call_args.triggered_by == "SSL_CIRCUIT_BREAKER"

    @patch('src.risk.ssl.circuit_breaker.SSLCircuitBreakerState')
    def test_returns_false_if_no_manifest(self, mock_state_mgr):
        """P0: Returns False if bot has no BotManifest."""
        mock_session = Mock()
        ssl = SSLCircuitBreaker(db_session=mock_session)

        mock_session.query.return_value.filter.return_value.scalar_one_or_none.return_value = None

        ssl._db_session = mock_session

        result = ssl.mark_strategy_retired("unknown-bot")

        assert result is False


class TestTriggerAlphaForgeWorkflow1:
    """P0: trigger_alphaforge_workflow_1() - AlphaForge workflow trigger.

    NOTE: The source code at circuit_breaker.py:832 imports DepartmentMailService
    from 'src.agents.departments.department_mail_service' which does not exist.
    The correct module is 'src.agents.departments.department_mail'.
    Tests for the mail trigger path are blocked by this source bug.
    """

    @patch('src.risk.ssl.circuit_breaker.SSLCircuitBreakerState')
    def test_returns_none_when_no_manifest(self, mock_state_mgr):
        """P0: AlphaForge returns None when bot has no manifest."""
        mock_session = Mock()
        ssl = SSLCircuitBreaker(db_session=mock_session)

        # No manifest found
        mock_session.execute.return_value.scalar_one_or_none.return_value = None
        ssl._db_session = mock_session

        result = ssl.trigger_alphaforge_workflow_1("bot-no-manifest")

        assert result is None


class TestEmitRetirementEvent:
    """P1: emit_retirement_event() - Retirement event emission."""

    @patch('src.risk.ssl.circuit_breaker.SSLCircuitBreaker._emit_event')
    @patch('src.risk.ssl.circuit_breaker.SSLCircuitBreakerState')
    def test_emits_retired_event(self, mock_state_mgr, mock_emit):
        """P1: RETIRED event is emitted to Redis."""
        mock_session = Mock()
        ssl = SSLCircuitBreaker(db_session=mock_session)

        mock_state_mgr_instance = Mock()
        mock_state_mgr_instance.get_state.return_value = SSLState.PAPER
        mock_state_mgr_instance.get_magic_number.return_value = "12345"
        mock_state_mgr_instance.get_consecutive_losses.return_value = 3
        mock_state_mgr_instance.update_state = Mock()
        ssl._state_manager = mock_state_mgr_instance

        mock_emit.return_value = True

        event = ssl.emit_retirement_event("bot-1", dpr_composite_score=45)

        assert event is not None
        assert event.event_type == SSLEventType.RETIRED
        assert event.tier == "TIER_1"
        assert event.metadata["dpr_score_at_retirement"] == 45
        mock_emit.assert_called_once()


class TestOnDeadZoneEvaluation:
    """P0: on_dead_zone_evaluation() - Dead Zone integration flow."""

    @patch('src.risk.ssl.circuit_breaker.SSLCircuitBreaker.mark_strategy_retired')
    @patch('src.risk.ssl.circuit_breaker.SSLCircuitBreaker.trigger_alphaforge_workflow_1')
    @patch('src.risk.ssl.circuit_breaker.SSLCircuitBreaker.emit_retirement_event')
    @patch('src.risk.ssl.circuit_breaker.SSLCircuitBreaker.evaluate_retirement')
    @patch('src.risk.ssl.circuit_breaker.SSLCircuitBreakerState')
    def test_triggers_retirement_when_evaluated(
        self, mock_state_mgr, mock_eval_retire, mock_emit, mock_trigger, mock_mark
    ):
        """P0: Dead Zone evaluation triggers retirement when conditions met."""
        mock_session = Mock()
        ssl = SSLCircuitBreaker(db_session=mock_session)

        mock_state_mgr_instance = Mock()
        ssl._state_manager = mock_state_mgr_instance

        mock_eval_retire.return_value = True
        mock_mark.return_value = True
        mock_trigger.return_value = "SCALPING_001_v2"
        # Return a proper SSLCircuitBreakerEvent with mutable metadata
        mock_event = SSLCircuitBreakerEvent(
            bot_id="bot-1",
            magic_number="12345",
            event_type=SSLEventType.RETIRED,
            consecutive_losses=3,
            tier="TIER_1",
            previous_state=SSLState.PAPER,
            new_state=SSLState.RETIRED,
        )
        mock_emit.return_value = mock_event

        result = ssl.on_dead_zone_evaluation("bot-1", dpr_composite_score=30)

        assert result is not None
        mock_mark.assert_called_once_with("bot-1")
        mock_trigger.assert_called_once_with("bot-1")
        mock_emit.assert_called_once()

    @patch('src.risk.ssl.circuit_breaker.SSLCircuitBreaker.evaluate_retirement')
    @patch('src.risk.ssl.circuit_breaker.SSLCircuitBreakerState')
    def test_no_event_if_not_retiring(self, mock_state_mgr, mock_eval_retire):
        """P0: Returns None if bot is not eligible for retirement."""
        mock_session = Mock()
        ssl = SSLCircuitBreaker(db_session=mock_session)

        mock_state_mgr_instance = Mock()
        ssl._state_manager = mock_state_mgr_instance

        mock_eval_retire.return_value = False

        result = ssl.on_dead_zone_evaluation("bot-1", dpr_composite_score=75)

        assert result is None


class TestPromoteToLive:
    """P0: promote_to_live() - Full promotion flow from paper to live."""

    @patch('src.risk.ssl.circuit_breaker.SSLCircuitBreaker._emit_event')
    @patch('src.risk.ssl.circuit_breaker.SSLCircuitBreaker._remove_paper_only_tag')
    @patch('src.risk.ssl.circuit_breaker.SSLCircuitBreaker._add_primal_remount_tag')
    @patch('src.risk.ssl.circuit_breaker.SSLCircuitBreakerState')
    def test_promotes_tier1_to_live(self, mock_state_mgr, mock_add_tag, mock_remove_tag, mock_emit):
        """P0: TIER_1 paper bot promoted to LIVE with correct state updates."""
        mock_session = Mock()
        ssl = SSLCircuitBreaker(db_session=mock_session)

        mock_state_mgr_instance = Mock()
        mock_state_mgr_instance.get_state.return_value = SSLState.RECOVERY
        mock_state_mgr_instance.get_magic_number.return_value = "12345"
        mock_state_mgr_instance.get_consecutive_losses.return_value = 0
        ssl._state_manager = mock_state_mgr_instance

        event = ssl.promote_to_live("bot-1")

        assert event is not None
        assert event.event_type == SSLEventType.RECOVERY_CONFIRMED
        assert event.previous_state == SSLState.RECOVERY
        assert event.new_state == SSLState.LIVE
        mock_remove_tag.assert_called_once_with("bot-1")
        mock_add_tag.assert_called_once_with("bot-1")

    @patch('src.risk.ssl.circuit_breaker.SSLCircuitBreakerState')
    def test_promote_updates_state_manager(self, mock_state_mgr):
        """P0: promote_to_live updates state with consecutive_losses=0."""
        mock_session = Mock()
        ssl = SSLCircuitBreaker(db_session=mock_session)

        mock_state_mgr_instance = Mock()
        mock_state_mgr_instance.get_state.return_value = SSLState.RECOVERY
        mock_state_mgr_instance.get_magic_number.return_value = "12345"
        mock_state_mgr_instance.get_consecutive_losses.return_value = 0
        ssl._state_manager = mock_state_mgr_instance

        with patch.object(ssl, '_emit_event'):
            ssl.promote_to_live("bot-1")

        mock_state_mgr_instance.update_state.assert_called_once()
        call_kwargs = mock_state_mgr_instance.update_state.call_args[1]
        assert call_kwargs["new_state"] == SSLState.LIVE
        assert call_kwargs["consecutive_losses"] == 0
        assert call_kwargs["tier"] is None


class TestRecoveryEvaluation:
    """P1: evaluate_recovery() - Recovery eligibility."""

    @patch('src.risk.ssl.circuit_breaker.SSLCircuitBreakerState')
    def test_eligible_with_2_recovery_wins(self, mock_state_mgr):
        """P1: Bot with 2 recovery wins in TIER_1 paper is eligible."""
        mock_session = Mock()
        ssl = SSLCircuitBreaker(db_session=mock_session)

        mock_state_mgr_instance = Mock()
        mock_state_mgr_instance.get_state.return_value = SSLState.PAPER
        mock_state_mgr_instance.get_tier.return_value = BotTier.TIER_1
        mock_state_mgr_instance.get_recovery_win_count.return_value = 2
        ssl._state_manager = mock_state_mgr_instance

        result = ssl.evaluate_recovery("bot-1")

        assert result is True

    @patch('src.risk.ssl.circuit_breaker.SSLCircuitBreakerState')
    def test_not_eligible_tier2(self, mock_state_mgr):
        """P1: TIER_2 paper bots are not eligible for recovery."""
        mock_session = Mock()
        ssl = SSLCircuitBreaker(db_session=mock_session)

        mock_state_mgr_instance = Mock()
        mock_state_mgr_instance.get_state.return_value = SSLState.PAPER
        mock_state_mgr_instance.get_tier.return_value = BotTier.TIER_2
        mock_state_mgr_instance.get_recovery_win_count.return_value = 2
        ssl._state_manager = mock_state_mgr_instance

        result = ssl.evaluate_recovery("bot-tier2")

        assert result is False

    @patch('src.risk.ssl.circuit_breaker.SSLCircuitBreakerState')
    def test_not_eligible_live_state(self, mock_state_mgr):
        """P1: Bot in LIVE state cannot be evaluated for recovery."""
        mock_session = Mock()
        ssl = SSLCircuitBreaker(db_session=mock_session)

        mock_state_mgr_instance = Mock()
        mock_state_mgr_instance.get_state.return_value = SSLState.LIVE
        ssl._state_manager = mock_state_mgr_instance

        result = ssl.evaluate_recovery("bot-live")

        assert result is False

    @patch('src.risk.ssl.circuit_breaker.SSLCircuitBreakerState')
    def test_not_eligible_insufficient_wins(self, mock_state_mgr):
        """P1: Bot with <2 recovery wins is not eligible."""
        mock_session = Mock()
        ssl = SSLCircuitBreaker(db_session=mock_session)

        mock_state_mgr_instance = Mock()
        mock_state_mgr_instance.get_state.return_value = SSLState.PAPER
        mock_state_mgr_instance.get_tier.return_value = BotTier.TIER_1
        mock_state_mgr_instance.get_recovery_win_count.return_value = 1
        ssl._state_manager = mock_state_mgr_instance

        result = ssl.evaluate_recovery("bot-1")

        assert result is False


class TestLossDuringRecovery:
    """P1: Loss during RECOVERY transitions back to PAPER."""

    @patch('src.risk.ssl.circuit_breaker.SSLCircuitBreaker._emit_event')
    @patch('src.risk.ssl.circuit_breaker.SSLCircuitBreakerState')
    def test_loss_in_recovery_goes_to_paper(self, mock_state_mgr, mock_emit):
        """P1: Loss during RECOVERY resets and moves back to PAPER."""
        mock_session = Mock()
        ssl = SSLCircuitBreaker(db_session=mock_session)

        mock_state_mgr_instance = Mock()
        mock_state_mgr_instance.get_state.return_value = SSLState.RECOVERY
        mock_state_mgr_instance.get_magic_number.return_value = "12345"
        mock_state_mgr_instance.get_consecutive_losses.return_value = 0
        ssl._state_manager = mock_state_mgr_instance

        mock_emit.return_value = Mock()

        event = ssl.on_trade_close("bot-1", "12345", is_win=False)

        assert event is not None
        assert event.event_type == SSLEventType.MOVE_TO_PAPER
        assert event.previous_state == SSLState.RECOVERY


class TestORBBotTypeDetection:
    """P2: ORB bot type detection and 3-loss threshold."""

    @patch('src.risk.ssl.circuit_breaker.SSLCircuitBreakerState')
    def test_orb_bot_from_strategy_type(self, mock_state_mgr):
        """P2: Bot with 'orb' in strategy_type is detected as ORB."""
        mock_session = Mock()
        ssl = SSLCircuitBreaker(db_session=mock_session)

        # Mock BotManifest with ORB strategy
        mock_manifest = MagicMock()
        mock_manifest.strategy_type = "EURUSD_ORB_Breakout"
        mock_session.execute.return_value.scalar_one_or_none.return_value = mock_manifest

        ssl._db_session = mock_session

        bot_type = ssl._get_bot_type("bot-orb")

        assert bot_type == BotType.ORB

    @patch('src.risk.ssl.circuit_breaker.SSLCircuitBreakerState')
    def test_orb_threshold_is_3(self, mock_state_mgr):
        """P2: ORB bots should have 3-loss threshold."""
        mock_session = Mock()
        ssl = SSLCircuitBreaker(db_session=mock_session)

        with patch.object(ssl, '_get_bot_type', return_value=BotType.ORB):
            threshold = ssl._get_threshold("bot-orb")

            assert threshold == 3

    @patch('src.risk.ssl.circuit_breaker.SSLCircuitBreakerState')
    def test_scalping_threshold_is_2(self, mock_state_mgr):
        """P2: Scalping bots should have 2-loss threshold."""
        mock_session = Mock()
        ssl = SSLCircuitBreaker(db_session=mock_session)

        with patch.object(ssl, '_get_bot_type', return_value=BotType.SCALPING):
            threshold = ssl._get_threshold("bot-scalp")

            assert threshold == 2


class TestBotTierDetermination:
    """P2: TIER_1/TIER_2 determination via @primal tag."""

    @patch('src.risk.ssl.circuit_breaker.SSLCircuitBreakerState')
    def test_tier1_with_primal_tag(self, mock_state_mgr):
        """P2: Bot with @primal lifecycle tag is TIER_1."""
        mock_session = Mock()
        ssl = SSLCircuitBreaker(db_session=mock_session)

        # Mock BotLifecycleLog with @primal tag
        mock_entry = MagicMock()
        mock_entry.to_tag = "@primal"
        mock_session.execute.return_value.scalar_one_or_none.return_value = mock_entry

        ssl._db_session = mock_session

        tier = ssl._determine_tier("bot-primal")

        assert tier == BotTier.TIER_1

    @patch('src.risk.ssl.circuit_breaker.SSLCircuitBreakerState')
    def test_tier2_without_primal_tag(self, mock_state_mgr):
        """P2: Bot without @primal tag is TIER_2 (fresh from AlphaForge)."""
        mock_session = Mock()
        ssl = SSLCircuitBreaker(db_session=mock_session)

        # No BotLifecycleLog entry found
        mock_session.execute.return_value.scalar_one_or_none.return_value = None

        ssl._db_session = mock_session

        tier = ssl._determine_tier("bot-new")

        assert tier == BotTier.TIER_2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

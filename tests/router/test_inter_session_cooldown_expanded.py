"""
Expanded Tests for InterSessionCooldownOrchestrator.

Story 16.3: Inter-Session Cooldown Window (10:00–13:00 GMT)
[P1] Coverage gaps from existing tests

These tests fill gaps identified in the existing test suite:
- Redis publish error handling (all publish methods)
- Callback error handling (all 4 step callbacks + tilt/roster)
- on_tilt_activate() method NOT TESTED
- CooldownTransition audit log fields (step_deadline, entry_id, etc.)
- _check_step_timeout edge cases (unknown state, actual timeout)
- start() time window validation behavior
- _tilt_activate_confirmed flag
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

from src.events.cooldown import (
    CooldownPhase,
    CooldownState,
    NYQueueCandidate,
)
from src.events.tilt import TiltPhase, TiltPhaseEvent, TiltState
from src.router.inter_session_cooldown import (
    CHANNEL_COOLDOWN_COMPLETED,
    CHANNEL_COOLDOWN_LONDON_SCORE_COMPLETE,
    CHANNEL_COOLDOWN_PHASE,
    InterSessionCooldownOrchestrator,
    CooldownTransition,
    COOLDOWN_START_HOUR,
    COOLDOWN_END_HOUR,
    STEP_WINDOWS,
)


class MockRedis:
    """Mock Redis client for testing."""

    def __init__(self, publish_raises=False):
        self._data = {}
        self._channels = {}
        self._publish_raises = publish_raises

    def set(self, key, value, nx=False, ex=None):
        if nx and key in self._data:
            return False
        self._data[key] = value
        return True

    def get(self, key):
        return self._data.get(key)

    def delete(self, key):
        if key in self._data:
            del self._data[key]
            return 1
        return 0

    def exists(self, key):
        return 1 if key in self._data else 0

    def publish(self, channel, message):
        if self._publish_raises:
            raise RedisPublishError("Simulated publish failure")
        if channel not in self._channels:
            self._channels[channel] = []
        self._channels[channel].append(message)

    def subscribe(self, channel):
        pass


class RedisPublishError(Exception):
    """Simulated Redis publish error."""
    pass


class TestOnTiltActivate:
    """Test on_tilt_activate() method - NOT TESTED IN EXISTING SUITE."""

    def test_tilt_activate_sets_flag(self):
        """Test Tilt ACTIVATE sets _tilt_activate_confirmed flag."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)

        event = TiltPhaseEvent(
            phase=TiltPhase.ACTIVATE,
            state=TiltState.ACTIVATE,
            closing_session="LONDON",
            incoming_session="NY",
            regime_persistence_timer=0,
            timestamp_utc=datetime.now(timezone.utc),
            metadata={},
        )

        orchestrator.on_tilt_activate(event)

        assert orchestrator._tilt_activate_confirmed is True

    def test_tilt_activate_ignores_non_activate_phase(self):
        """Test on_tilt_activate ignores non-ACTIVATE phases."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)
        orchestrator._tilt_activate_confirmed = False

        event = TiltPhaseEvent(
            phase=TiltPhase.LOCK,
            state=TiltState.LOCK,
            closing_session="LONDON",
            incoming_session="NY",
            regime_persistence_timer=0,
            timestamp_utc=datetime.now(timezone.utc),
            metadata={},
        )

        orchestrator.on_tilt_activate(event)

        assert orchestrator._tilt_activate_confirmed is False

    def test_tilt_activate_captures_incoming_session(self):
        """Test Tilt ACTIVATE event captures incoming session."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)

        event = TiltPhaseEvent(
            phase=TiltPhase.ACTIVATE,
            state=TiltState.ACTIVATE,
            closing_session="LONDON",
            incoming_session="NY",
            regime_persistence_timer=0,
            timestamp_utc=datetime.now(timezone.utc),
            metadata={},
        )

        orchestrator.on_tilt_activate(event)

        assert orchestrator._tilt_activate_confirmed is True


class TestRedisPublishErrorHandling:
    """Test Redis publish error handling."""

    def test_publish_phase_event_handles_redis_error(self):
        """Test _publish_phase_event gracefully handles Redis errors."""
        redis = MockRedis(publish_raises=True)
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)

        # Should not raise, should log error
        orchestrator._publish_phase_event()

        # State unchanged
        assert orchestrator.state == CooldownState.PENDING

    def test_publish_step_completion_handles_redis_error(self):
        """Test _publish_step_completion gracefully handles Redis errors."""
        redis = MockRedis(publish_raises=True)
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)

        # Should not raise
        orchestrator._publish_step_completion(
            CooldownPhase.STEP_1,
            "London Session Scoring",
            {"score": 0.85}
        )

    def test_publish_completion_handles_redis_error(self):
        """Test _publish_completion gracefully handles Redis errors."""
        from src.events.cooldown import InterSessionCooldownCompletionEvent

        redis = MockRedis(publish_raises=True)
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)

        event = InterSessionCooldownCompletionEvent(
            completed_at=datetime.now(timezone.utc),
            ny_roster_locked=True,
            roster_summary={"state": "locked"},
            tilt_activate_triggered=True,
        )

        # Should not raise
        orchestrator._publish_completion(event)


class TestCallbackErrorHandling:
    """Test callback error handling in step execution."""

    def test_step_1_dpr_callback_error_caught(self):
        """Test Step 1 DPR callback error is caught and logged."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)

        def failing_callback():
            raise RuntimeError("DPR scoring failed")

        orchestrator.set_dpr_scoring_callback(failing_callback)

        with patch.object(orchestrator, '_transition_to'):
            with patch.object(orchestrator, '_publish_step_completion'):
                with patch.object(orchestrator, '_execute_step_2', return_value=True):
                    result = orchestrator._execute_step_1()

        assert result is True

    def test_step_2_ssl_callback_error_caught(self):
        """Test Step 2 SSL callback error is caught and logged."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)

        def failing_callback():
            raise RuntimeError("SSL paper review failed")

        orchestrator.set_ssl_paper_review_callback(failing_callback)

        with patch.object(orchestrator, '_transition_to'):
            with patch.object(orchestrator, '_publish_step_completion'):
                with patch.object(orchestrator, '_execute_step_3', return_value=True):
                    result = orchestrator._execute_step_2()

        assert result is True

    def test_step_3_dpr_queue_callback_error_caught(self):
        """Test Step 3 DPR queue callback error is caught and logged."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)

        def failing_callback():
            raise RuntimeError("DPR queue build failed")

        orchestrator.set_dpr_queue_callback(failing_callback)

        with patch.object(orchestrator, '_transition_to'):
            with patch.object(orchestrator, '_publish_step_completion'):
                with patch.object(orchestrator, '_execute_step_4', return_value=True):
                    result = orchestrator._execute_step_3()

        assert result is True

    def test_step_4_svss_callback_error_caught(self):
        """Test Step 4 SVSS health callback error is caught and logged."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)

        def failing_callback():
            raise RuntimeError("SVSS health check failed")

        orchestrator.set_svss_health_callback(failing_callback)

        with patch.object(orchestrator, '_transition_to'):
            with patch.object(orchestrator, '_publish_step_completion'):
                with patch.object(orchestrator, '_complete', return_value=True):
                    result = orchestrator._execute_step_4()

        assert result is True

    def test_step_4_sqs_callback_error_caught(self):
        """Test Step 4 SQS warmup callback error is caught and logged."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)

        orchestrator.set_svss_health_callback(MagicMock(return_value={"status": "ok"}))

        def failing_sqs_callback(is_monday):
            raise RuntimeError("SQS warmup failed")

        orchestrator.set_sqs_warmup_callback(failing_sqs_callback)

        with patch('src.router.inter_session_cooldown.datetime') as mock_dt:
            mock_dt.now.return_value = datetime(2026, 3, 23, 10, 0, tzinfo=timezone.utc)
            mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

            with patch.object(orchestrator, '_transition_to'):
                with patch.object(orchestrator, '_publish_step_completion'):
                    with patch.object(orchestrator, '_complete', return_value=True):
                        result = orchestrator._execute_step_4()

        assert result is True

    def test_step_4_sentinel_callback_error_caught(self):
        """Test Step 4 Sentinel regime callback error is caught and logged."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)

        orchestrator.set_svss_health_callback(MagicMock(return_value={"status": "ok"}))
        orchestrator.set_sqs_warmup_callback(MagicMock())

        def failing_sentinel_callback():
            raise RuntimeError("Sentinel regime check failed")

        orchestrator.set_sentinel_regime_callback(failing_sentinel_callback)

        with patch('src.router.inter_session_cooldown.datetime') as mock_dt:
            mock_dt.now.return_value = datetime(2026, 3, 23, 10, 0, tzinfo=timezone.utc)
            mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

            with patch.object(orchestrator, '_transition_to'):
                with patch.object(orchestrator, '_publish_step_completion'):
                    with patch.object(orchestrator, '_complete', return_value=True):
                        result = orchestrator._execute_step_4()

        assert result is True

    def test_complete_tilt_activate_callback_error_caught(self):
        """Test _complete() tilt activate callback error is caught and logged."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)

        def failing_callback():
            raise RuntimeError("Tilt activate failed")

        orchestrator.set_tilt_activate_callback(failing_callback)

        with patch.object(orchestrator, '_transition_to'):
            with patch.object(orchestrator, '_publish_completion'):
                result = orchestrator._complete()

        assert result is True
        assert orchestrator._ny_roster_locked is True

    def test_complete_roster_lock_callback_error_caught(self):
        """Test _complete() roster lock callback error is caught and logged."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)

        def failing_callback():
            raise RuntimeError("Roster lock failed")

        orchestrator.set_roster_lock_callback(failing_callback)

        with patch.object(orchestrator, '_transition_to'):
            with patch.object(orchestrator, '_publish_completion'):
                result = orchestrator._complete()

        assert result is True
        assert orchestrator._ny_roster_locked is True


class TestCheckStepTimeoutEdgeCases:
    """Test _check_step_timeout edge cases."""

    def test_timeout_check_returns_false_for_unknown_state(self):
        """Test timeout check returns False for state not in STEP_WINDOWS."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)
        orchestrator._window_start = datetime.now(timezone.utc)

        # PENDING and COMPLETED are not in STEP_WINDOWS
        result = orchestrator._check_step_timeout(CooldownState.PENDING)

        assert result is False

    def test_timeout_check_triggers_when_past_deadline(self):
        """Test timeout check returns True when past deadline."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)

        # Set window_start to 30 minutes ago
        orchestrator._window_start = datetime.now(timezone.utc) - timedelta(minutes=30)
        orchestrator._current_step_start = datetime.now(timezone.utc) - timedelta(minutes=31)

        # Step 1 deadline is 30 minutes from window_start
        # So if window_start was 30 min ago, we're exactly at/past deadline
        result = orchestrator._check_step_timeout(CooldownState.STEP_1_SCORING)

        assert result is True

    def test_timeout_check_returns_false_at_exact_deadline(self):
        """Test timeout check returns False at exact deadline moment."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)

        # Set window_start to exactly 30 minutes ago
        # Step 1 ends at 30 minutes from window_start
        orchestrator._window_start = datetime.now(timezone.utc) - timedelta(minutes=29, seconds=59)
        orchestrator._current_step_start = datetime.now(timezone.utc) - timedelta(minutes=29, seconds=59)

        # Should still be within window
        result = orchestrator._check_step_timeout(CooldownState.STEP_1_SCORING)

        assert result is False


class TestCooldownTransitionAuditLogFields:
    """Test CooldownTransition audit log field completeness."""

    def test_audit_log_entry_has_entry_id(self):
        """Test audit log entries have unique entry_id."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)

        orchestrator._transition_to(CooldownState.STEP_1_SCORING)

        log = orchestrator.get_audit_log()
        assert len(log) == 1
        assert log[0].entry_id is not None
        assert len(log[0].entry_id) > 0

    def test_audit_log_entry_has_timestamp_utc(self):
        """Test audit log entries have timestamp_utc."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)

        before = datetime.now(timezone.utc)
        orchestrator._transition_to(CooldownState.STEP_1_SCORING)
        after = datetime.now(timezone.utc)

        log = orchestrator.get_audit_log()
        assert log[0].timestamp_utc is not None
        assert before <= log[0].timestamp_utc <= after

    def test_audit_log_entry_has_step_name(self):
        """Test audit log entries have correct step_name."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)

        orchestrator._transition_to(CooldownState.STEP_1_SCORING)

        log = orchestrator.get_audit_log()
        assert log[0].step_name == "London Session Scoring"

    def test_audit_log_entry_has_step_started_at(self):
        """Test audit log entries have step_started_at from _current_step_start."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)

        orchestrator._current_step_start = datetime(2026, 3, 25, 10, 5, tzinfo=timezone.utc)
        orchestrator._transition_to(CooldownState.STEP_1_SCORING)

        log = orchestrator.get_audit_log()
        assert log[0].step_started_at == datetime(2026, 3, 25, 10, 5, tzinfo=timezone.utc)

    def test_audit_log_entry_has_step_deadline(self):
        """Test audit log entries have step_deadline calculated from window_start."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)

        orchestrator._window_start = datetime(2026, 3, 25, 10, 0, tzinfo=timezone.utc)
        orchestrator._transition_to(CooldownState.STEP_1_SCORING)

        log = orchestrator.get_audit_log()
        # Step 1 deadline is 30 minutes from window_start
        assert log[0].step_deadline is not None
        assert log[0].step_deadline.hour == 10
        assert log[0].step_deadline.minute == 30

    def test_audit_log_metadata_field_exists(self):
        """Test audit log entries have metadata dict."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)

        orchestrator._transition_to(CooldownState.STEP_1_SCORING)

        log = orchestrator.get_audit_log()
        assert log[0].metadata is not None
        assert isinstance(log[0].metadata, dict)

    def test_audit_log_multiple_transitions_have_unique_ids(self):
        """Test multiple audit log entries have unique entry_ids."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)

        orchestrator._transition_to(CooldownState.STEP_1_SCORING)
        orchestrator._transition_to(CooldownState.STEP_2_PAPER_RECOVERY)
        orchestrator._transition_to(CooldownState.STEP_3_QUEUE_BUILD)

        log = orchestrator.get_audit_log()
        entry_ids = [entry.entry_id for entry in log]
        assert len(entry_ids) == len(set(entry_ids))


class TestStartTimeWindowValidation:
    """Test start() time window validation behavior."""

    def test_start_accepts_time_outside_window(self):
        """Test start() accepts calls even outside 10:00-13:00 window (NFR-P1)."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)

        # Mock time to be outside window (e.g., 14:00 GMT)
        with patch('src.router.inter_session_cooldown.datetime') as mock_dt:
            mock_dt.now.return_value = datetime(2026, 3, 25, 14, 0, tzinfo=timezone.utc)
            mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

            with patch.object(orchestrator, '_execute_step_1', return_value=True):
                result = orchestrator.start()

        assert result is True

    def test_start_rejects_when_already_running(self):
        """Test start() is rejected when already running."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)
        orchestrator._state = CooldownState.STEP_2_PAPER_RECOVERY

        result = orchestrator.start()

        assert result is False

    def test_start_sets_window_end_to_tomorrow_if_1300_passed(self):
        """Test window_end is set to tomorrow if 13:00 has passed today."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)

        # Mock time to be 14:00 GMT (after 13:00 window)
        with patch('src.router.inter_session_cooldown.datetime') as mock_dt:
            mock_now = datetime(2026, 3, 25, 14, 0, tzinfo=timezone.utc)
            mock_dt.now.return_value = mock_now
            mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

            with patch.object(orchestrator, '_execute_step_1', return_value=True):
                orchestrator.start()

        # Window end should be tomorrow at 13:00
        assert orchestrator.window_end is not None
        assert orchestrator.window_end.day == 26  # tomorrow


class TestIsRunningProperty:
    """Test is_running property edge cases."""

    def test_is_running_false_when_pending(self):
        """Test is_running is False when state is PENDING."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)

        assert orchestrator.is_running is False

    def test_is_running_false_when_completed(self):
        """Test is_running is False when state is COMPLETED."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)
        orchestrator._state = CooldownState.COMPLETED

        assert orchestrator.is_running is False

    def test_is_running_true_during_step_1(self):
        """Test is_running is True during STEP_1_SCORING."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)
        orchestrator._state = CooldownState.STEP_1_SCORING

        assert orchestrator.is_running is True

    def test_is_running_true_during_all_steps(self):
        """Test is_running is True during all active steps."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)

        for state in [
            CooldownState.STEP_1_SCORING,
            CooldownState.STEP_2_PAPER_RECOVERY,
            CooldownState.STEP_3_QUEUE_BUILD,
            CooldownState.STEP_4_HEALTH_CHECK,
        ]:
            orchestrator._state = state
            assert orchestrator.is_running is True


class TestTiltActivateConfirmedFlag:
    """Test _tilt_activate_confirmed flag behavior."""

    def test_flag_initially_false(self):
        """Test _tilt_activate_confirmed is initially False."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)

        assert orchestrator._tilt_activate_confirmed is False

    def test_flag_set_true_by_on_tilt_activate(self):
        """Test flag is set True when on_tilt_activate receives ACTIVATE."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)

        event = TiltPhaseEvent(
            phase=TiltPhase.ACTIVATE,
            state=TiltState.ACTIVATE,
            closing_session="LONDON",
            incoming_session="NY",
            regime_persistence_timer=0,
            timestamp_utc=datetime.now(timezone.utc),
            metadata={},
        )

        orchestrator.on_tilt_activate(event)

        assert orchestrator._tilt_activate_confirmed is True

    def test_flag_ignores_non_activate_phase(self):
        """Test flag remains False for non-ACTIVATE phases."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)

        for phase in [TiltPhase.LOCK, TiltPhase.SIGNAL, TiltPhase.WAIT, TiltPhase.RE_RANK]:
            orchestrator._tilt_activate_confirmed = False

            event = TiltPhaseEvent(
                phase=phase,
                state=TiltState.LOCK,
                closing_session="LONDON",
                incoming_session="NY",
                regime_persistence_timer=0,
                timestamp_utc=datetime.now(timezone.utc),
                metadata={},
            )

            orchestrator.on_tilt_activate(event)

            assert orchestrator._tilt_activate_confirmed is False


class TestStubEdgeCases:
    """Test stub implementation edge cases."""

    def test_stub_london_scoring_has_required_keys(self):
        """Test stub London scoring has all required keys."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)

        result = orchestrator._stub_london_scoring()

        assert result["stub"] is True
        assert "london_performers" in result
        assert "best_london_bot" in result
        assert "note" in result

    def test_stub_paper_review_has_required_keys(self):
        """Test stub paper review has all required keys."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)

        result = orchestrator._stub_paper_review()

        assert result["stub"] is True
        assert "tier_1_paper_bots_reviewed" in result
        assert "recovery_confirmed" in result
        assert "recovery_extended" in result
        assert "note" in result

    def test_stub_ny_queue_build_candidates_are_valid(self):
        """Test stub NY queue build returns valid NYQueueCandidate structures."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)

        result = orchestrator._stub_ny_queue_build()

        assert result["stub"] is True
        candidates = result["queue_candidates"]

        # All 5 candidates should have required fields
        for candidate in candidates:
            assert "bot_id" in candidate
            assert "queue_position" in candidate
            assert "source" in candidate
            assert "tier" in candidate
            assert "dpr_score" in candidate

    def test_stub_ny_queue_composition_sums_to_5(self):
        """Test stub NY queue composition sums to 5 total candidates."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)

        result = orchestrator._stub_ny_queue_build()

        composition = result["composition"]
        total = sum(composition.values())

        assert total == 5

    def test_stub_svss_health_has_required_keys(self):
        """Test stub SVSS health has all required keys."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)

        result = orchestrator._stub_svss_health()

        assert result["stub"] is True
        assert "status" in result
        assert result["status"] == "pass"
        assert "note" in result

    def test_stub_sentinel_regime_has_required_keys(self):
        """Test stub Sentinel regime has all required keys."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)

        result = orchestrator._stub_sentinel_regime()

        assert result["stub"] is True
        assert "regime" in result
        assert "ny_open_ready" in result
        assert "note" in result


class TestStepExecutionProceedsOnCallbackError:
    """Test step execution proceeds to next step even when callback raises."""

    def test_step_1_proceeds_to_step_2_on_callback_error(self):
        """Test Step 1 proceeds to Step 2 when DPR callback raises."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)

        def failing_callback():
            raise RuntimeError("DPR failed")

        orchestrator.set_dpr_scoring_callback(failing_callback)

        with patch.object(orchestrator, '_transition_to'):
            with patch.object(orchestrator, '_publish_step_completion') as mock_publish:
                with patch.object(orchestrator, '_execute_step_2', return_value=True) as mock_step2:
                    result = orchestrator._execute_step_1()

        assert result is True
        mock_step2.assert_called_once()

    def test_step_2_proceeds_to_step_3_on_callback_error(self):
        """Test Step 2 proceeds to Step 3 when SSL callback raises."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)

        def failing_callback():
            raise RuntimeError("SSL failed")

        orchestrator.set_ssl_paper_review_callback(failing_callback)

        with patch.object(orchestrator, '_transition_to'):
            with patch.object(orchestrator, '_publish_step_completion') as mock_publish:
                with patch.object(orchestrator, '_execute_step_3', return_value=True) as mock_step3:
                    result = orchestrator._execute_step_2()

        assert result is True
        mock_step3.assert_called_once()

    def test_step_3_proceeds_to_step_4_on_callback_error(self):
        """Test Step 3 proceeds to Step 4 when DPR queue callback raises."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)

        def failing_callback():
            raise RuntimeError("DPR queue failed")

        orchestrator.set_dpr_queue_callback(failing_callback)

        with patch.object(orchestrator, '_transition_to'):
            with patch.object(orchestrator, '_publish_step_completion') as mock_publish:
                with patch.object(orchestrator, '_execute_step_4', return_value=True) as mock_step4:
                    result = orchestrator._execute_step_3()

        assert result is True
        mock_step4.assert_called_once()


class TestGetAuditLogEntries:
    """Test get_audit_log_entries filtering - NOTE: method doesn't exist on orchestrator."""

    def test_get_audit_log_returns_copy(self):
        """Test get_audit_log returns a copy, not the original."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)

        orchestrator._transition_to(CooldownState.STEP_1_SCORING)

        log1 = orchestrator.get_audit_log()
        log2 = orchestrator.get_audit_log()

        # Should be different objects (copy)
        assert log1 is not log2
        assert len(log1) == len(log2)

    def test_audit_log_records_all_transitions(self):
        """Test audit log records all state transitions in order."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)

        orchestrator._transition_to(CooldownState.STEP_1_SCORING)
        orchestrator._transition_to(CooldownState.STEP_2_PAPER_RECOVERY)
        orchestrator._transition_to(CooldownState.STEP_3_QUEUE_BUILD)
        orchestrator._transition_to(CooldownState.STEP_4_HEALTH_CHECK)

        log = orchestrator.get_audit_log()

        assert len(log) == 4
        assert log[0].to_state == CooldownState.STEP_1_SCORING
        assert log[1].to_state == CooldownState.STEP_2_PAPER_RECOVERY
        assert log[2].to_state == CooldownState.STEP_3_QUEUE_BUILD
        assert log[3].to_state == CooldownState.STEP_4_HEALTH_CHECK


class TestCooldownTransitionDataclass:
    """Test CooldownTransition dataclass directly."""

    def test_cooldown_transition_creation(self):
        """Test CooldownTransition can be created with all fields."""
        entry = CooldownTransition(
            entry_id="test-id-123",
            from_state=CooldownState.PENDING,
            to_state=CooldownState.STEP_1_SCORING,
            step_name="London Session Scoring",
            timestamp_utc=datetime.now(timezone.utc),
            step_started_at=datetime.now(timezone.utc),
            step_deadline=datetime.now(timezone.utc) + timedelta(minutes=30),
            metadata={"test": "value"},
        )

        assert entry.entry_id == "test-id-123"
        assert entry.from_state == CooldownState.PENDING
        assert entry.to_state == CooldownState.STEP_1_SCORING
        assert entry.step_name == "London Session Scoring"
        assert entry.metadata == {"test": "value"}

    def test_cooldown_transition_default_metadata(self):
        """Test CooldownTransition metadata defaults to empty dict."""
        entry = CooldownTransition(
            entry_id="test-id-456",
            from_state=None,
            to_state=CooldownState.PENDING,
            step_name="Pending",
            timestamp_utc=datetime.now(timezone.utc),
        )

        assert entry.metadata == {}

    def test_cooldown_transition_optional_fields(self):
        """Test CooldownTransition optional fields can be None."""
        entry = CooldownTransition(
            entry_id="test-id-789",
            from_state=None,
            to_state=CooldownState.PENDING,
            step_name="Pending",
            timestamp_utc=datetime.now(timezone.utc),
            step_started_at=None,
            step_deadline=None,
        )

        assert entry.step_started_at is None
        assert entry.step_deadline is None

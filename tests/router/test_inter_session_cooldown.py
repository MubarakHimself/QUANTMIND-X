"""
Tests for InterSessionCooldownOrchestrator.

Story 16.3: Inter-Session Cooldown Window (10:00–13:00 GMT)

Tests for the InterSessionCooldownOrchestrator class implementing
the 4-step cooldown sequence.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from src.events.cooldown import (
    CooldownPhase,
    CooldownState,
    NYQueueCandidate,
)
from src.router.inter_session_cooldown import (
    CHANNEL_COOLDOWN_COMPLETED,
    CHANNEL_COOLDOWN_LONDON_SCORE_COMPLETE,
    CHANNEL_COOLDOWN_PHASE,
    InterSessionCooldownOrchestrator,
)


class MockRedis:
    """Mock Redis client for testing."""

    def __init__(self):
        self._data = {}
        self._channels = {}

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
        if channel not in self._channels:
            self._channels[channel] = []
        self._channels[channel].append(message)

    def subscribe(self, channel):
        pass


class TestInterSessionCooldownOrchestratorInit:
    """Test InterSessionCooldownOrchestrator initialization."""

    def test_initial_state_is_pending(self):
        """Verify orchestrator starts in PENDING state."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)

        assert orchestrator.state == CooldownState.PENDING
        assert orchestrator.is_running is False

    def test_initial_roster_not_locked(self):
        """Verify NY roster starts unlocked."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)

        assert orchestrator.ny_roster_locked is False

    def test_instance_id_is_set(self):
        """Verify instance_id is stored correctly."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(
            redis_client=redis,
            instance_id="cooldown-test"
        )

        assert orchestrator._instance_id == "cooldown-test"

    def test_window_times_initially_none(self):
        """Verify window times start as None."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)

        assert orchestrator.window_start is None
        assert orchestrator.window_end is None


class TestCooldownStart:
    """Test cooldown start behavior."""

    def test_start_transitions_to_step_1(self):
        """Test that start() transitions to STEP_1_SCORING."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)

        # Mock all step methods to just transition state without delay
        with patch.object(orchestrator, '_execute_step_1', return_value=True):
            with patch.object(orchestrator, '_execute_step_2', return_value=True):
                with patch.object(orchestrator, '_execute_step_3', return_value=True):
                    with patch.object(orchestrator, '_execute_step_4', return_value=True):
                        with patch.object(orchestrator, '_complete', return_value=True):
                            result = orchestrator.start()

        assert result is True

    def test_start_rejected_when_running(self):
        """Test that start() is rejected when already running."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)
        orchestrator._state = CooldownState.STEP_2_PAPER_RECOVERY

        result = orchestrator.start()

        assert result is False

    def test_start_records_window_times(self):
        """Test that start() records window start and end times."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)

        with patch.object(orchestrator, '_execute_step_1', return_value=True):
            with patch.object(orchestrator, '_execute_step_2', return_value=True):
                with patch.object(orchestrator, '_execute_step_3', return_value=True):
                    with patch.object(orchestrator, '_execute_step_4', return_value=True):
                        with patch.object(orchestrator, '_complete', return_value=True):
                            orchestrator.start()

        assert orchestrator.window_start is not None
        assert orchestrator.window_end is not None


class TestStepExecution:
    """Test individual step execution."""

    def test_step_1_uses_dpr_callback(self):
        """Test Step 1 uses DPR scoring callback when available."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)

        mock_callback = MagicMock(return_value={"score": 0.85})
        orchestrator.set_dpr_scoring_callback(mock_callback)

        with patch.object(orchestrator, '_transition_to'):
            with patch.object(orchestrator, '_publish_step_completion'):
                orchestrator._execute_step_1()

        mock_callback.assert_called_once()

    def test_step_1_uses_stub_when_no_callback(self):
        """Test Step 1 uses stub when DPR callback not set."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)

        with patch.object(orchestrator, '_transition_to'):
            with patch.object(orchestrator, '_publish_step_completion'):
                with patch.object(orchestrator, '_execute_step_2', return_value=True) as mock_step2:
                    result = orchestrator._execute_step_1()

        assert result is True
        # Step 1 should have called Step 2
        mock_step2.assert_called_once()

    def test_step_2_uses_ssl_callback(self):
        """Test Step 2 uses SSL paper review callback when available."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)

        mock_callback = MagicMock(return_value={"bots_reviewed": 5})
        orchestrator.set_ssl_paper_review_callback(mock_callback)

        with patch.object(orchestrator, '_transition_to'):
            with patch.object(orchestrator, '_publish_step_completion'):
                orchestrator._execute_step_2()

        mock_callback.assert_called_once()

    def test_step_3_uses_dpr_queue_callback(self):
        """Test Step 3 uses DPR queue callback when available."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)

        mock_callback = MagicMock(return_value=[
            NYQueueCandidate(bot_id="bot-1", queue_position=1, source="london_performer", tier=1)
        ])
        orchestrator.set_dpr_queue_callback(mock_callback)

        with patch.object(orchestrator, '_transition_to'):
            with patch.object(orchestrator, '_publish_step_completion'):
                orchestrator._execute_step_3()

        mock_callback.assert_called_once()

    def test_step_4_svss_health_callback(self):
        """Test Step 4 uses SVSS health callback when available."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)

        mock_callback = MagicMock(return_value={"status": "healthy"})
        orchestrator.set_svss_health_callback(mock_callback)

        with patch.object(orchestrator, '_transition_to'):
            with patch.object(orchestrator, '_publish_step_completion'):
                orchestrator._execute_step_4()

        mock_callback.assert_called_once()

    def test_step_4_sqs_warmup_on_monday(self):
        """Test Step 4 triggers SQS warm-up on Monday."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)

        mock_sqs = MagicMock()
        orchestrator.set_svss_health_callback(MagicMock(return_value={"status": "ok"}))
        orchestrator.set_sqs_warmup_callback(mock_sqs)

        # Mock is_monday to return True
        with patch('src.router.inter_session_cooldown.datetime') as mock_dt:
            mock_dt.now.return_value = datetime(2026, 3, 23, 10, 0, tzinfo=timezone.utc)  # Monday
            mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

            with patch.object(orchestrator, '_transition_to'):
                with patch.object(orchestrator, '_publish_step_completion'):
                    with patch.object(orchestrator, '_complete', return_value=True):
                        orchestrator._execute_step_4()

        # SQS warmup should have been called with True
        mock_sqs.assert_called_once()


class TestNYRosterLock:
    """Test NY roster locking."""

    def test_complete_locks_roster(self):
        """Test that _complete() locks the NY roster."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)

        with patch.object(orchestrator, '_transition_to'):
            with patch.object(orchestrator, '_publish_completion'):
                orchestrator._complete()

        assert orchestrator.ny_roster_locked is True

    def test_complete_calls_tilt_activate(self):
        """Test that _complete() triggers Tilt ACTIVATE."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)

        mock_callback = MagicMock()
        orchestrator.set_tilt_activate_callback(mock_callback)

        with patch.object(orchestrator, '_transition_to'):
            with patch.object(orchestrator, '_publish_completion'):
                orchestrator._complete()

        mock_callback.assert_called_once()

    def test_complete_calls_roster_lock_callback(self):
        """Test that _complete() calls roster lock callback."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)

        mock_callback = MagicMock()
        orchestrator.set_roster_lock_callback(mock_callback)

        with patch.object(orchestrator, '_transition_to'):
            with patch.object(orchestrator, '_publish_completion'):
                orchestrator._complete()

        mock_callback.assert_called_once()


class TestRedisPubSub:
    """Test Redis pub/sub behavior."""

    def test_publish_phase_event(self):
        """Test phase events are published to Redis."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)

        orchestrator._publish_phase_event()

        assert CHANNEL_COOLDOWN_PHASE in redis._channels
        assert len(redis._channels[CHANNEL_COOLDOWN_PHASE]) == 1

    def test_publish_step_completion_channels(self):
        """Test step completions are published to correct channels."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)

        # Step 1 completion
        orchestrator._publish_step_completion(
            CooldownPhase.STEP_1,
            "London Session Scoring",
            {"score": 0.85}
        )

        assert CHANNEL_COOLDOWN_LONDON_SCORE_COMPLETE in redis._channels
        assert CHANNEL_COOLDOWN_PHASE in redis._channels

    def test_publish_completion(self):
        """Test completion event is published."""
        from src.events.cooldown import InterSessionCooldownCompletionEvent

        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)

        event = InterSessionCooldownCompletionEvent(
            ny_roster_locked=True,
            tilt_activate_triggered=True,
        )
        orchestrator._publish_completion(event)

        assert CHANNEL_COOLDOWN_COMPLETED in redis._channels


class TestCallbackSetters:
    """Test callback setter methods."""

    def test_set_dpr_scoring_callback(self):
        """Test set_dpr_scoring_callback."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)

        callback = MagicMock()
        orchestrator.set_dpr_scoring_callback(callback)

        assert orchestrator._dpr_scoring_callback is callback

    def test_set_dpr_queue_callback(self):
        """Test set_dpr_queue_callback."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)

        callback = MagicMock()
        orchestrator.set_dpr_queue_callback(callback)

        assert orchestrator._dpr_queue_callback is callback

    def test_set_ssl_paper_review_callback(self):
        """Test set_ssl_paper_review_callback."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)

        callback = MagicMock()
        orchestrator.set_ssl_paper_review_callback(callback)

        assert orchestrator._ssl_paper_review_callback is callback

    def test_set_svss_health_callback(self):
        """Test set_svss_health_callback."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)

        callback = MagicMock()
        orchestrator.set_svss_health_callback(callback)

        assert orchestrator._svss_health_callback is callback

    def test_set_tilt_activate_callback(self):
        """Test set_tilt_activate_callback."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)

        callback = MagicMock()
        orchestrator.set_tilt_activate_callback(callback)

        assert orchestrator._tilt_activate_callback is callback


class TestAuditLog:
    """Test audit log functionality."""

    def test_get_audit_log(self):
        """Test get_audit_log returns copy of log."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)

        orchestrator._transition_to(CooldownState.STEP_1_SCORING)

        log = orchestrator.get_audit_log()
        assert len(log) == 1
        assert log[0].to_state == CooldownState.STEP_1_SCORING

    def test_audit_log_contains_transitions(self):
        """Test audit log records state transitions."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)

        orchestrator._transition_to(CooldownState.STEP_1_SCORING)
        orchestrator._transition_to(CooldownState.STEP_2_PAPER_RECOVERY)

        log = orchestrator.get_audit_log()
        assert len(log) == 2
        assert log[0].from_state == CooldownState.PENDING
        assert log[0].to_state == CooldownState.STEP_1_SCORING
        assert log[1].from_state == CooldownState.STEP_1_SCORING
        assert log[1].to_state == CooldownState.STEP_2_PAPER_RECOVERY


class TestStateEventGetters:
    """Test state event getter methods."""

    def test_get_current_state_event(self):
        """Test get_current_state_event returns correct event."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)
        orchestrator._state = CooldownState.STEP_2_PAPER_RECOVERY

        event = orchestrator.get_current_state_event()

        assert event.state == CooldownState.STEP_2_PAPER_RECOVERY
        assert event.current_step == 2
        assert event.step_name == "Paper Recovery Review"

    def test_get_current_phase_event(self):
        """Test get_current_phase_event returns correct event."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)
        orchestrator._state = CooldownState.STEP_3_QUEUE_BUILD

        event = orchestrator.get_current_phase_event()

        assert event.phase == CooldownPhase.STEP_3
        assert event.step_name == "NY Queue Order via DPR + Tier Remix"


class TestStubImplementations:
    """Test stub implementations when callbacks not set."""

    def test_stub_london_scoring(self):
        """Test stub London scoring returns expected structure."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)

        result = orchestrator._stub_london_scoring()

        assert result["stub"] is True
        assert "london_performers" in result
        assert "best_london_bot" in result

    def test_stub_paper_review(self):
        """Test stub paper review returns expected structure."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)

        result = orchestrator._stub_paper_review()

        assert result["stub"] is True
        assert "tier_1_paper_bots_reviewed" in result
        assert "recovery_confirmed" in result
        assert "recovery_extended" in result

    def test_stub_ny_queue_build(self):
        """Test stub NY queue build returns expected structure."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)

        result = orchestrator._stub_ny_queue_build()

        assert result["stub"] is True
        assert "queue_candidates" in result
        assert "composition" in result
        # Verify proper hybrid queue composition
        assert result["composition"]["london_performer"] == 1
        assert result["composition"]["recovery_candidate"] == 1
        assert result["composition"]["tier_3_dpr"] == 2
        assert result["composition"]["tier_2_fresh"] == 1
        # Verify queue candidates demonstrate proper structure
        candidates = result["queue_candidates"]
        assert len(candidates) == 5
        # Position 1: london_performer
        assert candidates[0]["queue_position"] == 1
        assert candidates[0]["source"] == "london_performer"
        assert candidates[0]["session_specialist"] is True
        # Position 2: recovery_candidate
        assert candidates[1]["queue_position"] == 2
        assert candidates[1]["source"] == "recovery_candidate"
        assert candidates[1]["consecutive_paper_wins"] == 2

    def test_stub_svss_health(self):
        """Test stub SVSS health returns expected structure."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)

        result = orchestrator._stub_svss_health()

        assert result["stub"] is True
        assert "status" in result
        assert result["status"] == "pass"

    def test_stub_sentinel_regime(self):
        """Test stub Sentinel regime returns expected structure."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)

        result = orchestrator._stub_sentinel_regime()

        assert result["stub"] is True
        assert "regime" in result
        assert "ny_open_ready" in result


class TestTimeoutHandling:
    """Test timeout handling for step execution."""

    def test_check_step_timeout_returns_false_when_no_window_start(self):
        """Test that timeout check returns False when window_start is None."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)

        # No start() called, so window_start is None
        result = orchestrator._check_step_timeout(CooldownState.STEP_1_SCORING)

        assert result is False

    def test_check_step_timeout_returns_false_within_window(self):
        """Test that timeout check returns False when within window."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)

        # Manually set window_start to now (within window)
        orchestrator._window_start = datetime.now(timezone.utc)
        orchestrator._current_step_start = datetime.now(timezone.utc)

        result = orchestrator._check_step_timeout(CooldownState.STEP_1_SCORING)

        assert result is False


class TestStepDeadlines:
    """Test step_deadlines property."""

    def test_step_deadlines_returns_empty_when_no_window_start(self):
        """Test that step_deadlines returns empty dict when window_start is None."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)

        deadlines = orchestrator.step_deadlines

        assert deadlines == {}

    def test_step_deadlines_returns_dict_with_all_steps(self):
        """Test that step_deadlines returns deadlines for all steps."""
        redis = MockRedis()
        orchestrator = InterSessionCooldownOrchestrator(redis_client=redis)

        orchestrator._window_start = datetime(2026, 3, 25, 10, 0, tzinfo=timezone.utc)

        deadlines = orchestrator.step_deadlines

        assert len(deadlines) == 4
        assert CooldownState.STEP_1_SCORING in deadlines
        assert CooldownState.STEP_2_PAPER_RECOVERY in deadlines
        assert CooldownState.STEP_3_QUEUE_BUILD in deadlines
        assert CooldownState.STEP_4_HEALTH_CHECK in deadlines
        # Step 1 deadline should be 10:30 (10:00 + 30 min)
        assert deadlines[CooldownState.STEP_1_SCORING].hour == 10
        assert deadlines[CooldownState.STEP_1_SCORING].minute == 30
        # Step 4 deadline should be 13:00 (10:00 + 180 min)
        assert deadlines[CooldownState.STEP_4_HEALTH_CHECK].hour == 13
        assert deadlines[CooldownState.STEP_4_HEALTH_CHECK].minute == 0

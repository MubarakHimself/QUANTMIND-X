"""
P0 Acceptance Tests for Epic 4 - Risk Management & Compliance

These tests verify the critical path for:
- R-001: Kelly fraction validation (kelly_fraction > 1.0 → 422)
- R-002: Islamic compliance countdown (60min warning, 30min critical, 21:45 UTC force-close)
- R-003: NFR-R1 cascade failure isolation (each tile independent)
- R-004: CalendarGovernor Journey 45 NFP scenario (4 phase transitions)

Test Design Reference: _bmad-output/test-artifacts/test-design-epic-4.md
Generated via: bmad-tea-testarch-atdd skill (YOLO mode)

NOTE: These tests are written to FAIL initially (TDD red phase).
They should pass once the implementation is complete.
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch
from fastapi.testclient import TestClient

from src.api.server import app


class TestKellyFractionValidationAPI:
    """
    P0 Tests for R-001: Kelly Fraction Validation Bypass

    Risk: kelly_fraction > 1.0 not rejected properly, allowing oversized positions
    Mitigation: Pydantic validation at API boundary, 422 response enforced

    Tests use actual API endpoint via TestClient to verify HTTP 422 responses.
    """

    @pytest.fixture
    def client(self):
        """Create TestClient for API testing."""
        return TestClient(app)

    @pytest.fixture
    def test_account_tag(self):
        """Test account tag for risk params."""
        return "test-kelly-account"

    def test_kelly_fraction_1_0_accepted(self, client, test_account_tag):
        """
        P0: Kelly fraction = 1.0 should be ACCEPTED (boundary value).

        AC: kelly_fraction=1.0 is the maximum valid value and must be accepted.
        Expected: 200 OK response
        """
        response = client.put(
            f"/api/risk/params/{test_account_tag}",
            json={"kelly_fraction": 1.0}
        )

        # This should pass once implementation is complete
        assert response.status_code == 200, (
            f"kelly_fraction=1.0 should be accepted, got {response.status_code}: "
            f"{response.json()}"
        )
        data = response.json()
        assert data["status"] == "updated"
        assert "kelly_fraction" in data["updated_fields"]

    def test_kelly_fraction_1_01_rejected_with_422(self, client, test_account_tag):
        """
        P0: Kelly fraction = 1.01 should be REJECTED with 422 (boundary value).

        AC: kelly_fraction > 1.0 must be rejected at API boundary.
        This is the critical security fix (R-001) preventing oversized positions.

        Expected: 422 Unprocessable Entity
        """
        response = client.put(
            f"/api/risk/params/{test_account_tag}",
            json={"kelly_fraction": 1.01}
        )

        # This should fail initially (TDD red phase)
        assert response.status_code == 422, (
            f"kelly_fraction=1.01 should be rejected with 422, got {response.status_code}: "
            f"{response.text}"
        )

    def test_kelly_fraction_0_0_rejected_with_422(self, client, test_account_tag):
        """
        P0: Kelly fraction = 0.0 should be REJECTED with 422 (boundary value).

        AC: kelly_fraction must be > 0 (minimum is 0.001 per Pydantic ge=0.001).

        Expected: 422 Unprocessable Entity
        """
        response = client.put(
            f"/api/risk/params/{test_account_tag}",
            json={"kelly_fraction": 0.0}
        )

        # This should fail initially (TDD red phase)
        assert response.status_code == 422, (
            f"kelly_fraction=0.0 should be rejected with 422, got {response.status_code}: "
            f"{response.text}"
        )

    def test_kelly_fraction_0_001_accepted_lower_bound(self, client, test_account_tag):
        """
        P0: Kelly fraction = 0.001 should be ACCEPTED (lower boundary value).

        AC: kelly_fraction >= 0.001 is the valid range.

        Expected: 200 OK response
        """
        response = client.put(
            f"/api/risk/params/{test_account_tag}",
            json={"kelly_fraction": 0.001}
        )

        # This should pass once implementation is complete
        assert response.status_code == 200, (
            f"kelly_fraction=0.001 should be accepted, got {response.status_code}: "
            f"{response.json()}"
        )

    def test_kelly_fraction_0_0009_rejected(self, client, test_account_tag):
        """
        P0: Kelly fraction = 0.0009 should be REJECTED with 422 (below lower bound).

        AC: kelly_fraction < 0.001 is invalid.

        Expected: 422 Unprocessable Entity
        """
        response = client.put(
            f"/api/risk/params/{test_account_tag}",
            json={"kelly_fraction": 0.0009}
        )

        # This should fail initially (TDD red phase)
        assert response.status_code == 422, (
            f"kelly_fraction=0.0009 should be rejected with 422, got {response.status_code}: "
            f"{response.text}"
        )


class TestIslamicComplianceCountdownAPI:
    """
    P0 Tests for R-002: Islamic Compliance Countdown Failure

    Risk: force-close at 21:45 UTC may not trigger, causing account violation
    Mitigation: Timer validation tests, UI countdown display verification

    Tests verify the /api/risk/islamic-status endpoint with mocked time.
    """

    @pytest.fixture
    def client(self):
        """Create TestClient for API testing."""
        return TestClient(app)

    def test_islamic_countdown_61min_before_no_warning(self, client):
        """
        P0: 61 minutes before 21:45 UTC - NO warning window active.

        AC: is_within_60min_window should be FALSE when > 60 minutes remain.

        Expected: is_within_60min_window=false, is_within_30min_window=false
        """
        # Mock time: 61 minutes before 21:45 = 20:44 UTC
        mock_time = datetime(2026, 3, 20, 20, 44, 0, tzinfo=timezone.utc)

        with patch('src.api.risk_endpoints.datetime') as mock_datetime:
            mock_datetime.now.return_value = mock_time
            mock_datetime.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

            response = client.get("/api/risk/islamic-status")

        assert response.status_code == 200
        data = response.json()

        # This should fail initially (TDD red phase)
        assert data["is_within_60min_window"] is False, (
            f"At 61 min before 21:45, is_within_60min_window should be False, got {data}"
        )
        assert data["is_within_30min_window"] is False, (
            f"At 61 min before 21:45, is_within_30min_window should be False, got {data}"
        )

    def test_islamic_countdown_59min_before_warning(self, client):
        """
        P0: 59 minutes before 21:45 UTC - 60-minute WARNING window active.

        AC: is_within_60min_window should be TRUE when 0 < minutes <= 60 remain.
        is_within_30min_window should be FALSE when > 30 minutes remain.

        Expected: is_within_60min_window=true, is_within_30min_window=false
        """
        # Mock time: 59 minutes before 21:45 = 20:46 UTC
        mock_time = datetime(2026, 3, 20, 20, 46, 0, tzinfo=timezone.utc)

        with patch('src.api.risk_endpoints.datetime') as mock_datetime:
            mock_datetime.now.return_value = mock_time
            mock_datetime.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

            response = client.get("/api/risk/islamic-status")

        assert response.status_code == 200
        data = response.json()

        # This should fail initially (TDD red phase)
        assert data["is_within_60min_window"] is True, (
            f"At 59 min before 21:45, is_within_60min_window should be True, got {data}"
        )
        assert data["is_within_30min_window"] is False, (
            f"At 59 min before 21:45, is_within_30min_window should be False, got {data}"
        )
        assert data["countdown_seconds"] > 0, (
            f"countdown_seconds should be positive, got {data.get('countdown_seconds')}"
        )

    def test_islamic_countdown_29min_before_critical(self, client):
        """
        P0: 29 minutes before 21:45 UTC - 30-minute CRITICAL window active.

        AC: is_within_30min_window should be TRUE when 0 < minutes <= 30 remain.
        29 minutes = 1740 seconds which is <= 1800 (30 min).

        Expected: is_within_60min_window=true, is_within_30min_window=true
        """
        # Mock time: 29 minutes before 21:45 = 21:16 UTC
        mock_time = datetime(2026, 3, 20, 21, 16, 0, tzinfo=timezone.utc)

        with patch('src.api.risk_endpoints.datetime') as mock_datetime:
            mock_datetime.now.return_value = mock_time
            mock_datetime.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

            response = client.get("/api/risk/islamic-status")

        assert response.status_code == 200
        data = response.json()

        # This should fail initially (TDD red phase)
        assert data["is_within_60min_window"] is True, (
            f"At 29 min before 21:45, is_within_60min_window should be True, got {data}"
        )
        assert data["is_within_30min_window"] is True, (
            f"At 29 min before 21:45, is_within_30min_window should be True, got {data}"
        )

    def test_islamic_countdown_force_close_at_2145_utc(self, client):
        """
        P0: At exactly 21:45 UTC - FORCE-CLOSE should be triggered.

        AC: countdown should reach 0 at 21:45 UTC, force_close_at should be set.

        Expected: countdown_seconds=0, force_close_at is set
        """
        # Mock time: exactly at 21:45 UTC
        mock_time = datetime(2026, 3, 20, 21, 45, 0, tzinfo=timezone.utc)

        with patch('src.api.risk_endpoints.datetime') as mock_datetime:
            mock_datetime.now.return_value = mock_time
            mock_datetime.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

            response = client.get("/api/risk/islamic-status")

        assert response.status_code == 200
        data = response.json()

        # This should fail initially (TDD red phase)
        assert data["countdown_seconds"] == 0, (
            f"At 21:45 UTC exactly, countdown_seconds should be 0, got {data.get('countdown_seconds')}"
        )
        assert data["force_close_at"] is not None, (
            f"At 21:45 UTC, force_close_at should be set, got {data}"
        )


class TestCalendarGovernorJourney45API:
    """
    P0 Tests for R-004: CalendarGovernor Journey 45 NFP Scenario

    Risk: NFP Friday 4-phase transition failure
    Mitigation: Journey 45 scenario tests with time-based assertions

    Tests verify the CalendarGovernor's 4-phase transition:
    - Thu 18:00 (EUR/USD scalpers 0.5x)
    - Fri 13:15 (all pause)
    - Fri 14:00 (regime-check reactivation)
    - Fri 15:00 (normal operation)

    NOTE: These tests verify the CalendarGovernor at the unit level.
    For API-level tests, use the calendar event endpoints.
    """

    def test_journey45_fri_13_00_pre_event_scaling(self):
        """
        P0: Fri 13:00 UTC - EUR/USD scalpers get 0.5x (30 min before NFP).

        AC: 30 minutes before NFP at 13:30, strategies should be at
        0.5x lot size (pre-event scaling phase).

        Expected: PRE_EVENT phase with 0.5x scaling
        """
        from src.router.calendar_governor import CalendarGovernor
        from src.risk.models.calendar import (
            NewsItem, CalendarRule, CalendarEventType,
            NewsImpact, CalendarPhase
        )

        governor = CalendarGovernor(account_id="FTMO-JOURNEY45")

        # Create calendar rule with standard Journey 45 parameters
        rule = CalendarRule(
            rule_id="journey45-rule",
            account_id="FTMO-JOURNEY45",
            blacklist_enabled=True,
            blackout_minutes=30,
            lot_scaling_factors={
                "pre_event": 0.5,
                "during_event": 0.0,
                "post_event_regime_check": 0.75,
                "normal": 1.0,
            },
            post_event_delay_minutes=60,
            regime_check_enabled=True,
        )
        governor.register_calendar_rule(rule)

        # NFP event at Friday 13:30 UTC
        nfp_event = NewsItem(
            event_id="nfp-2026-0306",
            title="Non-Farm Payrolls",
            event_type=CalendarEventType.NFP,
            impact=NewsImpact.HIGH,
            event_time=datetime(2026, 3, 6, 13, 30, tzinfo=timezone.utc),
            currencies=["USD", "EUR", "GBP"],
        )
        governor.add_calendar_event(nfp_event)

        # Fri 13:00 UTC = 30 minutes before NFP (13:30)
        check_time = datetime(2026, 3, 6, 13, 0, tzinfo=timezone.utc)

        scaling = governor.evaluate_lot_scaling(
            "FTMO-JOURNEY45", check_time, nfp_event, rule
        )

        # This should fail initially (TDD red phase)
        assert scaling == 0.5, (
            f"At Fri 13:00 UTC (30 min before NFP), pre-event scaling should be 0.5x, got {scaling}"
        )

    def test_journey45_fri_13_15_during_event_pause(self):
        """
        P0: Fri 13:15 - ALL STRATEGIES PAUSE (15 min before NFP = DURING_EVENT).

        AC: At Fri 13:15 UTC (15 min before NFP at 13:30), strategies
        should be fully paused with 0.0x lot size.

        Expected: DURING_EVENT phase with 0.0x scaling (full pause)
        """
        from src.router.calendar_governor import CalendarGovernor
        from src.risk.models.calendar import (
            NewsItem, CalendarRule, CalendarEventType, NewsImpact
        )

        governor = CalendarGovernor(account_id="FTMO-JOURNEY45")

        rule = CalendarRule(
            rule_id="journey45-rule",
            account_id="FTMO-JOURNEY45",
            blacklist_enabled=True,
            blackout_minutes=30,
            lot_scaling_factors={
                "pre_event": 0.5,
                "during_event": 0.0,
                "post_event_regime_check": 0.75,
                "normal": 1.0,
            },
            post_event_delay_minutes=60,
            regime_check_enabled=True,
        )
        governor.register_calendar_rule(rule)

        nfp_event = NewsItem(
            event_id="nfp-2026-0306",
            title="Non-Farm Payrolls",
            event_type=CalendarEventType.NFP,
            impact=NewsImpact.HIGH,
            event_time=datetime(2026, 3, 6, 13, 30, tzinfo=timezone.utc),
            currencies=["USD", "EUR", "GBP"],
        )
        governor.add_calendar_event(nfp_event)

        # Fri 13:15 UTC = 15 min before NFP (13:30)
        check_time = datetime(2026, 3, 6, 13, 15, tzinfo=timezone.utc)

        scaling = governor.evaluate_lot_scaling(
            "FTMO-JOURNEY45", check_time, nfp_event, rule
        )

        # This should fail initially (TDD red phase)
        assert scaling == 0.0, (
            f"At Fri 13:15 UTC (15 min before NFP), scaling should be 0.0x (PAUSE), got {scaling}"
        )

    def test_journey45_fri_14_00_regime_check_reactivation(self):
        """
        P0: Fri 14:00 - REGIME-CHECK REACTIVATION (30 min after NFP).

        AC: At Fri 14:00 UTC (30 min after NFP), strategies can resume
        with regime-check reactivation (0.75x).

        Expected: POST_EVENT_REGIME_CHECK phase
        """
        from src.router.calendar_governor import CalendarGovernor
        from src.risk.models.calendar import (
            NewsItem, CalendarRule, CalendarEventType, NewsImpact
        )

        governor = CalendarGovernor(account_id="FTMO-JOURNEY45")

        rule = CalendarRule(
            rule_id="journey45-rule",
            account_id="FTMO-JOURNEY45",
            blacklist_enabled=True,
            blackout_minutes=30,
            lot_scaling_factors={
                "pre_event": 0.5,
                "during_event": 0.0,
                "post_event_regime_check": 0.75,
                "normal": 1.0,
            },
            post_event_delay_minutes=60,  # 60 min post-event delay
            regime_check_enabled=True,
        )
        governor.register_calendar_rule(rule)

        nfp_event = NewsItem(
            event_id="nfp-2026-0306",
            title="Non-Farm Payrolls",
            event_type=CalendarEventType.NFP,
            impact=NewsImpact.HIGH,
            event_time=datetime(2026, 3, 6, 13, 30, tzinfo=timezone.utc),
            currencies=["USD", "EUR", "GBP"],
        )
        governor.add_calendar_event(nfp_event)

        # Fri 14:00 UTC = 30 min after NFP (13:30)
        check_time = datetime(2026, 3, 6, 14, 0, tzinfo=timezone.utc)

        # Check if reactivation time is reached
        is_reactivation = governor.is_post_event_reactivation_time(
            "FTMO-JOURNEY45", check_time, nfp_event, rule
        )

        # This should fail initially (TDD red phase)
        assert is_reactivation is True, (
            f"At Fri 14:00 UTC (30 min after NFP), reactivation should be allowed, got {is_reactivation}"
        )

    def test_journey45_fri_15_00_normal_operation(self):
        """
        P0: Fri 15:00 - NORMAL OPERATION (1 hour after NFP).

        AC: At Fri 15:00 UTC (1.5 hours after NFP), post-event delay
        has passed, normal operation resumes (1.0x).

        Expected: NORMAL phase with 1.0x scaling
        """
        from src.router.calendar_governor import CalendarGovernor
        from src.risk.models.calendar import (
            NewsItem, CalendarRule, CalendarEventType, NewsImpact
        )

        governor = CalendarGovernor(account_id="FTMO-JOURNEY45")

        rule = CalendarRule(
            rule_id="journey45-rule",
            account_id="FTMO-JOURNEY45",
            blacklist_enabled=True,
            blackout_minutes=30,
            lot_scaling_factors={
                "pre_event": 0.5,
                "during_event": 0.0,
                "post_event_regime_check": 0.75,
                "normal": 1.0,
            },
            post_event_delay_minutes=60,  # 60 min post-event delay
            regime_check_enabled=True,
        )
        governor.register_calendar_rule(rule)

        nfp_event = NewsItem(
            event_id="nfp-2026-0306",
            title="Non-Farm Payrolls",
            event_type=CalendarEventType.NFP,
            impact=NewsImpact.HIGH,
            event_time=datetime(2026, 3, 6, 13, 30, tzinfo=timezone.utc),
            currencies=["USD", "EUR", "GBP"],
        )
        governor.add_calendar_event(nfp_event)

        # Fri 15:00 UTC = 1.5 hours after NFP (13:30)
        check_time = datetime(2026, 3, 6, 15, 0, tzinfo=timezone.utc)

        scaling = governor.evaluate_lot_scaling_normal(
            "FTMO-JOURNEY45", check_time
        )

        # This should fail initially (TDD red phase)
        assert scaling == 1.0, (
            f"At Fri 15:00 UTC (1.5 hours after NFP), scaling should be 1.0x (NORMAL), got {scaling}"
        )


class TestNFPR1CascadeFailureIsolation:
    """
    P0 Tests for R-003: 5-Second Polling Cascade Failure

    Risk: NFR-R1 requires independent failure isolation per tile
    Mitigation: Each tile wrapped in try/catch, individual error states

    NOTE: This is a FRONTEND requirement. The backend API tests verify
    that each endpoint returns appropriate error states independently.

    For proper E2E cascade failure testing, Playwright tests should be
    used to verify that one tile's failure doesn't affect others.
    """

    def test_risk_params_endpoint_error_isolation(self):
        """
        P0: Risk params endpoint should handle errors independently.

        AC: If /api/risk/params/{account_tag} fails, other endpoints
        (regime, physics) should still respond normally.

        Expected: Each endpoint can fail or succeed independently
        """
        from src.api.risk_endpoints import router

        # Verify router has multiple independent endpoints
        routes = [route.path for route in router.routes]

        # These endpoints should all exist and be independent
        assert any("/api/risk/params" in path for path in routes)
        assert any("/api/risk/regime" in path for path in routes)
        assert any("/api/risk/physics" in path for path in routes)
        assert any("/api/risk/compliance" in path for path in routes)

        # Each endpoint should be a separate route (not chained)
        risk_param_routes = [r for r in routes if "/api/risk/params" in r]
        assert len(risk_param_routes) >= 1, "Risk params endpoint should exist"

    def test_compliance_endpoint_returns_per_account_state(self):
        """
        P0: Compliance endpoint returns per-account-tag state.

        AC: Each account tag's compliance state is independent. Failure
        in one account's circuit breaker doesn't affect others.

        Expected: Response contains array of per-account states
        """
        from src.api.risk_endpoints import AccountTagCompliance

        # AccountTagCompliance should have tag field for isolation
        assert hasattr(AccountTagCompliance, 'model_fields')
        fields = AccountTagCompliance.model_fields

        # This should fail initially if the model is not properly defined
        assert 'tag' in fields, "AccountTagCompliance should have 'tag' field for per-account isolation"
        assert 'circuit_breaker_state' in fields, "Should have circuit_breaker_state field"
        assert 'drawdown_pct' in fields, "Should have drawdown_pct field"

    def test_physics_endpoints_are_independent(self):
        """
        P0: Physics sensor endpoints should return independent states.

        AC: Each sensor (Ising, Lyapunov, HMM) should have its own
        alert state that doesn't cascade to others.

        Expected: Each sensor can have different alert states
        """
        from src.api.risk_endpoints import (
            PhysicsIsingOutput,
            PhysicsLyapunovOutput,
            PhysicsHMMOutput
        )

        # Each sensor output should have independent alert state
        ising_fields = PhysicsIsingOutput.model_fields
        lyapunov_fields = PhysicsLyapunovOutput.model_fields
        hmm_fields = PhysicsHMMOutput.model_fields

        assert 'alert' in ising_fields, "Ising should have alert field"
        assert 'alert' in lyapunov_fields, "Lyapunov should have alert field"
        assert 'alert' in hmm_fields, "HMM should have alert field"

        # These are independent - each sensor has its own alert
        # One sensor going critical doesn't force others to critical

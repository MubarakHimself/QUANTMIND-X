"""
Tests for CalendarGovernor - News Blackout & Calendar-Aware Trading Rules

Tests the CalendarGovernor mixin for economic calendar event handling,
blackout window evaluation, and lot scaling rules per account configuration.

Story: 4-1-calendargovernor-news-blackout-calendar-aware-trading-rules
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch
import logging

from src.risk.models.calendar import (
    NewsItem,
    CalendarRule,
    CalendarEventType,
    NewsImpact,
    CalendarPhase,
)
from src.router.calendar_governor import CalendarGovernor
from src.router.enhanced_governor import EnhancedGovernor
from src.router.governor import RiskMandate
from src.router.sentinel import RegimeReport

logger = logging.getLogger(__name__)


class TestNewsItem:
    """Test the NewsItem data model."""

    def test_news_item_creation_tier1(self):
        """Test creating a Tier 1 news event (NFP)."""
        event_time = datetime(2026, 3, 6, 13, 30, tzinfo=timezone.utc)  # NFP Friday

        news = NewsItem(
            event_id="nfp-2026-03-06",
            title="Non-Farm Payrolls",
            event_type=CalendarEventType.NFP,
            impact=NewsImpact.HIGH,
            event_time=event_time,
            currencies=["USD", "EUR", "GBP"],
        )

        assert news.event_id == "nfp-2026-03-06"
        assert news.event_type == CalendarEventType.NFP
        assert news.impact == NewsImpact.HIGH
        assert "USD" in news.currencies

    def test_news_item_tier_classification(self):
        """Test Tier 1 vs Tier 2 event classification."""
        # Tier 1 events
        nfp = NewsItem(
            event_id="nfp-1",
            title="Non-Farm Payrolls",
            event_type=CalendarEventType.NFP,
            impact=NewsImpact.HIGH,
            event_time=datetime.now(timezone.utc),
            currencies=["USD"],
        )
        assert nfp.is_tier1 is True

        ecb = NewsItem(
            event_id="ecb-1",
            title="ECB Rate Decision",
            event_type=CalendarEventType.ECB,
            impact=NewsImpact.HIGH,
            event_time=datetime.now(timezone.utc),
            currencies=["EUR"],
        )
        assert ecb.is_tier1 is True

        # Tier 2 events
        ppi = NewsItem(
            event_id="ppi-1",
            title="PPI",
            event_type=CalendarEventType.OTHER,
            impact=NewsImpact.MEDIUM,
            event_time=datetime.now(timezone.utc),
            currencies=["USD"],
        )
        assert ppi.is_tier1 is False


class TestCalendarRule:
    """Test the CalendarRule data model."""

    def test_calendar_rule_creation(self):
        """Test creating a calendar rule."""
        rule = CalendarRule(
            rule_id="rule-1",
            account_id="FTMO-12345",
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

        assert rule.rule_id == "rule-1"
        assert rule.account_id == "FTMO-12345"
        assert rule.blackout_minutes == 30
        assert rule.lot_scaling_factors["pre_event"] == 0.5
        assert rule.post_event_delay_minutes == 60

    def test_lot_scaling_factor_retrieval(self):
        """Test retrieving correct lot scaling factor for each phase."""
        rule = CalendarRule(
            rule_id="rule-1",
            account_id="FTMO-12345",
            blacklist_enabled=True,
            blackout_minutes=30,
            lot_scaling_factors={
                "pre_event": 0.5,
                "during_event": 0.0,
                "post_event_regime_check": 0.75,
                "normal": 1.0,
            },
        )

        assert rule.get_lot_scaling(CalendarPhase.PRE_EVENT) == 0.5
        assert rule.get_lot_scaling(CalendarPhase.DURING_EVENT) == 0.0
        assert rule.get_lot_scaling(CalendarPhase.POST_EVENT_REGIME_CHECK) == 0.75
        assert rule.get_lot_scaling(CalendarPhase.NORMAL) == 1.0


class TestCalendarGovernor:
    """Test the CalendarGovernor mixin functionality."""

    @pytest.fixture
    def calendar_governor(self):
        """Create a CalendarGovernor instance for testing."""
        return CalendarGovernor(account_id="FTMO-12345")

    @pytest.fixture
    def sample_regime_report(self):
        """Create a sample RegimeReport for testing."""
        return RegimeReport(
            regime="TREND",
            regime_quality=0.8,
            chaos_score=0.2,
            is_systemic_risk=False,
            susceptibility=0.3,
            news_state="SAFE",
            timestamp=datetime.now(timezone.utc),
        )

    @pytest.fixture
    def sample_trade_proposal(self):
        """Create a sample trade proposal."""
        return {
            "symbol": "EURUSD",
            "current_balance": 100000.0,
            "broker_id": "icmarkets_raw",
            "account_id": "FTMO-12345",
            "stop_loss_pips": 20.0,
        }

    @pytest.fixture
    def nfp_event(self):
        """Create an NFP event for testing."""
        # Event at 13:30 UTC on Friday March 6, 2026
        event_time = datetime(2026, 3, 6, 13, 30, tzinfo=timezone.utc)
        return NewsItem(
            event_id="nfp-2026-0306",
            title="Non-Farm Payrolls",
            event_type=CalendarEventType.NFP,
            impact=NewsImpact.HIGH,
            event_time=event_time,
            currencies=["USD", "EUR"],
        )

    @pytest.fixture
    def calendar_rule(self):
        """Create a calendar rule for testing."""
        return CalendarRule(
            rule_id="rule-1",
            account_id="FTMO-12345",
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

    def test_calendar_governor_initialization(self, calendar_governor):
        """Test CalendarGovernor initializes correctly."""
        assert calendar_governor.account_id == "FTMO-12345"
        assert calendar_governor._calendar_rules == {}
        assert calendar_governor._active_events == []
        assert calendar_governor._reactivation_timers == {}

    def test_register_calendar_rule(self, calendar_governor, calendar_rule):
        """Test registering a calendar rule."""
        calendar_governor.register_calendar_rule(calendar_rule)

        assert "FTMO-12345" in calendar_governor._calendar_rules
        assert calendar_governor._calendar_rules["FTMO-12345"].rule_id == "rule-1"

    def test_add_calendar_event(self, calendar_governor, nfp_event):
        """Test adding a calendar event."""
        calendar_governor.add_calendar_event(nfp_event)

        assert len(calendar_governor._active_events) == 1
        assert calendar_governor._active_events[0].event_id == "nfp-2026-0306"

    def test_is_within_blackout_window_pre_event(self, calendar_governor, nfp_event, calendar_rule):
        """Test blackout window detection 30 minutes before event."""
        calendar_governor.register_calendar_rule(calendar_rule)
        calendar_governor.add_calendar_event(nfp_event)

        # 20 minutes before event - should be in blackout
        check_time = nfp_event.event_time - timedelta(minutes=20)
        result = calendar_governor.is_within_blackout_window(
            "FTMO-12345", check_time, calendar_governor._calendar_rules.get("FTMO-12345")
        )
        assert result is True

    def test_is_outside_blackout_window(self, calendar_governor, nfp_event):
        """Test outside blackout window detection."""
        calendar_governor.add_calendar_event(nfp_event)

        # 2 hours before event - outside default 30-minute window
        check_time = nfp_event.event_time - timedelta(hours=2)
        result = calendar_governor.is_within_blackout_window(
            "FTMO-12345", check_time, calendar_governor._calendar_rules.get("FTMO-12345")
        )
        assert result is False

    def test_evaluate_lot_scaling_pre_event(self, calendar_governor, calendar_rule, nfp_event):
        """Test lot scaling factor for pre-event phase."""
        calendar_governor.register_calendar_rule(calendar_rule)
        calendar_governor.add_calendar_event(nfp_event)

        # 20 minutes before event (in pre-event phase)
        check_time = nfp_event.event_time - timedelta(minutes=20)
        scaling = calendar_governor.evaluate_lot_scaling(
            "FTMO-12345", check_time, nfp_event, calendar_rule
        )

        assert scaling == 0.5  # pre_event scaling

    def test_evaluate_lot_scaling_during_event(self, calendar_governor, calendar_rule, nfp_event):
        """Test lot scaling factor for during-event phase."""
        calendar_governor.register_calendar_rule(calendar_rule)
        calendar_governor.add_calendar_event(nfp_event)

        # During the event
        check_time = nfp_event.event_time + timedelta(minutes=5)
        scaling = calendar_governor.evaluate_lot_scaling(
            "FTMO-12345", check_time, nfp_event, calendar_rule
        )

        assert scaling == 0.0  # during_event = PAUSE

    def test_evaluate_lot_scaling_normal(self, calendar_governor, calendar_rule):
        """Test lot scaling factor for normal operation (no events)."""
        calendar_governor.register_calendar_rule(calendar_rule)

        # Normal time - no events
        check_time = datetime.now(timezone.utc)
        scaling = calendar_governor.evaluate_lot_scaling_normal("FTMO-12345", check_time)

        assert scaling == 1.0  # normal scaling

    def test_calculate_risk_with_calendar_rules(
        self, calendar_governor, sample_regime_report, sample_trade_proposal, calendar_rule, nfp_event
    ):
        """Test calculate_risk integrates calendar rules correctly.

        Uses patch on _get_current_utc_time so that clear_past_events() uses the
        mocked time — preventing stale events from being incorrectly removed.
        """
        # Setup: register rule and add event
        calendar_governor.register_calendar_rule(calendar_rule)
        calendar_governor.add_calendar_event(nfp_event)

        # 20 minutes before event - should apply 0.5x PRE_EVENT scaling
        check_time = nfp_event.event_time - timedelta(minutes=20)

        with patch.object(calendar_governor, '_get_current_utc_time', return_value=check_time):
            mandate = calendar_governor.calculate_risk(
                regime_report=sample_regime_report,
                trade_proposal=sample_trade_proposal,
                account_balance=100000.0,
            )

        # The mandate should reflect calendar-based scaling
        assert mandate is not None
        assert hasattr(mandate, 'kelly_adjustments')
        # CalendarGovernor note must be present since 0.5x scaling was applied
        assert mandate.notes is not None
        assert "CalendarGovernor" in mandate.notes


class TestJourney45NFPFriday:
    """Test Journey 45 (The Calendar Gate) scenario: NFP every Friday.

    Thu 18:00 (EUR/USD scalpers 0.5x)
    Fri 13:15 (all pause)
    Fri 14:00 (regime-check reactivate)
    Fri 15:00 (normal)
    """

    @pytest.fixture
    def journey45_governor(self):
        """Create CalendarGovernor for Journey 45 testing."""
        return CalendarGovernor(account_id="FTMO-JOURNEY45")

    @pytest.fixture
    def journey45_rule(self):
        """Create calendar rule for Journey 45."""
        return CalendarRule(
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

    @pytest.fixture
    def nfp_march_6_2026(self):
        """NFP event on Friday March 6, 2026 at 13:30 UTC."""
        return NewsItem(
            event_id="nfp-2026-0306",
            title="Non-Farm Payrolls",
            event_type=CalendarEventType.NFP,
            impact=NewsImpact.HIGH,
            event_time=datetime(2026, 3, 6, 13, 30, tzinfo=timezone.utc),
            currencies=["USD", "EUR", "GBP"],
        )

    def test_phase_1_eurusd_scalpers_30min_before(
        self, journey45_governor, journey45_rule, nfp_march_6_2026
    ):
        """Phase 1: Fri 13:00 - EUR/USD scalpers get 0.5x (30 min before NFP at 13:30)."""
        journey45_governor.register_calendar_rule(journey45_rule)
        journey45_governor.add_calendar_event(nfp_march_6_2026)

        # Fri 13:00 UTC = 30 minutes before NFP (13:30)
        check_time = datetime(2026, 3, 6, 13, 0, tzinfo=timezone.utc)  # Friday 13:00

        scaling = journey45_governor.evaluate_lot_scaling(
            "FTMO-JOURNEY45", check_time, nfp_march_6_2026, journey45_rule
        )

        assert scaling == 0.5  # pre_event = 0.5x

    def test_phase_2_all_pause_15min_before(
        self, journey45_governor, journey45_rule, nfp_march_6_2026
    ):
        """Phase 2: Fri 13:15 - all strategies PAUSE (15 min before NFP = DURING_EVENT 0.0x)."""
        journey45_governor.register_calendar_rule(journey45_rule)
        journey45_governor.add_calendar_event(nfp_march_6_2026)

        # Fri 13:15 UTC = 15 minutes before NFP (13:30) — falls in DURING_EVENT window
        check_time = datetime(2026, 3, 6, 13, 15, tzinfo=timezone.utc)  # Friday 13:15

        # At 13:15 (15 min before event), DURING_EVENT phase applies → 0.0x scaling (full pause)
        scaling = journey45_governor.evaluate_lot_scaling(
            "FTMO-JOURNEY45", check_time, nfp_march_6_2026, journey45_rule
        )

        assert scaling == 0.0  # during_event = full PAUSE

    def test_phase_3_regime_check_30min_after(
        self, journey45_governor, journey45_rule, nfp_march_6_2026
    ):
        """Phase 3: Fri 14:00 - regime-check reactivate (30 min after NFP)."""
        journey45_governor.register_calendar_rule(journey45_rule)
        journey45_governor.add_calendar_event(nfp_march_6_2026)

        # Fri 14:00 UTC = 30 minutes after NFP event ends
        check_time = datetime(2026, 3, 6, 14, 0, tzinfo=timezone.utc)  # Friday 14:00

        # Check if post_event delay has passed
        is_reactivation_time = journey45_governor.is_post_event_reactivation_time(
            "FTMO-JOURNEY45", check_time, nfp_march_6_2026, journey45_rule
        )

        assert is_reactivation_time is True

    def test_phase_4_normal_1h_after(
        self, journey45_governor, journey45_rule, nfp_march_6_2026
    ):
        """Phase 4: Fri 15:00 - normal operation (1 hour after NFP)."""
        journey45_governor.register_calendar_rule(journey45_rule)
        journey45_governor.add_calendar_event(nfp_march_6_2026)

        # Fri 15:00 UTC = 1 hour 30 min after NFP event
        check_time = datetime(2026, 3, 6, 15, 0, tzinfo=timezone.utc)  # Friday 15:00

        # After post_event_delay (60 min), should be normal
        scaling = journey45_governor.evaluate_lot_scaling_normal(
            "FTMO-JOURNEY45", check_time
        )

        assert scaling == 1.0  # normal = 1.0x


class TestAuditLogging:
    """Test audit trail logging for calendar events."""

    def test_audit_log_rule_activation(self):
        """Test that rule activation is logged."""
        governor = CalendarGovernor(account_id="FTMO-TEST")

        with patch('src.router.calendar_governor.logger') as mock_logger:
            # Simulate rule activation
            governor._log_rule_activation("FTMO-TEST", "pre_event", 0.5)

            # Verify logging was called
            assert mock_logger.info.called

    def test_audit_log_resumption(self):
        """Test that resumption is logged."""
        governor = CalendarGovernor(account_id="FTMO-TEST")

        with patch('src.router.calendar_governor.logger') as mock_logger:
            # Simulate resumption
            governor._log_resumption("FTMO-TEST", "TREND", 0.8)

            # Verify logging was called
            assert mock_logger.info.called
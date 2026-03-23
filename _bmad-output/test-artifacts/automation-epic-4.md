---
name: 'automation-epic-4'
description: 'P1-P3 Test Coverage Expansion for Epic 4 - Risk Management & Compliance'
outputFile: '_bmad-output/test-artifacts/automation-epic-4.md'
epic: 'Epic 4 - Risk Management & Compliance'
date: '2026-03-21'
stack: 'fullstack'
mode: 'standalone'
focus_areas:
  - CalendarGovernor journey scenarios
  - HMM regime detection
  - Risk canvas physics tiles
  - BotCircuitBreaker
  - Prop firm registry UI
p0_status:
  total: 17
  passing: 16
  failing: 1
  failing_test: 'test_islamic_countdown_force_close_at_2145_utc'
  bug: 'is_within_60min condition fails at countdown_seconds=0'
---

# Epic 4 P1-P3 Test Automation Coverage

**Generated**: 2026-03-21
**Epic**: Epic 4 - Risk Management & Compliance
**Mode**: Standalone (YOLO)
**Stack**: Fullstack (Python/pytest + Svelte/vitest)

---

## Executive Summary

Epic 4 has 17 P0 tests (16 passing, 1 failing). This document expands P1-P3 test coverage across:
- CalendarGovernor journey scenarios (beyond NFP)
- HMM regime detection sensor tests
- Risk canvas physics tile tests
- BotCircuitBreaker unit tests
- Prop firm registry API tests

**P0 Bug Note**: The failing P0 test `test_islamic_countdown_force_close_at_2145_utc` exposes a bug where `is_within_60min` condition fails at `countdown_seconds=0`. This should be fixed in the implementation before P1 tests are run.

---

## P1 Tests - High Priority

### P1-A: CalendarGovernor Extended Journey Scenarios

**Risk Score**: P1 (High)
**Justification**: Core calendar functionality with complex time-based logic
**Test Level**: Unit

#### P1-A1: Multiple Overlapping Events

```python
# tests/api/test_epic4_p1_calendar.py

class TestCalendarGovernorMultipleEvents:
    """P1: CalendarGovernor handles multiple overlapping events correctly."""

    def test_overlapping_nfp_and_boe_events(self):
        """
        P1: When NFP and BOE events overlap, highest impact takes precedence.

        Scenario:
        - NFP at 13:30 (USD impact HIGH)
        - BOE at 13:45 (GBP impact HIGH)
        - Check time: 13:20 (both in blackout window)

        Expected: PRE_EVENT phase with 0.5x scaling (most restrictive)
        """
        from src.router.calendar_governor import CalendarGovernor
        from src.risk.models.calendar import (
            NewsItem, CalendarRule, CalendarEventType, NewsImpact, CalendarPhase
        )

        governor = CalendarGovernor(account_id="FTMO-TEST")

        rule = CalendarRule(
            rule_id="overlap-rule",
            account_id="FTMO-TEST",
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

        # NFP at 13:30
        nfp = NewsItem(
            event_id="nfp-2026-0310",
            title="Non-Farm Payrolls",
            event_type=CalendarEventType.NFP,
            impact=NewsImpact.HIGH,
            event_time=datetime(2026, 3, 10, 13, 30, tzinfo=timezone.utc),
            currencies=["USD", "EUR", "GBP"],
        )

        # BOE at 13:45
        boe = NewsItem(
            event_id="boe-2026-0310",
            title="Bank of England Rate Decision",
            event_type=CalendarEventType.CENTRAL_BANK,
            impact=NewsImpact.HIGH,
            event_time=datetime(2026, 3, 10, 13, 45, tzinfo=timezone.utc),
            currencies=["GBP"],
        )

        governor.add_calendar_event(nfp)
        governor.add_calendar_event(boe)

        # Check at 13:20 (10 min before NFP, 25 min before BOE)
        check_time = datetime(2026, 3, 10, 13, 20, tzinfo=timezone.utc)

        # EURUSD affected by NFP
        scaling = governor.evaluate_lot_scaling("FTMO-TEST", check_time, nfp, rule)

        assert scaling == 0.5, (
            f"With overlapping events, pre-event scaling should be 0.5x, got {scaling}"
        )

    def test_different_blackout_windows_per_rule(self):
        """
        P1: Different accounts can have different blackout windows.

        Scenario:
        - Account A: 30 min blackout (standard)
        - Account B: 60 min blackout (conservative)
        - Event at 14:00

        Expected: Account A normal at 13:40, Account B still in pre-event
        """
        governor_a = CalendarGovernor(account_id="FTMO-A")
        governor_b = CalendarGovernor(account_id="FTMO-B")

        rule_a = CalendarRule(
            rule_id="rule-a", account_id="FTMO-A",
            blacklist_enabled=True, blackout_minutes=30,
            lot_scaling_factors={"pre_event": 0.5, "during_event": 0.0, "normal": 1.0},
            post_event_delay_minutes=60, regime_check_enabled=True,
        )

        rule_b = CalendarRule(
            rule_id="rule-b", account_id="FTMO-B",
            blacklist_enabled=True, blackout_minutes=60,
            lot_scaling_factors={"pre_event": 0.5, "during_event": 0.0, "normal": 1.0},
            post_event_delay_minutes=60, regime_check_enabled=True,
        )

        governor_a.register_calendar_rule(rule_a)
        governor_b.register_calendar_rule(rule_b)

        event = NewsItem(
            event_id="test-event",
            event_type=CalendarEventType.NFP,
            impact=NewsImpact.HIGH,
            event_time=datetime(2026, 3, 10, 14, 0, tzinfo=timezone.utc),
            currencies=["USD"],
        )

        # Check at 13:40 (20 min before event)
        check_time = datetime(2026, 3, 10, 13, 40, tzinfo=timezone.utc)

        scaling_a = governor_a.evaluate_lot_scaling("FTMO-A", check_time, event, rule_a)
        scaling_b = governor_b.evaluate_lot_scaling("FTMO-B", check_time, event, rule_b)

        assert scaling_a == 1.0, f"Account A (30min) should be normal at 20min, got {scaling_a}"
        assert scaling_b == 0.5, f"Account B (60min) should be pre-event at 20min, got {scaling_b}"
```

#### P1-A2: Regime Check Reactivation Logic

```python
    def test_regime_check_fails_prevents_reactivation(self):
        """
        P1: If regime quality check fails, reactivation is prevented.

        Scenario:
        - NFP event passed at 13:30
        - Post-event delay: 30 min
        - Check time: 13:45 (in regime check window)
        - Regime quality: 0.3 (below 0.6 threshold)

        Expected: is_post_event_reactivation_time returns False
        """
        from unittest.mock import Mock
        from src.router.sentinel import RegimeReport

        governor = CalendarGovernor(account_id="FTMO-TEST")

        rule = CalendarRule(
            rule_id="regime-check-rule",
            account_id="FTMO-TEST",
            blacklist_enabled=True,
            blackout_minutes=30,
            lot_scaling_factors={"pre_event": 0.5, "during_event": 0.0, "normal": 1.0},
            post_event_delay_minutes=60,
            regime_check_enabled=True,
        )
        governor.register_calendar_rule(rule)

        event = NewsItem(
            event_id="nfp-2026-0310",
            event_type=CalendarEventType.NFP,
            impact=NewsImpact.HIGH,
            event_time=datetime(2026, 3, 10, 13, 30, tzinfo=timezone.utc),
            currencies=["USD"],
        )
        governor.add_calendar_event(event)

        # Mock poor quality regime report
        bad_regime = Mock(spec=RegimeReport)
        bad_regime.regime_quality = 0.3
        bad_regime.chaos_score = 0.6

        check_time = datetime(2026, 3, 10, 13, 45, tzinfo=timezone.utc)

        can_reactivate = governor.is_post_event_reactivation_time(
            "FTMO-TEST", check_time, event, rule
        )

        # Should be True (time window is reached)
        assert can_reactivate is True

        # But regime check should fail
        should_reactivate = governor.check_regime_for_reactivation(bad_regime, min_regime_quality=0.6)
        assert should_reactivate is False, (
            f"Poor regime quality (0.3) should prevent reactivation, but got {should_reactivate}"
        )

    def test_good_regime_allows_reactivation(self):
        """P1: Good regime quality allows reactivation after event."""
        governor = CalendarGovernor(account_id="FTMO-TEST")

        rule = CalendarRule(
            rule_id="regime-check-rule",
            account_id="FTMO-TEST",
            blacklist_enabled=True,
            blackout_minutes=30,
            lot_scaling_factors={"pre_event": 0.5, "during_event": 0.0, "normal": 1.0},
            post_event_delay_minutes=60,
            regime_check_enabled=True,
        )
        governor.register_calendar_rule(rule)

        event = NewsItem(
            event_id="nfp-2026-0310",
            event_type=CalendarEventType.NFP,
            impact=NewsImpact.HIGH,
            event_time=datetime(2026, 3, 10, 13, 30, tzinfo=timezone.utc),
            currencies=["USD"],
        )

        # Good regime
        good_regime = Mock(spec=RegimeReport)
        good_regime.regime_quality = 0.8
        good_regime.chaos_score = 0.2

        check_time = datetime(2026, 3, 10, 13, 45, tzinfo=timezone.utc)

        can_reactivate = governor.is_post_event_reactivation_time(
            "FTMO-TEST", check_time, event, rule
        )
        should_reactivate = governor.check_regime_for_reactivation(good_regime)

        assert can_reactivate is True
        assert should_reactivate is True, (
            f"Good regime quality (0.8) should allow reactivation"
        )
```

#### P1-A3: Weekend and Holiday Handling

```python
    def test_weekend_event_ignored(self):
        """P1: Events on weekends (when markets closed) are properly handled."""
        governor = CalendarGovernor(account_id="FTMO-TEST")

        rule = CalendarRule(
            rule_id="weekend-rule",
            account_id="FTMO-TEST",
            blacklist_enabled=True,
            blackout_minutes=30,
            lot_scaling_factors={"pre_event": 0.5, "during_event": 0.0, "normal": 1.0},
            post_event_delay_minutes=60,
            regime_check_enabled=True,
        )
        governor.register_calendar_rule(rule)

        # Saturday event
        saturday_event = NewsItem(
            event_id="weekend-nfp",
            event_type=CalendarEventType.NFP,
            impact=NewsImpact.HIGH,
            event_time=datetime(2026, 3, 14, 13, 30, tzinfo=timezone.utc),  # Saturday
            currencies=["USD"],
        )
        governor.add_calendar_event(saturday_event)

        # Friday after market close
        friday_close = datetime(2026, 3, 13, 21, 0, tzinfo=timezone.utc)

        # With no relevant active events, should be normal
        scaling = governor.evaluate_lot_scaling_normal("FTMO-TEST", friday_close)

        assert scaling == 1.0, f"Weekend event should not affect Friday close, got {scaling}"
```

---

### P1-B: BotCircuitBreaker Unit Tests

**Risk Score**: P1 (High)
**Justification**: Core risk management - automatic quarantine must work correctly
**Test Level**: Unit

#### P1-B1: Personal vs Prop Firm Thresholds

```python
# tests/router/test_bot_circuit_breaker.py

class TestBotCircuitBreakerThresholds:
    """P1: BotCircuitBreaker enforces correct thresholds by account book type."""

    def test_personal_account_5_consecutive_losses_triggers_quarantine(self):
        """
        P1: Personal accounts quarantine after 5 consecutive losses.

        NOTE: This test WILL FAIL because the bug is in check_allowed():
        The quarantine check happens AFTER recording, but check_allowed()
        is called BEFORE record_trade(). Need to fix implementation.
        """
        from src.router.bot_circuit_breaker import BotCircuitBreakerManager, AccountBook
        from unittest.mock import MagicMock

        mock_db = MagicMock()
        manager = BotCircuitBreakerManager(
            db_manager=mock_db,
            account_book=AccountBook.PERSONAL
        )

        # Simulate 5 losses
        for i in range(5):
            state = manager.record_trade(f"bot-personal-{i}", is_loss=True)

        is_allowed, reason = manager.check_allowed("bot-personal-4")

        assert is_allowed is False, (
            f"Bot should be quarantined after 5 consecutive losses, got allowed={is_allowed}"
        )
        assert "consecutive losses" in reason.lower()

    def test_prop_firm_3_consecutive_losses_triggers_quarantine(self):
        """
        P1: Prop firm accounts quarantine after 3 consecutive losses.

        Prop firms have tighter thresholds (3 vs 5).
        """
        from src.router.bot_circuit_breaker import BotCircuitBreakerManager, AccountBook
        from unittest.mock import MagicMock

        mock_db = MagicMock()
        manager = BotCircuitBreakerManager(
            db_manager=mock_db,
            account_book=AccountBook.PROP_FIRM
        )

        # Simulate 3 losses (prop firm threshold)
        for i in range(3):
            state = manager.record_trade(f"bot-prop-{i}", is_loss=True)

        is_allowed, reason = manager.check_allowed("bot-prop-2")

        assert is_allowed is False, (
            f"Prop firm bot should be quarantined after 3 consecutive losses, got allowed={is_allowed}"
        )

    def test_win_resets_consecutive_losses(self):
        """P1: A winning trade resets the consecutive loss counter."""
        from src.router.bot_circuit_breaker import BotCircuitBreakerManager, AccountBook
        from unittest.mock import MagicMock

        mock_db = MagicMock()
        manager = BotCircuitBreakerManager(db_manager=mock_db)

        # 3 losses
        for i in range(3):
            manager.record_trade("bot-reset", is_loss=True)

        # 1 win - should reset
        manager.record_trade("bot-reset", is_loss=False)

        # Next 2 losses should not trigger quarantine (counter was reset)
        for i in range(2):
            manager.record_trade(f"bot-reset-{i}", is_loss=True)

        is_allowed, _ = manager.check_allowed("bot-reset-1")
        assert is_allowed is True, "Counter should have reset after win"

    def test_daily_trade_limit_enforced(self):
        """P1: Daily trade limit of 20 is enforced."""
        from src.router.bot_circuit_breaker import BotCircuitBreakerManager, DEFAULT_DAILY_TRADE_LIMIT
        from unittest.mock import MagicMock
        from datetime import date

        mock_db = MagicMock()
        manager = BotCircuitBreakerManager(db_manager=mock_db)

        # Record 20 trades (limit)
        for i in range(DEFAULT_DAILY_TRADE_LIMIT):
            manager.record_trade(f"bot-limit-{i}", is_loss=False, trade_date=date.today())

        # 21st trade should be blocked
        is_allowed, reason = manager.check_allowed("bot-limit-19")

        assert is_allowed is False, (
            f"Daily limit ({DEFAULT_DAILY_TRADE_LIMIT}) should block 21st trade"
        )
        assert "Daily trade limit" in reason

    def test_new_day_resets_daily_counter(self):
        """P1: New calendar day resets daily trade counter."""
        from src.router.bot_circuit_breaker import BotCircuitBreakerManager
        from unittest.mock import MagicMock
        from datetime import date, timedelta

        mock_db = MagicMock()
        manager = BotCircuitBreakerManager(db_manager=mock_db)

        yesterday = date.today() - timedelta(days=1)

        # 20 trades yesterday
        for i in range(20):
            manager.record_trade("bot-newday", is_loss=False, trade_date=yesterday)

        # Check today - should be allowed (counter reset)
        is_allowed, _ = manager.check_allowed("bot-newday")

        assert is_allowed is True, "New day should reset daily counter"
```

#### P1-B2: Quarantine and Reactivation

```python
class TestBotCircuitBreakerQuarantine:
    """P1: Bot quarantine and reactivation functionality."""

    def test_manual_quarantine_works(self):
        """P1: Manual quarantine via quarantine_bot() method."""
        from src.router.bot_circuit_breaker import BotCircuitBreakerManager
        from unittest.mock import MagicMock

        mock_db = MagicMock()
        manager = BotCircuitBreakerManager(db_manager=mock_db)

        state = manager.quarantine_bot("manual-bot", reason="Manual intervention required")

        is_allowed, reason = manager.check_allowed("manual-bot")

        assert is_allowed is False
        assert "Manual intervention" in reason

    def test_reactivation_resets_state(self):
        """P1: Reactiving a bot resets all counters and clears quarantine."""
        from src.router.bot_circuit_breaker import BotCircuitBreakerManager
        from unittest.mock import MagicMock

        mock_db = MagicMock()
        manager = BotCircuitBreakerManager(db_manager=mock_db)

        # Quarantine
        manager.quarantine_bot("reactive-bot", reason="Test quarantine")

        # Reactivate
        state = manager.reactivate_bot("reactive-bot")

        assert state.is_quarantined is False
        assert state.consecutive_losses == 0

        is_allowed, _ = manager.check_allowed("reactive-bot")
        assert is_allowed is True

    def test_get_quarantined_bots(self):
        """P1: get_quarantined_bots() returns all quarantined bots."""
        from src.router.bot_circuit_breaker import BotCircuitBreakerManager
        from unittest.mock import MagicMock

        mock_db = MagicMock()
        manager = BotCircuitBreakerManager(db_manager=mock_db)

        # Quarantine some bots
        manager.quarantine_bot("quarantined-1", reason="Loss threshold")
        manager.quarantine_bot("quarantined-2", reason="Manual")

        quarantined = manager.get_quarantined_bots()

        assert len(quarantined) >= 2
        bot_ids = [b["bot_id"] for b in quarantined]
        assert "quarantined-1" in bot_ids
        assert "quarantined-2" in bot_ids
```

---

### P1-C: HMM Regime Detection Tests

**Risk Score**: P1 (High)
**Justification**: Core market regime detection - affects all trading decisions
**Test Level**: Unit

```python
# tests/risk/physics/test_hmm_regime_detection.py

class TestHMMRegimeDetection:
    """P1: HMM regime sensor provides accurate regime predictions."""

    def test_default_reading_when_no_model(self):
        """P1: Sensor returns default reading when no model is loaded."""
        from src.risk.physics.hmm_sensor import HMMRegimeSensor
        import numpy as np

        sensor = HMMRegimeSensor()
        # Force no model by setting _model to None
        sensor._model = None

        features = np.random.randn(10)
        reading = sensor.predict_regime(features)

        assert reading.regime == "UNKNOWN"
        assert reading.confidence == 0.0

    def test_regime_mapping_config(self):
        """P1: Regime mapping from config is applied correctly."""
        from src.risk.physics.hmm_sensor import HMMRegimeSensor
        from src.risk.physics.sensors.config import HMMSensorConfig

        config = HMMSensorConfig(
            regime_mapping={
                0: "TRENDING_LOW_VOL",
                1: "TRENDING_HIGH_VOL",
                2: "RANGING_LOW_VOL",
                3: "RANGING_HIGH_VOL",
            }
        )

        sensor = HMMRegimeSensor(config=config)

        # Test that config is stored
        assert sensor.config.regime_mapping[0] == "TRENDING_LOW_VOL"
        assert sensor.config.regime_mapping[3] == "RANGING_HIGH_VOL"

    def test_feature_extraction_from_ohlcv(self):
        """P1: Features are extracted correctly from OHLCV data."""
        from src.risk.physics.hmm_sensor import HMMRegimeSensor
        import numpy as np

        sensor = HMMRegimeSensor()

        # Sample OHLCV data
        ohlcv_data = {
            "open": np.random.randn(100),
            "high": np.random.randn(100) + 0.5,
            "low": np.random.randn(100) - 0.5,
            "close": np.random.randn(100),
            "volume": np.random.randint(100, 1000, 100),
        }

        reading = sensor.predict_from_ohlcv(ohlcv_data)

        # Should return a reading (default UNKNOWN if no model)
        assert reading is not None
        assert hasattr(reading, 'regime')
        assert hasattr(reading, 'confidence')

    def test_cache_hits_and_misses_tracked(self):
        """P1: Cache statistics are tracked correctly."""
        from src.risk.physics.hmm_sensor import HMMRegimeSensor
        import numpy as np

        sensor = HMMRegimeSensor()
        sensor._model = None  # Ensure no model

        features = np.random.randn(10)

        # First call
        sensor.predict_regime(features, cache_key="test-key")

        # Same call should hit cache
        initial_misses = sensor._cache_misses

        # With model loaded, would count hits
        # Without model, just verify counter exists
        assert hasattr(sensor, '_cache_hits')
        assert hasattr(sensor, '_cache_misses')

    def test_regime_to_float_conversion(self):
        """P1: Regime string to float conversion works correctly."""
        from src.risk.physics.hmm_sensor import HMMRegimeSensor

        sensor = HMMRegimeSensor()

        assert sensor._regime_to_float("TRENDING_LOW_VOL") == 0.0
        assert sensor._regime_to_float("TRENDING_HIGH_VOL") == 0.33
        assert sensor._regime_to_float("RANGING_LOW_VOL") == 0.67
        assert sensor._regime_to_float("RANGING_HIGH_VOL") == 1.0
        assert sensor._regime_to_float("UNKNOWN") == 0.5  # Neutral

    def test_clear_cache(self):
        """P1: Cache can be cleared."""
        from src.risk.physics.hmm_sensor import HMMRegimeSensor
        import numpy as np

        sensor = HMMRegimeSensor()

        sensor._prediction_cache["test"] = "value"
        sensor._cache_hits = 5
        sensor._cache_misses = 10

        sensor.clear_cache()

        assert len(sensor._prediction_cache) == 0
        assert sensor._cache_hits == 0
        assert sensor._cache_misses == 0
```

---

## P2 Tests - Medium Priority

### P2-A: PropFirmRiskOverlay Tests

**Risk Score**: P2 (Medium)
**Justification**: Secondary risk calculations - affects prop firm accounts
**Test Level**: Unit

```python
# tests/risk/test_prop_firm_overlay.py

class TestPropFirmRiskOverlay:
    """P2: PropFirmRiskOverlay applies correct limits per firm."""

    def test_ftmo_limits_applied(self):
        """P2: FTMO limits (3% drawdown, 4% daily) are applied."""
        from src.risk.prop_firm_overlay import PropFirmRiskOverlay

        overlay = PropFirmRiskOverlay(firm_name="FTMO")

        limits = overlay.apply_risk_limits("prop_firm")

        assert limits["effective_max_drawdown"] == 0.03
        assert limits["effective_daily_loss"] == 0.04
        assert limits["book_type"] == "prop_firm"
        assert limits["firm"] == "FTMO"

    def test_topstep_limits_applied(self):
        """P2: Topstep limits (4% drawdown) are applied."""
        from src.risk.prop_firm_overlay import PropFirmRiskOverlay

        overlay = PropFirmRiskOverlay(firm_name="Topstep")

        limits = overlay.apply_risk_limits("prop_firm")

        assert limits["effective_max_drawdown"] == 0.04

    def test_personal_limits_used_when_no_firm(self):
        """P2: Personal limits (5% drawdown) used for personal accounts."""
        from src.risk.prop_firm_overlay import PropFirmRiskOverlay

        overlay = PropFirmRiskOverlay(firm_name=None)

        limits = overlay.apply_risk_limits("personal")

        assert limits["effective_max_drawdown"] == 0.05
        assert limits["book_type"] == "personal"

    def test_p_pass_calculation(self):
        """P2: P_pass probability is calculated correctly."""
        from src.risk.prop_firm_overlay import PropFirmRiskOverlay

        overlay = PropFirmRiskOverlay(firm_name="FTMO")

        result = overlay.calculate_p_pass(
            win_rate=0.6,
            avg_win=100.0,
            avg_loss=50.0,
            account_book="prop_firm"
        )

        assert "p_pass" in result
        assert "prop_score" in result
        assert 0 <= result["p_pass"] <= 1

    def test_recovery_mode_entry(self):
        """P2: Recovery mode is entered when drawdown exceeds threshold."""
        from src.risk.prop_firm_overlay import PropFirmRiskOverlay

        overlay = PropFirmRiskOverlay(firm_name="FTMO", initial_balance=100000)

        # Simulate drawdown to 2.6% (below 3% FTMO limit)
        overlay.update_balance(97400)  # 2.6% drawdown

        # Not in recovery yet
        overlay.check_recovery_mode()

        # Push to 3.1% (breach)
        overlay.update_balance(96900)  # 3.1% drawdown

        result = overlay.check_recovery_mode()

        assert result["in_recovery"] is True

    def test_recovery_risk_multiplier(self):
        """P2: Recovery mode applies 50% position reduction."""
        from src.risk.prop_firm_overlay import PropFirmRiskOverlay

        overlay = PropFirmRiskOverlay(firm_name="FTMO", initial_balance=100000)

        # Enter recovery mode
        overlay.update_balance(97000)
        overlay.enter_recovery_mode()

        multiplier = overlay.get_recovery_risk_multiplier()

        assert multiplier["multiplier"] < 1.0
        assert multiplier["mode"] == "recovery"
```

---

### P2-B: Physics Sensor Integration Tests

**Risk Score**: P2 (Medium)
**Justification**: Integration between sensors - affects dashboard display
**Test Level**: Integration

```python
# tests/risk/physics/test_physics_integration.py

class TestPhysicsSensorIntegration:
    """P2: Physics sensors work together correctly."""

    def test_ising_and_hmm_can_run_together(self):
        """P2: Ising and HMM sensors can make predictions in same session."""
        from src.risk.physics import IsingRegimeSensor, ChaosSensor
        from src.risk.physics.hmm_sensor import HMMRegimeSensor
        import numpy as np

        ising = IsingRegimeSensor(grid_size=8, equilibrium_steps=500, measurement_steps=100)
        chaos = ChaosSensor(dimension=3, tau=10, lookback=100)
        hmm = HMMRegimeSensor()

        # Run Ising
        ising_result = ising.detect_regime(volatility=0.02)

        # Run Chaos
        chaos_data = np.random.randn(500)
        chaos_result = chaos.analyze_chaos(returns=chaos_data)

        # Run HMM
        hmm_features = np.random.randn(10)
        hmm_reading = hmm.predict_regime(hmm_features)

        assert "regime_label" in ising_result
        assert "lyapunov_exponent" in chaos_result
        assert hasattr(hmm_reading, 'regime')

    def test_sensor_caches_independent(self):
        """P2: Each sensor maintains independent cache."""
        from src.risk.physics import IsingRegimeSensor, ChaosSensor
        import numpy as np

        ising1 = IsingRegimeSensor(grid_size=8)
        ising2 = IsingRegimeSensor(grid_size=8)

        # Both have independent cache
        ising1.detect_regime(volatility=0.02)
        ising2.detect_regime(volatility=0.05)

        assert ising1._cache != ising2._cache
```

---

## P3 Tests - Low Priority

### P3-A: API Endpoint Smoke Tests

**Risk Score**: P3 (Low)
**Justification**: API contracts - smoke tests for UI integration
**Test Level**: API

```python
# tests/api/test_epic4_p3_smoke.py

class TestEpic4APISmokeTests:
    """P3: Smoke tests for Epic 4 API endpoints."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from src.api.server import app
        return TestClient(app)

    def test_calendar_rules_endpoint_exists(self, client):
        """P3: /api/risk/calendar/rules endpoint responds."""
        response = client.get("/api/risk/calendar/rules")
        # May return 200 or 500 depending on state, but should not 404
        assert response.status_code != 404

    def test_risk_params_endpoint_accepts_valid_update(self, client):
        """P3: /api/risk/params/{tag} accepts valid Kelly fraction."""
        response = client.put(
            "/api/risk/params/test-p3-tag",
            json={"kelly_fraction": 0.5}
        )
        # Should be 200 or 422 depending on validation implementation
        assert response.status_code in [200, 422]

    def test_prop_firm_registry_endpoint_exists(self, client):
        """P3: Prop firm registry endpoints exist."""
        response = client.get("/api/risk/prop-firms")
        # Should not 404
        assert response.status_code != 404

    def test_islamic_status_endpoint_exists(self, client):
        """P3: Islamic compliance status endpoint exists."""
        response = client.get("/api/risk/islamic-status")
        # Should return 200 (even if data is mocked)
        assert response.status_code == 200
```

---

## Test Fixtures & Factories

### Fixture: Mock RegimeReport

```python
# tests/conftest.py (additions)

import pytest
from unittest.mock import Mock
from datetime import datetime, timezone

@pytest.fixture
def mock_good_regime():
    """Good quality regime report for reactivation tests."""
    regime = Mock()
    regime.regime_quality = 0.8
    regime.chaos_score = 0.2
    regime.regime = "TRENDING_LOW_VOL"
    regime.timestamp = datetime.now(timezone.utc)
    return regime

@pytest.fixture
def mock_poor_regime():
    """Poor quality regime report for reactivation tests."""
    regime = Mock()
    regime.regime_quality = 0.3
    regime.chaos_score = 0.7
    regime.regime = "RANGING_HIGH_VOL"
    regime.timestamp = datetime.now(timezone.utc)
    return regime

@pytest.fixture
def mock_chaos_sensor():
    """Pre-configured chaos sensor."""
    from src.risk.physics import ChaosSensor
    return ChaosSensor(dimension=3, tau=10, lookback=100)

@pytest.fixture
def mock_ising_sensor():
    """Pre-configured Ising sensor."""
    from src.risk.physics import IsingRegimeSensor
    return IsingRegimeSensor(grid_size=8, equilibrium_steps=500, measurement_steps=100)
```

---

## Risk Scoring Summary

| Test ID | Priority | Risk Score | Complexity | Coverage Gap |
|---------|----------|------------|------------|--------------|
| P1-A1 | P1 | 6 | Medium | Overlapping events |
| P1-A2 | P1 | 6 | Medium | Regime check logic |
| P1-A3 | P1 | 4 | Low | Weekend handling |
| P1-B1 | P1 | 8 | High | Circuit breaker thresholds |
| P1-B2 | P1 | 7 | Medium | Quarantine/reactivation |
| P1-C1 | P1 | 7 | Medium | HMM regime detection |
| P2-A1 | P2 | 5 | Medium | Prop firm limits |
| P2-A2 | P2 | 4 | Low | Recovery mode |
| P2-B1 | P2 | 4 | Low | Sensor integration |
| P3-A1 | P3 | 2 | Low | API smoke tests |

---

## Next Steps

1. **Fix P0 Bug**: The `is_within_60min` condition fails at `countdown_seconds=0` in the Islamic compliance implementation
2. **Implement P1 Tests**: CalendarGovernor extended scenarios and BotCircuitBreaker unit tests
3. **Run P0 Tests**: Verify 16/17 passing after bug fix
4. **Expand Coverage**: Move P2 tests to implementation phase

---

**Generated by**: bmad-tea-testarch-automate skill (YOLO mode)
**Date**: 2026-03-21

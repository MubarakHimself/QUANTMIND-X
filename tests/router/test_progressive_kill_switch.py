"""
Tests for Progressive Kill Switch System

Comprehensive tests for all 5 tiers of protection and alert progression.

**Validates: Phase 10 - Testing Strategy**
"""

import pytest
from datetime import datetime, timezone, time, timedelta
from unittest.mock import Mock, patch, MagicMock, AsyncMock, PropertyMock
from typing import Dict, Any
import asyncio

# Import components
from src.router.alert_manager import (
    AlertManager, AlertLevel, Alert,
    get_alert_manager, reset_alert_manager
)
from src.router.strategy_monitor import (
    StrategyFamilyMonitor, FamilyState,
    get_strategy_monitor, reset_strategy_monitor
)
from src.router.account_monitor import (
    AccountMonitor, AccountState,
    get_account_monitor, reset_account_monitor
)
from src.router.session_monitor import (
    SessionMonitor, SessionState,
    get_session_monitor, reset_session_monitor
)
from src.router.system_monitor import (
    SystemMonitor, SystemState,
    get_system_monitor, reset_system_monitor
)
from src.router.progressive_kill_switch import (
    ProgressiveKillSwitch, ProgressiveConfig,
    get_progressive_kill_switch, reset_progressive_kill_switch
)
from src.router.bot_manifest import BotManifest, StrategyType


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def reset_all_singletons():
    """Reset all singleton instances before each test."""
    reset_progressive_kill_switch()
    reset_alert_manager()
    reset_strategy_monitor()
    reset_account_monitor()
    reset_session_monitor()
    reset_system_monitor()
    yield
    reset_progressive_kill_switch()
    reset_alert_manager()
    reset_strategy_monitor()
    reset_account_monitor()
    reset_session_monitor()
    reset_system_monitor()


@pytest.fixture(autouse=True)
def stub_bot_circuit_breaker(monkeypatch):
    """Keep ProgressiveKillSwitch tests isolated from DB-backed circuit breaker setup."""

    class _StubBotCircuitBreakerManager:
        def check_allowed(self, bot_id):
            return True, None

        def get_quarantined_bots(self):
            return []

    monkeypatch.setattr(
        "src.router.progressive_kill_switch.BotCircuitBreakerManager",
        lambda *args, **kwargs: _StubBotCircuitBreakerManager(),
    )


@pytest.fixture
def alert_manager():
    """Create a fresh AlertManager for testing."""
    return AlertManager()


@pytest.fixture
def strategy_monitor(alert_manager):
    """Create a StrategyFamilyMonitor for testing."""
    return StrategyFamilyMonitor(alert_manager=alert_manager)


@pytest.fixture
def account_monitor(alert_manager):
    """Create an AccountMonitor for testing."""
    return AccountMonitor(alert_manager=alert_manager)


@pytest.fixture
def session_monitor(alert_manager):
    """Create a SessionMonitor for testing."""
    return SessionMonitor(alert_manager=alert_manager)


@pytest.fixture
def system_monitor(alert_manager):
    """Create a SystemMonitor for testing."""
    return SystemMonitor(alert_manager=alert_manager)


@pytest.fixture
def sample_bot_manifest():
    """Create a sample bot manifest for testing."""
    return BotManifest(
        bot_id="test_bot_01",
        name="Test Bot",
        description="Test bot for unit tests",
        strategy_type=StrategyType.SCALPER,
        frequency="HIGH",
        min_capital_req=100.0,
        symbols=["EURUSD"]
    )


@pytest.fixture
def mock_sentinel():
    """Create a mock Sentinel for testing."""
    sentinel = Mock()
    report = Mock()
    report.chaos_score = 0.0
    report.regime = "TREND_STABLE"
    sentinel.current_report = report
    return sentinel


@pytest.fixture
def mock_news_sensor():
    """Create a mock NewsSensor for testing."""
    sensor = Mock()
    sensor.check_state.return_value = "SAFE"
    return sensor


@pytest.fixture
def progressive_config():
    """Create a test configuration."""
    return ProgressiveConfig(
        enabled=True,
        tier2_max_failed_bots=3,
        tier2_max_family_loss_pct=0.20,
        tier3_max_daily_loss_pct=0.03,
        tier3_max_weekly_loss_pct=0.10,
        tier4_end_of_day_utc="21:00",
        tier4_max_trades_per_minute=10,
        tier5_max_chaos_score=0.75
    )


# =============================================================================
# Alert Manager Tests
# =============================================================================

class TestAlertManager:
    """Tests for AlertManager class."""

    def test_alert_level_calculation(self, alert_manager):
        """Test threshold to level mapping."""
        assert alert_manager.calculate_level(0) == AlertLevel.GREEN
        assert alert_manager.calculate_level(25) == AlertLevel.GREEN
        assert alert_manager.calculate_level(49) == AlertLevel.GREEN
        assert alert_manager.calculate_level(50) == AlertLevel.YELLOW
        assert alert_manager.calculate_level(60) == AlertLevel.YELLOW
        assert alert_manager.calculate_level(70) == AlertLevel.ORANGE
        assert alert_manager.calculate_level(80) == AlertLevel.ORANGE
        assert alert_manager.calculate_level(85) == AlertLevel.RED
        assert alert_manager.calculate_level(90) == AlertLevel.RED
        assert alert_manager.calculate_level(95) == AlertLevel.BLACK
        assert alert_manager.calculate_level(100) == AlertLevel.BLACK

    def test_raise_alert(self, alert_manager):
        """Test raising an alert."""
        alert = alert_manager.raise_alert(
            tier=2,
            message="Test alert",
            threshold_pct=75.0,
            source="test"
        )

        assert alert.level == AlertLevel.ORANGE
        assert alert.tier == 2
        assert alert.message == "Test alert"
        assert alert.threshold_pct == 75.0
        assert alert.source == "test"
        assert len(alert_manager.active_alerts) == 1
        assert alert_manager.current_level == AlertLevel.ORANGE

    def test_alert_progression(self, alert_manager):
        """Test GREEN→YELLOW→ORANGE→RED→BLACK transitions."""
        # Start at GREEN
        assert alert_manager.current_level == AlertLevel.GREEN

        # Raise YELLOW
        alert_manager.raise_alert(tier=1, message="Warning", threshold_pct=55.0, source="test")
        assert alert_manager.current_level == AlertLevel.YELLOW

        # Raise ORANGE (higher priority)
        alert_manager.raise_alert(tier=2, message="Elevated", threshold_pct=75.0, source="test")
        assert alert_manager.current_level == AlertLevel.ORANGE

        # Raise RED
        alert_manager.raise_alert(tier=3, message="Critical", threshold_pct=90.0, source="test")
        assert alert_manager.current_level == AlertLevel.RED

        # Raise BLACK
        alert_manager.raise_alert(tier=5, message="Emergency", threshold_pct=100.0, source="test")
        assert alert_manager.current_level == AlertLevel.BLACK

    def test_clear_alerts(self, alert_manager):
        """Test clearing alerts."""
        alert_manager.raise_alert(tier=1, message="Test", threshold_pct=60.0, source="test1")
        alert_manager.raise_alert(tier=2, message="Test", threshold_pct=70.0, source="test2")

        assert len(alert_manager.active_alerts) == 2

        cleared = alert_manager.clear_alerts_by_tier(1)
        assert cleared == 1
        assert len(alert_manager.active_alerts) == 1

        cleared = alert_manager.clear_alerts_by_source("test2")
        assert cleared == 1
        assert len(alert_manager.active_alerts) == 0
        assert alert_manager.current_level == AlertLevel.GREEN


# =============================================================================
# Tier 2: Strategy Monitor Tests
# =============================================================================

class TestStrategyMonitor:
    """Tests for StrategyFamilyMonitor class."""

    def test_record_bot_failure(self, strategy_monitor, sample_bot_manifest):
        """Test recording bot failures."""
        # First failure shouldn't quarantine
        quarantined = strategy_monitor.record_bot_failure(sample_bot_manifest)
        assert not quarantined
        assert strategy_monitor.is_family_allowed(StrategyType.SCALPER)

        # Second failure
        sample_bot_manifest.bot_id = "test_bot_02"
        quarantined = strategy_monitor.record_bot_failure(sample_bot_manifest)
        assert not quarantined

        # Third failure should quarantine
        sample_bot_manifest.bot_id = "test_bot_03"
        quarantined = strategy_monitor.record_bot_failure(sample_bot_manifest)
        assert quarantined
        assert not strategy_monitor.is_family_allowed(StrategyType.SCALPER)

    def test_family_loss_tracking(self, strategy_monitor):
        """Test family P&L tracking."""
        # Record losses
        quarantined = strategy_monitor.record_family_pnl(
            StrategyType.SCALPER,
            pnl=-500.0,
            initial_capital=10000.0
        )
        assert not quarantined

        # Check state
        state = strategy_monitor.get_family_state(StrategyType.SCALPER)
        assert state["total_pnl"] == -500.0
        assert state["loss_pct"] == 0.05  # 5%

    def test_strategy_family_quarantine(self, strategy_monitor):
        """Test full family quarantine on max loss."""
        # Record 20% loss
        quarantined = strategy_monitor.record_family_pnl(
            StrategyType.SCALPER,
            pnl=-2000.0,
            initial_capital=10000.0
        )
        assert quarantined
        assert not strategy_monitor.is_family_allowed(StrategyType.SCALPER)

        # Get quarantined families
        quarantined_families = strategy_monitor.get_quarantined_families()
        assert "SCALPER" in quarantined_families

    def test_reactivate_family(self, strategy_monitor, sample_bot_manifest):
        """Test reactivating a quarantined family."""
        # Quarantine by 3 failures
        for i in range(3):
            sample_bot_manifest.bot_id = f"test_bot_{i}"
            strategy_monitor.record_bot_failure(sample_bot_manifest)

        assert not strategy_monitor.is_family_allowed(StrategyType.SCALPER)

        # Reactivate
        result = strategy_monitor.reactivate_family(StrategyType.SCALPER)
        assert result
        assert strategy_monitor.is_family_allowed(StrategyType.SCALPER)


# =============================================================================
# Tier 3: Account Monitor Tests
# =============================================================================

class TestAccountMonitor:
    """Tests for AccountMonitor class."""

    def test_record_trade_pnl(self, account_monitor):
        """Test recording trade P&L."""
        stop_triggered = account_monitor.record_trade_pnl(
            account_id="test_account",
            pnl=-100.0,
            account_balance=10000.0
        )
        assert not stop_triggered

        state = account_monitor.get_stop_status("test_account")
        assert state["daily_pnl"] == -100.0
        assert state["daily_loss_pct"] == pytest.approx(0.01, rel=0.01)

    def test_daily_loss_limit(self, account_monitor):
        """Test 3% daily limit enforcement."""
        # Record 3% loss
        stop_triggered = account_monitor.record_trade_pnl(
            account_id="test_account",
            pnl=-300.0,
            account_balance=10000.0
        )
        assert stop_triggered

        # Check account is blocked
        allowed = account_monitor.is_account_allowed("test_account")
        assert not allowed

        status = account_monitor.get_stop_status("test_account")
        assert status["daily_stop"]

    def test_weekly_loss_limit(self, account_monitor):
        """Test 10% weekly limit enforcement."""
        # Record 10% loss
        stop_triggered = account_monitor.record_trade_pnl(
            account_id="test_account",
            pnl=-1000.0,
            account_balance=10000.0
        )
        assert stop_triggered

        status = account_monitor.get_stop_status("test_account")
        assert status["weekly_stop"]

    def test_accounts_at_risk(self, account_monitor):
        """Test getting accounts at risk."""
        # Create some risk
        account_monitor.record_trade_pnl("account1", pnl=-200.0, account_balance=10000.0)
        account_monitor.record_trade_pnl("account2", pnl=-50.0, account_balance=10000.0)

        at_risk = account_monitor.get_accounts_at_risk(threshold_pct=50.0)
        assert len(at_risk) == 1
        assert at_risk[0]["account_id"] == "account1"

    def test_reset_account_stops(self, account_monitor):
        """Test resetting account stops."""
        # Trigger daily stop
        account_monitor.record_trade_pnl("test_account", pnl=-300.0, account_balance=10000.0)
        assert not account_monitor.is_account_allowed("test_account")

        # Reset
        account_monitor.reset_account_stops("test_account")
        assert account_monitor.is_account_allowed("test_account")

    def test_pressure_state_visible_before_hard_lock(self, account_monitor):
        """Pressure bands should be visible before the hard stop is hit."""
        account_monitor.record_trade_pnl(
            account_id="test_account",
            pnl=-200.0,
            account_balance=10000.0,
        )

        status = account_monitor.get_stop_status("test_account")
        assert status["hard_lock_active"] is False
        assert status["pressure_state"] in ["CAUTION", "RESTRICTED"]
        assert status["lock_source"] == "ACCOUNT_PRESSURE"
        assert status["operating_budget_consumed_ratio"] > 0

    def test_hard_lock_status_includes_reason(self, account_monitor):
        """Hard lock should expose source and reason, not only booleans."""
        account_monitor.record_trade_pnl(
            account_id="test_account",
            pnl=-300.0,
            account_balance=10000.0,
        )

        status = account_monitor.get_stop_status("test_account")
        assert status["hard_lock_active"] is True
        assert status["lock_source"] == "ACCOUNT_HARD_LOCK"
        assert "reason" in status
        assert status["pressure_state"] == "STOPPED"


# =============================================================================
# Tier 4: Session Monitor Tests
# =============================================================================

class TestSessionMonitor:
    """Tests for SessionMonitor class."""

    def test_session_allowed_normal(self, session_monitor):
        """Test normal session allows trading."""
        allowed, reason = session_monitor.check_session_allowed()
        assert allowed
        assert reason is None

    def test_news_kill_zone(self, session_monitor, mock_news_sensor):
        """Test news kill zone blocking."""
        mock_news_sensor.check_state.return_value = "KILL_ZONE"
        session_monitor.set_news_sensor(mock_news_sensor)

        allowed, reason = session_monitor.check_session_allowed()
        assert not allowed
        assert "kill zone" in reason.lower()

    def test_end_of_day_shutdown(self, session_monitor):
        """Test end of day closure at 21:00 UTC."""
        # Mock current time to be 21:30 UTC
        with patch.object(
            SessionMonitor,
            'current_time_utc',
            new_callable=PropertyMock,
        ) as mock_time:
            mock_time.return_value = datetime(2026, 2, 14, 21, 30, tzinfo=timezone.utc)

            allowed, reason = session_monitor.check_session_allowed()
            assert not allowed
            assert "21:00" in reason

    def test_rate_limiting(self, session_monitor):
        """Test rate limiting."""
        # Record 10 trades (at limit)
        for _ in range(10):
            session_monitor.record_trade()

        allowed, reason = session_monitor.check_session_allowed()
        assert not allowed
        assert "rate limit" in reason.lower()

    def test_session_status(self, session_monitor):
        """Test getting session status."""
        status = session_monitor.get_session_status()

        assert "in_trading_window" in status
        assert "news_state" in status
        assert "rate_limit" in status
        assert status["rate_limit"]["max_per_minute"] == 10


# =============================================================================
# Tier 5: System Monitor Tests
# =============================================================================

class TestSystemMonitor:
    """Tests for SystemMonitor class."""

    def test_system_health_ok(self, system_monitor, mock_sentinel):
        """Test healthy system allows trading."""
        mock_sentinel.current_report.chaos_score = 0.3
        system_monitor.set_sentinel(mock_sentinel)
        system_monitor.update_broker_ping()

        allowed, reason = system_monitor.check_system_health()
        assert allowed
        assert reason is None

    def test_chaos_threshold(self, system_monitor, mock_sentinel):
        """Test chaos score >0.75 trigger."""
        mock_sentinel.current_report.chaos_score = 0.8
        system_monitor.set_sentinel(mock_sentinel)
        system_monitor.update_broker_ping()

        allowed, reason = system_monitor.check_system_health()
        assert not allowed
        assert "chaos" in reason.lower()

    def test_broker_timeout(self, system_monitor):
        """Test broker connection loss detection."""
        # Set old ping time
        system_monitor.state.last_broker_ping = datetime.now(timezone.utc) - timedelta(seconds=60)

        allowed, reason = system_monitor.check_system_health()
        assert not allowed
        assert "broker" in reason.lower()

    def test_nuclear_shutdown(self, system_monitor, mock_sentinel):
        """Test nuclear shutdown on critical chaos."""
        mock_sentinel.current_report.chaos_score = 0.9
        system_monitor.set_sentinel(mock_sentinel)

        allowed, reason = system_monitor.check_system_health()
        assert not allowed
        assert system_monitor.is_shutdown()

    def test_reset_shutdown(self, system_monitor, mock_sentinel):
        """Test resetting shutdown state."""
        mock_sentinel.current_report.chaos_score = 0.9
        system_monitor.set_sentinel(mock_sentinel)
        system_monitor.check_system_health()

        assert system_monitor.is_shutdown()

        system_monitor.reset_shutdown()
        assert not system_monitor.is_shutdown()


# =============================================================================
# Progressive Kill Switch Orchestrator Tests
# =============================================================================

class TestProgressiveKillSwitch:
    """Tests for ProgressiveKillSwitch orchestrator."""

    @pytest.mark.asyncio
    async def test_all_tiers_pass(self, progressive_config, mock_sentinel, mock_news_sensor):
        """Test when all tiers pass."""
        mock_sentinel.current_report.chaos_score = 0.2
        mock_news_sensor.check_state.return_value = "SAFE"

        pks = ProgressiveKillSwitch(
            config=progressive_config,
            sentinel=mock_sentinel,
            news_sensor=mock_news_sensor
        )
        pks.update_broker_ping()

        context = {
            "bot_id": "test_bot",
            "account_id": "test_account",
            "strategy_family": "SCALPER",
            "current_pnl_pct": 0.0
        }

        allowed, reason = await pks.check_all_tiers(context)
        assert allowed
        assert reason is None

    @pytest.mark.asyncio
    async def test_tier_priority_order(self, progressive_config, mock_sentinel, mock_news_sensor):
        """Verify Tier 5 overrides lower tiers."""
        # Set Tier 5 condition (chaos)
        mock_sentinel.current_report.chaos_score = 0.8
        # Set Tier 4 condition (news)
        mock_news_sensor.check_state.return_value = "KILL_ZONE"

        pks = ProgressiveKillSwitch(
            config=progressive_config,
            sentinel=mock_sentinel,
            news_sensor=mock_news_sensor
        )

        context = {"bot_id": "test", "account_id": "test"}

        allowed, reason = await pks.check_all_tiers(context)
        assert not allowed
        # Should be blocked by Tier 5 (chaos) first, not Tier 4 (news)
        assert "chaos" in reason.lower()

    @pytest.mark.asyncio
    async def test_full_tier_cascade(self, progressive_config, mock_sentinel, mock_news_sensor):
        """Test multi-tier failure scenario."""
        mock_sentinel.current_report.chaos_score = 0.2
        mock_news_sensor.check_state.return_value = "SAFE"

        pks = ProgressiveKillSwitch(
            config=progressive_config,
            sentinel=mock_sentinel,
            news_sensor=mock_news_sensor
        )
        pks.update_broker_ping()

        # Trigger Tier 3 (account loss)
        pks.record_trade_pnl("test_account", pnl=-300.0, account_balance=10000.0)

        context = {
            "bot_id": "test_bot",
            "account_id": "test_account",
            "strategy_family": "SCALPER"
        }

        allowed, reason = await pks.check_all_tiers(context)
        assert not allowed
        assert "account" in reason.lower()
        assert pks.get_status()["lock_state"]["source"] == "ACCOUNT_HARD_LOCK"

    def test_status_endpoint(self, progressive_config, mock_sentinel, mock_news_sensor):
        """Test getting comprehensive status."""
        mock_sentinel.current_report.chaos_score = 0.3
        mock_news_sensor.check_state.return_value = "SAFE"

        pks = ProgressiveKillSwitch(
            config=progressive_config,
            sentinel=mock_sentinel,
            news_sensor=mock_news_sensor
        )

        status = pks.get_status()

        assert status["enabled"]
        assert status["current_alert_level"] == "GREEN"
        assert "tiers" in status
        assert "tier1_bot" in status["tiers"]
        assert "tier5_system" in status["tiers"]

    def test_configuration_loading(self):
        """Test loading configuration from YAML."""
        config = ProgressiveConfig(
            tier2_max_failed_bots=5,
            tier3_max_daily_loss_pct=0.05
        )

        pks = ProgressiveKillSwitch(config=config)

        assert pks.config.tier2_max_failed_bots == 5
        assert pks.config.tier3_max_daily_loss_pct == 0.05

    @pytest.mark.asyncio
    async def test_black_condition_triggers_kill_switch_awaited(self, progressive_config, mock_sentinel):
        """Test that BLACK condition awaits kill_switch.trigger exactly once.
        
        This test verifies the fix for the issue where async SmartKillSwitch.trigger
        was called without awaiting, causing kill actions to never execute.
        """
        # Set up mock kill switch with async trigger
        mock_kill_switch = AsyncMock()
        mock_kill_switch.trigger = AsyncMock()
        
        # Set high chaos score to trigger BLACK condition (Tier 5)
        mock_sentinel.current_report.chaos_score = 0.9
        
        pks = ProgressiveKillSwitch(
            config=progressive_config,
            sentinel=mock_sentinel,
            kill_switch=mock_kill_switch
        )
        pks.update_broker_ping()
        
        context = {
            "bot_id": "test_bot",
            "account_id": "test_account",
            "strategy_family": "SCALPER"
        }
        
        # Trigger check - should cause BLACK condition
        allowed, reason = await pks.check_all_tiers(context)
        
        # Verify trade was blocked
        assert not allowed
        assert reason is not None
        assert "chaos" in reason.lower()
        
        # Verify kill_switch.trigger was awaited exactly once
        mock_kill_switch.trigger.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_manual_market_lock_is_sticky_until_resumed(
        self,
        progressive_config,
        mock_sentinel,
        mock_news_sensor,
    ):
        """Manual market lock should block trading until explicitly resumed."""
        mock_sentinel.current_report.chaos_score = 0.2
        mock_news_sensor.check_state.return_value = "SAFE"

        pks = ProgressiveKillSwitch(
            config=progressive_config,
            sentinel=mock_sentinel,
            news_sensor=mock_news_sensor,
        )
        pks.update_broker_ping()
        pks.activate_manual_market_lock(reason="operator review")

        context = {"bot_id": "test_bot", "account_id": "test_account", "strategy_family": "SCALPER"}
        allowed, reason = await pks.check_all_tiers(context)
        assert not allowed
        assert "manual market lock" in reason.lower()
        assert pks.get_status()["lock_state"]["source"] == "MANUAL_MARKET_LOCK"

        resume_result = pks.resume_manual_market_lock()
        assert resume_result is True

        allowed, reason = await pks.check_all_tiers(context)
        assert allowed
        assert reason is None

    @pytest.mark.asyncio
    async def test_account_hard_lock_beats_session_opportunity(
        self,
        progressive_config,
        mock_sentinel,
        mock_news_sensor,
    ):
        """A safe session cannot reopen trading after an account hard lock."""
        mock_sentinel.current_report.chaos_score = 0.2
        mock_news_sensor.check_state.return_value = "SAFE"

        pks = ProgressiveKillSwitch(
            config=progressive_config,
            sentinel=mock_sentinel,
            news_sensor=mock_news_sensor,
        )
        pks.update_broker_ping()
        pks.record_trade_pnl("test_account", pnl=-300.0, account_balance=10000.0)

        context = {"bot_id": "test_bot", "account_id": "test_account", "strategy_family": "SCALPER"}
        allowed, reason = await pks.check_all_tiers(context)
        assert not allowed
        assert "account" in reason.lower()

        status = pks.get_status()
        assert status["lock_state"]["source"] == "ACCOUNT_HARD_LOCK"
        assert status["last_check_result"]["allowed"] is False


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the complete system."""

    @pytest.mark.asyncio
    async def test_progressive_shutdown_sequence(self, mock_sentinel, mock_news_sensor):
        """Test proper shutdown execution order."""
        mock_sentinel.current_report.chaos_score = 0.1
        mock_news_sensor.check_state.return_value = "SAFE"

        pks = get_progressive_kill_switch()
        pks.system_monitor.set_sentinel(mock_sentinel)
        pks.session_monitor.set_news_sensor(mock_news_sensor)
        pks.update_broker_ping()

        context = {
            "bot_id": "integration_bot",
            "account_id": "integration_account",
            "strategy_family": "STRUCTURAL"
        }

        # Start allowed
        allowed, _ = await pks.check_all_tiers(context)
        assert allowed

        # Trigger account loss (Tier 3)
        pks.record_trade_pnl("integration_account", pnl=-200.0, account_balance=10000.0)
        allowed, _ = await pks.check_all_tiers(context)
        assert allowed  # Not at 3% yet

        # More losses to trigger daily stop
        pks.record_trade_pnl("integration_account", pnl=-150.0, account_balance=10000.0)
        allowed, reason = await pks.check_all_tiers(context)
        assert not allowed

    @pytest.mark.asyncio
    async def test_alert_notification_delivery(self, mock_sentinel, mock_news_sensor):
        """Test alert notification system."""
        mock_sentinel.current_report.chaos_score = 0.8

        pks = get_progressive_kill_switch()
        pks.system_monitor.set_sentinel(mock_sentinel)

        # Check health - should trigger BLACK alert
        await pks.check_all_tiers({})

        alerts = pks.get_all_alerts()
        assert len(alerts["active_alerts"]) >= 1
        assert alerts["current_level"] == "BLACK"

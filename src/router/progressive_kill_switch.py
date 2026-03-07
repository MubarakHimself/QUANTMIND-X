"""
Progressive Kill Switch Orchestrator

Coordinates all five tiers of protection and executes appropriate actions
based on alert levels (GREEN→YELLOW→ORANGE→RED→BLACK).

**Validates: Phase 6 - Progressive Kill Switch Orchestrator**
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List, TYPE_CHECKING
import yaml

from src.router.kill_switch import SmartKillSwitch, KillReason, get_smart_kill_switch
from src.router.bot_circuit_breaker import BotCircuitBreakerManager
from src.router.alert_manager import AlertManager, AlertLevel, get_alert_manager
from src.router.strategy_monitor import StrategyFamilyMonitor, get_strategy_monitor
from src.router.account_monitor import AccountMonitor, get_account_monitor
from src.router.session_monitor import SessionMonitor, get_session_monitor
from src.router.system_monitor import SystemMonitor, get_system_monitor
from src.router.bot_manifest import StrategyType, AccountBook

if TYPE_CHECKING:
    from src.router.sentinel import Sentinel
    from src.router.sensors.news import NewsSensor
    from mcp_mt5.alert_service import AlertService

logger = logging.getLogger(__name__)


@dataclass
class ProgressiveConfig:
    """Configuration for the progressive kill switch system."""
    enabled: bool = True

    # Tier 2: Strategy-Level
    tier2_max_failed_bots: int = 3
    tier2_max_family_loss_pct: float = 0.20

    # Tier 3: Account-Level
    tier3_max_daily_loss_pct: float = 0.03
    tier3_max_weekly_loss_pct: float = 0.10
    tier3_email_alerts: bool = True
    tier3_sms_alerts: bool = False

    # Tier 4: Session-Level
    tier4_end_of_day_utc: str = "21:00"
    tier4_trading_window_start: str = "00:00"
    tier4_trading_window_end: str = "23:59"
    tier4_max_trades_per_minute: int = 10

    # Tier 5: System-Level
    tier5_max_chaos_score: float = 0.75
    tier5_broker_timeout_seconds: int = 30

    # Alert Thresholds
    alert_threshold_yellow: int = 50
    alert_threshold_orange: int = 70
    alert_threshold_red: int = 85
    alert_threshold_black: int = 95


def load_progressive_config(config_path: str = "config/trading_system.yaml") -> ProgressiveConfig:
    """
    Load progressive kill switch configuration from YAML file.

    Args:
        config_path: Path to the configuration file

    Returns:
        ProgressiveConfig instance with loaded or default values
    """
    config = ProgressiveConfig()

    try:
        path = Path(config_path)
        if not path.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return config

        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        progressive = data.get('kill_switch', {}).get('progressive', {})
        if not progressive:
            logger.info("No progressive config found, using defaults")
            return config

        config.enabled = progressive.get('enabled', True)

        # Tier 2
        tier2 = progressive.get('tier2', {})
        config.tier2_max_failed_bots = tier2.get('max_failed_bots_per_family', 3)
        config.tier2_max_family_loss_pct = tier2.get('max_family_loss_pct', 0.20)

        # Tier 3
        tier3 = progressive.get('tier3', {})
        config.tier3_max_daily_loss_pct = tier3.get('max_daily_loss_pct', 0.03)
        config.tier3_max_weekly_loss_pct = tier3.get('max_weekly_loss_pct', 0.10)
        config.tier3_email_alerts = tier3.get('email_alerts', True)
        config.tier3_sms_alerts = tier3.get('sms_alerts', False)

        # Tier 4
        tier4 = progressive.get('tier4', {})
        config.tier4_end_of_day_utc = tier4.get('end_of_day_utc', '21:00')
        config.tier4_trading_window_start = tier4.get('trading_window_start', '00:00')
        config.tier4_trading_window_end = tier4.get('trading_window_end', '23:59')
        config.tier4_max_trades_per_minute = tier4.get('max_trades_per_minute', 10)

        # Tier 5
        tier5 = progressive.get('tier5', {})
        config.tier5_max_chaos_score = tier5.get('max_chaos_score', 0.75)
        config.tier5_broker_timeout_seconds = tier5.get('broker_timeout_seconds', 30)

        # Alert Thresholds
        thresholds = progressive.get('alert_thresholds', {})
        config.alert_threshold_yellow = thresholds.get('yellow', 50)
        config.alert_threshold_orange = thresholds.get('orange', 70)
        config.alert_threshold_red = thresholds.get('red', 85)
        config.alert_threshold_black = thresholds.get('black', 95)

        logger.info(f"Loaded progressive config from {config_path}")

    except Exception as e:
        logger.error(f"Failed to load progressive config: {e}, using defaults")

    return config


class ProgressiveKillSwitch:
    """
    Orchestrates all five tiers of protection with progressive alert levels.

    Tier Hierarchy:
    - Tier 5: System-Level (chaos, broker) - HIGHEST PRIORITY
    - Tier 4: Session-Level (news, EOD, rate limits)
    - Tier 3: Account-Level (daily/weekly limits)
    - Tier 2: Strategy-Level (family failures, losses)
    - Tier 1: Bot-Level (consecutive losses, daily limits) - LOWEST PRIORITY

    Alert Level Actions:
    - BLACK → Nuclear shutdown (close all positions, block all trading)
    - RED → Close all positions, block new trades
    - ORANGE → Halt new trades, keep existing positions
    - YELLOW → Warning only, increased monitoring
    - GREEN → Normal operation

    Usage:
        pks = ProgressiveKillSwitch()
        allowed, reason = pks.check_all_tiers(context)
    """

    def __init__(
        self,
        config: Optional[ProgressiveConfig] = None,
        kill_switch: Optional[SmartKillSwitch] = None,
        bot_circuit_breaker: Optional[BotCircuitBreakerManager] = None,
        sentinel: Optional["Sentinel"] = None,
        news_sensor: Optional["NewsSensor"] = None,
        alert_service: Optional["AlertService"] = None
    ):
        """
        Initialize ProgressiveKillSwitch.

        Args:
            config: Configuration instance
            kill_switch: SmartKillSwitch for executing kills
            bot_circuit_breaker: Tier 1 bot protection
            sentinel: Sentinel for chaos detection
            news_sensor: NewsSensor for news events
            alert_service: Optional AlertService for notifications
        """
        self.config = config or load_progressive_config()

        # Initialize alert manager with custom thresholds
        alert_thresholds = {
            AlertLevel.YELLOW: self.config.alert_threshold_yellow,
            AlertLevel.ORANGE: self.config.alert_threshold_orange,
            AlertLevel.RED: self.config.alert_threshold_red,
            AlertLevel.BLACK: self.config.alert_threshold_black
        }
        # Inject AlertService into AlertManager
        self.alert_manager = AlertManager(
            thresholds=alert_thresholds,
            alert_service=alert_service
        )

        # Tier 1: Bot-Level (reuse existing)
        self.bot_circuit_breaker = bot_circuit_breaker or BotCircuitBreakerManager()

        # Tier 2: Strategy-Level
        self.strategy_monitor = StrategyFamilyMonitor(
            alert_manager=self.alert_manager,
            max_failed_bots=self.config.tier2_max_failed_bots,
            max_family_loss_pct=self.config.tier2_max_family_loss_pct
        )

        # Tier 3: Account-Level
        self.account_monitor = AccountMonitor(
            alert_manager=self.alert_manager,
            max_daily_loss_pct=self.config.tier3_max_daily_loss_pct,
            max_weekly_loss_pct=self.config.tier3_max_weekly_loss_pct,
            email_alerts=self.config.tier3_email_alerts,
            sms_alerts=self.config.tier3_sms_alerts
        )

        # Tier 4: Session-Level
        self.session_monitor = SessionMonitor(
            alert_manager=self.alert_manager,
            news_sensor=news_sensor,
            sentinel=sentinel,
            end_of_day_utc=self._parse_time(self.config.tier4_end_of_day_utc),
            trading_window_start=self._parse_time(self.config.tier4_trading_window_start),
            trading_window_end=self._parse_time(self.config.tier4_trading_window_end),
            max_trades_per_minute=self.config.tier4_max_trades_per_minute
        )

        # Tier 5: System-Level
        self.system_monitor = SystemMonitor(
            alert_manager=self.alert_manager,
            sentinel=sentinel,
            max_chaos_score=self.config.tier5_max_chaos_score,
            broker_timeout_seconds=self.config.tier5_broker_timeout_seconds
        )

        # Smart Kill Switch for execution
        self.kill_switch = kill_switch or get_smart_kill_switch()

        # Set sentinel references
        if sentinel:
            self.session_monitor.set_sentinel(sentinel)
            self.system_monitor.set_sentinel(sentinel)

        self._last_check_result: Optional[Tuple[bool, Optional[str]]] = None

        # Account book mapping for prop firm emergency kills
        # Maps account_id -> AccountBook type
        self._account_book_map: Dict[str, AccountBook] = {}

        logger.info(
            f"ProgressiveKillSwitch initialized: "
            f"enabled={self.config.enabled}, "
            f"thresholds=({self.config.alert_threshold_yellow}/"
            f"{self.config.alert_threshold_orange}/"
            f"{self.config.alert_threshold_red}/"
            f"{self.config.alert_threshold_black})"
        )

    def _parse_time(self, time_str: str):
        """Parse time string to time object."""
        from datetime import time
        try:
            parts = time_str.split(':')
            return time(int(parts[0]), int(parts[1]))
        except (ValueError, IndexError):
            logger.warning(f"Invalid time format: {time_str}, using default")
            return time(0, 0)

    async def check_all_tiers(self, context: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Check all protection tiers in priority order (5→4→3→2→1)."""
        if not self.config.enabled:
            return True, None

        bot_id = context.get('bot_id')
        account_id = context.get('account_id')
        strategy_family = context.get('strategy_family')

        # Tier 5: System-Level
        allowed, reason = self.system_monitor.check_system_health()
        if not allowed:
            await self._execute_kill(5, reason, AlertLevel.BLACK)
            self._last_check_result = (False, reason)
            return False, reason

        # Tier 4: Session-Level
        allowed, reason = self.session_monitor.check_session_allowed()
        if not allowed:
            level = AlertLevel.RED if "End of trading day" in (reason or "") else AlertLevel.BLACK
            if "kill zone" in (reason or "").lower():
                level = AlertLevel.BLACK
            await self._execute_kill(4, reason, level)
            self._last_check_result = (False, reason)
            return False, reason

        # Tier 3: Account-Level
        if account_id:
            allowed = self.account_monitor.is_account_allowed(account_id)
            if not allowed:
                status = self.account_monitor.get_stop_status(account_id)
                reason = f"Account stop: daily={status['daily_stop']}, weekly={status['weekly_stop']}"
                await self._execute_kill(3, reason, AlertLevel.RED)
                self._last_check_result = (False, reason)
                return False, reason

        # Tier 2: Strategy-Level
        if strategy_family:
            try:
                family = StrategyType(strategy_family) if isinstance(strategy_family, str) else strategy_family
                if not self.strategy_monitor.is_family_allowed(family):
                    reason = f"Strategy family {family.value} quarantined"
                    await self._execute_kill(2, reason, AlertLevel.ORANGE)
                    self._last_check_result = (False, reason)
                    return False, reason
            except ValueError:
                pass

        # Tier 1: Bot-Level
        if bot_id:
            allowed, reason = self.bot_circuit_breaker.check_allowed(bot_id)
            if not allowed:
                self.alert_manager.raise_alert(tier=1, message=f"Bot {bot_id} blocked: {reason}",
                    threshold_pct=60.0, source="bot", metadata={"bot_id": bot_id, "reason": reason})
                self._last_check_result = (False, reason)
                return False, reason

        self._last_check_result = (True, None)
        return True, None

    async def _execute_kill(self, tier: int, reason: str, level: AlertLevel) -> None:
        """Execute appropriate kill action based on alert level."""
        import asyncio
        logger.warning(f"Kill: Tier {tier}, Level {level.value}, Reason: {reason}")

        if level == AlertLevel.BLACK:
            await self.kill_switch.trigger(reason=KillReason.SYSTEM_ERROR, triggered_by=f"progressive_tier{tier}",
                message=f"NUCLEAR: {reason}")
        elif level == AlertLevel.RED:
            await self.kill_switch.trigger(reason=KillReason.DRAWDOWN_LIMIT, triggered_by=f"progressive_tier{tier}",
                message=f"RED: {reason}")
        elif level == AlertLevel.ORANGE:
            try:
                loop = asyncio.get_event_loop()
                asyncio.create_task(self.kill_switch._send_halt_new_trades()) if loop.is_running() else \
                    loop.run_until_complete(self.kill_switch._send_halt_new_trades())
            except RuntimeError:
                asyncio.run(self.kill_switch._send_halt_new_trades())

    def record_bot_failure(self, bot_manifest: Any, reason: str = "") -> bool:
        return self.strategy_monitor.record_bot_failure(bot_manifest, reason)

    def record_trade_pnl(self, account_id: str, pnl: float, account_balance: Optional[float] = None) -> bool:
        self.session_monitor.record_trade()
        return self.account_monitor.record_trade_pnl(account_id, pnl, account_balance)

    def update_broker_ping(self) -> None:
        self.system_monitor.update_broker_ping()

    def get_status(self) -> Dict[str, Any]:
        return {"enabled": self.config.enabled, "current_alert_level": self.alert_manager.current_level.value,
            "active_alerts": len(self.alert_manager.active_alerts),
            "last_check_result": {"allowed": self._last_check_result[0] if self._last_check_result else None,
                "reason": self._last_check_result[1] if self._last_check_result else None},
            "tiers": {"tier1_bot": {"quarantined_count": len(self.bot_circuit_breaker.get_quarantined_bots())},
                "tier2_strategy": {"quarantined_families": self.strategy_monitor.get_quarantined_families()},
                "tier3_account": {"accounts_at_risk": len(self.account_monitor.get_accounts_at_risk())},
                "tier4_session": self.session_monitor.get_session_status(),
                "tier5_system": self.system_monitor.get_system_status()}}

    def get_all_alerts(self) -> Dict[str, Any]:
        return {"current_level": self.alert_manager.current_level.value,
            "active_alerts": [a.to_dict() for a in self.alert_manager.active_alerts],
            "history": self.alert_manager.get_history(limit=50)}

    def reset_tier(self, tier: int) -> bool:
        """Reset a specific tier's state (1-5). Returns True if successful."""
        if tier == 2:
            for family in StrategyType:
                self.strategy_monitor.reactivate_family(family)
            return True
        elif tier == 3:
            for account_id in self.account_monitor.account_states:
                self.account_monitor.reset_account_stops(account_id)
            return True
        elif tier == 4:
            self.session_monitor.reset_eod_flag()
            self.session_monitor.clear_rate_limit_history()
            return True
        elif tier == 5:
            return self.system_monitor.reset_shutdown()
        return False

    def reactivate_family(self, family: str) -> bool:
        """Reactivate a quarantined strategy family."""
        try:
            strategy_type = StrategyType(family)
            return self.strategy_monitor.reactivate_family(strategy_type)
        except ValueError:
            return False

    def register_account_book(self, account_id: str, book_type: AccountBook) -> None:
        """
        Register an account with its book type for filtering.

        Args:
            account_id: Account identifier
            book_type: AccountBook type (PERSONAL or PROP_FIRM)
        """
        self._account_book_map[account_id] = book_type
        logger.info(f"Registered account {account_id} to book type {book_type.value}")

    def get_accounts_by_book(self, book_type: AccountBook) -> List[str]:
        """
        Get all accounts filtered by book type.

        Args:
            book_type: AccountBook type to filter by

        Returns:
            List of account IDs matching the book type
        """
        return [
            account_id for account_id, book in self._account_book_map.items()
            if book == book_type
        ]

    async def emergency_kill_prop_firm(
        self,
        prop_firm_name: Optional[str] = None,
        reason: str = "Emergency prop firm kill triggered"
    ) -> Dict[str, Any]:
        """Emergency kill all prop firm accounts or specific firm."""
        logger.critical(f"EMERGENCY KILL: Prop firm={prop_firm_name or 'ALL'}, reason={reason}")

        prop_firm_accounts = self.get_accounts_by_book(AccountBook.PROP_FIRM)

        if prop_firm_name:
            logger.warning(f"Prop firm name filtering: {prop_firm_name}. Killing all prop firm accounts.")

        if not prop_firm_accounts:
            return {"success": True, "killed_count": 0, "accounts": [], "message": "No prop firm accounts"}

        killed_accounts = []
        for account_id in prop_firm_accounts:
            try:
                await self.kill_switch.trigger(
                    reason=KillReason.MANUAL,
                    triggered_by="emergency_prop_firm_kill",
                    message=f"Prop firm emergency kill: {reason}",
                    account_id=account_id
                )
                killed_accounts.append(account_id)
            except Exception as e:
                logger.error(f"Failed to kill {account_id}: {e}")

        self.alert_manager.raise_alert(
            tier=5,
            message=f"EMERGENCY KILL: {len(killed_accounts)} prop firm accounts",
            threshold_pct=100.0,
            source="emergency_kill",
            metadata={"prop_firm_name": prop_firm_name, "killed_accounts": killed_accounts, "reason": reason}
        )

        return {"success": True, "killed_count": len(killed_accounts), "accounts": killed_accounts,
                "prop_firm_name": prop_firm_name, "reason": reason}


# Global singleton instance
_global_progressive_kill_switch: Optional[ProgressiveKillSwitch] = None


def get_progressive_kill_switch(alert_service: Optional["AlertService"] = None) -> ProgressiveKillSwitch:
    """Get or create the global ProgressiveKillSwitch instance."""
    global _global_progressive_kill_switch
    if _global_progressive_kill_switch is None:
        _global_progressive_kill_switch = ProgressiveKillSwitch(alert_service=alert_service)
    elif alert_service is not None:
        _global_progressive_kill_switch.alert_manager.alert_service = alert_service
    return _global_progressive_kill_switch


def reset_progressive_kill_switch() -> None:
    """Reset the global ProgressiveKillSwitch instance (for testing)."""
    global _global_progressive_kill_switch
    _global_progressive_kill_switch = None

    from src.router.alert_manager import reset_alert_manager
    from src.router.strategy_monitor import reset_strategy_monitor
    from src.router.account_monitor import reset_account_monitor
    from src.router.session_monitor import reset_session_monitor
    from src.router.system_monitor import reset_system_monitor

    reset_alert_manager()
    reset_strategy_monitor()
    reset_account_monitor()
    reset_session_monitor()
    reset_system_monitor()

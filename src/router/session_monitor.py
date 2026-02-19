"""
Session Monitor for Progressive Kill Switch System

Implements Tier 4 protection by monitoring session-based controls
including news events, end-of-day, trading windows, and rate limits.

**Validates: Phase 4 - Tier 4 Session-Level Protection**
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, time, timedelta
from typing import Optional, Tuple, List, Dict, Any, TYPE_CHECKING

from src.router.alert_manager import AlertManager, AlertLevel, get_alert_manager
from src.router.sensors.news import NewsSensor

if TYPE_CHECKING:
    from src.router.sentinel import Sentinel

logger = logging.getLogger(__name__)


@dataclass
class SessionState:
    """
    Tracks session-related state.

    Attributes:
        trade_timestamps: List of recent trade timestamps for rate limiting
        last_eod_check: When we last checked end-of-day
        eod_positions_closed: Whether we closed positions at EOD today
    """
    trade_timestamps: List[datetime] = field(default_factory=list)
    last_eod_check: Optional[datetime] = None
    eod_positions_closed: bool = False


class SessionMonitor:
    """
    Monitors session-based controls for trading restrictions.

    Tier 4 Protection:
    - Checks for news kill zones via NewsSensor
    - Enforces end-of-day closure (default 21:00 UTC)
    - Enforces trading window hours
    - Implements rate limiting to prevent excessive trading

    Configuration:
    - END_OF_DAY_UTC = time(21, 0)
    - TRADING_WINDOW_START = time(0, 0)
    - TRADING_WINDOW_END = time(23, 59)
    - MAX_TRADES_PER_MINUTE = 10

    Usage:
        monitor = SessionMonitor(alert_manager, news_sensor)
        allowed, reason = monitor.check_session_allowed()
        monitor.record_trade()
    """

    # Default configuration (can be overridden via config)
    END_OF_DAY_UTC = time(21, 0)
    TRADING_WINDOW_START = time(0, 0)
    TRADING_WINDOW_END = time(23, 59)
    MAX_TRADES_PER_MINUTE = 10

    def __init__(
        self,
        alert_manager: Optional[AlertManager] = None,
        news_sensor: Optional[NewsSensor] = None,
        sentinel: Optional["Sentinel"] = None,
        end_of_day_utc: Optional[time] = None,
        trading_window_start: Optional[time] = None,
        trading_window_end: Optional[time] = None,
        max_trades_per_minute: int = MAX_TRADES_PER_MINUTE
    ):
        """
        Initialize SessionMonitor.

        Args:
            alert_manager: AlertManager for raising alerts
            news_sensor: NewsSensor for news event detection
            sentinel: Sentinel for regime detection
            end_of_day_utc: Time to close all positions (default 21:00 UTC)
            trading_window_start: Start of allowed trading hours
            trading_window_end: End of allowed trading hours
            max_trades_per_minute: Rate limit for trades
        """
        self.alert_manager = alert_manager or get_alert_manager()
        self.news_sensor = news_sensor
        self.sentinel = sentinel

        self.end_of_day_utc = end_of_day_utc or self.END_OF_DAY_UTC
        self.trading_window_start = trading_window_start or self.TRADING_WINDOW_START
        self.trading_window_end = trading_window_end or self.TRADING_WINDOW_END
        self.max_trades_per_minute = max_trades_per_minute

        self.state = SessionState()

        logger.info(
            f"SessionMonitor initialized: "
            f"EOD={self.end_of_day_utc}, "
            f"window={self.trading_window_start}-{self.trading_window_end}, "
            f"rate_limit={max_trades_per_minute}/min"
        )

    @property
    def current_time_utc(self) -> datetime:
        """Get current UTC time."""
        return datetime.now(timezone.utc)

    def check_session_allowed(self) -> Tuple[bool, Optional[str]]:
        """
        Check all session conditions.

        Returns:
            Tuple of (allowed, reason) where:
            - allowed: True if trading is permitted
            - reason: None if allowed, otherwise explanation
        """
        now = self.current_time_utc
        now_time = now.time()

        # 1. Check news events (highest priority)
        news_state = self._check_news_state()
        if news_state == "KILL_ZONE":
            self.alert_manager.raise_alert(
                tier=4,
                message="Trading blocked: News KILL ZONE active",
                threshold_pct=100.0,
                source="session",
                metadata={"news_state": news_state}
            )
            return False, "News event kill zone active"

        # 2. Check end of day
        if self._is_end_of_day(now_time):
            # Trigger EOD closure if not done today
            if not self.state.eod_positions_closed:
                # Comment 2: Raise RED/BLACK alert for mandatory close-out
                self.alert_manager.raise_alert(
                    tier=4,
                    message=f"End of day reached: {self.end_of_day_utc} - MANDATORY CLOSE-OUT",
                    threshold_pct=90.0,  # RED level
                    source="session",
                    metadata={
                        "eod_time": str(self.end_of_day_utc),
                        "action_required": "CLOSE_ALL_POSITIONS"
                    }
                )
                self.state.eod_positions_closed = True
            return False, f"End of trading day ({self.end_of_day_utc}) - Positions must be closed"

        # 3. Check trading window
        if not self._is_in_trading_window(now_time):
            self.alert_manager.raise_alert(
                tier=4,
                message=f"Outside trading window ({self.trading_window_start}-{self.trading_window_end})",
                threshold_pct=50.0,
                source="session",
                metadata={"current_time": str(now_time)}
            )
            return False, f"Outside trading window"

        # 4. Check rate limit
        if not self._check_rate_limit():
            self.alert_manager.raise_alert(
                tier=4,
                message=f"Rate limit exceeded: {self.max_trades_per_minute} trades/minute",
                threshold_pct=90.0,
                source="session",
                metadata={"trades_last_minute": len(self._get_recent_trades())}
            )
            return False, "Trade rate limit exceeded"

        # Reset EOD flag at midnight
        if now_time < time(0, 5) and self.state.eod_positions_closed:
            self.state.eod_positions_closed = False

        return True, None

    def _check_news_state(self) -> str:
        """Check current news state via NewsSensor."""
        if self.news_sensor:
            return self.news_sensor.check_state()
        return "SAFE"

    def _is_end_of_day(self, current_time: time) -> bool:
        """Check if we've reached end of trading day."""
        return current_time >= self.end_of_day_utc

    def _is_in_trading_window(self, current_time: time) -> bool:
        """Check if current time is within trading window."""
        if self.trading_window_start <= self.trading_window_end:
            # Normal case: window doesn't cross midnight
            return self.trading_window_start <= current_time <= self.trading_window_end
        else:
            # Window crosses midnight (e.g., 22:00 - 02:00)
            return current_time >= self.trading_window_start or current_time <= self.trading_window_end

    def _check_rate_limit(self) -> bool:
        """Check if rate limit has been exceeded."""
        recent_trades = self._get_recent_trades()
        return len(recent_trades) < self.max_trades_per_minute

    def _get_recent_trades(self, window_seconds: int = 60) -> List[datetime]:
        """Get trades within the last N seconds."""
        now = self.current_time_utc
        cutoff = now - timedelta(seconds=window_seconds)

        # Clean old timestamps
        self.state.trade_timestamps = [
            ts for ts in self.state.trade_timestamps
            if ts > cutoff
        ]

        return self.state.trade_timestamps

    def record_trade(self) -> None:
        """Record a trade timestamp for rate limiting."""
        self.state.trade_timestamps.append(self.current_time_utc)

    def get_upcoming_news_events(self, hours_ahead: int = 24) -> List[Dict[str, Any]]:
        """Get upcoming news events."""
        if not self.news_sensor:
            return []

        events = self.news_sensor.get_upcoming_events(hours_ahead)
        return [
            {
                "title": e.title,
                "impact": e.impact,
                "time": e.time.isoformat(),
                "currency": e.currency
            }
            for e in events
        ]

    def get_session_status(self) -> Dict[str, Any]:
        """Get current session status."""
        now = self.current_time_utc
        now_time = now.time()

        news_state = self._check_news_state()
        recent_trades = len(self._get_recent_trades())

        return {
            "current_time_utc": now.isoformat(),
            "in_trading_window": self._is_in_trading_window(now_time),
            "end_of_day_reached": self._is_end_of_day(now_time),
            "news_state": news_state,
            "rate_limit": {
                "trades_last_minute": recent_trades,
                "max_per_minute": self.max_trades_per_minute,
                "exceeded": recent_trades >= self.max_trades_per_minute
            },
            "eod_time": str(self.end_of_day_utc),
            "trading_window": {
                "start": str(self.trading_window_start),
                "end": str(self.trading_window_end)
            },
            "eod_positions_closed": self.state.eod_positions_closed
        }

    def set_news_sensor(self, news_sensor: NewsSensor) -> None:
        """Set the news sensor (for lazy loading)."""
        self.news_sensor = news_sensor
        logger.info("NewsSensor attached to SessionMonitor")

    def set_sentinel(self, sentinel: "Sentinel") -> None:
        """Set the sentinel (for regime detection)."""
        self.sentinel = sentinel
        logger.info("Sentinel attached to SessionMonitor")

    def reset_eod_flag(self) -> None:
        """Reset end-of-day flag (for testing or manual override)."""
        self.state.eod_positions_closed = False
        logger.info("EOD flag reset")

    def clear_rate_limit_history(self) -> None:
        """Clear rate limit history (for testing)."""
        self.state.trade_timestamps.clear()
        logger.info("Rate limit history cleared")


# Global singleton instance
_global_session_monitor: Optional[SessionMonitor] = None


def get_session_monitor() -> SessionMonitor:
    """Get or create the global SessionMonitor instance."""
    global _global_session_monitor
    if _global_session_monitor is None:
        _global_session_monitor = SessionMonitor()
    return _global_session_monitor


def reset_session_monitor() -> None:
    """Reset the global SessionMonitor instance (for testing)."""
    global _global_session_monitor
    _global_session_monitor = None

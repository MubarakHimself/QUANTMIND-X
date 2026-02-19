"""
System Monitor for Progressive Kill Switch System

Implements Tier 5 protection by monitoring system-wide safeguards
including chaos score, broker connectivity, and nuclear shutdown.

**Validates: Phase 5 - Tier 5 System-Level Protection**

V3: Integrated with Prometheus monitoring for:
- Chaos score gauge updates
- MT5 connection status tracking
- System shutdown event tracking
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple, Dict, Any, TYPE_CHECKING

from src.router.alert_manager import AlertManager, AlertLevel, get_alert_manager

if TYPE_CHECKING:
    from src.router.sentinel import Sentinel

logger = logging.getLogger(__name__)


@dataclass
class SystemState:
    """
    Tracks system-wide state.

    Attributes:
        last_broker_ping: Last successful broker communication
        system_shutdown: Whether nuclear shutdown is active
        shutdown_reason: Why shutdown was triggered
        shutdown_time: When shutdown occurred
        health_check_count: Number of health checks performed
        last_chaos_score: Most recent chaos score
    """
    last_broker_ping: Optional[datetime] = None
    system_shutdown: bool = False
    shutdown_reason: Optional[str] = None
    shutdown_time: Optional[datetime] = None
    health_check_count: int = 0
    last_chaos_score: float = 0.0


class SystemMonitor:
    """
    Monitors system-wide safeguards and triggers nuclear shutdown.

    Tier 5 Protection:
    - Monitors global chaos score from Sentinel
    - Tracks broker connection via heartbeat pings
    - Triggers nuclear shutdown on critical conditions
    - Notifies administrators of system emergencies

    Thresholds:
    - MAX_CHAOS_SCORE = 0.75 (75%)
    - BROKER_TIMEOUT_SECONDS = 30

    Usage:
        monitor = SystemMonitor(alert_manager, sentinel)
        allowed, reason = monitor.check_system_health()
        monitor.update_broker_ping()
    """

    # Default thresholds (can be overridden via config)
    MAX_CHAOS_SCORE = 0.75
    BROKER_TIMEOUT_SECONDS = 30

    def __init__(
        self,
        alert_manager: Optional[AlertManager] = None,
        sentinel: Optional["Sentinel"] = None,
        max_chaos_score: float = MAX_CHAOS_SCORE,
        broker_timeout_seconds: int = BROKER_TIMEOUT_SECONDS
    ):
        """
        Initialize SystemMonitor.

        Args:
            alert_manager: AlertManager for raising alerts
            sentinel: Sentinel for chaos score detection
            max_chaos_score: Maximum allowed chaos score (default 0.75)
            broker_timeout_seconds: Seconds before broker timeout (default 30)
        """
        self.alert_manager = alert_manager or get_alert_manager()
        self.sentinel = sentinel
        self.max_chaos_score = max_chaos_score
        self.broker_timeout_seconds = broker_timeout_seconds

        self.state = SystemState()

        logger.info(
            f"SystemMonitor initialized: "
            f"max_chaos={max_chaos_score}, "
            f"broker_timeout={broker_timeout_seconds}s"
        )

    def check_system_health(self) -> Tuple[bool, Optional[str]]:
        """
        Check all system health conditions.

        Returns:
            Tuple of (allowed, reason) where:
            - allowed: True if system is healthy
            - reason: None if healthy, otherwise explanation
        """
        self.state.health_check_count += 1

        # If already shutdown, stay shutdown
        if self.state.system_shutdown:
            return False, f"System shutdown: {self.state.shutdown_reason}"

        # 1. Check chaos score (highest priority for market conditions)
        chaos_exceeded, chaos_reason = self._check_chaos_score()
        if chaos_exceeded:
            self._trigger_nuclear_shutdown(chaos_reason)
            return False, chaos_reason

        # 2. Check broker connectivity
        broker_ok, broker_reason = self._check_broker_connection()
        if not broker_ok:
            self._trigger_nuclear_shutdown(broker_reason)
            return False, broker_reason

        return True, None

    def _check_chaos_score(self) -> Tuple[bool, Optional[str]]:
        """Check if chaos score exceeds threshold."""
        if not self.sentinel:
            # No sentinel available - allow trading
            return False, None

        try:
            report = self.sentinel.current_report
            if report is None:
                return False, None

            chaos_score = report.chaos_score
            self.state.last_chaos_score = chaos_score
            
            # Update chaos score gauge in Prometheus
            self._update_chaos_score_metric(chaos_score)

            if chaos_score > self.max_chaos_score:
                # Calculate threshold percentage
                threshold_pct = (chaos_score / self.max_chaos_score) * 100

                # Raise alert based on severity
                if chaos_score >= 0.9:
                    self.alert_manager.raise_alert(
                        tier=5,
                        message=f"CRITICAL: Chaos score at {chaos_score:.2f}",
                        threshold_pct=min(threshold_pct, 100.0),
                        source="system",
                        metadata={"chaos_score": chaos_score, "regime": report.regime}
                    )
                else:
                    self.alert_manager.raise_alert(
                        tier=5,
                        message=f"Elevated chaos score: {chaos_score:.2f}",
                        threshold_pct=threshold_pct,
                        source="system",
                        metadata={"chaos_score": chaos_score, "regime": report.regime}
                    )

                return True, f"Chaos score too high: {chaos_score:.2f} > {self.max_chaos_score}"

            # Warning for elevated but not critical chaos
            if chaos_score > self.max_chaos_score * 0.7:
                threshold_pct = (chaos_score / self.max_chaos_score) * 100
                self.alert_manager.raise_alert(
                    tier=5,
                    message=f"Warning: Chaos score elevated at {chaos_score:.2f}",
                    threshold_pct=threshold_pct,
                    source="system",
                    metadata={"chaos_score": chaos_score, "regime": report.regime}
                )

        except Exception as e:
            logger.error(f"Error checking chaos score: {e}")

        return False, None

    def _check_broker_connection(self) -> Tuple[bool, Optional[str]]:
        """Check if broker connection is alive."""
        if self.state.last_broker_ping is None:
            # No ping recorded yet - allow trading (will be caught on first ping)
            return True, None

        time_since_ping = datetime.now(timezone.utc) - self.state.last_broker_ping
        seconds_elapsed = time_since_ping.total_seconds()

        if seconds_elapsed > self.broker_timeout_seconds:
            return False, f"Broker connection lost: {seconds_elapsed:.0f}s since last ping"

        return True, None

    def _trigger_nuclear_shutdown(self, reason: str) -> None:
        """Trigger nuclear shutdown."""
        self.state.system_shutdown = True
        self.state.shutdown_reason = reason
        self.state.shutdown_time = datetime.now(timezone.utc)

        # Raise BLACK alert
        self.alert_manager.raise_alert(
            tier=5,
            message=f"NUCLEAR SHUTDOWN: {reason}",
            threshold_pct=100.0,
            source="system",
            metadata={
                "reason": reason,
                "shutdown_time": self.state.shutdown_time.isoformat(),
                "last_chaos_score": self.state.last_chaos_score
            }
        )

        # Notify administrator
        self._notify_admin(reason)

        logger.critical(
            f"🚨🚨🚨 NUCLEAR SHUTDOWN TRIGGERED 🚨🚨🚨"
            f"\n   Reason: {reason}"
            f"\n   Time: {self.state.shutdown_time.isoformat()}"
        )

    def _notify_admin(self, reason: str) -> None:
        """Send administrator notification."""
        # Log for external notification system
        logger.critical(
            f"[ADMIN NOTIFY] Nuclear shutdown triggered: {reason}"
        )

        # Track shutdown in Prometheus
        self._track_shutdown_metric(reason)

        # In production, this would send:
        # - Email to admin list
        # - SMS to on-call
        # - PagerDuty/OpsGenie alert

    def update_broker_ping(self) -> None:
        """Update broker heartbeat timestamp."""
        self.state.last_broker_ping = datetime.now(timezone.utc)
        logger.debug(f"Broker ping updated: {self.state.last_broker_ping.isoformat()}")
        
        # Update MT5 connection status metric
        self._update_mt5_status_metric(True)

    def is_shutdown(self) -> bool:
        """Check if system is in shutdown state."""
        return self.state.system_shutdown

    def get_shutdown_info(self) -> Optional[Dict[str, Any]]:
        """Get shutdown information if shutdown is active."""
        if not self.state.system_shutdown:
            return None

        return {
            "reason": self.state.shutdown_reason,
            "time": self.state.shutdown_time.isoformat() if self.state.shutdown_time else None,
            "duration_seconds": (
                (datetime.now(timezone.utc) - self.state.shutdown_time).total_seconds()
                if self.state.shutdown_time else 0
            )
        }

    def reset_shutdown(self) -> bool:
        """
        Reset shutdown state after manual review.

        Returns:
            True if reset, False if not in shutdown
        """
        if not self.state.system_shutdown:
            return False

        old_reason = self.state.shutdown_reason
        self.state.system_shutdown = False
        self.state.shutdown_reason = None
        self.state.shutdown_time = None

        # Clear related alerts
        self.alert_manager.clear_alerts_by_source("system")

        logger.warning(
            f"⚠️ System shutdown RESET (was: {old_reason}) - "
            f"Manual review required before resuming trading"
        )

        return True

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        broker_ok, broker_reason = self._check_broker_connection() if self.state.last_broker_ping else (True, None)

        return {
            "is_shutdown": self.state.system_shutdown,
            "shutdown_reason": self.state.shutdown_reason,
            "shutdown_time": self.state.shutdown_time.isoformat() if self.state.shutdown_time else None,
            "last_chaos_score": self.state.last_chaos_score,
            "max_chaos_score": self.max_chaos_score,
            "broker_connection": {
                "connected": broker_ok,
                "last_ping": self.state.last_broker_ping.isoformat() if self.state.last_broker_ping else None,
                "timeout_seconds": self.broker_timeout_seconds
            },
            "health_check_count": self.state.health_check_count
        }

    def set_sentinel(self, sentinel: "Sentinel") -> None:
        """Set the sentinel (for lazy loading)."""
        self.sentinel = sentinel
        logger.info("Sentinel attached to SystemMonitor")
    
    # ========== Prometheus Metrics Helpers ==========
    
    def _update_chaos_score_metric(self, score: float) -> None:
        """Update chaos score gauge in Prometheus."""
        try:
            from src.monitoring import update_chaos_score
            update_chaos_score(score)
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Could not update chaos score metric: {e}")
    
    def _track_shutdown_metric(self, reason: str) -> None:
        """Track shutdown event in Prometheus."""
        try:
            from src.monitoring import track_shutdown
            track_shutdown(reason)
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Could not track shutdown metric: {e}")
    
    def _update_mt5_status_metric(self, connected: bool) -> None:
        """Update MT5 connection status gauge in Prometheus."""
        try:
            from src.monitoring import update_mt5_status
            update_mt5_status(connected)
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Could not update MT5 status metric: {e}")


# Global singleton instance
_global_system_monitor: Optional[SystemMonitor] = None


def get_system_monitor() -> SystemMonitor:
    """Get or create the global SystemMonitor instance."""
    global _global_system_monitor
    if _global_system_monitor is None:
        _global_system_monitor = SystemMonitor()
    return _global_system_monitor


def reset_system_monitor() -> None:
    """Reset the global SystemMonitor instance (for testing)."""
    global _global_system_monitor
    _global_system_monitor = None

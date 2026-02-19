"""
Alert Manager for Progressive Kill Switch System

Implements progressive alert levels (GREEN→YELLOW→ORANGE→RED→BLACK)
that map threshold percentages to severity levels.

**Validates: Phase 1 - Alert Level System**
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, List, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from mcp_mt5.alert_service import AlertService, AlertSeverity, AlertCategory

logger = logging.getLogger(__name__)

# Comment 4: DB Session for database access
_DB_SESSION = None


def _get_db_session():
    """Get or create database session for DB access."""
    global _DB_SESSION
    if _DB_SESSION is None:
        try:
            from src.database.engine import SessionLocal
            _DB_SESSION = SessionLocal()
        except Exception as e:
            logger.warning(f"Could not load DB session: {e}")
    return _DB_SESSION


class AlertLevel(Enum):
    """
    Progressive alert levels from safe to critical.

    Threshold ranges:
    - GREEN: 0-50% (normal operation)
    - YELLOW: 50-70% (warning)
    - ORANGE: 70-85% (elevated risk)
    - RED: 85-95% (critical)
    - BLACK: >95% (emergency shutdown)
    """
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    ORANGE = "ORANGE"
    RED = "RED"
    BLACK = "BLACK"

    @property
    def color(self) -> str:
        """Get the color code for display."""
        colors = {
            AlertLevel.GREEN: "#22C55E",
            AlertLevel.YELLOW: "#EAB308",
            AlertLevel.ORANGE: "#F97316",
            AlertLevel.RED: "#EF4444",
            AlertLevel.BLACK: "#000000"
        }
        return colors[self]

    @property
    def priority(self) -> int:
        """Get priority for sorting (higher = more urgent)."""
        priorities = {
            AlertLevel.GREEN: 0,
            AlertLevel.YELLOW: 1,
            AlertLevel.ORANGE: 2,
            AlertLevel.RED: 3,
            AlertLevel.BLACK: 4
        }
        return priorities[self]


@dataclass
class Alert:
    """
    Represents a single alert event.

    Attributes:
        level: Alert severity level
        tier: Which protection tier triggered (1-5)
        message: Human-readable description
        threshold_pct: Current threshold percentage (0-100)
        triggered_at: When the alert was raised
        source: Component that raised the alert
        metadata: Additional context
    """
    level: AlertLevel
    tier: int  # 1=Bot, 2=Strategy, 3=Account, 4=Session, 5=System
    message: str
    threshold_pct: float
    triggered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize alert to dictionary."""
        return {
            "level": self.level.value,
            "tier": self.tier,
            "message": self.message,
            "threshold_pct": self.threshold_pct,
            "triggered_at": self.triggered_at.isoformat(),
            "source": self.source,
            "metadata": self.metadata
        }


class AlertManager:
    """
    Manages progressive alert levels for the kill switch system.

    Features:
    - Maps threshold percentages to alert levels
    - Tracks active alerts and history
    - Integrates with existing AlertService for notifications
    - Maintains current system-wide alert level

    Usage:
        manager = AlertManager()
        level = manager.calculate_level(75.0)  # Returns ORANGE
        manager.raise_alert(tier=2, message="3 bots failed", threshold_pct=100.0, source="strategy")
    """

    # Default threshold boundaries (can be overridden via config)
    DEFAULT_THRESHOLDS = {
        AlertLevel.GREEN: 0,
        AlertLevel.YELLOW: 50,
        AlertLevel.ORANGE: 70,
        AlertLevel.RED: 85,
        AlertLevel.BLACK: 95
    }

    def __init__(
        self,
        thresholds: Optional[Dict[AlertLevel, float]] = None,
        alert_service: Optional["AlertService"] = None
    ):
        """
        Initialize AlertManager.

        Args:
            thresholds: Custom threshold boundaries (overrides defaults)
            alert_service: Optional AlertService for notifications
        """
        self.thresholds = thresholds or self.DEFAULT_THRESHOLDS.copy()
        self.alert_service = alert_service
        self.active_alerts: List[Alert] = []
        self.alert_history: List[Alert] = []
        self._current_level = AlertLevel.GREEN
        self._max_history_size = 1000

        logger.info(f"AlertManager initialized with thresholds: {[(k.value, v) for k, v in self.thresholds.items()]}")

    def calculate_level(self, threshold_pct: float) -> AlertLevel:
        """
        Determine alert level from threshold percentage.

        Args:
            threshold_pct: Current threshold as percentage (0-100)

        Returns:
            Corresponding AlertLevel enum value
        """
        if threshold_pct >= self.thresholds[AlertLevel.BLACK]:
            return AlertLevel.BLACK
        elif threshold_pct >= self.thresholds[AlertLevel.RED]:
            return AlertLevel.RED
        elif threshold_pct >= self.thresholds[AlertLevel.ORANGE]:
            return AlertLevel.ORANGE
        elif threshold_pct >= self.thresholds[AlertLevel.YELLOW]:
            return AlertLevel.YELLOW
        else:
            return AlertLevel.GREEN

    def raise_alert(
        self,
        tier: int,
        message: str,
        threshold_pct: float,
        source: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Alert:
        """
        Create and record a new alert.

        Args:
            tier: Protection tier (1-5)
            message: Alert description
            threshold_pct: Current threshold percentage
            source: Component that raised the alert
            metadata: Additional context

        Returns:
            The created Alert object
        """
        level = self.calculate_level(threshold_pct)

        alert = Alert(
            level=level,
            tier=tier,
            message=message,
            threshold_pct=threshold_pct,
            source=source,
            metadata=metadata or {}
        )

        # Add to active alerts
        self.active_alerts.append(alert)

        # Add to history
        self.alert_history.append(alert)
        if len(self.alert_history) > self._max_history_size:
            self.alert_history = self.alert_history[-self._max_history_size:]

        # Update system level
        self._update_system_level()

        # Log the alert
        log_msg = f"[Tier {tier}] {level.value}: {message} ({threshold_pct:.1f}%)"
        if level == AlertLevel.BLACK:
            logger.critical(f"🚨🚨🚨 {log_msg}")
        elif level == AlertLevel.RED:
            logger.critical(f"🚨 {log_msg}")
        elif level == AlertLevel.ORANGE:
            logger.warning(f"⚠️ {log_msg}")
        elif level == AlertLevel.YELLOW:
            logger.warning(f"⚡ {log_msg}")
        else:
            logger.info(f"✓ {log_msg}")

        # Comment 4: Save alert to DB for persistence
        self._save_alert_to_db(alert)

        # Send notification via AlertService if available
        self._send_notification(alert)

        return alert
    
    def _save_alert_to_db(self, alert: Alert) -> None:
        """
        Save alert to database for persistence.
        
        Comment 4: Write alerts to DB instead of in-memory only.
        """
        try:
            session = _get_db_session()
            if session is None:
                return
                
            from src.database.models import AlertHistory
            db_record = AlertHistory(
                level=alert.level.value,
                tier=alert.tier,
                message=alert.message,
                threshold_pct=alert.threshold_pct,
                triggered_at=alert.triggered_at,
                source=alert.source,
                metadata=alert.metadata,
                is_active=True
            )
            session.add(db_record)
            session.commit()
        except Exception as e:
            logger.debug(f"Could not save alert to DB: {e}")

    def _update_system_level(self) -> None:
        """Update current system level based on highest active alert."""
        if not self.active_alerts:
            self._current_level = AlertLevel.GREEN
            return

        # Find highest priority alert
        max_level = AlertLevel.GREEN
        for alert in self.active_alerts:
            if alert.level.priority > max_level.priority:
                max_level = alert.level

        if max_level != self._current_level:
            old_level = self._current_level
            self._current_level = max_level
            logger.info(f"System alert level changed: {old_level.value} → {max_level.value}")

    def _send_notification(self, alert: Alert) -> None:
        """Send notification via AlertService if configured."""
        if not self.alert_service:
            return

        try:
            # Map alert level to severity
            severity_map = {
                AlertLevel.GREEN: "INFO",
                AlertLevel.YELLOW: "WARNING",
                AlertLevel.ORANGE: "WARNING",
                AlertLevel.RED: "ERROR",
                AlertLevel.BLACK: "CRITICAL"
            }

            # Only send notifications for RED and BLACK
            if alert.level in [AlertLevel.RED, AlertLevel.BLACK]:
                self.alert_service.send_alert_sync(
                    title=f"[Tier {alert.tier}] Kill Switch Alert: {alert.level.value}",
                    message=alert.message,
                    severity=severity_map[alert.level],
                    category="RISK",
                    metadata={
                        "tier": alert.tier,
                        "threshold_pct": alert.threshold_pct,
                        "source": alert.source,
                        **alert.metadata
                    }
                )
        except Exception as e:
            logger.error(f"Failed to send alert notification: {e}")

    def clear_alert(self, alert: Alert) -> None:
        """Clear an active alert."""
        if alert in self.active_alerts:
            self.active_alerts.remove(alert)
            self._update_system_level()
            logger.info(f"Alert cleared: {alert.level.value} from {alert.source}")

    def clear_alerts_by_tier(self, tier: int) -> int:
        """Clear all alerts from a specific tier."""
        count = 0
        for alert in self.active_alerts[:]:  # Copy to avoid modification during iteration
            if alert.tier == tier:
                self.active_alerts.remove(alert)
                count += 1
        self._update_system_level()
        if count > 0:
            logger.info(f"Cleared {count} alerts from Tier {tier}")
        return count

    def clear_alerts_by_source(self, source: str) -> int:
        """Clear all alerts from a specific source."""
        count = 0
        for alert in self.active_alerts[:]:
            if alert.source == source:
                self.active_alerts.remove(alert)
                count += 1
        self._update_system_level()
        if count > 0:
            logger.info(f"Cleared {count} alerts from source: {source}")
        return count

    @property
    def current_level(self) -> AlertLevel:
        """Get current system-wide alert level."""
        return self._current_level

    def get_status(self) -> Dict[str, Any]:
        """Get current alert manager status."""
        return {
            "current_level": self._current_level.value,
            "active_alerts_count": len(self.active_alerts),
            "active_alerts": [a.to_dict() for a in self.active_alerts],
            "thresholds": {k.value: v for k, v in self.thresholds.items()}
        }

    def get_history(
        self,
        tier: Optional[int] = None,
        level: Optional[AlertLevel] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get alert history with optional filtering.

        Args:
            tier: Filter by tier (optional)
            level: Filter by level (optional)
            limit: Maximum number of alerts to return

        Returns:
            List of alert dictionaries
        """
        alerts = self.alert_history

        if tier is not None:
            alerts = [a for a in alerts if a.tier == tier]

        if level is not None:
            alerts = [a for a in alerts if a.level == level]

        # Sort by most recent first
        alerts = sorted(alerts, key=lambda a: a.triggered_at, reverse=True)

        return [a.to_dict() for a in alerts[:limit]]


# Global singleton instance
_global_alert_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    """Get or create the global AlertManager instance."""
    global _global_alert_manager
    if _global_alert_manager is None:
        _global_alert_manager = AlertManager()
    return _global_alert_manager


def reset_alert_manager() -> None:
    """Reset the global AlertManager instance (for testing)."""
    global _global_alert_manager
    _global_alert_manager = None

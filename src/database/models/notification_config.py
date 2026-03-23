"""
Notification configuration models.

Stores user preferences for notification events across categories:
trade, strategy, risk, system, agent.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Column, String, DateTime, Boolean, Text
from ..models.base import Base

logger = logging.getLogger(__name__)


class NotificationConfig(Base):
    """
    Notification configuration for storing user preferences.

    Stores which notification events the user wants to receive.
    Always-on events (kill_switch, loss_cap, system_critical) cannot be disabled.

    Attributes:
        id: Primary key (UUID)
        event_type: Event type identifier (e.g., 'trade_executed', 'kill_switch_triggered')
        category: Category (trade, strategy, risk, system, agent)
        is_enabled: Whether this notification is enabled
        is_always_on: Whether this event cannot be disabled
        description: Human-readable description of the event
        created_at_utc: Creation timestamp in UTC
        updated_at: Last update timestamp
    """
    __tablename__ = 'notification_configs'

    id = Column(String(36), primary_key=True)
    event_type = Column(String(100), nullable=False, unique=True, index=True)
    category = Column(String(50), nullable=False, index=True)
    is_enabled = Column(Boolean, default=True, nullable=False)
    is_always_on = Column(Boolean, default=False, nullable=False)
    description = Column(String(500), nullable=True)
    created_at_utc = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)

    # Default notification events
    DEFAULT_EVENTS = [
        # Trade events
        {"event_type": "trade_executed", "category": "trade", "is_always_on": False, "description": "Trade executed"},
        {"event_type": "trade_closed", "category": "trade", "is_always_on": False, "description": "Trade closed"},
        {"event_type": "trade_error", "category": "trade", "is_always_on": False, "description": "Trade execution error"},

        # Strategy events
        {"event_type": "strategy_paused", "category": "strategy", "is_always_on": False, "description": "Strategy paused"},
        {"event_type": "strategy_resumed", "category": "strategy", "is_always_on": False, "description": "Strategy resumed"},
        {"event_type": "strategy_error", "category": "strategy", "is_always_on": False, "description": "Strategy error"},

        # Risk events
        {"event_type": "loss_cap_triggered", "category": "risk", "is_always_on": False, "description": "Loss cap reached"},
        {"event_type": "drawdown_alert", "category": "risk", "is_always_on": False, "description": "Drawdown threshold alert"},
        {"event_type": "regime_changed", "category": "risk", "is_always_on": False, "description": "Market regime changed"},

        # System events - always on
        {"event_type": "kill_switch_triggered", "category": "system", "is_always_on": True, "description": "Kill switch triggered"},
        {"event_type": "loss_cap_triggered_system", "category": "system", "is_always_on": True, "description": "System loss cap triggered"},
        {"event_type": "system_critical", "category": "system", "is_always_on": True, "description": "System critical error"},
        {"event_type": "server_health_alert", "category": "system", "is_always_on": False, "description": "Server health threshold breach"},

        # Agent events
        {"event_type": "agent_task_complete", "category": "agent", "is_always_on": False, "description": "Agent task completed"},
        {"event_type": "agent_task_failed", "category": "agent", "is_always_on": False, "description": "Agent task failed"},
        {"event_type": "department_mail", "category": "agent", "is_always_on": False, "description": "Department mail received"},
    ]

    def to_dict(self) -> dict:
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "event_type": self.event_type,
            "category": self.category,
            "is_enabled": self.is_enabled,
            "is_always_on": self.is_always_on,
            "description": self.description,
            "created_at_utc": self.created_at_utc.isoformat() if self.created_at_utc else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    def __repr__(self):
        return f"<NotificationConfig(event_type={self.event_type}, category={self.category}, is_enabled={self.is_enabled})>"


class LogRetentionPolicy(Base):
    """
    Log retention policy configuration.

    This table stores the configuration for log retention periods and
    cold storage sync settings.

    Attributes:
        id: Primary key (always 'default' - single row)
        hot_retention_days: Days to keep logs in hot storage (default: 90)
        cold_retention_days: Days to keep logs in cold storage (default: 1095 = 3 years)
        cold_storage_path: Path to cold storage on Contabo
        sync_enabled: Whether automatic cold storage sync is enabled
        sync_cron: Cron expression for sync schedule (default: "0 2 * * *" = 2 AM daily)
        last_sync_at: Timestamp of last successful sync
        last_sync_status: Status of last sync (success/failed)
        checksum_algorithm: Algorithm for integrity verification (default: sha256)
    """
    __tablename__ = 'log_retention_policy'

    id = Column(String(20), primary_key=True, default="default")
    hot_retention_days = Column(String(10), default="90", nullable=False)
    cold_retention_days = Column(String(10), default="1095", nullable=False)  # 3 years
    cold_storage_path = Column(String(500), nullable=True)
    sync_enabled = Column(Boolean, default=True, nullable=False)
    sync_cron = Column(String(50), default="0 2 * * *", nullable=False)  # 2 AM daily
    last_sync_at = Column(DateTime, nullable=True)
    last_sync_status = Column(String(20), nullable=True)
    checksum_algorithm = Column(String(20), default="sha256", nullable=False)

    def to_dict(self) -> dict:
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "hot_retention_days": int(self.hot_retention_days),
            "cold_retention_days": int(self.cold_retention_days),
            "cold_storage_path": self.cold_storage_path,
            "sync_enabled": self.sync_enabled,
            "sync_cron": self.sync_cron,
            "last_sync_at": self.last_sync_at.isoformat() if self.last_sync_at else None,
            "last_sync_status": self.last_sync_status,
            "checksum_algorithm": self.checksum_algorithm,
        }

    def __repr__(self):
        return f"<LogRetentionPolicy(hot={self.hot_retention_days}d, cold={self.cold_retention_days}d)>"

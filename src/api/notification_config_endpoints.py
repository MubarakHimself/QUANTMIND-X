"""
Notification Configuration API Endpoints

Manages user notification preferences including enabling/disabling events
and retrieving event types by category. Also manages log retention policy
and cold storage sync configuration.

AC1: GET /api/notifications/config - returns all configurable event types
AC2: PUT /api/notifications/config/{event_type} - toggles event notifications
AC3: High-priority events (kill_switch, loss_cap) are always-on
AC4: Cold storage sync with integrity verification
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from sqlalchemy.orm import Session

from src.database.models import NotificationConfig
from src.database.models.notification_config import LogRetentionPolicy
from src.database.models.base import get_db_session
from src.database.engine import get_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/notifications", tags=["notifications"])


def _get_notification_dir() -> Path:
    """Resolve the writable notification log directory."""
    configured = os.getenv("QUANTMIND_DATA_DIR")
    if configured:
        return Path(configured) / "notifications"

    container_data_dir = Path("/app/data")
    if container_data_dir.exists():
        return container_data_dir / "notifications"

    return Path(__file__).resolve().parents[2] / "data" / "notifications"


# =============================================================================
# Notification Sending Function
# =============================================================================

async def send_notification(subject: str, body: str, event_type: str = "system_notification") -> bool:
    """
    Send a notification message.

    This is a simple implementation that logs the notification and can be
    enhanced later with actual email/telegram/webhook integration.

    Args:
        subject: Notification subject line
        body: Notification body text
        event_type: Type of notification event

    Returns:
        True if notification was logged successfully
    """
    logger.info(f"Notification sent: {subject}")

    # Log to console/file for now (can be enhanced with email/telegram later)
    try:
        import json

        notif_dir = _get_notification_dir()
        notif_dir.mkdir(parents=True, exist_ok=True)

        notif_file = notif_dir / f"notif_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}.json"
        notif_file.write_text(json.dumps({
            "subject": subject,
            "body": body,
            "event_type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }, indent=2))

        logger.info(f"Notification logged to {notif_file}")
        return True
    except Exception as e:
        logger.error(f"Failed to log notification: {e}")
        return False


# =============================================================================
# Helper Functions
# =============================================================================

def get_event_severity(event_type: str, is_always_on: bool) -> str:
    """Determine event severity based on event type and always-on status."""
    if is_always_on:
        return "urgent"
    # Map specific event types to severity
    high_priority = {"kill_switch", "loss_cap", "system_critical", "mt5_disconnect"}
    if any(x in event_type.lower() for x in high_priority):
        return "high"
    return "normal"


def get_event_delivery_channel(event_type: str, category: str) -> str:
    """Determine default delivery channel based on event type and category."""
    # Category-based defaults
    channel_map = {
        "trade": "telegram",
        "strategy": "telegram",
        "risk": "telegram",
        "system": "telegram",
        "agent": "log",
    }
    # Event-specific overrides
    event_channel_map = {
        "daily_summary": "email",
        "backtest_complete": "log",
        "hypothesis_generated": "log",
        "skill_registered": "log",
        "department_task_complete": "log",
    }
    if event_type in event_channel_map:
        return event_channel_map[event_type]
    return channel_map.get(category, "telegram")


# Pydantic models
class NotificationEvent(BaseModel):
    """Response model for notification events - matches AC1 specification."""
    event_type: str
    category: str
    severity: str  # Added to match AC1: low, normal, high, urgent
    enabled: bool  # Renamed from is_enabled to match AC1
    delivery_channel: str  # Added to match AC1: telegram, email, webhook, log, console
    is_always_on: bool  # Keep for AC3: always-on events cannot be disabled
    description: Optional[str] = None


class NotificationUpdate(BaseModel):
    event_type: str
    is_enabled: bool


class NotificationSettingsResponse(BaseModel):
    events: List[NotificationEvent]
    categories: List[str]


@router.get("", response_model=NotificationSettingsResponse)
async def get_notification_settings():
    """
    Get all notification settings including event types, categories,
    and enabled/disabled status.
    """
    session = get_session()
    try:
        # Get all notification configs
        configs = session.query(NotificationConfig).all()

        # If no configs exist, initialize with defaults
        if not configs:
            for event_def in NotificationConfig.DEFAULT_EVENTS:
                config = NotificationConfig(
                    id=str(uuid.uuid4()),
                    event_type=event_def["event_type"],
                    category=event_def["category"],
                    is_enabled=True,
                    is_always_on=event_def["is_always_on"],
                    description=event_def.get("description", ""),
                    created_at_utc=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc)
                )
                session.add(config)
            session.commit()
            configs = session.query(NotificationConfig).all()

        # Build response with AC1 specified fields: event_type, category, severity, enabled, delivery_channel
        events = [
            NotificationEvent(
                event_type=c.event_type,
                category=c.category,
                severity=get_event_severity(c.event_type, c.is_always_on),
                enabled=c.is_enabled,
                delivery_channel=get_event_delivery_channel(c.event_type, c.category),
                is_always_on=c.is_always_on,
                description=c.description
            )
            for c in configs
        ]

        # Get unique categories
        categories = sorted(list(set(c.category for c in configs)))

        return NotificationSettingsResponse(events=events, categories=categories)

    except Exception as e:
        logger.error(f"Error fetching notification settings: {e}")
        session.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()


@router.put("")
async def update_notification_settings(update: NotificationUpdate):
    """
    Update a notification event's enabled status.
    Always-on events cannot be disabled.
    """
    session = get_session()
    try:
        config = session.query(NotificationConfig).filter(
            NotificationConfig.event_type == update.event_type
        ).first()

        if not config:
            raise HTTPException(status_code=404, detail=f"Event type '{update.event_type}' not found")

        # Check if this is an always-on event
        if config.is_always_on:
            raise HTTPException(
                status_code=400,
                detail=f"Event '{update.event_type}' is always-on and cannot be disabled"
            )

        config.is_enabled = update.is_enabled
        config.updated_at = datetime.now(timezone.utc)
        session.commit()

        return {
            "event_type": config.event_type,
            "is_enabled": config.is_enabled,
            "category": config.category,
            "is_always_on": config.is_always_on
        }

    except HTTPException:
        session.rollback()
        raise
    except Exception as e:
        logger.error(f"Error updating notification setting: {e}")
        session.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()


@router.get("/categories/{category}", response_model=List[NotificationEvent])
async def get_notifications_by_category(category: str):
    """Get all notification events for a specific category."""
    session = get_session()
    try:
        configs = session.query(NotificationConfig).filter(
            NotificationConfig.category == category
        ).all()

        return [
            NotificationEvent(
                event_type=c.event_type,
                category=c.category,
                severity=get_event_severity(c.event_type, c.is_always_on),
                enabled=c.is_enabled,
                delivery_channel=get_event_delivery_channel(c.event_type, c.category),
                is_always_on=c.is_always_on,
                description=c.description
            )
            for c in configs
        ]

    except Exception as e:
        logger.error(f"Error fetching notifications for category {category}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()


@router.post("/reset")
async def reset_notification_settings():
    """
    Reset all notification settings to defaults.
    Always-on events remain always-on.
    """
    session = get_session()
    try:
        # Clear existing configs
        session.query(NotificationConfig).delete()

        # Re-insert defaults
        for event_def in NotificationConfig.DEFAULT_EVENTS:
            config = NotificationConfig(
                id=str(uuid.uuid4()),
                event_type=event_def["event_type"],
                category=event_def["category"],
                is_enabled=True,
                is_always_on=event_def["is_always_on"],
                description=event_def.get("description", ""),
                created_at_utc=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc)
            )
            session.add(config)

        session.commit()
        return {"message": "Notification settings reset to defaults"}

    except Exception as e:
        logger.error(f"Error resetting notification settings: {e}")
        session.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()


# =============================================================================
# Log Retention Policy Endpoints (AC4)
# =============================================================================

class LogRetentionPolicyResponse(BaseModel):
    hot_retention_days: int
    cold_retention_days: int
    cold_storage_path: Optional[str]
    sync_enabled: bool
    sync_cron: str
    last_sync_at: Optional[str]
    last_sync_status: Optional[str]
    checksum_algorithm: str


class LogRetentionPolicyUpdate(BaseModel):
    hot_retention_days: Optional[int] = None
    cold_retention_days: Optional[int] = None
    cold_storage_path: Optional[str] = None
    sync_enabled: Optional[bool] = None
    sync_cron: Optional[str] = None
    checksum_algorithm: Optional[str] = None


@router.get("/retention", response_model=LogRetentionPolicyResponse)
async def get_log_retention_policy():
    """
    Get the log retention policy configuration.
    AC4: Returns retention periods and cold storage settings.
    """
    session = get_session()
    try:
        policy = session.query(LogRetentionPolicy).filter(
            LogRetentionPolicy.id == "default"
        ).first()

        if not policy:
            # Create default policy
            policy = LogRetentionPolicy(
                id="default",
                hot_retention_days="90",
                cold_retention_days="1095",
                cold_storage_path="/mnt/cold-storage/logs",
                sync_enabled=True,
                sync_cron="0 2 * * *",
                checksum_algorithm="sha256"
            )
            session.add(policy)
            session.commit()
            session.refresh(policy)

        return LogRetentionPolicyResponse(
            hot_retention_days=int(policy.hot_retention_days),
            cold_retention_days=int(policy.cold_retention_days),
            cold_storage_path=policy.cold_storage_path,
            sync_enabled=policy.sync_enabled,
            sync_cron=policy.sync_cron,
            last_sync_at=policy.last_sync_at.isoformat() if policy.last_sync_at else None,
            last_sync_status=policy.last_sync_status,
            checksum_algorithm=policy.checksum_algorithm
        )

    except Exception as e:
        logger.error(f"Error fetching log retention policy: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()


@router.put("/retention", response_model=LogRetentionPolicyResponse)
async def update_log_retention_policy(update: LogRetentionPolicyUpdate):
    """
    Update the log retention policy configuration.
    AC4: Configure retention periods and sync settings.
    """
    session = get_session()
    try:
        policy = session.query(LogRetentionPolicy).filter(
            LogRetentionPolicy.id == "default"
        ).first()

        if not policy:
            policy = LogRetentionPolicy(id="default")
            session.add(policy)

        # Update fields if provided
        if update.hot_retention_days is not None:
            policy.hot_retention_days = str(update.hot_retention_days)
        if update.cold_retention_days is not None:
            policy.cold_retention_days = str(update.cold_retention_days)
        if update.cold_storage_path is not None:
            policy.cold_storage_path = update.cold_storage_path
        if update.sync_enabled is not None:
            policy.sync_enabled = update.sync_enabled
        if update.sync_cron is not None:
            policy.sync_cron = update.sync_cron
        if update.checksum_algorithm is not None:
            policy.checksum_algorithm = update.checksum_algorithm

        session.commit()
        session.refresh(policy)

        return LogRetentionPolicyResponse(
            hot_retention_days=int(policy.hot_retention_days),
            cold_retention_days=int(policy.cold_retention_days),
            cold_storage_path=policy.cold_storage_path,
            sync_enabled=policy.sync_enabled,
            sync_cron=policy.sync_cron,
            last_sync_at=policy.last_sync_at.isoformat() if policy.last_sync_at else None,
            last_sync_status=policy.last_sync_status,
            checksum_algorithm=policy.checksum_algorithm
        )

    except Exception as e:
        logger.error(f"Error updating log retention policy: {e}")
        session.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()


@router.post("/retention/sync")
async def trigger_log_sync():
    """
    Trigger a manual log sync to cold storage.
    AC4: Nightly log sync with integrity verification (checksums).
    """
    try:
        from src.monitoring.cold_storage_sync import run_cold_storage_sync

        result = run_cold_storage_sync()

        # Update last sync status in database
        session = get_session()
        try:
            policy = session.query(LogRetentionPolicy).filter(
                LogRetentionPolicy.id == "default"
            ).first()

            if policy:
                policy.last_sync_at = datetime.now(timezone.utc)
                policy.last_sync_status = "success" if result.get("success") else "failed"
                session.commit()
        except Exception:
            pass
        finally:
            session.close()

        return result

    except Exception as e:
        logger.error(f"Error triggering log sync: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/retention/status")
async def get_cold_storage_status():
    """
    Get cold storage sync status and statistics.
    AC4: Returns sync status and pending logs count.
    """
    try:
        from src.monitoring.cold_storage_sync import get_cold_storage_status

        status = get_cold_storage_status()
        return status

    except Exception as e:
        logger.error(f"Error getting cold storage status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

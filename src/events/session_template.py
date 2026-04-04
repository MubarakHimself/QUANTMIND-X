"""
Session Template Event Models.

Story 16.2: Session Template Class — Configurable 10-Window Canonical Cycle

Contains:
- SessionTemplateEvent: Event published when session template configuration changes

Per NFR-M2: SessionTemplate is a data/configuration class — NO LLM calls.
Per NFR-D1: All event transitions logged with timestamps for audit.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional, List

from pydantic import BaseModel, Field


class SessionTemplateEventType(str, Enum):
    """Types of session template events."""
    CONFIG_CHANGED = "config_changed"
    WINDOW_UPDATED = "window_updated"
    RELOADED = "reloaded"


class SessionTemplateEvent(BaseModel):
    """
    Event published when session template configuration changes.

    Published to Redis 'session:template' channel on config changes
    for UI subscription and downstream system coordination.

    Attributes:
        event_type: Type of session template event
        template_name: Name of the affected template
        window_name: Name of affected window (if applicable)
        timestamp_utc: When the event occurred
        metadata: Additional event context
    """
    event_type: SessionTemplateEventType = Field(description="Type of session template event")
    template_name: str = Field(description="Name of the affected template")
    window_name: Optional[str] = Field(default=None, description="Affected window name (if applicable)")
    timestamp_utc: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Event timestamp")
    metadata: dict = Field(default_factory=dict, description="Additional event context")

    def model_dump(self, **kwargs):
        """Override to ensure timestamp is serialized properly."""
        data = super().model_dump(**kwargs)
        if isinstance(data.get("timestamp_utc"), datetime):
            data["timestamp_utc"] = self.timestamp_utc.isoformat()
        return data

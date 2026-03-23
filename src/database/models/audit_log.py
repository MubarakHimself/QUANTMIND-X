"""
Audit Log Models.

Immutable audit log for 5-layer system: trade events, strategy lifecycle,
risk param changes, agent actions, system health.

FR59: All system events logged at appropriate level — no silent exclusions
FR60: NL audit trail query
NFR-D2: Audit log entries are immutable once written — no deletion, no modification
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Optional, List

from sqlalchemy import Column, String, DateTime, Text, Index, JSON, event, DDL
from ..models.base import Base

# Audit layer enum values
class AuditLayer:
    """Audit layer constants for 5-layer system."""
    TRADE = "trade"
    STRATEGY_LIFECYCLE = "strategy_lifecycle"
    RISK_PARAM = "risk_param"
    AGENT_ACTION = "agent_action"
    SYSTEM_HEALTH = "system_health"

    ALL_LAYERS = [
        TRADE,
        STRATEGY_LIFECYCLE,
        RISK_PARAM,
        AGENT_ACTION,
        SYSTEM_HEALTH
    ]


class AuditLogEntry(Base):
    """
    Immutable audit log entry for 5-layer audit system.

    Stores events from all 5 layers: trade, strategy lifecycle, risk param,
    agent action, system health. Immutable - no updates or deletes allowed.

    Attributes:
        id: Primary key (UUID)
        layer: Audit layer (trade, strategy_lifecycle, risk_param, agent_action, system_health)
        event_type: Specific event type within the layer
        entity_type: Type of entity affected (ea, strategy, risk_params, agent, server, etc.)
        entity_id: ID of the entity affected
        action: Description of the action performed
        actor: Who/what caused this event (user, agent, system, etc.)
        reason: Reason/causal explanation for the event
        timestamp_utc: When the event occurred in UTC
        payload_json: Additional event data (JSON)
        metadata_json: Additional metadata (JSON)
        created_at_utc: Record creation timestamp (same as timestamp_utc for immutable logs)
    """
    __tablename__ = 'audit_log'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    layer = Column(String(30), nullable=False, index=True)  # trade, strategy_lifecycle, risk_param, agent_action, system_health
    event_type = Column(String(50), nullable=False, index=True)  # execution, pause, parameter_change, task_dispatch, health_breach, etc.
    entity_type = Column(String(50), nullable=True, index=True)  # ea, strategy, risk_params, agent, server, etc.
    entity_id = Column(String(100), nullable=True, index=True)  # EA_X, strategy_name, agent_id, etc.
    action = Column(Text, nullable=False)  # "EA_X paused", "risk_params updated", etc.
    actor = Column(String(100), nullable=True, index=True)  # "Commander", "RiskDepartment", "system", "user_mubarak"
    reason = Column(Text, nullable=True)  # Causal explanation for the event
    timestamp_utc = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False, index=True)
    payload_json = Column(JSON, nullable=True)  # Event-specific data
    metadata_json = Column(JSON, nullable=True)  # Additional context
    created_at_utc = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    __table_args__ = (
        Index('idx_audit_log_layer_timestamp', 'layer', 'timestamp_utc'),
        Index('idx_audit_log_entity', 'entity_type', 'entity_id'),
        Index('idx_audit_log_actor_timestamp', 'actor', 'timestamp_utc'),
        Index('idx_audit_log_layer_event', 'layer', 'event_type'),
    )

    def __repr__(self):
        return f"<AuditLogEntry(layer={self.layer}, event_type={self.event_type}, entity_id={self.entity_id})>"

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "layer": self.layer,
            "event_type": self.event_type,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "action": self.action,
            "actor": self.actor,
            "reason": self.reason,
            "timestamp_utc": self.timestamp_utc.isoformat() if self.timestamp_utc else None,
            "payload_json": self.payload_json,
            "metadata_json": self.metadata_json,
            "created_at_utc": self.created_at_utc.isoformat() if self.created_at_utc else None
        }


# SQLAlchemy event listeners to create SQLite triggers for immutability
# These triggers prevent DELETE and UPDATE operations on audit_log table

immutability_trigger_delete = DDL(
    "CREATE TRIGGER IF NOT EXISTS audit_log_immutable_delete "
    "BEFORE DELETE ON audit_log "
    "BEGIN SELECT RAISE(ABORT, 'immutable'); END"
)

immutability_trigger_update = DDL(
    "CREATE TRIGGER IF NOT EXISTS audit_log_immutable_update "
    "BEFORE UPDATE ON audit_log "
    "BEGIN SELECT RAISE(ABORT, 'immutable'); END"
)

event.listen(
    AuditLogEntry.__table__,
    "after_create",
    immutability_trigger_delete.execute_if(dialect="sqlite")
)

event.listen(
    AuditLogEntry.__table__,
    "after_create",
    immutability_trigger_update.execute_if(dialect="sqlite")
)


class AuditQueryResult:
    """
    Query result structure for NL audit queries.

    Represents a causal chain of events returned from audit queries.
    """
    def __init__(self, entries: List[AuditLogEntry], query: str, total_count: int):
        self.entries = entries
        self.query = query
        self.total_count = total_count

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "query": self.query,
            "results": [entry.to_dict() for entry in self.entries],
            "total_count": self.total_count,
            "causal_chain": self._build_causal_chain()
        }

    def _build_causal_chain(self) -> List[dict]:
        """Build chronological causal chain from entries."""
        # Sort by timestamp
        sorted_entries = sorted(self.entries, key=lambda e: e.timestamp_utc)
        return [
            {
                "timestamp_utc": entry.timestamp_utc.isoformat() if entry.timestamp_utc else None,
                "layer": entry.layer,
                "event_type": entry.event_type,
                "actor": entry.actor,
                "reason": entry.reason
            }
            for entry in sorted_entries
        ]


# Trade event types
class TradeEventType:
    """Trade event type constants."""
    EXECUTION = "execution"
    CLOSE = "close"
    MODIFY = "modify"
    CANCEL = "cancel"
    PARTIAL_CLOSE = "partial_close"


# Strategy lifecycle event types
class StrategyLifecycleEventType:
    """Strategy lifecycle event type constants."""
    START = "start"
    PAUSE = "pause"
    RESUME = "resume"
    STOP = "stop"
    REGIME_CHANGE = "regime_change"


# Risk param event types
class RiskParamEventType:
    """Risk param event type constants."""
    PARAMETER_CHANGE = "parameter_change"
    DAILY_LOSS_CAP_CHANGE = "daily_loss_cap_change"
    KELLY_FRACTION_CHANGE = "kelly_fraction_change"
    POSITION_MULTIPLIER_CHANGE = "position_multiplier_change"


# Agent action event types
class AgentActionEventType:
    """Agent action event type constants."""
    TASK_DISPATCH = "task_dispatch"
    TASK_COMPLETE = "task_complete"
    TASK_FAILED = "task_failed"
    OPINION_GENERATED = "opinion_generated"
    DEPARTMENT_MAIL = "department_mail"


# System health event types
class SystemHealthEventType:
    """System health event type constants."""
    SERVER_START = "server_start"
    SERVER_STOP = "server_stop"
    HEALTH_BREACH = "health_breach"
    ERROR = "error"
    DEGRADED = "degraded"
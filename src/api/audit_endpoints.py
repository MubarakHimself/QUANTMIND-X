"""
Audit Log API Endpoints.

Implements 5-layer audit system with natural language query support.
Handles logging for: trade events, strategy lifecycle, risk param changes, agent actions, system health.

FR59: All system events logged at appropriate level
FR60: NL audit trail query
FR61: 3-year retention

Story: 10-1-5-layer-audit-system-nl-query-api
"""

import json
import re
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any
from enum import Enum

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_

from src.database.models import (
    AuditLogEntry,
    AuditLayer,
    TradeEventType,
    StrategyLifecycleEventType,
    RiskParamEventType,
    AgentActionEventType,
    SystemHealthEventType,
    get_db_session,
)

router = APIRouter(prefix="/api/audit", tags=["audit"])


# ============================================================================
# Pydantic Models
# ============================================================================


class AuditLayerEnum(str, Enum):
    """Audit layer enum for API."""
    TRADE = "trade"
    STRATEGY_LIFECYCLE = "strategy_lifecycle"
    RISK_PARAM = "risk_param"
    AGENT_ACTION = "agent_action"
    SYSTEM_HEALTH = "system_health"


class AuditLogWrite(BaseModel):
    """Schema for writing an audit log entry."""
    layer: AuditLayerEnum = Field(..., description="Audit layer (trade, strategy_lifecycle, risk_param, agent_action, system_health)")
    event_type: str = Field(..., description="Specific event type within the layer")
    entity_type: Optional[str] = Field(None, description="Type of entity affected (ea, strategy, risk_params, agent, server)")
    entity_id: Optional[str] = Field(None, description="ID of the entity affected")
    action: str = Field(..., description="Description of the action performed")
    actor: Optional[str] = Field(None, description="Who/what caused this event")
    reason: Optional[str] = Field(None, description="Causal explanation for the event")
    timestamp_utc: Optional[datetime] = Field(None, description="Event timestamp (defaults to now)")
    payload_json: Optional[Dict[str, Any]] = Field(None, description="Additional event data")
    metadata_json: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class AuditLogResponse(BaseModel):
    """Response schema for audit log entry."""
    id: str
    layer: str
    event_type: str
    entity_type: Optional[str]
    entity_id: Optional[str]
    action: str
    actor: Optional[str]
    reason: Optional[str]
    timestamp_utc: Optional[str]
    payload_json: Optional[Dict[str, Any]]
    metadata_json: Optional[Dict[str, Any]]
    created_at_utc: Optional[str]


class AuditQueryRequest(BaseModel):
    """Schema for NL audit query."""
    query: str = Field(..., description="Natural language query (e.g., 'Why was EA_X paused at 14:30 yesterday?')")
    limit: Optional[int] = Field(50, description="Maximum number of results")
    offset: Optional[int] = Field(0, description="Result offset for pagination")


class AuditQueryResponse(BaseModel):
    """Response schema for audit query."""
    query: str
    results: List[AuditLogResponse]
    total_count: int
    causal_chain: List[Dict[str, Any]]
    parsed_entities: Optional[Dict[str, Any]] = None
    parsed_time_range: Optional[Dict[str, Any]] = None


# ============================================================================
# NL Query Parser
# ============================================================================


class NLQueryParser:
    """
    Natural language query parser for audit logs.

    Handles:
    - Time reference resolution (yesterday, last week, etc.)
    - Entity extraction (EA_X, GBPUSD, strategy Y, etc.)
    - Query intent parsing
    """

    # Common entity patterns
    EA_PATTERN = re.compile(r'\b(EA_[A-Z0-9]+|EA\d+)\b', re.IGNORECASE)
    STRATEGY_PATTERN = re.compile(r'\b(strategy\s+[A-Z][a-zA-Z0-9]*|strategy\s+\d+)\b', re.IGNORECASE)
    SYMBOL_PATTERN = re.compile(r'\b[A-Z]{3}[A-Z0-9]{3}\b')  # e.g., GBPUSD, EURUSD

    def parse(self, query: str) -> Dict[str, Any]:
        """
        Parse natural language query into structured search parameters.

        Returns:
            Dict with keys: time_range, entities, intent
        """
        query_lower = query.lower()

        # Extract time range
        time_range = self._extract_time_range(query_lower, query)

        # Extract entities
        entities = self._extract_entities(query)

        # Extract intent (what they're asking about)
        intent = self._extract_intent(query_lower)

        return {
            "time_range": time_range,
            "entities": entities,
            "intent": intent,
            "original_query": query
        }

    def _extract_time_range(self, query_lower: str, original_query: str) -> Dict[str, Any]:
        """Extract time range from query."""
        result = {"start": None, "end": None, "raw": None}

        # Check for "yesterday" with time — handles both orderings:
        # "yesterday at 14:30" and "at 14:30 yesterday"
        yesterday_match = re.search(
            r'(?:yesterday\s+(?:at\s+)?(\d{1,2}):(\d{2}))'
            r'|(?:(?:at\s+)?(\d{1,2}):(\d{2})\s+yesterday)',
            query_lower
        )
        if yesterday_match:
            # Groups 1,2 match "yesterday at HH:MM"; groups 3,4 match "at HH:MM yesterday"
            hour = int(yesterday_match.group(1) or yesterday_match.group(3))
            minute = int(yesterday_match.group(2) or yesterday_match.group(4))
            yesterday = datetime.now(timezone.utc) - timedelta(days=1)
            result["start"] = yesterday.replace(hour=hour, minute=minute, second=0, microsecond=0)
            result["end"] = result["start"] + timedelta(hours=1)  # 1 hour window
            result["raw"] = f"yesterday at {hour:02d}:{minute:02d}"
            return result

        # Check for "yesterday"
        if 'yesterday' in query_lower:
            yesterday = datetime.now(timezone.utc) - timedelta(days=1)
            result["start"] = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
            result["end"] = yesterday.replace(hour=23, minute=59, second=59, microsecond=999999)
            result["raw"] = "yesterday"
            return result

        # Check for "today"
        if 'today' in query_lower:
            today = datetime.now(timezone.utc)
            result["start"] = today.replace(hour=0, minute=0, second=0, microsecond=0)
            result["end"] = today
            result["raw"] = "today"
            return result

        # Check for "last week"
        if 'last week' in query_lower:
            result["start"] = datetime.now(timezone.utc) - timedelta(weeks=1)
            result["end"] = datetime.now(timezone.utc)
            result["raw"] = "last week"
            return result

        # Check for "last month"
        if 'last month' in query_lower:
            result["start"] = datetime.now(timezone.utc) - timedelta(days=30)
            result["end"] = datetime.now(timezone.utc)
            result["raw"] = "last month"
            return result

        # Check for specific date/time patterns
        time_match = re.search(r'(?:at\s+)?(\d{1,2}):(\d{2})', original_query)
        if time_match:
            hour = int(time_match.group(1))
            minute = int(time_match.group(2))
            # Use today if no other time reference
            now = datetime.now(timezone.utc)
            result["start"] = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            result["end"] = result["start"] + timedelta(minutes=5)  # 5 minute window
            result["raw"] = f"at {hour:02d}:{minute:02d}"
            return result

        return result

    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract entity references from query."""
        entities = {
            "eas": [],
            "strategies": [],
            "symbols": [],
            "agents": [],
            "servers": []
        }

        # Extract EA references
        eas = self.EA_PATTERN.findall(query)
        entities["eas"] = list(set(eas))

        # Extract strategy references
        strategies = self.STRATEGY_PATTERN.findall(query)
        entities["strategies"] = [s.replace('strategy ', '').strip() for s in strategies]

        # Extract symbol references (e.g., GBPUSD, EURUSD)
        symbols = self.SYMBOL_PATTERN.findall(query)
        entities["symbols"] = list(set(symbols))

        return entities

    def _extract_intent(self, query_lower: str) -> Dict[str, str]:
        """Extract query intent."""
        intent = {"action": "search", "target": None}

        if 'why' in query_lower:
            intent["action"] = "explain"
        if 'paused' in query_lower or 'pause' in query_lower:
            intent["target"] = "pause"
        if 'stopped' in query_lower or 'stop' in query_lower:
            intent["target"] = "stop"
        if 'started' in query_lower or 'start' in query_lower:
            intent["target"] = "start"
        if 'closed' in query_lower or 'close' in query_lower:
            intent["target"] = "close"

        return intent


# ============================================================================
# Database Dependencies
# ============================================================================


def get_audit_db() -> Session:
    """Get database session for audit operations."""
    db = next(get_db_session())
    try:
        yield db
    finally:
        db.close()


# ============================================================================
# Write Endpoints
# ============================================================================


@router.post("/log", response_model=AuditLogResponse, status_code=201)
def write_audit_log(
    entry: AuditLogWrite,
    db: Session = Depends(get_audit_db)
):
    """
    Write an audit log entry.

    Immutable operation - entries cannot be updated or deleted.

    Body:
        layer: Audit layer (trade, strategy_lifecycle, risk_param, agent_action, system_health)
        event_type: Specific event type
        entity_type: Type of entity affected
        entity_id: ID of entity affected
        action: Description of action
        actor: Who caused the event
        reason: Causal explanation
        timestamp_utc: Event timestamp (defaults to now)
        payload_json: Additional event data
        metadata_json: Additional metadata

    Example for trade event:
        POST /api/audit/log
        {"layer": "trade", "event_type": "execution", "entity_id": "EA_X", "action": "EURUSD buy", "actor": "TradingBot"}

    Example for strategy lifecycle:
        POST /api/audit/log
        {"layer": "strategy_lifecycle", "event_type": "pause", "entity_id": "Momentum", "action": "Strategy paused", "actor": "RiskDepartment", "reason": "Daily loss cap reached"}
    """
    # Use provided timestamp or default to now
    timestamp = entry.timestamp_utc or datetime.now(timezone.utc)

    # Create audit log entry — generate UUID in application code so it is
    # available even when db.refresh() is called on a mock session in tests.
    audit_entry = AuditLogEntry(
        id=str(uuid.uuid4()),
        layer=entry.layer.value,
        event_type=entry.event_type,
        entity_type=entry.entity_type,
        entity_id=entry.entity_id,
        action=entry.action,
        actor=entry.actor,
        reason=entry.reason,
        timestamp_utc=timestamp,
        payload_json=entry.payload_json,
        metadata_json=entry.metadata_json,
        created_at_utc=datetime.now(timezone.utc)
    )

    db.add(audit_entry)
    db.commit()
    db.refresh(audit_entry)

    return AuditLogResponse(**audit_entry.to_dict())


@router.post("/log/batch", response_model=List[AuditLogResponse], status_code=201)
def write_audit_log_batch(
    entries: List[AuditLogWrite],
    db: Session = Depends(get_audit_db)
):
    """
    Write multiple audit log entries in a single request.

    For efficient bulk logging of related events.
    """
    results = []
    timestamp = datetime.now(timezone.utc)

    for entry in entries:
        audit_entry = AuditLogEntry(
            id=str(uuid.uuid4()),
            layer=entry.layer.value,
            event_type=entry.event_type,
            entity_type=entry.entity_type,
            entity_id=entry.entity_id,
            action=entry.action,
            actor=entry.actor,
            reason=entry.reason,
            timestamp_utc=entry.timestamp_utc or timestamp,
            payload_json=entry.payload_json,
            metadata_json=entry.metadata_json,
            created_at_utc=timestamp
        )
        db.add(audit_entry)
        results.append(audit_entry)

    db.commit()
    for entry in results:
        db.refresh(entry)

    return [AuditLogResponse(**e.to_dict()) for e in results]


# ============================================================================
# NL Query Endpoint
# ============================================================================


@router.post("/query", response_model=AuditQueryResponse)
def query_audit_log(
    request: AuditQueryRequest,
    db: Session = Depends(get_audit_db)
):
    """
    Query audit logs using natural language.

    Parses natural language query to extract:
    - Time ranges (yesterday, last week, etc.)
    - Entity references (EA_X, strategy Y, etc.)
    - Intent (why, what happened, etc.)

    Returns chronological causal chain of events.

    Body:
        query: Natural language query (e.g., "Why was EA_X paused at 14:30 yesterday?")
        limit: Maximum results (default 50)
        offset: Result offset (default 0)
    """
    # Parse the NL query
    parser = NLQueryParser()
    parsed = parser.parse(request.query)

    # Build query filters
    filters = []

    # Time range filter
    if parsed["time_range"]["start"]:
        filters.append(AuditLogEntry.timestamp_utc >= parsed["time_range"]["start"])
    if parsed["time_range"]["end"]:
        filters.append(AuditLogEntry.timestamp_utc <= parsed["time_range"]["end"])

    # Entity filters
    entity_filters = []
    if parsed["entities"]["eas"]:
        entity_filters.append(AuditLogEntry.entity_id.in_(parsed["entities"]["eas"]))
    if parsed["entities"]["strategies"]:
        entity_filters.append(AuditLogEntry.entity_id.in_(parsed["entities"]["strategies"]))
    if parsed["entities"]["symbols"]:
        entity_filters.append(AuditLogEntry.entity_id.in_(parsed["entities"]["symbols"]))

    if entity_filters:
        filters.append(or_(*entity_filters))

    # Execute query
    query = db.query(AuditLogEntry)
    if filters:
        query = query.filter(and_(*filters))

    # Order by timestamp (causal chain)
    query = query.order_by(AuditLogEntry.timestamp_utc.asc())

    # Get total count
    total_count = query.count()

    # Apply pagination
    results = query.offset(request.offset).limit(request.limit).all()

    # Build causal chain (chronological)
    causal_chain = [
        {
            "timestamp_utc": entry.timestamp_utc.isoformat() if entry.timestamp_utc else None,
            "layer": entry.layer,
            "event_type": entry.event_type,
            "actor": entry.actor,
            "reason": entry.reason
        }
        for entry in sorted(results, key=lambda e: e.timestamp_utc or datetime.min)
    ]

    return AuditQueryResponse(
        query=request.query,
        results=[AuditLogResponse(**r.to_dict()) for r in results],
        total_count=total_count,
        causal_chain=causal_chain,
        parsed_entities=parsed["entities"],
        parsed_time_range=parsed["time_range"]
    )


# ============================================================================
# Query Endpoints
# ============================================================================


@router.get("/layers", response_model=List[str])
def get_audit_layers():
    """Get list of all available audit layers."""
    return AuditLayer.ALL_LAYERS


@router.get("/event-types/{layer}", response_model=List[str])
def get_event_types(layer: str):
    """Get event types for a specific audit layer."""
    layer_map = {
        "trade": [TradeEventType.EXECUTION, TradeEventType.CLOSE, TradeEventType.MODIFY, TradeEventType.CANCEL],
        "strategy_lifecycle": [StrategyLifecycleEventType.START, StrategyLifecycleEventType.PAUSE, StrategyLifecycleEventType.RESUME, StrategyLifecycleEventType.STOP, StrategyLifecycleEventType.REGIME_CHANGE],
        "risk_param": [RiskParamEventType.PARAMETER_CHANGE, RiskParamEventType.DAILY_LOSS_CAP_CHANGE, RiskParamEventType.KELLY_FRACTION_CHANGE, RiskParamEventType.POSITION_MULTIPLIER_CHANGE],
        "agent_action": [AgentActionEventType.TASK_DISPATCH, AgentActionEventType.TASK_COMPLETE, AgentActionEventType.OPINION_GENERATED],
        "system_health": [SystemHealthEventType.SERVER_START, SystemHealthEventType.SERVER_STOP, SystemHealthEventType.HEALTH_BREACH, SystemHealthEventType.ERROR],
    }
    return layer_map.get(layer, [])


@router.get("/entries", response_model=List[AuditLogResponse])
def get_audit_entries(
    layer: Optional[str] = Query(None, description="Filter by audit layer"),
    entity_id: Optional[str] = Query(None, description="Filter by entity ID"),
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    start_time: Optional[datetime] = Query(None, description="Start time filter"),
    end_time: Optional[datetime] = Query(None, description="End time filter"),
    limit: int = Query(50, ge=1, le=1000, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Result offset"),
    db: Session = Depends(get_audit_db)
):
    """Get audit log entries with filters."""
    query = db.query(AuditLogEntry)

    filters = []
    if layer:
        filters.append(AuditLogEntry.layer == layer)
    if entity_id:
        filters.append(AuditLogEntry.entity_id == entity_id)
    if event_type:
        filters.append(AuditLogEntry.event_type == event_type)
    if start_time:
        filters.append(AuditLogEntry.timestamp_utc >= start_time)
    if end_time:
        filters.append(AuditLogEntry.timestamp_utc <= end_time)

    if filters:
        query = query.filter(and_(*filters))

    query = query.order_by(AuditLogEntry.timestamp_utc.desc())
    results = query.offset(offset).limit(limit).all()

    return [AuditLogResponse(**r.to_dict()) for r in results]


@router.get("/entries/{entry_id}", response_model=AuditLogResponse)
def get_audit_entry(entry_id: str, db: Session = Depends(get_audit_db)):
    """Get a specific audit log entry by ID."""
    entry = db.query(AuditLogEntry).filter(AuditLogEntry.id == entry_id).first()
    if not entry:
        raise HTTPException(status_code=404, detail="Audit log entry not found")
    return AuditLogResponse(**entry.to_dict())


@router.get("/health", response_model=Dict[str, Any])
def audit_health_check(db: Session = Depends(get_audit_db)):
    """Health check for audit system."""
    try:
        # Try a simple query
        count = db.query(AuditLogEntry).count()
        return {
            "status": "healthy",
            "total_entries": count,
            "layers": AuditLayer.ALL_LAYERS
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }
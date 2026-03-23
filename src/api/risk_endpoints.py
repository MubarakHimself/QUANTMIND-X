"""
Risk API Endpoints - CalendarGovernor & Risk Parameters

Provides REST API endpoints for:
- Calendar rule CRUD operations
- Economic calendar event management
- Risk parameter configuration
- Prop firm registry management

Story: 4-2-risk-parameters-prop-firm-registry-apis
"""

import logging
from fastapi import APIRouter, HTTPException, Query, Depends, Request
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from src.risk.models.calendar import (
    NewsItem,
    CalendarRule,
    CalendarEventType,
    NewsImpact,
    CalendarPhase,
    DEFAULT_BLACKOUT_MINUTES,
    DEFAULT_POST_EVENT_DELAY_MINUTES,
)
from src.database.models import (
    RiskParams,
    RiskParamsAudit,
    PropFirmAccount,
    TradingMode,
    AccountType,
    get_db_session,
)

router = APIRouter(prefix="/api/risk", tags=["risk"])

logger = logging.getLogger(__name__)

# In-memory storage for demo (would be database in production)
_calendar_rules: Dict[str, CalendarRule] = {}
_calendar_events: List[NewsItem] = []

# Demo data for compliance and circuit breaker state
_demo_account_tags = ["prop-firm-001", "prop-firm-002", "demo-account"]


# =============================================================================
# Request/Response Models
# =============================================================================

class CalendarRuleCreateRequest(BaseModel):
    """Request to create a calendar rule."""
    rule_id: str = Field(..., description="Unique rule identifier")
    account_id: str = Field(..., description="Account this rule applies to")
    blacklist_enabled: bool = Field(True, description="Enable blackout rules")
    blackout_minutes: int = Field(DEFAULT_BLACKOUT_MINUTES, description="Blackout window in minutes")
    post_event_delay_minutes: int = Field(DEFAULT_POST_EVENT_DELAY_MINUTES, description="Post-event delay")
    regime_check_enabled: bool = Field(True, description="Enable regime check for reactivation")
    affected_symbols: List[str] = Field(default_factory=list, description="Affected symbols")
    enabled: bool = Field(True, description="Whether rule is active")


class CalendarRuleUpdateRequest(BaseModel):
    """Request to update a calendar rule."""
    blacklist_enabled: Optional[bool] = None
    blackout_minutes: Optional[int] = None
    post_event_delay_minutes: Optional[int] = None
    regime_check_enabled: Optional[bool] = None
    affected_symbols: Optional[List[str]] = None
    enabled: Optional[bool] = None


class NewsItemCreateRequest(BaseModel):
    """Request to create a news event."""
    event_id: str = Field(..., description="Unique event identifier")
    title: str = Field(..., description="Event title")
    event_type: CalendarEventType = Field(..., description="Type of economic event")
    impact: NewsImpact = Field(..., description="Impact level")
    event_time: datetime = Field(..., description="Event timestamp in UTC")
    currencies: List[str] = Field(default_factory=list, description="Affected currencies")
    description: Optional[str] = Field(None, description="Optional description")
    source: Optional[str] = Field("manual", description="Data source")


# =============================================================================
# Risk Parameters Models (Story 4.2)
# =============================================================================

class RiskParamsGetResponse(BaseModel):
    """Risk parameters response."""
    account_tag: str
    daily_loss_cap_pct: float
    max_trades_per_day: int
    kelly_fraction: float
    position_multiplier: float
    lyapunov_threshold: float
    hmm_retrain_trigger: float
    updated_at: datetime


class RiskParamsUpdateRequest(BaseModel):
    """Request to update risk parameters."""
    daily_loss_cap_pct: Optional[float] = Field(None, ge=0.001, le=100, description="Daily loss cap percentage")
    max_trades_per_day: Optional[int] = Field(None, ge=1, description="Max trades per day")
    kelly_fraction: Optional[float] = Field(None, ge=0.001, le=1.0, description="Kelly fraction (0-1)")
    position_multiplier: Optional[float] = Field(None, ge=0.1, le=10.0, description="Position multiplier")
    lyapunov_threshold: Optional[float] = Field(None, ge=0.001, le=1.0, description="Lyapunov threshold")
    hmm_retrain_trigger: Optional[float] = Field(None, ge=0.001, le=1.0, description="HMM retrain trigger")

    @field_validator('kelly_fraction')
    @classmethod
    def validate_kelly(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v > 1.0:
            raise ValueError('kelly_fraction must be <= 1.0')
        return v


class RiskParamsUpdateResponse(BaseModel):
    """Response after updating risk parameters."""
    status: str
    account_tag: str
    updated_fields: List[str]


# =============================================================================
# Prop Firm Registry Models (Story 4.2)
# =============================================================================

class PropFirmCreateRequest(BaseModel):
    """Request to create a prop firm entry."""
    firm_name: str = Field(..., description="Name of the prop firm")
    account_id: str = Field(..., description="MT5 account number")
    daily_loss_limit_pct: float = Field(..., ge=0.001, le=100, description="Daily loss limit percentage")
    target_profit_pct: float = Field(..., ge=0.001, description="Target profit percentage")
    risk_mode: str = Field(..., description="Risk mode: growth, scaling, guardian")


class PropFirmUpdateRequest(BaseModel):
    """Request to update a prop firm entry."""
    firm_name: Optional[str] = None
    daily_loss_limit_pct: Optional[float] = Field(None, ge=0.001, le=100)
    target_profit_pct: Optional[float] = Field(None, ge=0.001)
    risk_mode: Optional[str] = None
    enabled: Optional[bool] = None


class PropFirmResponse(BaseModel):
    """Prop firm entry response."""
    id: int
    firm_name: str
    account_id: str
    daily_loss_limit_pct: float
    target_profit_pct: float
    risk_mode: str
    account_type: str
    created_at: datetime
    updated_at: datetime


# =============================================================================
# Calendar Rule Endpoints
# =============================================================================

@router.post("/calendar/rules", response_model=CalendarRule)
async def create_calendar_rule(request: CalendarRuleCreateRequest):
    """Create a new calendar rule for an account."""
    rule = CalendarRule(
        rule_id=request.rule_id,
        account_id=request.account_id,
        blacklist_enabled=request.blacklist_enabled,
        blackout_minutes=request.blackout_minutes,
        post_event_delay_minutes=request.post_event_delay_minutes,
        regime_check_enabled=request.regime_check_enabled,
        affected_symbols=request.affected_symbols,
        enabled=request.enabled,
    )

    _calendar_rules[request.account_id] = rule
    logger.info(f"Calendar rule created: {request.rule_id} for account {request.account_id}")

    return rule


@router.get("/calendar/rules/{account_id}", response_model=CalendarRule)
async def get_calendar_rule(account_id: str):
    """Get calendar rule for an account."""
    if account_id not in _calendar_rules:
        raise HTTPException(status_code=404, detail=f"Calendar rule not found for account {account_id}")
    return _calendar_rules[account_id]


@router.get("/calendar/rules")
async def list_calendar_rules():
    """List all calendar rules."""
    return list(_calendar_rules.values())


@router.put("/calendar/rules/{account_id}", response_model=CalendarRule)
async def update_calendar_rule(account_id: str, request: CalendarRuleUpdateRequest):
    """Update calendar rule for an account."""
    if account_id not in _calendar_rules:
        raise HTTPException(status_code=404, detail=f"Calendar rule not found for account {account_id}")

    rule = _calendar_rules[account_id]

    # Update only provided fields
    if request.blacklist_enabled is not None:
        rule.blacklist_enabled = request.blacklist_enabled
    if request.blackout_minutes is not None:
        rule.blackout_minutes = request.blackout_minutes
    if request.post_event_delay_minutes is not None:
        rule.post_event_delay_minutes = request.post_event_delay_minutes
    if request.regime_check_enabled is not None:
        rule.regime_check_enabled = request.regime_check_enabled
    if request.affected_symbols is not None:
        rule.affected_symbols = request.affected_symbols
    if request.enabled is not None:
        rule.enabled = request.enabled

    rule.updated_at = datetime.now(timezone.utc)

    logger.info(f"Calendar rule updated for account {account_id}")
    return rule


@router.delete("/calendar/rules/{account_id}")
async def delete_calendar_rule(account_id: str):
    """Delete calendar rule for an account."""
    if account_id not in _calendar_rules:
        raise HTTPException(status_code=404, detail=f"Calendar rule not found for account {account_id}")

    del _calendar_rules[account_id]
    logger.info(f"Calendar rule deleted for account {account_id}")

    return {"status": "deleted", "account_id": account_id}


# =============================================================================
# Calendar Event Endpoints
# =============================================================================

@router.post("/calendar/events", response_model=NewsItem)
async def create_calendar_event(request: NewsItemCreateRequest):
    """Add a calendar event (news item)."""
    event = NewsItem(
        event_id=request.event_id,
        title=request.title,
        event_type=request.event_type,
        impact=request.impact,
        event_time=request.event_time,
        currencies=request.currencies,
        description=request.description,
        source=request.source,
    )

    # Check for duplicates
    existing_ids = {e.event_id for e in _calendar_events}
    if event.event_id not in existing_ids:
        _calendar_events.append(event)
        logger.info(f"Calendar event added: {event.event_id} at {event.event_time}")
    else:
        raise HTTPException(status_code=409, detail=f"Event {event.event_id} already exists")

    return event


@router.get("/calendar/events")
async def list_calendar_events(
    impact: Optional[NewsImpact] = Query(None, description="Filter by impact level"),
    from_time: Optional[datetime] = Query(None, description="Filter events from this time"),
):
    """List calendar events, optionally filtered."""
    events = _calendar_events

    if impact:
        events = [e for e in events if e.impact == impact]

    if from_time:
        events = [e for e in events if e.event_time >= from_time]

    # Sort by event time
    events.sort(key=lambda e: e.event_time)

    return events


@router.get("/calendar/events/{event_id}", response_model=NewsItem)
async def get_calendar_event(event_id: str):
    """Get a specific calendar event."""
    for event in _calendar_events:
        if event.event_id == event_id:
            return event

    raise HTTPException(status_code=404, detail=f"Event {event_id} not found")


@router.delete("/calendar/events/{event_id}")
async def delete_calendar_event(event_id: str):
    """Delete a calendar event."""
    global _calendar_events

    initial_count = len(_calendar_events)
    _calendar_events = [e for e in _calendar_events if e.event_id != event_id]

    if len(_calendar_events) == initial_count:
        raise HTTPException(status_code=404, detail=f"Event {event_id} not found")

    logger.info(f"Calendar event deleted: {event_id}")
    return {"status": "deleted", "event_id": event_id}


# =============================================================================
# Risk Parameters Endpoints (Story 4.2 - Full Implementation)
# =============================================================================

@router.get("/params", response_model=RiskParamsGetResponse)
async def get_default_risk_params(db: Session = Depends(get_db_session)):
    """Get risk params for the default account (used by RiskPanel settings)."""
    return await get_risk_params("default", db)


@router.post("/params")
async def save_default_risk_params(request: RiskParamsUpdateRequest, db: Session = Depends(get_db_session)):
    """Save risk params for the default account (used by RiskPanel settings)."""
    return await update_risk_params("default", request, db)


@router.get("/params/{account_tag}", response_model=RiskParamsGetResponse)
async def get_risk_params(account_tag: str, db: Session = Depends(get_db_session)):
    """
    Get risk parameters for an account tag.

    AC #1: Returns current risk params:
    { daily_loss_cap_pct, max_trades_per_day, kelly_fraction,
      position_multiplier, lyapunov_threshold, hmm_retrain_trigger }
    """
    # Try to get from database
    params = db.query(RiskParams).filter(RiskParams.account_tag == account_tag).first()

    if params:
        logger.info(f"Retrieved risk params for account_tag: {account_tag}")
        return RiskParamsGetResponse(
            account_tag=params.account_tag,
            daily_loss_cap_pct=params.daily_loss_cap_pct,
            max_trades_per_day=params.max_trades_per_day,
            kelly_fraction=params.kelly_fraction,
            position_multiplier=params.position_multiplier,
            lyapunov_threshold=params.lyapunov_threshold,
            hmm_retrain_trigger=params.hmm_retrain_trigger,
            updated_at=params.updated_at,
        )

    # Return default values if not found (for demo purposes)
    logger.info(f"No stored risk params for {account_tag}, returning defaults")
    return RiskParamsGetResponse(
        account_tag=account_tag,
        daily_loss_cap_pct=5.0,
        max_trades_per_day=10,
        kelly_fraction=0.5,
        position_multiplier=1.0,
        lyapunov_threshold=0.3,
        hmm_retrain_trigger=0.7,
        updated_at=datetime.now(timezone.utc),
    )


@router.put("/params/{account_tag}", response_model=RiskParamsUpdateResponse)
async def update_risk_params(
    account_tag: str,
    request: RiskParamsUpdateRequest,
    db: Session = Depends(get_db_session)
):
    """
    Update risk parameters for an account tag.

    AC #2: Only provided fields are updated, change takes effect on next
    risk evaluation cycle (≤30 seconds), change is written to audit layer.
    """
    # Validate Kelly fraction specifically (AC #5)
    if request.kelly_fraction is not None and request.kelly_fraction > 1.0:
        raise HTTPException(
            status_code=422,
            detail="kelly_fraction must be <= 1.0"
        )

    # Get existing params or create new
    params = db.query(RiskParams).filter(RiskParams.account_tag == account_tag).first()

    updated_fields = []

    if not params:
        # Create new params with defaults for non-provided fields
        params = RiskParams(
            account_tag=account_tag,
            daily_loss_cap_pct=request.daily_loss_cap_pct if request.daily_loss_cap_pct else 5.0,
            max_trades_per_day=request.max_trades_per_day if request.max_trades_per_day else 10,
            kelly_fraction=request.kelly_fraction if request.kelly_fraction else 0.5,
            position_multiplier=request.position_multiplier if request.position_multiplier else 1.0,
            lyapunov_threshold=request.lyapunov_threshold if request.lyapunov_threshold else 0.3,
            hmm_retrain_trigger=request.hmm_retrain_trigger if request.hmm_retrain_trigger else 0.7,
        )
        db.add(params)
        logger.info(f"Created new risk params for account_tag: {account_tag}")
    else:
        # Update only provided fields (partial update)
        updated_fields = []

        if request.daily_loss_cap_pct is not None:
            old_val = params.daily_loss_cap_pct
            params.daily_loss_cap_pct = request.daily_loss_cap_pct
            _log_audit(db, account_tag, 'daily_loss_cap_pct', str(old_val), str(request.daily_loss_cap_pct))
            updated_fields.append('daily_loss_cap_pct')

        if request.max_trades_per_day is not None:
            old_val = params.max_trades_per_day
            params.max_trades_per_day = request.max_trades_per_day
            _log_audit(db, account_tag, 'max_trades_per_day', str(old_val), str(request.max_trades_per_day))
            updated_fields.append('max_trades_per_day')

        if request.kelly_fraction is not None:
            old_val = params.kelly_fraction
            params.kelly_fraction = request.kelly_fraction
            _log_audit(db, account_tag, 'kelly_fraction', str(old_val), str(request.kelly_fraction))
            updated_fields.append('kelly_fraction')

        if request.position_multiplier is not None:
            old_val = params.position_multiplier
            params.position_multiplier = request.position_multiplier
            _log_audit(db, account_tag, 'position_multiplier', str(old_val), str(request.position_multiplier))
            updated_fields.append('position_multiplier')

        if request.lyapunov_threshold is not None:
            old_val = params.lyapunov_threshold
            params.lyapunov_threshold = request.lyapunov_threshold
            _log_audit(db, account_tag, 'lyapunov_threshold', str(old_val), str(request.lyapunov_threshold))
            updated_fields.append('lyapunov_threshold')

        if request.hmm_retrain_trigger is not None:
            old_val = params.hmm_retrain_trigger
            params.hmm_retrain_trigger = request.hmm_retrain_trigger
            _log_audit(db, account_tag, 'hmm_retrain_trigger', str(old_val), str(request.hmm_retrain_trigger))
            updated_fields.append('hmm_retrain_trigger')

        logger.info(f"Updated risk params for account_tag: {account_tag}, fields: {updated_fields}")

    params.updated_at = datetime.now(timezone.utc)
    db.commit()

    return RiskParamsUpdateResponse(
        status="updated",
        account_tag=account_tag,
        updated_fields=updated_fields,
    )


def _log_audit(db: Session, account_tag: str, field: str, old_val: str, new_val: str):
    """Log risk parameter change to audit table."""
    audit = RiskParamsAudit(
        account_tag=account_tag,
        field_changed=field,
        old_value=old_val,
        new_value=new_val,
        changed_by='api',
    )
    db.add(audit)
    db.commit()
    logger.info(f"Audit log: {account_tag} {field} changed from {old_val} to {new_val}")


# =============================================================================
# Prop Firm Registry Endpoints (Story 4.2 - Full Implementation)
# =============================================================================

@router.get("/prop-firms", response_model=List[PropFirmResponse])
async def list_prop_firms(db: Session = Depends(get_db_session)):
    """
    List all prop firm entries.

    AC #3: Returns all configured prop firm entries with their rule sets.
    """
    firms = db.query(PropFirmAccount).filter(
        PropFirmAccount.account_type == AccountType.PROP_FIRM
    ).all()

    logger.info(f"Retrieved {len(firms)} prop firm entries")
    return [
        PropFirmResponse(
            id=firm.id,
            firm_name=firm.firm_name,
            account_id=firm.account_id,
            daily_loss_limit_pct=firm.daily_loss_limit_pct,
            target_profit_pct=firm.target_profit_pct,
            risk_mode=firm.risk_mode,
            account_type=firm.account_type.value,
            created_at=firm.created_at,
            updated_at=firm.updated_at,
        )
        for firm in firms
    ]


@router.post("/prop-firms", response_model=PropFirmResponse, status_code=201)
async def create_prop_firm(request: PropFirmCreateRequest, db: Session = Depends(get_db_session)):
    """
    Create a new prop firm entry.

    AC #4: Entry is created and available in routing matrix account tag assignment.
    """
    # Check if account_id already exists
    existing = db.query(PropFirmAccount).filter(
        PropFirmAccount.account_id == request.account_id
    ).first()

    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"Prop firm with account_id {request.account_id} already exists"
        )

    # Validate risk_mode
    if request.risk_mode not in ['growth', 'scaling', 'guardian']:
        raise HTTPException(
            status_code=422,
            detail="risk_mode must be one of: growth, scaling, guardian"
        )

    firm = PropFirmAccount(
        firm_name=request.firm_name,
        account_id=request.account_id,
        daily_loss_limit_pct=request.daily_loss_limit_pct,
        target_profit_pct=request.target_profit_pct,
        risk_mode=request.risk_mode,
        account_type=AccountType.PROP_FIRM,
        mode=TradingMode.LIVE,
    )

    db.add(firm)
    db.commit()
    db.refresh(firm)

    logger.info(f"Created prop firm: {request.firm_name} with account_id: {request.account_id}")

    return PropFirmResponse(
        id=firm.id,
        firm_name=firm.firm_name,
        account_id=firm.account_id,
        daily_loss_limit_pct=firm.daily_loss_limit_pct,
        target_profit_pct=firm.target_profit_pct,
        risk_mode=firm.risk_mode,
        account_type=firm.account_type.value,
        created_at=firm.created_at,
        updated_at=firm.updated_at,
    )


@router.get("/prop-firms/{firm_id}", response_model=PropFirmResponse)
async def get_prop_firm(firm_id: int, db: Session = Depends(get_db_session)):
    """Get a specific prop firm entry by ID."""
    firm = db.query(PropFirmAccount).filter(PropFirmAccount.id == firm_id).first()

    if not firm:
        raise HTTPException(status_code=404, detail=f"Prop firm with id {firm_id} not found")

    return PropFirmResponse(
        id=firm.id,
        firm_name=firm.firm_name,
        account_id=firm.account_id,
        daily_loss_limit_pct=firm.daily_loss_limit_pct,
        target_profit_pct=firm.target_profit_pct,
        risk_mode=firm.risk_mode,
        account_type=firm.account_type.value,
        created_at=firm.created_at,
        updated_at=firm.updated_at,
    )


@router.put("/prop-firms/{firm_id}", response_model=PropFirmResponse)
async def update_prop_firm(
    firm_id: int,
    request: PropFirmUpdateRequest,
    db: Session = Depends(get_db_session)
):
    """Update a prop firm entry."""
    firm = db.query(PropFirmAccount).filter(PropFirmAccount.id == firm_id).first()

    if not firm:
        raise HTTPException(status_code=404, detail=f"Prop firm with id {firm_id} not found")

    if request.firm_name is not None:
        firm.firm_name = request.firm_name

    if request.daily_loss_limit_pct is not None:
        firm.daily_loss_limit_pct = request.daily_loss_limit_pct

    if request.target_profit_pct is not None:
        firm.target_profit_pct = request.target_profit_pct

    if request.risk_mode is not None:
        if request.risk_mode not in ['growth', 'scaling', 'guardian']:
            raise HTTPException(
                status_code=422,
                detail="risk_mode must be one of: growth, scaling, guardian"
            )
        firm.risk_mode = request.risk_mode

    firm.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(firm)

    logger.info(f"Updated prop firm id: {firm_id}")

    return PropFirmResponse(
        id=firm.id,
        firm_name=firm.firm_name,
        account_id=firm.account_id,
        daily_loss_limit_pct=firm.daily_loss_limit_pct,
        target_profit_pct=firm.target_profit_pct,
        risk_mode=firm.risk_mode,
        account_type=firm.account_type.value,
        created_at=firm.created_at,
        updated_at=firm.updated_at,
    )


@router.delete("/prop-firms/{firm_id}")
async def delete_prop_firm(firm_id: int, db: Session = Depends(get_db_session)):
    """Delete a prop firm entry."""
    firm = db.query(PropFirmAccount).filter(PropFirmAccount.id == firm_id).first()

    if not firm:
        raise HTTPException(status_code=404, detail=f"Prop firm with id {firm_id} not found")

    db.delete(firm)
    db.commit()

    logger.info(f"Deleted prop firm id: {firm_id}")
    return {"status": "deleted", "firm_id": firm_id}


# =============================================================================
# Strategy Router & Regime State Endpoints (Story 4.3)
# =============================================================================

# Import router components for regime detection
try:
    from src.router.engine import StrategyRouter
    from src.router.state import RouterState
    ROUTER_AVAILABLE = True
except ImportError:
    ROUTER_AVAILABLE = False
    StrategyRouter = None
    RouterState = None

# Import physics sensors
try:
    from src.risk.physics import IsingRegimeSensor, ChaosSensor, HMMRegimeSensor
    SENSORS_AVAILABLE = True
except ImportError:
    SENSORS_AVAILABLE = False
    IsingRegimeSensor = None
    ChaosSensor = None
    HMMRegimeSensor = None

# Global sensor instances (lazy initialization)
_ising_sensor: Optional[IsingRegimeSensor] = None
_chaos_sensor: Optional[ChaosSensor] = None
_hmm_sensor: Optional[HMMRegimeSensor] = None


def _get_ising_sensor() -> IsingRegimeSensor:
    """Get or create Ising sensor instance."""
    global _ising_sensor
    if _ising_sensor is None and SENSORS_AVAILABLE:
        _ising_sensor = IsingRegimeSensor()
    return _ising_sensor


def _get_chaos_sensor() -> ChaosSensor:
    """Get or create Chaos sensor instance."""
    global _chaos_sensor
    if _chaos_sensor is None and SENSORS_AVAILABLE:
        _chaos_sensor = ChaosSensor()
    return _chaos_sensor


def _get_hmm_sensor() -> Optional[HMMRegimeSensor]:
    """Get or create HMM sensor instance."""
    global _hmm_sensor
    if _hmm_sensor is None and SENSORS_AVAILABLE:
        try:
            _hmm_sensor = HMMRegimeSensor()
        except Exception:
            logger.warning("HMM sensor initialization failed, returning None")
            _hmm_sensor = None
    return _hmm_sensor


# Regime types as defined in the architecture
class RegimeType(str):
    TREND = "TREND"
    RANGE = "RANGE"
    BREAKOUT = "BREAKOUT"
    CHAOS = "CHAOS"
    UNKNOWN = "UNKNOWN"


class StrategyStatus(str):
    ACTIVE = "active"
    PAUSED = "paused"
    QUARANTINE = "quarantine"


class PauseReason(str):
    CALENDAR_RULE = "calendar_rule"
    RISK_BREACH = "risk_breach"
    MANUAL = "manual"
    REGIME_MISMATCH = "regime_mismatch"


class AlertState(str):
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"


# Response Models for Story 4.3

class RegimeResponse(BaseModel):
    """Response model for GET /api/risk/regime"""
    regime: str = Field(..., description="Current regime classification: TREND, RANGE, BREAKOUT, CHAOS, UNKNOWN")
    confidence_pct: float = Field(..., ge=0, le=100, description="Confidence percentage 0-100")
    transition_at_utc: datetime = Field(..., description="Last regime transition timestamp")
    previous_regime: Optional[str] = Field(None, description="Previous regime before current")
    active_strategy_count: int = Field(..., ge=0, description="Number of active strategies")
    paused_strategy_count: int = Field(..., ge=0, description="Number of paused strategies")


class StrategyStateItem(BaseModel):
    """Per-strategy router state item"""
    strategy_id: str = Field(..., description="Strategy identifier")
    status: str = Field(..., description="Status: active, paused, quarantine")
    pause_reason: Optional[str] = Field(None, description="Pause reason if paused")
    eligible_regimes: List[str] = Field(default_factory=list, description="Regimes where strategy can run")


class RouterStateResponse(BaseModel):
    """Response model for GET /api/risk/router/state"""
    strategies: List[StrategyStateItem] = Field(default_factory=list, description="Per-strategy router state")


class PhysicsIsingOutput(BaseModel):
    """Ising Model output"""
    magnetization: float = Field(..., description="System magnetization (-1 to 1)")
    correlation_matrix: Optional[Dict[str, Any]] = Field(None, description="Correlation matrix if available")
    alert: str = Field(..., description="Alert state: normal, warning, critical")


class PhysicsLyapunovOutput(BaseModel):
    """Lyapunov Exponent output"""
    exponent_value: float = Field(..., description="Lyapunov exponent value")
    divergence_rate: Optional[float] = Field(None, description="Divergence rate if calculated")
    alert: str = Field(..., description="Alert state: normal, warning, critical")


class PhysicsHMMOutput(BaseModel):
    """HMM regime output"""
    current_state: Optional[str] = Field(None, description="Current HMM state")
    transition_probabilities: Optional[Dict[str, float]] = Field(None, description="State transition probabilities")
    is_shadow_mode: bool = Field(False, description="Whether HMM is in shadow mode")
    alert: str = Field(..., description="Alert state: normal, warning, critical")


class PhysicsKellyOutput(BaseModel):
    """Kelly Engine output"""
    fraction: float = Field(..., description="Current Kelly fraction (0-1)")
    multiplier: float = Field(1.0, description="Physics multiplier from market conditions")
    house_of_money: bool = Field(False, description="Whether in house-of-money (favorable) state")
    kelly_fraction_setting: float = Field(0.5, description="Configured Kelly fraction setting")


class PhysicsResponse(BaseModel):
    """Response model for GET /api/risk/physics"""
    ising: PhysicsIsingOutput = Field(..., description="Ising Model sensor outputs")
    lyapunov: PhysicsLyapunovOutput = Field(..., description="Lyapunov exponent outputs")
    hmm: PhysicsHMMOutput = Field(..., description="HMM regime sensor outputs")
    kelly: PhysicsKellyOutput = Field(..., description="Kelly Engine outputs")


# =============================================================================
# Regime Classification Endpoint (AC #1)
# =============================================================================

@router.get("/regime", response_model=RegimeResponse)
async def get_regime():
    """
    Get current regime classification and strategy counts.

    AC #1: Returns { regime, confidence_pct, transition_at_utc, previous_regime,
                   active_strategy_count, paused_strategy_count }

    Regime types: TREND, RANGE, BREAKOUT, CHAOS, UNKNOWN
    """
    # Try to get regime from router if available
    regime = RegimeType.UNKNOWN
    confidence_pct = 50.0
    previous_regime = None
    transition_at = datetime.now(timezone.utc)
    active_count = 0
    paused_count = 0

    if ROUTER_AVAILABLE:
        try:
            # Try to get router state - check if there's a global instance
            from src.api.router_endpoints import _strategy_router
            if _strategy_router is not None:
                router = _strategy_router

                # Try to get regime from router
                if hasattr(router, 'get_regime_for_symbol'):
                    try:
                        regime_info = router.get_regime_for_symbol('EURUSD')
                        if regime_info:
                            regime = regime_info.get('regime', RegimeType.UNKNOWN)
                            confidence_pct = regime_info.get('confidence', 50.0)
                            previous_regime = regime_info.get('previous_regime')
                            if 'transition_at' in regime_info:
                                transition_at = regime_info['transition_at']
                    except Exception as e:
                        logger.warning(f"Could not get regime from router: {e}")

                # Try to get strategy counts
                if hasattr(router, 'commander') and router.commander:
                    commander = router.commander
                    if hasattr(commander, 'active_bots'):
                        active_count = len(commander.active_bots)

                if hasattr(router, 'governor') and router.governor:
                    governor = router.governor
                    # Check for paused strategies via kill switch
                    if hasattr(governor, 'progressive_kill_switch'):
                        pks = governor.progressive_kill_switch
                        if hasattr(pks, 'paused_bots'):
                            paused_count = len(pks.paused_bots)
                        elif hasattr(pks, 'get_paused_count'):
                            paused_count = pks.get_paused_count()
        except Exception as e:
            logger.warning(f"Could not get regime from router: {e}")

    logger.info(f"Regime response: {regime}, confidence: {confidence_pct}%, active: {active_count}, paused: {paused_count}")

    return RegimeResponse(
        regime=regime,
        confidence_pct=confidence_pct,
        transition_at_utc=transition_at,
        previous_regime=previous_regime,
        active_strategy_count=active_count,
        paused_strategy_count=paused_count
    )


# =============================================================================
# Strategy Router State Endpoint (AC #2)
# =============================================================================

@router.get("/router/state", response_model=RouterStateResponse)
async def get_router_state():
    """
    Get per-strategy router state.

    AC #2: Returns [{ strategy_id, status, pause_reason, eligible_regimes }]

    Status values: active, paused, quarantine
    Pause reasons: calendar_rule, risk_breach, manual, regime_mismatch
    """
    strategies = []

    if ROUTER_AVAILABLE:
        try:
            from src.api.router_endpoints import _strategy_router
            if _strategy_router is not None:
                router = _strategy_router

                # Get all registered bots/strategies
                if hasattr(router, 'registered_bots'):
                    for bot_id, bot in router.registered_bots.items():
                        status = StrategyStatus.ACTIVE
                        pause_reason = None
                        eligible_regimes = [RegimeType.TREND, RegimeType.RANGE, RegimeType.BREAKOUT]

                        # Check if paused via kill switch
                        if hasattr(router, 'governor') and router.governor:
                            governor = router.governor
                            if hasattr(governor, 'progressive_kill_switch'):
                                pks = governor.progressive_kill_switch
                                if hasattr(pks, 'paused_bots') and bot_id in pks.paused_bots:
                                    status = StrategyStatus.PAUSED
                                    pause_reason = PauseReason.RISK_BREACH
                                elif hasattr(pks, 'quarantined_bots') and bot_id in pks.quarantined_bots:
                                    status = StrategyStatus.QUARANTINE

                        # Get eligible regimes from strategy config
                        if hasattr(bot, 'eligible_regimes'):
                            eligible_regimes = bot.eligible_regimes

                        strategies.append(StrategyStateItem(
                            strategy_id=bot_id,
                            status=status.value,
                            pause_reason=pause_reason,
                            eligible_regimes=eligible_regimes
                        ))

                # If no registered bots, return demo data
                if not strategies:
                    strategies = _get_demo_strategy_states()
        except Exception as e:
            logger.warning(f"Could not get router state: {e}")
            strategies = _get_demo_strategy_states()
    else:
        # Return demo data if router not available
        strategies = _get_demo_strategy_states()

    logger.info(f"Router state response: {len(strategies)} strategies")

    return RouterStateResponse(strategies=strategies)


def _get_demo_strategy_states() -> List[StrategyStateItem]:
    """Return demo strategy states for testing when router unavailable."""
    return [
        StrategyStateItem(
            strategy_id="trend-follower-001",
            status=StrategyStatus.ACTIVE.value,
            pause_reason=None,
            eligible_regimes=[RegimeType.TREND, RegimeType.BREAKOUT]
        ),
        StrategyStateItem(
            strategy_id="range-trader-002",
            status=StrategyStatus.PAUSED.value,
            pause_reason=PauseReason.REGIME_MISMATCH.value,
            eligible_regimes=[RegimeType.RANGE]
        ),
        StrategyStateItem(
            strategy_id="breakout-scalper-003",
            status=StrategyStatus.ACTIVE.value,
            pause_reason=None,
            eligible_regimes=[RegimeType.BREAKOUT, RegimeType.TREND]
        ),
        StrategyStateItem(
            strategy_id="volatility-adaptor-004",
            status=StrategyStatus.PAUSED.value,
            pause_reason=PauseReason.CALENDAR_RULE.value,
            eligible_regimes=[RegimeType.TREND, RegimeType.RANGE, RegimeType.BREAKOUT, RegimeType.CHAOS]
        ),
    ]


# =============================================================================
# Physics Sensor Outputs Endpoint (AC #3)
# =============================================================================

@router.get("/physics", response_model=PhysicsResponse)
async def get_physics_outputs():
    """
    Get physics sensor outputs.

    AC #3: Returns { ising, lyapunov, hmm } with their outputs and alert states.

    - Ising: magnetization, correlation_matrix, alert
    - Lyapunov: exponent_value, divergence_rate, alert
    - HMM: current_state, transition_probabilities, alert
    """
    # Get Ising sensor output
    ising_alert = AlertState.NORMAL
    ising_magnetization = 0.0

    if SENSORS_AVAILABLE and _get_ising_sensor() is not None:
        try:
            sensor = _get_ising_sensor()
            result = sensor.detect_regime(market_volatility=0.15)
            ising_magnetization = result.get('magnetization', 0.0)

            # Determine alert based on magnetization
            abs_mag = abs(ising_magnetization)
            if abs_mag >= 0.8:
                ising_alert = AlertState.NORMAL  # Strong signal
            elif abs_mag >= 0.3:
                ising_alert = AlertState.WARNING  # Transitioning
            else:
                ising_alert = AlertState.CRITICAL  # Unclear/noise

            logger.info(f"Ising sensor: magnetization={ising_magnetization}, alert={ising_alert}")
        except Exception as e:
            logger.warning(f"Could not get Ising sensor output: {e}")

    ising_output = PhysicsIsingOutput(
        magnetization=ising_magnetization,
        correlation_matrix=None,  # Would require price data to compute
        alert=ising_alert.value
    )

    # Get Lyapunov/Chaos sensor output
    lyapunov_alert = AlertState.NORMAL
    lyapunov_exponent = 0.0
    divergence_rate = None

    if SENSORS_AVAILABLE and _get_chaos_sensor() is not None:
        try:
            sensor = _get_chaos_sensor()
            # For demo, create synthetic returns to get a reading
            import numpy as np
            synthetic_returns = np.random.randn(100) * 0.01
            result = sensor.analyze_chaos(synthetic_returns)
            lyapunov_exponent = result.lyapunov_exponent
            divergence_rate = result.divergence_rate if hasattr(result, 'divergence_rate') else None

            # Determine alert based on lyapunov exponent
            if lyapunov_exponent < 0.2:
                lyapunov_alert = AlertState.NORMAL
            elif lyapunov_exponent < 0.5:
                lyapunov_alert = AlertState.WARNING
            else:
                lyapunov_alert = AlertState.CRITICAL

            logger.info(f"Lyapunov sensor: exponent={lyapunov_exponent}, alert={lyapunov_alert}")
        except Exception as e:
            logger.warning(f"Could not get Lyapunov sensor output: {e}")

    lyapunov_output = PhysicsLyapunovOutput(
        exponent_value=lyapunov_exponent,
        divergence_rate=divergence_rate,
        alert=lyapunov_alert.value
    )

    # Get HMM sensor output
    hmm_alert = AlertState.NORMAL
    hmm_state = None
    hmm_transitions = None

    hmm_sensor = _get_hmm_sensor()
    if hmm_sensor is not None:
        try:
            # Try to get current state
            if hasattr(hmm_sensor, 'get_current_state'):
                hmm_state = hmm_sensor.get_current_state()

            # Try to get state distribution as proxy for transition probabilities
            if hasattr(hmm_sensor, 'get_state_distribution'):
                state_dist = hmm_sensor.get_state_distribution()
                if state_dist:
                    hmm_transitions = {str(k): v for k, v in state_dist.items()}

            # HMM is in shadow mode per story notes - always show as warning
            hmm_alert = AlertState.WARNING

            logger.info(f"HMM sensor: state={hmm_state}, alert={hmm_alert}")
        except Exception as e:
            logger.warning(f"Could not get HMM sensor output: {e}")

    hmm_output = PhysicsHMMOutput(
        current_state=hmm_state,
        transition_probabilities=hmm_transitions,
        is_shadow_mode=True,  # HMM is in shadow mode per story requirements
        alert=hmm_alert.value
    )

    # Get Kelly Engine output
    # For dashboard, we show simulated values based on market conditions
    # In production, this would come from the Kelly Engine
    kelly_fraction = 0.5  # Default half-Kelly
    kelly_multiplier = 1.0
    house_of_money = False

    # Determine house_of_money based on other sensor states
    # If all sensors are in NORMAL state, we're in favorable conditions
    if ising_alert == AlertState.NORMAL and lyapunov_alert == AlertState.NORMAL:
        house_of_money = True
        kelly_multiplier = 1.2  # Boost multiplier in favorable conditions
    elif ising_alert == AlertState.WARNING or lyapunov_alert == AlertState.WARNING:
        kelly_multiplier = 0.8  # Reduce in transitional states
    else:
        kelly_multiplier = 0.5  # Significant reduction in critical conditions

    kelly_output = PhysicsKellyOutput(
        fraction=kelly_fraction,
        multiplier=kelly_multiplier,
        house_of_money=house_of_money,
        kelly_fraction_setting=kelly_fraction
    )

    logger.info(f"Physics outputs: ising={ising_alert.value}, lyapunov={lyapunov_alert.value}, hmm={hmm_alert.value}, kelly={kelly_fraction}")

    return PhysicsResponse(
        ising=ising_output,
        lyapunov=lyapunov_output,
        hmm=hmm_output,
        kelly=kelly_output
    )


# =============================================================================
# Compliance & Circuit Breaker Endpoints (Story 4.6)
# =============================================================================

class AccountTagCompliance(BaseModel):
    """Compliance state per account tag."""
    tag: str = Field(..., description="Account tag identifier")
    circuit_breaker_state: str = Field(..., description="Circuit breaker state: normal, warning, triggered")
    drawdown_pct: float = Field(..., description="Current drawdown percentage")
    daily_halt_triggered: bool = Field(False, description="Whether daily halt has been triggered")
    paused_strategies: int = Field(0, description="Number of paused strategies")
    last_check_utc: datetime = Field(..., description="Last check timestamp")


class IslamicComplianceStatus(BaseModel):
    """Islamic compliance countdown status."""
    countdown_seconds: int = Field(..., description="Seconds until force-close (0 if not within window)")
    force_close_at: Optional[datetime] = Field(None, description="Scheduled force-close time")
    is_within_60min_window: bool = Field(False, description="Within 60-minute warning window")
    is_within_30min_window: bool = Field(False, description="Within 30-minute critical window")
    current_time_utc: datetime = Field(..., description="Current server time")
    active_positions_count: int = Field(0, description="Number of positions that will be closed")


class ComplianceResponse(BaseModel):
    """Compliance overview response."""
    account_tags: List[AccountTagCompliance] = Field(..., description="Per-account-tag compliance states")
    islamic: IslamicComplianceStatus = Field(..., description="Islamic compliance status")
    overall_status: str = Field(..., description="Overall compliance: compliant, warning, critical")


@router.get("/compliance", response_model=ComplianceResponse)
async def get_compliance_status():
    """
    Get compliance status including BotCircuitBreaker states.

    AC #1: Returns BotCircuitBreaker state per account tag with:
    { account_tags: [{ tag, circuit_breaker_state, drawdown_pct, daily_halt_triggered }],
      islamic: { countdown_seconds, force_close_at } }
    """
    import random
    from datetime import timedelta

    # Get current UTC time
    now = datetime.now(timezone.utc)

    # Calculate Islamic compliance
    # Force-close time: 21:45 UTC daily
    force_close_hour = 21
    force_close_minute = 45

    # Calculate seconds until force-close
    force_close_today = now.replace(hour=force_close_hour, minute=force_close_minute, second=0, microsecond=0)

    if now > force_close_today:
        # Force close already passed today, use tomorrow
        force_close = force_close_today + timedelta(days=1)
    else:
        force_close = force_close_today

    countdown_seconds = int((force_close - now).total_seconds())
    is_within_60min = 0 <= countdown_seconds <= 3600
    is_within_30min = 0 <= countdown_seconds <= 1800

    # Generate demo account tag compliance data
    account_tags = []
    for tag in _demo_account_tags:
        # Simulate random states for demo
        drawdown = round(random.uniform(0, 15), 2)
        cb_states = ["normal", "warning", "triggered"]
        weights = [0.7, 0.2, 0.1]
        cb_state = random.choices(cb_states, weights=weights)[0]

        account_tags.append(AccountTagCompliance(
            tag=tag,
            circuit_breaker_state=cb_state,
            drawdown_pct=drawdown,
            daily_halt_triggered=drawdown > 5.0,
            paused_strategies=random.randint(0, 3) if cb_state != "normal" else 0,
            last_check_utc=now
        ))

    # Determine overall status
    if any(t.circuit_breaker_state == "triggered" or t.daily_halt_triggered for t in account_tags):
        overall_status = "critical"
    elif any(t.circuit_breaker_state == "warning" for t in account_tags) or is_within_30min:
        overall_status = "warning"
    else:
        overall_status = "compliant"

    islamic = IslamicComplianceStatus(
        countdown_seconds=countdown_seconds if countdown_seconds > 0 else 0,
        force_close_at=force_close if is_within_60min else None,
        is_within_60min_window=is_within_60min,
        is_within_30min_window=is_within_30min,
        current_time_utc=now,
        active_positions_count=random.randint(0, 5) if is_within_60min else 0
    )

    logger.info(f"Compliance status: {overall_status}, accounts: {len(account_tags)}, islamic countdown: {countdown_seconds}s")

    return ComplianceResponse(
        account_tags=account_tags,
        islamic=islamic,
        overall_status=overall_status
    )


@router.get("/islamic-status", response_model=IslamicComplianceStatus)
async def get_islamic_status():
    """
    Get Islamic compliance countdown status.

    Returns countdown to daily force-close at 21:45 UTC.
    Shows warning when within 60 minutes, critical when within 30 minutes.
    """
    from datetime import timedelta

    now = datetime.now(timezone.utc)
    force_close_hour = 21
    force_close_minute = 45

    force_close_today = now.replace(hour=force_close_hour, minute=force_close_minute, second=0, microsecond=0)

    if now > force_close_today:
        force_close = force_close_today + timedelta(days=1)
    else:
        force_close = force_close_today

    countdown_seconds = int((force_close - now).total_seconds())
    is_within_60min = 0 <= countdown_seconds <= 3600
    is_within_30min = 0 <= countdown_seconds <= 1800

    import random
    return IslamicComplianceStatus(
        countdown_seconds=countdown_seconds if countdown_seconds > 0 else 0,
        force_close_at=force_close if is_within_60min else None,
        is_within_60min_window=is_within_60min,
        is_within_30min_window=is_within_30min,
        current_time_utc=now,
        active_positions_count=random.randint(0, 5) if is_within_60min else 0
    )


# =============================================================================
# Calendar Blackout Endpoints (Story 4.6)
# =============================================================================

class CalendarBlackoutResponse(BaseModel):
    """Calendar blackout windows response."""
    events: List[NewsItem] = Field(default_factory=list, description="Upcoming high-impact news events")
    blackouts: List[Dict[str, Any]] = Field(default_factory=list, description="Active blackout windows")


@router.get("/calendar/blackout", response_model=CalendarBlackoutResponse)
async def get_calendar_blackout():
    """
    Get calendar blackout windows and affected strategies.

    AC #3: Returns { events: [{ event_name, impact, datetime_utc, blackout_minutes }],
                    blackouts: [{ start_utc, end_utc, affected_strategies }] }
    """
    import random

    # Get high-impact events
    high_impact_events = [e for e in _calendar_events if e.impact == NewsImpact.HIGH]
    high_impact_events.sort(key=lambda e: e.event_time)

    # Generate demo blackout windows
    blackouts = []
    for tag in _demo_account_tags:
        if random.random() > 0.5:  # 50% chance of active blackout
            from datetime import timedelta
            blackout_start = datetime.now(timezone.utc) - timedelta(hours=1)
            blackout_end = datetime.now(timezone.utc) + timedelta(hours=random.randint(1, 3))

            blackouts.append({
                "start_utc": blackout_start.isoformat(),
                "end_utc": blackout_end.isoformat(),
                "affected_strategies": [f"strategy-{random.randint(1, 5)}" for _ in range(random.randint(1, 3))],
                "account_tag": tag,
                "reason": "high_impact_news"
            })

    logger.info(f"Calendar blackout: {len(high_impact_events)} events, {len(blackouts)} active blackouts")

    return CalendarBlackoutResponse(
        events=high_impact_events,
        blackouts=blackouts
    )
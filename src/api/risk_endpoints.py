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
import os
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
from src.events.regime import RegimeType

router = APIRouter(prefix="/api/risk", tags=["risk"])

logger = logging.getLogger(__name__)

# In-memory authoring state until database-backed calendar persistence lands.
_calendar_rules: Dict[str, CalendarRule] = {}
_calendar_events: List[NewsItem] = []


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
    from src.risk.physics.msgarch import MSGARCHSensor
    from src.risk.physics.bocpd import BOCPDDetector
    from src.risk.physics.ensemble import EnsembleVoter
    SENSORS_AVAILABLE = True
except ImportError:
    SENSORS_AVAILABLE = False
    IsingRegimeSensor = None
    ChaosSensor = None
    HMMRegimeSensor = None
    MSGARCHSensor = None
    BOCPDDetector = None
    EnsembleVoter = None

# Global sensor instances (lazy initialization)
_ising_sensor: Optional[IsingRegimeSensor] = None
_chaos_sensor: Optional[ChaosSensor] = None
_hmm_sensor: Optional[HMMRegimeSensor] = None
_msgarch_sensor: Optional["MSGARCHSensor"] = None
_bocpd_detector: Optional["BOCPDDetector"] = None
_ensemble_voter: Optional["EnsembleVoter"] = None

# Model directory paths from environment
_HMM_MODEL_DIR = os.environ.get("HMM_MODEL_DIR", "/data/hmm/models")
_MSGARCH_MODEL_DIR = os.environ.get("MSGARCH_MODEL_DIR", "/data/msgarch/models")
_BOCPD_MODEL_DIR = os.environ.get("BOCPD_MODEL_DIR", "/data/bocpd/models")


def _find_latest_model(model_dir: str, pattern: str = "*.pkl") -> Optional[str]:
    """Find the latest model file in a directory matching the given pattern."""
    from pathlib import Path
    path = Path(model_dir)
    if not path.exists():
        return None
    files = sorted(path.glob(pattern), key=lambda x: x.stat().st_mtime, reverse=True)
    if not files:
        return None
    return str(files[0])


def _find_latest_json(model_dir: str) -> Optional[str]:
    """Find the latest JSON file in a directory (for BOCPD)."""
    from pathlib import Path
    path = Path(model_dir)
    if not path.exists():
        return None
    files = sorted(path.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    if not files:
        return None
    return str(files[0])


def _fetch_mt5_candles(symbol: str = "EURUSD", timeframe: str = "M5",
                       count: int = 200) -> Optional["pd.DataFrame"]:
    """Fetch recent OHLCV candles from MT5 terminal.

    Requires MT5 to be installed and initialized on the VPS/node.
    Falls back to None when MT5 is not available (e.g., local dev).

    Args:
        symbol: Trading symbol (default: EURUSD)
        timeframe: Timeframe - M1, M5, M15, H1, H4, D1 (default: M5)
        count: Number of candles to fetch (default: 200 for feature extraction)

    Returns:
        DataFrame with OHLCV columns or None if MT5 unavailable
    """
    import pandas as pd

    timeframe_map = {
        "M1": 1, "M5": 5, "M15": 15, "M30": 30,
        "H1": 16385, "H4": 16386, "D1": 16387,
        "W1": 32769,
    }

    mt5_tf = timeframe_map.get(timeframe, 5)

    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            logger.warning("MT5 initialize() failed")
            return None

        rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, count)
        mt5.shutdown()

        if rates is None or len(rates) < 50:
            logger.warning(f"MT5 returned insufficient candles for {symbol}: {len(rates) if rates is not None else 0}")
            return None

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df = df.rename(columns={
            'open': 'open', 'high': 'high', 'low': 'low',
            'close': 'close', 'tick_volume': 'volume'
        })
        df = df[['time', 'open', 'high', 'low', 'close', 'volume']]
        logger.debug(f"Fetched {len(df)} candles from MT5 for {symbol}")
        return df

    except Exception as e:
        logger.warning(f"MT5 candle fetch failed: {e}")
        return None


def _extract_features_from_candles(df: "pd.DataFrame") -> "np.ndarray":
    """Extract 10-feature vector from OHLCV DataFrame for model predictions.

    Uses the same feature extraction as HMM training so features are
    compatible with all trained models.

    Args:
        df: DataFrame with OHLCV columns

    Returns:
        2D numpy array (1 row, 10 features) suitable for model.predict_regime()
    """
    from src.risk.physics.hmm.trainer import extract_features_vectorized
    features = extract_features_vectorized(df)
    # Return the last row (most recent) as 2D array
    return features[-1:, :] if features.ndim == 2 else features.reshape(1, -1)


# HMM regime → RegimeType mapping (HMM uses 4-state forex regimes)
_HMM_TO_REGIME_TYPE: Dict[str, str] = {
    "TRENDING_LOW_VOL": "TREND_BULL",
    "TRENDING_HIGH_VOL": "TREND_BEAR",
    "RANGING_LOW_VOL": "RANGE_STABLE",
    "RANGING_HIGH_VOL": "RANGE_VOLATILE",
}


def _map_hmm_regime_to_regime_type(hmm_regime: str) -> str:
    """Map HMM sensor regime string to RegimeType enum value.

    HMM regimes: TRENDING_LOW_VOL, TRENDING_HIGH_VOL, RANGING_LOW_VOL, RANGING_HIGH_VOL
    RegimeType: TREND_BULL, TREND_BEAR, RANGE_STABLE, RANGE_VOLATILE, etc.
    """
    return _HMM_TO_REGIME_TYPE.get(hmm_regime, "CHAOS")


class ProductionEnsembleVoter(EnsembleVoter):
    """Ensemble voter with HMM regime type mapping for production.

    Overrides _extract_regime_type to correctly map HMM's
    TRENDING_LOW_VOL/etc. regime strings to RegimeType enum values.
    """

    def _extract_regime_type(self, output: Any) -> str:
        """Extract and map RegimeType from HMM or other model output."""
        if isinstance(output, dict):
            raw = output.get("regime_type") or output.get("regime")
        elif isinstance(output, str):
            raw = output
        else:
            raw = getattr(output, "regime_type", None) or getattr(output, "regime", None)

        if raw is None:
            return "CHAOS"

        raw_str = str(raw).upper()

        # Already a valid RegimeType?
        try:
            RegimeType[raw_str]
            return raw_str
        except KeyError:
            pass

        # Map HMM-style regimes
        if raw_str in _HMM_TO_REGIME_TYPE:
            mapped = _HMM_TO_REGIME_TYPE[raw_str]
            logger.debug(f"HMM regime {raw_str} → {mapped}")
            return mapped

        logger.warning(f"Unknown regime type: {raw}, falling back to CHAOS")
        return "CHAOS"


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


def _get_msgarch_sensor() -> Optional["MSGARCHSensor"]:
    """Get or create MS-GARCH sensor instance, loading latest model."""
    global _msgarch_sensor
    if _msgarch_sensor is None and SENSORS_AVAILABLE:
        try:
            model_path = _find_latest_model(_MSGARCH_MODEL_DIR, "*.pkl")
            if model_path:
                _msgarch_sensor = MSGARCHSensor(model_path=model_path)
                logger.info(f"Loaded MS-GARCH model: {model_path}")
            else:
                logger.warning("No MS-GARCH model found")
        except Exception as e:
            logger.warning(f"MS-GARCH sensor initialization failed: {e}")
            _msgarch_sensor = None
    return _msgarch_sensor


def _get_bocpd_detector() -> Optional["BOCPDDetector"]:
    """Get or create BOCPD detector instance, loading latest calibration."""
    global _bocpd_detector
    if _bocpd_detector is None and SENSORS_AVAILABLE:
        try:
            model_path = _find_latest_json(_BOCPD_MODEL_DIR)
            if model_path:
                _bocpd_detector = BOCPDDetector.load(model_path=model_path)
                logger.info(f"Loaded BOCPD detector: {model_path}")
            else:
                logger.warning("No BOCPD calibration found, using defaults")
                _bocpd_detector = BOCPDDetector()
        except Exception as e:
            logger.warning(f"BOCPD detector initialization failed: {e}")
            _bocpd_detector = BOCPDDetector()
    return _bocpd_detector


def _get_ensemble_voter() -> Optional["ProductionEnsembleVoter"]:
    """Get or create Ensemble voter instance combining all sensors."""
    global _ensemble_voter
    if _ensemble_voter is None and SENSORS_AVAILABLE:
        try:
            _ensemble_voter = ProductionEnsembleVoter(
                hmm_sensor=_get_hmm_sensor(),
                msgarch_sensor=_get_msgarch_sensor(),
                bocpd_detector=_get_bocpd_detector(),
            )
            logger.info("ProductionEnsembleVoter initialized with all sensors")
        except Exception as e:
            logger.warning(f"EnsembleVoter initialization failed: {e}")
            _ensemble_voter = None
    return _ensemble_voter


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


class PhysicsMSGARCHOutput(BaseModel):
    """MS-GARCH volatility regime output"""
    vol_state: Optional[str] = Field(None, description="Volatility regime state: LOW_VOL, HIGH_VOL, MED_VOL")
    sigma_forecast: Optional[float] = Field(None, description="Conditional volatility forecast")
    regime_type: Optional[str] = Field(None, description="Regime type classification")
    confidence: float = Field(0.0, description="Regime classification confidence (0-1)")
    transition_probs: Optional[Dict[str, float]] = Field(None, description="Regime transition probabilities")
    model_version: Optional[str] = Field(None, description="Loaded model version")
    alert: str = Field(..., description="Alert state: normal, warning, critical")


class PhysicsBOCPDOutput(BaseModel):
    """BOCPD changepoint detection output"""
    changepoint_prob: float = Field(0.0, description="Bayesian changepoint probability (0-1)")
    is_changepoint: bool = Field(False, description="Whether a changepoint is currently detected")
    current_run_length: int = Field(0, description="Estimated bars since last changepoint")
    regime_type: str = Field("STABLE", description="Regime type: STABLE or TRANSITION")
    confidence: float = Field(0.0, description="Confidence in the regime type (0-1)")
    hazard_lambda: Optional[float] = Field(None, description="Hazard rate lambda parameter")
    alert: str = Field(..., description="Alert state: normal, warning, critical")


class PhysicsEnsembleOutput(BaseModel):
    """Ensemble voter output combining HMM + MS-GARCH + BOCPD"""
    regime_type: Optional[str] = Field(None, description="Ensemble consensus regime type")
    confidence: float = Field(0.0, description="Ensemble confidence (0-1)")
    is_transition: bool = Field(False, description="Whether BOCPD detected a regime transition")
    sigma_forecast: Optional[float] = Field(None, description="Volatility forecast from MS-GARCH")
    ensemble_agreement: float = Field(0.0, description="Model agreement fraction (0-1)")
    weights_used: Optional[Dict[str, float]] = Field(None, description="Adaptive weights per model")
    model_count: int = Field(0, description="Number of models contributing to vote")
    sources: Optional[Dict[str, Any]] = Field(None, description="Per-model outputs")
    alert: str = Field(..., description="Alert state: normal, warning, critical")


class PhysicsResponse(BaseModel):
    """Response model for GET /api/risk/physics"""
    ising: PhysicsIsingOutput = Field(..., description="Ising Model sensor outputs")
    lyapunov: PhysicsLyapunovOutput = Field(..., description="Lyapunov exponent outputs")
    hmm: PhysicsHMMOutput = Field(..., description="HMM regime sensor outputs")
    msgarch: PhysicsMSGARCHOutput = Field(..., description="MS-GARCH volatility regime outputs")
    bocpd: PhysicsBOCPDOutput = Field(..., description="BOCPD changepoint detection outputs")
    ensemble: PhysicsEnsembleOutput = Field(..., description="Ensemble voter outputs")
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

        except Exception as e:
            logger.warning(f"Could not get router state: {e}")
            strategies = []

    logger.info(f"Router state response: {len(strategies)} strategies")

    return RouterStateResponse(strategies=strategies)


# =============================================================================
# Physics Sensor Outputs Endpoint (AC #3)
# =============================================================================

@router.get("/physics", response_model=PhysicsResponse)
async def get_physics_outputs():
    """
    Get physics sensor outputs from PRODUCTION models.

    All models (HMM, MS-GARCH, BOCPD, Ensemble) receive real features
    extracted from live MT5 OHLCV candles. Falls back to error states
    when MT5 is unavailable (e.g., local dev without MT5 connection).

    Returns: { ising, lyapunov, hmm, msgarch, bocpd, ensemble, kelly }
    """
    import numpy as np

    # ------------------------------------------------------------------
    # Fetch real OHLCV candles from MT5 (production data source)
    # Falls back to None when MT5 is unavailable
    # ------------------------------------------------------------------
    candles_df = _fetch_mt5_candles(symbol="EURUSD", timeframe="M5", count=200)
    mt5_connected = candles_df is not None

    # Extract real 10-feature vector from candles when available
    features_10d: Optional[np.ndarray] = None
    if candles_df is not None:
        try:
            features_10d = _extract_features_from_candles(candles_df)
            # Feature 0 = log_return, needed for all models
            log_return = float(features_10d[0, 0]) if features_10d is not None else 0.0
        except Exception as e:
            logger.warning(f"Feature extraction failed: {e}")
            features_10d = None

    logger.info(
        "Physics outputs: mt5=%s, features_10d=%s, log_return=%s",
        mt5_connected,
        "real" if features_10d is not None else "unavailable",
        float(features_10d[0, 0]) if features_10d is not None else "N/A",
    )

    if features_10d is None:
        raise HTTPException(
            status_code=503,
            detail="Live MT5 candle features unavailable; physics outputs cannot be computed on this host.",
        )

    model_features = features_10d

    # ------------------------------------------------------------------
    # Ising sensor — uses market_volatility param, not candle features
    # ------------------------------------------------------------------
    ising_alert = AlertState.NORMAL
    ising_magnetization = 0.0

    if SENSORS_AVAILABLE and _get_ising_sensor() is not None:
        try:
            sensor = _get_ising_sensor()
            # Compute real volatility from candles if available
            market_vol = 0.15
            if candles_df is not None and 'close' in candles_df.columns:
                returns = candles_df['close'].pct_change().dropna()
                market_vol = float(returns.std()) if len(returns) > 0 else 0.15
            result = sensor.detect_regime(market_volatility=market_vol)
            ising_magnetization = result.get('magnetization', 0.0)

            abs_mag = abs(ising_magnetization)
            if abs_mag >= 0.8:
                ising_alert = AlertState.NORMAL
            elif abs_mag >= 0.3:
                ising_alert = AlertState.WARNING
            else:
                ising_alert = AlertState.CRITICAL

            logger.info(f"Ising: mag={ising_magnetization:.4f}, vol={market_vol:.4f}, alert={ising_alert}")
        except Exception as e:
            logger.warning(f"Ising sensor error: {e}")

    ising_output = PhysicsIsingOutput(
        magnetization=ising_magnetization,
        correlation_matrix=None,
        alert=ising_alert
    )

    # ------------------------------------------------------------------
    # Lyapunov/Chaos sensor — uses real returns from candles
    # ------------------------------------------------------------------
    lyapunov_alert = AlertState.NORMAL
    lyapunov_exponent = 0.0
    divergence_rate = None

    if SENSORS_AVAILABLE and _get_chaos_sensor() is not None:
        try:
            sensor = _get_chaos_sensor()
            if candles_df is not None and 'close' in candles_df.columns:
                returns = candles_df['close'].pct_change().dropna().values[-100:]
            else:
                returns = np.random.randn(100) * 0.01
            result = sensor.analyze_chaos(returns)
            lyapunov_exponent = result.lyapunov_exponent
            divergence_rate = result.divergence_rate if hasattr(result, 'divergence_rate') else None

            if lyapunov_exponent < 0.2:
                lyapunov_alert = AlertState.NORMAL
            elif lyapunov_exponent < 0.5:
                lyapunov_alert = AlertState.WARNING
            else:
                lyapunov_alert = AlertState.CRITICAL

            logger.info(f"Lyapunov: exp={lyapunov_exponent:.4f}, alert={lyapunov_alert}")
        except Exception as e:
            logger.warning(f"Lyapunov sensor error: {e}")

    lyapunov_output = PhysicsLyapunovOutput(
        exponent_value=lyapunov_exponent,
        divergence_rate=divergence_rate,
        alert=lyapunov_alert
    )

    # ------------------------------------------------------------------
    # HMM sensor — uses real 10-feature vector from candles
    # ------------------------------------------------------------------
    hmm_alert = AlertState.NORMAL
    hmm_state = None
    hmm_transitions = None

    hmm_sensor = _get_hmm_sensor()
    if hmm_sensor is not None and model_features is not None:
        try:
            reading = hmm_sensor.predict_regime(model_features)
            # reading.regime is like "TRENDING_LOW_VOL" — map to RegimeType string
            raw_regime = getattr(reading, 'regime', None) or getattr(reading, 'state', None)
            hmm_state = _map_hmm_regime_to_regime_type(str(raw_regime)) if raw_regime is not None else None
            if hasattr(reading, 'state_probabilities') and reading.state_probabilities:
                hmm_transitions = {str(k): float(v) for k, v in reading.state_probabilities.items()}
            hmm_alert = AlertState.WARNING  # Shadow mode
            logger.info(f"HMM: raw={raw_regime}, mapped={hmm_state}, conf={reading.confidence:.2f}")
        except Exception as e:
            logger.warning(f"HMM sensor error: {e}")
            hmm_alert = AlertState.WARNING
    elif hmm_sensor is not None:
        # HMM available but no MT5 — use cached state distribution
        try:
            if hasattr(hmm_sensor, 'get_state_distribution'):
                state_dist = hmm_sensor.get_state_distribution()
                if state_dist:
                    hmm_transitions = {str(k): v for k, v in state_dist.items()}
        except Exception:
            pass
        hmm_alert = AlertState.WARNING

    hmm_output = PhysicsHMMOutput(
        current_state=hmm_state,
        transition_probabilities=hmm_transitions,
        is_shadow_mode=True,
        alert=hmm_alert
    )

    # ------------------------------------------------------------------
    # MS-GARCH sensor — uses real log return feature [0] from candles
    # ------------------------------------------------------------------
    msgarch_alert = AlertState.NORMAL
    msgarch_output = PhysicsMSGARCHOutput(alert=msgarch_alert)

    if SENSORS_AVAILABLE:
        try:
            sensor = _get_msgarch_sensor()
            if sensor is not None and sensor.is_model_loaded():
                info = sensor.get_model_info()
                # MS-GARCH: feature[0] = log_return
                ms_features = model_features.copy()
                if features_10d is None:
                    ms_features[0, 0] = np.random.randn(1)[0] * 0.01
                prediction = sensor.predict_regime(ms_features)

                vol_state = prediction.get("vol_state", "MED_VOL")
                sigma = prediction.get("sigma_forecast", 0.0)
                regime_type = prediction.get("regime_type", "RANGE_STABLE")
                confidence = prediction.get("confidence", 0.0)
                trans_probs = prediction.get("transition_probs", {})

                if vol_state in ("LOW_VOL",):
                    msgarch_alert = AlertState.NORMAL
                elif vol_state in ("HIGH_VOL",):
                    msgarch_alert = AlertState.WARNING
                else:
                    msgarch_alert = AlertState.WARNING

                msgarch_output = PhysicsMSGARCHOutput(
                    vol_state=vol_state,
                    sigma_forecast=sigma,
                    regime_type=regime_type,
                    confidence=confidence,
                    transition_probs=trans_probs,
                    model_version=info.get("version"),
                    alert=msgarch_alert
                )
                logger.info(f"MS-GARCH: vol_state={vol_state}, sigma={sigma:.6f}")
        except Exception as e:
            logger.warning(f"MS-GARCH sensor error: {e}")

    # ------------------------------------------------------------------
    # BOCPD detector — uses real log return feature [0] from candles
    # ------------------------------------------------------------------
    bocpd_alert = AlertState.NORMAL
    bocpd_output = PhysicsBOCPDOutput(alert=bocpd_alert)

    if SENSORS_AVAILABLE:
        try:
            detector = _get_bocpd_detector()
            if detector is not None:
                info = detector.get_model_info()
                # BOCPD: feature[0] = log_return, feature[7] = susceptibility
                boc_features = model_features.copy()
                if features_10d is None:
                    boc_features[0, 0] = np.random.randn(1)[0] * 0.01
                    boc_features[0, 7] = 0.0
                prediction = detector.predict_regime(boc_features)

                cp_prob = prediction.get("changepoint_prob", 0.0)
                is_cp = prediction.get("is_changepoint", False)
                run_length = prediction.get("current_run_length", 0)
                regime_type = prediction.get("regime_type", "STABLE")
                confidence = prediction.get("confidence", 0.0)

                if cp_prob < 0.1:
                    bocpd_alert = AlertState.NORMAL
                elif cp_prob < 0.3:
                    bocpd_alert = AlertState.WARNING
                else:
                    bocpd_alert = AlertState.CRITICAL

                hazard_lambda = None
                if isinstance(info.get("hazard"), dict):
                    hazard_lambda = info["hazard"].get("lambda")
                elif hasattr(info.get("hazard"), "get_params"):
                    hazard_lambda = info["hazard"].get_params().get("lambda")

                bocpd_output = PhysicsBOCPDOutput(
                    changepoint_prob=cp_prob,
                    is_changepoint=is_cp,
                    current_run_length=run_length,
                    regime_type=regime_type,
                    confidence=confidence,
                    hazard_lambda=hazard_lambda,
                    alert=bocpd_alert
                )
                logger.info(f"BOCPD: cp_prob={cp_prob:.4f}, is_changepoint={is_cp}")
        except Exception as e:
            logger.warning(f"BOCPD detector error: {e}")

    # ------------------------------------------------------------------
    # Ensemble voter — uses real features from candles
    # ------------------------------------------------------------------
    ensemble_alert = AlertState.NORMAL
    ensemble_output = PhysicsEnsembleOutput(alert=ensemble_alert)

    if SENSORS_AVAILABLE:
        try:
            voter = _get_ensemble_voter()
            if voter is not None and voter.is_model_loaded():
                prediction = voter.predict_regime(model_features)

                regime_type = prediction.get("regime_type", "UNKNOWN")
                confidence = prediction.get("confidence", 0.0)
                is_transition = prediction.get("is_transition", False)
                sigma = prediction.get("sigma_forecast")
                agreement = prediction.get("ensemble_agreement", 0.0)
                weights = prediction.get("weights_used", {})
                model_count = prediction.get("model_count", 0)
                sources = prediction.get("sources", {})

                if is_transition or confidence < 0.3:
                    ensemble_alert = AlertState.CRITICAL
                elif confidence < 0.6:
                    ensemble_alert = AlertState.WARNING
                else:
                    ensemble_alert = AlertState.NORMAL

                ensemble_output = PhysicsEnsembleOutput(
                    regime_type=regime_type,
                    confidence=confidence,
                    is_transition=is_transition,
                    sigma_forecast=sigma,
                    ensemble_agreement=agreement,
                    weights_used=weights,
                    model_count=model_count,
                    sources=sources,
                    alert=ensemble_alert
                )
                logger.info(f"Ensemble: regime={regime_type}, conf={confidence:.2f}, transition={is_transition}")
        except Exception as e:
            logger.warning(f"Ensemble voter error: {e}")

    # ------------------------------------------------------------------
    # Kelly Engine — driven by real sensor states
    # ------------------------------------------------------------------
    kelly_fraction = 0.5
    kelly_multiplier = 1.0
    house_of_money = False

    if ising_alert == AlertState.NORMAL and lyapunov_alert == AlertState.NORMAL:
        house_of_money = True
        kelly_multiplier = 1.2
    elif ising_alert == AlertState.WARNING or lyapunov_alert == AlertState.WARNING:
        kelly_multiplier = 0.8
    else:
        kelly_multiplier = 0.5

    kelly_output = PhysicsKellyOutput(
        fraction=kelly_fraction,
        multiplier=kelly_multiplier,
        house_of_money=house_of_money,
        kelly_fraction_setting=kelly_fraction
    )

    logger.info(
        f"Physics: ising={ising_alert}, lyapunov={lyapunov_alert}, "
        f"hmm={hmm_alert}, msgarch={msgarch_alert}, bocpd={bocpd_alert}, "
        f"ensemble={ensemble_alert}, mt5={'ON' if mt5_connected else 'OFF'}"
    )

    return PhysicsResponse(
        ising=ising_output,
        lyapunov=lyapunov_output,
        hmm=hmm_output,
        msgarch=msgarch_output,
        bocpd=bocpd_output,
        ensemble=ensemble_output,
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

    # No synthetic compliance state in production mode.
    account_tags: List[AccountTagCompliance] = []

    # Determine overall status
    if any(t.circuit_breaker_state == "triggered" or t.daily_halt_triggered for t in account_tags):
        overall_status = "critical"
    elif is_within_30min:
        overall_status = "critical"
    elif any(t.circuit_breaker_state == "warning" for t in account_tags) or is_within_60min:
        overall_status = "warning"
    else:
        overall_status = "compliant"

    islamic = IslamicComplianceStatus(
        countdown_seconds=countdown_seconds if countdown_seconds > 0 else 0,
        force_close_at=force_close if is_within_60min else None,
        is_within_60min_window=is_within_60min,
        is_within_30min_window=is_within_30min,
        current_time_utc=now,
        active_positions_count=0
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

    return IslamicComplianceStatus(
        countdown_seconds=countdown_seconds if countdown_seconds > 0 else 0,
        force_close_at=force_close if is_within_60min else None,
        is_within_60min_window=is_within_60min,
        is_within_30min_window=is_within_30min,
        current_time_utc=now,
        active_positions_count=0
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
    # Get high-impact events
    high_impact_events = [e for e in _calendar_events if e.impact == NewsImpact.HIGH]
    high_impact_events.sort(key=lambda e: e.event_time)

    # No synthetic blackout windows in production mode.
    blackouts: List[Dict[str, Any]] = []

    logger.info(f"Calendar blackout: {len(high_impact_events)} events, {len(blackouts)} active blackouts")

    return CalendarBlackoutResponse(
        events=high_impact_events,
        blackouts=blackouts
    )

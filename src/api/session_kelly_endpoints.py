"""
Session Kelly Endpoints

Exposes Session Kelly modifier state via REST API for the UI.
Provides current Kelly modifiers per session and historical data.

Endpoints:
- GET /api/risk/kelly/current - Current Kelly modifiers per session
- GET /api/risk/kelly/history - Kelly modifier history for the day

Part of Story 4.10: Session-Scoped Kelly Modifiers
"""

import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from src.risk.sizing.session_kelly_modifiers import (
    SessionKellyModifiers,
    SessionKellyState,
    PremiumSessionAssault,
)
from src.router.sessions import SessionDetector, TradingSession

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/risk/kelly", tags=["risk", "kelly"])

# Singleton instances per account_id
_modifiers: dict[str, SessionKellyModifiers] = {}

# In-memory history for the day (reset on new day)
_history: List[Dict[str, Any]] = []
_last_history_date: Optional[str] = None


def get_modifiers(account_id: str = "default") -> SessionKellyModifiers:
    """Get or create SessionKellyModifiers instance for account."""
    if account_id not in _modifiers:
        _modifiers[account_id] = SessionKellyModifiers(account_id=account_id)
    return _modifiers[account_id]


def _record_to_history(state: SessionKellyState) -> None:
    """Record a state snapshot to history, resetting if new day."""
    global _history, _last_history_date
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    if _last_history_date != today:
        _history = []
        _last_history_date = today

    _history.append({
        "timestamp": state.last_updated.isoformat() if state.last_updated else datetime.now(timezone.utc).isoformat(),
        "current_session": state.current_session,
        "hmm_multiplier": state.hmm_multiplier,
        "reverse_hmm_multiplier": state.reverse_hmm_multiplier,
        "session_kelly_multiplier": state.session_kelly_multiplier,
        "is_house_money_active": state.is_house_money_active,
        "is_preservation_mode": state.is_preservation_mode,
        "daily_pnl_pct": state.daily_pnl_pct,
        "session_loss_counter": state.session_loss_counter,
        "premium_boost_active": state.premium_boost_active,
        "is_premium_session": state.is_premium_session,
        "premium_assault": state.premium_assault.value if state.premium_assault else None,
    })


def _get_all_session_modifiers(
    account_id: str = "default",
) -> List[Dict[str, Any]]:
    """
    Compute Kelly modifiers for all known trading sessions.

    Returns a list of session modifier states.
    """
    modifiers = get_modifiers(account_id)
    utc_now = datetime.now(timezone.utc)
    current_session = SessionDetector.detect_session(utc_now)

    # HMM thresholds
    profit_threshold = modifiers.hmm_profit_threshold  # +8% normal
    loss_threshold = modifiers.hmm_loss_threshold      # -10%
    premium_threshold = profit_threshold - modifiers.premium_threshold_boost  # +6% for premium

    sessions_data = []

    for session in TradingSession:
        if session == TradingSession.CLOSED:
            continue

        # Determine if this session is currently active
        is_active = session == current_session

        # Check if premium session
        is_premium, premium_assault = modifiers.is_premium_session(session, utc_now)

        # Determine status color
        if modifiers._is_preservation_mode and is_active:
            status = "CRITICAL"
        elif modifiers._is_house_money_active and is_active:
            status = "NORMAL"
        elif is_active and modifiers._session_loss_counter >= 4:
            status = "STRESS"
        elif is_active and modifiers._session_loss_counter >= 2:
            status = "WARNING"
        else:
            status = "NORMAL"

        # Effective threshold for this session
        effective_threshold = premium_threshold if is_premium else profit_threshold

        sessions_data.append({
            "name": session.value,
            "is_active": is_active,
            "is_premium": is_premium,
            "premium_assault": premium_assault.value if premium_assault else None,
            "kelly_fraction": 0.0,  # Base Kelly fraction (from settings, not computed here)
            "kelly_dollar": 0.0,    # Dollar Kelly (requires account balance)
            "house_money_threshold": effective_threshold,
            "status": status,
            "hmm_multiplier": modifiers._hmm_multiplier if is_active else 1.0,
            "reverse_hmm_multiplier": modifiers._reverse_hmm_multiplier if is_active else 1.0,
            "session_kelly_multiplier": modifiers.session_kelly_multiplier if is_active else 1.0,
            "session_loss_counter": modifiers.session_loss_counter if is_active else 0,
            "daily_pnl_pct": modifiers._hmm_multiplier,  # Current daily P&L % approximation
        })

    return sessions_data


# =============================================================================
# Request/Response Models
# =============================================================================

class SessionKellyResponse(BaseModel):
    """Response model for a single session's Kelly data."""
    name: str
    is_active: bool
    is_premium: bool
    premium_assault: Optional[str] = None
    kelly_fraction: float
    kelly_dollar: float
    house_money_threshold: float
    status: str
    hmm_multiplier: float
    reverse_hmm_multiplier: float
    session_kelly_multiplier: float
    session_loss_counter: int
    daily_pnl_pct: float


class KellyCurrentResponse(BaseModel):
    """Response model for GET /api/risk/kelly/current."""
    sessions: List[SessionKellyResponse]
    current_session: str
    timestamp: str


class KellyHistoryEntry(BaseModel):
    """Single history entry for Kelly modifier changes."""
    timestamp: str
    current_session: str
    hmm_multiplier: float
    reverse_hmm_multiplier: float
    session_kelly_multiplier: float
    is_house_money_active: bool
    is_preservation_mode: bool
    daily_pnl_pct: float
    session_loss_counter: int
    premium_boost_active: bool
    is_premium_session: bool
    premium_assault: Optional[str] = None


class KellyHistoryResponse(BaseModel):
    """Response model for GET /api/risk/kelly/history."""
    history: List[KellyHistoryEntry]
    count: int
    date: str


# =============================================================================
# Endpoints
# =============================================================================

@router.get("/current", response_model=KellyCurrentResponse)
async def get_current_kelly(
    account_id: str = Query(default="default", description="Trading account identifier"),
) -> KellyCurrentResponse:
    """
    Get current Kelly modifiers per session.

    Returns the current Kelly modifier state for all trading sessions,
    including house money status, premium session indicators, and
    London-NY overlap special threshold (+4% vs normal +8%).

    Args:
        account_id: Trading account identifier (default: "default")

    Returns:
        KellyCurrentResponse with sessions array containing per-session Kelly data
    """
    try:
        modifiers = get_modifiers(account_id)
        utc_now = datetime.now(timezone.utc)
        current_session = SessionDetector.detect_session(utc_now)

        # Get all session modifiers
        sessions_data = _get_all_session_modifiers(account_id)

        # Record current state to history
        state = modifiers.get_current_state()
        _record_to_history(state)

        return KellyCurrentResponse(
            sessions=[SessionKellyResponse(**s) for s in sessions_data],
            current_session=current_session.value,
            timestamp=utc_now.isoformat(),
        )

    except Exception as e:
        logger.error(f"Failed to get current Kelly modifiers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history", response_model=KellyHistoryResponse)
async def get_kelly_history(
    account_id: str = Query(default="default", description="Trading account identifier"),
    limit: int = Query(default=100, description="Maximum number of history entries", ge=1, le=1000),
) -> KellyHistoryResponse:
    """
    Get Kelly modifier history for the current day.

    Returns a chronological list of Kelly modifier state changes
    recorded throughout the trading day.

    Args:
        account_id: Trading account identifier (default: "default")
        limit: Maximum number of entries to return (default: 100, max: 1000)

    Returns:
        KellyHistoryResponse with history array of state snapshots
    """
    global _history

    try:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Filter history for today and apply limit
        today_history = [
            entry for entry in _history
            if entry["timestamp"].startswith(today)
        ][-limit:]

        return KellyHistoryResponse(
            history=[KellyHistoryEntry(**entry) for entry in today_history],
            count=len(today_history),
            date=today,
        )

    except Exception as e:
        logger.error(f"Failed to get Kelly history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/record")
async def record_kelly_state(
    account_id: str = Query(default="default", description="Trading account identifier"),
    daily_pnl_pct: float = Query(..., description="Daily P&L percentage"),
    is_win: bool = Query(..., description="Whether the last trade was a win"),
) -> SessionKellyState:
    """
    Record a trade result and update Kelly modifier state.

    This endpoint should be called after each trade to update
    the session loss counter and recompute modifiers.

    Args:
        account_id: Trading account identifier
        daily_pnl_pct: Current daily P&L as percentage
        is_win: Whether the last trade was a winner

    Returns:
        Updated SessionKellyState
    """
    try:
        modifiers = get_modifiers(account_id)
        utc_now = datetime.now(timezone.utc)
        current_session = SessionDetector.detect_session(utc_now)

        # Update with trade result
        state = modifiers.on_trade_result(is_win=is_win)

        # Recompute with current P&L
        state = modifiers.compute_session_kelly_modifier(
            daily_pnl_pct=daily_pnl_pct,
            current_session=current_session,
            utc_now=utc_now,
        )

        # Record to history
        _record_to_history(state)

        return state

    except Exception as e:
        logger.error(f"Failed to record Kelly state: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/session-close")
async def on_session_boundary(
    account_id: str = Query(default="default", description="Trading account identifier"),
) -> Dict[str, Any]:
    """
    Reset modifiers on session boundary.

    Called when a trading session ends to reset session-scoped
    modifiers while preserving HMM state (which is daily).

    Args:
        account_id: Trading account identifier

    Returns:
        Confirmation message
    """
    try:
        modifiers = get_modifiers(account_id)
        modifiers.on_session_close()
        modifiers.on_session_start(SessionDetector.get_current_session())

        return {
            "success": True,
            "message": "Session modifiers reset",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to reset session modifiers: {e}")
        raise HTTPException(status_code=500, detail=str(e))

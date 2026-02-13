"""
Session Management REST API Endpoints

Provides endpoints for:
- Current session information
- All session statuses
- Timezone conversion utilities
"""

from fastapi import APIRouter, HTTPException
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
import logging
from zoneinfo import ZoneInfo

from src.router.sessions import SessionDetector, TradingSession, get_current_session, is_market_open, get_next_session_time

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/sessions", tags=["sessions"])


# =============================================================================
# Request/Response Models
# =============================================================================

class SessionInfoResponse(BaseModel):
    """Session information response."""
    session: str = Field(..., description="Current trading session")
    utc_time: str = Field(..., description="Current UTC time in ISO format")
    next_session: Optional[str] = Field(None, description="Next session that will open")
    is_active: bool = Field(..., description="Whether any session is active")
    time_until_open: Optional[int] = Field(None, description="Minutes until next session opens")
    time_until_close: Optional[int] = Field(None, description="Minutes until current session closes")
    time_until_close_str: Optional[str] = Field(None, description="Human-readable time until close")


class SessionStatusResponse(BaseModel):
    """Individual session status."""
    active: bool = Field(..., description="Whether session is currently active")
    name: str = Field(..., description="Human-readable session name")


class AllSessionsResponse(BaseModel):
    """All sessions status response."""
    ASIAN: SessionStatusResponse
    LONDON: SessionStatusResponse
    NEW_YORK: SessionStatusResponse
    OVERLAP: SessionStatusResponse
    CLOSED: SessionStatusResponse


class TimezoneConvertRequest(BaseModel):
    """Timezone conversion request."""
    time_str: str = Field(..., description="ISO format time string", examples=["2026-02-12T10:00:00"])
    from_timezone: str = Field(..., description="IANA timezone to convert from", examples=["America/New_York"])
    to_timezone: str = Field(default="UTC", description="IANA timezone to convert to (default: UTC)")


class TimezoneConvertResponse(BaseModel):
    """Timezone conversion response."""
    original_time: str = Field(..., description="Original input time")
    original_timezone: str = Field(..., description="Original timezone")
    converted_time: str = Field(..., description="Converted time in ISO format")
    converted_timezone: str = Field(..., description="Target timezone")
    session_context: Optional[str] = Field(None, description="Trading session at converted time")


# =============================================================================
# Session Endpoints
# =============================================================================

@router.get("/current", response_model=SessionInfoResponse)
async def get_current_session_endpoint() -> SessionInfoResponse:
    """
    Get current trading session information.
    
    Returns CLOSED with is_active=false on weekends (Saturday/Sunday).

    Returns:
        Current active session, next session, and timing information (in minutes)

    Example:
        GET /api/sessions/current

        Response (weekday):
        {
            "session": "LONDON",
            "utc_time": "2026-02-12T10:00:00Z",
            "next_session": "OVERLAP",
            "is_active": true,
            "time_until_open": 150,
            "time_until_close": 180,
            "time_until_close_str": "3h 0m"
        }
        
        Response (weekend):
        {
            "session": "CLOSED",
            "utc_time": "2026-02-15T10:00:00Z",
            "next_session": "ASIAN",
            "is_active": false,
            "time_until_open": null,
            "time_until_close": null,
            "time_until_close_str": null
        }
    """
    try:
        utc_now = datetime.now(timezone.utc)
        info = SessionDetector.get_session_info(utc_now)

        return SessionInfoResponse(
            session=info.session.value,
            utc_time=utc_now.isoformat(),
            next_session=info.next_session.value if info.next_session else None,
            is_active=info.is_active,
            time_until_open=info.time_until_open,
            time_until_close=info.time_until_close,
            time_until_close_str=info.time_until_close_str
        )
    except Exception as e:
        logger.error(f"Error getting current session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get session info: {str(e)}")


@router.get("/all", response_model=AllSessionsResponse)
async def get_all_sessions_status() -> AllSessionsResponse:
    """
    Get status of all trading sessions.
    
    During London/NY overlap, only OVERLAP is marked active (LONDON/NEW_YORK forced inactive).
    This ensures only one session is active at any given time.
    CLOSED is marked active only when no other sessions are active.
    
    Returns CLOSED as active on weekends (Saturday/Sunday), with all other sessions inactive.

    Returns:
        Status of all sessions (ASIAN, LONDON, NEW_YORK, OVERLAP, CLOSED)

    Example (London/NY Overlap - 14:00 UTC Mon-Fri):
        GET /api/sessions/all

        Response:
        {
            "ASIAN": {"active": false, "name": "Asian Session"},
            "LONDON": {"active": false, "name": "London Session"},
            "NEW_YORK": {"active": false, "name": "New York Session"},
            "OVERLAP": {"active": true, "name": "London/NY Overlap"},
            "CLOSED": {"active": false, "name": "Market Closed"}
        }
    
    Example (London only - 10:00 UTC Mon-Fri):
        GET /api/sessions/all

        Response:
        {
            "ASIAN": {"active": false, "name": "Asian Session"},
            "LONDON": {"active": true, "name": "London Session"},
            "NEW_YORK": {"active": false, "name": "New York Session"},
            "OVERLAP": {"active": false, "name": "London/NY Overlap"},
            "CLOSED": {"active": false, "name": "Market Closed"}
        }
        
    Example (weekend):
        GET /api/sessions/all

        Response:
        {
            "ASIAN": {"active": false, "name": "Asian Session"},
            "LONDON": {"active": false, "name": "London Session"},
            "NEW_YORK": {"active": false, "name": "New York Session"},
            "OVERLAP": {"active": false, "name": "London/NY Overlap"},
            "CLOSED": {"active": true, "name": "Market Closed"}
        }
    """
    try:
        utc_now = datetime.now(timezone.utc)

        # Check if overlap is active first - this determines LONDON/NEW_YORK status
        overlap_active = SessionDetector.is_in_session("OVERLAP", utc_now)
        
        # Check individual sessions
        statuses = {}
        
        for session in TradingSession:
            if session == TradingSession.CLOSED:
                # CLOSED is active when no other sessions are active
                asian_active = SessionDetector.is_in_session("ASIAN", utc_now)
                london_active = SessionDetector.is_in_session("LONDON", utc_now)
                ny_active = SessionDetector.is_in_session("NEW_YORK", utc_now)
                is_active = not (asian_active or london_active or ny_active)
            elif session == TradingSession.OVERLAP:
                # OVERLAP is active when in overlap period
                is_active = overlap_active
            elif session in (TradingSession.LONDON, TradingSession.NEW_YORK):
                # During overlap, LONDON and NEW_YORK are forced inactive
                # Only active when in their respective sessions but NOT in overlap
                is_active = SessionDetector.is_in_session(session.value, utc_now) and not overlap_active
            else:
                # For other sessions (ASIAN), check if currently active
                is_active = SessionDetector.is_in_session(session.value, utc_now)
            
            session_name = SessionDetector.SESSIONS.get(session.value, {}).get("name", session.value)
            statuses[session.value] = SessionStatusResponse(
                active=is_active,
                name=session_name
            )

        return AllSessionsResponse(**statuses)
    except Exception as e:
        logger.error(f"Error getting all sessions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get session statuses: {str(e)}")


@router.post("/convert", response_model=TimezoneConvertResponse)
async def convert_timezone(request: TimezoneConvertRequest) -> TimezoneConvertResponse:
    """
    Convert time between timezones with session context.

    Args:
        request: Timezone conversion request with time_str, from_timezone, to_timezone

    Returns:
        Converted time with trading session context

    Example:
        POST /api/sessions/convert
        {
            "time_str": "2026-02-12T08:30:00",
            "from_timezone": "America/New_York",
            "to_timezone": "UTC"
        }

        Response:
        {
            "original_time": "2026-02-12T08:30:00",
            "original_timezone": "America/New_York",
            "converted_time": "2026-02-12T13:30:00Z",
            "converted_timezone": "UTC",
            "session_context": "LONDON"
        }
    """
    try:
        # Parse input time
        from datetime import datetime
        try:
            # Try parsing as ISO format
            if 'T' in request.time_str:
                input_time = datetime.fromisoformat(request.time_str.replace('Z', '+00:00'))
            else:
                # Try basic format
                input_time = datetime.fromisoformat(request.time_str)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid time format: {request.time_str}. Use ISO format (YYYY-MM-DDTHH:MM:SS)"
            )

        # Validate and create source timezone, fall back to UTC on invalid
        try:
            source_tz = ZoneInfo(request.from_timezone)
        except Exception:
            logger.warning(f"Invalid timezone '{request.from_timezone}', falling back to UTC")
            source_tz = ZoneInfo("UTC")
            request.from_timezone = "UTC"

        # Validate and create target timezone
        if request.to_timezone.upper() != "UTC":
            try:
                target_tz = ZoneInfo(request.to_timezone)
            except Exception:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid timezone: {request.to_timezone}"
                )
        else:
            target_tz = None

        # Make input time aware in source timezone
        converted_time = SessionDetector.convert_to_utc(input_time, request.from_timezone)

        # If target is not UTC, convert from UTC to target
        if target_tz is not None:
            converted_time = converted_time.astimezone(target_tz)

        # Get session context for converted time
        session_at_time = SessionDetector.detect_session(converted_time)

        return TimezoneConvertResponse(
            original_time=request.time_str,
            original_timezone=request.from_timezone,
            converted_time=converted_time.isoformat(),
            converted_timezone=request.to_timezone,
            session_context=session_at_time.value
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error converting timezone: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to convert timezone: {str(e)}")


@router.get("/check/{session_name}")
async def check_session_active(session_name: str) -> Dict[str, Any]:
    """
    Check if a specific session is currently active.
    
    Special handling for CLOSED session:
    - Returns is_active=true when no other session is active (i.e., market is closed)
    - Returns is_active=false when any trading session is active (market is open)
    - On weekends (Saturday/Sunday), CLOSED is always active since no trading sessions run
    
    For all other sessions (ASIAN, LONDON, NEW_YORK, OVERLAP):
    - Returns is_active=true during their respective trading hours on weekdays
    - Returns is_active=false outside trading hours and on weekends

    Args:
        session_name: Session name to check (ASIAN, LONDON, NEW_YORK, OVERLAP, CLOSED)

    Returns:
        Boolean indicating if session is active

    Example (weekday during London session):
        GET /api/sessions/check/LONDON

        Response:
        {
            "session": "LONDON",
            "is_active": true,
            "utc_time": "2026-02-12T10:00:00Z"
        }
        
    Example (weekend):
        GET /api/sessions/check/LONDON

        Response:
        {
            "session": "LONDON",
            "is_active": false,
            "utc_time": "2026-02-15T10:00:00Z"
        }
        
    Example (weekend - CLOSED is active):
        GET /api/sessions/check/CLOSED

        Response:
        {
            "session": "CLOSED",
            "is_active": true,
            "utc_time": "2026-02-15T10:00:00Z"
        }
        
    Example (weekday outside trading hours - CLOSED is active):
        GET /api/sessions/check/CLOSED

        Response:
        {
            "session": "CLOSED",
            "is_active": true,
            "utc_time": "2026-02-12T20:00:00Z"
        }
        
    Example (weekday during trading - CLOSED is not active):
        GET /api/sessions/check/CLOSED

        Response:
        {
            "session": "CLOSED",
            "is_active": false,
            "utc_time": "2026-02-12T10:00:00Z"
        }
    """
    try:
        session_upper = session_name.upper()
        
        # Special case: CLOSED session returns true when no other session is active
        if session_upper == "CLOSED":
            utc_now = datetime.now(timezone.utc)
            
            # Check if any trading session is active
            asian_active = SessionDetector.is_in_session("ASIAN", utc_now)
            london_active = SessionDetector.is_in_session("LONDON", utc_now)
            ny_active = SessionDetector.is_in_session("NEW_YORK", utc_now)
            
            # CLOSED is active when no trading sessions are active
            is_active = not (asian_active or london_active or ny_active)
            
            return {
                "session": "CLOSED",
                "is_active": is_active,
                "utc_time": utc_now.isoformat()
            }

        # Validate session name for other sessions
        try:
            target_session = TradingSession(session_upper)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid session name: {session_name}. Must be one of: {[s.value for s in TradingSession]}"
            )

        utc_now = datetime.now(timezone.utc)
        is_active = SessionDetector.is_in_session(session_upper, utc_now)

        return {
            "session": session_upper,
            "is_active": is_active,
            "utc_time": utc_now.isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to check session: {str(e)}")


@router.get("/time-window")
async def check_time_window(
    start_time: str,
    end_time: str,
    timezone_name: str = "UTC"
) -> Dict[str, Any]:
    """
    Check if current time falls within a specific time window.

    Used for ICT strategy validation.

    Args:
        start_time: Window start time in HH:MM format
        end_time: Window end time in HH:MM format
        timezone_name: IANA timezone for the window (default: UTC)

    Returns:
        Boolean indicating if current time is within window

    Example:
        GET /api/sessions/time-window?start_time=09:50&end_time=10:10&timezone_name=America/New_York

        Response:
        {
            "start_time": "09:50",
            "end_time": "10:10",
            "timezone": "America/New_York",
            "is_in_window": true,
            "utc_time": "2026-02-12T14:55:00Z"
        }
    """
    try:
        utc_now = datetime.now(timezone.utc)
        is_in_window = SessionDetector.is_in_time_window(
            utc_now,
            start_time,
            end_time,
            timezone_name
        )

        return {
            "start_time": start_time,
            "end_time": end_time,
            "timezone": timezone_name,
            "is_in_window": is_in_window,
            "utc_time": utc_now.isoformat()
        }
    except Exception as e:
        logger.error(f"Error checking time window: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to check time window: {str(e)}")


@router.get("/market-open")
async def check_market_open() -> Dict[str, Any]:
    """
    Check if the market is currently open.
    
    Returns is_open=false on weekends (Saturday/Sunday) regardless of time.

    Returns:
        Boolean indicating if any trading session is active

    Example (weekday trading hours):
        GET /api/sessions/market-open

        Response:
        {
            "is_open": true,
            "current_session": "LONDON",
            "utc_time": "2026-02-12T10:00:00Z"
        }
        
    Example (weekend):
        GET /api/sessions/market-open

        Response:
        {
            "is_open": false,
            "current_session": null,
            "utc_time": "2026-02-15T10:00:00Z"
        }
        
    Example (weekday outside trading hours):
        GET /api/sessions/market-open

        Response:
        {
            "is_open": false,
            "current_session": null,
            "utc_time": "2026-02-12T20:00:00Z"
        }
    """
    try:
        utc_now = datetime.now(timezone.utc)
        info = SessionDetector.get_session_info(utc_now)

        return {
            "is_open": info.is_active,
            "current_session": info.session.value if info.is_active else None,
            "utc_time": utc_now.isoformat()
        }
    except Exception as e:
        logger.error(f"Error checking market status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to check market status: {str(e)}")

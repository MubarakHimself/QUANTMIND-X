"""
Session Management REST API Endpoints

Provides endpoints for:
- Current session information
- All session statuses
- Timezone conversion utilities
"""

from fastapi import APIRouter, HTTPException
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
import logging
import asyncio
from zoneinfo import ZoneInfo
import gc
import os
from pathlib import Path

from src.router.session_detector import SessionDetector, TradingSession, get_current_session, is_market_open, get_next_session_time
import src.router.sessions as session_runtime
from src.agents.memory.compaction import SessionCompactor, CompactionConfig, CompactionResult

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/sessions", tags=["sessions"])


def _utc_now() -> datetime:
    """Use the canonical session runtime clock so session tests patch one source."""
    return session_runtime.datetime.now(timezone.utc)


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
# Session Compaction Models
# =============================================================================

class CompactionRequest(BaseModel):
    """Request to compact a session."""
    session_id: str = Field(..., description="Session ID to compact")
    context_limit: int = Field(default=10000, description="Token limit after compaction")
    preserve_recent: int = Field(default=4, description="Number of recent messages to preserve")


class CompactionResponse(BaseModel):
    """Response from session compaction."""
    session_id: str
    success: bool
    original_tokens: int
    compacted_tokens: int
    reduction_ratio: float
    summary: str
    duration_ms: float
    timestamp: str


class CleanupRequest(BaseModel):
    """Request to cleanup old sessions."""
    older_than_hours: int = Field(default=24, description="Remove sessions inactive longer than this")


class CleanupResponse(BaseModel):
    """Response from session cleanup."""
    removed_sessions: List[str]
    removed_count: int
    timestamp: str


class MemoryStatsResponse(BaseModel):
    """Response with memory statistics."""
    active_sessions: int
    total_memory_mb: float
    sessions: List[Dict[str, Any]]
    timestamp: str


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
        utc_now = _utc_now()
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
        utc_now = _utc_now()
        current_session = SessionDetector.detect_session(utc_now)
        statuses = {}
        
        for session in TradingSession:
            is_active = session == current_session
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
            utc_now = _utc_now()
            is_active = SessionDetector.detect_session(utc_now) == TradingSession.CLOSED
            
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

        utc_now = _utc_now()
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
        utc_now = _utc_now()
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
        utc_now = _utc_now()
        current_session = SessionDetector.detect_session(utc_now)

        return {
            "is_open": current_session != TradingSession.CLOSED,
            "current_session": current_session.value if current_session != TradingSession.CLOSED else None,
            "utc_time": utc_now.isoformat()
        }
    except Exception as e:
        logger.error(f"Error checking market status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to check market status: {str(e)}")


# =============================================================================
# Session Compaction Endpoints
# =============================================================================

# Global session compactor instance
_session_compactor: Optional[SessionCompactor] = None
_session_messages: Dict[str, List[Dict[str, Any]]] = {}


def get_session_compactor() -> SessionCompactor:
    """Get or create the global session compactor."""
    global _session_compactor
    if _session_compactor is None:
        config = CompactionConfig(
            context_limit=10000,
            preserve_recent_count=4,
            enable_background=False
        )
        _session_compactor = SessionCompactor(config=config)
    return _session_compactor


@router.post("/compaction/compact", response_model=CompactionResponse)
async def compact_session(request: CompactionRequest) -> CompactionResponse:
    """
    Compact a session's messages to reduce memory usage.

    Uses summarization to preserve context while reducing token count.
    Keeps recent messages intact and summarizes older ones.

    Example:
        POST /api/sessions/compaction/compact
        {
            "session_id": "chat_001",
            "context_limit": 10000,
            "preserve_recent": 4
        }

        Response:
        {
            "session_id": "chat_001",
            "success": true,
            "original_tokens": 15000,
            "compacted_tokens": 8500,
            "reduction_ratio": 0.43,
            "summary": "Previous discussion covered...",
            "duration_ms": 125.5,
            "timestamp": "2026-03-05T10:00:00Z"
        }
    """
    try:
        compactor = get_session_compactor()

        # Get session messages or create sample
        messages = _session_messages.get(request.session_id, [])

        if not messages:
            # Create sample messages for demonstration if none exist
            messages = [
                {"role": "system", "content": "You are a helpful trading assistant."},
                {"role": "user", "content": f"Analyze the current market conditions for session {request.session_id}."},
                {"role": "assistant", "content": "I'll analyze the market data for you."}
            ]
            _session_messages[request.session_id] = messages

        # Configure compactor based on request
        config = CompactionConfig(
            context_limit=request.context_limit,
            preserve_recent_count=request.preserve_recent
        )
        compactor.config = config

        # Perform compaction
        result = compactor.compact(messages)

        # Update stored messages
        _session_messages[request.session_id] = result.preserved_messages

        return CompactionResponse(
            session_id=request.session_id,
            success=True,
            original_tokens=result.original_tokens,
            compacted_tokens=result.compacted_tokens,
            reduction_ratio=result.reduction_ratio,
            summary=result.summary,
            duration_ms=result.duration_ms,
            timestamp=result.timestamp.isoformat()
        )
    except Exception as e:
        logger.error(f"Error compacting session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to compact session: {str(e)}")


@router.post("/compaction/cleanup", response_model=CleanupResponse)
async def cleanup_old_sessions(request: CleanupRequest) -> CleanupResponse:
    """
    Clean up old inactive sessions to free memory.

    Removes session data that hasn't been active for the specified duration.

    Example:
        POST /api/sessions/compaction/cleanup
        {
            "older_than_hours": 24
        }

        Response:
        {
            "removed_sessions": ["session_001", "session_002"],
            "removed_count": 2,
            "timestamp": "2026-03-05T10:00:00Z"
        }
    """
    try:
        # Get list of session files from common locations
        session_dirs = [
            Path("sessions"),
            Path(".sessions"),
            Path("/tmp/sessions"),
        ]

        removed_sessions = []
        cutoff = datetime.now() - timedelta(hours=request.older_than_hours)

        for session_dir in session_dirs:
            if not session_dir.exists():
                continue

            for session_file in session_dir.glob("*.json"):
                try:
                    # Check file modification time
                    mtime = datetime.fromtimestamp(session_file.stat().st_mtime)
                    if mtime < cutoff:
                        session_id = session_file.stem
                        # Remove old session file
                        session_file.unlink()
                        removed_sessions.append(session_id)
                        logger.info(f"Removed old session file: {session_file}")
                except Exception as e:
                    logger.warning(f"Failed to process session file {session_file}: {e}")

        # Also clean from in-memory sessions
        global _session_messages
        sessions_to_remove = [
            sid for sid, msgs in _session_messages.items()
            if len(msgs) == 0  # Empty sessions
        ]
        for sid in sessions_to_remove:
            del _session_messages[sid]
            removed_sessions.append(sid)

        # Force garbage collection
        gc.collect()

        return CleanupResponse(
            removed_sessions=removed_sessions,
            removed_count=len(removed_sessions),
            timestamp=datetime.now(timezone.utc).isoformat()
        )
    except Exception as e:
        logger.error(f"Error cleaning up sessions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cleanup sessions: {str(e)}")


@router.get("/compaction/stats", response_model=MemoryStatsResponse)
async def get_memory_stats() -> MemoryStatsResponse:
    """
    Get current memory statistics for all sessions.

    Returns memory usage and session information to help with
    memory management decisions.

    Example:
        GET /api/sessions/compaction/stats

        Response:
        {
            "active_sessions": 5,
            "total_memory_mb": 125.5,
            "sessions": [
                {
                    "session_id": "chat_001",
                    "message_count": 50,
                    "estimated_tokens": 15000
                }
            ],
            "timestamp": "2026-03-05T10:00:00Z"
        }
    """
    try:
        sessions = []
        total_memory = 0.0

        # Get in-memory session info
        for session_id, messages in _session_messages.items():
            # Estimate tokens (rough: 4 chars per token)
            total_chars = sum(len(str(m.get("content", ""))) for m in messages)
            estimated_tokens = total_chars // 4

            sessions.append({
                "session_id": session_id,
                "message_count": len(messages),
                "estimated_tokens": estimated_tokens,
                "memory_bytes": total_chars * 2  # Rough estimate
            })
            total_memory += total_chars * 2

        # Also check session files on disk
        session_dirs = [
            Path("sessions"),
            Path(".sessions"),
        ]

        for session_dir in session_dirs:
            if not session_dir.exists():
                continue

            for session_file in session_dir.glob("*.json"):
                session_id = session_file.stem
                # Skip if already in memory
                if any(s["session_id"] == session_id for s in sessions):
                    continue

                try:
                    size = session_file.stat().st_size
                    total_memory += size
                    sessions.append({
                        "session_id": session_id,
                        "message_count": 0,
                        "estimated_tokens": size // 8,
                        "memory_bytes": size,
                        "source": "disk"
                    })
                except Exception as e:
                    logger.warning(f"Failed to stat session file {session_file}: {e}")

        return MemoryStatsResponse(
            active_sessions=len(sessions),
            total_memory_mb=total_memory / (1024 * 1024),
            sessions=sessions,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
    except Exception as e:
        logger.error(f"Error getting memory stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get memory stats: {str(e)}")


@router.delete("/compaction/{session_id}")
async def clear_session(session_id: str) -> Dict[str, Any]:
    """
    Clear a specific session from memory.

    Removes all messages for the given session ID.

    Example:
        DELETE /api/sessions/compaction/chat_001

        Response:
        {
            "success": true,
            "session_id": "chat_001",
            "messages_removed": 50
        }
    """
    try:
        global _session_messages

        messages_removed = 0
        if session_id in _session_messages:
            messages_removed = len(_session_messages[session_id])
            del _session_messages[session_id]

        # Also try to remove session file
        session_files = [
            Path(f"sessions/{session_id}.json"),
            Path(f".sessions/{session_id}.json"),
        ]

        for session_file in session_files:
            if session_file.exists():
                session_file.unlink()
                logger.info(f"Removed session file: {session_file}")

        gc.collect()

        return {
            "success": True,
            "session_id": session_id,
            "messages_removed": messages_removed
        }
    except Exception as e:
        logger.error(f"Error clearing session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear session: {str(e)}")

"""
SSL REST API Endpoints.

Story 18.1: Per-Bot Consecutive Loss Counter & Paper Rotation

Provides REST API endpoints for:
- GET /api/ssl/state/{bot_id} — Current SSL state for a bot
- GET /api/ssl/paper_candidates — All bots in paper tier
- GET /api/ssl/recovery_candidates — TIER_1 paper bots eligible for recovery
- POST /api/ssl/evaluate/{bot_id} — Manually trigger SSL evaluation
"""

import logging
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.database.models import get_db_session
from src.risk.ssl import SSLCircuitBreaker, SSLState, BotTier


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ssl", tags=["SSL"])


# Pydantic request/response models

class SSLStateResponse(BaseModel):
    """Response model for SSL state endpoint."""
    bot_id: str = Field(..., description="Bot identifier")
    ssl_state: str = Field(..., description="SSL state (live/paper/recovery/retired)")
    consecutive_losses: int = Field(..., description="Current consecutive loss count")
    tier: Optional[str] = Field(None, description="Paper tier (TIER_1/TIER_2) if in paper")
    magic_number: Optional[str] = Field(None, description="MT5 magic number")
    recovery_win_count: int = Field(default=0, description="Recovery win count")
    paper_entry_timestamp: Optional[str] = Field(None, description="When bot entered paper tier")


class PaperCandidatesResponse(BaseModel):
    """Response model for paper candidates endpoint."""
    paper_bots: List[str] = Field(..., description="List of bot IDs in paper tier")
    count: int = Field(..., description="Number of paper bots")


class RecoveryCandidatesResponse(BaseModel):
    """Response model for recovery candidates endpoint."""
    recovery_candidates: List[str] = Field(..., description="List of bot IDs eligible for recovery")
    count: int = Field(..., description="Number of recovery candidates")


class SSLEvaluateRequest(BaseModel):
    """Request model for manual SSL evaluation."""
    magic_number: str = Field(..., description="MT5 magic number")
    is_win: bool = Field(..., description="Whether the trade was a win")


class SSLEvaluateResponse(BaseModel):
    """Response model for SSL evaluation."""
    bot_id: str = Field(..., description="Bot identifier")
    event_type: Optional[str] = Field(None, description="Event type if state transition occurred")
    previous_state: Optional[str] = Field(None, description="Previous SSL state")
    new_state: Optional[str] = Field(None, description="New SSL state")
    consecutive_losses: int = Field(..., description="Current consecutive loss count")
    transition_occurred: bool = Field(..., description="Whether a state transition occurred")


# Dependency
def get_ssl_circuit_breaker(db: Session = Depends(get_db_session)) -> SSLCircuitBreaker:
    """Dependency to get SSL circuit breaker instance."""
    return SSLCircuitBreaker(db_session=db)


@router.get("/state/{bot_id}", response_model=SSLStateResponse)
async def get_ssl_state(
    bot_id: str,
    ssl: SSLCircuitBreaker = Depends(get_ssl_circuit_breaker),
):
    """
    Get current SSL state for a bot.

    Returns the current SSL state, consecutive loss count, tier (if in paper),
    and related timestamps.
    """
    try:
        state = ssl.get_ssl_state(bot_id)
        consecutive_losses = ssl.get_consecutive_losses(bot_id)
        tier = ssl.get_tier(bot_id)

        from src.risk.ssl.state import SSLCircuitBreakerState
        state_mgr = SSLCircuitBreakerState(db_session=ssl.db_session)
        magic_number = state_mgr.get_magic_number(bot_id)
        recovery_win_count = state_mgr.get_recovery_win_count(bot_id)
        paper_entry_ts = state_mgr.get_paper_entry_timestamp(bot_id)

        return SSLStateResponse(
            bot_id=bot_id,
            ssl_state=state.value,
            consecutive_losses=consecutive_losses,
            tier=tier.value if tier else None,
            magic_number=magic_number,
            recovery_win_count=recovery_win_count,
            paper_entry_timestamp=paper_entry_ts.isoformat() if paper_entry_ts else None,
        )

    except Exception as e:
        logger.error(f"Error getting SSL state for bot {bot_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/paper_candidates", response_model=PaperCandidatesResponse)
async def get_paper_candidates(
    ssl: SSLCircuitBreaker = Depends(get_ssl_circuit_breaker),
):
    """
    Get all bots currently in paper tier.

    Returns a list of all bot IDs in paper trading (both TIER_1 and TIER_2).
    """
    try:
        paper_bots = ssl.get_paper_candidates()
        return PaperCandidatesResponse(
            paper_bots=paper_bots,
            count=len(paper_bots),
        )

    except Exception as e:
        logger.error(f"Error getting paper candidates: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recovery_candidates", response_model=RecoveryCandidatesResponse)
async def get_recovery_candidates(
    ssl: SSLCircuitBreaker = Depends(get_ssl_circuit_breaker),
):
    """
    Get TIER_1 paper bots eligible for recovery.

    Returns bots that:
    - Are in PAPER state
    - Are TIER_1 (have lived before)
    - Have 2+ consecutive paper wins
    """
    try:
        recovery_candidates = ssl.get_recovery_candidates()
        return RecoveryCandidatesResponse(
            recovery_candidates=recovery_candidates,
            count=len(recovery_candidates),
        )

    except Exception as e:
        logger.error(f"Error getting recovery candidates: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evaluate/{bot_id}", response_model=SSLEvaluateResponse)
async def evaluate_ssl(
    bot_id: str,
    request: SSLEvaluateRequest,
    ssl: SSLCircuitBreaker = Depends(get_ssl_circuit_breaker),
):
    """
    Manually trigger SSL evaluation for a bot.

    Evaluates a trade result (win/loss) and triggers any necessary state transitions.
    Returns the evaluation result including whether a transition occurred.
    """
    try:
        previous_state = ssl.get_ssl_state(bot_id)

        # Call on_trade_close which handles the full SSL evaluation
        event = ssl.on_trade_close(
            bot_id=bot_id,
            magic_number=request.magic_number,
            is_win=request.is_win,
        )

        new_state = ssl.get_ssl_state(bot_id)
        consecutive_losses = ssl.get_consecutive_losses(bot_id)

        return SSLEvaluateResponse(
            bot_id=bot_id,
            event_type=event.event_type.value if event else None,
            previous_state=previous_state.value if previous_state else None,
            new_state=new_state.value if new_state else None,
            consecutive_losses=consecutive_losses,
            transition_occurred=event is not None,
        )

    except Exception as e:
        logger.error(f"Error evaluating SSL for bot {bot_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

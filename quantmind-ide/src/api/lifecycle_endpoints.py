"""
Bot Lifecycle Stage Report API Endpoints

Provides REST API for viewing bot lifecycle stage progression and reports.
Bots have a 5-stage lifecycle: Born -> Backtest -> Paper -> Live -> Review

Each stage contains:
- Q1-Q20 structured Q&A answers
- Performance metrics (win rate, drawdown, PnL, Sharpe, etc.)
- Stage transition timestamps
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from enum import Enum

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/bots/lifecycle", tags=["bot-lifecycle"])


class LifecycleStage(str, Enum):
    """The 5 stages of bot lifecycle."""
    BORN = "Born"
    BACKTEST = "Backtest"
    PAPER = "Paper"
    LIVE = "Live"
    REVIEW = "Review"


class QAAnswer(BaseModel):
    """A single Q&A answer from the stage report."""
    question_id: str = Field(description="Question identifier, e.g., 'q1', 'q2', etc.")
    question: str = Field(description="The question text")
    answer: Any = Field(description="The answer value")
    passed: Optional[bool] = Field(default=None, description="Whether the answer passed criteria")


class StageMetrics(BaseModel):
    """Performance metrics for a lifecycle stage."""
    win_rate: Optional[float] = Field(default=None, description="Win rate percentage")
    drawdown: Optional[float] = Field(default=None, description="Maximum drawdown percentage")
    pnl: Optional[float] = Field(default=None, description="Profit/Loss")
    sharpe_ratio: Optional[float] = Field(default=None, description="Sharpe ratio")
    profit_factor: Optional[float] = Field(default=None, description="Profit factor")
    total_trades: Optional[int] = Field(default=None, description="Total number of trades")
    consecutive_losses: Optional[int] = Field(default=None, description="Consecutive losing trades")
    avg_win: Optional[float] = Field(default=None, description="Average win amount")
    avg_loss: Optional[float] = Field(default=None, description="Average loss amount")
    recovery_factor: Optional[float] = Field(default=None, description="Recovery factor")
    max_drawdown_duration: Optional[int] = Field(default=None, description="Max drawdown duration in days")


class StageReport(BaseModel):
    """Full report for a specific lifecycle stage."""
    stage: LifecycleStage = Field(description="The stage this report is for")
    entered_at: str = Field(description="ISO timestamp when bot entered this stage")
    exited_at: Optional[str] = Field(default=None, description="ISO timestamp when bot exited this stage")
    q1_q20_answers: List[QAAnswer] = Field(default_factory=list, description="Q1-Q20 structured answers")
    metrics: StageMetrics = Field(default_factory=StageMetrics, description="Stage performance metrics")
    decline_recovery_status: Optional[str] = Field(
        default=None,
        description="Decline/recovery loop status if applicable: 'none', 'declining', 'recovering', 'recovered'"
    )
    notes: Optional[str] = Field(default=None, description="Additional notes about the stage")


class BotLifecycle(BaseModel):
    """Complete lifecycle data for a bot."""
    bot_id: str = Field(description="Unique bot identifier")
    current_stage: LifecycleStage = Field(description="Current lifecycle stage")
    stage_history: List[StageReport] = Field(default_factory=list, description="History of all stage reports")
    current_report: StageReport = Field(description="Report for the current stage")
    created_at: Optional[str] = Field(default=None, description="ISO timestamp when bot was created")
    updated_at: Optional[str] = Field(default=None, description="ISO timestamp of last update")


class LifecycleStats(BaseModel):
    """Statistics across all bots for the lifecycle panel."""
    total_bots: int = Field(description="Total number of bots")
    bots_by_stage: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of bots in each stage, e.g., {'Born': 5, 'Backtest': 10, ...}"
    )
    promotions_today: int = Field(description="Number of promotions today")
    demotions_today: int = Field(description="Number of demotions today (quarantine, etc.)")
    in_recovery: int = Field(description="Number of bots currently in recovery")
    next_check: str = Field(description="Next scheduled lifecycle check time")


# =============================================================================
# API Endpoints
# =============================================================================

@router.get("/{bot_id}", response_model=BotLifecycle)
async def get_bot_lifecycle(bot_id: str) -> BotLifecycle:
    """
    Get complete lifecycle data for a bot.

    Returns:
    - bot_id: Unique bot identifier
    - current_stage: Current lifecycle stage (Born/Backtest/Paper/Live/Review)
    - stage_history: List of all stage reports in chronological order
    - current_report: Report for the current stage with Q1-Q20 answers and metrics
    """
    # Database not wired yet - return 503
    raise HTTPException(
        status_code=503,
        detail="Lifecycle data not available. Database not wired."
    )


@router.get("/{bot_id}/stage/{stage}", response_model=StageReport)
async def get_bot_stage_report(bot_id: str, stage: LifecycleStage) -> StageReport:
    """
    Get the full report for a specific lifecycle stage of a bot.

    Args:
    - bot_id: Unique bot identifier
    - stage: The stage to retrieve (Born/Backtest/Paper/Live/Review)

    Returns:
    - Full stage report including Q1-Q20 answers and metrics
    """
    # Database not wired yet - return 503
    raise HTTPException(
        status_code=503,
        detail="Lifecycle data not available. Database not wired."
    )


@router.get("/stats/overview", response_model=LifecycleStats)
async def get_lifecycle_stats() -> LifecycleStats:
    """
    Get lifecycle statistics across all bots.

    Returns aggregated statistics including:
    - Total bot counts by stage
    - Promotions/demotions today
    - Bots in recovery status
    - Next scheduled check time
    """
    # Database not wired yet - return 503
    raise HTTPException(
        status_code=503,
        detail="Lifecycle statistics not available. Database not wired."
    )


@router.get("/")
async def list_bots_with_lifecycle(
    stage: Optional[LifecycleStage] = Query(default=None, description="Filter by current stage"),
    limit: int = Query(default=50, ge=1, le=200, description="Max results to return"),
    offset: int = Query(default=0, ge=0, description="Number of results to skip")
) -> Dict[str, Any]:
    """
    List all bots with their lifecycle summary.

    Args:
    - stage: Optional filter by current lifecycle stage
    - limit: Maximum number of results (default 50, max 200)
    - offset: Number of results to skip for pagination

    Returns:
    - List of bot lifecycle summaries
    - Total count for pagination
    """
    # Database not wired yet - return 503
    raise HTTPException(
        status_code=503,
        detail="Lifecycle data not available. Database not wired."
    )

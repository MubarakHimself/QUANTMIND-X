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
# Mock Data Store (replace with actual database calls)
# =============================================================================

# In-memory mock data for demonstration
_mock_lifecycle_data: Dict[str, BotLifecycle] = {}


def _get_mock_lifecycle(bot_id: str) -> Optional[BotLifecycle]:
    """
    DEPRECATED: Mock data source removed from production.
    This function now always returns None.
    Replace with actual database query.
    """
    return None


def _init_mock_data():
    """Initialize mock data for testing. Remove in production."""
    global _mock_lifecycle_data

    now = datetime.now(timezone.utc).isoformat()

    # Example bot with full lifecycle history
    _mock_lifecycle_data["bot-alpha-001"] = BotLifecycle(
        bot_id="bot-alpha-001",
        current_stage=LifecycleStage.LIVE,
        created_at="2025-06-01T00:00:00Z",
        updated_at=now,
        stage_history=[
            StageReport(
                stage=LifecycleStage.BORN,
                entered_at="2025-06-01T00:00:00Z",
                exited_at="2025-06-15T12:00:00Z",
                q1_q20_answers=[
                    QAAnswer(question_id="q1", question="Strategy documented?", answer=True, passed=True),
                    QAAnswer(question_id="q2", question="Risk parameters defined?", answer=True, passed=True),
                    QAAnswer(question_id="q3", question="Entry signals coded?", answer=True, passed=True),
                    QAAnswer(question_id="q4", question="Exit signals coded?", answer=True, passed=True),
                    QAAnswer(question_id="q5", question="Position sizing implemented?", answer=True, passed=True),
                    QAAnswer(question_id="q6", question="MT5 connector ready?", answer=True, passed=True),
                    QAAnswer(question_id="q7", question="Backtest harness ready?", answer=True, passed=True),
                    QAAnswer(question_id="q8", question="No critical errors in code?", answer=True, passed=True),
                    QAAnswer(question_id="q9", question="Min 20 trades threshold met?", answer=True, passed=True),
                    QAAnswer(question_id="q10", question="Win rate >= 50%?", answer=True, passed=True),
                    QAAnswer(question_id="q11", question="Avg trade duration reasonable?", answer=True, passed=True),
                    QAAnswer(question_id="q12", question="Max lot size defined?", answer=True, passed=True),
                    QAAnswer(question_id="q13", question="Session filter configured?", answer=True, passed=True),
                    QAAnswer(question_id="q14", question="News filter active?", answer=True, passed=True),
                    QAAnswer(question_id="q15", question="Spread filter configured?", answer=True, passed=True),
                    QAAnswer(question_id="q16", question="Slippage tolerance set?", answer=True, passed=True),
                    QAAnswer(question_id="q17", question="Broker compatible?", answer=True, passed=True),
                    QAAnswer(question_id="q18", question="Timezone configured?", answer=True, passed=True),
                    QAAnswer(question_id="q19", question="Logging enabled?", answer=True, passed=True),
                    QAAnswer(question_id="q20", question="Ready for backtesting?", answer=True, passed=True),
                ],
                metrics=StageMetrics(
                    win_rate=0.0,
                    total_trades=0,
                    consecutive_losses=0
                ),
                notes="Bot born from strategy alpha-v1"
            ),
            StageReport(
                stage=LifecycleStage.BACKTEST,
                entered_at="2025-06-15T12:00:00Z",
                exited_at="2025-07-15T00:00:00Z",
                q1_q20_answers=[
                    QAAnswer(question_id="q1", question="Min 50 trades completed?", answer=True, passed=True),
                    QAAnswer(question_id="q2", question="Win rate >= 55%?", answer=True, passed=True),
                    QAAnswer(question_id="q3", question="Sharpe ratio >= 1.5?", answer=True, passed=True),
                    QAAnswer(question_id="q4", question="Max drawdown < 15%?", answer=True, passed=True),
                    QAAnswer(question_id="q5", question="Profit factor > 1.5?", answer=True, passed=True),
                    QAAnswer(question_id="q6", question="Min 30 days in backtest?", answer=True, passed=True),
                    QAAnswer(question_id="q7", question="Consistent daily returns?", answer=True, passed=True),
                    QAAnswer(question_id="q8", question="No critical drawdown events?", answer=True, passed=True),
                    QAAnswer(question_id="q9", question="Risk/reward ratio acceptable?", answer=True, passed=True),
                    QAAnswer(question_id="q10", question="Spread impact assessed?", answer=True, passed=True),
                    QAAnswer(question_id="q11", question="Slippage impact assessed?", answer=True, passed=True),
                    QAAnswer(question_id="q12", question="Weekend gap risk acceptable?", answer=True, passed=True),
                    QAAnswer(question_id="q13", question="News event risk assessed?", answer=True, passed=True),
                    QAAnswer(question_id="q14", question="Holiday trading risk assessed?", answer=True, passed=True),
                    QAAnswer(question_id="q15", question="Correlated pairs risk assessed?", answer=True, passed=True),
                    QAAnswer(question_id="q16", question="Drawdown recovery time acceptable?", answer=True, passed=True),
                    QAAnswer(question_id="q17", question="Equity curve stable?", answer=True, passed=True),
                    QAAnswer(question_id="q18", question="Monte Carlo passed?", answer=True, passed=True),
                    QAAnswer(question_id="q19", question="Walk-forward analysis passed?", answer=True, passed=True),
                    QAAnswer(question_id="q20", question="Ready for paper trading?", answer=True, passed=True),
                ],
                metrics=StageMetrics(
                    win_rate=58.5,
                    drawdown=8.2,
                    pnl=12500.00,
                    sharpe_ratio=2.1,
                    profit_factor=1.8,
                    total_trades=187,
                    consecutive_losses=3,
                    avg_win=125.00,
                    avg_loss=85.00,
                    recovery_factor=3.2
                ),
                notes="Excellent backtest results across 3 market regimes"
            ),
            StageReport(
                stage=LifecycleStage.PAPER,
                entered_at="2025-07-15T00:00:00Z",
                exited_at="2025-09-15T00:00:00Z",
                q1_q20_answers=[
                    QAAnswer(question_id="q1", question="Min 100 trades in paper?", answer=True, passed=True),
                    QAAnswer(question_id="q2", question="Win rate >= 55%?", answer=True, passed=True),
                    QAAnswer(question_id="q3", question="Sharpe ratio >= 1.5?", answer=True, passed=True),
                    QAAnswer(question_id="q4", question="Max drawdown < 12%?", answer=True, passed=True),
                    QAAnswer(question_id="q5", question="Profit factor > 1.5?", answer=True, passed=True),
                    QAAnswer(question_id="q6", question="Min 60 days in paper?", answer=True, passed=True),
                    QAAnswer(question_id="q7", question="Live spread execution similar to backtest?", answer=True, passed=True),
                    QAAnswer(question_id="q8", question="Slippage within expectations?", answer=True, passed=True),
                    QAAnswer(question_id="q9", question="Execution latency acceptable?", answer=True, passed=True),
                    QAAnswer(question_id="q10", question="Broker connectivity stable?", answer=True, passed=True),
                    QAAnswer(question_id="q11", question="No requotes issues?", answer=True, passed=True),
                    QAAnswer(question_id="q12", question="Order fills match signals?", answer=True, passed=True),
                    QAAnswer(question_id="q13", question="Platform stability confirmed?", answer=True, passed=True),
                    QAAnswer(question_id="q14", question="Weekend positions managed correctly?", answer=True, passed=True),
                    QAAnswer(question_id="q15", question="News events handled correctly?", answer=True, passed=True),
                    QAAnswer(question_id="q16", question="Session transitions smooth?", answer=True, passed=True),
                    QAAnswer(question_id="q17", question="Drawdown matches backtest expectations?", answer=True, passed=True),
                    QAAnswer(question_id="q18", question="P&L matches backtest ratio?", answer=True, passed=True),
                    QAAnswer(question_id="q19", question="Risk management working correctly?", answer=True, passed=True),
                    QAAnswer(question_id="q20", question="Ready for live trading?", answer=True, passed=True),
                ],
                metrics=StageMetrics(
                    win_rate=56.8,
                    drawdown=10.5,
                    pnl=8200.00,
                    sharpe_ratio=1.85,
                    profit_factor=1.65,
                    total_trades=245,
                    consecutive_losses=4,
                    avg_win=118.00,
                    avg_loss=82.00,
                    recovery_factor=2.8,
                    max_drawdown_duration=12
                ),
                notes="Paper trading results closely match backtest expectations"
            ),
            StageReport(
                stage=LifecycleStage.LIVE,
                entered_at="2025-09-15T00:00:00Z",
                exited_at=None,
                q1_q20_answers=[
                    QAAnswer(question_id="q1", question="Min 100 live trades completed?", answer=True, passed=True),
                    QAAnswer(question_id="q2", question="Win rate >= 55%?", answer=True, passed=True),
                    QAAnswer(question_id="q3", question="Sharpe ratio >= 1.5?", answer=True, passed=True),
                    QAAnswer(question_id="q4", question="Max drawdown < 10%?", answer=True, passed=True),
                    QAAnswer(question_id="q5", question="Profit factor > 1.5?", answer=True, passed=True),
                    QAAnswer(question_id="q6", question="Min 60 days live?", answer=True, passed=True),
                    QAAnswer(question_id="q7", question="Execution quality maintained?", answer=True, passed=True),
                    QAAnswer(question_id="q8", question="No major outages?", answer=True, passed=True),
                    QAAnswer(question_id="q9", question="Realized slippage within bounds?", answer=True, passed=True),
                    QAAnswer(question_id="q10", question="Commission structure as expected?", answer=True, passed=True),
                    QAAnswer(question_id="q11", question="Funding/withdrawal process working?", answer=True, passed=True),
                    QAAnswer(question_id="q12", question="Account integrity confirmed?", answer=True, passed=True),
                    QAAnswer(question_id="q13", question="Risk limits respected?", answer=True, passed=True),
                    QAAnswer(question_id="q14", question="Drawdown recovery successful?", answer=True, passed=True),
                    QAAnswer(question_id="q15", question="Profit split conditions met?", answer=True, passed=True),
                    QAAnswer(question_id="q16", question="Evaluation metrics on track?", answer=True, passed=True),
                    QAAnswer(question_id="q17", question="Psychological performance stable?", answer=True, passed=True),
                    QAAnswer(question_id="q18", question="Strategy still relevant?", answer=True, passed=True),
                    QAAnswer(question_id="q19", question="Market regime adaptation adequate?", answer=True, passed=True),
                    QAAnswer(question_id="q20", question="Recommended for Review?", answer=True, passed=True),
                ],
                metrics=StageMetrics(
                    win_rate=57.2,
                    drawdown=7.8,
                    pnl=15200.00,
                    sharpe_ratio=1.95,
                    profit_factor=1.72,
                    total_trades=312,
                    consecutive_losses=3,
                    avg_win=122.00,
                    avg_loss=79.00,
                    recovery_factor=3.1,
                    max_drawdown_duration=8
                ),
                decline_recovery_status="none",
                notes="Strong live performance. Bot qualifies for Review stage."
            ),
        ],
        current_report=StageReport(
            stage=LifecycleStage.LIVE,
            entered_at="2025-09-15T00:00:00Z",
            exited_at=None,
            q1_q20_answers=[
                QAAnswer(question_id="q1", question="Min 100 live trades completed?", answer=True, passed=True),
                QAAnswer(question_id="q2", question="Win rate >= 55%?", answer=True, passed=True),
                QAAnswer(question_id="q3", question="Sharpe ratio >= 1.5?", answer=True, passed=True),
                QAAnswer(question_id="q4", question="Max drawdown < 10%?", answer=True, passed=True),
                QAAnswer(question_id="q5", question="Profit factor > 1.5?", answer=True, passed=True),
                QAAnswer(question_id="q6", question="Min 60 days live?", answer=True, passed=True),
                QAAnswer(question_id="q7", question="Execution quality maintained?", answer=True, passed=True),
                QAAnswer(question_id="q8", question="No major outages?", answer=True, passed=True),
                QAAnswer(question_id="q9", question="Realized slippage within bounds?", answer=True, passed=True),
                QAAnswer(question_id="q10", question="Commission structure as expected?", answer=True, passed=True),
                QAAnswer(question_id="q11", question="Funding/withdrawal process working?", answer=True, passed=True),
                QAAnswer(question_id="q12", question="Account integrity confirmed?", answer=True, passed=True),
                QAAnswer(question_id="q13", question="Risk limits respected?", answer=True, passed=True),
                QAAnswer(question_id="q14", question="Drawdown recovery successful?", answer=True, passed=True),
                QAAnswer(question_id="q15", question="Profit split conditions met?", answer=True, passed=True),
                QAAnswer(question_id="q16", question="Evaluation metrics on track?", answer=True, passed=True),
                QAAnswer(question_id="q17", question="Psychological performance stable?", answer=True, passed=True),
                QAAnswer(question_id="q18", question="Strategy still relevant?", answer=True, passed=True),
                QAAnswer(question_id="q19", question="Market regime adaptation adequate?", answer=True, passed=True),
                QAAnswer(question_id="q20", question="Recommended for Review?", answer=True, passed=True),
            ],
            metrics=StageMetrics(
                win_rate=57.2,
                drawdown=7.8,
                pnl=15200.00,
                sharpe_ratio=1.95,
                profit_factor=1.72,
                total_trades=312,
                consecutive_losses=3,
                avg_win=122.00,
                avg_loss=79.00,
                recovery_factor=3.1,
                max_drawdown_duration=8
            ),
            decline_recovery_status="none"
        )
    )

    # Example bot in Backtest stage
    _mock_lifecycle_data["bot-beta-002"] = BotLifecycle(
        bot_id="bot-beta-002",
        current_stage=LifecycleStage.BACKTEST,
        created_at="2025-11-01T00:00:00Z",
        updated_at=now,
        stage_history=[
            StageReport(
                stage=LifecycleStage.BORN,
                entered_at="2025-11-01T00:00:00Z",
                exited_at="2025-11-20T00:00:00Z",
                q1_q20_answers=[
                    QAAnswer(question_id="q1", question="Strategy documented?", answer=True, passed=True),
                    QAAnswer(question_id="q2", question="Risk parameters defined?", answer=True, passed=True),
                    QAAnswer(question_id="q3", question="Entry signals coded?", answer=True, passed=True),
                    QAAnswer(question_id="q4", question="Exit signals coded?", answer=True, passed=True),
                    QAAnswer(question_id="q5", question="Position sizing implemented?", answer=True, passed=True),
                    QAAnswer(question_id="q6", question="MT5 connector ready?", answer=True, passed=True),
                    QAAnswer(question_id="q7", question="Backtest harness ready?", answer=True, passed=True),
                    QAAnswer(question_id="q8", question="No critical errors in code?", answer=True, passed=True),
                    QAAnswer(question_id="q9", question="Min 20 trades threshold met?", answer=True, passed=True),
                    QAAnswer(question_id="q10", question="Win rate >= 50%?", answer=True, passed=True),
                    QAAnswer(question_id="q11", question="Avg trade duration reasonable?", answer=True, passed=True),
                    QAAnswer(question_id="q12", question="Max lot size defined?", answer=True, passed=True),
                    QAAnswer(question_id="q13", question="Session filter configured?", answer=True, passed=True),
                    QAAnswer(question_id="q14", question="News filter active?", answer=False, passed=False),
                    QAAnswer(question_id="q15", question="Spread filter configured?", answer=True, passed=True),
                    QAAnswer(question_id="q16", question="Slippage tolerance set?", answer=True, passed=True),
                    QAAnswer(question_id="q17", question="Broker compatible?", answer=True, passed=True),
                    QAAnswer(question_id="q18", question="Timezone configured?", answer=True, passed=True),
                    QAAnswer(question_id="q19", question="Logging enabled?", answer=True, passed=True),
                    QAAnswer(question_id="q20", question="Ready for backtesting?", answer=True, passed=True),
                ],
                metrics=StageMetrics(
                    win_rate=0.0,
                    total_trades=0,
                    consecutive_losses=0
                ),
                notes="Bot born with news filter needing configuration"
            ),
            StageReport(
                stage=LifecycleStage.BACKTEST,
                entered_at="2025-11-20T00:00:00Z",
                exited_at=None,
                q1_q20_answers=[
                    QAAnswer(question_id="q1", question="Min 50 trades completed?", answer=True, passed=True),
                    QAAnswer(question_id="q2", question="Win rate >= 55%?", answer=False, passed=False),
                    QAAnswer(question_id="q3", question="Sharpe ratio >= 1.5?", answer=False, passed=False),
                    QAAnswer(question_id="q4", question="Max drawdown < 15%?", answer=True, passed=True),
                    QAAnswer(question_id="q5", question="Profit factor > 1.5?", answer=False, passed=False),
                    QAAnswer(question_id="q6", question="Min 30 days in backtest?", answer=True, passed=True),
                    QAAnswer(question_id="q7", question="Consistent daily returns?", answer=False, passed=False),
                    QAAnswer(question_id="q8", question="No critical drawdown events?", answer=True, passed=True),
                    QAAnswer(question_id="q9", question="Risk/reward ratio acceptable?", answer=True, passed=True),
                    QAAnswer(question_id="q10", question="Spread impact assessed?", answer=True, passed=True),
                    QAAnswer(question_id="q11", question="Slippage impact assessed?", answer=True, passed=True),
                    QAAnswer(question_id="q12", question="Weekend gap risk acceptable?", answer=True, passed=True),
                    QAAnswer(question_id="q13", question="News event risk assessed?", answer=False, passed=False),
                    QAAnswer(question_id="q14", question="Holiday trading risk assessed?", answer=True, passed=True),
                    QAAnswer(question_id="q15", question="Correlated pairs risk assessed?", answer=True, passed=True),
                    QAAnswer(question_id="q16", question="Drawdown recovery time acceptable?", answer=False, passed=False),
                    QAAnswer(question_id="q17", question="Equity curve stable?", answer=False, passed=False),
                    QAAnswer(question_id="q18", question="Monte Carlo passed?", answer=False, passed=False),
                    QAAnswer(question_id="q19", question="Walk-forward analysis passed?", answer=False, passed=False),
                    QAAnswer(question_id="q20", question="Ready for paper trading?", answer=False, passed=False),
                ],
                metrics=StageMetrics(
                    win_rate=48.5,
                    drawdown=18.2,
                    pnl=-3200.00,
                    sharpe_ratio=0.85,
                    profit_factor=1.1,
                    total_trades=78,
                    consecutive_losses=7,
                    avg_win=95.00,
                    avg_loss=110.00,
                    recovery_factor=0.8,
                    max_drawdown_duration=22
                ),
                decline_recovery_status="declining",
                notes="Backtest showing underperformance. Needs strategy adjustment before paper."
            ),
        ],
        current_report=StageReport(
            stage=LifecycleStage.BACKTEST,
            entered_at="2025-11-20T00:00:00Z",
            exited_at=None,
            q1_q20_answers=[
                QAAnswer(question_id="q1", question="Min 50 trades completed?", answer=True, passed=True),
                QAAnswer(question_id="q2", question="Win rate >= 55%?", answer=False, passed=False),
                QAAnswer(question_id="q3", question="Sharpe ratio >= 1.5?", answer=False, passed=False),
                QAAnswer(question_id="q4", question="Max drawdown < 15%?", answer=True, passed=True),
                QAAnswer(question_id="q5", question="Profit factor > 1.5?", answer=False, passed=False),
                QAAnswer(question_id="q6", question="Min 30 days in backtest?", answer=True, passed=True),
                QAAnswer(question_id="q7", question="Consistent daily returns?", answer=False, passed=False),
                QAAnswer(question_id="q8", question="No critical drawdown events?", answer=True, passed=True),
                QAAnswer(question_id="q9", question="Risk/reward ratio acceptable?", answer=True, passed=True),
                QAAnswer(question_id="q10", question="Spread impact assessed?", answer=True, passed=True),
                QAAnswer(question_id="q11", question="Slippage impact assessed?", answer=True, passed=True),
                QAAnswer(question_id="q12", question="Weekend gap risk acceptable?", answer=True, passed=True),
                QAAnswer(question_id="q13", question="News event risk assessed?", answer=False, passed=False),
                QAAnswer(question_id="q14", question="Holiday trading risk assessed?", answer=True, passed=True),
                QAAnswer(question_id="q15", question="Correlated pairs risk assessed?", answer=True, passed=True),
                QAAnswer(question_id="q16", question="Drawdown recovery time acceptable?", answer=False, passed=False),
                QAAnswer(question_id="q17", question="Equity curve stable?", answer=False, passed=False),
                QAAnswer(question_id="q18", question="Monte Carlo passed?", answer=False, passed=False),
                QAAnswer(question_id="q19", question="Walk-forward analysis passed?", answer=False, passed=False),
                QAAnswer(question_id="q20", question="Ready for paper trading?", answer=False, passed=False),
            ],
            metrics=StageMetrics(
                win_rate=48.5,
                drawdown=18.2,
                pnl=-3200.00,
                sharpe_ratio=0.85,
                profit_factor=1.1,
                total_trades=78,
                consecutive_losses=7,
                avg_win=95.00,
                avg_loss=110.00,
                recovery_factor=0.8,
                max_drawdown_duration=22
            ),
            decline_recovery_status="declining"
        )
    )


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

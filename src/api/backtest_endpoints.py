"""
Backtest Results API Endpoints

Provides REST API endpoints for:
- Listing all completed backtests
- Getting backtest detail by ID
- Getting running backtests with progress

Story: 4-4-backtest-results-api
"""

import logging
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from enum import Enum
import uuid

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/backtests", tags=["backtest"])


# =============================================================================
# Enums and Constants
# =============================================================================

class BacktestMode(str, Enum):
    """Backtest modes confirmed working."""
    VANILLA = "VANILLA"
    SPICED = "SPICED"
    VANILLA_FULL = "VANILLA_FULL"
    SPICED_FULL = "SPICED_FULL"
    MODE_B = "MODE_B"
    MODE_C = "MODE_C"


class BacktestStatus(str, Enum):
    """Backtest execution status."""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ReportType(str, Enum):
    """Report types for backtest detail."""
    BASIC = "basic"
    MONTE_CARLO = "monte_carlo"
    WALK_FORWARD = "walk_forward"
    PBO = "pbo"


# =============================================================================
# Response Models
# =============================================================================

class BacktestSummary(BaseModel):
    """
    Summary of a completed backtest.

    AC #1: Returns { id, ea_name, mode, run_at_utc, net_pnl, sharpe, max_drawdown, win_rate }
    """
    id: str = Field(..., description="Unique backtest identifier")
    ea_name: str = Field(..., description="EA/Strategy name")
    mode: BacktestMode = Field(..., description="Backtest mode")
    run_at_utc: datetime = Field(..., description="Run timestamp in UTC")
    net_pnl: float = Field(..., description="Net profit/loss percentage")
    sharpe: float = Field(..., description="Sharpe ratio")
    max_drawdown: float = Field(..., description="Maximum drawdown percentage")
    win_rate: float = Field(..., description="Win rate percentage (0-100)")


class BacktestDetail(BaseModel):
    """
    Full backtest detail.

    AC #2: Returns full backtest detail including equity curve data points,
    trade distribution, and mode-specific parameters.
    """
    id: str = Field(..., description="Unique backtest identifier")
    ea_name: str = Field(..., description="EA/Strategy name")
    mode: BacktestMode = Field(..., description="Backtest mode")
    run_at_utc: datetime = Field(..., description="Run timestamp in UTC")
    net_pnl: float = Field(..., description="Net profit/loss percentage")
    sharpe: float = Field(..., description="Sharpe ratio")
    max_drawdown: float = Field(..., description="Maximum drawdown percentage")
    win_rate: float = Field(..., description="Win rate percentage (0-100)")
    equity_curve: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Equity curve data points {timestamp, equity}"
    )
    trade_distribution: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Trade distribution histogram {bin, count}"
    )
    mode_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Mode-specific parameters"
    )
    report_type: Optional[ReportType] = Field(None, description="Report type generated")
    total_trades: int = Field(0, description="Total number of trades")
    profit_factor: float = Field(0.0, description="Profit factor")
    avg_trade_pnl: float = Field(0.0, description="Average trade P&L")


class RunningBacktest(BaseModel):
    """
    Running backtest with progress.

    AC #3: Returns running backtests with progress pct and partial metrics.
    """
    id: str = Field(..., description="Unique backtest identifier")
    ea_name: str = Field(..., description="EA/Strategy name")
    mode: BacktestMode = Field(..., description="Backtest mode")
    progress_pct: float = Field(..., ge=0, le=100, description="Progress percentage (0-100)")
    started_at_utc: datetime = Field(..., description="Start timestamp in UTC")
    partial_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Partial metrics {net_pnl, drawdown, etc.}"
    )


# =============================================================================
# In-Memory Storage (Demo/Dev - would be database in production)
# =============================================================================

# Store completed backtests
_completed_backtests: Dict[str, BacktestDetail] = {}

# Store running backtests
_running_backtests: Dict[str, RunningBacktest] = {}

# Initialize with demo data for development
def _init_demo_data():
    """Initialize demo backtest data for development/testing."""
    global _completed_backtests, _running_backtests

    # Demo completed backtests
    demo_backtests = [
        {
            "id": "bt-001",
            "ea_name": "TrendFollower_v2.1",
            "mode": BacktestMode.VANILLA,
            "net_pnl": 24.5,
            "sharpe": 1.82,
            "max_drawdown": 8.3,
            "win_rate": 62.5,
            "equity_curve": [
                {"timestamp": "2026-01-01T00:00:00Z", "equity": 10000},
                {"timestamp": "2026-01-15T00:00:00Z", "equity": 10500},
                {"timestamp": "2026-02-01T00:00:00Z", "equity": 11200},
                {"timestamp": "2026-02-15T00:00:00Z", "equity": 11800},
                {"timestamp": "2026-03-01T00:00:00Z", "equity": 12450},
            ],
            "trade_distribution": [
                {"bin": "0-10", "count": 12},
                {"bin": "10-20", "count": 8},
                {"bin": "20-30", "count": 15},
                {"bin": "30-50", "count": 5},
                {"bin": "50+", "count": 2},
            ],
            "mode_params": {
                "lookback_period": 20,
                "entry_threshold": 0.02,
                "exit_threshold": 0.015,
            },
            "report_type": ReportType.BASIC,
            "total_trades": 42,
            "profit_factor": 1.85,
            "avg_trade_pnl": 0.58,
        },
        {
            "id": "bt-002",
            "ea_name": "TrendFollower_v2.1",
            "mode": BacktestMode.SPICED,
            "net_pnl": 31.2,
            "sharpe": 2.15,
            "max_drawdown": 6.1,
            "win_rate": 68.3,
            "equity_curve": [
                {"timestamp": "2026-01-01T00:00:00Z", "equity": 10000},
                {"timestamp": "2026-01-15T00:00:00Z", "equity": 10800},
                {"timestamp": "2026-02-01T00:00:00Z", "equity": 11650},
                {"timestamp": "2026-02-15T00:00:00Z", "equity": 12500},
                {"timestamp": "2026-03-01T00:00:00Z", "equity": 13120},
            ],
            "trade_distribution": [
                {"bin": "0-10", "count": 8},
                {"bin": "10-20", "count": 12},
                {"bin": "20-30", "count": 18},
                {"bin": "30-50", "count": 8},
                {"bin": "50+", "count": 4},
            ],
            "mode_params": {
                "lookback_period": 20,
                "entry_threshold": 0.02,
                "exit_threshold": 0.015,
                "regime_filter": True,
                "skip_high_chaos": True,
                "skip_news_events": True,
            },
            "report_type": ReportType.BASIC,
            "total_trades": 50,
            "profit_factor": 2.1,
            "avg_trade_pnl": 0.62,
        },
        {
            "id": "bt-003",
            "ea_name": "RangeTrader_v1.5",
            "mode": BacktestMode.VANILLA,
            "net_pnl": 12.8,
            "sharpe": 1.34,
            "max_drawdown": 12.4,
            "win_rate": 55.2,
            "equity_curve": [
                {"timestamp": "2026-01-01T00:00:00Z", "equity": 10000},
                {"timestamp": "2026-01-15T00:00:00Z", "equity": 10350},
                {"timestamp": "2026-02-01T00:00:00Z", "equity": 10800},
                {"timestamp": "2026-02-15T00:00:00Z", "equity": 11100},
                {"timestamp": "2026-03-01T00:00:00Z", "equity": 11280},
            ],
            "trade_distribution": [
                {"bin": "0-10", "count": 20},
                {"bin": "10-20", "count": 10},
                {"bin": "20-30", "count": 5},
                {"bin": "30-50", "count": 3},
                {"bin": "50+", "count": 1},
            ],
            "mode_params": {
                "bb_period": 20,
                "bb_std": 2.0,
                "rsi_period": 14,
                "rsi_threshold": 30,
            },
            "report_type": ReportType.BASIC,
            "total_trades": 39,
            "profit_factor": 1.45,
            "avg_trade_pnl": 0.33,
        },
        {
            "id": "bt-004",
            "ea_name": "BreakoutScaler_v3.0",
            "mode": BacktestMode.MODE_B,
            "net_pnl": 18.9,
            "sharpe": 1.56,
            "max_drawdown": 9.8,
            "win_rate": 58.7,
            "equity_curve": [
                {"timestamp": "2026-01-01T00:00:00Z", "equity": 10000},
                {"timestamp": "2026-01-15T00:00:00Z", "equity": 10420},
                {"timestamp": "2026-02-01T00:00:00Z", "equity": 11050},
                {"timestamp": "2026-02-15T00:00:00Z", "equity": 11500},
                {"timestamp": "2026-03-01T00:00:00Z", "equity": 11890},
            ],
            "trade_distribution": [
                {"bin": "0-10", "count": 15},
                {"bin": "10-20", "count": 12},
                {"bin": "20-30", "count": 8},
                {"bin": "30-50", "count": 4},
                {"bin": "50+", "count": 2},
            ],
            "mode_params": {
                "atr_period": 14,
                "atr_multiplier": 2.5,
                "volume_filter": True,
                "volatility_adaptation": True,
            },
            "report_type": ReportType.BASIC,
            "total_trades": 41,
            "profit_factor": 1.62,
            "avg_trade_pnl": 0.46,
        },
        {
            "id": "bt-005",
            "ea_name": "TrendFollower_v2.1",
            "mode": BacktestMode.VANILLA_FULL,
            "net_pnl": 22.1,
            "sharpe": 1.75,
            "max_drawdown": 7.5,
            "win_rate": 64.2,
            "equity_curve": [
                {"timestamp": "2026-01-01T00:00:00Z", "equity": 10000},
                {"timestamp": "2026-01-15T00:00:00Z", "equity": 10600},
                {"timestamp": "2026-02-01T00:00:00Z", "equity": 11300},
                {"timestamp": "2026-02-15T00:00:00Z", "equity": 11900},
                {"timestamp": "2026-03-01T00:00:00Z", "equity": 12210},
            ],
            "trade_distribution": [
                {"bin": "0-10", "count": 10},
                {"bin": "10-20", "count": 14},
                {"bin": "20-30", "count": 16},
                {"bin": "30-50", "count": 6},
                {"bin": "50+", "count": 3},
            ],
            "mode_params": {
                "lookback_period": 20,
                "entry_threshold": 0.02,
                "exit_threshold": 0.015,
                "walk_forward_window": 90,
                "walk_forward_step": 30,
                "optimization_metric": "sharpe",
            },
            "report_type": ReportType.WALK_FORWARD,
            "total_trades": 49,
            "profit_factor": 1.78,
            "avg_trade_pnl": 0.45,
        },
        {
            "id": "bt-006",
            "ea_name": "TrendFollower_v2.1",
            "mode": BacktestMode.SPICED_FULL,
            "net_pnl": 28.7,
            "sharpe": 2.05,
            "max_drawdown": 5.8,
            "win_rate": 70.1,
            "equity_curve": [
                {"timestamp": "2026-01-01T00:00:00Z", "equity": 10000},
                {"timestamp": "2026-01-15T00:00:00Z", "equity": 10850},
                {"timestamp": "2026-02-01T00:00:00Z", "equity": 11800},
                {"timestamp": "2026-02-15T00:00:00Z", "equity": 12600},
                {"timestamp": "2026-03-01T00:00:00Z", "equity": 12870},
            ],
            "trade_distribution": [
                {"bin": "0-10", "count": 6},
                {"bin": "10-20", "count": 10},
                {"bin": "20-30", "count": 20},
                {"bin": "30-50", "count": 10},
                {"bin": "50+", "count": 5},
            ],
            "mode_params": {
                "lookback_period": 20,
                "entry_threshold": 0.02,
                "exit_threshold": 0.015,
                "regime_filter": True,
                "skip_high_chaos": True,
                "skip_news_events": True,
                "walk_forward_window": 90,
                "walk_forward_step": 30,
                "optimization_metric": "sharpe",
            },
            "report_type": ReportType.WALK_FORWARD,
            "total_trades": 51,
            "profit_factor": 2.05,
            "avg_trade_pnl": 0.56,
        },
        {
            "id": "bt-007",
            "ea_name": "TrendFollower_v2.1",
            "mode": BacktestMode.MODE_C,
            "net_pnl": 19.8,
            "sharpe": 1.68,
            "max_drawdown": 8.9,
            "win_rate": 61.5,
            "equity_curve": [
                {"timestamp": "2026-01-01T00:00:00Z", "equity": 10000},
                {"timestamp": "2026-01-15T00:00:00Z", "equity": 10550},
                {"timestamp": "2026-02-01T00:00:00Z", "equity": 11200},
                {"timestamp": "2026-02-15T00:00:00Z", "equity": 11650},
                {"timestamp": "2026-03-01T00:00:00Z", "equity": 11980},
            ],
            "trade_distribution": [
                {"bin": "0-10", "count": 11},
                {"bin": "10-20", "count": 13},
                {"bin": "20-30", "count": 14},
                {"bin": "30-50", "count": 5},
                {"bin": "50+", "count": 2},
            ],
            "mode_params": {
                "lookback_period": 20,
                "entry_threshold": 0.02,
                "exit_threshold": 0.015,
                "enhanced_momentum": True,
                "adaptive_stops": True,
            },
            "report_type": ReportType.BASIC,
            "total_trades": 45,
            "profit_factor": 1.72,
            "avg_trade_pnl": 0.44,
        },
    ]

    for bt in demo_backtests:
        run_at = datetime(2026, 3, 1, 12, 0, 0, tzinfo=timezone.utc)
        detail = BacktestDetail(
            id=bt["id"],
            ea_name=bt["ea_name"],
            mode=bt["mode"],
            run_at_utc=run_at,
            net_pnl=bt["net_pnl"],
            sharpe=bt["sharpe"],
            max_drawdown=bt["max_drawdown"],
            win_rate=bt["win_rate"],
            equity_curve=bt["equity_curve"],
            trade_distribution=bt["trade_distribution"],
            mode_params=bt["mode_params"],
            report_type=bt.get("report_type"),
            total_trades=bt.get("total_trades", 0),
            profit_factor=bt.get("profit_factor", 0.0),
            avg_trade_pnl=bt.get("avg_trade_pnl", 0.0),
        )
        _completed_backtests[bt["id"]] = detail

    # Demo running backtest
    running = RunningBacktest(
        id="bt-running-001",
        ea_name="TrendFollower_v2.2",
        mode=BacktestMode.SPICED_FULL,
        progress_pct=67.5,
        started_at_utc=datetime(2026, 3, 15, 8, 0, 0, tzinfo=timezone.utc),
        partial_metrics={
            "net_pnl": 15.2,
            "drawdown": 4.8,
            "sharpe": 1.45,
        },
    )
    _running_backtests[running.id] = running

    logger.info(f"Initialized {len(_completed_backtests)} demo backtests and {len(_running_backtests)} running backtests")


# Initialize demo data on module load
_init_demo_data()


# =============================================================================
# Endpoints
# =============================================================================

@router.get("", response_model=List[BacktestSummary])
async def list_backtests(
    mode: Optional[BacktestMode] = Query(None, description="Filter by backtest mode"),
    ea_name: Optional[str] = Query(None, description="Filter by EA name"),
    limit: int = Query(50, ge=1, le=200, description="Maximum results to return"),
):
    """
    List all completed backtests.

    AC #1: Returns all completed backtests:
    { id, ea_name, mode, run_at_utc, net_pnl, sharpe, max_drawdown, win_rate }
    """
    results = []

    for bt in _completed_backtests.values():
        # Apply filters
        if mode and bt.mode != mode:
            continue
        if ea_name and bt.ea_name != ea_name:
            continue

        results.append(BacktestSummary(
            id=bt.id,
            ea_name=bt.ea_name,
            mode=bt.mode,
            run_at_utc=bt.run_at_utc,
            net_pnl=bt.net_pnl,
            sharpe=bt.sharpe,
            max_drawdown=bt.max_drawdown,
            win_rate=bt.win_rate,
        ))

    # Sort by run_at_utc descending (newest first)
    results.sort(key=lambda x: x.run_at_utc, reverse=True)

    # Apply limit
    results = results[:limit]

    logger.info(f"Returning {len(results)} backtest summaries (mode filter: {mode}, ea_name filter: {ea_name})")
    return results


@router.get("/running", response_model=List[RunningBacktest])
async def list_running_backtests():
    """
    List running backtests with progress.

    AC #3: Returns running backtests with progress pct and partial metrics.
    """
    results = list(_running_backtests.values())
    logger.info(f"Returning {len(results)} running backtests")
    return results


@router.get("/{backtest_id}", response_model=BacktestDetail)
async def get_backtest_detail(backtest_id: str):
    """
    Get full backtest detail by ID.

    AC #2: Returns full backtest detail including equity curve data points,
    trade distribution, and mode-specific parameters.
    """
    if backtest_id not in _completed_backtests:
        raise HTTPException(
            status_code=404,
            detail=f"Backtest {backtest_id} not found"
        )

    detail = _completed_backtests[backtest_id]
    logger.info(f"Returning backtest detail for {backtest_id}")
    return detail


@router.delete("/{backtest_id}")
async def delete_backtest(backtest_id: str):
    """Delete a completed backtest result."""
    if backtest_id not in _completed_backtests:
        raise HTTPException(
            status_code=404,
            detail=f"Backtest {backtest_id} not found"
        )

    del _completed_backtests[backtest_id]
    logger.info(f"Deleted backtest {backtest_id}")
    return {"status": "deleted", "backtest_id": backtest_id}


# =============================================================================
# Demo/Test Endpoints
# =============================================================================

@router.post("/demo/seed")
async def seed_demo_backtests(count: int = Query(6, ge=1, le=50)):
    """
    Seed additional demo backtest data for testing.

    This is a dev-only endpoint for testing purposes.
    """
    global _completed_backtests

    modes = list(BacktestMode)
    ea_names = ["TrendFollower_v2.1", "RangeTrader_v1.5", "BreakoutScaler_v3.0", "ScalperPro_v1.0"]

    import random

    for i in range(count):
        bt_id = f"bt-demo-{uuid.uuid4().hex[:8]}"
        mode = random.choice(modes)
        ea_name = random.choice(ea_names)

        detail = BacktestDetail(
            id=bt_id,
            ea_name=ea_name,
            mode=mode,
            run_at_utc=datetime.now(timezone.utc),
            net_pnl=round(random.uniform(-5, 35), 2),
            sharpe=round(random.uniform(0.5, 2.5), 2),
            max_drawdown=round(random.uniform(3, 15), 2),
            win_rate=round(random.uniform(40, 80), 1),
            equity_curve=[
                {"timestamp": "2026-01-01T00:00:00Z", "equity": 10000},
                {"timestamp": "2026-02-01T00:00:00Z", "equity": 11000},
                {"timestamp": "2026-03-01T00:00:00Z", "equity": 12000},
            ],
            trade_distribution=[
                {"bin": "0-10", "count": random.randint(5, 20)},
                {"bin": "10-20", "count": random.randint(5, 15)},
                {"bin": "20-30", "count": random.randint(3, 10)},
            ],
            mode_params={"lookback_period": random.randint(10, 30)},
            report_type=ReportType.BASIC,
            total_trades=random.randint(20, 60),
            profit_factor=round(random.uniform(1.2, 2.5), 2),
            avg_trade_pnl=round(random.uniform(0.2, 0.8), 2),
        )
        _completed_backtests[bt_id] = detail

    logger.info(f"Seeded {count} demo backtests")
    return {"status": "seeded", "count": count, "total_backtests": len(_completed_backtests)}


@router.post("/demo/reset")
async def reset_demo_backtests():
    """Reset all demo backtest data."""
    global _completed_backtests, _running_backtests

    _completed_backtests.clear()
    _running_backtests.clear()

    _init_demo_data()

    logger.info("Reset demo backtest data")
    return {"status": "reset", "total_backtests": len(_completed_backtests)}

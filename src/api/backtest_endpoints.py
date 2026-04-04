"""
Backtest results API endpoints.

Production mode only:
- reads persisted backtest artifacts from the canonical WF1/shared-assets tree
- never seeds demo rows
- returns honest empty states when no backtests exist yet
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from src.api.wf1_artifacts import iter_strategy_roots

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/backtests", tags=["backtest"])


class BacktestMode(str, Enum):
    VANILLA = "VANILLA"
    SPICED = "SPICED"
    VANILLA_FULL = "VANILLA_FULL"
    SPICED_FULL = "SPICED_FULL"
    MODE_B = "MODE_B"
    MODE_C = "MODE_C"


class BacktestStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ReportType(str, Enum):
    BASIC = "basic"
    MONTE_CARLO = "monte_carlo"
    WALK_FORWARD = "walk_forward"
    PBO = "pbo"


class BacktestSummary(BaseModel):
    id: str = Field(..., description="Unique backtest identifier")
    ea_name: str = Field(..., description="EA/Strategy name")
    mode: BacktestMode = Field(..., description="Backtest mode")
    run_at_utc: datetime = Field(..., description="Run timestamp in UTC")
    net_pnl: float = Field(..., description="Net profit/loss percentage")
    sharpe: float = Field(..., description="Sharpe ratio")
    max_drawdown: float = Field(..., description="Maximum drawdown percentage")
    win_rate: float = Field(..., description="Win rate percentage (0-100)")


class BacktestDetail(BaseModel):
    id: str = Field(..., description="Unique backtest identifier")
    ea_name: str = Field(..., description="EA/Strategy name")
    mode: BacktestMode = Field(..., description="Backtest mode")
    run_at_utc: datetime = Field(..., description="Run timestamp in UTC")
    net_pnl: float = Field(..., description="Net profit/loss percentage")
    sharpe: float = Field(..., description="Sharpe ratio")
    max_drawdown: float = Field(..., description="Maximum drawdown percentage")
    win_rate: float = Field(..., description="Win rate percentage (0-100)")
    equity_curve: List[Dict[str, Any]] = Field(default_factory=list)
    trade_distribution: List[Dict[str, Any]] = Field(default_factory=list)
    mode_params: Dict[str, Any] = Field(default_factory=dict)
    report_type: Optional[ReportType] = Field(None)
    total_trades: int = Field(0)
    profit_factor: float = Field(0.0)
    avg_trade_pnl: float = Field(0.0)


class RunningBacktest(BaseModel):
    id: str = Field(..., description="Unique backtest identifier")
    ea_name: str = Field(..., description="EA/Strategy name")
    mode: BacktestMode = Field(..., description="Backtest mode")
    progress_pct: float = Field(..., ge=0, le=100)
    started_at_utc: datetime = Field(..., description="Start timestamp in UTC")
    partial_metrics: Dict[str, float] = Field(default_factory=dict)


def _parse_iso(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc)
    if isinstance(value, str) and value:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    return datetime.now(timezone.utc)


def _coerce_mode(value: Any) -> BacktestMode:
    raw = str(value or "").strip().upper()
    alias_map = {
        "VANILLA": BacktestMode.VANILLA,
        "SPICED": BacktestMode.SPICED,
        "VANILLA_FULL": BacktestMode.VANILLA_FULL,
        "SPICED_FULL": BacktestMode.SPICED_FULL,
        "MODE_B": BacktestMode.MODE_B,
        "MODE_C": BacktestMode.MODE_C,
        "MONTE_CARLO": BacktestMode.SPICED_FULL,
        "WALK_FORWARD": BacktestMode.VANILLA_FULL,
    }
    return alias_map.get(raw, BacktestMode.VANILLA)


def _coerce_report_type(value: Any) -> Optional[ReportType]:
    if not value:
        return None
    raw = str(value).strip().lower()
    for item in ReportType:
        if item.value == raw:
            return item
    return None


def _normalize_detail(payload: Dict[str, Any], source_path: Path) -> Optional[BacktestDetail]:
    if not isinstance(payload, dict):
        return None

    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    mode_params = payload.get("mode_params") if isinstance(payload.get("mode_params"), dict) else {}

    backtest_id = str(
        payload.get("id")
        or payload.get("backtest_id")
        or summary.get("id")
        or source_path.stem
    )
    ea_name = str(
        payload.get("ea_name")
        or payload.get("strategy_name")
        or payload.get("variant_name")
        or summary.get("ea_name")
        or source_path.parent.parent.name
    )

    detail = BacktestDetail(
        id=backtest_id,
        ea_name=ea_name,
        mode=_coerce_mode(payload.get("mode") or summary.get("mode") or payload.get("variant_type")),
        run_at_utc=_parse_iso(payload.get("run_at_utc") or payload.get("updated_at") or payload.get("created_at")),
        net_pnl=float(payload.get("net_pnl", summary.get("net_pnl", metrics.get("net_pnl", 0.0))) or 0.0),
        sharpe=float(payload.get("sharpe", summary.get("sharpe", metrics.get("sharpe", 0.0))) or 0.0),
        max_drawdown=float(
            payload.get("max_drawdown", summary.get("max_drawdown", metrics.get("max_drawdown", 0.0))) or 0.0
        ),
        win_rate=float(payload.get("win_rate", summary.get("win_rate", metrics.get("win_rate", 0.0))) or 0.0),
        equity_curve=list(payload.get("equity_curve") or metrics.get("equity_curve") or []),
        trade_distribution=list(payload.get("trade_distribution") or metrics.get("trade_distribution") or []),
        mode_params=mode_params,
        report_type=_coerce_report_type(payload.get("report_type") or summary.get("report_type")),
        total_trades=int(payload.get("total_trades", summary.get("total_trades", metrics.get("total_trades", 0))) or 0),
        profit_factor=float(
            payload.get("profit_factor", summary.get("profit_factor", metrics.get("profit_factor", 0.0))) or 0.0
        ),
        avg_trade_pnl=float(
            payload.get("avg_trade_pnl", summary.get("avg_trade_pnl", metrics.get("avg_trade_pnl", 0.0))) or 0.0
        ),
    )
    return detail


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Skipping unreadable backtest artifact %s: %s", path, exc)
        return None


def _scan_backtest_files() -> tuple[Dict[str, BacktestDetail], Dict[str, RunningBacktest]]:
    completed: Dict[str, BacktestDetail] = {}
    running: Dict[str, RunningBacktest] = {}

    for root in iter_strategy_roots() or []:
        for section_name in ("backtests", "reports"):
            section = root / section_name
            if not section.exists():
                continue

            for json_path in sorted(section.rglob("*.json")):
                payload = _load_json(json_path)
                if not payload:
                    continue

                status_raw = str(payload.get("status", "")).lower()
                if status_raw == BacktestStatus.RUNNING.value:
                    running_id = str(payload.get("id") or payload.get("backtest_id") or json_path.stem)
                    running[running_id] = RunningBacktest(
                        id=running_id,
                        ea_name=str(payload.get("ea_name") or payload.get("strategy_name") or root.name),
                        mode=_coerce_mode(payload.get("mode") or payload.get("variant_type")),
                        progress_pct=float(payload.get("progress_pct", payload.get("progress", 0.0)) or 0.0),
                        started_at_utc=_parse_iso(payload.get("started_at_utc") or payload.get("started_at")),
                        partial_metrics=dict(payload.get("partial_metrics") or payload.get("metrics") or {}),
                    )
                    continue

                detail = _normalize_detail(payload, json_path)
                if detail:
                    completed[detail.id] = detail

    return completed, running


_completed_backtests: Dict[str, BacktestDetail] = {}
_running_backtests: Dict[str, RunningBacktest] = {}


def refresh_backtests_from_storage() -> None:
    completed, running = _scan_backtest_files()
    _completed_backtests.clear()
    _completed_backtests.update(completed)
    _running_backtests.clear()
    _running_backtests.update(running)


@router.get("", response_model=List[BacktestSummary])
async def list_backtests(
    mode: Optional[BacktestMode] = Query(None, description="Filter by backtest mode"),
    ea_name: Optional[str] = Query(None, description="Filter by EA name"),
    limit: int = Query(50, ge=1, le=200, description="Maximum results to return"),
):
    refresh_backtests_from_storage()

    results: List[BacktestSummary] = []
    for bt in _completed_backtests.values():
        if mode and bt.mode != mode:
            continue
        if ea_name and bt.ea_name != ea_name:
            continue
        results.append(
            BacktestSummary(
                id=bt.id,
                ea_name=bt.ea_name,
                mode=bt.mode,
                run_at_utc=bt.run_at_utc,
                net_pnl=bt.net_pnl,
                sharpe=bt.sharpe,
                max_drawdown=bt.max_drawdown,
                win_rate=bt.win_rate,
            )
        )

    results.sort(key=lambda item: item.run_at_utc, reverse=True)
    return results[:limit]


@router.get("/running", response_model=List[RunningBacktest])
async def list_running_backtests():
    refresh_backtests_from_storage()
    return list(_running_backtests.values())


@router.get("/{backtest_id}", response_model=BacktestDetail)
async def get_backtest_detail(backtest_id: str):
    refresh_backtests_from_storage()
    detail = _completed_backtests.get(backtest_id)
    if not detail:
        raise HTTPException(status_code=404, detail=f"Backtest {backtest_id} not found")
    return detail


@router.delete("/{backtest_id}")
async def delete_backtest(backtest_id: str):
    refresh_backtests_from_storage()
    detail = _completed_backtests.get(backtest_id)
    if not detail:
        raise HTTPException(status_code=404, detail=f"Backtest {backtest_id} not found")

    removed = False
    for root in iter_strategy_roots() or []:
        for section_name in ("backtests", "reports"):
            for json_path in (root / section_name).rglob("*.json") if (root / section_name).exists() else []:
                payload = _load_json(json_path)
                if not payload:
                    continue
                candidate_id = str(payload.get("id") or payload.get("backtest_id") or json_path.stem)
                if candidate_id == backtest_id:
                    json_path.unlink(missing_ok=True)
                    removed = True

    _completed_backtests.pop(backtest_id, None)
    return {"status": "deleted" if removed else "missing_on_disk", "backtest_id": backtest_id}

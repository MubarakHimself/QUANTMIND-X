"""
Dead Zone Workflow 3 API Endpoints
==================================

Endpoints for:
- Workflow 3 status and triggering
- DPR scores and queue status
- EOD Report retrieval
- Dead Zone scheduler management
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.router.dead_zone_workflow_3 import get_dead_zone_workflow
from src.router.dead_zone_scheduler import get_dead_zone_scheduler
from src.router.dpr_scoring_engine import get_dpr_scoring_engine
from src.router.eod_report_generator import get_eod_report_generator
from src.router.cold_storage_writer import get_cold_storage_writer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/dead-zone", tags=["dead-zone"])


# ============= Request/Response Models =============
class WorkflowTriggerResponse(BaseModel):
    run_id: str
    status: str
    message: str


class DPRScoreResponse(BaseModel):
    bot_id: str
    composite_score: float
    tier: str
    rank: int
    session_specialist: bool
    session_concern: bool


class QueueStatusResponse(BaseModel):
    queue: list[str]
    concerns: list[str]
    concerns_count: int


class EODReportResponse(BaseModel):
    """Full EOD Report response with all details."""
    report_id: str
    trading_date: str
    total_pnl: float
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    net_pnl: float
    best_trade: float
    worst_trade: float
    circuit_breaker_events: list[dict]
    force_close_reasons: list[dict]
    session_breakdown: dict
    regime_at_close: str
    generated_at: str


class EODReportSummary(BaseModel):
    """Summary of an EOD Report for listing."""
    report_id: str
    trading_date: str
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    net_pnl: float
    regime_at_close: str
    generated_at: str


# ============= Workflow Endpoints =============
@router.get("/workflow/status")
async def get_workflow_status():
    """Get current Dead Zone Workflow 3 status."""
    workflow = get_dead_zone_workflow()
    return workflow.get_status()


@router.post("/workflow/trigger", response_model=WorkflowTriggerResponse)
async def trigger_workflow():
    """Manually trigger Dead Zone Workflow 3."""
    workflow = get_dead_zone_workflow()

    try:
        result = await workflow.execute()
        return WorkflowTriggerResponse(
            run_id=result.run_id,
            status=result.status,
            message="Workflow 3 triggered successfully" if result.status == "completed" else f"Workflow failed: {result.error}"
        )
    except Exception as e:
        logger.error(f"Workflow trigger failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/workflow/history")
async def get_workflow_history():
    """Get recent workflow run history."""
    # Placeholder - would store workflow runs in database
    return {"runs": [], "note": "Workflow history not yet implemented"}


# ============= DPR Endpoints =============
@router.get("/dpr/scores")
async def get_dpr_scores():
    """Get current DPR scores for all active bots."""
    try:
        from src.router.bot_manifest import BotRegistry

        registry = BotRegistry()
        active_bots = registry.list_live_trading()
        bot_ids = [bot.bot_id for bot in active_bots]

        scoring_engine = get_dpr_scoring_engine()
        scores = await scoring_engine.score_all_bots(bot_ids)

        return {
            "scores": [s.to_dict() for s in scores],
            "total_bots": len(scores),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to get DPR scores: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dpr/scores/{bot_id}")
async def get_bot_dpr_score(bot_id: str):
    """Get DPR score for a specific bot."""
    try:
        scoring_engine = get_dpr_scoring_engine()
        scores = await scoring_engine.score_all_bots([bot_id])

        if not scores:
            raise HTTPException(status_code=404, detail=f"Bot {bot_id} not found")

        return scores[0].to_dict()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get DPR score for {bot_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============= Queue Endpoints =============
@router.get("/queue/status", response_model=QueueStatusResponse)
async def get_queue_status():
    """Get current queue status including SESSION_CONCERN flags."""
    try:
        from src.router.bot_manifest import BotRegistry
        from src.router.queue_reranker import get_queue_reranker

        registry = BotRegistry()
        active_bots = registry.list_live_trading()
        bot_ids = [bot.bot_id for bot in active_bots]

        scoring_engine = get_dpr_scoring_engine()
        scores = await scoring_engine.score_all_bots(bot_ids)

        reranker = get_queue_reranker()
        result = await reranker.run(scores)

        return QueueStatusResponse(
            queue=result.queue,
            concerns=result.concerns,
            concerns_count=len(result.concerns),
        )

    except Exception as e:
        logger.error(f"Failed to get queue status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/queue/specialists")
async def get_session_specialists():
    """Get all bots tagged as SESSION_SPECIALIST."""
    try:
        from src.router.bot_manifest import BotRegistry

        registry = BotRegistry()
        active_bots = registry.list_live_trading()

        specialists = [
            {
                "bot_id": bot.bot_id,
                "strategy_name": bot.strategy_type.value if hasattr(bot.strategy_type, 'value') else str(bot.strategy_type),
            }
            for bot in active_bots
            if 'SESSION_SPECIALIST' in bot.tags
        ]

        return {
            "specialists": specialists,
            "count": len(specialists),
        }

    except Exception as e:
        logger.error(f"Failed to get specialists: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/queue/concerns")
async def get_session_concerns():
    """Get all bots flagged with SESSION_CONCERN."""
    try:
        from src.router.bot_manifest import BotRegistry

        registry = BotRegistry()
        active_bots = registry.list_live_trading()

        concerns = [
            {
                "bot_id": bot.bot_id,
                "strategy_name": bot.strategy_type.value if hasattr(bot.strategy_type, 'value') else str(bot.strategy_type),
                "consecutive_negative_ev": 0,  # Placeholder - tracked externally
            }
            for bot in active_bots
            if 'SESSION_CONCERN' in bot.tags
        ]

        return {
            "concerns": concerns,
            "count": len(concerns),
        }

    except Exception as e:
        logger.error(f"Failed to get concerns: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============= EOD Report Endpoints =============
@router.get("/eod-report/today")
async def get_today_eod_report():
    """Get today's EOD Report."""
    try:
        generator = get_eod_report_generator()
        report = await generator.generate()

        return report.to_dict()

    except Exception as e:
        logger.error(f"Failed to generate EOD report: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/eod-report/{trading_date}")
async def get_eod_report_by_date(trading_date: str):
    """Get EOD Report for a specific trading date from cold storage."""
    try:
        writer = get_cold_storage_writer()
        file_key = f"eod_reports/{trading_date}_eod_report.json"
        data = await writer.read(file_key)

        if data is None:
            raise HTTPException(
                status_code=404,
                detail=f"EOD Report for {trading_date} not found"
            )

        # Transform to enhanced format with circuit breaker events and force close reasons
        return _transform_eod_report(data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get EOD report for {trading_date}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/eod-report/latest")
async def get_latest_eod_report():
    """Get the most recent EOD Report from cold storage."""
    try:
        writer = get_cold_storage_writer()

        # List all EOD report files
        files = await writer.list_files("eod_reports")
        eod_files = [f for f in files if f.endswith("_eod_report.json")]

        if not eod_files:
            raise HTTPException(
                status_code=404,
                detail="No EOD reports found in cold storage"
            )

        # Sort by trading date (extract from filename: YYYY-MM-DD_eod_report.json)
        def extract_date(f: str) -> str:
            return f.split("_")[0]

        eod_files.sort(key=extract_date, reverse=True)
        latest_file = eod_files[0]

        # Read and return the latest report
        data = await writer.read(latest_file)
        if data is None:
            raise HTTPException(
                status_code=404,
                detail=f"Failed to read latest EOD report: {latest_file}"
            )

        return _transform_eod_report(data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get latest EOD report: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/eod-reports", response_model=list[EODReportSummary])
async def list_eod_reports():
    """Get list of all available EOD reports from cold storage."""
    try:
        writer = get_cold_storage_writer()

        # List all EOD report files
        files = await writer.list_files("eod_reports")
        eod_files = [f for f in files if f.endswith("_eod_report.json")]

        if not eod_files:
            return []

        reports = []
        for file_key in eod_files:
            data = await writer.read(file_key)
            if data:
                reports.append(EODReportSummary(
                    report_id=data.get("report_id", ""),
                    trading_date=data.get("trading_date", ""),
                    total_trades=data.get("total_trades", 0),
                    wins=data.get("day_outcome", {}).get("winning_trades", 0),
                    losses=data.get("day_outcome", {}).get("losing_trades", 0),
                    win_rate=data.get("day_outcome", {}).get("win_rate", 0.0),
                    net_pnl=data.get("day_outcome", {}).get("net_pnl", 0.0),
                    regime_at_close=data.get("regime_at_close", "UNKNOWN"),
                    generated_at=data.get("generated_at", ""),
                ))

        # Sort by trading date descending
        reports.sort(key=lambda x: x.trading_date, reverse=True)
        return reports

    except Exception as e:
        logger.error(f"Failed to list EOD reports: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def _transform_eod_report(data: dict) -> dict:
    """
    Transform raw EOD report data into the enhanced format with:
    - circuit_breaker_events (extracted from anomaly_events)
    - force_close_reasons (extracted from anomaly_events)
    - session_breakdown (london/ny session stats)
    """
    # Extract circuit breaker events and force close reasons from anomaly_events
    circuit_breaker_events = []
    force_close_reasons = []

    for event in data.get("anomaly_events", []):
        event_type = event.get("event_type", "")
        if "CIRCUIT_BREAKER" in event_type.upper() or "CIRCUIT_BREAKER" in event.get("description", "").upper():
            circuit_breaker_events.append({
                "timestamp": event.get("timestamp", ""),
                "event_type": event_type,
                "severity": event.get("severity", "UNKNOWN"),
                "description": event.get("description", ""),
                "affected_bots": event.get("affected_bots", []),
            })
        elif "FORCE_CLOSE" in event_type.upper() or "FORCE_CLOSE" in event.get("description", "").upper():
            force_close_reasons.append({
                "timestamp": event.get("timestamp", ""),
                "event_type": event_type,
                "severity": event.get("severity", "UNKNOWN"),
                "description": event.get("description", ""),
                "affected_bots": event.get("affected_bots", []),
            })

    # Build session breakdown (london/ny) from pnl_attribution if available
    # This is a placeholder - actual implementation would group by session
    london_pnl = 0.0
    ny_pnl = 0.0
    london_trades = 0
    ny_trades = 0

    for attr in data.get("pnl_attribution", []):
        # Placeholder: in real implementation, bots would have session tags
        pnl = attr.get("pnl", 0.0)
        trades = attr.get("trades", 0)
        # For now, split evenly (would be based on actual session attribution)
        london_pnl += pnl * 0.4  # 40% during London session
        ny_pnl += pnl * 0.6     # 60% during NY session
        london_trades += int(trades * 0.4)
        ny_trades += int(trades * 0.6)

    day_outcome = data.get("day_outcome", {})

    return {
        "date": data.get("trading_date", ""),
        "total_trades": data.get("total_trades", 0),
        "wins": day_outcome.get("winning_trades", 0),
        "losses": day_outcome.get("losing_trades", 0),
        "win_rate": day_outcome.get("win_rate", 0.0),
        "net_pnl": day_outcome.get("net_pnl", 0.0),
        "best_trade": day_outcome.get("largest_win", 0.0),
        "worst_trade": day_outcome.get("largest_loss", 0.0),
        "circuit_breaker_events": circuit_breaker_events,
        "force_close_reasons": force_close_reasons,
        "session_breakdown": {
            "london": {
                "pnl": london_pnl,
                "trades": london_trades,
            },
            "ny": {
                "pnl": ny_pnl,
                "trades": ny_trades,
            },
        },
        "regime_at_close": data.get("regime_at_close", "UNKNOWN"),
        "timestamp": data.get("generated_at", ""),
    }


# ============= Scheduler Endpoints =============
@router.get("/scheduler/status")
async def get_scheduler_status():
    """Get Dead Zone scheduler status."""
    scheduler = get_dead_zone_scheduler()

    return {
        "running": scheduler._running,
        "workflow_3_start": scheduler.WORKFLOW_3_START,
    }


@router.post("/scheduler/start")
async def start_scheduler():
    """Start the Dead Zone scheduler."""
    scheduler = get_dead_zone_scheduler()

    if scheduler._running:
        return {"message": "Scheduler already running"}

    await scheduler.start()
    return {"message": "Scheduler started"}


@router.post("/scheduler/stop")
async def stop_scheduler():
    """Stop the Dead Zone scheduler."""
    scheduler = get_dead_zone_scheduler()

    if not scheduler._running:
        return {"message": "Scheduler not running"}

    scheduler.stop()
    return {"message": "Scheduler stopped"}

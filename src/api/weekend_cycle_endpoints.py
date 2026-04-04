"""
Weekend Cycle API Endpoints
==========================

Endpoints for:
- Workflow 4 status and triggering
- Weekend cycle scheduler management
- Roster deployment status
- Weekday block status

Reference: Story 8.13 (8-13-workflow-4-weekend-update-cycle)
"""

import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.router.weekend_update_cycle_workflow import get_weekend_update_cycle_workflow
from src.router.weekend_cycle_scheduler import get_weekend_cycle_scheduler
from src.router.weekday_parameter_guard import get_weekday_parameter_guard
from src.router.weekend_roster_manager import get_weekend_roster_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/weekend-cycle", tags=["weekend-cycle"])


# ============= Request/Response Models =============
class WorkflowTriggerResponse(BaseModel):
    run_id: str
    status: str
    message: str


class StepTriggerResponse(BaseModel):
    step_name: str
    status: str
    message: str


class BlockStatusResponse(BaseModel):
    is_blocked: bool
    current_day: str
    current_time: str
    next_allowed_window: Optional[str]
    message: str


# ============= Workflow Endpoints =============
@router.get("/workflow/status")
async def get_workflow_status():
    """Get current Weekend Update Cycle Workflow 4 status."""
    workflow = get_weekend_update_cycle_workflow()
    return workflow.get_status()


@router.post("/workflow/trigger", response_model=WorkflowTriggerResponse)
async def trigger_workflow():
    """Manually trigger full Weekend Update Cycle (not typically used)."""
    workflow = get_weekend_update_cycle_workflow()

    try:
        result = await workflow.execute()
        return WorkflowTriggerResponse(
            run_id=result.run_id,
            status=result.status,
            message="Weekend cycle triggered successfully" if result.status == "completed" else f"Workflow failed: {result.error}"
        )
    except Exception as e:
        logger.error(f"Weekend cycle trigger failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/workflow/step/{step_name}/trigger", response_model=StepTriggerResponse)
async def trigger_step(step_name: str):
    """Manually trigger a specific workflow step."""
    workflow = get_weekend_update_cycle_workflow()

    valid_steps = [
        "friday_analysis", "saturday_refinement", "saturday_wfa", "saturday_hmm_retrain",
        "sunday_calibration", "sunday_spread_profiles", "sunday_sqs_refresh",
        "sunday_kelly_calibration", "monday_roster_deploy"
    ]

    if step_name not in valid_steps:
        raise HTTPException(status_code=400, detail=f"Invalid step name: {step_name}")

    try:
        await workflow.execute_single_step(step_name)
        return StepTriggerResponse(
            step_name=step_name,
            status="completed",
            message=f"Step {step_name} triggered successfully"
        )
    except Exception as e:
        logger.error(f"Step trigger failed for {step_name}: {e}", exc_info=True)
        return StepTriggerResponse(
            step_name=step_name,
            status="failed",
            message=str(e)
        )


@router.get("/workflow/history")
async def get_workflow_history():
    """Get recent workflow run history."""
    # Placeholder - would store workflow runs in database
    return {"runs": [], "note": "Workflow history not yet implemented"}


# ============= Scheduler Endpoints =============
@router.get("/scheduler/status")
async def get_scheduler_status():
    """Get Weekend Cycle scheduler status."""
    scheduler = get_weekend_cycle_scheduler()
    return scheduler.get_status()


@router.post("/scheduler/start")
async def start_scheduler():
    """Start the Weekend Cycle scheduler."""
    scheduler = get_weekend_cycle_scheduler()

    if scheduler._running:
        return {"message": "Scheduler already running"}

    await scheduler.start()
    return {"message": "Scheduler started"}


@router.post("/scheduler/stop")
async def stop_scheduler():
    """Stop the Weekend Cycle scheduler."""
    scheduler = get_weekend_cycle_scheduler()

    if not scheduler._running:
        return {"message": "Scheduler not running"}

    scheduler.stop()
    return {"message": "Scheduler stopped"}


@router.post("/scheduler/trigger/friday")
async def trigger_friday():
    """Manually trigger Friday analysis step."""
    scheduler = get_weekend_cycle_scheduler()
    await scheduler.trigger_friday_analysis()
    return {"message": "Friday analysis triggered"}


@router.post("/scheduler/trigger/saturday")
async def trigger_saturday():
    """Manually trigger Saturday steps."""
    scheduler = get_weekend_cycle_scheduler()
    await scheduler.trigger_saturday_steps()
    return {"message": "Saturday steps triggered"}


@router.post("/scheduler/trigger/sunday")
async def trigger_sunday():
    """Manually trigger Sunday steps."""
    scheduler = get_weekend_cycle_scheduler()
    await scheduler.trigger_sunday_steps()
    return {"message": "Sunday steps triggered"}


@router.post("/scheduler/trigger/monday")
async def trigger_monday():
    """Manually trigger Monday roster deployment."""
    scheduler = get_weekend_cycle_scheduler()
    await scheduler.trigger_monday_deploy()
    return {"message": "Monday roster deployment triggered"}


# ============= Weekday Block Endpoints =============
@router.get("/block/status", response_model=BlockStatusResponse)
async def get_block_status():
    """Get current weekday parameter block status."""
    guard = get_weekday_parameter_guard()
    status = guard.get_block_status()

    return BlockStatusResponse(
        is_blocked=status["is_blocked"],
        current_day=status["current_day"],
        current_time=status["current_time"],
        next_allowed_window=status["next_allowed_window"],
        message=status["message"],
    )


@router.get("/block/check")
async def check_block():
    """Check if parameter changes are currently allowed."""
    guard = get_weekday_parameter_guard()
    is_allowed = guard.is_change_allowed()

    return {
        "change_allowed": is_allowed,
        "current_time": datetime.now(timezone.utc).strftime("%A %H:%M GMT"),
        "message": (
            "Parameter changes are allowed" if is_allowed
            else "Parameter changes are BLOCKED (weekday)"
        ),
    }


# ============= Roster Endpoints =============
@router.get("/roster/current")
async def get_current_roster():
    """Get current weekend roster (if prepared)."""
    manager = get_weekend_roster_manager()

    if manager._current_roster is None:
        return {"roster": None, "message": "No roster prepared yet"}

    return {
        "roster": manager._current_roster,
        "message": "Roster available"
    }


@router.post("/roster/prepare")
async def prepare_roster():
    """Prepare fresh roster for Monday deployment."""
    manager = get_weekend_roster_manager()

    try:
        roster = await manager.prepare_roster()

        if roster is None:
            return {"status": "failed", "message": "Failed to prepare roster"}

        return {
            "status": "prepared",
            "roster": roster,
            "message": "Roster prepared successfully"
        }

    except Exception as e:
        logger.error(f"Roster preparation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/roster/deploy")
async def deploy_roster():
    """Deploy roster to SessionDetector."""
    manager = get_weekend_roster_manager()

    try:
        result = await manager.deploy_roster()

        return {
            "status": result.status,
            "bots_deployed": result.bots_deployed,
            "deployed_at": result.deployed_at.isoformat() if result.deployed_at else None,
            "error": result.error,
        }

    except Exception as e:
        logger.error(f"Roster deployment failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============= SQS Warmup Endpoints =============
@router.get("/warmup/status")
async def get_warmup_status():
    """Get SQS Monday warmup status."""
    try:
        from src.router.sqs_monday_warmup import get_sqs_monday_warmup

        warmup = get_sqs_monday_warmup()
        return warmup.get_warmup_status()

    except Exception as e:
        logger.error(f"Error getting warmup status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/warmup/start")
async def start_warmup():
    """Manually start SQS Monday warmup."""
    try:
        from src.router.sqs_monday_warmup import get_sqs_monday_warmup

        warmup = get_sqs_monday_warmup()
        result = await warmup.execute_warmup()

        return result

    except Exception as e:
        logger.error(f"Error starting warmup: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

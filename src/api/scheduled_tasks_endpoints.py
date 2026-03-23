"""
Scheduled Tasks Endpoints - API for Weekend Compute Task Queries

This module provides API endpoints for querying weekend compute task status.
Used by FloorManager for "What's running this weekend?" queries.

Reference: Story 11-2-weekend-compute-protocol-scheduled-background-tasks
"""

import logging
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/scheduled-tasks", tags=["scheduled-tasks"])


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class TaskStatus:
    """Task status information for weekend compute queries."""
    task_name: str
    status: str  # "pending", "running", "completed", "failed", "retrying"
    progress_percent: float = 0.0
    estimated_completion: Optional[datetime] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    message: str = ""
    error: Optional[str] = None


class WeekendTaskResponse(BaseModel):
    """Response model for weekend task query."""
    query_time: datetime
    weekend_start: datetime
    tasks: List[Dict[str, Any]]
    total_tasks: int
    running_count: int
    completed_count: int
    failed_count: int


class WeekendTaskQueryRequest(BaseModel):
    """Request model for weekend task query."""
    weekend_date: Optional[str] = None  # Format: YYYY-MM-DD, defaults to current/most recent


# ============================================================================
# Task Status Calculation Helpers
# ============================================================================

def calculate_progress(run_start: datetime, estimated_duration: float) -> float:
    """Calculate task progress percentage.

    Args:
        run_start: When the task started
        estimated_duration: Estimated total duration in seconds

    Returns:
        Progress percentage (0-100)
    """
    elapsed = (datetime.now(timezone.utc) - run_start).total_seconds()
    progress = (elapsed / estimated_duration) * 100 if estimated_duration > 0 else 0
    return min(progress, 100.0)


def estimate_completion(
    run_start: datetime,
    estimated_duration: float,
    progress: float
) -> Optional[datetime]:
    """Estimate task completion time.

    Args:
        run_start: When the task started
        estimated_duration: Estimated total duration in seconds
        progress: Current progress percentage

    Returns:
        Estimated completion time or None if cannot estimate
    """
    if progress >= 100:
        return None

    remaining_seconds = estimated_duration * (1 - progress / 100)
    from datetime import timedelta
    return (datetime.now(timezone.utc) + timedelta(seconds=remaining_seconds)).replace(tzinfo=None)


# ============================================================================
# API Endpoints
# ============================================================================

@router.get("/weekend-tasks", response_model=WeekendTaskResponse)
async def get_weekend_tasks(
    weekend_date: Optional[str] = None
) -> WeekendTaskResponse:
    """
    Query weekend compute tasks - answers "What's running this weekend?"

    Returns list of tasks with status, progress, and estimated completion.

    Args:
        weekend_date: Optional specific weekend date (YYYY-MM-DD).
                     Defaults to most recent weekend.

    Returns:
        WeekendTaskResponse with task list and status summary
    """
    logger.info(f"Querying weekend tasks for: {weekend_date or 'current weekend'}")

    try:
        # Get Prefect client for task status
        from prefect import get_client
        client = get_client()

        # Determine weekend date
        if weekend_date:
            weekend = datetime.strptime(weekend_date, "%Y-%m-%d")
        else:
            # Default to most recent Saturday
            now = datetime.now(timezone.utc)
            days_since_saturday = (now.weekday() - 5) % 7
            weekend = now.replace(hour=0, minute=0, second=0, microsecond=0)
            if days_since_saturday != 0:
                weekend = weekend.replace(day=now.day - days_since_saturday)

        weekend_start = weekend.replace(tzinfo=None)
        weekend_end = weekend_start.replace(hour=23, minute=59, second=59)

        # Query Prefect for weekend compute flow runs
        try:
            flow_runs = await client.read_flow_runs(
                flow_name="weekend-compute-flow",
                limit=10
            )
        except Exception as e:
            logger.warning(f"Could not query Prefect: {e}")
            flow_runs = []

        # Build task status list
        tasks = []
        running_count = 0
        completed_count = 0
        failed_count = 0

        # Task duration estimates (in seconds)
        task_duration_estimates = {
            "monte_carlo_simulation": 600,  # 10 min
            "hmm_retraining": 1800,         # 30 min
            "pageindex_semantic": 1200,     # 20 min
            "correlation_refresh": 900      # 15 min
        }

        for run in flow_runs:
            run_start = run.start_time.replace(tzinfo=None) if run.start_time else None
            run_end = run.end_time.replace(tzinfo=None) if run.end_time else None

            if not run_start:
                continue

            # Determine task state
            state = run.state_name.lower() if run.state_name else "unknown"

            # Map Prefect states to our states
            if state in ["running", "pending"]:
                task_state = "running"
                running_count += 1
                progress = calculate_progress(
                    run_start,
                    task_duration_estimates.get("monte_carlo_simulation", 600)
                )
                eta = estimate_completion(run_start, 2700, progress)
            elif state in ["completed", "success"]:
                task_state = "completed"
                completed_count += 1
                progress = 100.0
                eta = None
            elif state in ["failed", "crashed"]:
                task_state = "failed"
                failed_count += 1
                progress = 100.0
                eta = None
            else:
                task_state = "pending"
                progress = 0.0
                eta = None

            # Calculate duration
            duration = 0.0
            if run_start:
                end = run_end or datetime.now(timezone.utc).replace(tzinfo=None)
                duration = (end - run_start).total_seconds()

            tasks.append({
                "run_id": run.id,
                "task_name": run.name or "weekend-compute-flow",
                "status": task_state,
                "progress_percent": progress,
                "estimated_completion": eta.isoformat() if eta else None,
                "start_time": run_start.isoformat() if run_start else None,
                "end_time": run_end.isoformat() if run_end else None,
                "duration_seconds": duration,
                "message": f"State: {state}",
                "error": None
            })

        # If no runs found, show the scheduled tasks
        if not tasks:
            # Return scheduled task info for upcoming weekend
            next_saturday = get_next_saturday()
            tasks = [
                {
                    "task_name": "monte_carlo_simulation",
                    "status": "scheduled",
                    "progress_percent": 0.0,
                    "estimated_completion": next_saturday.isoformat(),
                    "start_time": next_saturday.isoformat(),
                    "message": "Scheduled for Saturday 00:00 UTC"
                },
                {
                    "task_name": "hmm_retraining",
                    "status": "scheduled",
                    "progress_percent": 0.0,
                    "estimated_completion": next_saturday.isoformat(),
                    "start_time": next_saturday.isoformat(),
                    "message": "Scheduled for Saturday 00:00 UTC"
                },
                {
                    "task_name": "pageindex_semantic",
                    "status": "scheduled",
                    "progress_percent": 0.0,
                    "estimated_completion": next_saturday.isoformat(),
                    "start_time": next_saturday.isoformat(),
                    "message": "Scheduled for Saturday 00:00 UTC"
                },
                {
                    "task_name": "correlation_refresh",
                    "status": "scheduled",
                    "progress_percent": 0.0,
                    "estimated_completion": next_saturday.isoformat(),
                    "start_time": next_saturday.isoformat(),
                    "message": "Scheduled for Saturday 00:00 UTC"
                }
            ]
            running_count = 0
            completed_count = 0
            failed_count = 0

        return WeekendTaskResponse(
            query_time=datetime.now(timezone.utc).replace(tzinfo=None),
            weekend_start=weekend_start,
            tasks=tasks,
            total_tasks=len(tasks),
            running_count=running_count,
            completed_count=completed_count,
            failed_count=failed_count
        )

    except Exception as e:
        logger.error(f"Error querying weekend tasks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/task/{task_name}/status")
async def get_task_status(task_name: str) -> Dict[str, Any]:
    """Get status of a specific weekend task.

    Args:
        task_name: Name of the task (monte_carlo, hmm_retrain, pageindex, correlation)

    Returns:
        Task status information
    """
    logger.info(f"Querying task status for: {task_name}")

    try:
        from prefect import get_client
        client = get_client()

        # Map friendly names to Prefect task names
        task_map = {
            "monte_carlo": "monte_carlo_simulation",
            "hmm_retrain": "hmm_retraining",
            "hmm": "hmm_retraining",
            "pageindex": "pageindex_semantic",
            "correlation": "correlation_refresh"
        }

        prefect_task_name = task_map.get(task_name.lower(), task_name)

        # Query for recent flow runs
        flow_runs = await client.read_flow_runs(
            flow_name="weekend-compute-flow",
            limit=1
        )

        if not flow_runs:
            return {
                "task_name": task_name,
                "status": "no_runs",
                "message": "No recent weekend compute runs found"
            }

        run = flow_runs[0]
        state = run.state_name.lower() if run.state_name else "unknown"

        return {
            "task_name": task_name,
            "status": state,
            "run_id": run.id,
            "start_time": run.start_time.isoformat() if run.start_time else None,
            "end_time": run.end_time.isoformat() if run.end_time else None
        }

    except Exception as e:
        logger.error(f"Error getting task status: {e}")
        return {
            "task_name": task_name,
            "status": "error",
            "message": str(e)
        }


@router.post("/trigger")
async def trigger_weekend_compute() -> Dict[str, Any]:
    """Manually trigger weekend compute flow (for testing).

    Returns:
        Trigger confirmation
    """
    logger.info("Manual weekend compute trigger requested")

    try:
        from prefect import get_client
        client = get_client()

        # Create a flow run
        flow = await client.read_flow_by_name("weekend-compute-flow")
        deployment = await client.read_deployment_by_name("weekend-compute")

        # This would trigger the flow - commented out for safety
        # await client.create_flow_run(flow_id=flow.id, deployment_id=deployment.id)

        return {
            "status": "success",
            "message": "Weekend compute trigger submitted (disabled for safety)",
            "note": "Enable by uncommenting create_flow_run in production"
        }

    except Exception as e:
        logger.error(f"Error triggering weekend compute: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Helper Functions
# ============================================================================

def get_next_saturday() -> datetime:
    """Get next Saturday at 00:00 UTC.

    Returns:
        datetime of next Saturday 00:00 UTC
    """
    now = datetime.now(timezone.utc)
    days_until_saturday = (5 - now.weekday()) % 7
    if days_until_saturday == 0:
        days_until_saturday = 7  # Next week
    return now.replace(hour=0, minute=0, second=0, microsecond=0) + __import__('datetime').timedelta(days=days_until_saturday)


# ============================================================================
# Module Exports
# ============================================================================

__all__ = ["router", "TaskStatus", "WeekendTaskResponse", "WeekendTaskQueryRequest"]
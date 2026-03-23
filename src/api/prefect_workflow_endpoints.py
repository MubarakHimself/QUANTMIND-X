# Prefect Workflow API Endpoints
"""
Prefect workflow API endpoints for FlowForge canvas.

Provides:
- List all Prefect workflows with status
- Get workflow details (including task graph)
- Cancel specific workflow (per-card kill switch)
"""
from fastapi import APIRouter, HTTPException
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/prefect", tags=["prefect"])

# Workflow state enum matching the 6 columns
WORKFLOW_STATES = {
    "PENDING": "PENDING",
    "RUNNING": "RUNNING",
    "PENDING_REVIEW": "PENDING_REVIEW",
    "DONE": "DONE",
    "CANCELLED": "CANCELLED",
    "EXPIRED_REVIEW": "EXPIRED_REVIEW"
}


# Mock data for demonstration - in production, connect to Prefect API
MOCK_WORKFLOWS = [
    {
        "id": "flow-run-001",
        "flow_id": "flow-001",
        "name": "Alpha Research Pipeline",
        "department": "Research",
        "state": "RUNNING",
        "started_at": "2026-03-21T08:30:00Z",
        "duration_seconds": 3600,
        "completed_steps": 3,
        "total_steps": 8,
        "next_step": "Hypothesis Validation",
        "tasks": [
            {"id": "task-1", "name": "Data Ingestion", "state": "COMPLETED", "x": 100, "y": 50},
            {"id": "task-2", "name": "Feature Engineering", "state": "COMPLETED", "x": 250, "y": 50},
            {"id": "task-3", "name": "Model Training", "state": "RUNNING", "x": 400, "y": 50},
            {"id": "task-4", "name": "Hypothesis Validation", "state": "PENDING", "x": 550, "y": 50},
            {"id": "task-5", "name": "Backtest", "state": "PENDING", "x": 700, "y": 50},
            {"id": "task-6", "name": "Report Generation", "state": "PENDING", "x": 850, "y": 50},
        ],
        "dependencies": [
            {"from": "task-1", "to": "task-2"},
            {"from": "task-2", "to": "task-3"},
            {"from": "task-3", "to": "task-4"},
            {"from": "task-4", "to": "task-5"},
            {"from": "task-5", "to": "task-6"},
        ]
    },
    {
        "id": "flow-run-002",
        "flow_id": "flow-002",
        "name": "MQL5 Strategy Build",
        "department": "Development",
        "state": "PENDING",
        "started_at": None,
        "duration_seconds": 0,
        "completed_steps": 0,
        "total_steps": 5,
        "next_step": "Code Generation",
        "tasks": [
            {"id": "task-1", "name": "Strategy Spec", "state": "PENDING", "x": 100, "y": 50},
            {"id": "task-2", "name": "Code Generation", "state": "PENDING", "x": 250, "y": 50},
            {"id": "task-3", "name": "Compilation", "state": "PENDING", "x": 400, "y": 50},
            {"id": "task-4", "name": "Testing", "state": "PENDING", "x": 550, "y": 50},
            {"id": "task-5", "name": "Deployment", "state": "PENDING", "x": 700, "y": 50},
        ],
        "dependencies": [
            {"from": "task-1", "to": "task-2"},
            {"from": "task-2", "to": "task-3"},
            {"from": "task-3", "to": "task-4"},
            {"from": "task-4", "to": "task-5"},
        ]
    },
    {
        "id": "flow-run-003",
        "flow_id": "flow-003",
        "name": "Risk Analysis Batch",
        "department": "Risk",
        "state": "PENDING_REVIEW",
        "started_at": "2026-03-20T14:00:00Z",
        "duration_seconds": 7200,
        "completed_steps": 4,
        "total_steps": 4,
        "next_step": "Awaiting Human Approval",
        "tasks": [
            {"id": "task-1", "name": "Data Collection", "state": "COMPLETED", "x": 100, "y": 50},
            {"id": "task-2", "name": "Risk Calculation", "state": "COMPLETED", "x": 250, "y": 50},
            {"id": "task-3", "name": "Scenario Analysis", "state": "COMPLETED", "x": 400, "y": 50},
            {"id": "task-4", "name": "Report Review", "state": "COMPLETED", "x": 550, "y": 50},
        ],
        "dependencies": [
            {"from": "task-1", "to": "task-2"},
            {"from": "task-2", "to": "task-3"},
            {"from": "task-3", "to": "task-4"},
        ]
    },
    {
        "id": "flow-run-004",
        "flow_id": "flow-004",
        "name": "Portfolio Rebalance",
        "department": "Portfolio",
        "state": "DONE",
        "started_at": "2026-03-19T10:00:00Z",
        "duration_seconds": 1800,
        "completed_steps": 6,
        "total_steps": 6,
        "next_step": "Completed",
        "tasks": [
            {"id": "task-1", "name": "Position Analysis", "state": "COMPLETED", "x": 100, "y": 50},
            {"id": "task-2", "name": "Signal Generation", "state": "COMPLETED", "x": 250, "y": 50},
            {"id": "task-3", "name": "Order Planning", "state": "COMPLETED", "x": 400, "y": 50},
            {"id": "task-4", "name": "Execution", "state": "COMPLETED", "x": 550, "y": 50},
            {"id": "task-5", "name": "Reconciliation", "state": "COMPLETED", "x": 700, "y": 50},
            {"id": "task-6", "name": "Logging", "state": "COMPLETED", "x": 850, "y": 50},
        ],
        "dependencies": [
            {"from": "task-1", "to": "task-2"},
            {"from": "task-2", "to": "task-3"},
            {"from": "task-3", "to": "task-4"},
            {"from": "task-4", "to": "task-5"},
            {"from": "task-5", "to": "task-6"},
        ]
    },
    {
        "id": "flow-run-005",
        "flow_id": "flow-005",
        "name": "Backtest Runner",
        "department": "Development",
        "state": "CANCELLED",
        "started_at": "2026-03-21T06:00:00Z",
        "duration_seconds": 5400,
        "completed_steps": 2,
        "total_steps": 7,
        "next_step": "Cancelled",
        "tasks": [
            {"id": "task-1", "name": "Strategy Load", "state": "COMPLETED", "x": 100, "y": 50},
            {"id": "task-2", "name": "Data Fetch", "state": "COMPLETED", "x": 250, "y": 50},
            {"id": "task-3", "name": "Run Backtest", "state": "CANCELLED", "x": 400, "y": 50},
            {"id": "task-4", "name": "Metrics Calc", "state": "CANCELLED", "x": 550, "y": 50},
            {"id": "task-5", "name": "Optimize", "state": "PENDING", "x": 700, "y": 50},
            {"id": "task-6", "name": "Report", "state": "PENDING", "x": 850, "y": 50},
            {"id": "task-7", "name": "Archive", "state": "PENDING", "x": 1000, "y": 50},
        ],
        "dependencies": [
            {"from": "task-1", "to": "task-2"},
            {"from": "task-2", "to": "task-3"},
            {"from": "task-3", "to": "task-4"},
            {"from": "task-4", "to": "task-5"},
            {"from": "task-5", "to": "task-6"},
            {"from": "task-6", "to": "task-7"},
        ]
    },
    {
        "id": "flow-run-006",
        "flow_id": "flow-006",
        "name": "News Sentiment Analysis",
        "department": "Research",
        "state": "EXPIRED_REVIEW",
        "started_at": "2026-03-18T16:00:00Z",
        "duration_seconds": 3600,
        "completed_steps": 3,
        "total_steps": 3,
        "next_step": "Review Expired",
        "tasks": [
            {"id": "task-1", "name": "News Fetch", "state": "COMPLETED", "x": 100, "y": 50},
            {"id": "task-2", "name": "Sentiment Analysis", "state": "COMPLETED", "x": 250, "y": 50},
            {"id": "task-3", "name": "Signal Generation", "state": "COMPLETED", "x": 400, "y": 50},
        ],
        "dependencies": [
            {"from": "task-1", "to": "task-2"},
            {"from": "task-2", "to": "task-3"},
        ]
    },
]


@router.get("/workflows")
async def list_workflows() -> Dict[str, Any]:
    """
    List all Prefect workflows with status.

    Returns workflows grouped by state for the Kanban board.
    """
    try:
        # Group by state for Kanban columns
        workflows_by_state = {
            "PENDING": [],
            "RUNNING": [],
            "PENDING_REVIEW": [],
            "DONE": [],
            "CANCELLED": [],
            "EXPIRED_REVIEW": []
        }

        for workflow in MOCK_WORKFLOWS:
            state = workflow.get("state", "PENDING")
            if state in workflows_by_state:
                workflows_by_state[state].append(workflow)
            else:
                workflows_by_state["PENDING"].append(workflow)

        return {
            "workflows": MOCK_WORKFLOWS,
            "by_state": workflows_by_state,
            "total": len(MOCK_WORKFLOWS)
        }
    except Exception as e:
        logger.error(f"Error listing workflows: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/workflows/{workflow_id}")
async def get_workflow(workflow_id: str) -> Dict[str, Any]:
    """
    Get detailed workflow information including task graph.
    """
    try:
        for workflow in MOCK_WORKFLOWS:
            if workflow["id"] == workflow_id:
                return workflow

        raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting workflow {workflow_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/workflows/{workflow_id}/cancel")
async def cancel_workflow(workflow_id: str) -> Dict[str, Any]:
    """
    Cancel a specific Prefect workflow.

    This is the per-card workflow kill switch - only cancels the specified workflow,
    leaving all other workflows and live trading unaffected.
    """
    try:
        # Find workflow
        workflow = None
        for wf in MOCK_WORKFLOWS:
            if wf["id"] == workflow_id:
                workflow = wf
                break

        if not workflow:
            raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")

        # Only running workflows can be cancelled
        if workflow["state"] != "RUNNING":
            raise HTTPException(
                status_code=400,
                detail=f"Cannot cancel workflow in state: {workflow['state']}"
            )

        # Simulate cancellation
        # In production, this would call: await client.cancel_flow_run(workflow_id)
        workflow["state"] = "CANCELLED"

        logger.info(f"Workflow {workflow_id} cancelled successfully")

        return {
            "success": True,
            "workflow_id": workflow_id,
            "message": f"Workflow '{workflow['name']}' has been cancelled",
            "previous_state": "RUNNING",
            "new_state": "CANCELLED"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling workflow {workflow_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/workflows/{workflow_id}/resume")
async def resume_workflow(workflow_id: str) -> Dict[str, Any]:
    """
    Resume a cancelled workflow from the last completed step.

    This implements the /resume-workflow command pattern.
    """
    try:
        workflow = None
        for wf in MOCK_WORKFLOWS:
            if wf["id"] == workflow_id:
                workflow = wf
                break

        if not workflow:
            raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")

        if workflow["state"] != "CANCELLED":
            raise HTTPException(
                status_code=400,
                detail=f"Cannot resume workflow in state: {workflow['state']}"
            )

        # Resume from last completed step
        workflow["state"] = "RUNNING"

        return {
            "success": True,
            "workflow_id": workflow_id,
            "message": f"Workflow '{workflow['name']}' resumed from step {workflow['completed_steps']}",
            "resumed_from_step": workflow["completed_steps"]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resuming workflow {workflow_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
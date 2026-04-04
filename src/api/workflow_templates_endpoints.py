"""
Workflow Templates API

Provides predefined workflow templates that can be triggered via the Copilot.
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List

from src.agents.departments.floor_manager import get_floor_manager
from src.agents.departments.types import Department

router = APIRouter(prefix="/api/workflow-templates")

WORKFLOW_TEMPLATES = [
    {
        "id": "weekly_war_room",
        "name": "Weekly War Room",
        "description": "Full weekly review: DPR ranking audit, underperformer review, strategy retirement decisions, Alpha Forge queue check",
        "trigger_message": "Run weekly war room review. Audit all strategy DPR scores from the past 7 days. Flag any strategies with SESSION_CONCERN. Identify bottom-quartile performers. Check Alpha Forge queue for any pending paper trade graduations. Generate weekly summary.",
        "departments": ["portfolio", "research", "risk"],
        "estimated_duration": "15-20 minutes",
    },
    {
        "id": "alpha_forge_queue_check",
        "name": "Alpha Forge Queue Check",
        "description": "Check for pending TRDs, stuck backtests, and paper trade reviews",
        "trigger_message": "Check Alpha Forge queue. List all workflows by status. Flag any stuck in PENDING_REVIEW over 24 hours. Flag any backtests in progress over 2 hours.",
        "departments": ["development"],
        "estimated_duration": "2-3 minutes",
    }
]


@router.get("")
async def get_workflow_templates():
    """Get all available workflow templates."""
    return WORKFLOW_TEMPLATES


@router.post("/{template_id}/trigger")
async def trigger_workflow_template(template_id: str, background_tasks: BackgroundTasks):
    """
    Trigger a workflow template by ID.

    Dispatches the template's trigger_message to each listed department via
    FloorManager so the trading floor receives and processes the workflow.
    """
    template = next((t for t in WORKFLOW_TEMPLATES if t["id"] == template_id), None)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")

    manager = get_floor_manager()
    dispatch_results = []
    for dept in template["departments"]:
        result = manager.dispatch(
            to_dept=Department(dept),
            task=template["trigger_message"],
            priority="normal",
        )
        dispatch_results.append({"department": dept, "result": result})

    return {
        "status": "triggered",
        "template_id": template_id,
        "message": template["trigger_message"],
        "dispatch_results": dispatch_results,
    }

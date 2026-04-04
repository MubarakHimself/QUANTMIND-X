"""
Copilot Kill Switch API Endpoints

API endpoints for Copilot/agent kill switch - independent from trading kill switch.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime

from src.router.copilot_kill_switch import (
    get_copilot_kill_switch,
    CopilotKillSwitch,
)

router = APIRouter(prefix="/api")


# Request/Response models
class CopilotKillSwitchActivateRequest(BaseModel):
    """Request to activate copilot kill switch."""
    activator: str = "user"


class CopilotKillSwitchResponse(BaseModel):
    """Response for copilot kill switch operations."""
    success: bool
    suspended_at_utc: Optional[str] = None
    activated_by: Optional[str] = None
    terminated_tasks: List[str] = []
    already_active: Optional[bool] = None


class CopilotKillSwitchResumeResponse(BaseModel):
    """Response for resuming copilot."""
    success: bool
    resumed_at_utc: Optional[str] = None
    not_active: Optional[bool] = None


class CopilotKillSwitchStatusResponse(BaseModel):
    """Response for kill switch status check."""
    active: bool
    suspended_at_utc: Optional[str] = None
    activated_by: Optional[str] = None
    terminated_tasks_count: int


@router.post("/copilot/kill-switch", response_model=CopilotKillSwitchResponse)
async def activate_copilot_kill_switch(
    request: CopilotKillSwitchActivateRequest
) -> Dict[str, Any]:
    """
    Activate the copilot kill switch.

    This will:
    - Terminate all running FloorManager and department agent tasks
    - Show "Agent activity suspended" in the UI
    - NOT affect live trading
    """
    kill_switch = get_copilot_kill_switch()
    result = await kill_switch.activate(activator=request.activator)

    return result


@router.post("/copilot/kill-switch/resume", response_model=CopilotKillSwitchResumeResponse)
async def resume_copilot() -> Dict[str, Any]:
    """
    Resume copilot - reactivate the agent system.
    """
    kill_switch = get_copilot_kill_switch()
    result = await kill_switch.resume()

    return result


@router.get("/copilot/kill-switch/status", response_model=CopilotKillSwitchStatusResponse)
async def get_copilot_kill_switch_status() -> Dict[str, Any]:
    """
    Get current copilot kill switch status.
    """
    kill_switch = get_copilot_kill_switch()
    status = kill_switch.get_status()

    return status


@router.get("/copilot/kill-switch/history")
async def get_copilot_kill_switch_history() -> List[Dict[str, Any]]:
    """
    Get copilot kill switch activation history.
    """
    kill_switch = get_copilot_kill_switch()
    return kill_switch.get_history()

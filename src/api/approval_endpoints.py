"""
Human-in-the-Loop Approval API Endpoints

REST + SSE endpoints for the Svelte UI to display pending approvals
and allow the human to approve/reject them.

Endpoints:
    GET  /api/approvals/pending        — List all pending approval requests
    GET  /api/approvals/history         — Get resolved approval history
    GET  /api/approvals/count           — Get pending approval count (badge)
    GET  /api/approvals/{id}            — Get specific approval request
    GET  /api/approvals/{id}/resume     — Get resume context for an approval
    POST /api/approvals/{id}/approve    — Approve a request
    POST /api/approvals/{id}/reject     — Reject a request
    POST /api/approvals/{id}/cancel     — Cancel a request
    GET  /api/approvals/stream          — SSE stream of approval events
    POST /api/approvals/workflow-gate   — Resolve a workflow gate
    POST /api/approvals/resume          — Resume all pending approvals from DB
    POST /api/approvals/resume-workflows — Resume paused workflows from DB

Architecture reference: PRD §11.8 human-in-the-loop
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.agents.approval_manager import get_approval_manager, ApprovalStatus

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/approvals", tags=["approvals"])


# =============================================================================
# Request Models
# =============================================================================

class RejectRequest(BaseModel):
    """Body for rejecting an approval."""
    reason: str = ""


class WorkflowGateRequest(BaseModel):
    """Body for resolving a workflow gate."""
    workflow_id: str
    approved: bool
    resolved_by: str = "human"
    reason: str = ""


# =============================================================================
# REST Endpoints
# =============================================================================

@router.get("/pending")
async def list_pending_approvals():
    """Get all pending approval requests.

    Returns a flat array of approval objects for the UI.
    """
    return get_approval_manager().get_pending()


@router.get("/count")
async def get_pending_count():
    """Get count of pending approvals (for badge display).

    Returns:
        {"count": N}
    """
    return {"count": get_approval_manager().get_pending_count()}


@router.get("/history")
async def get_approval_history(limit: int = 50):
    """Get resolved approval history.

    Query Params:
        limit: Max entries to return (default 50)
    """
    return get_approval_manager().get_history(limit=limit)


@router.get("/{approval_id}")
async def get_approval(approval_id: str):
    """Get a specific approval request by ID."""
    req = get_approval_manager().get_request(approval_id)
    if not req:
        raise HTTPException(status_code=404, detail="Approval request not found")
    return req


@router.post("/{approval_id}/approve")
async def approve_request(approval_id: str):
    """Approve a pending approval request.

    Unblocks any agent/workflow awaiting this approval.
    """
    success = get_approval_manager().approve(approval_id, approved_by="human")
    if not success:
        raise HTTPException(
            status_code=400,
            detail="Approval request not found or not pending",
        )
    return {"status": "approved", "approval_id": approval_id}


@router.post("/{approval_id}/reject")
async def reject_request(approval_id: str, body: RejectRequest):
    """Reject a pending approval request.

    Unblocks any agent/workflow awaiting this approval.
    """
    success = get_approval_manager().reject(
        approval_id, reason=body.reason, rejected_by="human",
    )
    if not success:
        raise HTTPException(
            status_code=400,
            detail="Approval request not found or not pending",
        )
    return {"status": "rejected", "approval_id": approval_id}


@router.post("/{approval_id}/cancel")
async def cancel_request(approval_id: str):
    """Cancel a pending approval request (system-initiated)."""
    success = get_approval_manager().cancel(approval_id)
    if not success:
        raise HTTPException(
            status_code=400,
            detail="Approval request not found or not pending",
        )
    return {"status": "cancelled", "approval_id": approval_id}


# =============================================================================
# Workflow Gate Resolution
# =============================================================================

@router.post("/workflow-gate")
async def resolve_workflow_gate(body: WorkflowGateRequest):
    """Resolve a workflow HITL gate (SIT_GATE or HUMAN_APPROVAL_GATE).

    This both resolves the approval request AND advances/fails the workflow.
    """
    try:
        from src.agents.departments.workflow_coordinator import get_workflow_coordinator
        coord = get_workflow_coordinator()
        result = coord.resolve_gate(
            workflow_id=body.workflow_id,
            approved=body.approved,
            resolved_by=body.resolved_by,
            reason=body.reason,
        )
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Workflow gate resolution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Resume / Recovery Endpoints
# =============================================================================

@router.get("/{approval_id}/resume")
async def get_resume_context(approval_id: str):
    """Get the resume context for a specific approval request.

    Returns the saved snapshot needed to restart a paused workflow or agent.
    """
    ctx = get_approval_manager().get_resume_context(approval_id)
    if ctx is None:
        raise HTTPException(
            status_code=404,
            detail="No resume context found for this approval",
        )
    return {"approval_id": approval_id, "resume_context": ctx}


@router.post("/resume")
async def resume_pending_approvals():
    """Resume all pending approvals from the database after server restart.

    Reloads pending approval requests from DB, re-creates in-memory state,
    and re-sends SSE + department mail notifications.

    Returns the count of approvals resumed.
    """
    count = get_approval_manager().resume_pending()
    return {"resumed_count": count, "pending_count": get_approval_manager().get_pending_count()}


@router.post("/resume-workflows")
async def resume_waiting_workflows():
    """Resume all paused workflows from the database after server restart.

    Reloads workflows paused at HITL gates, re-creates in-memory workflow
    state, re-sends department mail reminders, and re-attaches approval
    request IDs.

    Returns the count of workflows resumed.
    """
    try:
        from src.agents.departments.workflow_coordinator import get_workflow_coordinator
        coord = get_workflow_coordinator()
        count = coord.resume_waiting_workflows()
        return {
            "resumed_workflows": count,
            "active_workflows": len(coord.get_active_workflows()),
        }
    except Exception as e:
        logger.error(f"Workflow resume failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# SSE Stream for Approval Events
# =============================================================================

@router.get("/stream")
async def stream_approval_events():
    """SSE endpoint for real-time approval notifications.

    The UI can connect to this to get notified when:
    - New approval requests are created
    - Approval requests are resolved
    - Approval requests expire

    Events are JSON-encoded with type: "approval_event".
    """
    queue: asyncio.Queue = asyncio.Queue(maxsize=50)

    # Register callback to push events to this stream
    def on_approval(req):
        try:
            queue.put_nowait(req.to_dict())
        except asyncio.QueueFull:
            pass

    mgr = get_approval_manager()
    mgr.on_request(on_approval)

    async def event_generator():
        # Send initial state
        pending = mgr.get_pending()
        yield f"data: {json.dumps({'type': 'init', 'pending': pending})}\n\n"

        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30)
                    yield f"data: {json.dumps({'type': 'approval_event', 'approval': event})}\n\n"
                except asyncio.TimeoutError:
                    yield f"data: {json.dumps({'type': 'heartbeat', 'pending_count': mgr.get_pending_count()})}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            # Remove callback (best effort)
            try:
                mgr._on_request_callbacks.remove(on_approval)
            except ValueError:
                pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

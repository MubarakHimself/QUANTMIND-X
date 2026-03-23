"""Canvas Context API Endpoints.

Provides endpoints for loading CanvasContextTemplate and assembling canvas context.
"""
import logging
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.canvas_context.loader import (
    load_template,
    get_all_templates,
    get_canvas_list,
    SUPPORTED_CANVASES,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/canvas-context", tags=["canvas-context"])


class CanvasContextRequest(BaseModel):
    """Request to load canvas context."""

    canvas: str
    session_id: Optional[str] = None
    include_memory_identifiers: bool = True


class CanvasContextResponse(BaseModel):
    """Response with canvas context."""

    canvas: str
    template: dict
    memory_identifiers: list[str] = []
    session_id: Optional[str] = None
    loaded_at: str


@router.get("/templates")
async def list_templates():
    """List all available canvas templates."""
    try:
        templates = get_all_templates()
        return {
            "templates": [
                {
                    "canvas": name,
                    "display_name": t.canvas_display_name,
                    "icon": t.canvas_icon,
                    "department_head": t.department_head,
                }
                for name, t in templates.items()
            ]
        }
    except Exception as e:
        logger.error(f"Failed to list templates: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/canvases")
async def list_canvases():
    """List all available canvases."""
    try:
        canvases = get_canvas_list()
        return {"canvases": canvases}
    except Exception as e:
        logger.error(f"Failed to list canvases: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/template/{canvas_name}")
async def get_template(canvas_name: str):
    """Get a specific canvas template.

    Args:
        canvas_name: Canvas identifier (e.g., 'risk', 'live_trading')
    """
    try:
        template = load_template(canvas_name)
        return {"template": template.model_dump()}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to load template: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/load")
async def load_canvas_context(request: CanvasContextRequest):
    """Load canvas context including template and memory identifiers.

    This endpoint loads the canvas-specific template and optionally retrieves
    memory identifiers from the graph for the given session.
    """
    try:
        # Load template
        template = load_template(request.canvas)

        # Get memory identifiers if requested
        memory_identifiers = []
        if request.include_memory_identifiers:
            # Get session ID or generate one
            session_id = request.session_id or str(uuid4())

            # Try to get committed nodes from graph memory
            try:
                from src.memory.graph.facade import get_graph_memory

                graph_memory = get_graph_memory()
                committed_state = graph_memory.load_committed_state(
                    session_id=session_id,
                    include_content=False  # Only identifiers, not content
                )
                memory_identifiers = [
                    node["id"] for node in committed_state.get("nodes", [])
                ]
            except Exception as e:
                logger.warning(f"Failed to load memory identifiers: {e}")
                memory_identifiers = []

        # Build response
        from datetime import datetime, timezone

        return CanvasContextResponse(
            canvas=request.canvas,
            template=template.model_dump(),
            memory_identifiers=memory_identifiers[: template.max_identifiers],  # Enforce token budget
            session_id=request.session_id,
            loaded_at=datetime.now(timezone.utc).isoformat(),
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to load canvas context: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check for canvas context system."""
    return {
        "status": "healthy",
        "supported_canvases": list(SUPPORTED_CANVASES),
    }
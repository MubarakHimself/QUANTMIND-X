"""
Tool Forge API Endpoints

REST interface for the dynamic Tool Forge system.
Allows agents and operators to create, list, execute, and deactivate runtime tools.
"""
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.agents.tools.tool_forge import get_tool_forge, ForgeTool

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/tool-forge", tags=["tool-forge"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class RegisterToolRequest(BaseModel):
    name: str
    description: str
    input_schema: Dict[str, Any] = {}
    implementation: str
    department: str = "shared"
    created_by: str = "system"


class ExecuteToolRequest(BaseModel):
    inputs: Dict[str, Any] = {}


def _tool_to_dict(tool: ForgeTool) -> Dict[str, Any]:
    return {
        "tool_id": tool.tool_id,
        "name": tool.name,
        "description": tool.description,
        "department": tool.department,
        "input_schema": tool.input_schema,
        "created_by": tool.created_by,
        "created_at": tool.created_at,
        "usage_count": tool.usage_count,
        "active": tool.active,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/tools")
def list_tools(department: Optional[str] = None) -> Dict[str, Any]:
    """List all active tools, optionally filtered by department."""
    try:
        forge = get_tool_forge()
        tools = forge.list_tools(department=department)
        return {
            "tools": [_tool_to_dict(t) for t in tools],
            "count": len(tools),
            "department_filter": department,
        }
    except Exception as e:
        logger.error(f"Failed to list tools: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tools", status_code=201)
def register_tool(body: RegisterToolRequest) -> Dict[str, Any]:
    """Register a new forged tool."""
    try:
        forge = get_tool_forge()
        tool = forge.register_tool(
            name=body.name,
            description=body.description,
            input_schema=body.input_schema,
            implementation=body.implementation,
            department=body.department,
            created_by=body.created_by,
        )
        return {"status": "registered", "tool": _tool_to_dict(tool)}
    except Exception as e:
        logger.error(f"Failed to register tool '{body.name}': {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/tools/{name}/execute")
def execute_tool(name: str, body: ExecuteToolRequest) -> Dict[str, Any]:
    """Execute a forged tool by name with given inputs."""
    forge = get_tool_forge()
    tool = forge.get_tool(name)
    if not tool:
        raise HTTPException(status_code=404, detail=f"Tool '{name}' not found")
    result = forge.execute_tool(name, body.inputs)
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])
    return result


@router.delete("/tools/{name}")
def deactivate_tool(name: str) -> Dict[str, Any]:
    """Deactivate a tool (soft delete)."""
    forge = get_tool_forge()
    deactivated = forge.deactivate_tool(name)
    if not deactivated:
        raise HTTPException(status_code=404, detail=f"Tool '{name}' not found or already inactive")
    return {"status": "deactivated", "name": name}

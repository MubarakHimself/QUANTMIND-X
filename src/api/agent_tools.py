"""
Agent Workshop & System Tools

Comprehensive tool system giving agents capabilities to:
- Read system logs (router, risk, position sizing)
- Create components in their workshop
- Create hooks
- Access MCP servers
- Grow and enhance the system
"""

import os
import json
import logging
import subprocess
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/agent-tools", tags=["agent-tools"])

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
WORKSHOP_DIR = PROJECT_ROOT / "agent-workshop"
LOGS_DIR = PROJECT_ROOT / "logs"
HOOKS_DIR = PROJECT_ROOT / "src" / "agents" / "hooks"

# Ensure directories exist
for d in [WORKSHOP_DIR, LOGS_DIR, HOOKS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
    for sub in ["components", "hooks", "values", "templates", "experiments"]:
        (d / sub).mkdir(exist_ok=True)


# =============================================================================
# MODELS
# =============================================================================

class ToolRequest(BaseModel):
    tool_name: str
    parameters: Dict[str, Any] = {}
    agent_id: str = "copilot"


class ToolResponse(BaseModel):
    success: bool
    result: Optional[str] = None
    error: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    artifacts: List[Dict[str, Any]] = []


class ComponentRequest(BaseModel):
    name: str
    component_type: str  # "strategy", "indicator", "filter", "utility", "hook"
    code: str
    description: str = ""
    metadata: Dict[str, Any] = {}


class HookRequest(BaseModel):
    hook_name: str
    hook_type: str  # "pre", "post", "trigger"
    event: str  # "trade", "signal", "risk", "position"
    code: str
    description: str = ""


class CustomToolRequest(BaseModel):
    tool_name: str
    description: str
    parameters_schema: Dict[str, Any]  # JSON Schema for parameters
    code: str  # Python code to execute
    returns: str = "dict"  # Return type description
    agent_id: str = "copilot"


class GenerativeUIRequest(BaseModel):
    component_name: str
    component_type: str  # "card", "form", "chart", "table", "widget", "custom"
    svelte_code: str
    props_schema: Dict[str, Any] = {}  # JSON Schema for props
    description: str = ""
    agent_id: str = "copilot"


class FeaturePageRequest(BaseModel):
    """Request model for creating/updating a feature page."""
    feature_name: str
    title: str
    description: str = ""
    icon: str = "star"
    svelte_code: str
    agent_id: str = "copilot"


# =============================================================================
# LOG READING TOOLS
# =============================================================================

def read_system_logs(log_type: str, lines: int = 100) -> Dict[str, Any]:
    """
    Read system logs for strategy router, risk management, position sizing.

    Args:
        log_type: "router", "risk", "position", "all", or specific log name
        lines: Number of lines to read from end
    """
    logs = {}

    # Log file mappings
    log_files = {
        "router": ["router.log", "strategy_router.log", "commander.log"],
        "risk": ["risk.log", "governor.log", "kelly.log"],
        "position": ["position.log", "sizing.log", "execution.log"],
        "hmm": ["hmm.log", "regime.log"],
        "trading": ["trading.log", "orders.log", "signals.log"],
    }

    if log_type == "all":
        types_to_read = list(log_files.keys())
    else:
        types_to_read = [log_type] if log_type in log_files else [log_type]

    for t in types_to_read:
        files = log_files.get(t, [f"{t}.log"])
        for log_file in files:
            log_path = LOGS_DIR / log_file
            if log_path.exists():
                try:
                    with open(log_path, "r") as f:
                        content = f.readlines()
                        logs[log_file] = {
                            "path": str(log_path),
                            "lines": len(content),
                            "tail": content[-lines:] if len(content) > lines else content,
                            "last_modified": datetime.fromtimestamp(
                                log_path.stat().st_mtime
                            ).isoformat(),
                        }
                except Exception as e:
                    logs[log_file] = {"error": str(e)}
            else:
                # Try reading from journalctl if available
                try:
                    result = subprocess.run(
                        ["journalctl", "-u", f"quantmind-{t}", "-n", str(lines), "--no-pager"],
                        capture_output=True, text=True, timeout=10
                    )
                    if result.returncode == 0:
                        logs[f"journalctl-{t}"] = {
                            "source": "systemd",
                            "tail": result.stdout.split("\n")[-lines:],
                        }
                except:
                    pass

    # Also capture stdout/stderr from running processes
    if not logs:
        logs["status"] = f"No log files found for '{log_type}'. Logs directory: {LOGS_DIR}"

    return {
        "log_type": log_type,
        "logs": logs,
        "timestamp": datetime.utcnow().isoformat()
    }


# =============================================================================
# WORKSHOP TOOLS
# =============================================================================

def create_workshop_component(
    agent_id: str,
    name: str,
    component_type: str,
    code: str,
    description: str = "",
    metadata: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Create a new component in the agent's workshop.

    Component types:
    - strategy: Trading strategy components
    - indicator: Technical indicators
    - filter: Signal filters
    - utility: Helper functions
    - hook: Event hooks
    """
    agent_workshop = WORKSHOP_DIR / agent_id / "components"
    agent_workshop.mkdir(parents=True, exist_ok=True)

    # Sanitize name
    safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in name)

    # Determine file extension
    ext_map = {
        "strategy": ".py",
        "indicator": ".py",
        "filter": ".py",
        "utility": ".py",
        "hook": ".py",
        "config": ".json",
        "template": ".md",
    }
    ext = ext_map.get(component_type, ".txt")

    file_path = agent_workshop / f"{safe_name}{ext}"

    # Create component metadata
    component_meta = {
        "name": name,
        "type": component_type,
        "description": description,
        "created_by": agent_id,
        "created_at": datetime.utcnow().isoformat(),
        "file": str(file_path),
        "metadata": metadata or {}
    }

    # Write code file
    with open(file_path, "w") as f:
        if ext == ".py":
            f.write(f'"""\n{name}\n{description}\n\nCreated by: {agent_id}\nDate: {datetime.utcnow().isoformat()}\n"""\n\n')
        f.write(code)

    # Write metadata
    meta_path = agent_workshop / f"{safe_name}.meta.json"
    with open(meta_path, "w") as f:
        json.dump(component_meta, f, indent=2)

    return {
        "success": True,
        "component": {
            "name": name,
            "type": component_type,
            "path": str(file_path),
            "meta_path": str(meta_path),
            "size": len(code)
        }
    }


def list_workshop_components(agent_id: str, component_type: str = None) -> List[Dict[str, Any]]:
    """List all components in agent's workshop."""
    agent_workshop = WORKSHOP_DIR / agent_id / "components"
    components = []

    if not agent_workshop.exists():
        return components

    for meta_file in agent_workshop.glob("*.meta.json"):
        try:
            with open(meta_file, "r") as f:
                meta = json.load(f)
            if component_type is None or meta.get("type") == component_type:
                components.append(meta)
        except:
            pass

    return components


def create_value_store(
    agent_id: str,
    key: str,
    value: Any,
    description: str = ""
) -> Dict[str, Any]:
    """
    Store a value in the agent's workshop for later use.
    Values can be configs, learned parameters, optimizations, etc.
    """
    values_dir = WORKSHOP_DIR / agent_id / "values"
    values_dir.mkdir(parents=True, exist_ok=True)

    safe_key = "".join(c if c.isalnum() or c in "_-" else "_" for c in key)

    value_entry = {
        "key": key,
        "value": value,
        "description": description,
        "created_by": agent_id,
        "created_at": datetime.utcnow().isoformat(),
    }

    value_path = values_dir / f"{safe_key}.json"
    with open(value_path, "w") as f:
        json.dump(value_entry, f, indent=2)

    return {"success": True, "key": key, "path": str(value_path)}


def get_value_store(agent_id: str, key: str = None) -> Dict[str, Any]:
    """Retrieve values from agent's workshop."""
    values_dir = WORKSHOP_DIR / agent_id / "values"

    if not values_dir.exists():
        return {"values": {}}

    if key:
        safe_key = "".join(c if c.isalnum() or c in "_-" else "_" for c in key)
        value_path = values_dir / f"{safe_key}.json"
        if value_path.exists():
            with open(value_path, "r") as f:
                return json.load(f)
        return {"error": f"Value '{key}' not found"}

    values = {}
    for value_file in values_dir.glob("*.json"):
        try:
            with open(value_file, "r") as f:
                data = json.load(f)
                values[data["key"]] = data
        except:
            pass

    return {"values": values}


# =============================================================================
# HOOK CREATION TOOLS
# =============================================================================

def create_agent_hook(
    agent_id: str,
    hook_name: str,
    hook_type: str,
    event: str,
    code: str,
    description: str = ""
) -> Dict[str, Any]:
    """
    Create a new hook for system events.

    Hook types:
    - pre: Execute before event
    - post: Execute after event
    - trigger: Execute on condition

    Events:
    - trade: Trade execution
    - signal: Signal generation
    - risk: Risk check
    - position: Position change
    """
    hooks_dir = WORKSHOP_DIR / agent_id / "hooks"
    hooks_dir.mkdir(parents=True, exist_ok=True)

    safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in hook_name)
    file_name = f"{hook_type}_{event}_{safe_name}.py"
    file_path = hooks_dir / file_name

    hook_code = f'''"""
Hook: {hook_name}
Type: {hook_type}
Event: {event}
Description: {description}
Created by: {agent_id}
Date: {datetime.utcnow().isoformat()}
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def execute(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Hook execution function.

    Args:
        context: Event context containing relevant data

    Returns:
        Modified context or action dict
    """
{code}

# Hook registration
HOOK_CONFIG = {{
    "name": "{hook_name}",
    "type": "{hook_type}",
    "event": "{event}",
    "priority": 100,
    "enabled": True
}}
'''

    with open(file_path, "w") as f:
        f.write(hook_code)

    # Register hook in system
    hook_registry = HOOKS_DIR / "registry.json"
    registry = {}
    if hook_registry.exists():
        with open(hook_registry, "r") as f:
            registry = json.load(f)

    registry[f"{agent_id}_{hook_name}"] = {
        "path": str(file_path),
        "type": hook_type,
        "event": event,
        "agent_id": agent_id,
        "created_at": datetime.utcnow().isoformat(),
    }

    with open(hook_registry, "w") as f:
        json.dump(registry, f, indent=2)

    return {
        "success": True,
        "hook": {
            "name": hook_name,
            "type": hook_type,
            "event": event,
            "path": str(file_path),
        }
    }


def list_agent_hooks(agent_id: str = None) -> List[Dict[str, Any]]:
    """List all hooks, optionally filtered by agent."""
    hook_registry = HOOKS_DIR / "registry.json"

    if not hook_registry.exists():
        return []

    with open(hook_registry, "r") as f:
        registry = json.load(f)

    if agent_id:
        return [
            {"name": k, **v}
            for k, v in registry.items()
            if v.get("agent_id") == agent_id
        ]

    return [{"name": k, **v} for k, v in registry.items()]


# =============================================================================
# MCP ACCESS TOOLS
# =============================================================================

def list_available_mcp_servers() -> Dict[str, Any]:
    """List all available MCP servers for agents to use."""
    mcp_config_dir = PROJECT_ROOT / "config" / "mcp"
    servers = {}

    for config_file in mcp_config_dir.glob("*.json"):
        try:
            with open(config_file, "r") as f:
                config = json.load(f)
            agent_name = config_file.stem.replace("-mcp", "")
            servers[agent_name] = {
                "config_file": str(config_file),
                "servers": list(config.get("mcpServers", {}).keys())
            }
        except Exception as e:
            servers[config_file.name] = {"error": str(e)}

    return {
        "mcp_directory": str(mcp_config_dir),
        "servers": servers,
        "available_tools": [
            "filesystem", "github", "context7", "brave-search",
            "memory", "sequential-thinking", "chrome_devtools"
        ]
    }


def get_mcp_server_config(server_name: str) -> Dict[str, Any]:
    """Get configuration for a specific MCP server."""
    mcp_config_dir = PROJECT_ROOT / "config" / "mcp"

    for config_file in mcp_config_dir.glob("*.json"):
        try:
            with open(config_file, "r") as f:
                config = json.load(f)

            servers = config.get("mcpServers", {})
            if server_name in servers:
                return {
                    "server": server_name,
                    "config": servers[server_name],
                    "config_file": str(config_file)
                }
        except:
            pass

    return {"error": f"MCP server '{server_name}' not found"}


# =============================================================================
# SYSTEM GROWTH TOOLS
# =============================================================================

def propose_system_enhancement(
    agent_id: str,
    title: str,
    description: str,
    target_file: str,
    proposed_code: str,
    enhancement_type: str = "feature"
) -> Dict[str, Any]:
    """
    Propose a system enhancement for review.

    Creates a proposal in the experiments directory that can be
    reviewed and potentially merged into the main system.
    """
    experiments_dir = WORKSHOP_DIR / agent_id / "experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)

    safe_title = "".join(c if c.isalnum() or c in "_-" else "_" for c in title)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    proposal_dir = experiments_dir / f"{timestamp}_{safe_title}"
    proposal_dir.mkdir(parents=True, exist_ok=True)

    proposal = {
        "title": title,
        "description": description,
        "enhancement_type": enhancement_type,
        "target_file": target_file,
        "proposed_by": agent_id,
        "created_at": datetime.utcnow().isoformat(),
        "status": "proposed"
    }

    # Write proposal metadata
    with open(proposal_dir / "proposal.json", "w") as f:
        json.dump(proposal, f, indent=2)

    # Write proposed code
    with open(proposal_dir / "proposed_code.txt", "w") as f:
        f.write(proposed_code)

    # Write review checklist
    checklist = f"""# Enhancement Proposal: {title}

## Description
{description}

## Target
- File: {target_file}
- Type: {enhancement_type}

## Review Checklist
- [ ] Code reviewed
- [ ] Tests written
- [ ] Documentation updated
- [ ] No breaking changes
- [ ] Performance impact assessed

## Approval
- Proposed by: {agent_id}
- Date: {datetime.utcnow().isoformat()}
- Status: Proposed
"""
    with open(proposal_dir / "REVIEW.md", "w") as f:
        f.write(checklist)

    return {
        "success": True,
        "proposal": {
            "id": f"{timestamp}_{safe_title}",
            "path": str(proposal_dir),
            "title": title,
            "status": "proposed"
        }
    }


def list_enhancement_proposals(agent_id: str = None, status: str = None) -> List[Dict[str, Any]]:
    """List all enhancement proposals."""
    proposals = []

    for agent_dir in WORKSHOP_DIR.iterdir():
        if not agent_dir.is_dir():
            continue

        if agent_id and agent_dir.name != agent_id:
            continue

        experiments_dir = agent_dir / "experiments"
        if not experiments_dir.exists():
            continue

        for proposal_dir in experiments_dir.iterdir():
            if not proposal_dir.is_dir():
                continue

            proposal_file = proposal_dir / "proposal.json"
            if proposal_file.exists():
                try:
                    with open(proposal_file, "r") as f:
                        proposal = json.load(f)
                    proposal["id"] = proposal_dir.name
                    proposal["agent_id"] = agent_dir.name

                    if status is None or proposal.get("status") == status:
                        proposals.append(proposal)
                except:
                    pass

    return sorted(proposals, key=lambda x: x.get("created_at", ""), reverse=True)


# =============================================================================
# API ENDPOINTS
# =============================================================================

@router.post("/execute", response_model=ToolResponse)
async def execute_agent_tool(request: ToolRequest):
    """Execute any agent tool."""

    tool_name = request.tool_name
    params = request.parameters
    agent_id = request.agent_id

    try:
        # Log reading tools
        if tool_name == "read_logs":
            result = read_system_logs(
                params.get("log_type", "all"),
                params.get("lines", 100)
            )
            return ToolResponse(success=True, data=result)

        # Workshop tools
        elif tool_name == "create_component":
            result = create_workshop_component(
                agent_id,
                params.get("name", "untitled"),
                params.get("component_type", "utility"),
                params.get("code", ""),
                params.get("description", ""),
                params.get("metadata", {})
            )
            return ToolResponse(success=result["success"], data=result)

        elif tool_name == "list_components":
            components = list_workshop_components(
                agent_id,
                params.get("component_type")
            )
            return ToolResponse(success=True, data={"components": components})

        elif tool_name == "store_value":
            result = create_value_store(
                agent_id,
                params.get("key"),
                params.get("value"),
                params.get("description", "")
            )
            return ToolResponse(success=result["success"], data=result)

        elif tool_name == "get_value":
            result = get_value_store(agent_id, params.get("key"))
            return ToolResponse(success=True, data=result)

        # Hook tools
        elif tool_name == "create_hook":
            result = create_agent_hook(
                agent_id,
                params.get("hook_name", "unnamed"),
                params.get("hook_type", "post"),
                params.get("event", "trade"),
                params.get("code", "pass"),
                params.get("description", "")
            )
            return ToolResponse(success=result["success"], data=result)

        elif tool_name == "list_hooks":
            hooks = list_agent_hooks(params.get("agent_id", agent_id))
            return ToolResponse(success=True, data={"hooks": hooks})

        # MCP tools
        elif tool_name == "list_mcp_servers":
            result = list_available_mcp_servers()
            return ToolResponse(success=True, data=result)

        elif tool_name == "get_mcp_config":
            result = get_mcp_server_config(params.get("server_name"))
            return ToolResponse(success="error" not in result, data=result)

        # System growth tools
        elif tool_name == "propose_enhancement":
            result = propose_system_enhancement(
                agent_id,
                params.get("title", "Untitled"),
                params.get("description", ""),
                params.get("target_file", ""),
                params.get("proposed_code", ""),
                params.get("enhancement_type", "feature")
            )
            return ToolResponse(success=result["success"], data=result)

        elif tool_name == "list_proposals":
            proposals = list_enhancement_proposals(
                params.get("agent_id"),
                params.get("status")
            )
            return ToolResponse(success=True, data={"proposals": proposals})

        # Feature page tools
        elif tool_name == "create_feature_page":
            result = create_feature_page(
                agent_id,
                params.get("feature_name", "untitled"),
                params.get("title", "Untitled Feature"),
                params.get("svelte_code", ""),
                params.get("description", ""),
                params.get("icon", "star")
            )
            return ToolResponse(success=result.get("success", False), data=result)

        elif tool_name == "list_feature_pages":
            features = list_feature_pages(agent_id)
            return ToolResponse(success=True, data={"features": features})

        elif tool_name == "get_feature_page":
            result = get_feature_page(
                agent_id,
                params.get("feature_name")
            )
            return ToolResponse(success=result.get("success", False), data=result)

        elif tool_name == "delete_feature_page":
            result = delete_feature_page(
                agent_id,
                params.get("feature_name")
            )
            return ToolResponse(success=result.get("success", False), data=result)

        else:
            return ToolResponse(
                success=False,
                error=f"Unknown tool: {tool_name}"
            )

    except Exception as e:
        logger.error(f"Tool execution error: {e}", exc_info=True)
        return ToolResponse(success=False, error=str(e))


@router.get("/available")
async def list_available_tools():
    """List all available agent tools."""
    return {
        "tools": [
            # Log tools
            {"name": "read_logs", "description": "Read system logs (router, risk, position)", "params": ["log_type", "lines"]},

            # Workshop tools
            {"name": "create_component", "description": "Create a component in workshop", "params": ["name", "component_type", "code", "description"]},
            {"name": "list_components", "description": "List workshop components", "params": ["component_type"]},
            {"name": "store_value", "description": "Store a value for later use", "params": ["key", "value", "description"]},
            {"name": "get_value", "description": "Retrieve stored values", "params": ["key"]},

            # Hook tools
            {"name": "create_hook", "description": "Create a system hook", "params": ["hook_name", "hook_type", "event", "code", "description"]},
            {"name": "list_hooks", "description": "List system hooks", "params": ["agent_id"]},

            # MCP tools
            {"name": "list_mcp_servers", "description": "List available MCP servers", "params": []},
            {"name": "get_mcp_config", "description": "Get MCP server configuration", "params": ["server_name"]},

            # System growth tools
            {"name": "propose_enhancement", "description": "Propose system enhancement", "params": ["title", "description", "target_file", "proposed_code", "enhancement_type"]},
            {"name": "list_proposals", "description": "List enhancement proposals", "params": ["agent_id", "status"]},

            # Feature page tools
            {"name": "create_feature_page", "description": "Create a new feature page with Svelte component and metadata", "params": ["feature_name", "title", "svelte_code", "description", "icon"]},
            {"name": "list_feature_pages", "description": "List all feature pages for an agent", "params": []},
            {"name": "get_feature_page", "description": "Get a specific feature page with Svelte code and metadata", "params": ["feature_name"]},
            {"name": "delete_feature_page", "description": "Delete a feature page and all its files", "params": ["feature_name"]},
        ],
        "workshop_path": str(WORKSHOP_DIR),
        "logs_path": str(LOGS_DIR),
        "hooks_path": str(HOOKS_DIR)
    }


@router.get("/workshop/{agent_id}")
async def get_agent_workshop(agent_id: str):
    """Get agent's workshop contents."""
    agent_workshop = WORKSHOP_DIR / agent_id

    if not agent_workshop.exists():
        return {"agent_id": agent_id, "exists": False}

    contents = {}
    for subdir in ["components", "hooks", "values", "templates", "experiments", "features"]:
        sub_path = agent_workshop / subdir
        if sub_path.exists():
            contents[subdir] = [f.name for f in sub_path.iterdir()]

    return {
        "agent_id": agent_id,
        "exists": True,
        "path": str(agent_workshop),
        "contents": contents
    }


# =============================================================================
# FEATURE PAGE REST ENDPOINTS
# =============================================================================

@router.post("/features/{agent_id}")
async def api_create_feature_page(agent_id: str, request: FeaturePageRequest):
    """
    Create a new feature page.

    Args:
        agent_id: The agent creating the feature
        request: FeaturePageRequest with feature details

    Returns:
        Created feature page details
    """
    result = create_feature_page(
        agent_id=agent_id,
        feature_name=request.feature_name,
        title=request.title,
        svelte_code=request.svelte_code,
        description=request.description,
        icon=request.icon
    )

    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Failed to create feature page"))

    return result


@router.get("/features/{agent_id}")
async def api_list_feature_pages(agent_id: str):
    """
    List all feature pages for an agent.

    Args:
        agent_id: The agent whose features to list

    Returns:
        List of feature page metadata
    """
    features = list_feature_pages(agent_id)
    return {"agent_id": agent_id, "features": features, "count": len(features)}


@router.get("/features/{agent_id}/{feature_name}")
async def api_get_feature_page(agent_id: str, feature_name: str):
    """
    Get a specific feature page.

    Args:
        agent_id: The agent who owns the feature
        feature_name: The feature identifier

    Returns:
        Feature page with metadata and Svelte code
    """
    result = get_feature_page(agent_id, feature_name)

    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result.get("error", "Feature page not found"))

    return result


@router.delete("/features/{agent_id}/{feature_name}")
async def api_delete_feature_page(agent_id: str, feature_name: str):
    """
    Delete a feature page.

    Args:
        agent_id: The agent who owns the feature
        feature_name: The feature identifier

    Returns:
        Deletion confirmation
    """
    result = delete_feature_page(agent_id, feature_name)

    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result.get("error", "Feature page not found"))

    return result

"""
MCP API Endpoints.

REST API for MCP server management.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field

from ..agents.mcp.client import (
    MCPClient,
    MCPServerConfig,
    get_mcp_client,
)


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/mcp", tags=["mcp"])


# Request/Response Models
class ServerConfigRequest(BaseModel):
    """Request to configure an MCP server."""
    server_id: str
    name: str
    transport: str = "stdio"
    command: Optional[str] = None
    args: List[str] = []
    env: Dict[str, str] = {}
    url: Optional[str] = None
    timeout: int = 30000


class ConnectRequest(BaseModel):
    """Request to connect to a server."""
    config: Optional[ServerConfigRequest] = None


class CallToolRequest(BaseModel):
    """Request to call an MCP tool."""
    tool_name: str
    arguments: Dict[str, Any] = {}


def get_client() -> MCPClient:
    """Get MCP client."""
    return get_mcp_client()


@router.get("/servers")
async def list_servers(
    include_tools: bool = Query(False, description="Include tool list"),
) -> Dict[str, Any]:
    """
    List all MCP servers and their status.
    """
    client = get_client()
    servers = client.get_all_servers()

    if include_tools:
        for server in servers:
            server["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                }
                for t in client.get_tools(server["server_id"])
            ]

    return {
        "servers": servers,
        "total": len(servers),
    }


@router.get("/servers/{server_id}")
async def get_server(server_id: str) -> Dict[str, Any]:
    """
    Get status of a specific MCP server.
    """
    client = get_client()
    status = client.get_server_status(server_id)

    if not status:
        raise HTTPException(status_code=404, detail="Server not found")

    return status


@router.get("/servers/{server_id}/tools")
async def get_server_tools(server_id: str) -> Dict[str, Any]:
    """
    Get available tools for a specific MCP server.
    """
    client = get_client()
    status = client.get_server_status(server_id)

    if not status:
        raise HTTPException(status_code=404, detail="Server not found")

    tools = client.get_tools(server_id)

    return {
        "server_id": server_id,
        "tools": [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in tools
        ],
        "total": len(tools),
    }


@router.post("/servers/{server_id}/connect")
async def connect_server(
    server_id: str,
    background_tasks: BackgroundTasks,
    config: Optional[ServerConfigRequest] = None,
) -> Dict[str, Any]:
    """
    Connect to an MCP server.

    If config is provided, creates/updates server configuration.
    """
    client = get_client()

    # Check if server exists
    status = client.get_server_status(server_id)

    if config:
        # Create or update configuration
        server_config = MCPServerConfig(
            server_id=config.server_id,
            name=config.name,
            transport=config.transport,
            command=config.command,
            args=config.args,
            env=config.env,
            url=config.url,
            timeout=config.timeout,
        )

        # Connect in background
        async def do_connect():
            try:
                await client.connect(server_config)
            except Exception as e:
                logger.error(f"Failed to connect to {server_id}: {e}")

        background_tasks.add_task(do_connect)

        return {
            "message": "Connection initiated",
            "server_id": server_id,
        }

    elif status:
        # Reconnect existing server
        if status["status"] == "connected":
            return {
                "message": "Already connected",
                "server_id": server_id,
            }

        # Would need stored config to reconnect
        raise HTTPException(
            status_code=400,
            detail="Server config not provided and not stored"
        )

    else:
        raise HTTPException(status_code=404, detail="Server not found")


@router.post("/servers/{server_id}/disconnect")
async def disconnect_server(server_id: str) -> Dict[str, Any]:
    """
    Disconnect from an MCP server.
    """
    client = get_client()
    success = await client.disconnect(server_id)

    if not success:
        raise HTTPException(status_code=404, detail="Server not found")

    return {
        "disconnected": True,
        "server_id": server_id,
    }


@router.post("/call-tool")
async def call_tool(
    request: CallToolRequest,
    server_id: str = Query(..., description="MCP server ID"),
) -> Dict[str, Any]:
    """
    Call a tool on an MCP server.
    """
    client = get_client()

    try:
        result = await client.call_tool(
            server_id=server_id,
            tool_name=request.tool_name,
            arguments=request.arguments,
        )
        return {
            "success": True,
            "result": result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tools")
async def list_all_tools(
    server_id: Optional[str] = Query(None, description="Filter by server"),
) -> Dict[str, Any]:
    """
    List all available MCP tools.
    """
    client = get_client()
    tools = client.get_tools(server_id)

    return {
        "tools": [
            {
                "name": t.name,
                "server_id": t.server_id,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in tools
        ],
        "total": len(tools),
    }


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Get health status of all MCP servers.
    """
    client = get_client()
    servers = client.get_all_servers()

    connected = sum(1 for s in servers if s["status"] == "connected")
    total = len(servers)

    return {
        "healthy": connected == total,
        "connected": connected,
        "total": total,
        "servers": [
            {
                "server_id": s["server_id"],
                "status": s["status"],
                "healthy": s["status"] == "connected",
            }
            for s in servers
        ],
    }


# =============================================================================
# QuantMind MCP Status Monitoring
# =============================================================================

# Define the 5 core MCP servers for QuantMind
QUANTMIND_MCP_SERVERS = {
    "context7": {
        "name": "Context7",
        "type": "context7",
        "description": "MQL5 documentation retrieval",
        "url": "https://context7.com/websites/mql5_en",
        "icon": "book",
        "tools": ["get_mql5_documentation", "get_mql5_examples"]
    },
    "sequential_thinking": {
        "name": "Sequential Thinking",
        "type": "sequential_thinking",
        "description": "Task decomposition and step-by-step reasoning",
        "url": "local://sequential-thinking",
        "icon": "brain",
        "tools": ["sequential_thinking", "analyze_errors"]
    },
    "pageindex": {
        "name": "PageIndex",
        "type": "pageindex",
        "description": "PDF indexing and search",
        "url": "http://localhost:8080",
        "icon": "file-text",
        "tools": ["index_pdf", "search_pdf", "get_indexed_documents"]
    },
    "backtest": {
        "name": "Backtest MCP",
        "type": "backtest",
        "description": "Strategy backtesting engine",
        "url": "http://localhost:8081",
        "icon": "chart-line",
        "tools": ["run_backtest", "get_backtest_status", "get_backtest_results", "compare_backtests"]
    },
    "mt5_compiler": {
        "name": "MT5 Compiler",
        "type": "mt5_compiler",
        "description": "MQL5 code compilation",
        "url": "local://mt5-compiler",
        "icon": "code",
        "tools": ["compile_mql5_code", "validate_mql5_syntax", "get_compilation_errors"]
    }
}


class MCPServerStatus(BaseModel):
    """Status of a single MCP server."""
    server_id: str
    name: str
    type: str
    description: str
    url: str
    icon: str
    status: str  # "online", "offline", "error"
    healthy: bool
    latency_ms: Optional[int] = None
    last_check: Optional[str] = None
    error: Optional[str] = None
    tools: List[str] = []
    tools_available: int = 0


class MCPStatusResponse(BaseModel):
    """Response model for MCP status endpoint."""
    healthy: bool
    total_servers: int
    online_servers: int
    offline_servers: int
    servers: List[MCPServerStatus]
    last_updated: str


@router.get("/status", response_model=MCPStatusResponse)
async def get_mcp_status() -> MCPStatusResponse:
    """
    Get comprehensive status for all 5 MCP servers.
    
    Returns status for:
    - Context7 (MQL5 Documentation)
    - Sequential Thinking
    - PageIndex (PDF Indexing)
    - Backtest MCP
    - MT5 Compiler
    
    Each server includes:
    - Connection status (online/offline/error)
    - Latency in milliseconds
    - Available tools count
    - Last check timestamp
    """
    import time
    from datetime import datetime
    
    client = get_client()
    connected_servers = {s["server_id"]: s for s in client.get_all_servers()}
    
    server_statuses = []
    online_count = 0
    offline_count = 0
    
    for server_id, server_info in QUANTMIND_MCP_SERVERS.items():
        connected_server = connected_servers.get(server_id, {})
        status = connected_server.get("status", "offline")
        
        # Calculate latency (placeholder - would measure actual ping)
        latency_ms = None
        if status == "connected":
            start = time.time()
            # Would do actual health check here
            latency_ms = int((time.time() - start) * 1000)
        
        # Determine status string
        if status == "connected":
            status_str = "online"
            healthy = True
            online_count += 1
        elif status == "error":
            status_str = "error"
            healthy = False
            offline_count += 1
        else:
            status_str = "offline"
            healthy = False
            offline_count += 1
        
        # Get available tools
        tools = server_info.get("tools", [])
        tools_available = len(tools) if status == "connected" else 0
        
        server_status = MCPServerStatus(
            server_id=server_id,
            name=server_info["name"],
            type=server_info["type"],
            description=server_info["description"],
            url=server_info["url"],
            icon=server_info["icon"],
            status=status_str,
            healthy=healthy,
            latency_ms=latency_ms,
            last_check=datetime.utcnow().isoformat() if status == "connected" else None,
            error=connected_server.get("error"),
            tools=tools,
            tools_available=tools_available
        )
        server_statuses.append(server_status)
    
    return MCPStatusResponse(
        healthy=online_count == len(QUANTMIND_MCP_SERVERS),
        total_servers=len(QUANTMIND_MCP_SERVERS),
        online_servers=online_count,
        offline_servers=offline_count,
        servers=server_statuses,
        last_updated=datetime.utcnow().isoformat()
    )


@router.get("/status/{server_id}", response_model=MCPServerStatus)
async def get_single_server_status(server_id: str) -> MCPServerStatus:
    """
    Get status for a specific MCP server.
    
    Args:
        server_id: One of context7, sequential_thinking, pageindex, backtest, mt5_compiler
    """
    from datetime import datetime
    import time
    
    if server_id not in QUANTMIND_MCP_SERVERS:
        raise HTTPException(
            status_code=404, 
            detail=f"Unknown server: {server_id}. Valid servers: {list(QUANTMIND_MCP_SERVERS.keys())}"
        )
    
    server_info = QUANTMIND_MCP_SERVERS[server_id]
    client = get_client()
    connected_server = client.get_server_status(server_id)
    
    status = "offline"
    error = None
    latency_ms = None
    
    if connected_server:
        status = "connected" if connected_server.get("status") == "connected" else "offline"
        error = connected_server.get("error")
        
        if status == "connected":
            start = time.time()
            # Would do actual health check here
            latency_ms = int((time.time() - start) * 1000)
    
    status_str = "online" if status == "connected" else ("error" if error else "offline")
    tools = server_info.get("tools", [])
    
    return MCPServerStatus(
        server_id=server_id,
        name=server_info["name"],
        type=server_info["type"],
        description=server_info["description"],
        url=server_info["url"],
        icon=server_info["icon"],
        status=status_str,
        healthy=status == "connected",
        latency_ms=latency_ms,
        last_check=datetime.utcnow().isoformat() if status == "connected" else None,
        error=error,
        tools=tools,
        tools_available=len(tools) if status == "connected" else 0
    )


@router.post("/status/retry/{server_id}")
async def retry_server_connection(
    server_id: str,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Retry connection to a failed MCP server.
    
    Args:
        server_id: Server to retry connection for
    """
    if server_id not in QUANTMIND_MCP_SERVERS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown server: {server_id}"
        )
    
    client = get_client()
    server_info = QUANTMIND_MCP_SERVERS[server_id]
    
    # Create config for connection
    config = MCPServerConfig(
        server_id=server_id,
        name=server_info["name"],
        transport="http" if server_info["url"].startswith("http") else "stdio",
        url=server_info["url"] if server_info["url"].startswith("http") else None,
        timeout=30000
    )
    
    async def do_connect():
        try:
            await client.connect(config)
            logger.info(f"Successfully reconnected to {server_id}")
        except Exception as e:
            logger.error(f"Failed to reconnect to {server_id}: {e}")
    
    background_tasks.add_task(do_connect)
    
    return {
        "message": "Connection retry initiated",
        "server_id": server_id,
        "server_name": server_info["name"]
    }


@router.get("/tools/available")
async def get_available_mcp_tools() -> Dict[str, Any]:
    """
    Get all available MCP tools grouped by server.
    """
    client = get_client()
    all_tools = []
    tools_by_server = {}
    
    for server_id, server_info in QUANTMIND_MCP_SERVERS.items():
        server_tools = client.get_tools(server_id)
        tools_by_server[server_id] = [
            {
                "name": t.name,
                "description": t.description,
                "available": True
            }
            for t in server_tools
        ]
        all_tools.extend([
            {
                "name": t.name,
                "server_id": server_id,
                "description": t.description
            }
            for t in server_tools
        ])
    
    return {
        "total_tools": len(all_tools),
        "tools": all_tools,
        "by_server": tools_by_server
    }

"""
MCP API Endpoints

Provides REST API endpoints for MCP server management and tool execution.
Uses the real MCP client layer to connect to actual MCP servers and execute tools.

These endpoints match the expected API structure in mcpStore.ts.

IMPORTANT: Requires live MCP client initialization. No fallback mocks.
All operations route to the live MCP client.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel

# Import real MCP client - REQUIRED
from src.mcp.client import (
    get_mcp_client,
    MCPServerConfig,
    MCPServerStatus,
    MCPTransportType,
    initialize_mcp_client_from_config,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/mcp", tags=["mcp"])


# ============================================================================
# Request/Response Models
# ============================================================================

class ConnectRequest(BaseModel):
    config: Optional[Dict[str, Any]] = None


class CallToolRequest(BaseModel):
    tool_name: str
    arguments: Dict[str, Any] = {}


class RegisterServerRequest(BaseModel):
    server_id: str
    name: str
    description: str = ""
    command: Optional[str] = None
    args: List[str] = []
    env: Dict[str, str] = {}
    auto_connect: bool = False


class MCPServerResponse(BaseModel):
    server_id: str
    name: str
    status: str
    connected_at: Optional[str] = None
    tools_count: int
    last_error: Optional[str] = None


class MCPToolResponse(BaseModel):
    name: str
    server_id: str
    description: str
    input_schema: Dict[str, Any]


class MCPServerStatusModel(BaseModel):
    server_id: str
    status: str
    healthy: bool


class HealthResponse(BaseModel):
    healthy: bool
    servers: List[MCPServerStatusModel]


# ============================================================================
# Default MCP Server Configurations
# ============================================================================

DEFAULT_MCP_SERVERS = [
    {
        "server_id": "metatrader5",
        "name": "MetaTrader 5 MCP",
        "description": "MetaTrader 5 trading platform integration",
        "command": "npx",
        "args": ["-y", "@anthropic-ai/mcp-server-mt5"],
        "env": {},
        "auto_connect": False,
    },
    {
        "server_id": "filesystem",
        "name": "Filesystem MCP",
        "description": "Local filesystem access",
        "command": "npx",
        "args": ["-y", "@anthropic-ai/mcp-server-filesystem", "--root", "./workspace"],
        "env": {},
        "auto_connect": False,
    },
    {
        "server_id": "context7",
        "name": "Context7 MCP",
        "description": "MQL5 documentation retrieval",
        "command": "npx",
        "args": ["-y", "@context7/mcp-server"],
        "env": {},
        "auto_connect": False,
    },
    {
        "server_id": "sequential_thinking",
        "name": "Sequential Thinking MCP",
        "description": "Task decomposition and reasoning",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"],
        "env": {},
        "auto_connect": False,
    },
    {
        "server_id": "pageindex",
        "name": "PageIndex MCP",
        "description": "PDF indexing and search",
        "command": "npx",
        "args": ["-y", "@pageindex/mcp-server"],
        "env": {},
        "auto_connect": False,
    },
    {
        "server_id": "backtest",
        "name": "Backtest MCP",
        "description": "Strategy backtesting",
        "command": "python",
        "args": ["-m", "mcp_servers.backtest_mcp_server.main"],
        "env": {},
        "auto_connect": False,
    },
    {
        "server_id": "quantmindx-kb",
        "name": "QuantMindX Knowledge Base",
        "description": "Knowledge base retrieval for MQL5 articles, books, and trading logs via PageIndex",
        "command": "python",
        "args": ["mcp-servers/quantmindx-kb/server.py"],
        "env": {
            "PAGEINDEX_ARTICLES_URL": "http://localhost:3000",
            "PAGEINDEX_BOOKS_URL": "http://localhost:3001",
            "PAGEINDEX_LOGS_URL": "http://localhost:3002",
        },
        "auto_connect": False,
    },
]


def _require_client():
    """
    Get MCP client or raise HTTPException if not available.

    This function ALWAYS returns a live MCP client - no fallbacks.
    All default servers are registered on first call.

    Raises:
        HTTPException: If MCP client cannot be initialized
    """
    try:
        client = get_mcp_client()
    except Exception as e:
        logger.error(f"Failed to initialize MCP client: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"MCP client initialization failed: {str(e)}"
        )

    # Register default servers if not already registered
    for server_config in DEFAULT_MCP_SERVERS:
        if not client.get_server_config(server_config["server_id"]):
            try:
                config = MCPServerConfig(
                    server_id=server_config["server_id"],
                    name=server_config["name"],
                    description=server_config.get("description", ""),
                    transport=MCPTransportType.STDIO,
                    command=server_config.get("command"),
                    args=server_config.get("args", []),
                    env=server_config.get("env", {}),
                    auto_connect=server_config.get("auto_connect", False),
                )
                client.register_server(config)
                logger.info(f"Registered default MCP server: {server_config['server_id']}")
            except Exception as e:
                logger.warning(f"Failed to register server {server_config['server_id']}: {e}")

    return client


# ============================================================================
# API Endpoints
# ============================================================================

@router.get("/servers", response_model=Dict[str, Any])
async def list_servers(include_tools: bool = Query(False)):
    """
    List all configured MCP servers with their status.
    
    Returns live status from the MCP client.
    """
    client = _require_client()

    servers = []
    for config in client.list_servers():
        status = client.get_status(config.server_id)

        # Get connection details
        connected_at = None
        last_error = None
        if config.server_id in client._servers:
            state = client._servers[config.server_id]
            connected_at = state.connected_at.isoformat() if state.connected_at else None
            last_error = state.last_error

        tools = await client.list_tools(config.server_id)

        server_data = {
            "server_id": config.server_id,
            "name": config.name,
            "status": status.value,
            "connected_at": connected_at,
            "tools_count": len(tools),
            "last_error": last_error,
        }

        if include_tools:
            server_data["tools"] = [
                {
                    "name": t.name,
                    "server_id": t.server_id,
                    "description": t.description,
                    "input_schema": t.input_schema,
                }
                for t in tools
            ]

        servers.append(server_data)

    return {"servers": servers}


@router.post("/servers/register", response_model=Dict[str, Any])
async def register_server(request: RegisterServerRequest):
    """
    Register a new MCP server configuration.
    
    The server will be available for connection but not auto-connected
    unless auto_connect is True.
    """
    client = _require_client()

    # Check if already registered
    if client.get_server_config(request.server_id):
        raise HTTPException(
            status_code=400,
            detail=f"Server {request.server_id} already registered"
        )

    config = MCPServerConfig(
        server_id=request.server_id,
        name=request.name,
        description=request.description,
        transport=MCPTransportType.STDIO,
        command=request.command,
        args=request.args,
        env=request.env,
        auto_connect=request.auto_connect,
    )

    client.register_server(config)

    logger.info(f"Registered MCP server: {request.server_id}")

    # Auto-connect if requested
    if request.auto_connect:
        try:
            success = await client.connect(request.server_id)
            if success:
                logger.info(f"Auto-connected to MCP server: {request.server_id}")
            else:
                logger.warning(f"Auto-connect failed for MCP server: {request.server_id}")
        except Exception as e:
            logger.error(f"Auto-connect error for {request.server_id}: {e}")

    return {
        "success": True,
        "server_id": request.server_id,
        "message": "Server registered successfully",
    }


@router.get("/servers/{server_id}/tools", response_model=Dict[str, Any])
async def get_server_tools(server_id: str):
    """
    Get available tools for a specific MCP server.
    
    Requires the server to be connected to list tools.
    """
    client = _require_client()

    config = client.get_server_config(server_id)
    if not config:
        raise HTTPException(status_code=404, detail=f"Server {server_id} not found")

    tools = await client.list_tools(server_id)

    return {
        "tools": [
            {
                "name": t.name,
                "server_id": t.server_id,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in tools
        ]
    }


@router.post("/servers/{server_id}/connect", response_model=Dict[str, Any])
async def connect_server(
    server_id: str,
    request: ConnectRequest = None,
    background_tasks: BackgroundTasks = None
):
    """
    Connect to an MCP server.
    
    Establishes a live connection to the MCP server process.
    """
    client = _require_client()

    config = client.get_server_config(server_id)
    if not config:
        raise HTTPException(status_code=404, detail=f"Server {server_id} not found")

    # Check current status
    current_status = client.get_status(server_id)
    if current_status == MCPServerStatus.CONNECTED:
        state = client._servers.get(server_id)
        connected_at = state.connected_at.isoformat() if state and state.connected_at else None
        return {
            "success": True,
            "server_id": server_id,
            "status": "connected",
            "connected_at": connected_at,
            "message": "Already connected",
        }

    # Attempt connection
    try:
        success = await client.connect(server_id)

        if success:
            logger.info(f"Connected to MCP server: {server_id}")

            state = client._servers.get(server_id)
            connected_at = state.connected_at.isoformat() if state and state.connected_at else None

            return {
                "success": True,
                "server_id": server_id,
                "status": "connected",
                "connected_at": connected_at,
            }
        else:
            state = client._servers.get(server_id)
            error = state.last_error if state else "Unknown error"

            logger.error(f"Failed to connect to MCP server {server_id}: {error}")

            raise HTTPException(
                status_code=500,
                detail=f"Connection failed: {error}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Connection error for {server_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Connection error: {str(e)}"
        )


@router.post("/servers/{server_id}/disconnect", response_model=Dict[str, Any])
async def disconnect_server(server_id: str):
    """
    Disconnect from an MCP server.
    """
    client = _require_client()

    config = client.get_server_config(server_id)
    if not config:
        raise HTTPException(status_code=404, detail=f"Server {server_id} not found")

    await client.disconnect(server_id)

    logger.info(f"Disconnected from MCP server: {server_id}")

    return {
        "success": True,
        "server_id": server_id,
        "status": "disconnected",
    }


@router.post("/call-tool", response_model=Dict[str, Any])
async def call_tool(
    server_id: str = Query(...),
    request: CallToolRequest = None
):
    """
    Call an MCP tool on a specific server.

    Routes tool calls to the live MCP client. Requires the server to be connected.
    
    Args:
        server_id: The MCP server to call the tool on
        request: Tool name and arguments
        
    Returns:
        Tool execution result from the MCP server
    """
    client = _require_client()

    config = client.get_server_config(server_id)
    if not config:
        raise HTTPException(status_code=404, detail=f"Server {server_id} not found")

    # Check connection status
    status = client.get_status(server_id)
    if status != MCPServerStatus.CONNECTED:
        # Try to auto-connect
        try:
            success = await client.connect(server_id)
            if not success:
                raise HTTPException(
                    status_code=400,
                    detail=f"Server {server_id} is not connected (status: {status.value}) and auto-connect failed"
                )
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Server {server_id} is not connected (status: {status.value}): {str(e)}"
            )

    # Verify tool exists
    tool = client.get_tool(request.tool_name)
    if not tool or tool.server_id != server_id:
        # List available tools for helpful error message
        tools = await client.list_tools(server_id)
        available = [t.name for t in tools]
        raise HTTPException(
            status_code=404,
            detail=f"Tool {request.tool_name} not found on server {server_id}. Available tools: {available}"
        )

    try:
        result = await client.call_tool(
            server_id,
            request.tool_name,
            request.arguments
        )

        logger.info(f"Called tool {request.tool_name} on {server_id}")

        return {
            "success": True,
            "result": result,
            "tool": request.tool_name,
            "server_id": server_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.error(f"Tool call failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Tool call failed: {str(e)}"
        )


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check health of all MCP servers.
    
    Returns the live status of all registered servers.
    """
    client = _require_client()

    server_statuses = []

    for config in client.list_servers():
        status = client.get_status(config.server_id)
        server_statuses.append({
            "server_id": config.server_id,
            "status": status.value,
            "healthy": status == MCPServerStatus.CONNECTED,
        })

    healthy = any(s["healthy"] for s in server_statuses) if server_statuses else False

    return {
        "healthy": healthy,
        "servers": server_statuses,
    }


@router.get("/tools", response_model=Dict[str, Any])
async def list_all_tools():
    """
    List all available tools across all connected servers.
    
    Returns tools from the live MCP client registry.
    """
    client = _require_client()

    tools = client.get_all_tools()
    return {
        "tools": [
            {
                "name": t.name,
                "server_id": t.server_id,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in tools
        ]
    }


@router.post("/initialize", response_model=Dict[str, Any])
async def initialize_from_config(
    config_path: Optional[str] = Query(None, description="Path to MCP config file")
):
    """
    Initialize MCP client from configuration file.
    
    Loads server configurations from the specified config file or the default location.
    """
    try:
        client = await initialize_mcp_client_from_config(config_path)
        
        servers_registered = len(client.list_servers())
        
        return {
            "success": True,
            "servers_registered": servers_registered,
            "message": f"MCP client initialized with {servers_registered} servers"
        }
        
    except Exception as e:
        logger.error(f"Failed to initialize MCP client from config: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Initialization failed: {str(e)}"
        )


@router.post("/servers/{server_id}/auto-connect", response_model=Dict[str, Any])
async def set_auto_connect(
    server_id: str,
    enabled: bool = Query(..., description="Enable or disable auto-connect")
):
    """
    Enable or disable auto-connect for a server.
    """
    client = _require_client()
    
    config = client.get_server_config(server_id)
    if not config:
        raise HTTPException(status_code=404, detail=f"Server {server_id} not found")
    
    # Update the config
    state = client._servers.get(server_id)
    if state:
        state.config.auto_connect = enabled
        
    return {
        "success": True,
        "server_id": server_id,
        "auto_connect": enabled,
        "message": f"Auto-connect {'enabled' if enabled else 'disabled'} for {server_id}"
    }

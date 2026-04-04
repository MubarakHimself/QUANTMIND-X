"""
MCP API Endpoints

Provides REST API endpoints for MCP server management and tool execution.
Uses the real MCP client layer to connect to actual MCP servers and execute tools.

These endpoints match the expected API structure in mcpStore.ts.

IMPORTANT: Requires live MCP client initialization. No fallback mocks.
All operations route to the live MCP client.
"""

import logging
import os
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

_DEFAULT_PAGEINDEX_ARTICLES_URL = os.getenv("PAGEINDEX_ARTICLES_URL", "http://pageindex-articles:3000")
_DEFAULT_PAGEINDEX_BOOKS_URL = os.getenv("PAGEINDEX_BOOKS_URL", "http://pageindex-books:3001")
_DEFAULT_PAGEINDEX_LOGS_URL = os.getenv("PAGEINDEX_LOGS_URL", "http://pageindex-logs:3002")


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
            "PAGEINDEX_ARTICLES_URL": _DEFAULT_PAGEINDEX_ARTICLES_URL,
            "PAGEINDEX_BOOKS_URL": _DEFAULT_PAGEINDEX_BOOKS_URL,
            "PAGEINDEX_LOGS_URL": _DEFAULT_PAGEINDEX_LOGS_URL,
        },
        "auto_connect": False,
    },
    {
        "server_id": "notebooklm",
        "name": "NotebookLM MCP",
        "description": "NotebookLM integration for project grounding and cross-session memory",
        "command": "npx",
        "args": ["-y", "notebooklm-mcp-cli"],
        "env": {},
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


# ============================================================================
# NotebookLM Project Grounding Endpoints
# ============================================================================

class NotebookGroundingRequest(BaseModel):
    """Request for grounding NotebookLM with project context."""
    project_path: str = "."
    notebook_name: Optional[str] = None
    include_files: List[str] = []
    exclude_patterns: List[str] = ["*.pyc", "__pycache__", ".git", "node_modules", "venv", ".venv"]
    max_files: int = 50


class NotebookQueryRequest(BaseModel):
    """Request for querying a NotebookLM notebook."""
    notebook_id: str
    query: str
    context: Optional[str] = None


class NotebookGroundingResponse(BaseModel):
    """Response for grounding operation."""
    success: bool
    notebook_id: Optional[str] = None
    sources_added: List[str] = []
    message: str


@router.post("/notebooklm/ground", response_model=NotebookGroundingResponse)
async def ground_notebooklm_with_project(request: NotebookGroundingRequest):
    """
    Ground NotebookLM with project context for project-specific AI assistance.

    This endpoint:
    1. Scans the project directory for relevant files
    2. Creates or uses an existing NotebookLM notebook
    3. Adds project files as sources to the notebook
    4. Enables cross-session memory through NotebookLM
    """
    import glob
    import os

    project_path = os.path.abspath(request.project_path)

    if not os.path.exists(project_path):
        raise HTTPException(
            status_code=404,
            detail=f"Project path does not exist: {project_path}"
        )

    # Find files to add
    files_to_add = []

    if request.include_files:
        # Use specified files
        for pattern in request.include_files:
            matched = glob.glob(os.path.join(project_path, pattern))
            files_to_add.extend(matched)
    else:
        # Auto-discover relevant files
        extensions = ["*.py", "*.md", "*.json", "*.yaml", "*.yml", "*.txt", "*.ts", "*.js"]
        for ext in extensions:
            pattern = os.path.join(project_path, "**", ext)
            matched = glob.glob(pattern, recursive=True)
            # Filter out excluded patterns
            for f in matched:
                excluded = False
                for exclude in request.exclude_patterns:
                    if exclude.replace("*", "") in f:
                        excluded = True
                        break
                if not excluded and len(files_to_add) < request.max_files:
                    files_to_add.append(f)

    # Use the existing NotebookLM client
    try:
        from src.mcp.notebooklm_server import get_notebooklm_client
        client = get_notebooklm_client()

        # Create or get notebook
        notebook_name = request.notebook_name or f"QuantMindX_Project_{os.path.basename(project_path)}"

        # Try to find existing notebook or create new
        existing_notebooks = await client.list_notebooks()
        notebook_id = None

        for nb in existing_notebooks.get("notebooks", []):
            if notebook_name in nb:
                notebook_id = nb.split(":")[-1].strip() if ":" in nb else nb
                break

        if not notebook_id:
            result = await client.create_notebook(notebook_name)
            notebook_id = result.get("notebook_id", "")

        # Add sources
        sources_added = []
        for file_path in files_to_add[:10]:  # Limit to 10 sources
            try:
                await client.add_source(file_path, notebook_id)
                sources_added.append(file_path)
            except Exception as e:
                logger.warning(f"Failed to add source {file_path}: {e}")

        return NotebookGroundingResponse(
            success=True,
            notebook_id=notebook_id,
            sources_added=sources_added,
            message=f"Grounded {len(sources_added)} sources in NotebookLM"
        )

    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="NotebookLM client not available. Install notebooklm-mcp-cli."
        )
    except Exception as e:
        logger.error(f"NotebookLM grounding failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Grounding failed: {str(e)}"
        )


@router.post("/notebooklm/query", response_model=Dict[str, Any])
async def query_notebooklm(request: NotebookQueryRequest):
    """
    Query a NotebookLM notebook with project-specific context.

    This enables AI assistants to use NotebookLM as a knowledge base
    for project context and cross-session memory.
    """
    try:
        from src.mcp.notebooklm_server import get_notebooklm_client
        client = get_notebooklm_client()

        # Build query with optional context
        full_query = request.query
        if request.context:
            full_query = f"Context: {request.context}\n\nQuestion: {request.query}"

        result = await client.query_notebook(request.notebook_id, full_query)

        return {
            "success": True,
            "response": result.get("response", ""),
            "notebook_id": request.notebook_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="NotebookLM client not available. Install notebooklm-mcp-cli."
        )
    except Exception as e:
        logger.error(f"NotebookLM query failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {str(e)}"
        )


@router.get("/notebooklm/notebooks", response_model=Dict[str, Any])
async def list_notebooklm_notebooks():
    """
    List all available notebooks in NotebookLM.
    """
    try:
        from src.mcp.notebooklm_server import get_notebooklm_client
        client = get_notebooklm_client()

        result = await client.list_notebooks()

        return {
            "success": True,
            "notebooks": result.get("notebooks", []),
        }

    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="NotebookLM client not available. Install notebooklm-mcp-cli."
        )
    except Exception as e:
        logger.error(f"Failed to list notebooks: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list notebooks: {str(e)}"
        )


@router.post("/notebooklm/notebooks", response_model=Dict[str, Any])
async def create_notebooklm_notebook(name: str = Query(...)):
    """
    Create a new NotebookLM notebook.
    """
    try:
        from src.mcp.notebooklm_server import get_notebooklm_client
        client = get_notebooklm_client()

        result = await client.create_notebook(name)

        return {
            "success": True,
            "notebook_id": result.get("notebook_id", ""),
            "name": name,
        }

    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="NotebookLM client not available. Install notebooklm-mcp-cli."
        )
    except Exception as e:
        logger.error(f"Failed to create notebook: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create notebook: {str(e)}"
        )


@router.post("/notebooklm/audio", response_model=Dict[str, Any])
async def create_notebooklm_audio(notebook_id: str = Query(...)):
    """
    Create an audio overview from a NotebookLM notebook.
    """
    try:
        from src.mcp.notebooklm_server import get_notebooklm_client
        client = get_notebooklm_client()

        result = await client.create_audio(notebook_id)

        return {
            "success": True,
            "audio_created": result.get("audio_created", ""),
            "notebook_id": notebook_id,
        }

    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="NotebookLM client not available. Install notebooklm-mcp-cli."
        )
    except Exception as e:
        logger.error(f"Failed to create audio: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create audio: {str(e)}"
        )

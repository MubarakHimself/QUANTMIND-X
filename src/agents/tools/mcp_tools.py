"""
MCP Tool Wrappers for QuantMind Agents.

This module provides tool wrappers for all MCP servers:
- Context7: MQL5 documentation retrieval
- Sequential Thinking: Task decomposition
- PageIndex: PDF indexing and search
- Backtest: Strategy backtesting
- MT5 Compiler: MQL5 code compilation

Uses the real MCP client layer for actual server connections.
"""

import asyncio
import json
import logging
import aiohttp
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Import real MCP client
try:
    from src.mcp.client import (
        get_mcp_client,
        MCPServerConfig,
        MCPServerStatus,
        MCPTransportType,
        initialize_mcp_client_from_config,
        MCPClient,
    )
    MCP_CLIENT_AVAILABLE = True
except ImportError:
    MCP_CLIENT_AVAILABLE = False
    logger.warning("Real MCP client not available, using fallback")


# Path to MCP config file
MCP_CONFIG_PATH = Path(__file__).parent.parent / ".analyst" / "mcp_config.json"


# =============================================================================
# MCP Server Configuration Loader
# =============================================================================

def _load_mcp_config() -> Dict[str, Any]:
    """Load MCP server configurations from the config file."""
    config_path = MCP_CONFIG_PATH
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                config_data = json.load(f)
            logger.info(f"Loaded MCP config from {config_path}")
            return config_data.get("mcpServers", {})
        except Exception as e:
            logger.error(f"Failed to load MCP config from {config_path}: {e}")
    return {}


# =============================================================================
# MCP Client Manager (Real Implementation)
# =============================================================================

class MCPClientManager:
    """
    Manager for MCP server connections and tool invocations.

    Uses the real MCP client layer for actual server connections.
    Supports both stdio and HTTP transports.
    """

    def __init__(self, configs: Optional[Dict[str, Any]] = None):
        """
        Initialize MCP client manager.

        Args:
            configs: Optional custom server configurations
        """
        self.configs = configs or _load_mcp_config()
        self._real_client: Optional[Any] = None
        self._http_sessions: Dict[str, aiohttp.ClientSession] = {}
        self._status: Dict[str, Dict[str, Any]] = {}
        self._initialized = False

        logger.info(f"MCPClientManager initialized with {len(self.configs)} servers")

    async def initialize(self) -> None:
        """Initialize connections to all MCP servers using real client."""
        if self._initialized:
            return

        if MCP_CLIENT_AVAILABLE:
            self._real_client = get_mcp_client()

            # Register servers from config
            for server_id, config in self.configs.items():
                if config.get("enabled", True):
                    await self._register_server(server_id, config)
        else:
            logger.warning("MCP client unavailable, tools will raise errors on invocation")

        self._initialized = True

    async def _register_server(self, server_id: str, config: Dict[str, Any]) -> None:
        """Register a server based on its configuration."""
        server_type = config.get("type", "")
        url = config.get("url", "")
        
        # Check if this is an HTTP-based server
        if url and url.startswith("http"):
            # HTTP transport - create session
            if server_id not in self._http_sessions:
                self._http_sessions[server_id] = aiohttp.ClientSession(
                    base_url=url,
                    timeout=aiohttp.ClientTimeout(total=config.get("config", {}).get("timeout", 30))
                )
            self._status[server_id] = {
                "status": "online",
                "last_check": datetime.now().isoformat(),
                "error": None,
                "transport": "http",
                "url": url
            }
            logger.info(f"Registered HTTP MCP server: {server_id} at {url}")
        elif url and url.startswith("local://"):
            # Local/stdio transport - use real MCP client
            if MCP_CLIENT_AVAILABLE and self._real_client:
                try:
                    # Determine command based on server type
                    command, args = self._get_stdio_command(server_id, config)
                    if command:
                        mcp_config = MCPServerConfig(
                            server_id=server_id,
                            name=config.get("name", server_id),
                            description=config.get("description", ""),
                            transport=MCPTransportType.STDIO,
                            command=command,
                            args=args,
                            env=config.get("env", {}),
                            enabled=True,
                            auto_connect=False,
                        )
                        self._real_client.register_server(mcp_config)
                        logger.info(f"Registered stdio MCP server: {server_id}")
                except Exception as e:
                    logger.error(f"Failed to register MCP server {server_id}: {e}")
        else:
            # Default: try stdio with real client
            if MCP_CLIENT_AVAILABLE and self._real_client:
                try:
                    command, args = self._get_stdio_command(server_id, config)
                    if command:
                        mcp_config = MCPServerConfig(
                            server_id=server_id,
                            name=config.get("name", server_id),
                            description=config.get("description", ""),
                            transport=MCPTransportType.STDIO,
                            command=command,
                            args=args,
                            env=config.get("env", {}),
                            enabled=True,
                            auto_connect=False,
                        )
                        self._real_client.register_server(mcp_config)
                        logger.info(f"Registered stdio MCP server: {server_id}")
                except Exception as e:
                    logger.error(f"Failed to register MCP server {server_id}: {e}")

    def _get_stdio_command(self, server_id: str, config: Dict[str, Any]) -> tuple:
        """Get stdio command and args for a server type."""
        server_type = config.get("type", server_id)
        
        # Map server types to their npx packages
        type_to_package = {
            "context7": ("npx", ["-y", "@context7/mcp-server"]),
            "sequential_thinking": ("npx", ["-y", "@modelcontextprotocol/server-sequential-thinking"]),
            "pageindex": ("npx", ["-y", "@pageindex/mcp-server"]),
            "backtest": ("python", ["-m", "mcp_servers.backtest_mcp_server.main"]),
            "mt5_compiler": ("python", ["-m", "mcp_servers.mt5_compiler"]),
        }
        
        if server_type in type_to_package:
            return type_to_package[server_type]
        
        # Check if config has explicit command/args
        if "command" in config:
            return config["command"], config.get("args", [])
        
        return None, []

    async def connect_server(self, server_id: str) -> bool:
        """Connect to a specific MCP server."""
        config = self.configs.get(server_id)
        if not config:
            raise ValueError(f"Unknown server: {server_id}")

        url = config.get("url", "")
        
        # HTTP servers are "always connected" via session
        if url and url.startswith("http"):
            if server_id in self._http_sessions:
                self._status[server_id] = {
                    "status": "online",
                    "last_check": datetime.now().isoformat(),
                    "error": None
                }
                return True
            # Create session if not exists
            self._http_sessions[server_id] = aiohttp.ClientSession(
                base_url=url,
                timeout=aiohttp.ClientTimeout(total=config.get("config", {}).get("timeout", 30))
            )
            return True

        # Stdio servers need real connection
        if MCP_CLIENT_AVAILABLE and self._real_client:
            server_config = self._real_client.get_server_config(server_id)
            if server_config:
                success = await self._real_client.connect(server_id)
                if success:
                    self._status[server_id] = {
                        "status": "online",
                        "last_check": datetime.now().isoformat(),
                        "error": None
                    }
                else:
                    state = self._real_client._servers.get(server_id)
                    self._status[server_id] = {
                        "status": "offline",
                        "last_check": datetime.now().isoformat(),
                        "error": state.last_error if state else "Unknown error"
                    }
                return success
        return False

    async def call_tool(
        self,
        server_id: str,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Call a tool on an MCP server using real client or HTTP.

        Args:
            server_id: Server identifier
            tool_name: Name of the tool to call
            arguments: Tool arguments

        Returns:
            Tool result from actual MCP server
        """
        # Ensure initialized
        if not self._initialized:
            await self.initialize()

        config = self.configs.get(server_id)
        if not config:
            raise ValueError(f"Unknown server: {server_id}")

        url = config.get("url", "")

        # HTTP transport
        if url and url.startswith("http"):
            return await self._call_http_tool(server_id, tool_name, arguments)

        # Stdio transport via real MCP client
        if MCP_CLIENT_AVAILABLE and self._real_client:
            # Check if server is connected
            status = self._real_client.get_status(server_id)
            if status != MCPServerStatus.CONNECTED:
                # Try to connect
                connected = await self.connect_server(server_id)
                if not connected:
                    raise ConnectionError(f"Failed to connect to MCP server: {server_id}")

            # Call the tool using real client
            try:
                result = await self._real_client.call_tool(server_id, tool_name, arguments)
                logger.debug(f"Called tool {tool_name} on {server_id}: success")
                return result
            except Exception as e:
                logger.error(f"Tool call failed on {server_id}/{tool_name}: {e}")
                raise

        raise RuntimeError(f"MCP client not available - cannot call {server_id}/{tool_name}")

    async def _call_http_tool(
        self,
        server_id: str,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Call a tool via HTTP transport."""
        session = self._http_sessions.get(server_id)
        if not session:
            await self.connect_server(server_id)
            session = self._http_sessions.get(server_id)
        
        if not session:
            raise ConnectionError(f"No HTTP session for server: {server_id}")

        # MCP over HTTP uses JSON-RPC
        request_body = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments or {}
            }
        }

        try:
            async with session.post("/mcp", json=request_body) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"HTTP error {response.status}: {error_text}")
                
                result = await response.json()
                
                # Check for JSON-RPC error
                if "error" in result:
                    raise RuntimeError(f"MCP error: {result['error']}")
                
                return result.get("result", {})
                
        except aiohttp.ClientError as e:
            logger.error(f"HTTP call failed to {server_id}: {e}")
            raise ConnectionError(f"Failed to call HTTP MCP server {server_id}: {e}")

    def get_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all MCP servers."""
        statuses = {}
        
        for server_id, config in self.configs.items():
            url = config.get("url", "")
            
            if url and url.startswith("http"):
                # HTTP server status
                statuses[server_id] = self._status.get(server_id, {
                    "status": "unknown",
                    "last_check": datetime.now().isoformat(),
                    "error": None
                })
            elif MCP_CLIENT_AVAILABLE and self._real_client:
                # Stdio server status
                status = self._real_client.get_status(server_id)
                state = self._real_client._servers.get(server_id)
                statuses[server_id] = {
                    "status": status.value,
                    "last_check": datetime.now().isoformat(),
                    "error": state.last_error if state else None
                }
            else:
                statuses[server_id] = self._status.get(server_id, {
                    "status": "unavailable",
                    "last_check": datetime.now().isoformat(),
                    "error": "MCP client not available"
                })
        
        return statuses

    async def health_check(self) -> Dict[str, bool]:
        """Perform health check on all servers."""
        results = {}
        
        for server_id, config in self.configs.items():
            url = config.get("url", "")
            
            if url and url.startswith("http"):
                # HTTP health check
                session = self._http_sessions.get(server_id)
                if session:
                    try:
                        async with session.get("/health") as response:
                            results[server_id] = response.status == 200
                    except:
                        results[server_id] = False
                else:
                    results[server_id] = False
            elif MCP_CLIENT_AVAILABLE and self._real_client:
                status = self._real_client.get_status(server_id)
                results[server_id] = status == MCPServerStatus.CONNECTED
            else:
                results[server_id] = False
        
        return results

    async def list_tools(self, server_id: str) -> List[Dict[str, Any]]:
        """List tools available from a server."""
        config = self.configs.get(server_id)
        if not config:
            return []
        
        url = config.get("url", "")
        
        # HTTP transport
        if url and url.startswith("http"):
            session = self._http_sessions.get(server_id)
            if session:
                try:
                    request_body = {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "tools/list",
                        "params": {}
                    }
                    async with session.post("/mcp", json=request_body) as response:
                        if response.status == 200:
                            result = await response.json()
                            tools = result.get("result", {}).get("tools", [])
                            return [
                                {
                                    "name": t.get("name", "unknown"),
                                    "description": t.get("description", ""),
                                    "input_schema": t.get("inputSchema", {}),
                                    "server_id": server_id
                                }
                                for t in tools
                            ]
                except Exception as e:
                    logger.error(f"Failed to list tools from {server_id}: {e}")
            return []
        
        # Stdio transport
        if MCP_CLIENT_AVAILABLE and self._real_client:
            tools = await self._real_client.list_tools(server_id)
            return [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                    "server_id": t.server_id
                }
                for t in tools
            ]
        return []

    async def close(self) -> None:
        """Close all connections."""
        # Close HTTP sessions
        for session in self._http_sessions.values():
            await session.close()
        self._http_sessions.clear()
        
        # Disconnect stdio servers
        if MCP_CLIENT_AVAILABLE and self._real_client:
            for server_id in list(self._real_client._servers.keys()):
                await self._real_client.disconnect(server_id)


# Global MCP client manager instance
_mcp_manager: Optional[MCPClientManager] = None


async def get_mcp_manager() -> MCPClientManager:
    """Get or create the global MCP client manager."""
    global _mcp_manager
    if _mcp_manager is None:
        _mcp_manager = MCPClientManager()
        await _mcp_manager.initialize()
    return _mcp_manager


# =============================================================================
# Context7 MCP Tools (MQL5 Documentation)
# =============================================================================

async def get_mql5_documentation(
    query: str,
    context: Optional[str] = None,
    max_results: int = 5
) -> Dict[str, Any]:
    """
    Retrieve MQL5 documentation from Context7.

    This tool queries the MQL5 documentation through Context7 MCP server
    to get relevant documentation, examples, and API references.

    Args:
        query: Search query for MQL5 documentation
        context: Optional context to narrow search (e.g., "indicators", "trading")
        max_results: Maximum number of results to return

    Returns:
        Dictionary containing:
        - results: List of documentation entries
        - total: Total number of matching documents
        - query: Original query
    """
    logger.info(f"Querying MQL5 documentation: {query}")

    manager = await get_mcp_manager()

    try:
        result = await manager.call_tool(
            "context7",
            "query-docs",
            {
                "query": query,
                "context": context,
                "max_results": max_results
            }
        )

        # Parse result from MCP server
        if isinstance(result, dict):
            return {
                "results": result.get("results", []),
                "total": result.get("total", 0),
                "query": query
            }
        return {"results": [], "total": 0, "query": query}

    except Exception as e:
        logger.error(f"Context7 query failed: {e}")
        raise RuntimeError(f"Failed to query MQL5 documentation: {e}")


async def get_mql5_examples(
    topic: str,
    language: str = "mql5"
) -> Dict[str, Any]:
    """
    Get MQL5 code examples for a specific topic.

    Args:
        topic: Topic to get examples for (e.g., "moving average", "order send")
        language: Programming language (default: mql5)

    Returns:
        Dictionary containing code examples with explanations
    """
    logger.info(f"Getting MQL5 examples for: {topic}")

    manager = await get_mcp_manager()

    try:
        result = await manager.call_tool(
            "context7",
            "get-examples",
            {
                "topic": topic,
                "language": language
            }
        )

        if isinstance(result, dict):
            return {
                "examples": result.get("examples", []),
                "topic": topic,
                "language": language
            }
        return {"examples": [], "topic": topic, "language": language}

    except Exception as e:
        logger.error(f"Failed to get MQL5 examples: {e}")
        raise RuntimeError(f"Failed to get MQL5 examples: {e}")


# =============================================================================
# Sequential Thinking MCP Tools
# =============================================================================

async def sequential_thinking(
    task: str,
    context: Optional[str] = None,
    max_steps: int = 10
) -> Dict[str, Any]:
    """
    Decompose a complex task into sequential steps.

    This tool uses the Sequential Thinking MCP server to break down
    complex tasks into manageable steps with reasoning.

    Args:
        task: Task description to decompose
        context: Optional context for the task
        max_steps: Maximum number of steps to generate

    Returns:
        Dictionary containing:
        - steps: List of sequential steps
        - reasoning: Overall reasoning
        - estimated_complexity: Complexity assessment
    """
    logger.info(f"Sequential thinking for task: {task[:100]}...")

    manager = await get_mcp_manager()

    try:
        result = await manager.call_tool(
            "sequential_thinking",
            "decompose",
            {
                "task": task,
                "context": context,
                "max_steps": max_steps
            }
        )

        if isinstance(result, dict):
            return {
                "steps": result.get("steps", []),
                "reasoning": result.get("reasoning", ""),
                "estimated_complexity": result.get("complexity", "medium"),
                "total_steps": len(result.get("steps", []))
            }
        return {
            "steps": [],
            "reasoning": "",
            "estimated_complexity": "unknown",
            "total_steps": 0
        }

    except Exception as e:
        logger.error(f"Sequential thinking failed: {e}")
        raise RuntimeError(f"Failed to decompose task: {e}")


async def analyze_errors(
    errors: List[str],
    code: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze errors and suggest fixes using sequential thinking.

    Args:
        errors: List of error messages
        code: Optional code that caused the errors

    Returns:
        Dictionary containing error analysis and suggested fixes
    """
    logger.info(f"Analyzing {len(errors)} errors")

    manager = await get_mcp_manager()

    try:
        result = await manager.call_tool(
            "sequential_thinking",
            "analyze-errors",
            {
                "errors": errors,
                "code": code
            }
        )

        if isinstance(result, dict):
            return {
                "analysis": result.get("analysis", []),
                "overall_assessment": result.get("assessment", "Errors require attention"),
                "suggested_approach": result.get("approach", "Fix errors in order of priority")
            }
        return {
            "analysis": [],
            "overall_assessment": "Analysis failed",
            "suggested_approach": "Manual review required"
        }

    except Exception as e:
        logger.error(f"Error analysis failed: {e}")
        raise RuntimeError(f"Failed to analyze errors: {e}")


# =============================================================================
# PageIndex MCP Tools (PDF Indexing)
# =============================================================================

async def index_pdf(
    pdf_path: str,
    namespace: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Index a PDF document using PageIndex.

    This tool indexes a PDF file into the PageIndex knowledge base
    for later retrieval and search.

    Args:
        pdf_path: Path to the PDF file
        namespace: Namespace to index into (e.g., "mql5_book", "strategies")
        metadata: Optional metadata for the document

    Returns:
        Dictionary containing:
        - job_id: Indexing job identifier
        - status: Indexing status
        - pages_indexed: Number of pages indexed
    """
    logger.info(f"Indexing PDF: {pdf_path} into namespace: {namespace}")

    # Validate PDF path
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    manager = await get_mcp_manager()

    try:
        result = await manager.call_tool(
            "pageindex",
            "index-document",
            {
                "path": pdf_path,
                "namespace": namespace,
                "metadata": metadata or {}
            }
        )

        if isinstance(result, dict):
            return {
                "job_id": result.get("job_id", f"idx_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
                "status": result.get("status", "completed"),
                "pdf_path": pdf_path,
                "namespace": namespace,
                "pages_indexed": result.get("pages_indexed", 0),
                "metadata": metadata or {}
            }
        return {
            "job_id": f"idx_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "status": "unknown",
            "pdf_path": pdf_path,
            "namespace": namespace,
            "pages_indexed": 0,
            "metadata": metadata or {}
        }

    except Exception as e:
        logger.error(f"PDF indexing failed: {e}")
        raise RuntimeError(f"Failed to index PDF: {e}")


async def search_pdf(
    query: str,
    namespace: str,
    max_results: int = 5,
    include_content: bool = True
) -> Dict[str, Any]:
    """
    Search indexed PDFs using PageIndex.

    Args:
        query: Search query
        namespace: Namespace to search in
        max_results: Maximum number of results
        include_content: Whether to include full content in results

    Returns:
        Dictionary containing search results
    """
    logger.info(f"Searching PDF namespace {namespace}: {query}")

    manager = await get_mcp_manager()

    try:
        result = await manager.call_tool(
            "pageindex",
            "search",
            {
                "query": query,
                "namespace": namespace,
                "max_results": max_results,
                "include_content": include_content
            }
        )

        if isinstance(result, dict):
            return {
                "results": result.get("results", []),
                "total": result.get("total", 0),
                "query": query,
                "namespace": namespace
            }
        return {
            "results": [],
            "total": 0,
            "query": query,
            "namespace": namespace
        }

    except Exception as e:
        logger.error(f"PDF search failed: {e}")
        raise RuntimeError(f"Failed to search PDFs: {e}")


async def get_indexed_documents(namespace: str) -> Dict[str, Any]:
    """
    Get list of indexed documents in a namespace.

    Args:
        namespace: Namespace to list documents from

    Returns:
        Dictionary containing list of indexed documents
    """
    logger.info(f"Getting indexed documents for namespace: {namespace}")

    manager = await get_mcp_manager()

    try:
        result = await manager.call_tool(
            "pageindex",
            "list-documents",
            {"namespace": namespace}
        )

        if isinstance(result, dict):
            return {
                "namespace": namespace,
                "documents": result.get("documents", []),
                "total": result.get("total", 0)
            }
        return {"namespace": namespace, "documents": [], "total": 0}

    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise RuntimeError(f"Failed to list indexed documents: {e}")


# =============================================================================
# Backtest MCP Tools
# =============================================================================

async def run_backtest(
    code: str,
    config: Dict[str, Any],
    strategy_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run a backtest for MQL5 strategy code.

    This tool submits a backtest job to the Backtest MCP server.

    Args:
        code: MQL5 strategy code to backtest
        config: Backtest configuration including:
            - symbol: Trading symbol (e.g., "EURUSD")
            - timeframe: Timeframe (e.g., "H1")
            - start_date: Start date for backtest
            - end_date: End date for backtest
            - initial_deposit: Initial deposit amount
        strategy_name: Optional name for the strategy

    Returns:
        Dictionary containing:
        - backtest_id: Backtest job identifier
        - status: Job status
        - estimated_time: Estimated completion time
    """
    logger.info(f"Running backtest for strategy: {strategy_name or 'unnamed'}")

    manager = await get_mcp_manager()

    try:
        result = await manager.call_tool(
            "backtest",
            "run",
            {
                "code": code,
                "config": config,
                "strategy_name": strategy_name
            }
        )

        if isinstance(result, dict):
            return {
                "backtest_id": result.get("backtest_id", f"bt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
                "status": result.get("status", "queued"),
                "strategy_name": strategy_name,
                "config": config,
                "estimated_time": result.get("estimated_time", "2 minutes")
            }
        return {
            "backtest_id": f"bt_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "status": "unknown",
            "strategy_name": strategy_name,
            "config": config,
            "estimated_time": "unknown"
        }

    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise RuntimeError(f"Failed to run backtest: {e}")


async def get_backtest_status(backtest_id: str) -> Dict[str, Any]:
    """
    Get status of a backtest job.

    Args:
        backtest_id: Backtest job identifier

    Returns:
        Dictionary containing job status and progress
    """
    logger.info(f"Getting backtest status: {backtest_id}")

    manager = await get_mcp_manager()

    try:
        result = await manager.call_tool(
            "backtest",
            "status",
            {"backtest_id": backtest_id}
        )

        if isinstance(result, dict):
            return {
                "backtest_id": backtest_id,
                "status": result.get("status", "unknown"),
                "progress": result.get("progress", 0),
                "started_at": result.get("started_at"),
                "estimated_completion": result.get("estimated_completion")
            }
        return {
            "backtest_id": backtest_id,
            "status": "unknown",
            "progress": 0,
            "started_at": None,
            "estimated_completion": None
        }

    except Exception as e:
        logger.error(f"Failed to get backtest status: {e}")
        raise RuntimeError(f"Failed to get backtest status: {e}")


async def get_backtest_results(backtest_id: str) -> Dict[str, Any]:
    """
    Get results of a completed backtest.

    Args:
        backtest_id: Backtest job identifier

    Returns:
        Dictionary containing backtest results:
        - metrics: Performance metrics
        - trades: List of trades
        - equity_curve: Equity curve data
    """
    logger.info(f"Getting backtest results: {backtest_id}")

    manager = await get_mcp_manager()

    try:
        result = await manager.call_tool(
            "backtest",
            "results",
            {"backtest_id": backtest_id}
        )

        if isinstance(result, dict):
            return {
                "backtest_id": backtest_id,
                "status": result.get("status", "completed"),
                "metrics": result.get("metrics", {}),
                "trades": result.get("trades", []),
                "equity_curve": result.get("equity_curve", []),
                "completed_at": result.get("completed_at", datetime.now().isoformat())
            }
        return {
            "backtest_id": backtest_id,
            "status": "unknown",
            "metrics": {},
            "trades": [],
            "equity_curve": [],
            "completed_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to get backtest results: {e}")
        raise RuntimeError(f"Failed to get backtest results: {e}")


async def compare_backtests(
    backtest_ids: List[str]
) -> Dict[str, Any]:
    """
    Compare results from multiple backtests.

    Args:
        backtest_ids: List of backtest identifiers to compare

    Returns:
        Dictionary containing comparison results
    """
    logger.info(f"Comparing {len(backtest_ids)} backtests")

    manager = await get_mcp_manager()

    try:
        result = await manager.call_tool(
            "backtest",
            "compare",
            {"backtest_ids": backtest_ids}
        )

        if isinstance(result, dict):
            return {
                "backtest_ids": backtest_ids,
                "comparison": result.get("comparison", {}),
                "metrics_table": result.get("metrics_table", [])
            }
        return {
            "backtest_ids": backtest_ids,
            "comparison": {},
            "metrics_table": []
        }

    except Exception as e:
        logger.error(f"Failed to compare backtests: {e}")
        raise RuntimeError(f"Failed to compare backtests: {e}")


# =============================================================================
# MT5 Compiler MCP Tools
# =============================================================================

async def compile_mql5_code(
    code: str,
    filename: str,
    code_type: str = "expert"
) -> Dict[str, Any]:
    """
    Compile MQL5 code using MT5 Compiler MCP.

    This tool compiles MQL5 code and returns compilation results
    including any errors or warnings.

    Args:
        code: MQL5 source code to compile
        filename: Output filename (without extension)
        code_type: Type of code ("expert", "indicator", "script", "library")

    Returns:
        Dictionary containing:
        - success: Whether compilation succeeded
        - errors: List of compilation errors
        - warnings: List of compilation warnings
        - output_path: Path to compiled file (if successful)
    """
    logger.info(f"Compiling MQL5 code: {filename}")

    manager = await get_mcp_manager()

    try:
        result = await manager.call_tool(
            "metatrader5",
            "compile",
            {
                "code": code,
                "filename": filename,
                "code_type": code_type
            }
        )

        if isinstance(result, dict):
            return {
                "success": result.get("success", False),
                "filename": filename,
                "code_type": code_type,
                "errors": result.get("errors", []),
                "warnings": result.get("warnings", []),
                "output_path": result.get("output_path", f"/MT5/MQL5/Experts/{filename}.ex5"),
                "compiled_at": result.get("compiled_at", datetime.now().isoformat())
            }
        return {
            "success": False,
            "filename": filename,
            "code_type": code_type,
            "errors": ["Unknown compilation result"],
            "warnings": [],
            "output_path": None,
            "compiled_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Compilation failed: {e}")
        raise RuntimeError(f"Failed to compile MQL5 code: {e}")


async def validate_mql5_syntax(code: str) -> Dict[str, Any]:
    """
    Validate MQL5 code syntax without full compilation.

    Args:
        code: MQL5 source code to validate

    Returns:
        Dictionary containing validation results
    """
    logger.info("Validating MQL5 syntax")

    manager = await get_mcp_manager()

    try:
        result = await manager.call_tool(
            "metatrader5",
            "validate-syntax",
            {"code": code}
        )

        if isinstance(result, dict):
            return {
                "valid": result.get("valid", False),
                "errors": result.get("errors", []),
                "warnings": result.get("warnings", []),
                "suggestions": result.get("suggestions", [])
            }
        return {
            "valid": False,
            "errors": ["Unknown validation result"],
            "warnings": [],
            "suggestions": []
        }

    except Exception as e:
        logger.error(f"Syntax validation failed: {e}")
        raise RuntimeError(f"Failed to validate MQL5 syntax: {e}")


async def get_compilation_errors(
    error_codes: List[str]
) -> Dict[str, Any]:
    """
    Get detailed information about MQL5 compilation errors.

    Args:
        error_codes: List of MQL5 error codes

    Returns:
        Dictionary containing error details and suggested fixes
    """
    logger.info(f"Getting details for {len(error_codes)} error codes")

    manager = await get_mcp_manager()

    try:
        result = await manager.call_tool(
            "metatrader5",
            "get-error-details",
            {"error_codes": error_codes}
        )

        if isinstance(result, dict):
            return {
                "errors": result.get("errors", [])
            }
        return {"errors": []}

    except Exception as e:
        logger.error(f"Failed to get error details: {e}")
        raise RuntimeError(f"Failed to get compilation error details: {e}")


# =============================================================================
# Tool Registry for LangGraph Integration
# =============================================================================

MCP_TOOLS = {
    # Context7 tools
    "get_mql5_documentation": {
        "function": get_mql5_documentation,
        "description": "Retrieve MQL5 documentation from Context7",
        "parameters": {
            "query": {"type": "string", "required": True},
            "context": {"type": "string", "required": False},
            "max_results": {"type": "integer", "required": False, "default": 5}
        }
    },
    "get_mql5_examples": {
        "function": get_mql5_examples,
        "description": "Get MQL5 code examples for a specific topic",
        "parameters": {
            "topic": {"type": "string", "required": True},
            "language": {"type": "string", "required": False, "default": "mql5"}
        }
    },

    # Sequential Thinking tools
    "sequential_thinking": {
        "function": sequential_thinking,
        "description": "Decompose a complex task into sequential steps",
        "parameters": {
            "task": {"type": "string", "required": True},
            "context": {"type": "string", "required": False},
            "max_steps": {"type": "integer", "required": False, "default": 10}
        }
    },
    "analyze_errors": {
        "function": analyze_errors,
        "description": "Analyze errors and suggest fixes using sequential thinking",
        "parameters": {
            "errors": {"type": "array", "items": {"type": "string"}, "required": True},
            "code": {"type": "string", "required": False}
        }
    },

    # PageIndex tools
    "index_pdf": {
        "function": index_pdf,
        "description": "Index a PDF document using PageIndex",
        "parameters": {
            "pdf_path": {"type": "string", "required": True},
            "namespace": {"type": "string", "required": True},
            "metadata": {"type": "object", "required": False}
        }
    },
    "search_pdf": {
        "function": search_pdf,
        "description": "Search indexed PDFs using PageIndex",
        "parameters": {
            "query": {"type": "string", "required": True},
            "namespace": {"type": "string", "required": True},
            "max_results": {"type": "integer", "required": False, "default": 5},
            "include_content": {"type": "boolean", "required": False, "default": True}
        }
    },
    "get_indexed_documents": {
        "function": get_indexed_documents,
        "description": "Get list of indexed documents in a namespace",
        "parameters": {
            "namespace": {"type": "string", "required": True}
        }
    },

    # Backtest tools
    "run_backtest": {
        "function": run_backtest,
        "description": "Run a backtest for MQL5 strategy code",
        "parameters": {
            "code": {"type": "string", "required": True},
            "config": {"type": "object", "required": True},
            "strategy_name": {"type": "string", "required": False}
        }
    },
    "get_backtest_status": {
        "function": get_backtest_status,
        "description": "Get status of a backtest job",
        "parameters": {
            "backtest_id": {"type": "string", "required": True}
        }
    },
    "get_backtest_results": {
        "function": get_backtest_results,
        "description": "Get results of a completed backtest",
        "parameters": {
            "backtest_id": {"type": "string", "required": True}
        }
    },
    "compare_backtests": {
        "function": compare_backtests,
        "description": "Compare results from multiple backtests",
        "parameters": {
            "backtest_ids": {"type": "array", "items": {"type": "string"}, "required": True}
        }
    },

    # MT5 Compiler tools
    "compile_mql5_code": {
        "function": compile_mql5_code,
        "description": "Compile MQL5 code using MT5 Compiler MCP",
        "parameters": {
            "code": {"type": "string", "required": True},
            "filename": {"type": "string", "required": True},
            "code_type": {"type": "string", "required": False, "default": "expert"}
        }
    },
    "validate_mql5_syntax": {
        "function": validate_mql5_syntax,
        "description": "Validate MQL5 code syntax without full compilation",
        "parameters": {
            "code": {"type": "string", "required": True}
        }
    },
    "get_compilation_errors": {
        "function": get_compilation_errors,
        "description": "Get detailed information about MQL5 compilation errors",
        "parameters": {
            "error_codes": {"type": "array", "items": {"type": "string"}, "required": True}
        }
    },
}


def get_mcp_tool(name: str) -> Optional[Dict[str, Any]]:
    """
    Get an MCP tool by name.

    Args:
        name: Tool name

    Returns:
        Tool definition or None if not found
    """
    return MCP_TOOLS.get(name)


def list_mcp_tools() -> List[str]:
    """
    List all available MCP tools.

    Returns:
        List of tool names
    """
    return list(MCP_TOOLS.keys())


async def invoke_mcp_tool(name: str, **kwargs) -> Dict[str, Any]:
    """
    Invoke an MCP tool by name.

    Args:
        name: Tool name
        **kwargs: Tool arguments

    Returns:
        Tool result
    """
    tool = get_mcp_tool(name)
    if not tool:
        raise ValueError(f"Unknown MCP tool: {name}")

    func = tool["function"]
    return await func(**kwargs)

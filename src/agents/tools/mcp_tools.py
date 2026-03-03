"""
MCP Tool Wrappers for QuantMind Agents.

This module provides tool wrappers for all MCP servers:
- Context7: MQL5 documentation retrieval
- Sequential Thinking: Task decomposition
- PageIndex: PDF indexing and search
- Backtest: Strategy backtesting
- MT5 Compiler: MQL5 code compilation

Uses the real MCP client layer for actual server connections.

NOTE: This module has been refactored into a modular structure.
For new code, prefer importing from src.agents.tools.mcp subpackage:
    from src.agents.tools.mcp import get_mql5_documentation

This module is maintained for backward compatibility.
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
# Re-export from modular structure for backward compatibility
# =============================================================================

# Manager
from src.agents.tools.mcp.manager import (
    MCPClientManager,
    get_mcp_manager,
)

# Context7 tools
from src.agents.tools.mcp.context7 import (
    get_mql5_documentation,
    get_mql5_examples,
)

# Sequential Thinking tools
from src.agents.tools.mcp.sequential_thinking import (
    sequential_thinking,
    analyze_errors,
)

# PageIndex tools
from src.agents.tools.mcp.page_index import (
    index_pdf,
    search_pdf,
    get_indexed_documents,
)

# Backtest tools
from src.agents.tools.mcp.backtest import (
    run_backtest,
    get_backtest_status,
    get_backtest_results,
    compare_backtests,
)

# MT5 Compiler tools
from src.agents.tools.mcp.mt5_compiler import (
    compile_mql5_code,
    validate_mql5_syntax,
    get_compilation_errors,
)


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


# Backward compatibility: Keep the old module-level _mcp_manager for direct usage
_mcp_manager: Optional[MCPClientManager] = None

"""
SDK Tools for Claude Agent SDK

Converts existing tool registries to SDK-compatible format.
Provides tool definitions for use with the SDK orchestrator.
"""

import logging
from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ToolCategory(Enum):
    """Categories of tools available."""
    MEMORY = "memory"
    KNOWLEDGE = "knowledge"
    TRADING = "trading"
    ANALYSIS = "analysis"
    MCP = "mcp"


@dataclass
class SDKTool:
    """SDK-compatible tool definition."""
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Optional[Callable] = None
    category: ToolCategory = ToolCategory.ANALYSIS

    def to_anthropic_format(self) -> Dict[str, Any]:
        """Convert to Anthropic tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": self.parameters,
                "required": [k for k, v in self.parameters.items()
                           if v.get("required", False)]
            }
        }


# =============================================================================
# Memory Tools (from tools/memory_tools.py)
# =============================================================================

MEMORY_TOOLS: Dict[str, SDKTool] = {
    "add_semantic_memory": SDKTool(
        name="add_semantic_memory",
        description="Add a semantic memory to the agent's memory store",
        parameters={
            "content": {"type": "string", "description": "The memory content to store"},
            "agent_name": {"type": "string", "description": "Agent identifier", "default": "analyst"},
        },
        category=ToolCategory.MEMORY,
    ),
    "search_memories": SDKTool(
        name="search_memories",
        description="Search through stored memories",
        parameters={
            "query": {"type": "string", "description": "Search query"},
            "limit": {"type": "integer", "description": "Max results", "default": 10},
        },
        category=ToolCategory.MEMORY,
    ),
    "get_memory_stats": SDKTool(
        name="get_memory_stats",
        description="Get statistics about stored memories",
        parameters={},
        category=ToolCategory.MEMORY,
    ),
    "clear_memories": SDKTool(
        name="clear_memories",
        description="Clear all stored memories for an agent",
        parameters={
            "agent_name": {"type": "string", "description": "Agent to clear memories for"},
        },
        category=ToolCategory.MEMORY,
    ),
    "add_episodic_memory": SDKTool(
        name="add_episodic_memory",
        description="Add an episodic memory (time-based event)",
        parameters={
            "event": {"type": "string", "description": "The event to remember"},
            "timestamp": {"type": "string", "description": "ISO timestamp"},
        },
        category=ToolCategory.MEMORY,
    ),
    "recall_episodes": SDKTool(
        name="recall_episodes",
        description="Recall episodic memories within a time range",
        parameters={
            "start_time": {"type": "string", "description": "Start ISO timestamp"},
            "end_time": {"type": "string", "description": "End ISO timestamp"},
        },
        category=ToolCategory.MEMORY,
    ),
}


# =============================================================================
# Knowledge Tools (from tools/knowledge_tools.py)
# =============================================================================

KNOWLEDGE_TOOLS: Dict[str, SDKTool] = {
    "search_knowledge": SDKTool(
        name="search_knowledge",
        description="Search the knowledge base for relevant information",
        parameters={
            "query": {"type": "string", "description": "Search query"},
            "namespace": {"type": "string", "description": "Knowledge namespace", "default": "default"},
        },
        category=ToolCategory.KNOWLEDGE,
    ),
    "add_knowledge": SDKTool(
        name="add_knowledge",
        description="Add new knowledge to the knowledge base",
        parameters={
            "content": {"type": "string", "description": "Knowledge content"},
            "metadata": {"type": "object", "description": "Optional metadata"},
        },
        category=ToolCategory.KNOWLEDGE,
    ),
    "list_namespaces": SDKTool(
        name="list_namespaces",
        description="List all knowledge namespaces",
        parameters={},
        category=ToolCategory.KNOWLEDGE,
    ),
    "get_knowledge_stats": SDKTool(
        name="get_knowledge_stats",
        description="Get knowledge base statistics",
        parameters={},
        category=ToolCategory.KNOWLEDGE,
    ),
}


# =============================================================================
# Trading Tools
# =============================================================================

TRADING_TOOLS: Dict[str, SDKTool] = {
    "get_positions": SDKTool(
        name="get_positions",
        description="Get current trading positions",
        parameters={
            "symbol": {"type": "string", "description": "Optional symbol filter"},
        },
        category=ToolCategory.TRADING,
    ),
    "get_account_info": SDKTool(
        name="get_account_info",
        description="Get account information and balance",
        parameters={},
        category=ToolCategory.TRADING,
    ),
    "get_market_data": SDKTool(
        name="get_market_data",
        description="Get market data for a symbol",
        parameters={
            "symbol": {"type": "string", "description": "Trading symbol"},
            "timeframe": {"type": "string", "description": "Timeframe (M1, M5, H1, etc.)"},
            "count": {"type": "integer", "description": "Number of bars", "default": 100},
        },
        category=ToolCategory.TRADING,
    ),
    "place_order": SDKTool(
        name="place_order",
        description="Place a trading order (requires confirmation)",
        parameters={
            "symbol": {"type": "string", "description": "Trading symbol"},
            "order_type": {"type": "string", "description": "BUY or SELL"},
            "volume": {"type": "number", "description": "Lot size"},
            "stop_loss": {"type": "number", "description": "Stop loss price"},
            "take_profit": {"type": "number", "description": "Take profit price"},
        },
        category=ToolCategory.TRADING,
    ),
    "get_trade_history": SDKTool(
        name="get_trade_history",
        description="Get historical trades",
        parameters={
            "symbol": {"type": "string", "description": "Optional symbol filter"},
            "limit": {"type": "integer", "description": "Max results", "default": 50},
        },
        category=ToolCategory.TRADING,
    ),
}


# =============================================================================
# Analysis Tools
# =============================================================================

ANALYSIS_TOOLS: Dict[str, SDKTool] = {
    "run_backtest": SDKTool(
        name="run_backtest",
        description="Run a backtest on a strategy",
        parameters={
            "strategy_id": {"type": "string", "description": "Strategy identifier"},
            "symbol": {"type": "string", "description": "Trading symbol"},
            "start_date": {"type": "string", "description": "Start date YYYY-MM-DD"},
            "end_date": {"type": "string", "description": "End date YYYY-MM-DD"},
        },
        category=ToolCategory.ANALYSIS,
    ),
    "get_regime": SDKTool(
        name="get_regime",
        description="Get current market regime analysis",
        parameters={
            "symbol": {"type": "string", "description": "Trading symbol"},
        },
        category=ToolCategory.ANALYSIS,
    ),
    "calculate_indicators": SDKTool(
        name="calculate_indicators",
        description="Calculate technical indicators",
        parameters={
            "symbol": {"type": "string", "description": "Trading symbol"},
            "indicators": {"type": "array", "description": "List of indicators to calculate"},
        },
        category=ToolCategory.ANALYSIS,
    ),
    "analyze_strategy": SDKTool(
        name="analyze_strategy",
        description="Analyze a trading strategy's performance",
        parameters={
            "strategy_id": {"type": "string", "description": "Strategy identifier"},
        },
        category=ToolCategory.ANALYSIS,
    ),
}


# =============================================================================
# Tool Registry
# =============================================================================

ALL_SDK_TOOLS: Dict[str, SDKTool] = {
    **MEMORY_TOOLS,
    **KNOWLEDGE_TOOLS,
    **TRADING_TOOLS,
    **ANALYSIS_TOOLS,
}


def get_tools_for_agent(agent_id: str) -> List[SDKTool]:
    """
    Get appropriate tools for a specific agent type.

    Args:
        agent_id: Agent identifier (analyst, copilot, quantcode, etc.)

    Returns:
        List of SDKTool instances for the agent
    """
    # Define tool sets for each agent type
    agent_tool_sets = {
        "analyst": list(MEMORY_TOOLS.values()) + list(KNOWLEDGE_TOOLS.values()) + list(ANALYSIS_TOOLS.values()),
        "copilot": list(ALL_SDK_TOOLS.values()),  # Copilot has access to all tools
        "quantcode": list(ANALYSIS_TOOLS.values()) + list(TRADING_TOOLS.values()),
        "pinescript": list(ANALYSIS_TOOLS.values()),
        "router": [TRADING_TOOLS["get_positions"], TRADING_TOOLS["get_account_info"]],
        "executor": list(TRADING_TOOLS.values()),
    }

    return agent_tool_sets.get(agent_id, [])


def get_tool_definitions_for_agent(agent_id: str) -> List[Dict[str, Any]]:
    """
    Get Anthropic-compatible tool definitions for an agent.

    Args:
        agent_id: Agent identifier

    Returns:
        List of tool definitions in Anthropic format
    """
    tools = get_tools_for_agent(agent_id)
    return [tool.to_anthropic_format() for tool in tools]


def get_tool_by_name(tool_name: str) -> Optional[SDKTool]:
    """
    Get a tool by its name.

    Args:
        tool_name: Name of the tool

    Returns:
        SDKTool instance or None if not found
    """
    return ALL_SDK_TOOLS.get(tool_name)


def list_all_tools() -> List[str]:
    """
    List all available tool names.

    Returns:
        List of tool names
    """
    return list(ALL_SDK_TOOLS.keys())


def get_tools_by_category(category: ToolCategory) -> List[SDKTool]:
    """
    Get all tools in a specific category.

    Args:
        category: Tool category to filter by

    Returns:
        List of SDKTool instances in the category
    """
    return [tool for tool in ALL_SDK_TOOLS.values() if tool.category == category]

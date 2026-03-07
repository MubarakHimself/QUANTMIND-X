"""
Tool access control for department agents.

LIVE TRADING IS PROHIBITED for all agents.
Risk/Position and Broker are READ-ONLY for all departments.
"""

from enum import Enum
from typing import Dict, Set, List

from .types import Department


class ToolPermission(Enum):
    """Tool permission levels."""
    READ = "read"
    WRITE = "write"


# Module-level constant for easy import
LIVE_TRADING_PROHIBITED = True


# Tool access configuration per department (Option B)
# All departments get: memory_tools, knowledge_tools, mail (WRITE)
# Strategy Router, Risk/Position, Broker are READ-ONLY for most
TOOL_ACCESS: Dict[str, Dict[str, Set[ToolPermission]]] = {
    # RESEARCH: Strategy development + Analysis
    # Tools: knowledge_tools, backtest_tools, memory_tools, mail (WRITE)
    "research": {
        "memory_tools": {ToolPermission.READ, ToolPermission.WRITE},
        "knowledge_tools": {ToolPermission.READ, ToolPermission.WRITE},
        "backtest_tools": {ToolPermission.READ, ToolPermission.WRITE},
        "strategy_router": {ToolPermission.READ},  # READ-ONLY
        "risk_tools": {ToolPermission.READ},       # READ-ONLY
        "broker_tools": {ToolPermission.READ},      # READ-ONLY
        "strategy_extraction": {ToolPermission.READ, ToolPermission.WRITE},
        "video_ingest": {ToolPermission.READ, ToolPermission.WRITE},  # Video processing
        "gemini_cli": {ToolPermission.READ, ToolPermission.WRITE},  # Gemini CLI research
        "prop_firm_research": {ToolPermission.READ, ToolPermission.WRITE},  # Prop firm research
        "task_list": {ToolPermission.READ, ToolPermission.WRITE},  # Task list management
        "mail": {ToolPermission.READ, ToolPermission.WRITE},
    },
    # DEVELOPMENT: EA/Bot building (Python, PineScript, MQL5)
    # Tools: mql5_tools, pinescript_tools, backtest_tools, knowledge_tools, memory_tools, mail
    "development": {
        "memory_tools": {ToolPermission.READ, ToolPermission.WRITE},
        "knowledge_tools": {ToolPermission.READ, ToolPermission.WRITE},
        "backtest_tools": {ToolPermission.READ, ToolPermission.WRITE},
        "pinescript_tools": {ToolPermission.READ, ToolPermission.WRITE},
        "mql5_tools": {ToolPermission.READ, ToolPermission.WRITE},
        "ea_lifecycle": {ToolPermission.READ, ToolPermission.WRITE},
        "strategy_router": {ToolPermission.READ},   # READ-ONLY
        "risk_tools": {ToolPermission.READ},         # READ-ONLY
        "broker_tools": {ToolPermission.READ},       # READ-ONLY
        "strategy_extraction": {ToolPermission.READ, ToolPermission.WRITE},
        "video_ingest": {ToolPermission.READ, ToolPermission.WRITE},  # Video processing
        "gemini_cli": {ToolPermission.READ, ToolPermission.WRITE},  # Gemini CLI research
        "prop_firm_research": {ToolPermission.READ},  # Prop firm research (READ)
        "task_list": {ToolPermission.READ, ToolPermission.WRITE},  # Task list management
        "mail": {ToolPermission.READ, ToolPermission.WRITE},
    },
    # TRADING: Order execution (READ-ONLY for broker)
    # Tools: broker_tools (READ), memory_tools, mail
    "trading": {
        "memory_tools": {ToolPermission.READ, ToolPermission.WRITE},
        "knowledge_tools": {ToolPermission.READ},
        "broker_tools": {ToolPermission.READ},       # READ-ONLY (no placing orders)
        "strategy_router": {ToolPermission.READ},     # READ-ONLY
        "risk_tools": {ToolPermission.READ},         # READ-ONLY
        "prop_firm_research": {ToolPermission.READ},  # Prop firm research (READ)
        "task_list": {ToolPermission.READ, ToolPermission.WRITE},  # Task list management
        "shared_assets": {ToolPermission.READ},      # Shared assets (READ)
        "mail": {ToolPermission.READ, ToolPermission.WRITE},
    },
    # RISK: Risk management (READ-ONLY)
    # Tools: risk_tools (READ), memory_tools, mail
    "risk": {
        "memory_tools": {ToolPermission.READ, ToolPermission.WRITE},
        "knowledge_tools": {ToolPermission.READ},
        "risk_tools": {ToolPermission.READ},          # READ-ONLY (no setting limits)
        "strategy_router": {ToolPermission.READ},      # READ-ONLY
        "broker_tools": {ToolPermission.READ},        # READ-ONLY
        "prop_firm_research": {ToolPermission.READ},   # Prop firm research (READ)
        "task_list": {ToolPermission.READ, ToolPermission.WRITE},  # Task list management
        "shared_assets": {ToolPermission.READ},       # Shared assets (READ)
        "mail": {ToolPermission.READ, ToolPermission.WRITE},
    },
    # PORTFOLIO: Allocation
    # Tools: memory_tools, mail
    "portfolio": {
        "memory_tools": {ToolPermission.READ, ToolPermission.WRITE},
        "knowledge_tools": {ToolPermission.READ},
        "strategy_router": {ToolPermission.READ},      # READ-ONLY
        "risk_tools": {ToolPermission.READ},           # READ-ONLY
        "broker_tools": {ToolPermission.READ},         # READ-ONLY
        "strategy_extraction": {ToolPermission.READ},
        "prop_firm_research": {ToolPermission.READ},   # Prop firm research (READ)
        "task_list": {ToolPermission.READ, ToolPermission.WRITE},  # Task list management
        "shared_assets": {ToolPermission.READ, ToolPermission.WRITE},  # Shared assets
        "mail": {ToolPermission.READ, ToolPermission.WRITE},
    },
    # FLOOR MANAGER: Cross-department coordination
    "floor_manager": {
        "memory_tools": {ToolPermission.READ, ToolPermission.WRITE},
        "memory_all_depts": {ToolPermission.READ},     # Cross-dept read access
        "knowledge_tools": {ToolPermission.READ},
        "strategy_router": {ToolPermission.READ},
        "risk_tools": {ToolPermission.READ},
        "broker_tools": {ToolPermission.READ},
        "gemini_cli": {ToolPermission.READ, ToolPermission.WRITE},  # Gemini CLI research
        "prop_firm_research": {ToolPermission.READ, ToolPermission.WRITE},  # Prop firm research
        "task_list": {ToolPermission.READ, ToolPermission.WRITE},  # Task list management
        "shared_assets": {ToolPermission.READ, ToolPermission.WRITE},  # Shared assets
        "mail": {ToolPermission.READ, ToolPermission.WRITE},
    },
}


class ToolAccessController:
    """Controls tool access for department agents."""

    # Hard constraint: LIVE TRADING IS PROHIBITED
    LIVE_TRADING_PROHIBITED = LIVE_TRADING_PROHIBITED

    def __init__(self, department: Department):
        self.department = department
        self.permissions = TOOL_ACCESS.get(department.value, {})

    def can_access(self, tool_name: str, permission: ToolPermission) -> bool:
        """Check if department has permission for tool."""
        if tool_name == "live_trading":
            return False  # Always prohibited
        return permission in self.permissions.get(tool_name, set())

    def filter_tools(self, tools: List[str]) -> List[str]:
        """Filter available tools based on department permissions."""
        return [
            tool for tool in tools
            if self.can_access(tool, ToolPermission.READ) or
               self.can_access(tool, ToolPermission.WRITE)
        ]

    def get_available_tools(self) -> List[str]:
        """Get list of all available tools for this department."""
        return list(self.permissions.keys())

    def has_write_access(self, tool_name: str) -> bool:
        """Check if department has WRITE access to a tool."""
        return self.can_access(tool_name, ToolPermission.WRITE)

    def has_read_access(self, tool_name: str) -> bool:
        """Check if department has READ access to a tool."""
        return self.can_access(tool_name, ToolPermission.READ)

    def is_live_trading_prohibited(self) -> bool:
        """Check if live trading is prohibited for this department."""
        return self.LIVE_TRADING_PROHIBITED


__all__ = [
    "ToolPermission",
    "ToolAccessController",
    "TOOL_ACCESS",
    "LIVE_TRADING_PROHIBITED",
]

"""
Analyst Agent CLI - Converts video/strategy content to TRD files.

This package provides a CLI tool that:
1. Reads NPRD video outputs (unstructured trading content)
2. Cross-references with a filtered knowledge base (ChromaDB)
3. Generates structured Markdown TRD (Technical Requirements Document) files
4. Supports human-in-the-loop interaction for missing information

Version: 1.0 (Fast/MVP)
"""

__version__ = "1.0.0"
__author__ = "QuantMindX"

# Package exports
from .kb.client import ChromaKBClient
from .graph.state import AnalystState

# Agent framework exports
from .agent import (
    BaseAgent,
    AgentConfig,
    AgentCapabilities,
    Skill,
    SkillRegistry,
    Tool,
    ToolRegistry,
    MCPClient,
    MCPTool,
    AnalystAgent,
    create_analyst_agent,
    get_analyst_agent
)

__all__ = [
    "ChromaKBClient",
    "AnalystState",
    "BaseAgent",
    "AgentConfig",
    "AgentCapabilities",
    "Skill",
    "SkillRegistry",
    "Tool",
    "ToolRegistry",
    "MCPClient",
    "MCPTool",
    "AnalystAgent",
    "create_analyst_agent",
    "get_analyst_agent",
    "__version__",
]

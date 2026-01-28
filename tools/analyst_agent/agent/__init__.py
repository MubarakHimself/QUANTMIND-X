"""
QuantMindX Agent Framework

Provides a unified base for all agents with:
- Skills system
- Tools framework
- MCP (Model Context Protocol) access
- Memory integration with LangMem
- RAG (Retrieval-Augmented Generation)
- Centralized Analyst Agent
"""

from .base import BaseAgent, AgentConfig, AgentCapabilities
from .skills import Skill, SkillRegistry
from .tools import Tool, ToolRegistry
from .mcp import MCPClient, MCPTool
from .analyst_agent import AnalystAgent, create_analyst_agent, get_analyst_agent
from .rag import (
    ChromaDBRetriever,
    create_retriever_tool,
    create_rag_chain,
    create_agent_with_retrieval,
    RAGEnabledAgent,
    enable_rag_for_agent,
    create_analyst_agent_with_rag
)

__all__ = [
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
    "ChromaDBRetriever",
    "create_retriever_tool",
    "create_rag_chain",
    "create_agent_with_retrieval",
    "RAGEnabledAgent",
    "enable_rag_for_agent",
    "create_analyst_agent_with_rag",
]

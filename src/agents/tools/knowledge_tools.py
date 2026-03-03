"""
Knowledge Base Tools for QuantMind Agents.

This module provides tools for agents to interact with the PageIndex knowledge base,
including MQL5 book search and custom PDF indexing/retrieval.

Uses real PageIndex MCP calls for all indexing and search operations.

NOTE: This module has been refactored into a modular structure.
The functionality has been moved to src/agents/tools/knowledge/
This file is kept for backward compatibility and imports from the new modules.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


# =============================================================================
# Backward Compatibility Imports
# =============================================================================
# Import all functions and classes from the new modular structure
# to maintain backward compatibility

from src.agents.tools.knowledge.client import (
    get_pageindex_manager,
    call_pageindex_tool,
    PageIndexClient,
)

from src.agents.tools.knowledge.mql5_book import (
    search_mql5_book,
    get_mql5_book_section,
)

from src.agents.tools.knowledge.knowledge_hub import (
    search_knowledge_hub,
    get_article_content,
    list_knowledge_namespaces,
)

from src.agents.tools.knowledge.pdf_indexing import (
    index_pdf_document,
    get_indexing_status,
    list_indexed_documents,
    remove_indexed_document,
)

from src.agents.tools.knowledge.strategies import (
    search_strategy_patterns,
    get_indicator_template,
)

from src.agents.tools.knowledge.registry import (
    KNOWLEDGE_TOOLS,
    get_knowledge_tool,
    list_knowledge_tools,
    invoke_knowledge_tool,
)


# =============================================================================
# Tool Registry for LangGraph Integration
# =============================================================================
# KNOWLEDGE_TOOLS is imported from registry for backward compatibility

def get_knowledge_tool(name: str) -> Optional[Dict[str, Any]]:
    """
    Get a knowledge tool by name.

    Args:
        name: Tool name

    Returns:
        Tool definition or None if not found
    """
    return KNOWLEDGE_TOOLS.get(name)


def list_knowledge_tools() -> List[str]:
    """
    List all available knowledge tools.

    Returns:
        List of tool names
    """
    return list(KNOWLEDGE_TOOLS.keys())


async def invoke_knowledge_tool(name: str, **kwargs) -> Dict[str, Any]:
    """
    Invoke a knowledge tool by name.

    Args:
        name: Tool name
        **kwargs: Tool arguments

    Returns:
        Tool result
    """
    tool = get_knowledge_tool(name)
    if not tool:
        raise ValueError(f"Unknown knowledge tool: {name}")

    func = tool["function"]
    return await func(**kwargs)

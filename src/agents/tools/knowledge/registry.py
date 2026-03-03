# src/agents/tools/knowledge/registry.py
"""Tool Registry for LangGraph Integration."""

import logging
from typing import Dict, Any, List, Optional, Callable

from src.agents.tools.knowledge.mql5_book import search_mql5_book, get_mql5_book_section
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

logger = logging.getLogger(__name__)

# =============================================================================
# Tool Registry
# =============================================================================

KNOWLEDGE_TOOLS: Dict[str, Dict[str, Any]] = {
    "search_mql5_book": {
        "function": search_mql5_book,
        "description": "Search the MQL5 programming book for relevant content",
        "parameters": {
            "query": {"type": "string", "required": True},
            "max_results": {"type": "integer", "required": False, "default": 5},
            "include_content": {"type": "boolean", "required": False, "default": True}
        }
    },
    "get_mql5_book_section": {
        "function": get_mql5_book_section,
        "description": "Get a specific section from the MQL5 book by page number",
        "parameters": {
            "page_number": {"type": "integer", "required": True}
        }
    },
    "search_knowledge_hub": {
        "function": search_knowledge_hub,
        "description": "Search all indexed PDFs in the knowledge hub",
        "parameters": {
            "query": {"type": "string", "required": True},
            "namespaces": {"type": "array", "items": {"type": "string"}, "required": False},
            "max_results": {"type": "integer", "required": False, "default": 10},
            "include_content": {"type": "boolean", "required": False, "default": True}
        }
    },
    "get_article_content": {
        "function": get_article_content,
        "description": "Retrieve full content of an indexed article/document",
        "parameters": {
            "article_id": {"type": "string", "required": True},
            "namespace": {"type": "string", "required": True}
        }
    },
    "list_knowledge_namespaces": {
        "function": list_knowledge_namespaces,
        "description": "List all available knowledge namespaces",
        "parameters": {}
    },
    "index_pdf_document": {
        "function": index_pdf_document,
        "description": "Index a PDF document into the knowledge hub",
        "parameters": {
            "pdf_path": {"type": "string", "required": True},
            "namespace": {"type": "string", "required": True},
            "metadata": {"type": "object", "required": False}
        }
    },
    "get_indexing_status": {
        "function": get_indexing_status,
        "description": "Get status of a PDF indexing job",
        "parameters": {
            "job_id": {"type": "string", "required": True}
        }
    },
    "list_indexed_documents": {
        "function": list_indexed_documents,
        "description": "List all indexed documents in a namespace",
        "parameters": {
            "namespace": {"type": "string", "required": True}
        }
    },
    "remove_indexed_document": {
        "function": remove_indexed_document,
        "description": "Remove an indexed document from the knowledge hub",
        "parameters": {
            "document_id": {"type": "string", "required": True},
            "namespace": {"type": "string", "required": True}
        }
    },
    "search_strategy_patterns": {
        "function": search_strategy_patterns,
        "description": "Search for trading strategy patterns and examples",
        "parameters": {
            "pattern_type": {"type": "string", "required": True},
            "context": {"type": "string", "required": False}
        }
    },
    "get_indicator_template": {
        "function": get_indicator_template,
        "description": "Get MQL5 code template for a specific indicator",
        "parameters": {
            "indicator_name": {"type": "string", "required": True}
        }
    },
}


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

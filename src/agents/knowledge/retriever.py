"""
Knowledge Base Retrieval
Standardized RAG tool for QuantMindAgents using PageIndex.

NOTE: LangChain tool decorator removed - pending migration to Anthropic Agent SDK (Epic 7).
"""

import json
from typing import List, Optional, Dict, Any

# LangChain tool decorator removed - using plain functions
# from langchain_core.tools import tool

from pydantic import BaseModel, Field

from src.agents.knowledge.router import kb_router, PageIndexClient

# Re-export PageIndexClient for external use
__all__ = ["PageIndexClient", "search_knowledge_base", "search_all_collections", "get_retrieval_tool", "get_all_retrieval_tools"]


# =============================================================================
# Tool Decorator Stub
# =============================================================================

def tool(name: str, args_schema=None):
    """Stub decorator for langchain tool - pending Anthropic Agent SDK migration."""
    def decorator(func):
        func._name = name
        func._is_tool = True
        return func
    return decorator


class SearchInput(BaseModel):
    query: str = Field(description="The research query or topic to search for.")
    collection: str = Field(
        default="articles",
        description="The KB collection: 'articles' (MQL5 articles), 'books' (PDFs/docs), or 'logs' (trading logs)."
    )


class SearchResult(BaseModel):
    """Structured search result from PageIndex."""
    content: str
    source: str
    score: float
    page: Optional[int] = None
    section: Optional[str] = None
    collection: str


@tool("search_knowledge_base", args_schema=SearchInput)
def search_knowledge_base(query: str, collection: str = "articles") -> Dict[str, Any]:
    """
    Search the QuantMindX Knowledge Base for specialized information.
    Uses PageIndex reasoning-based retrieval for better explainability.
    
    Collections:
    - articles: MQL5 articles from mql5.com (trading strategies, indicators, EAs)
    - books: PDF books and documentation
    - logs: Trading logs and historical data
    
    Returns:
        A structured dict with either:
        - {"results": [...], "total": n} on success
        - {"error": "error message"} on failure
    """
    # Validate collection
    valid_collections = ["articles", "books", "logs"]
    if collection not in valid_collections:
        return {"error": f"Invalid collection '{collection}'. Valid options: {valid_collections}"}
    
    # Use the knowledge router to search PageIndex
    try:
        results = kb_router.search(query, collection=collection, limit=5)
    except Exception as e:
        return {"error": f"Search failed: {str(e)}"}
    
    if not results:
        return {"results": [], "total": 0, "query": query, "collection": collection}
    
    # Return structured results
    return {
        "results": results,
        "total": len(results),
        "query": query,
        "collection": collection
    }


@tool("search_all_collections")
def search_all_collections(query: str, limit_per_collection: int = 3) -> str:
    """
    Search across all knowledge base collections at once.
    Useful for comprehensive research across articles, books, and logs.
    
    Args:
        query: The search query
        limit_per_collection: Max results per collection (default 3)
        
    Returns:
        Formatted results from all collections
    """
    all_results = kb_router.search_all(query, limit_per_collection=limit_per_collection)
    
    if not all_results or all(len(r) == 0 for r in all_results.values()):
        return f"No results found for '{query}' across any collection."
    
    response_parts = [f"Search results for '{query}' across all collections:\n"]
    
    for collection, results in all_results.items():
        if results:
            response_parts.append(f"\n=== {collection.upper()} ({len(results)} results) ===")
            for i, result in enumerate(results, 1):
                citation = result.get("source", "unknown")
                if result.get("page"):
                    citation += f" (page {result.get('page')})"
                response_parts.append(
                    f"[{i}] {result.get('content', '')[:300]}...\n"
                    f"    Source: {citation}"
                )
    
    return "\n".join(response_parts)


def get_retrieval_tool():
    """Get the primary knowledge retrieval tool."""
    return search_knowledge_base


def get_all_retrieval_tools():
    """Get all knowledge retrieval tools."""
    return [search_knowledge_base, search_all_collections]
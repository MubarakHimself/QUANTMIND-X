# src/agents/tools/knowledge/__init__.py
"""Knowledge tools modular package.

This package contains modular implementations of knowledge base tools:
- client: PageIndex MCP client
- mql5_book: MQL5 book search tools
- knowledge_hub: Knowledge hub search tools
- pdf_indexing: PDF indexing tools
- strategies: Strategy-specific tools
- registry: Tool registry for LangGraph integration
"""

from src.agents.tools.knowledge.client import PageIndexClient
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
from src.agents.tools.knowledge.registry import (
    KNOWLEDGE_TOOLS,
    get_knowledge_tool,
    list_knowledge_tools,
    invoke_knowledge_tool,
)

__all__ = [
    "PageIndexClient",
    "search_mql5_book",
    "get_mql5_book_section",
    "search_knowledge_hub",
    "get_article_content",
    "list_knowledge_namespaces",
    "index_pdf_document",
    "get_indexing_status",
    "list_indexed_documents",
    "remove_indexed_document",
    "search_strategy_patterns",
    "get_indicator_template",
    "KNOWLEDGE_TOOLS",
    "get_knowledge_tool",
    "list_knowledge_tools",
    "invoke_knowledge_tool",
]

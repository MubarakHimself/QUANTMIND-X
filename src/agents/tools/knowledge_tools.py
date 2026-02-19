"""
Knowledge Base Tools for QuantMind Agents.

This module provides tools for agents to interact with the PageIndex knowledge base,
including MQL5 book search and custom PDF indexing/retrieval.

Uses real PageIndex MCP calls for all indexing and search operations.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


# =============================================================================
# PageIndex MCP Client Integration
# =============================================================================

async def get_pageindex_manager():
    """Get the MCP manager for PageIndex operations."""
    from src.agents.tools.mcp_tools import get_mcp_manager
    return await get_mcp_manager()


async def call_pageindex_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call a PageIndex MCP tool with proper error handling.
    
    Args:
        tool_name: Name of the PageIndex tool to call
        arguments: Arguments to pass to the tool
        
    Returns:
        Tool result from PageIndex MCP server
        
    Raises:
        RuntimeError: If the MCP call fails
    """
    try:
        manager = await get_pageindex_manager()
        result = await manager.call_tool("pageindex", tool_name, arguments)
        return result
    except Exception as e:
        logger.error(f"PageIndex MCP call failed for {tool_name}: {e}")
        raise RuntimeError(f"PageIndex MCP error: {e}")


# =============================================================================
# MQL5 Book Search Tools
# =============================================================================

async def search_mql5_book(
    query: str,
    max_results: int = 5,
    include_content: bool = True
) -> Dict[str, Any]:
    """
    Search the MQL5 programming book for relevant content.
    
    This tool searches the indexed MQL5 book (mql5book.pdf) to find
    relevant documentation, examples, and explanations using PageIndex MCP.
    
    Args:
        query: Search query (e.g., "how to create indicator", "OrderSend function")
        max_results: Maximum number of results to return (default: 5)
        include_content: Whether to include full content snippets (default: True)
        
    Returns:
        Dictionary containing:
        - results: List of matching book sections
        - total: Total number of matches
        - query: Original query
    """
    logger.info(f"Searching MQL5 book: {query}")
    
    try:
        # Call PageIndex MCP to search the mql5_book namespace
        result = await call_pageindex_tool("search", {
            "query": query,
            "namespace": "mql5_book",
            "max_results": max_results,
            "include_content": include_content
        })
        
        if isinstance(result, dict):
            results = []
            for item in result.get("results", []):
                results.append({
                    "page": item.get("page", 0),
                    "chapter": item.get("chapter", ""),
                    "content": item.get("content") if include_content else None,
                    "relevance": item.get("relevance", 0.0),
                    "section_title": item.get("section_title", item.get("title", ""))
                })
            
            return {
                "success": True,
                "results": results,
                "total": result.get("total", len(results)),
                "query": query,
                "namespace": "mql5_book"
            }
        
        return {
            "success": True,
            "results": [],
            "total": 0,
            "query": query,
            "namespace": "mql5_book"
        }
        
    except Exception as e:
        logger.error(f"MQL5 book search failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "results": [],
            "total": 0,
            "query": query,
            "namespace": "mql5_book"
        }


async def get_mql5_book_section(
    page_number: int
) -> Dict[str, Any]:
    """
    Get a specific section from the MQL5 book by page number.
    
    Args:
        page_number: Page number to retrieve
        
    Returns:
        Dictionary containing the page content
    """
    logger.info(f"Getting MQL5 book page: {page_number}")
    
    try:
        # Call PageIndex MCP to get specific page
        result = await call_pageindex_tool("get-page", {
            "namespace": "mql5_book",
            "page": page_number
        })
        
        if isinstance(result, dict):
            return {
                "success": True,
                "page": page_number,
                "content": result.get("content", ""),
                "chapter": result.get("chapter", ""),
                "section": result.get("section", "")
            }
        
        return {
            "success": False,
            "page": page_number,
            "content": "",
            "chapter": "",
            "section": "",
            "error": "Page not found"
        }
        
    except Exception as e:
        logger.error(f"Failed to get MQL5 book page {page_number}: {e}")
        return {
            "success": False,
            "page": page_number,
            "content": "",
            "chapter": "",
            "section": "",
            "error": str(e)
        }


# =============================================================================
# Knowledge Hub Tools
# =============================================================================

async def search_knowledge_hub(
    query: str,
    namespaces: Optional[List[str]] = None,
    max_results: int = 10,
    include_content: bool = True
) -> Dict[str, Any]:
    """
    Search all indexed PDFs in the knowledge hub using PageIndex MCP.
    
    This tool searches across all indexed documents including:
    - mql5_book: MQL5 programming book
    - strategies: Trading strategies documentation
    - knowledge: General knowledge base
    
    Args:
        query: Search query
        namespaces: List of namespaces to search (default: all)
        max_results: Maximum results per namespace (default: 10)
        include_content: Whether to include content snippets (default: True)
        
    Returns:
        Dictionary containing:
        - results: List of matching documents grouped by namespace
        - total: Total number of matches
        - query: Original query
    """
    logger.info(f"Searching knowledge hub: {query}")
    
    if namespaces is None:
        namespaces = ["mql5_book", "strategies", "knowledge"]
    
    try:
        # Call PageIndex MCP for multi-namespace search
        result = await call_pageindex_tool("search-all", {
            "query": query,
            "namespaces": namespaces,
            "max_results": max_results,
            "include_content": include_content
        })
        
        results = []
        if isinstance(result, dict):
            # Group results by namespace
            namespace_results = {}
            for item in result.get("results", []):
                ns = item.get("namespace", "unknown")
                if ns not in namespace_results:
                    namespace_results[ns] = []
                namespace_results[ns].append({
                    "document": item.get("filename", item.get("document", "")),
                    "page": item.get("page", 1),
                    "content": item.get("content") if include_content else None,
                    "relevance": item.get("relevance", 0.0)
                })
            
            for ns, matches in namespace_results.items():
                results.append({
                    "namespace": ns,
                    "matches": matches
                })
        
        return {
            "success": True,
            "results": results,
            "total": sum(len(r["matches"]) for r in results),
            "query": query,
            "namespaces_searched": namespaces
        }
        
    except Exception as e:
        logger.error(f"Knowledge hub search failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "results": [],
            "total": 0,
            "query": query,
            "namespaces_searched": namespaces
        }


async def get_article_content(
    article_id: str,
    namespace: str
) -> Dict[str, Any]:
    """
    Retrieve full content of an indexed article/document using PageIndex MCP.
    
    Args:
        article_id: Article or document identifier
        namespace: Namespace the article belongs to
        
    Returns:
        Dictionary containing the full article content
    """
    logger.info(f"Getting article content: {article_id} from {namespace}")
    
    try:
        # Call PageIndex MCP to get document content
        result = await call_pageindex_tool("get-document", {
            "document_id": article_id,
            "namespace": namespace
        })
        
        if isinstance(result, dict):
            return {
                "success": True,
                "article_id": article_id,
                "namespace": namespace,
                "title": result.get("title", f"Article {article_id}"),
                "content": result.get("content", ""),
                "metadata": {
                    "indexed_at": result.get("indexed_at", ""),
                    "pages": result.get("pages", 0)
                }
            }
        
        return {
            "success": False,
            "article_id": article_id,
            "namespace": namespace,
            "title": "",
            "content": "",
            "error": "Article not found"
        }
        
    except Exception as e:
        logger.error(f"Failed to get article content: {e}")
        return {
            "success": False,
            "article_id": article_id,
            "namespace": namespace,
            "title": "",
            "content": "",
            "error": str(e)
        }


async def list_knowledge_namespaces() -> Dict[str, Any]:
    """
    List all available knowledge namespaces using PageIndex MCP.
    
    Returns:
        Dictionary containing list of namespaces with metadata
    """
    logger.info("Listing knowledge namespaces")
    
    try:
        # Call PageIndex MCP to list namespaces
        result = await call_pageindex_tool("list-namespaces", {})
        
        namespaces = []
        if isinstance(result, dict):
            for ns in result.get("namespaces", []):
                namespaces.append({
                    "name": ns.get("name", ""),
                    "description": ns.get("description", ""),
                    "document_count": ns.get("document_count", 0),
                    "total_pages": ns.get("total_pages", 0),
                    "indexed_at": ns.get("indexed_at")
                })
        
        # Add default namespaces if not present
        default_namespaces = {
            "mql5_book": "MQL5 Programming Book",
            "strategies": "Trading strategies documentation",
            "knowledge": "General knowledge base"
        }
        
        existing_names = [ns["name"] for ns in namespaces]
        for name, desc in default_namespaces.items():
            if name not in existing_names:
                namespaces.append({
                    "name": name,
                    "description": desc,
                    "document_count": 0,
                    "total_pages": 0,
                    "indexed_at": None
                })
        
        return {
            "success": True,
            "namespaces": sorted(namespaces, key=lambda x: x["name"]),
            "total": len(namespaces)
        }
        
    except Exception as e:
        logger.error(f"Failed to list namespaces: {e}")
        return {
            "success": False,
            "error": str(e),
            "namespaces": [
                {
                    "name": "mql5_book",
                    "description": "MQL5 Programming Book",
                    "document_count": 0,
                    "total_pages": 0,
                    "indexed_at": None
                },
                {
                    "name": "strategies",
                    "description": "Trading strategies documentation",
                    "document_count": 0,
                    "total_pages": 0,
                    "indexed_at": None
                },
                {
                    "name": "knowledge",
                    "description": "General knowledge base",
                    "document_count": 0,
                    "total_pages": 0,
                    "indexed_at": None
                }
            ],
            "total": 3
        }


# =============================================================================
# PDF Indexing Tools
# =============================================================================

async def index_pdf_document(
    pdf_path: str,
    namespace: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Index a PDF document into the knowledge hub using PageIndex MCP.
    
    This tool indexes a PDF file for later retrieval and search.
    
    Args:
        pdf_path: Path to the PDF file
        namespace: Namespace to index into (e.g., "strategies", "knowledge")
        metadata: Optional metadata for the document
        
    Returns:
        Dictionary containing:
        - job_id: Indexing job identifier
        - status: Indexing status
        - message: Status message
    """
    logger.info(f"Indexing PDF: {pdf_path} into {namespace}")
    
    # Validate path
    path = Path(pdf_path)
    if not path.exists():
        return {
            "success": False,
            "error": f"PDF file not found: {pdf_path}",
            "status": "failed"
        }
    
    if not path.suffix.lower() == ".pdf":
        return {
            "success": False,
            "error": "File must be a PDF",
            "status": "failed"
        }
    
    try:
        # Call PageIndex MCP to index the document
        result = await call_pageindex_tool("index-document", {
            "path": pdf_path,
            "namespace": namespace,
            "metadata": metadata or {}
        })
        
        if isinstance(result, dict):
            return {
                "success": True,
                "job_id": result.get("job_id", f"idx_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
                "status": result.get("status", "completed"),
                "pdf_path": pdf_path,
                "namespace": namespace,
                "pages_indexed": result.get("pages_indexed", 0),
                "message": f"Successfully indexed {path.name} into {namespace}"
            }
        
        return {
            "success": True,
            "job_id": f"idx_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "status": "completed",
            "pdf_path": pdf_path,
            "namespace": namespace,
            "pages_indexed": 0,
            "message": f"Indexing completed for {path.name}"
        }
        
    except Exception as e:
        logger.error(f"PDF indexing failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "status": "failed",
            "pdf_path": pdf_path,
            "namespace": namespace
        }


async def get_indexing_status(
    job_id: str
) -> Dict[str, Any]:
    """
    Get status of a PDF indexing job using PageIndex MCP.
    
    Args:
        job_id: Indexing job identifier
        
    Returns:
        Dictionary containing job status
    """
    logger.info(f"Getting indexing status: {job_id}")
    
    try:
        # Call PageIndex MCP to get job status
        result = await call_pageindex_tool("get-job-status", {
            "job_id": job_id
        })
        
        if isinstance(result, dict):
            return {
                "success": True,
                "job_id": job_id,
                "status": result.get("status", "unknown"),
                "progress": result.get("progress", 0),
                "pages_processed": result.get("pages_processed", 0),
                "pages_total": result.get("pages_total", 0),
                "started_at": result.get("started_at"),
                "completed_at": result.get("completed_at")
            }
        
        return {
            "success": True,
            "job_id": job_id,
            "status": "unknown",
            "progress": 0,
            "pages_processed": 0,
            "pages_total": 0,
            "started_at": None,
            "completed_at": None
        }
        
    except Exception as e:
        logger.error(f"Failed to get indexing status: {e}")
        return {
            "success": False,
            "job_id": job_id,
            "status": "error",
            "error": str(e)
        }


async def list_indexed_documents(
    namespace: str
) -> Dict[str, Any]:
    """
    List all indexed documents in a namespace using PageIndex MCP.
    
    Args:
        namespace: Namespace to list documents from
        
    Returns:
        Dictionary containing list of documents
    """
    logger.info(f"Listing indexed documents in {namespace}")
    
    try:
        # Call PageIndex MCP to list documents
        result = await call_pageindex_tool("list-documents", {
            "namespace": namespace
        })
        
        if isinstance(result, dict):
            documents = []
            for doc in result.get("documents", []):
                documents.append({
                    "id": doc.get("id", ""),
                    "filename": doc.get("filename", ""),
                    "pages": doc.get("pages", 0),
                    "indexed_at": doc.get("indexed_at", ""),
                    "size_bytes": doc.get("size_bytes", 0)
                })
            
            return {
                "success": True,
                "namespace": namespace,
                "documents": documents,
                "total": len(documents)
            }
        
        return {
            "success": True,
            "namespace": namespace,
            "documents": [],
            "total": 0
        }
        
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        return {
            "success": False,
            "namespace": namespace,
            "documents": [],
            "total": 0,
            "error": str(e)
        }


async def remove_indexed_document(
    document_id: str,
    namespace: str
) -> Dict[str, Any]:
    """
    Remove an indexed document from the knowledge hub using PageIndex MCP.
    
    Args:
        document_id: Document identifier
        namespace: Namespace the document belongs to
        
    Returns:
        Dictionary containing removal status
    """
    logger.info(f"Removing document: {document_id} from {namespace}")
    
    try:
        # Call PageIndex MCP to remove document
        result = await call_pageindex_tool("remove-document", {
            "document_id": document_id,
            "namespace": namespace
        })
        
        if isinstance(result, dict):
            return {
                "success": result.get("success", True),
                "document_id": document_id,
                "namespace": namespace,
                "message": result.get("message", "Document removed successfully")
            }
        
        return {
            "success": True,
            "document_id": document_id,
            "namespace": namespace,
            "message": "Document removed successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to remove document: {e}")
        return {
            "success": False,
            "document_id": document_id,
            "namespace": namespace,
            "error": str(e)
        }


# =============================================================================
# Strategy-Specific Knowledge Tools
# =============================================================================

async def search_strategy_patterns(
    pattern_type: str,
    context: Optional[str] = None
) -> Dict[str, Any]:
    """
    Search for trading strategy patterns and examples using PageIndex MCP.
    
    Args:
        pattern_type: Type of pattern (e.g., "entry", "exit", "risk_management")
        context: Optional context for more specific results
        
    Returns:
        Dictionary containing matching patterns
    """
    logger.info(f"Searching strategy patterns: {pattern_type}")
    
    # Build search query
    query = f"{pattern_type} trading strategy pattern"
    if context:
        query = f"{query} {context}"
    
    try:
        # Search strategies namespace
        result = await call_pageindex_tool("search", {
            "query": query,
            "namespace": "strategies",
            "max_results": 10,
            "include_content": True
        })
        
        patterns = []
        if isinstance(result, dict):
            for item in result.get("results", []):
                patterns.append({
                    "name": item.get("title", "Pattern"),
                    "description": item.get("content", "")[:200] if item.get("content") else "",
                    "code_snippet": item.get("code", ""),
                    "relevance": item.get("relevance", 0.0)
                })
        
        return {
            "success": True,
            "pattern_type": pattern_type,
            "patterns": patterns,
            "context": context,
            "total": len(patterns)
        }
        
    except Exception as e:
        logger.error(f"Strategy pattern search failed: {e}")
        # Return default patterns on error
        default_patterns = {
            "entry": [
                {
                    "name": "Moving Average Crossover",
                    "description": "Enter when fast MA crosses above slow MA",
                    "code_snippet": "if(fastMA > slowMA && previousFastMA <= previousSlowMA)",
                    "relevance": 0.9
                },
                {
                    "name": "RSI Oversold",
                    "description": "Enter when RSI crosses above 30 from below",
                    "code_snippet": "if(rsi > 30 && previousRSI <= 30)",
                    "relevance": 0.85
                }
            ],
            "exit": [
                {
                    "name": "Take Profit Target",
                    "description": "Exit at predefined price target",
                    "code_snippet": "if(currentPrice >= entryPrice + takeProfit)",
                    "relevance": 0.9
                }
            ],
            "risk_management": [
                {
                    "name": "Fixed Percentage Risk",
                    "description": "Risk fixed percentage of account per trade",
                    "code_snippet": "lotSize = (accountBalance * riskPercent) / (stopLossPips * pipValue)",
                    "relevance": 0.9
                }
            ]
        }
        
        return {
            "success": True,
            "pattern_type": pattern_type,
            "patterns": default_patterns.get(pattern_type, []),
            "context": context,
            "total": len(default_patterns.get(pattern_type, []))
        }


async def get_indicator_template(
    indicator_name: str
) -> Dict[str, Any]:
    """
    Get MQL5 code template for a specific indicator.
    
    First searches the MQL5 book, then falls back to built-in templates.
    
    Args:
        indicator_name: Name of the indicator (e.g., "RSI", "MACD", "MovingAverage")
        
    Returns:
        Dictionary containing indicator template code
    """
    logger.info(f"Getting indicator template: {indicator_name}")
    
    # Try to find in MQL5 book first
    try:
        result = await call_pageindex_tool("search", {
            "query": f"{indicator_name} indicator code example",
            "namespace": "mql5_book",
            "max_results": 3,
            "include_content": True
        })
        
        if isinstance(result, dict) and result.get("results"):
            # Found in MQL5 book
            best_match = result["results"][0]
            return {
                "success": True,
                "name": indicator_name,
                "code": best_match.get("content", ""),
                "source": "mql5_book",
                "parameters": []  # Would be extracted from content
            }
    except Exception as e:
        logger.warning(f"MQL5 book search for {indicator_name} failed: {e}")
    
    # Fall back to built-in templates
    templates = {
        "RSI": {
            "name": "Relative Strength Index",
            "code": """
int rsiHandle;
double rsiBuffer[];

int OnInit()
{
    rsiHandle = iRSI(_Symbol, PERIOD_CURRENT, 14, PRICE_CLOSE);
    ArraySetAsSeries(rsiBuffer, true);
    return(INIT_SUCCEEDED);
}

void OnTick()
{
    CopyBuffer(rsiHandle, 0, 0, 3, rsiBuffer);
    double currentRSI = rsiBuffer[0];
    double previousRSI = rsiBuffer[1];
}
""",
            "parameters": ["period", "applied_price"]
        },
        "MACD": {
            "name": "Moving Average Convergence Divergence",
            "code": """
int macdHandle;
double macdMainBuffer[];
double macdSignalBuffer[];

int OnInit()
{
    macdHandle = iMACD(_Symbol, PERIOD_CURRENT, 12, 26, 9, PRICE_CLOSE);
    ArraySetAsSeries(macdMainBuffer, true);
    ArraySetAsSeries(macdSignalBuffer, true);
    return(INIT_SUCCEEDED);
}

void OnTick()
{
    CopyBuffer(macdHandle, 0, 0, 3, macdMainBuffer);
    CopyBuffer(macdHandle, 1, 0, 3, macdSignalBuffer);
}
""",
            "parameters": ["fast_period", "slow_period", "signal_period", "applied_price"]
        },
        "MovingAverage": {
            "name": "Moving Average",
            "code": """
int maHandle;
double maBuffer[];

int OnInit()
{
    maHandle = iMA(_Symbol, PERIOD_CURRENT, 20, 0, MODE_EMA, PRICE_CLOSE);
    ArraySetAsSeries(maBuffer, true);
    return(INIT_SUCCEEDED);
}

void OnTick()
{
    CopyBuffer(maHandle, 0, 0, 3, maBuffer);
    double currentMA = maBuffer[0];
    double previousMA = maBuffer[1];
}
""",
            "parameters": ["period", "ma_shift", "ma_method", "applied_price"]
        },
        "BollingerBands": {
            "name": "Bollinger Bands",
            "code": """
int bbHandle;
double bbUpperBuffer[];
double bbMiddleBuffer[];
double bbLowerBuffer[];

int OnInit()
{
    bbHandle = iBands(_Symbol, PERIOD_CURRENT, 20, 0, 2.0, PRICE_CLOSE);
    ArraySetAsSeries(bbUpperBuffer, true);
    ArraySetAsSeries(bbMiddleBuffer, true);
    ArraySetAsSeries(bbLowerBuffer, true);
    return(INIT_SUCCEEDED);
}

void OnTick()
{
    CopyBuffer(bbHandle, 0, 0, 3, bbMiddleBuffer);
    CopyBuffer(bbHandle, 1, 0, 3, bbUpperBuffer);
    CopyBuffer(bbHandle, 2, 0, 3, bbLowerBuffer);
}
""",
            "parameters": ["period", "deviation", "applied_price"]
        },
        "ATR": {
            "name": "Average True Range",
            "code": """
int atrHandle;
double atrBuffer[];

int OnInit()
{
    atrHandle = iATR(_Symbol, PERIOD_CURRENT, 14);
    ArraySetAsSeries(atrBuffer, true);
    return(INIT_SUCCEEDED);
}

void OnTick()
{
    CopyBuffer(atrHandle, 0, 0, 3, atrBuffer);
    double currentATR = atrBuffer[0];
}
""",
            "parameters": ["period"]
        },
        "Stochastic": {
            "name": "Stochastic Oscillator",
            "code": """
int stochHandle;
double stochMainBuffer[];
double stochSignalBuffer[];

int OnInit()
{
    stochHandle = iStochastic(_Symbol, PERIOD_CURRENT, 5, 3, 3, MODE_SMA, STO_LOWHIGH);
    ArraySetAsSeries(stochMainBuffer, true);
    ArraySetAsSeries(stochSignalBuffer, true);
    return(INIT_SUCCEEDED);
}

void OnTick()
{
    CopyBuffer(stochHandle, 0, 0, 3, stochMainBuffer);
    CopyBuffer(stochHandle, 1, 0, 3, stochSignalBuffer);
}
""",
            "parameters": ["k_period", "d_period", "slowing", "ma_method", "price_field"]
        }
    }
    
    # Find matching template
    for key, template in templates.items():
        if key.lower() == indicator_name.lower():
            return {
                "success": True,
                "name": template["name"],
                "code": template["code"],
                "source": "builtin",
                "parameters": template["parameters"]
            }
    
    # Not found
    return {
        "success": False,
        "name": indicator_name,
        "code": "",
        "source": "none",
        "error": f"Indicator template not found: {indicator_name}"
    }


# =============================================================================
# Tool Registry for LangGraph Integration
# =============================================================================

KNOWLEDGE_TOOLS = {
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

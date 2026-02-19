#!/usr/bin/env python3
"""
PageIndex MCP Server for QuantMindX Knowledge Base
Provides reasoning-based retrieval over MQL5 articles, books, and logs via PageIndex services.

Architecture:
- PageIndex Articles (port 3000): MQL5 scraped articles
- PageIndex Books (port 3001): PDF books and documentation
- PageIndex Logs (port 3002): Trading logs and historical data
"""

import json
import asyncio
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import httpx

# Security check: Only run from QuantMindX directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
if PROJECT_ROOT.name != "QUANTMINDX":
    print(f"Error: Must be run from QuantMindX directory", file=sys.stderr)
    print(f"Current: {PROJECT_ROOT}", file=sys.stderr)
    sys.exit(1)

os.chdir(PROJECT_ROOT)

try:
    from mcp.server import Server
    from mcp.types import Tool, TextContent
except ImportError:
    print("MCP SDK not found. Install with: pip install mcp")
    exit(1)

# PageIndex Configuration
PAGEINDEX_URLS = {
    "articles": os.environ.get("PAGEINDEX_ARTICLES_URL", "http://localhost:3000"),
    "books": os.environ.get("PAGEINDEX_BOOKS_URL", "http://localhost:3001"),
    "logs": os.environ.get("PAGEINDEX_LOGS_URL", "http://localhost:3002"),
}

# HTTP client timeout
HTTP_TIMEOUT = httpx.Timeout(30.0)


async def pageindex_search(query: str, collection: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Search PageIndex service asynchronously.
    
    Args:
        query: Search query string
        collection: Collection name (articles, books, logs)
        limit: Maximum results to return
        
    Returns:
        List of search results
    """
    if collection not in PAGEINDEX_URLS:
        return []
    
    base_url = PAGEINDEX_URLS[collection]
    search_url = f"{base_url}/search"
    
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            response = await client.post(
                search_url,
                json={
                    "query": query,
                    "limit": limit,
                }
            )
            response.raise_for_status()
            data = response.json()
            
            # Format results with page/section references
            results = []
            for item in data.get("results", []):
                results.append({
                    "title": item.get("title", "Unknown"),
                    "content": item.get("content", ""),
                    "source": item.get("source", "unknown"),
                    "page": item.get("page"),
                    "section": item.get("section"),
                    "score": item.get("score", 0.0),
                    "file_path": item.get("file_path", ""),
                    "collection": collection,
                })
            return results
            
    except httpx.HTTPStatusError as e:
        print(f"PageIndex search failed with status {e.response.status_code}: {e}", file=sys.stderr)
        return []
    except httpx.RequestError as e:
        print(f"PageIndex request failed: {e}", file=sys.stderr)
        return []
    except Exception as e:
        print(f"Unexpected error during PageIndex search: {e}", file=sys.stderr)
        return []


async def pageindex_health(collection: str) -> Dict[str, Any]:
    """Check health of a PageIndex service."""
    if collection not in PAGEINDEX_URLS:
        return {"status": "unknown", "error": f"Unknown collection: {collection}"}
    
    base_url = PAGEINDEX_URLS[collection]
    health_url = f"{base_url}/health"
    
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
            response = await client.get(health_url)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


async def pageindex_stats(collection: str) -> Dict[str, Any]:
    """Get statistics from a PageIndex service."""
    if collection not in PAGEINDEX_URLS:
        return {"error": f"Unknown collection: {collection}"}
    
    base_url = PAGEINDEX_URLS[collection]
    stats_url = f"{base_url}/stats"
    
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            response = await client.get(stats_url)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        return {"error": str(e)}


# MCP Server
server = Server("quantmindx-kb")


@server.list_tools()
async def list_tools():
    """List available tools."""
    return [
        Tool(
            name="search_knowledge_base",
            description="""Search the QuantMindX knowledge base for articles, books, and logs about trading strategies, indicators, Expert Advisors, and MQL5 programming.
            
Uses PageIndex reasoning-based retrieval for better explainability with page/section references.

Collections:
- articles: MQL5 scraped articles from mql5.com
- books: PDF books and documentation
- logs: Trading logs and historical data""",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query"
                    },
                    "collection": {
                        "type": "string",
                        "description": "Collection to search: 'articles', 'books', 'logs', or 'all' (default: articles)",
                        "enum": ["articles", "books", "logs", "all"],
                        "default": "articles"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results per collection (default 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_article_content",
            description="Get the full content of a specific article by file path.",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Relative file path from search results"
                    }
                },
                "required": ["file_path"]
            }
        ),
        Tool(
            name="kb_stats",
            description="Get statistics about the knowledge base including PageIndex service health.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    """Handle tool calls."""
    
    if name == "search_knowledge_base":
        query = arguments.get("query", "")
        collection = arguments.get("collection", "articles")
        limit = arguments.get("limit", 5)
        
        if collection == "all":
            # Search all collections
            all_results = {}
            for coll in ["articles", "books", "logs"]:
                results = await pageindex_search(query, coll, limit)
                all_results[coll] = results
            
            return [TextContent(
                type="text",
                text=json.dumps({
                    "query": query,
                    "results_by_collection": all_results,
                    "total_results": sum(len(r) for r in all_results.values())
                }, indent=2)
            )]
        else:
            # Search specific collection
            results = await pageindex_search(query, collection, limit)
            
            return [TextContent(
                type="text",
                text=json.dumps({
                    "query": query,
                    "collection": collection,
                    "results": results,
                    "total_results": len(results)
                }, indent=2)
            )]
    
    elif name == "get_article_content":
        file_path = arguments.get("file_path", "")
        full_path = PROJECT_ROOT / "data" / "scraped_articles" / file_path

        if full_path.exists():
            content = full_path.read_text(encoding='utf-8')
            return [TextContent(type="text", text=content)]
        else:
            return [TextContent(type="text", text=f"Article not found: {file_path}")]
    
    elif name == "kb_stats":
        # Get health and stats from all PageIndex services
        stats = {
            "pageindex_services": {},
            "data_directories": {}
        }
        
        for collection in ["articles", "books", "logs"]:
            # Health check
            health = await pageindex_health(collection)
            # Stats
            collection_stats = await pageindex_stats(collection)
            
            stats["pageindex_services"][collection] = {
                "url": PAGEINDEX_URLS[collection],
                "health": health,
                "stats": collection_stats
            }
        
        # Check data directories
        data_dirs = {
            "scraped_articles": PROJECT_ROOT / "data" / "scraped_articles",
            "books": PROJECT_ROOT / "data" / "knowledge_base" / "books",
            "logs": PROJECT_ROOT / "data" / "logs",
        }
        
        for name, path in data_dirs.items():
            if path.exists():
                file_count = sum(1 for _ in path.rglob("*") if _.is_file())
                stats["data_directories"][name] = {
                    "path": str(path),
                    "exists": True,
                    "file_count": file_count
                }
            else:
                stats["data_directories"][name] = {
                    "path": str(path),
                    "exists": False
                }
        
        return [TextContent(type="text", text=json.dumps(stats, indent=2))]
    
    return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def main():
    """Run the MCP server."""
    from mcp.server.stdio import stdio_server
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
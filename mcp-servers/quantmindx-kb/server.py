#!/usr/bin/env python3
"""
Qdrant MCP Server for QuantMindX Knowledge Base
Provides semantic search over MQL5 articles via MCP protocol.
"""

import json
import asyncio
from pathlib import Path

try:
    from mcp.server import Server
    from mcp.types import Tool, TextContent
except ImportError:
    print("MCP SDK not found. Install with: pip install mcp")
    exit(1)

try:
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Missing dependencies. Install with:")
    print("  pip install qdrant-client sentence-transformers")
    exit(1)

# Configuration
QDRANT_PATH = Path("data/qdrant_db")
COLLECTION_NAME = "mql5_knowledge"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Initialize clients
client = QdrantClient(path=str(QDRANT_PATH))
model = SentenceTransformer(EMBEDDING_MODEL)

# MCP Server
server = Server("quantmindx-kb")

@server.list_tools()
async def list_tools():
    """List available tools."""
    return [
        Tool(
            name="search_knowledge_base",
            description="Search the MQL5 knowledge base for articles about trading strategies, indicators, Expert Advisors, and MQL5 programming.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results (default 5)",
                        "default": 5
                    },
                    "category_filter": {
                        "type": "string",
                        "description": "Optional category filter (e.g., 'Trading Systems', 'Integration')"
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
            description="Get statistics about the knowledge base.",
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
        limit = arguments.get("limit", 5)
        category_filter = arguments.get("category_filter")
        
        # Embed query
        query_vector = model.encode(query).tolist()
        
        # Search
        results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=limit * 2  # Get more, filter later
        )
        
        # Filter and dedupe by title
        seen_titles = set()
        filtered = []
        
        for r in results:
            title = r.payload.get('title', '')
            if title in seen_titles:
                continue
            
            if category_filter:
                if category_filter.lower() not in r.payload.get('categories', '').lower():
                    continue
            
            seen_titles.add(title)
            filtered.append({
                "title": r.payload.get('title', 'Unknown'),
                "url": r.payload.get('url', ''),
                "categories": r.payload.get('categories', ''),
                "relevance_score": r.payload.get('relevance_score', 0),
                "file_path": r.payload.get('file_path', ''),
                "preview": r.payload.get('text', '')[:300],
                "match_score": round(r.score, 3)
            })
            
            if len(filtered) >= limit:
                break
        
        return [TextContent(
            type="text",
            text=json.dumps({"results": filtered, "query": query}, indent=2)
        )]
    
    elif name == "get_article_content":
        file_path = arguments.get("file_path", "")
        full_path = Path("data/scraped_articles") / file_path
        
        if full_path.exists():
            content = full_path.read_text(encoding='utf-8')
            return [TextContent(type="text", text=content)]
        else:
            return [TextContent(type="text", text=f"Article not found: {file_path}")]
    
    elif name == "kb_stats":
        collection_info = client.get_collection(COLLECTION_NAME)
        
        stats = {
            "collection": COLLECTION_NAME,
            "total_vectors": collection_info.points_count,
            "embedding_model": EMBEDDING_MODEL,
            "storage_path": str(QDRANT_PATH)
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

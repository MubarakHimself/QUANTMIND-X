#!/usr/bin/env python3
"""
ChromaDB MCP Server for QuantMindX Knowledge Base
Provides semantic search over MQL5 articles via MCP protocol.
Only works when run from the QuantMindX directory.
"""

import json
import asyncio
import os
import sys
import re
from pathlib import Path
from typing import Any, Dict, List

# Security check: Only run from QuantMindX directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if PROJECT_ROOT.name != "QUANTMINDX":
    # Also check if we're already in QUANTMINDX directory
    cwd = Path.cwd()
    if cwd.name == "QUANTMINDX":
        PROJECT_ROOT = cwd
    else:
        print(f"Error: Must be run from QuantMindX directory", file=sys.stderr)
        print(f"Script location: {PROJECT_ROOT}", file=sys.stderr)
        print(f"Current directory: {cwd}", file=sys.stderr)
        sys.exit(1)

os.chdir(PROJECT_ROOT)

try:
    from mcp.server import Server  # type: ignore
    from mcp.types import Tool, TextContent  # type: ignore
except ImportError:
    print("MCP SDK not found. Install with: pip install mcp")
    exit(1)

# Type aliases for clarity
MCPResult = List[TextContent]
ToolList = List[Tool]

try:
    import chromadb
except ImportError:
    print("ChromaDB not found. Install with: pip install chromadb")
    exit(1)

# Configuration
CHROMA_PATH = PROJECT_ROOT / "data" / "chromadb"
COLLECTION_NAME = "mql5_knowledge"
SCRAPED_DIR = PROJECT_ROOT / "data" / "scraped_articles"

# Initialize ChromaDB client
client = chromadb.PersistentClient(path=str(CHROMA_PATH))
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)

def extract_title_from_frontmatter(doc: str) -> str:
    """Extract title from YAML frontmatter or markdown heading."""
    # Try YAML frontmatter first
    if doc.startswith('---'):
        lines = doc.split('\n')
        for line in lines[1:10]:  # Check first 10 lines
            if line.startswith('title:'):
                return line.split(':', 1)[1].strip()
            if line == '---':
                break

    # Try first # heading
    match = re.search(r'^#\s+(.+)$', doc, re.MULTILINE)
    if match:
        return match.group(1).strip()

    return "Untitled"

# MCP Server
server = Server("quantmindx-kb-chroma")

@server.list_tools()
async def list_tools() -> ToolList:
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
        ),
        Tool(
            name="list_categories",
            description="List all available categories in the knowledge base.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> MCPResult:
    """Handle tool calls."""

    if name == "search_knowledge_base":
        query = arguments.get("query", "")
        limit = arguments.get("limit", 5)
        category_filter = arguments.get("category_filter")

        # Search ChromaDB (built-in embeddings)
        results = collection.query(
            query_texts=[query],
            n_results=limit * 2  # Get more, filter later
        )

        # Process results
        filtered = []
        seen_titles = set()
        distances = results.get('distances', [[]])
        metadatas = results.get('metadatas', [[]])

        if results['documents'] and len(results['documents'][0]) > 0:
            for i, doc in enumerate(results['documents'][0]):
                metadata = metadatas[0][i] if metadatas and len(metadatas) > 0 and i < len(metadatas[0]) else {}
                # Extract title from frontmatter since metadata title is "---"
                title = extract_title_from_frontmatter(doc)
                distance = distances[0][i] if distances and len(distances[0]) > i else 0

                # Dedupe by title
                if title in seen_titles:
                    continue

                # Category filter
                if category_filter:
                    categories = str(metadata.get('categories', '') or '')
                    if category_filter.lower() not in categories.lower():
                        continue

                seen_titles.add(title)

                # Convert distance to relevance score (cosine distance -> similarity)
                relevance_score = 1 - distance if distance else 0

                # Get preview (skip frontmatter)
                preview_lines = doc.split('\n')
                preview_start = 0
                for j, line in enumerate(preview_lines):
                    if j > 0 and line == '---':
                        preview_start = j + 1
                        break
                preview = '\n'.join(preview_lines[preview_start:preview_start + 10]).strip()
                if len(preview) > 300:
                    preview = preview[:300] + "..."

                filtered.append({
                    "title": title,
                    "file_path": metadata.get('file_path', ''),
                    "categories": metadata.get('categories', ''),
                    "relevance_score": round(relevance_score, 3),
                    "preview": preview
                })

                if len(filtered) >= limit:
                    break

        return [TextContent(
            type="text",
            text=json.dumps({"results": filtered, "query": query}, indent=2)
        )]

    elif name == "get_article_content":
        file_path = arguments.get("file_path", "")
        full_path = SCRAPED_DIR / file_path

        if full_path.exists():
            content = full_path.read_text(encoding='utf-8')
            return [TextContent(type="text", text=content)]
        else:
            return [TextContent(type="text", text=f"Article not found: {file_path}")]

    elif name == "kb_stats":
        count = collection.count()

        stats = {
            "collection": COLLECTION_NAME,
            "total_articles": count,
            "storage_path": str(CHROMA_PATH),
            "embedding_function": "ChromaDB built-in (all-MiniLM-L6-v2)"
        }

        return [TextContent(type="text", text=json.dumps(stats, indent=2))]

    elif name == "list_categories":
        # Get unique categories from all metadata
        results = collection.get(include=["metadatas"])
        categories = set()

        if results['metadatas']:
            for meta in results['metadatas']:
                cat = meta.get('categories', '')
                if cat:
                    categories.add(cat)

        return [TextContent(type="text", text=json.dumps({
            "categories": sorted(list(categories))
        }, indent=2))]

    return [TextContent(type="text", text=f"Unknown tool: {name}")]

async def main():
    """Run the MCP server."""
    from mcp.server.stdio import stdio_server  # type: ignore

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())

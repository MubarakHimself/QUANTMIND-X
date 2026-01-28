#!/usr/bin/env python3
"""
Simple Qdrant MCP Server for QuantMindX Knowledge Base
Provides direct text search over MQL5 articles via MCP protocol.
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
except ImportError:
    print("Qdrant client not found. Install with: pip install qdrant-client")
    exit(1)

# Configuration
QDRANT_PATH = Path("data/qdrant_db")
COLLECTION_NAME = "mql5_knowledge"

# Initialize client
client = QdrantClient(path=str(QDRANT_PATH))

# MCP Server
server = Server("quantmindx-kb-simple")

@server.list_tools()
async def list_tools():
    """List available tools."""
    return [
        Tool(
            name="kb_stats",
            description="Get statistics about the knowledge base.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="list_collections",
            description="List all collections in the knowledge base.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    """Handle tool calls."""

    if name == "kb_stats":
        try:
            collection_info = client.get_collection(COLLECTION_NAME)

            stats = {
                "collection": COLLECTION_NAME,
                "total_vectors": collection_info.points_count,
                "storage_path": str(QDRANT_PATH)
            }

            return [TextContent(type="text", text=json.dumps(stats, indent=2))]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]

    elif name == "list_collections":
        try:
            collections = client.get_collections()
            return [TextContent(type="text", text=json.dumps({
                "collections": [c.name for c in collections.collections]
            }, indent=2))]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]

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

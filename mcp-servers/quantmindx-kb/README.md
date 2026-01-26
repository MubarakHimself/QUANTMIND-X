# QuantMindX Knowledge Base MCP Server

MCP server for semantic search over the QuantMindX knowledge base.

## Setup

### 1. Install Dependencies

```bash
pip install mcp qdrant-client sentence-transformers
```

### 2. Start Qdrant (Docker)

```bash
docker run -p 6333:6333 qdrant/qdrant
```

Or use local storage (default): `data/qdrant_db/`

### 3. Configure Claude Code MCP

Add to your Claude Code MCP configuration (usually `~/.config/claude/mcp_config.json`):

```json
{
  "mcpServers": {
    "quantmindx-kb": {
      "command": "python",
      "args": ["/home/mubarkahimself/Desktop/QUANTMINDX/mcp-servers/quantmindx-kb/server.py"],
      "env": {
        "PYTHONPATH": "/home/mubarkahimself/Desktop/QUANTMINDX"
      }
    }
  }
}
```

### 4. Index Content

```bash
# Scrape articles
python scripts/firecrawl_scraper.py

# Index to Qdrant
python scripts/index_to_qdrant.py
```

## Available Tools

| Tool | Description |
|------|-------------|
| `search_knowledge_base` | Semantic search with optional category filter |
| `get_article_content` | Get full content by file path |
| `kb_stats` | Get knowledge base statistics |

## Usage Example

```
User: "How do I implement a trailing stop in MQL5?"
Claude: [Uses search_knowledge_base] → finds relevant articles → provides code examples
```

## Configuration

- **Collection**: `mql5_knowledge`
- **Embedding Model**: `all-MiniLM-L6-v2`
- **Storage**: `data/qdrant_db/` (local) or Qdrant cloud

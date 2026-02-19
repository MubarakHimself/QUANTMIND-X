# QuantMindX Knowledge Base MCP Server (PageIndex)

MCP server for reasoning-based knowledge retrieval over the QuantMindX knowledge base using PageIndex services.

## Architecture

This server uses PageIndex for knowledge retrieval instead of traditional vector databases:
- **Articles Service** (port 3000): MQL5 articles and documentation
- **Books Service** (port 3001): Trading books and guides
- **Logs Service** (port 3002): System logs and trading history

PageIndex provides reasoning-based retrieval with page/section references for better explainability.

## Setup

### 1. Start PageIndex Services

```bash
# Start all PageIndex services
docker-compose -f docker-compose.pageindex.yml up -d

# Verify services are running
curl http://localhost:3000/health
curl http://localhost:3001/health
curl http://localhost:3002/health
```

### 2. Install Dependencies

```bash
pip install mcp httpx
```

### 3. Configure Environment Variables

Set the following environment variables (defaults are provided):

```bash
export PAGEINDEX_ARTICLES_URL="http://localhost:3000"
export PAGEINDEX_BOOKS_URL="http://localhost:3001"
export PAGEINDEX_LOGS_URL="http://localhost:3002"
```

### 4. Configure Claude Code MCP

Add to your Claude Code MCP configuration (usually `~/.config/claude/mcp_config.json`):

```json
{
  "mcpServers": {
    "quantmindx-kb": {
      "command": "python",
      "args": ["/home/mubarkahimself/Desktop/QUANTMINDX/mcp-servers/quantmindx-kb/server.py"],
      "env": {
        "PYTHONPATH": "/home/mubarkahimself/Desktop/QUANTMINDX",
        "PAGEINDEX_ARTICLES_URL": "http://localhost:3000",
        "PAGEINDEX_BOOKS_URL": "http://localhost:3001",
        "PAGEINDEX_LOGS_URL": "http://localhost:3002"
      }
    }
  }
}
```

### 5. Index Content

```bash
# Scrape articles
python scripts/firecrawl_scraper.py

# Index to PageIndex
python scripts/index_to_pageindex.py --collection articles --directory data/scraped_articles/
```

## Available Tools

| Tool | Description |
|------|-------------|
| `search_knowledge_base` | Search knowledge base with reasoning-based retrieval. Returns results with page/section references. |
| `get_article_content` | Get full content by file path (file-based retrieval) |
| `kb_stats` | Get knowledge base statistics from all PageIndex services |

## Usage Examples

### Search for MQL5 Information

```
User: "How do I implement a trailing stop in MQL5?"
Claude: [Uses search_knowledge_base with collection="articles"]
        Returns: Found in "Trailing Stop Implementation" (Page 3, Section 2)
                 Code example: ...
```

### Search Books

```
User: "What does the book say about risk management?"
Claude: [Uses search_knowledge_base with collection="books"]
        Returns: "Risk Management Strategies" (Page 45-52)
                 Key principles: ...
```

### Search Logs

```
User: "Find errors related to order execution"
Claude: [Uses search_knowledge_base with collection="logs"]
        Returns: Found 3 matching log entries with timestamps...
```

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `PAGEINDEX_ARTICLES_URL` | `http://localhost:3000` | URL for articles Pageindex service |
| `PAGEINDEX_BOOKS_URL` | `http://localhost:3001` | URL for books pageindex service |
| `PAGEINDEX_LOGS_URL` | `http://localhost:3002` | URL for logs pageindex service |

## Collections

| Collection | Service | Content |
|------------|---------|---------|
| `articles` | Port 3000 | MQL5 articles, documentation, tutorials |
| `books` | Port 3001 | Trading books, guides, references |
| `logs` | Port 3002 | System logs, trading history, errors |

## Benefits Over Vector Databases

1. **Reasoning-based Retrieval**: Returns results based on relevance reasoning, not just vector similarity
2. **Page/Section References**: Results include exact page and section references for verification
3. **No Embedding Overhead**: No need to generate embeddings for queries
4. **Faster Retrieval**: Direct text matching without vector operations
5. **Better Explainability**: Clear provenance for each result

## Troubleshooting

### Services Not Running

```bash
# Check service status
docker ps | grep pageindex

# Restart services
docker-compose -f docker-compose.pageindex.yml restart
```

### Connection Errors

Ensure PageIndex services are accessible:
```bash
curl http://localhost:3000/health
curl http://localhost:3001/health
curl http://localhost:3002/health
```

### Empty Search Results

Index content first:
```bash
python scripts/index_to_pageindex.py --collection articles --directory data/scraped_articles/
# QuantMindX Knowledge Base Setup Guide

## Overview

The QuantMindX knowledge base provides semantic search over trading articles, videos, and documentation using Qdrant vector database.

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Input Sources  │────▶│  Processing     │────▶│  Qdrant Vector  │
│  (URLs, Videos) │     │  (Embeddings)   │     │  Database       │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                           │
                                                           ▼
                                                    ┌─────────────────┐
                                                    │  MCP Server     │
                                                    │  (Search API)   │
                                                    └─────────────────┘
                                                           │
                                                           ▼
                                                    ┌─────────────────┐
                                                    │  Claude Code    │
                                                    │  (Query)        │
                                                    └─────────────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
cd /home/mubarkahimself/Desktop/QUANTMINDX
source venv/bin/activate
pip install -r requirements.txt
pip install mcp qdrant-client sentence-transformers
```

### 2. Start Qdrant

**Option A: Docker (Recommended)**
```bash
docker run -d -p 6333:6333 -v $(pwd)/data/qdrant:/qdrant/storage qdrant/qdrant
```

**Option B: Local Storage**
The MCP server uses local storage by default (`data/qdrant_db/`)

### 3. Configure Claude Code MCP

Edit `~/.config/claude/mcp_config.json`:

```json
{
  "mcpServers": {
    "quantmindx-kb": {
      "command": "python",
      "args": ["/home/mubarkahimself/Desktop/QUANTMINDX/mcp-servers/quantmindx-kb/server.py"],
      "env": {
        "PYTHONPATH": "/home/mubarkahimself/Desktop/QUANTMINDX",
        "QDRANT_URL": "http://localhost:6333"
      }
    }
  }
}
```

Restart Claude Code after editing.

### 4. Index Content

```bash
# Create input file
echo "https://www.mql5.com/en/articles/100" > data/inputs/articles.txt

# Scrape and index
python scripts/firecrawl_scraper.py --batch data/inputs/articles.txt
python scripts/index_to_qdrant.py --input data/scraped_articles/
```

## Usage

### From Claude Code

```
User: "Search for articles about trailing stop loss implementation"
Claude: [Calls search_knowledge_base] → Returns relevant articles with code examples
```

### Direct Python API

```python
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

client = QdrantClient(url="http://localhost:6333")
model = SentenceTransformer("all-MiniLM-L6-v2")

query = "trailing stop loss"
vector = model.encode(query).tolist()

results = client.search(
    collection_name="mql5_knowledge",
    query_vector=vector,
    limit=5
)
```

## Data Structure

### Article Document
```json
{
  "title": "Trailing Stop Implementation",
  "url": "https://www.mql5.com/en/articles/123",
  "categories": "Trading Systems, Risk Management",
  "text": "Full article content...",
  "file_path": "mql5/123-trailing-stop.md",
  "relevance_score": 0.95
}
```

### Video Clip (NPRD)
```json
{
  "video_id": "vid_12345",
  "clip_id": 1,
  "timestamp": "04:15-05:30",
  "speaker": "SPEAKER_02",
  "transcript": "...",
  "visual_description": "...",
  "ocr_content": ["..."]
}
```

## Maintenance

```bash
# Check knowledge base stats
python -c "from mcp_servers.quantmindx_kb.server import kb_stats; print(kb_stats())"

# Re-index all content
python scripts/index_to_qdrant.py --rebuild

# Backup Qdrant data
cp -r data/qdrant_db/ data/backups/qdrant_$(date +%Y%m%d)/
```

## Troubleshooting

**No search results:**
- Check Qdrant is running: `curl http://localhost:6333/`
- Verify collection exists: `python scripts/index_to_qdrant.py --stats`

**MCP connection error:**
- Check MCP config path: `~/.config/claude/mcp_config.json`
- Verify Python path in config
- Check server logs: `python mcp-servers/quantmindx-kb/server.py`

**Embedding errors:**
- Install sentence-transformers: `pip install sentence-transformers`
- First download may take time (model ~100MB)

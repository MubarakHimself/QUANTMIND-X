# QUANTMIND-X

An AI-powered quantitative trading platform with automated strategy extraction, coding, and deployment.

## Project Structure

```
QUANTMINDX/
├── scripts/                # Python scripts for scraping & processing
├── notebooks/              # Jupyter notebooks for exploration
├── tools/                  # NPRD multimodal video indexer
├── mcp-servers/            # Knowledge base MCP server
├── mcp-metatrader5-server/ # MT5 live trading integration
├── data/                   # Data storage (gitignored)
├── config/                 # Configuration files
└── requirements.txt        # Python dependencies
```

## Setup

### 1. Install Python Dependencies

```bash
cd /home/mubarkahimself/Desktop/QUANTMINDX
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Set Up API Keys

Create a `.env` file in the project root:

```
FIRECRAWL_API_KEY=your_api_key_here
QDRANT_URL=http://localhost:6333
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
```

Get your Firecrawl API key: https://www.firecrawl.dev/

### 3. Connect Knowledge Base (MCP)

Add to `~/.config/claude/mcp_config.json`:

```json
{
  "mcpServers": {
    "quantmindx-kb": {
      "command": "python",
      "args": ["mcp-servers/quantmindx-kb/server.py"]
    }
  }
}
```

See [docs/KNOWLEDGE_BASE_SETUP.md](docs/KNOWLEDGE_BASE_SETUP.md) for details.

## Workflow

1. **Input Sources**: Add article URLs or video links to `data/inputs/`
2. **Scrape**: Firecrawl extracts content from web articles
3. **Process**: NPRD processes videos with timeline extraction
4. **Store**: Content embedded and stored in Qdrant vector database
5. **Query**: Use MCP server or direct API to search knowledge base

## Components

### Knowledge Base Scraper
Extracts articles using Firecrawl, handles multi-part series automatically.

### NPRD (Non-Processing Research Data)
Multimodal video/audio indexer with:
- Timeline extraction and semantic block analysis
- Speaker diarization
- OCR content extraction from slides
- Gemini 1.5 Pro integration for video understanding

### Vector Database (Qdrant)
Stores categorized knowledge with semantic search capabilities.

### MCP Knowledge Base Server
Provides MCP interface for querying the knowledge base from Claude Code.

## Current Status

- [x] Firecrawl scraper
- [x] Qdrant vector database setup
- [x] MCP knowledge base server
- [x] NPRD video processing framework
- [ ] Trading bot coding agent
- [ ] Code review agent
- [ ] Live trading integration

## Knowledge Base

The knowledge base is accessible via:
- **Direct API**: Qdrant REST interface
- **MCP Server**: `mcp-servers/quantmindx-kb/`
- **Python Scripts**: `scripts/index_to_qdrant.py`, `scripts/generate_document_index.py`

Run `python scripts/index_to_qdrant.py --help` for indexing options.

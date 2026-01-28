# QUANTMIND-X

An AI-powered quantitative trading platform with automated strategy extraction, coding, and deployment.

## Quick Start

```bash
# One-time setup (installs everything)
./setup.sh

# Index articles to knowledge base
python3 scripts/index_to_qdrant.py

# Start using the knowledge base
./scripts/kb.sh status
```

## Project Structure

```
QUANTMINDX/
├── setup.sh               # One-time installation script
├── scripts/               # Python scripts for scraping & processing
│   ├── index_to_qdrant.py # Knowledge base indexer
│   └── kb.sh              # Knowledge base manager
├── tools/                 # NPRD multimodal video indexer
├── mcp-servers/           # Knowledge base MCP server
├── mcp-metatrader5-server/# MT5 live trading integration
├── data/                  # Data storage (gitignored)
└── requirements.txt       # Python dependencies
```

## Setup

### One-Time Setup (Recommended)

```bash
cd /home/mubarkahimself/Desktop/QUANTMINDX
./setup.sh
```

This installs:
- Python virtual environment
- All dependencies (qdrant-client, sentence-transformers, etc.)
- Embedding models
- Qdrant database
- MCP server

### Manual Setup

If you prefer manual setup:

```bash
# Create venv
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
cat > .env <<EOF
FIRECRAWL_API_KEY=your_api_key_here
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
EOF
```

## Knowledge Base

### Index Articles

```bash
# From QuantMindX directory
python3 scripts/index_to_qdrant.py
```

This processes all scraped articles and creates vector embeddings for semantic search.

### Knowledge Base Commands

```bash
./scripts/kb.sh status     # Check knowledge base stats
./scripts/kb.sh search "RSI"  # Search articles
./scripts/kb.sh start      # Start MCP server manually
```

### Auto-Start Hooks (Optional)

Enable automatic activation when entering the directory:

```bash
./scripts/setup-kb-hooks.sh
source ~/.bashrc  # or ~/.zshrc
```

## Workflow

1. **Input Sources**: Add article URLs to `data/inputs/`
2. **Scrape**: Firecrawl extracts content from web articles
3. **Process**: NPRD processes videos with timeline extraction
4. **Index**: Run `python3 scripts/index_to_qdrant.py`
5. **Query**: Use MCP tools or `./scripts/kb.sh search`

## Components

### Knowledge Base
- **Scraper**: Firecrawl integration for web articles
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Vector DB**: Qdrant for semantic search
- **MCP Server**: Interface for Claude Code integration

### NPRD (Non-Processing Research Data)
Multimodal video/audio indexer with:
- Timeline extraction and semantic block analysis
- Speaker diarization
- OCR content extraction from slides
- Gemini 1.5 Pro integration

### MetaTrader5 Integration
Live trading connectivity via MCP server.

## Current Status

- [x] One-time setup script
- [x] Firecrawl scraper
- [x] Qdrant vector database
- [x] MCP knowledge base server
- [x] NPRD video processing framework
- [ ] Trading bot coding agent
- [ ] Code review agent
- [ ] Live trading integration

## Documentation

- [docs/KNOWLEDGE_BASE_SETUP.md](docs/KNOWLEDGE_BASE_SETUP.md) - Detailed setup guide
- [docs/KNOWLEDGE_BASE_HOOKS.md](docs/KNOWLEDGE_BASE_HOOKS.md) - Auto-start hooks guide

# Analyst Agent CLI - Quick Start Guide

**Version:** 1.0 (Fast/MVP)
**Status:** Ready for Implementation

## Overview

Analyst Agent CLI converts unstructured trading content (NPRD video outputs, strategy documents) into structured Technical Requirements Document (TRD) markdown files using ChromaDB knowledge base cross-referencing and human-in-the-loop interaction.

## Key Features

- **NPRD Video Processing**: Reads JSON outputs from NPRD video analysis
- **Knowledge Base Integration**: Cross-references with ChromaDB knowledge base
- **TRD Generation**: Creates structured markdown TRD files
- **Human-in-the-Loop**: Interactive prompts for missing information
- **Auto Mode**: Optional autonomous generation with defaults

## Quick Start

### Prerequisites

1. Python 3.10+
2. ChromaDB installed at `data/chromadb/`
3. OpenAI or Anthropic API key

### Installation

```bash
# Navigate to project directory
cd /home/mubarkahimself/Desktop/QUANTMINDX

# Install dependencies
pip install -r requirements.txt

# Create filtered knowledge base collection
python scripts/filter_analyst_kb.py
```

### Basic Usage

```bash
# Interactive mode
python tools/analyst_cli.py

# Generate from NPRD output
python tools/analyst_cli.py generate --input outputs/videos/strategy_xyz.json

# Autonomous mode (use defaults)
python tools/analyst_cli.py generate --auto --input outputs/videos/strategy_xyz.json

# List available NPRD outputs
python tools/analyst_cli.py list

# Show knowledge base statistics
python tools/analyst_cli.py stats
```

## Command Reference

### generate

Generate TRD from input file.

```bash
analyst-cli generate [OPTIONS]

Options:
  -i, --input PATH      Input file path (NPRD JSON or strategy doc)
  -o, --output PATH     Output directory (default: docs/trds/)
  --auto                Autonomous mode (no interactive prompts)
  --kb-collection TEXT  ChromaDB collection name (default: analyst_kb)
  --type TEXT           Input type: nprd or strategy_doc (default: nprd)
```

### list

List available NPRD outputs.

```bash
analyst-cli list
```

### stats

Display knowledge base statistics.

```bash
analyst-cli stats
```

### complete

Complete missing fields in existing TRD.

```bash
analyst-cli complete --trd docs/trds/strategy.md
```

## Input Formats

### NPRD Video Output (JSON)

```json
{
  "video_id": "abc123",
  "video_title": "ORB Trading Strategy Explained",
  "video_url": "https://youtube.com/watch?v=abc123",
  "processed_at": "2026-01-26T14:30:00Z",
  "chunks": [
    {
      "chunk_id": 1,
      "start_time": "00:00",
      "end_time": "15:00",
      "transcript": "In this video, I'll explain the ORB strategy...",
      "visual_description": "Chart showing EURUSD...",
      "ocr_text": "RSI: 45.2, ATR: 0.0012",
      "keywords": ["ORB", "opening range", "breakout"]
    }
  ],
  "summary": {
    "full_transcript": "combined transcript...",
    "all_keywords": ["ORB", "breakout", "london session"],
    "visual_elements": ["charts", "indicators"]
  }
}
```

### Strategy Document (Markdown)

Any markdown file containing strategy notes can be used as input.

## Output Format

Generated TRD files are saved to `docs/trds/{strategy_name}_{timestamp}.md` with the following structure:

- YAML frontmatter (metadata)
- Overview section
- Entry Logic
- Exit Logic
- Filters (time, market conditions)
- Indicators & Settings
- Position Sizing & Risk Management
- Knowledge Base References
- Missing Information (if any)

## Knowledge Base

### Collections

- **Main Collection**: `mql5_knowledge` (1,805 MQL5 articles)
- **Filtered Collection**: `analyst_kb` (~330 articles)
  - Categories: Trading Systems, Trading, Expert Advisors, Indicators
  - Excludes: Machine Learning, Integration

### Creating Filtered Collection

```bash
# Run filter script
python scripts/filter_analyst_kb.py

# Output:
# Created analyst_kb collection
# Source: mql5_knowledge (1,805 articles)
# Filtered: ~330 articles
```

## Configuration

Create `.analyst_config.json` in project root:

```json
{
  "llm_provider": "openai",
  "llm_model": "gpt-4o",
  "temperature": 0.1,
  "kb_collection": "analyst_kb",
  "default_output_dir": "docs/trds",
  "auto_mode_defaults": {
    "position_sizing": "risk_based",
    "max_risk_per_trade": "0.5%",
    "default_sl": "ATR 1.5",
    "default_tp": "1.5x risk"
  }
}
```

Environment variables in `.env`:

```bash
# LLM Configuration
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# KB Settings
CHROMA_PATH=data/chromadb
ANALYST_COLLECTION=analyst_kb
```

## Interactive Mode Example

```bash
$ python tools/analyst_cli.py

Select input source:
[1] NPRD Video Output
[2] Strategy Document
[3] List available outputs
> 1

Available NPRD outputs:
[1] orb_strategy_20260126.json (2 hours ago)
[2] ict_concepts_20260125.json (1 day ago)
> 1

Loading: orb_strategy_20260126.json...
Parsed 4 chunks
Extracted concepts: entry=3, exit=2, filters=4
Searched KB: 12 relevant articles found

Missing information detected:
1. Position sizing method
   Enter value (or 'skip'): risk_based

2. Take profit calculation
   Enter value (or 'skip'): 1.5x risk

3. Stop loss placement
   Enter value (or 'skip'): ATR 1.5

Generating TRD...
Saved to: docs/trds/orb_breakout_scalper_20260126.md
```

## Troubleshooting

### ChromaDB Connection Issues

```bash
# Verify ChromaDB path
ls -la data/chromadb/

# Check collection exists
python -c "import chromadb; c = chromadb.PersistentClient('data/chromadb'); print(c.list_collections())"
```

### Empty KB Results

```bash
# Verify analyst_kb collection exists
python -c "import chromadb; c = chromadb.PersistentClient('data/chromadb'); col = c.get_collection('analyst_kb'); print(col.count())"

# Re-create filtered collection
python scripts/filter_analyst_kb.py
```

### LLM API Errors

```bash
# Check API key is set
echo $OPENAI_API_KEY

# Verify .env file exists
cat .env
```

### Missing Dependencies

```bash
# Re-install requirements
pip install -r requirements.txt --upgrade
```

## Project Structure

```
tools/analyst_agent/
├── cli/                     # CLI commands
├── graph/                   # LangGraph workflow
├── chains/                  # LangChain chains
├── kb/                      # ChromaDB client
├── prompts/                 # Prompt templates
└── utils/                   # Utilities
```

## Next Steps

1. Review generated TRD file
2. Complete missing fields if needed
3. Use TRD for MQL5 code generation (future)

## Documentation

- **Detailed Manual**: `docs/user_guides/analyst_cli_manual.md`
- **TRD**: `docs/trds/analyst_agent_cli_v1.md`
- **KB Integration**: `docs/KNOWLEDGE_BASES_AND_IDE.md`

## Notes

- Uses **ChromaDB** (NOT Qdrant)
- No autonomous trading - human approval required
- No risk management system in v1.0
- Follows Spec Driven Development (SDD) methodology

---

**Last Updated:** 2026-01-27
**Version:** 1.0

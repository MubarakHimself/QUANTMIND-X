# Analyst Agent CLI - User Manual

**Version:** 1.0 (Fast/MVP)
**Last Updated:** 2026-01-27

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Installation & Setup](#2-installation--setup)
3. [Knowledge Base Configuration](#3-knowledge-base-configuration)
4. [CLI Commands Reference](#4-cli-commands-reference)
5. [Input Formats](#5-input-formats)
6. [Output Structure](#6-output-structure)
7. [Workflow Overview](#7-workflow-overview)
8. [Configuration](#8-configuration)
9. [Usage Examples](#9-usage-examples)
10. [Troubleshooting](#10-troubleshooting)
11. [Best Practices](#11-best-practices)

---

## 1. Introduction

### 1.1 Purpose

The Analyst Agent CLI converts unstructured trading content into structured Technical Requirements Document (TRD) markdown files. It processes:

- **NPRD Video Outputs**: JSON files from video analysis
- **Strategy Documents**: Markdown files with trading strategy notes

### 1.2 Key Features

| Feature | Description |
|---------|-------------|
| **Knowledge Base Integration** | Cross-references with ChromaDB for relevant MQL5 articles |
| **Human-in-the-Loop** | Interactive prompts for missing information |
| **Auto Mode** | Autonomous generation with configurable defaults |
| **Structured Output** | TRD files ready for implementation |
| **ChromaDB Native** | Uses existing ChromaDB installation |

### 1.3 System Architecture

```
Input (NPRD/Strategy) -> Analyst Agent -> ChromaDB Search -> TRD Output
                                                      ^
                                                      |
                                               analyst_kb (filtered)
```

### 1.4 Technology Stack

| Component | Technology | Version |
|-----------|------------|---------|
| CLI Framework | Click | 8.1+ |
| Orchestration | LangChain | 0.2+ |
| Workflow | LangGraph | 0.2+ |
| Vector DB | ChromaDB | 0.4+ |
| LLM | OpenAI/Anthropic | - |

---

## 2. Installation & Setup

### 2.1 Prerequisites

```bash
# Check Python version (3.10+)
python --version

# Verify ChromaDB exists
ls -la data/chromadb/

# Check for API keys
echo $OPENAI_API_KEY
```

### 2.2 Install Dependencies

```bash
# Navigate to project
cd /home/mubarkahimself/Desktop/QUANTMINDX

# Install requirements
pip install -r requirements.txt

# Verify installation
python -c "import click, chromadb, langchain; print('Dependencies OK')"
```

### 2.3 Environment Setup

Create `.env` file in project root:

```bash
# LLM Configuration
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# LLM Settings
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o
TEMPERATURE=0.1

# KB Settings
CHROMA_PATH=data/chromadb
ANALYST_COLLECTION=analyst_kb

# Output Settings
TRD_OUTPUT_DIR=docs/trds
NPRD_INPUT_DIR=outputs/videos
```

Load environment variables:

```bash
# For Linux/Mac
source .env

# Or install python-dotenv (recommended)
pip install python-dotenv
```

### 2.4 Verify Installation

```bash
# Run help command
python tools/analyst_cli.py --help

# Expected output:
# Usage: analyst_cli.py [OPTIONS] COMMAND [ARGS]...
#   Analyst Agent CLI - Convert video/strategy content to TRD files.
#
# Options:
#   --help  Show this message and exit.
#
# Commands:
#   generate  Generate TRD from input
#   list      List available NPRD outputs
#   stats     Show knowledge base statistics
#   complete  Complete existing TRD
```

---

## 3. Knowledge Base Configuration

### 3.1 ChromaDB Structure

The system uses ChromaDB with two collections:

| Collection | Articles | Purpose |
|------------|----------|---------|
| `mql5_knowledge` | 1,805 | Main knowledge base |
| `analyst_kb` | ~330 | Filtered for Analyst Agent |

### 3.2 Create Filtered Collection

Run the filter script to create `analyst_kb`:

```bash
python scripts/filter_analyst_kb.py
```

**Expected Output:**

```
Created analyst_kb collection
Source: mql5_knowledge (1,805 articles)
Filtered: 330 articles
Filter criteria: Trading Systems, Trading, Expert Advisors, Indicators
Excluded: Machine Learning, Integration
```

### 3.3 Filter Criteria

The filtered collection includes:

- **Categories**: Trading Systems, Trading, Expert Advisors, Indicators
- **Excludes**: Machine Learning, Integration articles

### 3.4 Verify Collection

```bash
# Check collection count
python -c "
import chromadb
client = chromadb.PersistentClient('data/chromadb')
col = client.get_collection('analyst_kb')
print(f'Articles: {col.count()}')
"

# Test search
python -c "
import chromadb
client = chromadb.PersistentClient('data/chromadb')
col = client.get_collection('analyst_kb')
results = col.query(query_texts=['stop loss'], n_results=3)
print(f'Found {len(results[\"documents\"][0])} results')
"
```

### 3.5 Manual Collection Management

```bash
# List all collections
python -c "
import chromadb
client = chromadb.PersistentClient('data/chromadb')
print(client.list_collections())
"

# Delete and recreate analyst_kb
python -c "
import chromadb
client = chromadb.PersistentClient('data/chromadb')
client.delete_collection('analyst_kb')
print('Deleted analyst_kb')
"
```

---

## 4. CLI Commands Reference

### 4.1 Command Overview

| Command | Purpose | Options |
|---------|---------|---------|
| `generate` | Create TRD from input | `--input`, `--output`, `--auto`, `--type` |
| `list` | List available inputs | None |
| `stats` | Show KB statistics | None |
| `complete` | Fill missing TRD fields | `--trd` |

### 4.2 generate Command

**Syntax:**

```bash
python tools/analyst_cli.py generate [OPTIONS]
```

**Options:**

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--input` | `-i` | PATH | - | Input file path (required) |
| `--output` | `-o` | PATH | `docs/trds/` | Output directory |
| `--auto` | - | flag | False | Skip interactive prompts |
| `--kb-collection` | - | TEXT | `analyst_kb` | ChromaDB collection |
| `--type` | - | TEXT | `nprd` | Input type: `nprd` or `strategy_doc` |

**Examples:**

```bash
# Basic generation
python tools/analyst_cli.py generate --input outputs/videos/strategy.json

# Custom output
python tools/analyst_cli.py generate -i outputs/videos/strategy.json -o custom_trds/

# Auto mode
python tools/analyst_cli.py generate --auto --input outputs/videos/strategy.json

# Strategy document
python tools/analyst_cli.py generate --input strategy.md --type strategy_doc
```

### 4.3 list Command

**Syntax:**

```bash
python tools/analyst_cli.py list
```

**Output:**

```
Available NPRD outputs:
[1] orb_strategy_20260126.json (2 hours ago)
    Video: ORB Trading Strategy Explained
    Chunks: 4

[2] ict_concepts_20260125.json (1 day ago)
    Video: ICT Concepts
    Chunks: 7
```

### 4.4 stats Command

**Syntax:**

```bash
python tools/analyst_cli.py stats
```

**Output:**

```
Knowledge Base Statistics
==========================
Collection: analyst_kb
Total Articles: 330
Storage Path: data/chromadb
Embedding Model: ChromaDB built-in (all-MiniLM-L6-v2)
Distance Metric: Cosine
```

### 4.5 complete Command

**Syntax:**

```bash
python tools/analyst_cli.py complete --trd PATH
```

**Purpose:** Fill missing fields in existing TRD interactively.

---

## 5. Input Formats

### 5.1 NPRD Video Output (JSON)

**File Location:** `outputs/videos/{video_id}_{timestamp}.json`

**Structure:**

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
      "visual_description": "Chart showing EURUSD 1-minute timeframe...",
      "ocr_text": "RSI: 45.2, ATR: 0.0012",
      "keywords": ["ORB", "opening range", "breakout"]
    }
  ],
  "summary": {
    "full_transcript": "combined transcript...",
    "all_keywords": ["ORB", "breakout", "london session", "EURUSD"],
    "visual_elements": ["charts", "indicators", "trade examples"]
  }
}
```

**Required Fields:**

- `video_id`: Unique identifier
- `video_title`: Strategy name
- `chunks`: Array of video segments
  - `transcript`: Spoken content
  - `ocr_text`: Text from visuals
  - `keywords`: Key terms
- `summary.full_transcript`: Combined text

### 5.2 Strategy Document (Markdown)

Any markdown file can be used:

```markdown
# Trading Strategy: RSI Oversold

## Entry Conditions
- RSI < 30
- Price above EMA 20

## Exit Conditions
- RSI > 70
- Take profit: 1:2 risk-reward

## Filters
- London session only
- Avoid news events
```

### 5.3 Validation

```bash
# Validate NPRD format
python -c "
import json
import sys
with open('outputs/videos/strategy.json') as f:
    data = json.load(f)
    required = ['video_id', 'video_title', 'chunks', 'summary']
    missing = [r for r in required if r not in data]
    if missing:
        print(f'Missing fields: {missing}')
        sys.exit(1)
    print('Valid NPRD format')
"
```

---

## 6. Output Structure

### 6.1 TRD File Format

**Location:** `docs/trds/{strategy_name}_{timestamp}.md`

**Structure:**

```markdown
---
strategy_name: "ORB Breakout Scalper"
source: "NPRD: abc123 | ORB Trading Strategy Explained"
generated_at: "2026-01-26T14:30:00Z"
status: "draft"
version: "1.0"
---

# Trading Strategy: ORB Breakout Scalper

## Overview
[Brief strategy description]

## Entry Logic
### Primary Entry Trigger
- Entry condition 1
- Entry condition 2

### Entry Confirmation
- Confirmation signal

## Exit Logic
### Take Profit
- TP strategy

### Stop Loss
- SL strategy

## Filters
### Time Filters
- Trading session
- Time of day

### Market Condition Filters
- Volatility
- Spread

## Indicators & Settings
| Indicator | Settings | Purpose |
|-----------|----------|---------|
| RSI | Period: 14 | Entry signal |

## Position Sizing & Risk Management
### Risk Per Trade
- 0.5% of account

## Knowledge Base References
1. Article Title
   - URL: link
   - Relevance: description

## Missing Information (Requires Input)
- [ ] Field 1: description
```

### 6.2 Section Details

#### YAML Frontmatter

```yaml
strategy_name: "Strategy Name"
source: "Source description"
generated_at: "ISO 8601 timestamp"
status: "draft" | "complete"
version: "1.0"
```

#### Overview Section

Strategy description, source, and generation metadata.

#### Entry Logic

- Primary Entry Trigger
- Entry Confirmation
- Entry Example (pseudo-code)

#### Exit Logic

- Take Profit strategy
- Stop Loss placement
- Trailing Stop method
- Time Exit conditions

#### Filters

- **Time Filters**: Trading session, time of day, days to avoid
- **Market Condition Filters**: Volatility, spread, trend, news events

#### Indicators & Settings

Table format with indicator name, settings, and purpose.

#### Position Sizing & Risk Management

- Risk per trade
- Position sizing method
- Max drawdown limit

#### Knowledge Base References

List of relevant MQL5 articles found in ChromaDB.

#### Missing Information

Fields not found in input, requires user input.

---

## 7. Workflow Overview

### 7.1 LangGraph State Machine

```
START -> PARSE INPUT -> EXTRACT CONCEPTS -> SEARCH KB -> IDENTIFY GAPS
                                                          |
                                          +---------------+---------------+
                                          |                               |
                                    Has Gaps                        No Gaps
                                          |                               |
                                          v                               v
                                    ASK USER -> GENERATE TRD -> END     GENERATE TRD -> END
```

### 7.2 Processing Steps

| Step | Description | Output |
|------|-------------|--------|
| 1. Parse Input | Read NPRD JSON or strategy doc | Raw content |
| 2. Extract Concepts | LLM extracts trading concepts | Entry/exit/filters |
| 3. Search KB | Query ChromaDB for references | Relevant articles |
| 4. Identify Gaps | Detect missing information | Missing fields list |
| 5. Ask User | Interactive prompts (if gaps) | User answers |
| 6. Generate TRD | Create markdown file | TRD document |

### 7.3 State Management

The workflow maintains state through LangGraph:

```python
{
    "input_path": "path/to/input.json",
    "input_type": "nprd",
    "transcript": "full text content",
    "entry_conditions": ["condition1", "condition2"],
    "exit_conditions": ["tp", "sl"],
    "kb_references": [...],
    "missing_info": ["field1", "field2"],
    "user_answers": {},
    "trd_path": "output/path.md",
    "completed": false
}
```

### 7.4 Human-in-the-Loop

When gaps are detected:

```bash
Missing information detected:

1. Position sizing method
   Enter value (or 'skip'): risk_based

2. Take profit calculation
   Enter value (or 'skip'): 1.5x risk
```

**Responses:**
- Type value to provide input
- Type 'skip' to use default/TODO
- Press Enter for auto-mode default

---

## 8. Configuration

### 8.1 Configuration File

**Location:** `.analyst_config.json` (project root)

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
  },
  "search_settings": {
    "default_n_results": 5,
    "min_relevance_score": 0.6
  }
}
```

### 8.2 Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `llm_provider` | string | `openai` | LLM provider: `openai` or `anthropic` |
| `llm_model` | string | `gpt-4o` | Model name |
| `temperature` | float | `0.1` | LLM temperature (0-1) |
| `kb_collection` | string | `analyst_kb` | ChromaDB collection name |
| `default_output_dir` | string | `docs/trds` | TRD output directory |
| `auto_mode_defaults` | object | - | Default values for auto mode |

### 8.3 Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes* | - | OpenAI API key |
| `ANTHROPIC_API_KEY` | Yes* | - | Anthropic API key |
| `CHROMA_PATH` | No | `data/chromadb` | ChromaDB path |
| `ANALYST_COLLECTION` | No | `analyst_kb` | Collection name |

*Required depending on `llm_provider`

### 8.4 Precedence Order

1. Command-line flags (highest)
2. Environment variables
3. Configuration file
4. Default values (lowest)

---

## 9. Usage Examples

### 9.1 Basic Generation

```bash
# Generate from NPRD output
python tools/analyst_cli.py generate --input outputs/videos/orb_strategy.json

# Output:
# Loading: orb_strategy.json...
# Parsed 4 chunks
# Extracted concepts: entry=3, exit=2, filters=4
# Searched KB: 12 relevant articles found
# Generating TRD...
# Saved to: docs/trds/orb_breakout_scalper_20260126.md
```

### 9.2 Interactive Mode

```bash
python tools/analyst_cli.py

# Follow prompts:
# Select input source: [1] NPRD Video Output
# Select file: [1] orb_strategy.json
# Provide missing info interactively
```

### 9.3 Auto Mode

```bash
# Skip all prompts, use defaults
python tools/analyst_cli.py generate --auto --input outputs/videos/strategy.json

# No interactive prompts
# Missing fields marked as TODO
```

### 9.4 Custom Output Directory

```bash
python tools/analyst_cli.py generate \
  --input outputs/videos/strategy.json \
  --output custom_trds/
```

### 9.5 Strategy Document Input

```bash
# From markdown file
python tools/analyst_cli.py generate \
  --input my_strategy.md \
  --type strategy_doc
```

### 9.6 Complete Existing TRD

```bash
# Fill missing fields
python tools/analyst_cli.py complete --trd docs/trds/strategy.md
```

### 9.7 Batch Processing (Manual)

```bash
# Process all NPRD outputs
for file in outputs/videos/*.json; do
  python tools/analyst_cli.py generate --auto --input "$file"
done
```

---

## 10. Troubleshooting

### 10.1 Common Issues

#### ChromaDB Connection Failed

**Error:**
```
ConnectionError: Cannot connect to ChromaDB
```

**Solutions:**

```bash
# Verify path
ls -la data/chromadb/

# Check permissions
chmod -R 755 data/chromadb/

# Reinstall ChromaDB
pip install --upgrade chromadb
```

#### Collection Not Found

**Error:**
```
ValueError: Collection analyst_kb does not exist
```

**Solutions:**

```bash
# Create collection
python scripts/filter_analyst_kb.py

# Verify exists
python -c "
import chromadb
client = chromadb.PersistentClient('data/chromadb')
print('Collections:', client.list_collections())
"
```

#### Empty KB Search Results

**Symptoms:** KB references section is empty

**Solutions:**

```bash
# Check collection count
python -c "
import chromadb
col = chromadb.PersistentClient('data/chromadb').get_collection('analyst_kb')
print('Count:', col.count())
"

# Test search manually
python -c "
import chromadb
col = chromadb.PersistentClient('data/chromadb').get_collection('analyst_kb')
results = col.query(query_texts=['stop loss'], n_results=5)
print('Results:', len(results['documents'][0]))
"
```

#### LLM API Errors

**Error:**
```
AuthenticationError: Incorrect API key provided
```

**Solutions:**

```bash
# Check API key
echo $OPENAI_API_KEY

# Verify .env file
cat .env | grep API_KEY

# Test API key
python -c "
import openai
openai.api_key = 'your-key'
models = openai.Model.list()
print('OK')
"
```

#### Missing Dependencies

**Error:**
```
ModuleNotFoundError: No module named 'langchain'
```

**Solutions:**

```bash
# Reinstall all requirements
pip install -r requirements.txt --upgrade

# Verify specific package
python -c "import langchain; print(langchain.__version__)"
```

### 10.2 Debug Mode

Enable verbose logging:

```bash
# Set environment variable
export LOG_LEVEL=DEBUG

# Run with debug output
python tools/analyst_cli.py generate --input test.json
```

### 10.3 Log Files

Check for log files:

```bash
# Application logs
logs/analyst_cli.log

# Error logs
logs/errors.log
```

### 10.4 Recovery Procedures

```bash
# Reset ChromaDB collection
python -c "
import chromadb
client = chromadb.PersistentClient('data/chromadb')
client.delete_collection('analyst_kb')
print('Deleted analyst_kb')
"

# Recreate collection
python scripts/filter_analyst_kb.py

# Clear cache
rm -rf .cache/
```

---

## 11. Best Practices

### 11.1 Input Preparation

1. **Use complete NPRD outputs**: Ensure all chunks are present
2. **Include OCR text**: Visual indicators are important
3. **Verify JSON format**: Check structure before processing

### 11.2 Knowledge Base Management

1. **Keep analyst_kb updated**: Re-run filter after adding articles
2. **Monitor collection size**: ~330 articles is optimal
3. **Test search queries**: Verify relevant results appear

### 11.3 Workflow Optimization

1. **Use auto mode for batch processing**: Skip prompts when processing many files
2. **Review generated TRDs**: Always verify output quality
3. **Complete missing fields**: Use `complete` command to finish TRDs

### 11.4 Configuration

1. **Adjust temperature**: Lower (0.1) for deterministic output
2. **Choose appropriate model**: GPT-4o for quality, GPT-3.5 for speed
3. **Set reasonable defaults**: Auto mode defaults should be safe

### 11.5 Quality Assurance

**TRD Checklist:**

- [ ] All sections present
- [ ] Entry conditions clear
- [ ] Exit conditions defined
- [ ] Filters specified
- [ ] KB references relevant
- [ ] No TODO placeholders (or marked appropriately)

### 11.6 Performance Tips

```bash
# Use faster LLM for batch processing
export LLM_MODEL=gpt-3.5-turbo

# Increase search results for comprehensive analysis
python tools/analyst_cli.py generate --input strategy.json --kb-collection analyst_kb

# Process in parallel (manual)
for file in outputs/videos/*.json; do
  python tools/analyst_cli.py generate --auto --input "$file" &
done
wait
```

### 11.7 Version Control

```bash
# Track generated TRDs
git add docs/trds/*.md
git commit -m "Add generated TRDs"

# Track configuration
git add .analyst_config.json
git commit -m "Update analyst config"
```

---

## Appendix A: Quick Reference

### Common Commands

```bash
# Generate TRD
python tools/analyst_cli.py generate --input file.json

# List inputs
python tools/analyst_cli.py list

# Show stats
python tools/analyst_cli.py stats

# Complete TRD
python tools/analyst_cli.py complete --trd file.md
```

### Keyboard Shortcuts (Interactive Mode)

| Key | Action |
|-----|--------|
| Enter | Confirm selection |
| Ctrl+C | Cancel operation |
| Tab | Auto-complete (if available) |

### File Locations

| Type | Path |
|------|------|
| Inputs | `outputs/videos/*.json` |
| Outputs | `docs/trds/*.md` |
| Config | `.analyst_config.json` |
| ChromaDB | `data/chromadb/` |

---

## Appendix B: Support

### Documentation

- **TRD**: `docs/trds/analyst_agent_cli_v1.md`
- **Quick Start**: `tools/analyst_agent/README.md`
- **KB Setup**: `docs/KNOWLEDGE_BASE_SETUP.md`

### Getting Help

1. Check this manual first
2. Review TRD for technical details
3. Check ChromaDB documentation
4. Review LangGraph documentation

---

**Document Version:** 1.0
**Last Updated:** 2026-01-27
**Maintainer:** QuantMindX Project

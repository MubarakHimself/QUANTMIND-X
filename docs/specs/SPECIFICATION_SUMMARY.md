# Analyst Agent CLI - Specification Summary

**Version:** 1.0
**Date:** 2026-01-27
**Status:** Production Specification

## Quick Reference

### What is the Analyst Agent CLI?

A command-line tool that converts unstructured trading content from NPRD video outputs into structured Technical Requirements Document (TRD) files, enriched with relevant MQL5 articles from a filtered ChromaDB knowledge base.

### Key Features

1. **Multi-Source Input**
   - NPRD video output JSON files
   - Strategy markdown documents

2. **LLM-Powered Extraction**
   - Entry/exit conditions
   - Filters and indicators
   - Risk management parameters

3. **Knowledge Base Integration**
   - Semantic search against 330+ MQL5 articles
   - Relevant implementation patterns

4. **Gap Detection & Collection**
   - Interactive mode: Prompt user for missing info
   - Auto mode: Use defaults without prompts

5. **Structured Output**
   - TRD markdown files
   - YAML frontmatter
   - KB references included

### System Architecture

```
Input (NPRD/Doc) → Parse → Extract (LLM) → Search KB → Identify Gaps
                                                          ↓
                                              ┌────────────┴────────────┐
                                              │                         │
                                         Gaps? YES                  Gaps? NO
                                              │                         │
                                              ▼                         ▼
                                    Ask User / Use Defaults          Generate
                                              │                         │
                                              └────────────┬────────────┘
                                                           ▼
                                                     Generate TRD
                                                           │
                                                           ▼
                                                     Save to File
```

## Functional Requirements Summary

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-1 | Parse NPRD JSON outputs | P0 |
| FR-2 | Parse strategy documents | P1 |
| FR-3 | Extract entry/exit conditions | P0 |
| FR-4 | Extract filters and indicators | P0 |
| FR-5 | Search knowledge base | P0 |
| FR-6 | Identify missing information | P0 |
| FR-7 | Interactive data collection | P1 |
| FR-8 | Autonomous mode | P2 |
| FR-9 | Generate TRD markdown | P0 |
| FR-10 | List available inputs | P2 |
| FR-11 | Show KB statistics | P3 |

## CLI Interface

### Generate Command
```bash
# Interactive mode
python tools/analyst_cli.py generate

# Specific input
python tools/analyst_cli.py generate -i outputs/videos/abc123.json

# Auto mode (no prompts)
python tools/analyst_cli.py generate -i outputs/videos/abc123.json --auto

# Strategy document
python tools/analyst_cli.py generate -i strategy.md --type strategy_doc
```

### Other Commands
```bash
# List available inputs
python tools/analyst_cli.py list

# Show KB stats
python tools/analyst_cli.py stats

# Complete existing TRD
python tools/analyst_cli.py complete --trd docs/trds/strategy.md
```

## Data Flow

1. **Parse Input**
   - Read NPRD JSON or markdown file
   - Extract transcript, keywords, OCR text

2. **Extract Concepts** (LangChain + LLM)
   - Entry conditions
   - Exit conditions
   - Filters (time, volatility, spread)
   - Indicators with settings

3. **Search KB** (ChromaDB)
   - Generate semantic queries
   - Retrieve relevant MQL5 articles
   - Rank by relevance

4. **Identify Gaps**
   - Check against required fields
   - List missing information
   - Categorize by severity

5. **Collect Data**
   - Interactive: Prompt user
   - Auto: Use defaults from config

6. **Generate TRD**
   - Combine all data
   - Format as markdown
   - Save to `docs/trds/`

## Input Formats

### NPRD Output (JSON)
```json
{
  "video_id": "abc123",
  "video_title": "ORB Strategy Explained",
  "video_url": "https://youtube.com/watch?v=abc123",
  "processed_at": "2026-01-26T14:30:00Z",
  "chunks": [
    {
      "chunk_id": 1,
      "start_time": "00:00",
      "end_time": "15:00",
      "transcript": "In this video...",
      "visual_description": "Chart showing...",
      "ocr_text": "RSI: 45.2",
      "keywords": ["ORB", "breakout"]
    }
  ],
  "summary": {
    "full_transcript": "...",
    "all_keywords": ["ORB", "breakout"],
    "visual_elements": ["charts"]
  }
}
```

### Strategy Document (Markdown)
- Free-form text describing strategy
- MAY include sections and formatting
- Length: 100 - 50,000 characters

## Output Format

### TRD File Structure
```markdown
---
strategy_name: "orb_breakout_scalper"
source: "NPRD: abc123 | ORB Strategy Explained"
generated_at: "2026-01-26T14:30:00Z"
status: "draft"
version: "1.0"
---

# Trading Strategy: ORB Breakout Scalper

## Overview
[Brief description]

## Entry Logic
[Entry conditions, confirmations, examples]

## Exit Logic
[Take profit, stop loss, trailing]

## Filters
[Time filters, market conditions]

## Indicators & Settings
[Table of indicators]

## Position Sizing & Risk Management
[Risk per trade, position sizing]

## Knowledge Base References
[Relevant MQL5 articles]

## Missing Information
[List of gaps if any]

## Next Steps
[Review, complete, code, backtest]
```

## Technology Stack

| Component | Technology | Version |
|-----------|------------|---------|
| CLI Framework | Click | >=8.1.0 |
| Orchestration | LangGraph | >=0.2.0 |
| Chain Composition | LangChain | >=0.2.0 |
| Vector DB | ChromaDB | >=0.4.0 |
| LLM | OpenAI/Anthropic | - |
| Formatting | Rich | >=13.0.0 |
| Python | - | >=3.10 |

## Performance Targets

| Metric | Target |
|--------|--------|
| CLI startup | < 2 seconds |
| TRD generation | < 30 seconds |
| KB search | < 500ms |
| File parsing | < 3 seconds |
| Memory usage | < 500MB |

## Configuration Files

### .env (Required)
```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

### .analyst_config.json (Optional)
```json
{
  "llm": {
    "provider": "openai",
    "model": "gpt-4o",
    "temperature": 0.1
  },
  "knowledge_base": {
    "collection_name": "analyst_kb",
    "search_n_results": 5
  },
  "auto_mode": {
    "defaults": {
      "position_sizing": "risk_based",
      "max_risk_per_trade": "0.5%"
    }
  }
}
```

## Error Handling

| Code | Name | Recovery |
|------|------|----------|
| E001 | INPUT_FILE_NOT_FOUND | Abort |
| E003 | LLM_API_ERROR | Retry 3x |
| E005 | KB_NOT_ACCESSIBLE | Continue without KB |
| E009 | EXTRACTION_FAILED | Abort with message |

## Testing Requirements

- Unit tests: 80% coverage minimum
- Integration tests: E2E workflows
- Performance tests: All targets met
- Manual testing: Interactive mode

## Out of Scope (Future)

- v2.0: Risk management system
- v2.0: Dynamic position sizing
- v2.0: Prop firm challenge rules
- v3.0: IDE integration
- v3.0: Multi-strategy batch processing
- Autonomous trading (explicitly excluded)

## File Locations

```
tools/
├── analyst_cli.py          # Main CLI entry point
└── analyst_agent/          # Analyst Agent package
    ├── cli/                # CLI commands
    ├── graph/              # LangGraph workflow
    ├── chains/             # LangChain chains
    ├── kb/                 # Knowledge base client
    ├── prompts/            # Prompt templates
    └── utils/              # Utilities

docs/
├── specs/                  # Specification documents
│   └── analyst_agent_spec_v1.md
└── trds/                   # Generated TRD files
    └── {strategy}_{timestamp}.md

outputs/
└── videos/                 # NPRD outputs (input)
    └── {video}_{timestamp}.json

data/
└── chromadb/               # Knowledge base storage
    └── analyst_kb/         # Filtered collection
```

## Getting Started

1. **Setup**
   ```bash
   ./setup.sh
   ```

2. **Create KB Collection**
   ```bash
   python scripts/create_analyst_kb.py
   ```

3. **Generate TRD**
   ```bash
   python tools/analyst_cli.py generate -i outputs/videos/strategy.json
   ```

4. **Review Output**
   ```bash
   cat docs/trds/strategy_20260126.md
   ```

## Key Dependencies

```
click>=8.1.0
langchain>=0.2.0
langgraph>=0.2.0
chromadb>=0.4.0
openai>=1.0.0
anthropic>=0.18.0
rich>=13.0.0
pydantic>=2.0.0
```

## Acceptance Criteria

- [ ] Parses NPRD JSON and strategy docs
- [ ] Extracts trading concepts with LLM
- [ ] Searches knowledge base for references
- [ ] Identifies missing information
- [ ] Prompts for gaps (interactive mode)
- [ ] Uses defaults (auto mode)
- [ ] Generates valid TRD markdown
- [ ] Meets all performance targets
- [ ] 80%+ test coverage
- [ ] PEP 8 compliant code

## Support

- Full specification: `docs/specs/analyst_agent_spec_v1.md`
- Technical Requirements: `docs/trds/analyst_agent_cli_v1.md`
- Issues: GitHub Issues

---

**Version:** 1.0
**Last Updated:** 2026-01-27
**Author:** Specification Writer Agent

# Technical Requirements Document: Analyst Agent CLI v1.0

> **Project:** QuantMindX
> **Component:** Analyst Agent CLI
> **Version:** 1.0 (Fast/MVP)
> **Status:** Ready for Implementation
> **Generated:** 2026-01-26
> **Last Updated:** 2026-01-27 (Enhanced for SDD Implementation)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Overview](#2-system-overview)
3. [Architecture](#3-architecture)
4. [Components](#4-components)
5. [Data Structures](#5-data-structures)
6. [Knowledge Base Integration](#6-knowledge-base-integration)
7. [CLI Interface Specification](#7-cli-interface-specification)
8. [LangGraph Workflow](#8-langgraph-workflow)
9. [File Organization](#9-file-organization)
10. [Dependencies](#10-dependencies)
11. [Testing Strategy](#11-testing-strategy)
12. [Manual Testing Procedures](#12-manual-testing-procedures)
13. [Implementation Phases](#13-implementation-phases)
14. [Acceptance Criteria](#14-acceptance-criteria)
15. [Appendix: Project Context](#15-appendix-project-context)

---

## 1. Executive Summary

### Purpose

Build a fast CLI tool that:
1. Reads NPRD video outputs (unstructured trading content)
2. Cross-references with a filtered knowledge base (Qdrant, NOT ChromaDB)
3. Generates structured Markdown TRD (Technical Requirements Document) files
4. Supports human-in-the-loop interaction for missing information

### Key Constraints

- **NO autonomous trading** - Human approval required at every stage
- **NO risk management system yet** - Will be added in v2.0
- **Simple extraction first** - Don't interpret, just structure
- **CLI only** - IDE integration is future scope
- **Use ChromaDB** - Existing installation in `data/chromadb/`

### Target User

Mubarak (project owner) who:
- Has NPRD outputs in `outputs/videos/`
- Has ChromaDB with 1,805 MQL5 articles in `data/chromadb/`
- Wants to convert video content into TRD files
- Wants human-in-the-loop questions during generation

### Vector Database Setup

**Vector DB:** ChromaDB (existing installation in `data/chromadb/`)
- **Existing Collection:** `mql5_knowledge` (1,805 articles)
- **Filtered Collection:** `analyst_kb` (~330 articles) - create via filter script
- **No Migration Required:** Use existing ChromaDB installation

**Filter Script:** See `scripts/filter_analyst_kb.py` (to be created)

---

## 2. System Overview

### Data Flow

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  NPRD Output    │────►│  Analyst Agent  │────►│  TRD Markdown   │
│  (JSON/MD)      │     │  CLI            │     │  Files          │
└─────────────────┘     └────────┬────────┘     └─────────────────┘
                                 │
                                 │ Semantic Search
                                 ▼
                        ┌─────────────────┐
                        │  ChromaDB       │
                        │  analyst_kb     │
                        │  (~330 articles)│
                        └─────────────────┘
```

### Input Sources

| Source | Format | Location | Content |
|--------|--------|----------|---------|
| NPRD Video Output | JSON | `outputs/videos/*.json` | Transcript, OCR, keywords |
| Strategy Document | MD/PDF | User-provided | Compiled strategy notes |

### Output Format

**TRD Markdown File:**
- Location: `docs/trds/{strategy_name}_{timestamp}.md`
- Structure: YAML frontmatter + sections
- Template: Specified in [Section 5.2](#52-trd-template)

---

## 3. Architecture

### Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **CLI Framework** | Click | Command-line interface |
| **Orchestration** | LangChain | Chain composition |
| **Workflow** | LangGraph | State machine for HITL |
| **Vector DB** | ChromaDB | Knowledge base queries |
| **LLM** | OpenAI/Anthropic | Text generation |
| **Config** | YAML | Settings persistence |

### System Components

```
analyst_agent/
├── cli.py                 # Main CLI entry point (Click)
├── graph/
│   ├── __init__.py
│   ├── workflow.py        # LangGraph StateGraph
│   ├── nodes.py           # Individual nodes (extract, search, etc.)
│   └── state.py           # TypedDict for state
├── chains/
│   ├── __init__.py
│   ├── extraction.py      # LangChain for content extraction
│   ├── search.py          # KB search chain
│   └── generation.py      # TRD generation chain
├── kb/
│   ├── __init__.py
│   └── client.py          # ChromaDB client wrapper
├── prompts/
│   └── templates.py       # All prompt templates
└── utils/
    ├── file_io.py         # NPRD output parsing
    └── trd_template.py    # TRD markdown generator
```

### LangGraph State Machine

```
                    ┌─────────────┐
                    │   START     │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  PARSE      │  Parse NPRD/strategy doc
                    │  INPUT      │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  EXTRACT    │  Extract trading concepts
                    │  CONCEPTS   │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  SEARCH KB  │  Query ChromaDB for refs
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
              ┌─────►IDENTIFY     │
              │     │GAPS         │
              │     └──────┬──────┘
              │            │
              │    ┌───────┴───────┐
              │    │               │
              │    ▼               ▼
              │ Missing      No Missing
              │    │               │
              │    ▼               ▼
              │ ┌─────────┐  ┌─────────────┐
              │ │  ASK    │  │  GENERATE   │
              │ │  USER   │  │  TRD        │
              │ └────┬────┘  └──────┬──────┘
              │      │              │
              │      └──────┬───────┘
              │             ▼
              │      ┌─────────────┐
              │      │  UPDATE     │
              │      │  STATE      │
              │      └──────┬──────┘
              │             │
              └─────────────┘
                          │
                          ▼
                    ┌─────────────┐
                    │     END     │
                    └─────────────┘
```

---

## 4. Components

### 4.1 CLI Module (`cli.py`)

**Purpose:** Entry point, user interaction, command routing

**Commands:**
```bash
# Interactive mode (default)
python tools/analyst_cli.py

# Specify input file
python tools/analyst_cli.py --input outputs/videos/strategy_xyz.json

# Autonomous mode (skip questions, use defaults)
python tools/analyst_cli.py --auto --input outputs/videos/strategy_xyz.json

# List available NPRD outputs
python tools/analyst_cli.py --list

# Show KB stats
python tools/analyst_cli.py --stats
```

**Click Groups:**
```python
@click.group()
def cli():
    """Analyst Agent CLI - Convert video/strategy content to TRD files."""

@cli.command()
@click.option('--input', '-i', type=Path, help='Input file path')
@click.option('--auto', is_flag=True, help='Autonomous mode (no questions)')
@click.option('--output', '-o', type=Path, help='Output directory')
def generate(input, auto, output):
    """Generate TRD from input."""

@cli.command()
def list():
    """List available NPRD outputs."""

@cli.command()
def stats():
    """Show knowledge base statistics."""
```

### 4.2 LangGraph Workflow (`graph/workflow.py`)

**Purpose:** State machine orchestrating the TRD generation pipeline

**State Definition:**
```python
from typing import TypedDict, List, Optional

class AnalystState(TypedDict):
    # Input
    input_path: str
    input_type: str  # "nprd" | "strategy_doc"

    # Parsed content
    raw_content: dict
    transcript: str
    ocr_text: str
    keywords: List[str]

    # Extracted concepts
    entry_conditions: List[str]
    exit_conditions: List[str]
    filters: List[str]
    time_filters: List[str]

    # KB search results
    kb_references: List[dict]
    relevant_articles: List[str]

    # Identified gaps
    missing_info: List[str]
    user_answers: dict

    # Output
    trd_content: str
    trd_path: str

    # Flow control
    current_step: str
    needs_user_input: bool
    completed: bool
```

**Graph Structure:**
```python
from langgraph.graph import StateGraph, END

def create_analyst_graph() -> StateGraph:
    workflow = StateGraph(AnalystState)

    # Add nodes
    workflow.add_node("parse_input", parse_input_node)
    workflow.add_node("extract_concepts", extract_concepts_node)
    workflow.add_node("search_knowledge_base", search_kb_node)
    workflow.add_node("identify_gaps", identify_gaps_node)
    workflow.add_node("ask_user", ask_user_node)
    workflow.add_node("generate_trd", generate_trd_node)

    # Define edges
    workflow.set_entry_point("parse_input")
    workflow.add_edge("parse_input", "extract_concepts")
    workflow.add_edge("extract_concepts", "search_knowledge_base")
    workflow.add_edge("search_knowledge_base", "identify_gaps")

    # Conditional: gaps or no gaps
    workflow.add_conditional_edges(
        "identify_gaps",
        has_gaps,
        {
            "ask_user": "ask_user",
            "generate": "generate_trd"
        }
    )

    workflow.add_edge("ask_user", "generate_trd")
    workflow.add_edge("generate_trd", END)

    return workflow.compile()
```

### 4.3 Extraction Chain (`chains/extraction.py`)

**Purpose:** Extract structured trading concepts from unstructured text

**LangChain Pipeline:**
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

EXTRACTION_PROMPT = """You are a trading strategy analyst. Extract structured information from the following content.

Content:
{content}

Extract and return ONLY a JSON with this structure:
{{
    "strategy_name": "string",
    "overview": "brief description",
    "entry_conditions": ["condition 1", "condition 2"],
    "exit_conditions": ["take profit", "stop loss", "trailing"],
    "filters": ["session", "volatility", "spread"],
    "time_filters": ["london session", "avoid news"],
    "indicators_used": ["RSI", "MACD", "ATR"],
    "mentioned_concepts": ["support/resistance", "order flow"]
}}

Return ONLY valid JSON, no markdown."""

extraction_chain = ChatPromptTemplate.from_template(EXTRACTION_PROMPT) | llm | JsonOutputParser()
```

### 4.4 KB Search Chain (`chains/search.py`)

**Purpose:** Query ChromaDB for relevant articles

**Search Strategy:**
1. Extract key concepts from extracted data
2. Generate 3-5 search queries
3. Query ChromaDB `analyst_kb` collection
4. Return top 5 articles per query

```python
def search_knowledge_base(state: AnalystState) -> AnalystState:
    """Search ChromaDB for relevant articles."""
    client = ChromaClient(collection="analyst_kb")

    # Generate search queries from concepts
    queries = generate_search_queries(state)

    # Search and collect results
    all_results = []
    for query in queries:
        results = client.search(query, n_results=5)
        all_results.extend(results)

    # Deduplicate by title
    seen = set()
    unique_results = []
    for r in all_results:
        if r['title'] not in seen:
            seen.add(r['title'])
            unique_results.append(r)

    state['kb_references'] = unique_results[:10]  # Top 10
    return state
```

### 4.5 Gap Identification (`nodes.py`)

**Purpose:** Identify missing information needed for complete TRD

**Required Fields:**
```python
REQUIRED_TRD_FIELDS = {
    "entry_logic": ["entry trigger", "entry confirmation"],
    "exit_logic": ["take profit", "stop loss"],
    "risk_management": ["position sizing", "max risk"],
    "filters": ["time filters", "market condition filters"],
    "indicators": ["indicator settings", "timeframes"],
}

def identify_gaps(state: AnalystState) -> AnalystState:
    """Identify missing information."""
    extracted = {k: v for k, v in state.items() if v}
    missing = []

    for field, subfields in REQUIRED_TRD_FIELDS.items():
        for subfield in subfields:
            if subfield not in str(extracted):
                missing.append(f"{field}: {subfield}")

    state['missing_info'] = missing
    state['needs_user_input'] = len(missing) > 0
    return state
```

---

## 5. Data Structures

### 5.1 NPRD Output Format (Input)

**File:** `outputs/videos/{video_id}_{timestamp}.json`

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

### 5.2 TRD Template (Output)

**File:** `docs/trds/{strategy_name}_{timestamp}.md`

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

[Brief strategy description from video/extracted content]

**Source:** ORB Trading Strategy Explained (YouTube)
**Analyst:** Analyst Agent v1.0
**Generated:** 2026-01-26

---

## Entry Logic

### Primary Entry Trigger

- [Extracted entry condition 1]
- [Extracted entry condition 2]

### Entry Confirmation

- [Confirmation signal if mentioned]
- [Timeframe: e.g., 1-minute, 5-minute]

### Entry Example

```
IF [condition 1] AND [condition 2]
AND time is within [session]
THEN enter [LONG/SHORT] at [price level]
```

---

## Exit Logic

### Take Profit

- [TP strategy: fixed pips, percentage, resistance]
- [TP value if specified]

### Stop Loss

- [SL strategy: fixed pips, ATR-based, support]
- [SL value if specified]

### Trailing Stop

- [Trailing method if mentioned]
- [Trail distance]

### Time Exit

- [Close at specific time if mentioned]

---

## Filters

### Time Filters

- **Trading Session:** [London, NY, Asian, overlap]
- **Time of Day:** [e.g., 8am-12pm GMT]
- **Days to Avoid:** [Friday, news days, etc.]

### Market Condition Filters

- **Volatility:** [ATR threshold, avoid low volatility]
- **Spread:** [Max spread allowed]
- **Trend:** [Trend following or range]
- **News Events:** [Avoid high-impact news]

---

## Indicators & Settings

| Indicator | Settings | Purpose |
|-----------|----------|---------|
| [RSI] | [Period: 14] | [Entry/exit signal] |
| [ATR] | [Period: 14] | [Stop loss calculation] |
| [MA] | [Period: 20, EMA] | [Trend filter] |

---

## Position Sizing & Risk Management

> **Note:** This section will be enhanced in v2.0 with dynamic risk management system.

### Risk Per Trade

- [Mentioned: e.g., 0.5% of account]
- [Or: TODO - specify risk amount]

### Position Sizing

- [Method: fixed lots, risk-based, volatility-based]
- [Or: TODO - implement position sizing]

### Max Drawdown Limit

- [Daily limit if mentioned]
- [Or: TODO - specify drawdown limit]

---

## Knowledge Base References

### Relevant Articles Found

1. **[Article Title 1]**
   - URL: [link from KB]
   - Relevance: [why it matters]
   - Key Insight: [specific implementation detail]

2. **[Article Title 2]**
   - URL: [link from KB]
   - Relevance: [implementation pattern]

### Implementation Notes

- [Technical considerations from KB articles]
- [Potential pitfalls mentioned in literature]
- [Suggested MQL5 patterns]

---

## Missing Information (Requires Input)

The following information was not found in the source and needs user input:

- [ ] [Field 1]: [description]
- [ ] [Field 2]: [description]

**To complete this TRD, run:**
```bash
python tools/analyst_cli.py --trd docs/trds/{this_file}.md --complete
```

---

## Next Steps

1. **Review this TRD** - Verify all sections are accurate
2. **Complete missing fields** - Provide input for TODO items
3. **Generate MQL5 code** - Use QuantCode CLI (future)
4. **Backtest** - Validate strategy performance

---

**Generated by:** QuantMindX Analyst Agent v1.0
**Knowledge Base:** analyst_kb (~330 articles)
**LangGraph Version:** 0.2.x
```

---

## 6. Knowledge Base Integration

### 6.1 Filtered Collection Creation

**New Collection:** `analyst_kb`

**Filter Criteria:**
```python
ACCEPTED_CATEGORIES = [
    "Trading Systems",
    "Trading",
    "Expert Advisors",
    "Indicators"
]

REJECTED_COMBINATIONS = [
    "Integration",
    "Machine Learning"
]
```

**Creation Script:** `scripts/create_analyst_kb.py`

```python
#!/usr/bin/env python3
"""
Create filtered ChromaDB collection for Analyst Agent.
Selects only relevant articles from main KB.
"""

import chromadb
from pathlib import Path
import json

# Configuration
MAIN_COLLECTION = "mql5_knowledge"
ANALYST_COLLECTION = "analyst_kb"
SCRAPED_DIR = Path("data/scraped_articles")

# Filter criteria
ACCEPTED_PATTERNS = [
    "Trading Systems",
    "Trading",
    "Expert Advisors",
    "Indicators"
]

def should_include(article_meta: dict) -> bool:
    """Check if article should be in analyst KB."""
    categories = article_meta.get('categories', '')

    # Reject if has ML or Integration
    if 'Machine Learning' in categories or 'Integration' in categories:
        return False

    # Accept if matches target categories
    for pattern in ACCEPTED_PATTERNS:
        if pattern in categories:
            return True

    return False

def create_analyst_kb():
    """Create filtered collection."""
    client = chromadb.PersistentClient(path=str(Path("data/chromadb")))

    # Get main collection
    main_col = client.get_collection(MAIN_COLLECTION)

    # Create analyst collection
    analyst_col = client.get_or_create_collection(
        name=ANALYST_COLLECTION,
        metadata={"hnsw:space": "cosine", "purpose": "analyst_agent"}
    )

    # Get all articles from main collection
    results = main_col.get(include=['metadatas', 'documents'])

    # Filter and add to analyst collection
    filtered_ids = []
    filtered_docs = []
    filtered_metas = []

    for i, (meta, doc) in enumerate(zip(results['metadatas'], results['documents'])):
        if should_include(meta):
            filtered_ids.append(f"analyst_{i}")
            filtered_docs.append(doc)
            filtered_metas.append(meta)

    # Batch add
    analyst_col.add(
        ids=filtered_ids,
        documents=filtered_docs,
        metadatas=filtered_metas
    )

    print(f"✅ Created analyst_kb collection")
    print(f"   Source: {MAIN_COLLECTION} ({len(results['documents'])} articles)")
    print(f"   Filtered: {len(filtered_docs)} articles")
    print(f"   Filtered by: {ACCEPTED_PATTERNS}")
    print(f"   Excluded: Machine Learning, Integration")

if __name__ == "__main__":
    create_analyst_kb()
```

### 6.2 KB Client Wrapper

**File:** `analyst_agent/kb/client.py`

```python
"""ChromaDB client wrapper for Analyst Agent."""

import chromadb
from pathlib import Path
from typing import List, Dict, Optional

class AnalystKBClient:
    """Client for querying analyst_kb collection."""

    def __init__(self, chroma_path: str = None):
        self.chroma_path = chroma_path or str(Path("data/chromadb"))
        self.client = chromadb.PersistentClient(path=self.chroma_path)
        self.collection = self.client.get_collection("analyst_kb")

    def search(
        self,
        query: str,
        n_results: int = 5,
        category_filter: Optional[str] = None
    ) -> List[Dict]:
        """Search knowledge base."""

        results = self.collection.query(
            query_texts=[query],
            n_results=n_results * 2  # Get more for filtering
        )

        # Process results
        formatted = []
        seen = set()

        if results['documents'] and len(results['documents'][0]) > 0:
            for i, doc in enumerate(results['documents'][0]):
                meta = results['metadatas'][0][i]

                # Extract title
                title = self._extract_title(doc)
                if title in seen:
                    continue
                seen.add(title)

                # Category filter
                if category_filter:
                    if category_filter.lower() not in meta.get('categories', '').lower():
                        continue

                formatted.append({
                    'title': title,
                    'file_path': meta.get('file_path', ''),
                    'categories': meta.get('categories', ''),
                    'content': doc[:500],  # Preview
                    'distance': results['distances'][0][i] if 'distances' in results else 0
                })

                if len(formatted) >= n_results:
                    break

        return formatted

    def get_stats(self) -> Dict:
        """Get collection statistics."""
        count = self.collection.count()
        return {
            'collection': 'analyst_kb',
            'total_articles': count,
            'storage_path': self.chroma_path
        }

    def _extract_title(self, doc: str) -> str:
        """Extract title from YAML frontmatter."""
        if doc.startswith('---'):
            lines = doc.split('\n')
            for line in lines[1:10]:
                if line.startswith('title:'):
                    return line.split(':', 1)[1].strip()
                if line == '---':
                    break
        return "Untitled"
```

---

## 7. CLI Interface Specification

### 7.1 Interactive Mode

```bash
$ python tools/analyst_cli.py

╔══════════════════════════════════════════════════════════════════╗
║          ____ _  _ _  _    ___  ____    _  _ _  _ _    _ ___     ║
║         |__| |\/| |_\|    |__] |__|    |_\| |_\| |    |  |      ║
║         |  | |  | | |      |__| |  |    | | | | | |    |  |      ║
║                                                                  ║
║                    Analyst Agent CLI v1.0                         ║
╚══════════════════════════════════════════════════════════════════╝

Select input source:

[1] NPRD Video Output
[2] Strategy Document
[3] List available outputs

> 1

Available NPRD outputs:

[1] orb_strategy_20260126.json (2 hours ago)
[2] ict_concepts_20260125.json (1 day ago)
[3] scalping_setup_20260124.json (2 days ago)

> 1

Loading: orb_strategy_20260126.json...
✅ Parsed 4 chunks
✅ Extracted concepts: entry=3, exit=2, filters=4
✅ Searched KB: 12 relevant articles found

⚠️  Missing information detected:

1. Position sizing method
   Enter value (or 'skip'): risk_based

2. Take profit calculation
   Enter value (or 'skip'): 1.5x risk

3. Stop loss placement
   Enter value (or 'skip'): ATR 1.5

✅ Generating TRD...
✅ Saved to: docs/trds/orb_breakout_scalper_20260126.md

Next steps:
- Review the TRD file
- Run: python tools/analyst_cli.py --complete docs/trds/orb_breakout_scalper_20260126.md
```

### 7.2 Command-Line Options

```bash
# Generate from specific file
python tools/analyst_cli.py --input outputs/videos/abc123.json

# Autonomous mode (use defaults for missing info)
python tools/analyst_cli.py --auto --input outputs/videos/abc123.json

# Custom output directory
python tools/analyst_cli.py --input outputs/videos/abc123.json --output custom_trds/

# List available inputs
python tools/analyst_cli.py --list

# Show KB stats
python tools/analyst_cli.py --stats

# Complete existing TRD
python tools/analyst_cli.py --complete docs/trds/orb_strategy.md

# Generate from strategy document
python tools/analyst_cli.py --input strategy_notes.md --type strategy_doc
```

### 7.3 Configuration File

**File:** `.analyst_config.json` (in project root)

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

---

## 8. LangGraph Workflow

### 8.1 Node Implementations

**File:** `analyst_agent/graph/nodes.py`

```python
"""LangGraph nodes for Analyst Agent workflow."""

from typing import Dict, Any
from .state import AnalystState

def parse_input_node(state: AnalystState) -> AnalystState:
    """Parse input file (NPRD JSON or strategy doc)."""
    input_path = state['input_path']
    input_type = state['input_type']

    if input_type == "nprd":
        with open(input_path) as f:
            data = json.load(f)

        state['raw_content'] = data
        state['transcript'] = data['summary']['full_transcript']
        state['keywords'] = data['summary']['all_keywords']

    elif input_type == "strategy_doc":
        content = Path(input_path).read_text()
        state['raw_content'] = {"content": content}
        state['transcript'] = content

    state['current_step'] = "parsed"
    return state

def extract_concepts_node(state: AnalystState) -> AnalystState:
    """Extract trading concepts using LangChain."""
    from ..chains.extraction import extraction_chain

    result = extraction_chain.invoke({"content": state['transcript']})

    state['entry_conditions'] = result.get('entry_conditions', [])
    state['exit_conditions'] = result.get('exit_conditions', [])
    state['filters'] = result.get('filters', [])
    state['time_filters'] = result.get('time_filters', [])
    state['indicators'] = result.get('indicators_used', [])

    state['current_step'] = "extracted"
    return state

def search_kb_node(state: AnalystState) -> AnalystState:
    """Search knowledge base for relevant articles."""
    from ..kb.client import AnalystKBClient

    client = AnalystKBClient()

    # Generate search queries
    queries = generate_search_queries(state)

    # Search
    all_results = []
    for query in queries:
        results = client.search(query, n_results=5)
        all_results.extend(results)

    # Deduplicate
    seen = set()
    unique = []
    for r in all_results:
        if r['title'] not in seen:
            seen.add(r['title'])
            unique.append(r)

    state['kb_references'] = unique[:10]
    state['current_step'] = "searched"
    return state

def identify_gaps_node(state: AnalystState) -> AnalystState:
    """Identify missing information."""
    from ..utils.gap_detector import detect_gaps

    missing = detect_gaps(state)
    state['missing_info'] = missing
    state['needs_user_input'] = len(missing) > 0

    state['current_step'] = "gaps_identified"
    return state

def ask_user_node(state: AnalystState) -> AnalystState:
    """Ask user for missing information (interactive)."""
    answers = {}

    for gap in state['missing_info']:
        print(f"\n❓ Missing: {gap}")
        answer = input("Enter value (or 'skip'): ").strip()
        if answer.lower() != 'skip':
            answers[gap] = answer

    state['user_answers'] = answers
    state['current_step'] = "user_input_received"
    return state

def generate_trd_node(state: AnalystState) -> AnalystState:
    """Generate final TRD markdown file."""
    from ..utils.trd_template import generate_trd

    trd_content = generate_trd(state)

    # Save file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    strategy_name = sanitize_filename(state.get('strategy_name', 'strategy'))
    filename = f"{strategy_name}_{timestamp}.md"
    output_path = Path("docs/trds") / filename

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(trd_content)

    state['trd_content'] = trd_content
    state['trd_path'] = str(output_path)
    state['completed'] = True

    return state

def has_gaps(state: AnalystState) -> str:
    """Conditional edge: check if gaps exist."""
    return "ask_user" if state['needs_user_input'] else "generate"
```

### 8.2 Graph Invocation

**File:** `analyst_agent/graph/workflow.py`

```python
"""LangGraph workflow orchestration."""

from langgraph.graph import StateGraph, END
from .state import AnalystState
from .nodes import (
    parse_input_node,
    extract_concepts_node,
    search_kb_node,
    identify_gaps_node,
    ask_user_node,
    generate_trd_node,
    has_gaps
)

def create_analyst_graph():
    """Create and compile the analyst workflow graph."""
    workflow = StateGraph(AnalystState)

    # Add all nodes
    workflow.add_node("parse_input", parse_input_node)
    workflow.add_node("extract_concepts", extract_concepts_node)
    workflow.add_node("search_knowledge_base", search_kb_node)
    workflow.add_node("identify_gaps", identify_gaps_node)
    workflow.add_node("ask_user", ask_user_node)
    workflow.add_node("generate_trd", generate_trd_node)

    # Define flow
    workflow.set_entry_point("parse_input")

    workflow.add_edge("parse_input", "extract_concepts")
    workflow.add_edge("extract_concepts", "search_knowledge_base")
    workflow.add_edge("search_knowledge_base", "identify_gaps")

    # Conditional: ask user or go straight to generation
    workflow.add_conditional_edges(
        "identify_gaps",
        has_gaps,
        {
            "ask_user": "ask_user",
            "generate": "generate_trd"
        }
    )

    workflow.add_edge("ask_user", "generate_trd")
    workflow.add_edge("generate_trd", END)

    return workflow.compile()

# Usage
def run_analyst_workflow(input_path: str, input_type: str = "nprd", auto_mode: bool = False):
    """Run the analyst workflow."""

    # Initialize state
    initial_state: AnalystState = {
        "input_path": input_path,
        "input_type": input_type,
        "transcript": "",
        "keywords": [],
        "entry_conditions": [],
        "exit_conditions": [],
        "filters": [],
        "time_filters": [],
        "kb_references": [],
        "missing_info": [],
        "user_answers": {},
        "trd_content": "",
        "trd_path": "",
        "current_step": "start",
        "needs_user_input": False,
        "completed": False
    }

    # Create and run graph
    graph = create_analyst_graph()

    if auto_mode:
        # In auto mode, replace ask_user node with default answers
        config = {"recursion_limit": 20}
    else:
        config = {"recursion_limit": 20}

    result = graph.invoke(initial_state, config)

    return result
```

---

## 9. File Organization

### 9.1 Directory Structure

```
tools/
├── analyst_cli.py              # Main CLI entry point
│
├── analyst_agent/              # Analyst Agent package
│   ├── __init__.py
│   │
│   ├── cli/                    # CLI module
│   │   ├── __init__.py
│   │   ├── commands.py         # Click command definitions
│   │   └── interface.py        # Interactive prompts
│   │
│   ├── graph/                  # LangGraph workflow
│   │   ├── __init__.py
│   │   ├── workflow.py         # StateGraph definition
│   │   ├── nodes.py            # Individual node functions
│   │   └── state.py            # TypedDict state definition
│   │
│   ├── chains/                 # LangChain chains
│   │   ├── __init__.py
│   │   ├── extraction.py       # Concept extraction
│   │   ├── search.py           # KB search
│   │   └── generation.py       # TRD generation
│   │
│   ├── kb/                     # Knowledge base client
│   │   ├── __init__.py
│   │   └── client.py           # ChromaDB wrapper
│   │
│   ├── prompts/                # Prompt templates
│   │   ├── __init__.py
│   │   └── templates.py        # All prompt strings
│   │
│   └── utils/                  # Utilities
│       ├── __init__.py
│       ├── file_io.py          # File parsing
│       ├── trd_template.py     # TRD markdown generator
│       ├── gap_detector.py     # Missing info detection
│       └── search_queries.py   # Query generation
│
├── nprd_cli.py                 # Existing NPRD CLI
└── README_ANALYST.md           # Analyst Agent documentation
```

### 9.2 Output Directories

```
docs/
└── trds/                       # Generated TRD files
    ├── orb_breakout_scalper_20260126.md
    ├── ict_concept_strategy_20260125.md
    └── ...

outputs/
└── videos/                     # NPRD outputs (existing)
    ├── orb_strategy_20260126.json
    └── ...
```

---

## 10. Dependencies

### 10.1 Python Packages

```txt
# Core
click>=8.1.0
pydantic>=2.0.0
pyyaml>=6.0

# LangChain
langchain>=0.2.0
langchain-openai>=0.1.0
langchain-anthropic>=0.1.0
langchain-community>=0.2.0

# LangGraph
langgraph>=0.2.0

# ChromaDB
chromadb>=0.4.0

# Utilities
python-dotenv>=1.0.0
rich>=13.0.0          # For CLI formatting
pyfiglet>=1.0.0       # ASCII art
```

### 10.2 API Requirements

| Service | Purpose | Required |
|---------|---------|----------|
| OpenAI API | LLM for extraction/generation | Yes |
| Anthropic API | Alternative LLM option | Optional |
| ChromaDB (local) | Knowledge base | Yes (local) |

### 10.3 Configuration Files

**File:** `.env` (in project root)

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
```

---

## 11. Implementation Phases

### Phase 1: Foundation (Week 1)

**Goal:** Basic structure and KB integration

| Task | File | Description |
|------|------|-------------|
| Create filtered KB | `scripts/create_analyst_kb.py` | Filter articles from main KB |
| KB client wrapper | `analyst_agent/kb/client.py` | ChromaDB query interface |
| State definition | `analyst_agent/graph/state.py` | TypedDict for workflow |
| Basic CLI | `tools/analyst_cli.py` | Click commands |

**Acceptance:**
- `analyst_kb` collection exists with ~330 articles
- Can search KB via Python client
- CLI runs and shows menu

### Phase 2: Extraction & Search (Week 1-2)

**Goal:** Extract concepts and search KB

| Task | File | Description |
|------|------|-------------|
| Extraction chain | `analyst_agent/chains/extraction.py` | LangChain for concept extraction |
| Search chain | `analyst_agent/chains/search.py` | KB search with queries |
| Prompt templates | `analyst_agent/prompts/templates.py` | All prompt strings |
| Gap detection | `analyst_agent/utils/gap_detector.py` | Identify missing info |

**Acceptance:**
- Can parse NPRD JSON output
- Extracts entry/exit/filters from transcript
- Returns relevant KB articles

### Phase 3: Workflow & Generation (Week 2)

**Goal:** LangGraph orchestration

| Task | File | Description |
|------|------|-------------|
| Node implementations | `analyst_agent/graph/nodes.py` | All workflow nodes |
| Workflow graph | `analyst_agent/graph/workflow.py` | StateGraph definition |
| TRD template | `analyst_agent/utils/trd_template.py` | Markdown generator |
| Conditional edges | `analyst_agent/graph/workflow.py` | Gap checking logic |

**Acceptance:**
- LangGraph runs end-to-end
- Generates TRD markdown file
- Saves to `docs/trds/`

### Phase 4: Interactive Mode (Week 2-3)

**Goal:** Human-in-the-loop

| Task | File | Description |
|------|------|-------------|
| Interactive prompts | `analyst_agent/cli/interface.py` | Rich console UI |
| User answers handler | `analyst_agent/graph/nodes.py` | `ask_user_node` |
| Auto mode | `analyst_agent/cli/commands.py` | `--auto` flag |
| Config management | `.analyst_config.json` | Settings persistence |

**Acceptance:**
- Asks user for missing info interactively
- Auto mode uses defaults
- Config file saves preferences

### Phase 5: Testing & Polish (Week 3)

**Goal:** Production ready

| Task | Description |
|------|-------------|
| Test with real NPRD outputs | Validate extraction quality |
| Test with strategy documents | Alternative input path |
| Error handling | Graceful failures |
| Documentation | README, usage examples |

**Acceptance:**
- Processes 5+ real NPRD outputs
- Generates valid TRD files
- No crashes on edge cases

---

## 12. Acceptance Criteria

### 12.1 Functional Requirements

| ID | Requirement | Priority | Test Case |
|----|-------------|----------|-----------|
| FR-1 | Parse NPRD JSON outputs | P1 | Load and parse valid NPRD file |
| FR-2 | Parse strategy documents | P1 | Load markdown strategy doc |
| FR-3 | Extract entry conditions | P1 | Extract from transcript with LLM |
| FR-4 | Extract exit conditions | P1 | Extract TP/SL/trailing |
| FR-5 | Extract filters | P1 | Extract time/volatility filters |
| FR-6 | Search analyst_kb | P1 | Return relevant articles |
| FR-7 | Identify missing info | P1 | Detect gaps in extracted data |
| FR-8 | Ask user for gaps (interactive) | P1 | Prompt for missing fields |
| FR-9 | Generate TRD markdown | P1 | Create valid TRD file |
| FR-10 | Support auto mode | P2 | Use defaults without prompts |
| FR-11 | List available inputs | P2 | Show NPRD outputs |
| FR-12 | Show KB stats | P3 | Display collection info |

### 12.2 Non-Functional Requirements

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-1 | CLI startup time | < 2 seconds |
| NFR-2 | TRD generation time | < 30 seconds |
| NFR-3 | KB search latency | < 500ms |
| NFR-4 | Code quality | PEP8 compliant, type hints |
| NFR-5 | Error handling | Graceful, helpful messages |
| NFR-6 | Documentation | Docstrings on all functions |

### 12.3 Integration Points

| Component | Interface | Data Format |
|-----------|-----------|-------------|
| NPRD → Analyst | File read | JSON |
| Analyst → ChromaDB | Query/Response | Vectors |
| Analyst → File System | Write | Markdown |
| CLI → User | Interactive | Terminal I/O |

---

## 13. Future Enhancements (Out of Scope for v1.0)

- **v2.0:** Risk management system integration
- **v2.0:** Position sizing module
- **v2.0:** Prop firm challenge rules
- **v2.1:** Strategy document validation
- **v2.2:** TRD comparison/diff tool
- **v3.0:** IDE integration (VS Code extension)
- **v3.0:** Multi-strategy batch processing

---

## 14. Notes for Implementation Agent

1. **Start with Phase 1** - Build foundation first (KB filtering, basic CLI)
2. **Use existing patterns** - NPRD CLI structure is a good reference
3. **Keep it simple** - Don't over-engineer, focus on MVP
4. **Test each phase** - Don't move forward until acceptance criteria met
5. **Document as you go** - Docstrings on all functions
6. **Handle errors** - Try/catch with helpful messages
7. **Use type hints** - Helps with LangGraph TypedDict state
8. **ChromaDB client** - Reuse patterns from `server_chroma.py`
9. **TRD template** - Make it easy to modify sections
10. **Config file** - Support both `.env` and `.analyst_config.json`

---

**End of TRD**

*This document contains all necessary information for implementation. Ask questions if any section requires clarification.*

---

## 11. Testing Strategy

### Testing Pyramid

```
                    ┌─────────────────┐
                    │   Manual E2E    │  <- User validates TRD quality
                    │   (5 scenarios)  │
                    └─────────────────┘
                           │
                    ┌─────────────────┐
                    │  Integration    │  <- Test KB search, chains
                    │    Tests        │
                    └─────────────────┘
                           │
                    ┌─────────────────┐
                    │   Unit Tests    │  <- Test individual functions
                    │   (pytest)      │
                    └─────────────────┘
```

### Unit Testing Requirements

**Coverage Target:** 70% minimum, 80% ideal

| Module | Tests Required | Key Test Cases |
|--------|----------------|----------------|
| `kb/client.py` | Qdrant connection, search | Connection success, empty results, query formatting |
| `chains/extraction.py` | Concept extraction | Valid input, missing fields, malformed JSON |
| `chains/search.py` | KB search | Query construction, result ranking, no matches |
| `chains/generation.py` | TRD generation | Template filling, missing info handling |
| `graph/workflow.py` | LangGraph state | State transitions, human input handling |
| `utils/file_io.py` | File parsing | Valid NPRD, chunked files, missing files |

### Integration Testing Requirements

| Scenario | Test Case | Expected Outcome |
|----------|-----------|------------------|
| End-to-end TRD generation | Run `analyst-cli generate --input test_nprd.json` | Valid TRD markdown file created |
| KB search with hits | Query with matching KB articles | References section populated |
| KB search with no hits | Query with no matches | Empty references, warning logged |
| Human-in-the-loop | Missing strategy info | Prompt user, incorporate answer |
| Chunked NPRD input | Multiple JSON chunks | Aggregate before processing |

### Manual Testing Requirements

**Golden Dataset:** 5 manual TRDs created from known good NPRD outputs

| Test Case | Input | Expected Output | Validation |
|-----------|-------|----------------|------------|
| Simple strategy | Basic RSI strategy TRD | All sections complete | Section checklist |
| Complex strategy | Multi-indicator system | KB references found | Reference count ≥ 3 |
| Missing info | Incomplete NPRD | HITL prompts user | Prompt count ≥ 1 |
| Edge case | Very long transcript | No truncation | Word count check |
| Invalid input | Malformed JSON | Graceful error | Error message |

---

## 12. Manual Testing Procedures

### Test Environment Setup

```bash
# 1. Create test environment
cd /home/mubarkahimself/Desktop/QUANTMINDX
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Setup Qdrant with test collection
# (See scripts/setup_test_kb.py)

# 3. Run health check
analyst-cli --help
analyst-cli health check
```

### Test Case 1: Basic TRD Generation

```bash
# Given: A sample NPRD output in JSON format
# When: Run analyst-cli generate
# Then: TRD file is created

analyst-cli generate \
  --input outputs/videos/test_rsi_strategy.json \
  --output docs/trds/test_rsi_strategy_$(date +%Y%m%d).md \
  --kb-collection analyst_kb

# Validation:
# - File exists at docs/trds/test_rsi_strategy_YYYYMMDD.md
# - Has all required sections (1-12)
# - Markdown is valid
# - No "TODO" placeholders except in approved sections
```

### Test Case 2: Human-in-the-Loop Interaction

```bash
# Given: NPRD with missing stop-loss strategy
# When: Run analyst-cli generate interactively
# Then: System prompts for missing info

analyst-cli generate \
  --input outputs/videos/incomplete_strategy.json \
  --interactive

# Expected prompts:
# "What is the stop-loss strategy for this system?"
# "Enter initial lot size (default: 0.01):"

# Validation:
# - Prompt appears and waits for input
# - User input incorporated into TRD
# - Process completes successfully
```

### Test Case 3: Knowledge Base Search

```bash
# Given: NPRD mentioning "trailing stop"
# When: Run analyst-cli generate
# Then: KB articles about trailing stop referenced

# Validation:
grep -i "trailing stop" docs/trds/test_*.md
# Expected: Reference to KB article found
# Check: Reference count > 0
```

### Test Case 4: Chunked NPRD Input

```bash
# Given: Large NPRD split across 3 JSON chunks
# When: Run analyst-cli generate with chunk directory
# Then: Chunks aggregated before processing

analyst-cli generate \
  --input-dir outputs/videos/chunked_strategy/ \
  --pattern "strategy_part_*.json" \
  --output docs/trds/chunked_strategy.md

# Validation:
# - All chunks read (check logs)
# - Aggregated content is coherent
# - No duplication in TRD
```

### Test Case 5: Error Handling

```bash
# Given: Invalid JSON input
# When: Run analyst-cli generate
# Then: Graceful error, no crash

analyst-cli generate \
  --input outputs/videos/corrupt.json

# Expected output:
# ERROR: Failed to parse NPRD output: Invalid JSON at line 42
# HINT: Run analyst-cli validate --input to check file format

# Validation:
# - Exit code = 1
# - Error message is helpful
# - No traceback to user
```

### Quality Validation Checklist

For each generated TRD, manually verify:

```markdown
## TRD Quality Checklist

### Structure (All must be PASS)
- [ ] Has YAML frontmatter with title, date, version
- [ ] Has Executive Summary section
- [ ] Has System Overview with data flow diagram
- [ ] Has Architecture section with component diagram
- [ ] Has Components section with detailed specs
- [ ] Has Data Structures section
- [ ] Has CLI Interface section
- [ ] Has Implementation Phases section
- [ ] Has Acceptance Criteria section

### Content Quality
- [ ] No "TODO" or "FIXME" in generated content
- [ ] All sections have meaningful content (not "N/A")
- [ ] KB references are relevant to the strategy
- [ ] Code examples are syntactically valid (if present)
- [ ] Technical terminology is used correctly

### Completeness
- [ ] Strategy entry conditions are clear
- [ ] Strategy exit conditions are clear
- [ ] Risk parameters are specified (if mentioned in NPRD)
- [ ] Timeframe is specified
- [ ] Indicators required are listed

### Readability
- [ ] Markdown formatting is correct
- [ ] Code blocks have language identifiers
- [ ] Tables are properly formatted
- [ ] No broken links or references
```

---

## 13. Implementation Phases

**IMPORTANT:** Use Spec Driven Development (SDD) methodology:
1. **Write test first** (red)
2. **Implement minimal code** to pass (green)
3. **Refactor** while keeping tests green (refactor)

### Phase 0: Pre-Implementation (Days 1-2)

**Objective:** Setup environment and create filtered ChromaDB collection

#### Task 0.1: Filter ChromaDB to Create analyst_kb Collection
```bash
# Create filter script
cat > scripts/filter_analyst_kb.py << 'EOF'
"""
Filter ChromaDB mql5_knowledge to create analyst_kb collection.

Filters to ~330 articles:
- Trading Systems
- Trading
- Expert Advisors
- Indicators
"""
import chromadb
from pathlib import Path

# Connect to existing ChromaDB
client = chromadb.PersistentClient(path=str(Path("data/chromadb")))
source_collection = client.get_collection("mql5_knowledge")

# Get all data
results = source_collection.get()

# Filter categories
categories = ["Trading Systems", "Trading", "Expert Advisors", "Indicators"]
filtered_docs = []
filtered_metadatas = []
filtered_ids = []

for i, (doc, metadata, id) in enumerate(zip(results['documents'], results['metadatas'], results['ids'])):
    if metadata.get('categories', '') in categories:
        filtered_docs.append(doc)
        filtered_metadatas.append(metadata)
        filtered_ids.append(id)

# Create or get filtered collection
try:
    analyst_kb = client.create_collection(
        name="analyst_kb",
        metadata={"description": "Filtered KB for Analyst Agent (~330 articles)"}
    )
    print(f"Created new collection: analyst_kb")
except:
    analyst_kb = client.get_collection("analyst_kb")
    print(f"Using existing collection: analyst_kb")

# Add filtered documents
if filtered_ids:
    analyst_kb.add(
        documents=filtered_docs,
        metadatas=filtered_metadatas,
        ids=filtered_ids
    )
    print(f"Added {len(filtered_ids)} articles to analyst_kb")
else:
    print("No articles matched the filter criteria")

print(f"analyst_kb now has {analyst_kb.count()} articles")
EOF

python scripts/filter_analyst_kb.py
```

#### Task 0.2: Project Structure
```bash
mkdir -p tools/analyst_agent/{cli,graph,chains,kb,prompts,utils,tests}
touch tools/analyst_agent/__init__.py
touch tools/analyst_cli.py
```

### Phase 1: Foundation (Week 1)

**SDD Approach:** Write tests before implementation

| Day | Task | Test First | Implement |
|-----|------|------------|-----------|
| 1 | Qdrant client | Test connection, search | Implement client |
| 2 | NPRD parser | Test JSON parsing | Implement parser |
| 3 | CLI skeleton | Test --help, --version | Implement Click app |
| 4 | State machine | Test state transitions | Implement LangGraph |
| 5 | Extraction chain | Test concept extraction | Implement chain |

### Phase 2: Core Features (Week 2)

| Day | Task | Deliverable |
|-----|------|-------------|
| 1-2 | KB search chain | Query Qdrant, rank results |
| 3-4 | Generation chain | TRD markdown output |
| 5 | HITL integration | User prompts, state persistence |

### Phase 3: Polish (Week 3-4)

| Day | Task | Deliverable |
|-----|------|-------------|
| 1-2 | Error handling | Graceful failures, logging |
| 3-4 | Testing | Unit tests, integration tests |
| 5 | Documentation | README, API docs |

---

## 14. Acceptance Criteria

### Functional Requirements

| ID | Requirement | Test Method |
|----|-------------|-------------|
| FR-001 | Parse NPRD JSON output | Unit test with sample JSON |
| FR-002 | Extract trading concepts | Integration test with LLM |
| FR-003 | Search Qdrant KB | Unit test with mock Qdrant |
| FR-004 | Generate TRD markdown | Integration test E2E |
| FR-005 | HITL for missing info | Manual interactive test |
| FR-006 | Handle chunked NPRD files | Integration test |
| FR-007 | Validate generated TRD | Manual quality check |

### Non-Functional Requirements

| ID | Requirement | Target | Test Method |
|----|-------------|--------|-------------|
| NFR-001 | CLI response time | <500ms for help, <30s for generation | Benchmark |
| NFR-002 | Code coverage | 70% minimum | pytest --cov |
| NFR-003 | Error handling | No uncaught exceptions | Manual edge case testing |
| NFR-004 | Documentation | All functions documented | pydocstyle check |
| NFR-005 | Type safety | 90% type hints | mypy check |

### Definition of Done

A feature is complete when:
- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] Code reviewed
- [ ] Documentation updated
- [ ] Manual testing completed
- [ ] Acceptance criteria verified

---

## 15. Appendix: Project Context

### A.1 Project Structure

```
QUANTMINDX/
├── data/
│   ├── chromadb/                    # ChromaDB format (to be migrated)
│   │   └── chroma.sqlite3
│   └── qdrant_storage/              # Qdrant persistent storage
│       └── ...
├── docs/
│   ├── trds/                        # Technical Requirements (output)
│   ├── architecture/                # Architecture diagrams
│   ├── KNOWLEDGE_BASES_AND_IDE.md   # KB integration guide
│   └── KNOWLEDGE_BASE_SETUP.md      # KB setup instructions
├── mcp-servers/
│   └── quantmindx-kb/               # Knowledge base MCP server
│       ├── server_chroma.py          # ChromaDB MCP (to be replaced)
│       └── server_simple.py
├── outputs/
│   └── videos/                      # NPRD outputs (input)
│       ├── strategy_1.json
│       ├── strategy_2_chunk_1.json
│       └── ...
├── scripts/
│   ├── index_to_qdrant.py            # Existing Qdrant indexer
│   └── migrate_to_qdrant.py          # KB migration script (to create)
├── tools/
│   └── analyst_agent/               # NEW: Analyst Agent CLI
│       ├── cli/
│       ├── graph/
│       ├── chains/
│       ├── kb/
│       ├── prompts/
│       ├── utils/
│       └── tests/
└── requirements.txt
```

### A.2 Existing Infrastructure

**Qdrant Vector Database:**
- URL: http://localhost:6333
- Collections: `mql5_knowledge` (1,805 articles)
- Embedding Model: sentence-transformers (384 dimensions)
- Distance: Cosine

**Knowledge Base Articles:**
- Categories: Trading, Trading Systems, Expert Advisors, Indicators
- Target Filter: ~330 articles for analyst_kb
- Format: Markdown with YAML frontmatter

**NPRD Output Format:**
```json
{
  "video_title": "RSI Oversold Strategy",
  "transcript": "Full video text transcript...",
  "ocr_text": "Text from video screenshots...",
  "keywords": ["RSI", "oversold", "entry", "exit"],
  "timestamp_ranges": [
    {"start": "00:00", "end": "02:30", "topic": "Introduction"},
    {"start": "02:30", "end": "15:00", "topic": "Strategy Rules"}
  ],
  "detected_concepts": {
    "indicators": ["RSI", "EMA"],
    "timeframe": "H1",
    "entry_conditions": ["RSI < 30", "Price crosses EMA up"],
    "exit_conditions": ["RSI > 70", "Take profit at 1:2 R:R"]
  }
}
```

### A.3 Integration Points

**MCP Server:**
The analyst_agent will integrate with the existing MCP server structure:
- Current: `mcp-servers/quantmindx-kb/server_chroma.py`
- Future: Support Qdrant queries via MCP

**RiskMind Component (v2.0):**
- Will validate strategies generated by Analyst Agent
- Will check against risk limits before approving for implementation

**Strategy Router (v2.0):**
- Will route generated TRDs to appropriate implementation teams
- Will track TRD status (draft, approved, implemented, deprecated)

### A.4 Testing Commands Reference

```bash
# Run all tests
pytest tools/analyst_agent/tests/ -v

# Run with coverage
pytest tools/analyst_agent/tests/ --cov=tools/analyst_agent --cov-report=html

# Run specific test
pytest tools/analyst_agent/tests/test_kb_client.py -v

# Run integration tests
pytest tools/analyst_agent/tests/integration/ -v

# Run manual test
analyst-cli generate --input outputs/videos/test.json --dry-run

# Validate NPRD format
analyst-cli validate --input outputs/videos/test.json

# Health check
analyst-cli health check
```

### A.5 Common Issues & Solutions

| Issue | Cause | Solution |
|-------|--------|----------|
| Qdrant connection refused | Qdrant not running | `docker-compose up -d qdrant` |
| Empty KB results | Wrong collection name | Check collection exists with `analyst-cli kb list` |
| LLM API errors | Invalid API key | Check `.env` has `OPENAI_API_KEY` |
| Chunked files not aggregating | Wrong pattern | Ensure pattern matches all chunks |
| TRD missing sections | LLM hallucination | Regenerate with lower temperature |

### A.6 Environment Variables

```bash
# .env file
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=analyst_kb
LOG_LEVEL=INFO
TRD_OUTPUT_DIR=docs/trds
NPRD_INPUT_DIR=outputs/videos
```

### A.7 Quick Start Commands

```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python scripts/migrate_to_qdrant.py

# Run analyst agent
analyst-cli generate --input outputs/videos/my_strategy.json

# Interactive mode
analyst-cli generate --input outputs/videos/incomplete.json --interactive

# Batch processing
analyst-cli batch --input-dir outputs/videos/ --output-dir docs/trds/

# Health check
analyst-cli health check

# Validate input
analyst-cli validate --input outputs/videos/my_strategy.json
```

---

**TRD Version:** 1.1 (Enhanced for SDD Implementation)
**Last Modified:** 2026-01-27
**Ready for Swarm Implementation:** YES

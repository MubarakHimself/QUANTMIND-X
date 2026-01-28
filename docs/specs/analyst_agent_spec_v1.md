# Analyst Agent CLI - Specification Document v1.0

> **Project:** QuantMindX
> **Component:** Analyst Agent CLI
> **Version:** 1.0 (Fast/MVP)
> **Status:** Production Specification
> **Date:** 2026-01-27
> **Author:** Specification Writer Agent

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [System Overview](#2-system-overview)
3. [Functional Requirements](#3-functional-requirements)
4. [Non-Functional Requirements](#4-non-functional-requirements)
5. [API Specifications](#5-api-specifications)
6. [Data Models](#6-data-models)
7. [Error Handling](#7-error-handling)
8. [Security Requirements](#8-security-requirements)
9. [Testing Requirements](#9-testing-requirements)
10. [Deployment Requirements](#10-deployment-requirements)
11. [Acceptance Criteria](#11-acceptance-criteria)
12. [Glossary](#12-glossary)

---

## 1. Introduction

### 1.1 Purpose

This specification document defines the requirements for the **Analyst Agent CLI**, a command-line tool that converts unstructured trading content from NPRD (Non-Processing Research Data) video outputs into structured Technical Requirements Document (TRD) files. The system leverages a filtered ChromaDB knowledge base to enrich extracted content with relevant MQL5 articles and supports human-in-the-loop interaction for collecting missing information.

### 1.2 Scope

**In Scope:**
- Parsing NPRD video output JSON files
- Parsing strategy documents (markdown format)
- Extracting trading concepts using LLMs
- Semantic search against filtered ChromaDB knowledge base
- Identifying missing information gaps
- Interactive human-in-the-loop data collection
- Generating structured TRD markdown files
- CLI interface with interactive and autonomous modes

**Out of Scope (Future Enhancements):**
- Risk management system integration (planned v2.0)
- Dynamic position sizing module (planned v2.0)
- Prop firm challenge rules enforcement (planned v2.0)
- IDE integration (planned v3.0)
- Multi-strategy batch processing (planned v3.0)
- Autonomous trading capabilities (explicitly excluded)

### 1.3 Target Audience

**Primary Users:**
- Mubarak (project owner) - Quantitative trading strategy developer
- Strategy analysts - Converting video content to documentation
- MQL5 developers - Using generated TRDs for EA development

**Secondary Users:**
- System administrators - Managing knowledge base and deployments
- Quality assurance - Testing TRD generation quality

### 1.4 Document Conventions

| Convention | Meaning |
|------------|---------|
| **MUST** | Absolute requirement |
| **MUST NOT** | Absolute prohibition |
| **REQUIRED** | Same as MUST |
| **SHALL** | Same as MUST |
| **SHALL NOT** | Same as MUST NOT |
| **SHOULD** | Recommended practice |
| **SHOULD NOT** | Not recommended practice |
| **MAY** | Optional feature |
| **OPTIONAL** | Same as MAY |

---

## 2. System Overview

### 2.1 System Context

```
┌─────────────────────────────────────────────────────────────────┐
│                          External Systems                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│  │   NPRD       │    │ ChromaDB KB  │    │   User       │     │
│  │   Outputs    │    │ analyst_kb   │    │   Terminal   │     │
│  │  (JSON)      │    │ (~330 arts)  │    │   (CLI)      │     │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘     │
│         │                   │                   │              │
│         │ File Read         │ Semantic Search   │ Interactive  │
│         │                   │                   │ I/O          │
└─────────┼───────────────────┼───────────────────┼──────────────┘
          │                   │                   │
          ▼                   ▼                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Analyst Agent CLI                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │   CLI    │  │ LangGraph│  │LangChain │  │   KB     │       │
│  │ Interface│  │ Workflow │  │  Chains  │  │ Client   │       │
│  │ (Click)  │  │Orchestration│  │ Extraction│  │(ChromaDB)│       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
│                                                                   │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              │ File Write
                              ▼
                    ┌──────────────────┐
                    │   TRD Files      │
                    │   (Markdown)     │
                    │ docs/trds/       │
                    └──────────────────┘
```

### 2.2 Data Flow

```
┌────────────────┐
│   User Input   │
│  (CLI Command) │
└───────┬────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│ 1. PARSE INPUT                                                │
│    - Read NPRD JSON or Strategy Doc                           │
│    - Extract transcript, OCR text, keywords                  │
└───────┬──────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│ 2. EXTRACT CONCEPTS (LangChain + LLM)                         │
│    - Entry conditions                                         │
│    - Exit conditions                                          │
│    - Filters (time, volatility, spread)                       │
│    - Indicators used                                          │
└───────┬──────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│ 3. SEARCH KNOWLEDGE BASE (ChromaDB)                           │
│    - Generate semantic queries                                │
│    - Retrieve relevant MQL5 articles                          │
│    - Rank by relevance                                        │
└───────┬──────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│ 4. IDENTIFY GAPS                                              │
│    - Check against required TRD fields                        │
│    - List missing information                                 │
└───────┬──────────────────────────────────────────────────────┘
        │
        ▼
     ┌──┴───┐
     │ Gaps?│
     └──┬───┘
      ┌─┴─┐
      │NO │    ┌────────────────────────────────────────────────┐
      └───┘    │ 6A. GENERATE TRD (Direct)                      │
               │    - Use extracted data only                   │
               └────────┬───────────────────────────────────────┘
      ┌─┴─┐           │
      │YES│           ▼
      └───┘    ┌────────────────────────────────────────────────┐
               │ 6B. ASK USER (Interactive) or USE DEFAULTS (Auto)│
               │    - Collect missing information                │
               │    - Update state with answers                  │
               └────────┬───────────────────────────────────────┘
                        │
                        ▼
               ┌────────────────────────────────────────────────┐
               │ 7. GENERATE TRD                                 │
               │    - Combine all collected data                │
               │    - Format as markdown TRD                    │
               │    - Write to docs/trds/                       │
               └────────┬───────────────────────────────────────┘
                        │
                        ▼
               ┌────────────────────────────────────────────────┐
               │ 8. OUTPUT                                       │
               │    - Display TRD location                      │
               │    - Show completion status                    │
               │    - Suggest next steps                        │
               └────────────────────────────────────────────────┘
```

### 2.3 Technology Stack

| Layer | Technology | Version | Purpose |
|-------|------------|---------|---------|
| **CLI Framework** | Click | >=8.1.0 | Command-line interface |
| **Orchestration** | LangGraph | >=0.2.0 | State machine workflow |
| **Chain Composition** | LangChain | >=0.2.0 | LLM chain management |
| **Vector Database** | ChromaDB | >=0.4.0 | Knowledge base storage |
| **LLM Providers** | OpenAI/Anthropic | - | Text generation |
| **Configuration** | YAML/JSON | - | Settings persistence |
| **CLI Formatting** | Rich | >=13.0.0 | Terminal UI |
| **Python Runtime** | Python | >=3.10 | Execution environment |

---

## 3. Functional Requirements

### 3.1 User Stories

#### US-1: NPRD Video Processing
**As a** strategy analyst
**I want to** process NPRD video output JSON files
**So that** I can convert unstructured video content into structured TRD files

**Acceptance Criteria:**
- System MUST accept NPRD JSON files via `--input` flag
- System MUST parse video metadata (video_id, title, URL)
- System MUST extract transcript from all chunks
- System MUST extract OCR text from visual descriptions
- System MUST aggregate keywords across all chunks
- System MUST display parsing progress with status messages

**Priority:** P0 (Required for MVP)

#### US-2: Strategy Document Processing
**As a** strategy developer
**I want to** process strategy markdown documents
**So that** I can generate TRDs from compiled strategy notes

**Acceptance Criteria:**
- System MUST accept markdown files via `--input --type strategy_doc`
- System MUST parse markdown content as plain text
- System MUST apply same extraction pipeline as NPRD
- System MUST handle markdown formatting gracefully

**Priority:** P1 (Required for MVP)

#### US-3: Trading Concept Extraction
**As a** strategy analyst
**I want to** automatically extract trading concepts from unstructured text
**So that** I don't have to manually identify strategy components

**Acceptance Criteria:**
- System MUST extract entry conditions (minimum 1 per strategy)
- System MUST extract exit conditions (take profit, stop loss, trailing)
- System MUST extract filters (time, volatility, spread, trend)
- System MUST extract time filters (trading sessions, days to avoid)
- System MUST extract indicators mentioned with settings
- System MUST identify trading concepts (support/resistance, order flow, etc.)
- Extraction MUST return structured JSON format
- Extraction MUST handle ambiguous content with confidence scores

**Priority:** P0 (Required for MVP)

#### US-4: Knowledge Base Enrichment
**As a** strategy analyst
**I want to** find relevant MQL5 articles for my strategy
**So that** I can leverage proven implementation patterns

**Acceptance Criteria:**
- System MUST query ChromaDB `analyst_kb` collection
- System MUST generate 3-5 semantic search queries from extracted concepts
- System MUST retrieve top 5 articles per query
- System MUST deduplicate results by article title
- System MUST return maximum 10 unique articles
- Results MUST include: title, URL, relevance score, content preview
- Search MUST complete within 500ms

**Priority:** P0 (Required for MVP)

#### US-5: Gap Identification
**As a** strategy analyst
**I want to** know what information is missing from my strategy
**So that** I can complete the strategy before implementation

**Acceptance Criteria:**
- System MUST validate against required TRD fields:
  - Entry logic (trigger, confirmation)
  - Exit logic (take profit, stop loss, trailing stop)
  - Risk management (position sizing, max risk)
  - Filters (time filters, market condition filters)
  - Indicators (settings, timeframes)
- System MUST list all missing fields
- System MUST categorize gaps by severity (critical, important, optional)
- System MUST provide examples for each missing field

**Priority:** P0 (Required for MVP)

#### US-6: Interactive Data Collection
**As a** strategy analyst
**I want to** be prompted for missing information during TRD generation
**So that** I can complete the TRD in one session

**Acceptance Criteria:**
- System MUST prompt for each missing field sequentially
- System MUST display field description and example
- System MUST accept free-form text input
- System MUST support 'skip' command to omit fields
- System MUST validate input format (numbers for percentages, etc.)
- System MUST store answers in workflow state
- System MUST allow corrections before final generation

**Priority:** P1 (Required for MVP)

#### US-7: Autonomous Mode
**As a** strategy analyst
**I want to** generate TRDs without interactive prompts
**So that** I can batch process multiple strategies

**Acceptance Criteria:**
- System MUST support `--auto` flag to skip prompts
- System MUST use default values from configuration file
- System MUST mark auto-filled fields in TRD
- System MUST log all default values used

**Priority:** P2 (High priority)

#### US-8: TRD Generation
**As a** strategy analyst
**I want to** receive a formatted TRD markdown file
**So that** I can share it with developers or use it for coding

**Acceptance Criteria:**
- System MUST generate markdown file with YAML frontmatter
- System MUST include: strategy name, source, timestamp, status
- System MUST organize content into sections: Overview, Entry Logic, Exit Logic, Filters, Indicators, Risk Management, KB References, Missing Info, Next Steps
- System MUST save to `docs/trds/{strategy_name}_{timestamp}.md`
- System MUST create output directory if not exists
- System MUST validate markdown syntax before writing
- System MUST display file path after generation

**Priority:** P0 (Required for MVP)

#### US-9: Input Discovery
**As a** strategy analyst
**I want to** list available NPRD outputs
**So that** I can select which strategy to process

**Acceptance Criteria:**
- System MUST support `--list` flag
- System MUST display all JSON files in `outputs/videos/`
- System MUST show: filename, timestamp, file size
- System MUST sort by modification time (newest first)
- List MUST support pagination if >20 files

**Priority:** P2 (High priority)

#### US-10: Knowledge Base Statistics
**As a** system administrator
**I want to** view knowledge base statistics
**So that** I can verify data integrity

**Acceptance Criteria:**
- System MUST support `--stats` flag
- System MUST display: total articles, collection name, storage path
- System MUST show last update timestamp
- System MUST verify collection accessibility

**Priority:** P3 (Nice to have)

### 3.2 Input Specifications

#### 3.2.1 NPRD Video Output Format

**File Location:** `outputs/videos/{video_id}_{timestamp}.json`

**Schema:**
```json
{
  "type": "object",
  "required": ["video_id", "video_title", "video_url", "processed_at", "chunks", "summary"],
  "properties": {
    "video_id": {
      "type": "string",
      "description": "Unique video identifier",
      "pattern": "^[a-zA-Z0-9_-]+$"
    },
    "video_title": {
      "type": "string",
      "description": "Video title from source",
      "minLength": 1,
      "maxLength": 200
    },
    "video_url": {
      "type": "string",
      "description": "Full URL to video source",
      "format": "uri"
    },
    "processed_at": {
      "type": "string",
      "description": "ISO 8601 timestamp",
      "format": "date-time"
    },
    "chunks": {
      "type": "array",
      "description": "Video content chunks",
      "items": {
        "type": "object",
        "required": ["chunk_id", "start_time", "end_time", "transcript"],
        "properties": {
          "chunk_id": {"type": "integer"},
          "start_time": {
            "type": "string",
            "description": "Timestamp in HH:MM format",
            "pattern": "^\\d{2}:\\d{2}$"
          },
          "end_time": {
            "type": "string",
            "pattern": "^\\d{2}:\\d{2}$"
          },
          "transcript": {
            "type": "string",
            "description": "Speech-to-text transcript"
          },
          "visual_description": {
            "type": "string",
            "description": "Description of visual content"
          },
          "ocr_text": {
            "type": "string",
            "description": "Extracted text from visuals"
          },
          "keywords": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Extracted keywords"
          }
        }
      }
    },
    "summary": {
      "type": "object",
      "required": ["full_transcript", "all_keywords"],
      "properties": {
        "full_transcript": {
          "type": "string",
          "description": "Concatenated transcript from all chunks"
        },
        "all_keywords": {
          "type": "array",
          "items": {"type": "string"},
          "description": "Unique keywords from all chunks"
        },
        "visual_elements": {
          "type": "array",
          "items": {"type": "string"},
          "description": "Types of visual content detected"
        }
      }
    }
  }
}
```

**Validation Rules:**
- `chunks` array MUST contain at least 1 element
- `transcript` in each chunk MUST NOT be empty
- `full_transcript` MUST be concatenation of all chunk transcripts
- `all_keywords` MUST be deduplicated
- Timestamps MUST be in chronological order

#### 3.2.2 Strategy Document Format

**File Location:** User-provided path (any directory)

**Format:** Markdown (`.md`)

**Expected Content:**
- Free-form text describing trading strategy
- MAY include markdown formatting (headers, lists, code blocks)
- MAY include sections (Entry, Exit, Risk Management, etc.)
- Length: 100 - 50,000 characters

**Validation Rules:**
- File MUST be readable
- Content MUST NOT be empty (>100 characters)
- System SHOULD extract text from markdown formatting

### 3.3 Output Specifications

#### 3.3.1 TRD File Structure

**File Location:** `docs/trds/{strategy_name}_{timestamp}.md`

**Filename Format:**
- Strategy name: lowercase, spaces to underscores, special chars removed
- Timestamp: `YYYYMMDD_HHMMSS`
- Example: `orb_breakout_scalper_20260126_143000.md`

**Content Structure:**

```markdown
---
strategy_name: "{Sanitized strategy name}"
source: "{Input source}"
generated_at: "{ISO 8601 timestamp}"
status: "draft|complete"
version: "1.0"
analyst_version: "1.0"
kb_collection: "analyst_kb"
kb_articles_count: {Number of referenced articles}
---

# Trading Strategy: {Strategy Name}

## Overview

{Brief strategy description (1-2 paragraphs)}

**Source:** {Video title or document name}
**Analyst:** Analyst Agent v1.0
**Generated:** {Timestamp}
**Status:** {Draft/Complete}

---

## Entry Logic

### Primary Entry Trigger

- {Entry condition 1}
- {Entry condition 2}
- {Entry condition N}

### Entry Confirmation

- {Confirmation signal if mentioned}
- {Timeframe: e.g., 1-minute, 5-minute}

### Entry Example

```
IF [condition 1] AND [condition 2]
AND time is within [session]
THEN enter [LONG/SHORT] at [price level]
```

---

## Exit Logic

### Take Profit

- {TP strategy: fixed pips, percentage, resistance}
- {TP value if specified}

### Stop Loss

- {SL strategy: fixed pips, ATR-based, support}
- {SL value if specified}

### Trailing Stop

- {Trailing method if mentioned}
- {Trail distance}

### Time Exit

- {Close at specific time if mentioned}

---

## Filters

### Time Filters

- **Trading Session:** {London, NY, Asian, overlap}
- **Time of Day:** {e.g., 8am-12pm GMT}
- **Days to Avoid:** {Friday, news days, etc.}

### Market Condition Filters

- **Volatility:** {ATR threshold, avoid low volatility}
- **Spread:** {Max spread allowed}
- **Trend:** {Trend following or range}
- **News Events:** {Avoid high-impact news}

---

## Indicators & Settings

| Indicator | Settings | Purpose |
|-----------|----------|---------|
| {RSI} | {Period: 14} | {Entry/exit signal} |
| {ATR} | {Period: 14} | {Stop loss calculation} |
| {MA} | {Period: 20, EMA} | {Trend filter} |

---

## Position Sizing & Risk Management

> **Note:** This section will be enhanced in v2.0 with dynamic risk management system.

### Risk Per Trade

- {Mentioned: e.g., 0.5% of account}
- {Or: TODO - specify risk amount}

### Position Sizing

- {Method: fixed lots, risk-based, volatility-based}
- {Or: TODO - implement position sizing}

### Max Drawdown Limit

- {Daily limit if mentioned}
- {Or: TODO - specify drawdown limit}

---

## Knowledge Base References

### Relevant Articles Found

1. **[{Article Title 1}]({URL})**
   - **Relevance:** {why it matters}
   - **Key Insight:** {specific implementation detail}
   - **Category:** {MQL5 category}

2. **[{Article Title 2}]({URL})**
   - **Relevance:** {implementation pattern}
   - **Key Insight:** {technical consideration}

### Implementation Notes

- {Technical considerations from KB articles}
- {Potential pitfalls mentioned in literature}
- {Suggested MQL5 patterns}

---

## Missing Information (Requires Input)

The following information was not found in the source and needs user input:

- [ ] [{Field 1}]: {description}
- [ ] [{Field 2}]: {description}

**To complete this TRD, run:**
```bash
python tools/analyst_cli.py --complete docs/trds/{this_file}.md
```

---

## Next Steps

1. **Review this TRD** - Verify all sections are accurate
2. **Complete missing fields** - Provide input for TODO items
3. **Generate MQL5 code** - Use QuantCode CLI (future)
4. **Backtest** - Validate strategy performance

---

**Generated by:** QuantMindX Analyst Agent v1.0
**Knowledge Base:** analyst_kb ({count} articles)
**LangGraph Version:** 0.2.x
```

**Validation Rules:**
- YAML frontmatter MUST be valid
- All required sections MUST be present
- Missing Info section MUST be present if gaps exist
- Filename MUST be filesystem-safe
- File MUST be valid markdown

---

## 4. Non-Functional Requirements

### 4.1 Performance Requirements

| Requirement | Metric | Measurement Method | Priority |
|-------------|--------|-------------------|----------|
| **CLI Startup Time** | < 2 seconds | Time from command to first prompt | P0 |
| **TRD Generation Time** | < 30 seconds | Time from input to file output | P0 |
| **KB Search Latency** | < 500ms | Time per semantic search query | P0 |
| **File Parsing Time** | < 3 seconds | Time to parse NPRD JSON | P1 |
| **Extraction Time** | < 10 seconds | LLM extraction chain execution | P1 |
| **Memory Usage** | < 500MB | Peak memory during execution | P2 |
| **File Size Limits** | Support 10MB NPRD files | Max input file size | P1 |

**Performance Testing:**
- MUST measure with standard NPRD output (10 chunks, 5000 words)
- MUST test with knowledge base of 330 articles
- MUST measure on standard development machine (8GB RAM, 4 CPU)

### 4.2 Reliability Requirements

| Requirement | Specification | Priority |
|-------------|---------------|----------|
| **Uptime** | CLI MUST execute successfully 99% of times | P0 |
| **Error Recovery** | MUST recover from LLM API failures with retry | P0 |
| **Graceful Degradation** | MUST continue if KB search fails | P1 |
| **Data Integrity** | MUST validate all generated TRDs | P0 |
| **State Persistence** | MUST NOT lose data on interruption | P1 |

**Failure Scenarios:**
1. **LLM API Failure:** Retry 3 times with exponential backoff, then use defaults
2. **ChromaDB Unavailable:** Continue without KB references, log warning
3. **File Write Error:** Display error, keep data in memory for retry
4. **Invalid Input:** Display specific error message, suggest corrections

### 4.3 Availability Requirements

| Requirement | Specification | Priority |
|-------------|---------------|----------|
| **CLI Availability** | Available whenever Python runtime is available | P0 |
| **KB Availability** | ChromaDB MUST be accessible or CLI must handle absence | P1 |
| **LLM Availability** | Support at least one provider (OpenAI or Anthropic) | P0 |
| **Offline Mode** | MAY operate without KB (with reduced functionality) | P2 |

### 4.4 Scalability Requirements

| Requirement | Specification | Priority |
|-------------|---------------|----------|
| **File Size** | Support NPRD outputs up to 10MB | P1 |
| **KB Size** | Support knowledge base up to 1,000 articles | P2 |
| **Batch Processing** | Support processing 10+ files in sequence | P2 |
| **Concurrent Users** | Single user (no multi-user requirement) | P3 |

**Future Scalability (Out of Scope):**
- Distributed processing
- Multi-user support
- Cloud deployment
- Real-time streaming

### 4.5 Maintainability Requirements

| Requirement | Specification | Priority |
|-------------|---------------|----------|
| **Code Quality** | PEP 8 compliant, type hints on all functions | P0 |
| **Documentation** | Docstrings on all public functions | P0 |
| **Test Coverage** | Minimum 80% code coverage | P1 |
| **Modularity** | Component-based architecture, clear interfaces | P0 |
| **Configuration** | Externalized configuration (no hardcoded values) | P0 |
| **Logging** | Structured logging with levels (DEBUG, INFO, WARNING, ERROR) | P1 |

### 4.6 Usability Requirements

| Requirement | Specification | Priority |
|-------------|---------------|----------|
| **CLI Help** | `--help` MUST display all commands and options | P0 |
| **Error Messages** | MUST be clear, actionable, and non-technical | P0 |
| **Progress Indication** | MUST show progress for long operations | P1 |
| **Interactive Prompts** | MUST be intuitive with default values | P1 |
| **Output Formatting** | MUST use readable formatting (Rich library) | P1 |

---

## 5. API Specifications

### 5.1 CLI Command Interface

#### 5.1.1 Main Command

```bash
python tools/analyst_cli.py [OPTIONS] COMMAND [ARGS]...
```

**Global Options:**
- `--version` - Show version and exit
- `--help` - Show help message
- `--verbose, -v` - Enable verbose logging
- `--config PATH` - Specify custom config file

#### 5.1.2 Generate Command

```bash
python tools/analyst_cli.py generate [OPTIONS]
```

**Options:**
| Option | Short | Type | Required | Description | Default |
|--------|-------|------|----------|-------------|---------|
| `--input` | `-i` | PATH | No | Input file path (NPRD JSON or strategy doc) | Interactive selection |
| `--type` | `-t` | CHOICE | No | Input type | `nprd` |
| `--auto` | `-a` | FLAG | No | Autonomous mode (no interactive prompts) | False |
| `--output` | `-o` | PATH | No | Output directory for TRD | `docs/trds/` |
| `--model` | `-m` | STRING | No | LLM model to use | From config |
| `--temperature` | FLOAT | No | LLM temperature (0.0-1.0) | 0.1 |

**Input Type Choices:**
- `nprd` - NPRD video output JSON
- `strategy_doc` - Strategy markdown document

**Exit Codes:**
- `0` - Success
- `1` - General error
- `2` - Input file not found
- `3` - Invalid input format
- `4` - LLM API error
- `5` - Knowledge base error

#### 5.1.3 List Command

```bash
python tools/analyst_cli.py list [OPTIONS]
```

**Options:**
| Option | Short | Type | Required | Description | Default |
|--------|-------|------|----------|-------------|---------|
| `--source` | `-s` | CHOICE | No | Source to list | `all` |
| `--limit` | `-l` | INTEGER | No | Maximum items to display | 20 |

**Source Choices:**
- `all` - All available inputs
- `nprd` - Only NPRD outputs
- `strategy_docs` - Only strategy documents

**Output Format:**
```
Available NPRD Outputs:

[1] orb_strategy_20260126.json (2 hours ago, 45 KB)
[2] ict_concepts_20260125.json (1 day ago, 38 KB)
[3] scalping_setup_20260124.json (2 days ago, 52 KB)

Total: 3 files
```

#### 5.1.4 Stats Command

```bash
python tools/analyst_cli.py stats [OPTIONS]
```

**Options:**
| Option | Short | Type | Required | Description | Default |
|--------|-------|------|----------|-------------|---------|
| `--kb` | FLAG | No | Show knowledge base statistics | False |

**Output Format:**
```
Knowledge Base Statistics:

Collection: analyst_kb
Total Articles: 330
Storage Path: /home/user/project/data/chromadb
Last Updated: 2026-01-26 14:30:00

Categories:
  - Trading Systems: 180 articles
  - Trading: 95 articles
  - Expert Advisors: 35 articles
  - Indicators: 20 articles
```

#### 5.1.5 Complete Command

```bash
python tools/analyst_cli.py complete [OPTIONS] TRD_FILE
```

**Options:**
| Option | Short | Type | Required | Description | Default |
|--------|-------|------|----------|-------------|---------|
| `--trd` | PATH | Yes | Path to incomplete TRD file | - |
| `--interactive` | FLAG | No | Force interactive mode | True |

**Behavior:**
- Loads existing TRD file
- Parses "Missing Information" section
- Prompts user for each missing field
- Updates TRD with new information
- Removes "Missing Information" section when complete

### 5.2 MCP Server Tools (Future Enhancement)

**Note:** MCP server integration is planned for v2.0

#### 5.2.1 Tool: generate_trd

```python
{
  "name": "generate_trd",
  "description": "Generate a TRD from NPRD output or strategy document",
  "inputSchema": {
    "type": "object",
    "properties": {
      "input_path": {
        "type": "string",
        "description": "Path to input file"
      },
      "input_type": {
        "type": "string",
        "enum": ["nprd", "strategy_doc"],
        "description": "Type of input file"
      },
      "auto_mode": {
        "type": "boolean",
        "description": "Use defaults for missing information",
        "default": false
      },
      "output_dir": {
        "type": "string",
        "description": "Output directory for TRD",
        "default": "docs/trds/"
      }
    },
    "required": ["input_path"]
  }
}
```

**Response:**
```python
{
  "trd_path": "docs/trds/orb_breakout_scalper_20260126.md",
  "status": "complete|draft",
  "missing_fields": ["position_sizing", "take_profit"],
  "kb_articles_referenced": 12,
  "processing_time_seconds": 25.3
}
```

#### 5.2.2 Tool: search_knowledge_base

```python
{
  "name": "search_knowledge_base",
  "description": "Search the analyst knowledge base",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Search query"
      },
      "n_results": {
        "type": "integer",
        "description": "Number of results to return",
        "default": 5,
        "minimum": 1,
        "maximum": 20
      },
      "category_filter": {
        "type": "string",
        "description": "Filter by category",
        "enum": ["Trading Systems", "Trading", "Expert Advisors", "Indicators"]
      }
    },
    "required": ["query"]
  }
}
```

**Response:**
```python
{
  "results": [
    {
      "title": "Implementing ORB Strategy",
      "url": "https://www.mql5.com/en/articles/...",
      "category": "Trading Systems",
      "relevance_score": 0.92,
      "preview": "Opening Range Breakout strategy implementation..."
    }
  ],
  "total_count": 12
}
```

### 5.3 Internal Component APIs

#### 5.3.1 KB Client API

**File:** `analyst_agent/kb/client.py`

```python
class AnalystKBClient:
    """Client for querying analyst_kb collection."""

    def __init__(self, chroma_path: str = None) -> None:
        """Initialize ChromaDB client.

        Args:
            chroma_path: Path to ChromaDB persistence directory
        """
        ...

    def search(
        self,
        query: str,
        n_results: int = 5,
        category_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search knowledge base.

        Args:
            query: Search query text
            n_results: Number of results to return
            category_filter: Optional category filter

        Returns:
            List of search results with metadata
        """
        ...

    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics.

        Returns:
            Dictionary with collection stats
        """
        ...
```

#### 5.3.2 Workflow State API

**File:** `analyst_agent/graph/state.py`

```python
class AnalystState(TypedDict):
    """State for analyst workflow graph."""

    # Input
    input_path: str
    input_type: str  # "nprd" | "strategy_doc"

    # Parsed content
    raw_content: Dict[str, Any]
    transcript: str
    ocr_text: str
    keywords: List[str]

    # Extracted concepts
    strategy_name: str
    entry_conditions: List[str]
    exit_conditions: List[str]
    filters: List[str]
    time_filters: List[str]
    indicators: List[str]

    # KB search results
    kb_references: List[Dict[str, Any]]

    # Identified gaps
    missing_info: List[str]
    user_answers: Dict[str, str]

    # Output
    trd_content: str
    trd_path: str

    # Flow control
    current_step: str
    needs_user_input: bool
    completed: bool
```

---

## 6. Data Models

### 6.1 Configuration Data Model

**File:** `.analyst_config.json`

```json
{
  "$schema": "analyst_config_v1.json",
  "version": "1.0",
  "llm": {
    "provider": "openai",
    "model": "gpt-4o",
    "temperature": 0.1,
    "max_tokens": 2000,
    "retry_attempts": 3,
    "retry_delay": 1.0
  },
  "knowledge_base": {
    "chroma_path": "data/chromadb",
    "collection_name": "analyst_kb",
    "embedding_model": "all-MiniLM-L6-v2",
    "search_n_results": 5,
    "min_relevance_score": 0.6
  },
  "output": {
    "default_dir": "docs/trds",
    "filename_template": "{strategy_name}_{timestamp}.md",
    "timestamp_format": "%Y%m%d_%H%M%S"
  },
  "auto_mode": {
    "defaults": {
      "position_sizing": "risk_based",
      "max_risk_per_trade": "0.5%",
      "default_sl": "ATR 1.5",
      "default_tp": "1.5x risk",
      "trailing_stop": "Not specified"
    }
  },
  "validation": {
    "required_fields": [
      "entry_logic",
      "exit_logic",
      "risk_management",
      "filters",
      "indicators"
    ],
    "min_confidence_score": 0.7
  },
  "logging": {
    "level": "INFO",
    "file": "logs/analyst_agent.log",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  }
}
```

### 6.2 TRD Metadata Model

**YAML Frontmatter Schema:**

```yaml
---
strategy_name: string  # Required, max 100 chars
source: string  # Required, format: "NPRD: {video_id} | {title}" or "Strategy Doc: {path}"
generated_at: datetime  # Required, ISO 8601
status: enum  # Required: "draft" | "complete"
version: string  # Required, semver format
analyst_version: string  # Required, semver format
kb_collection: string  # Required
kb_articles_count: integer  # Required, >= 0
processing_time_seconds: float  # Optional
confidence_score: float  # Optional, 0.0-1.0
---
```

### 6.3 Gap Detection Model

**Gap Definition:**

```python
@dataclass
class InformationGap:
    """Represents a missing piece of information in TRD."""

    field_name: str  # Field identifier (e.g., "position_sizing")
    category: str  # Category: "critical" | "important" | "optional"
    description: str  # Human-readable description
    example: str  # Example value
    context: str  # Why this field is needed
    validation_rule: Optional[str]  # Regex or validation logic
    default_value: Optional[str]  # Default for auto mode
```

**Gap Categories:**

| Category | Description | Impact |
|----------|-------------|--------|
| **Critical** | Required for implementation | Cannot proceed without this |
| **Important** | Significantly improves implementation | Should be provided |
| **Optional** | Nice to have | Can be added later |

**Required Fields by Category:**

```python
REQUIRED_TRD_FIELDS = {
    "entry_logic": {
        "critical": [
            "entry_trigger",
            "entry_confirmation"
        ],
        "important": [
            "entry_timeframe",
            "entry_conditions_count"
        ]
    },
    "exit_logic": {
        "critical": [
            "take_profit",
            "stop_loss"
        ],
        "important": [
            "trailing_stop",
            "time_exit"
        ]
    },
    "risk_management": {
        "critical": [
            "position_sizing",
            "max_risk_per_trade"
        ],
        "important": [
            "daily_drawdown_limit",
            "max_concurrent_trades"
        ]
    },
    "filters": {
        "critical": [
            "time_filters"
        ],
        "important": [
            "volatility_filter",
            "spread_filter",
            "news_filter"
        ]
    },
    "indicators": {
        "critical": [
            "indicator_settings"
        ],
        "important": [
            "indicator_timeframes",
            "indicator_combinations"
        ]
    }
}
```

### 6.4 Knowledge Base Reference Model

```python
@dataclass
class KBReference:
    """Reference to a knowledge base article."""

    title: str
    url: str
    category: str
    relevance_score: float  # 0.0-1.0
    distance: float  # Vector distance
    preview: str  # Content preview (first 500 chars)
    key_insights: List[str]  # Extracted insights
    code_snippets: List[str]  # Relevant code examples

    def to_markdown(self) -> str:
        """Format as markdown for TRD."""
        ...
```

---

## 7. Error Handling

### 7.1 Error Categories

| Category | Severity | Recovery Strategy | User Action |
|----------|----------|-------------------|-------------|
| **Input Error** | ERROR | Abort with message | Fix input file |
| **LLM API Error** | ERROR | Retry 3x, then abort | Check API key, retry |
| **KB Search Error** | WARNING | Continue without KB | Check KB, retry |
| **File Write Error** | ERROR | Retry, save to temp | Check permissions |
| **Validation Error** | ERROR | Show specific issue | Fix data |
| **Configuration Error** | ERROR | Use defaults or abort | Fix config file |

### 7.2 Error Codes

| Code | Name | Description |
|------|------|-------------|
| `E001` | `INPUT_FILE_NOT_FOUND` | Input file does not exist |
| `E002` | `INPUT_INVALID_FORMAT` | Input file format is invalid |
| `E003` | `LLM_API_ERROR` | LLM API request failed |
| `E004` | `LLM_RATE_LIMIT` | LLM API rate limit exceeded |
| `E005` | `KB_NOT_ACCESSIBLE` | ChromaDB not accessible |
| `E006` | `KB_COLLECTION_NOT_FOUND` | analyst_kb collection not found |
| `E007` | `OUTPUT_WRITE_FAILED` | Cannot write TRD file |
| `E008` | `CONFIG_INVALID` | Configuration file is invalid |
| `E009` | `EXTRACTION_FAILED` | LLM extraction returned invalid data |
| `E010` | `VALIDATION_FAILED` | Generated TRD failed validation |

### 7.3 Error Messages

**Format:** `{ERROR_CODE}: {Human-readable message}\n{Suggestion}`

**Examples:**

```
E001: INPUT_FILE_NOT_FOUND
The input file 'outputs/videos/missing.json' does not exist.

Suggestion:
  - Run 'python tools/analyst_cli.py --list' to see available files
  - Check the file path is correct
  - Generate NPRD output first if needed
```

```
E003: LLM_API_ERROR
Failed to connect to OpenAI API: Connection timeout (attempt 1/3)

Suggestion:
  - Check your internet connection
  - Verify OPENAI_API_KEY is set in .env
  - The system will retry automatically (2 attempts remaining)
```

```
E009: EXTRACTION_FAILED
LLM extraction returned invalid JSON: Expected 'entry_conditions' array

Suggestion:
  - The source text may be too vague
  - Try editing the source to be more specific
  - Run with '--verbose' to see extraction details
```

### 7.4 Logging Strategy

**Log Levels:**
- `DEBUG` - Detailed execution flow
- `INFO` - Key steps and progress
- `WARNING` - Non-critical issues (e.g., KB search failed)
- `ERROR` - Failures that stop execution

**Log Format:**
```
YYYY-MM-DD HH:MM:SS - LEVEL - analyst_agent.module - Message
```

**Log Locations:**
- Console: INFO and above (colored with Rich)
- File: All levels to `logs/analyst_agent.log`
- Rotation: Daily, keep 7 days

**Key Events to Log:**
| Event | Level | Context |
|-------|-------|---------|
| CLI started | INFO | Command, arguments |
| File parsed | INFO | File path, size |
| LLM extraction started | INFO | Model, input length |
| LLM extraction completed | INFO | Output length, duration |
| KB search performed | INFO | Query, results count |
| Gaps identified | INFO | Missing fields count |
| User prompted | INFO | Field name |
| TRD generated | INFO | Output path |
| Error occurred | ERROR | Error code, message |

---

## 8. Security Requirements

### 8.1 API Key Management

**Requirements:**
- API keys MUST be stored in `.env` file (gitignored)
- `.env` file MUST have restricted permissions (600)
- API keys MUST NOT be hardcoded in source code
- API keys MUST NOT be logged or displayed in output
- System MUST validate API key format on startup

**Supported Providers:**
| Provider | Environment Variable | Validation |
|----------|---------------------|------------|
| OpenAI | `OPENAI_API_KEY` | Starts with `sk-` |
| Anthropic | `ANTHROPIC_API_KEY` | Starts with `sk-ant-` |

### 8.2 Input Validation

**File Path Validation:**
- MUST prevent directory traversal attacks
- MUST restrict to project directory or user-specified paths
- MUST validate file extensions (`.json`, `.md`)

**Content Validation:**
- MUST sanitize user input to prevent injection attacks
- MUST limit input size to prevent DoS (max 10MB files)
- MUST validate JSON structure before processing

**LLM Prompt Injection:**
- MUST validate prompt templates before use
- MUST not include untrusted user input in system prompts
- SHOULD use prompt sanitization libraries

### 8.3 Output Security

**File Write Permissions:**
- MUST create output directories with secure permissions (755)
- MUST create TRD files with appropriate permissions (644)
- MUST verify write path is within allowed directories

**Data Sanitization:**
- MUST escape special characters in markdown output
- MUST validate URLs in KB references
- MUST not expose system information in TRDs

### 8.4 Knowledge Base Security

**Access Control:**
- ChromaDB instance MUST be local-only (no network exposure)
- MUST validate collection names before access
- MUST not expose full database contents in error messages

**Data Integrity:**
- SHOULD validate checksums of KB articles
- MUST handle corrupted KB data gracefully

---

## 9. Testing Requirements

### 9.1 Unit Testing

**Coverage Target:** 80% minimum

**Test Categories:**

| Component | Tests Required | Key Scenarios |
|-----------|----------------|---------------|
| **File Parsing** | 5+ tests | Valid NPRD, invalid JSON, missing fields, large files, empty files |
| **Extraction Chain** | 5+ tests | Valid extraction, LLM failure, malformed JSON, timeout, retry logic |
| **KB Search** | 5+ tests | Valid search, empty results, KB unavailable, filtering, deduplication |
| **Gap Detection** | 10+ tests | All required fields present, partial missing, all missing, edge cases |
| **TRD Generation** | 5+ tests | Valid state, missing fields, special characters, large content |
| **CLI Commands** | 3+ tests per command | Valid args, invalid args, help flag |

**Example Test:**

```python
def test_extract_concepts_from_transcript():
    """Test trading concept extraction from transcript."""
    transcript = """
    The ORB strategy enters when price breaks the opening range.
    Use a 15-pip stop loss and take profit at 1.5x risk.
    Only trade during London session, avoid news events.
    """

    result = extract_concepts(transcript)

    assert "entry_conditions" in result
    assert len(result["entry_conditions"]) > 0
    assert "orb" in str(result["entry_conditions"]).lower()
    assert "stop_loss" in result
    assert result["stop_loss"] == "15 pips"
```

### 9.2 Integration Testing

**Test Scenarios:**

1. **End-to-End TRD Generation:**
   - Input: Valid NPRD JSON file
   - Process: Full workflow execution
   - Output: Valid TRD markdown file
   - Validation: All sections present, valid markdown

2. **Knowledge Base Integration:**
   - Setup: Ensure ChromaDB is running with analyst_kb
   - Action: Perform semantic search
   - Validation: Relevant articles returned

3. **LLM Integration:**
   - Setup: Configure API keys
   - Action: Call extraction chain
   - Validation: Structured JSON returned

4. **Interactive Mode:**
   - Setup: Provide input with gaps
   - Action: Simulate user input
   - Validation: Gaps filled correctly

5. **Auto Mode:**
   - Setup: Provide input with gaps
   - Action: Run with --auto flag
   - Validation: Default values used

### 9.3 Performance Testing

**Benchmarks:**

| Operation | Target | Measurement |
|-----------|--------|-------------|
| CLI startup | < 2s | Time to first prompt |
| File parsing | < 3s | 10MB NPRD file |
| LLM extraction | < 10s | 5000 word transcript |
| KB search | < 500ms | Single query |
| TRD generation | < 30s | End-to-end |

**Load Testing:**
- Process 10 files sequentially
- Measure memory usage throughout
- Verify no memory leaks

### 9.4 Acceptance Testing

**Test Cases:**

| ID | Scenario | Expected Result | Priority |
|----|----------|-----------------|----------|
| TC-1 | Generate TRD from valid NPRD | Valid TRD file created | P0 |
| TC-2 | Generate TRD from strategy doc | Valid TRD file created | P0 |
| TC-3 | Generate with missing info | Prompts for gaps or uses defaults | P0 |
| TC-4 | Generate in auto mode | TRD created with defaults | P1 |
| TC-5 | List available outputs | Shows all NPRD files | P2 |
| TC-6 | Show KB stats | Displays collection info | P3 |
| TC-7 | Invalid input file | Clear error message | P0 |
| TC-8 | LLM API failure | Retry then graceful error | P0 |
| TC-9 | KB unavailable | Continues without KB | P1 |
| TC-10 | Complete incomplete TRD | Updates TRD with user input | P1 |

---

## 10. Deployment Requirements

### 10.1 Environment Requirements

**System Requirements:**
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **OS** | Linux, macOS, Windows 10+ | Ubuntu 22.04 LTS |
| **Python** | 3.10 | 3.11 |
| **RAM** | 4 GB | 8 GB |
| **Disk** | 1 GB free | 5 GB free |
| **Network** | Required for LLM APIs | Stable connection |

**Dependencies:**
See `requirements.txt` in TRD document

### 10.2 Installation

**Automated Setup:**
```bash
./setup.sh
```

**Manual Setup:**
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
cat > .env <<EOF
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
EOF

# Create knowledge base collection
python scripts/create_analyst_kb.py
```

### 10.3 Configuration

**Required Configuration Files:**
1. `.env` - API keys and environment variables
2. `.analyst_config.json` - Analyst Agent settings (optional, will use defaults)
3. `data/chromadb/` - ChromaDB persistence directory

**Configuration Validation:**
- System MUST validate configuration on startup
- MUST create default config if not exists
- MUST warn about deprecated settings

### 10.4 Runtime Environment

**Startup Process:**
1. Load configuration files
2. Validate API keys
3. Initialize ChromaDB client
4. Initialize LLM clients
5. Display CLI interface

**Shutdown Process:**
1. Close database connections
2. Flush log buffers
3. Clean up temporary files
4. Exit with appropriate code

---

## 11. Acceptance Criteria

### 11.1 Functional Acceptance

| Criterion | Verification Method |
|-----------|---------------------|
| **Parse NPRD outputs** | Unit test with valid NPRD JSON |
| **Parse strategy documents** | Unit test with markdown file |
| **Extract trading concepts** | Integration test with LLM |
| **Search knowledge base** | Integration test with ChromaDB |
| **Identify gaps** | Unit test with various inputs |
| **Interactive prompts** | Manual testing |
| **Auto mode** | Integration test with --auto flag |
| **Generate TRD** | End-to-end test |
| **List inputs** | Manual testing |
| **Show stats** | Manual testing |

### 11.2 Non-Functional Acceptance

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| **CLI startup time** | < 2 seconds | Performance test |
| **TRD generation** | < 30 seconds | Performance test |
| **KB search latency** | < 500ms | Performance test |
| **Code quality** | PEP 8 compliant | Linting (pylint, flake8) |
| **Test coverage** | > 80% | Coverage report |
| **Documentation** | 100% public functions | Docstring check |

### 11.3 Definition of Done

A feature is **done** when:
- [ ] Code implements the specification
- [ ] Unit tests pass (100%)
- [ ] Integration tests pass
- [ ] Code review approved
- [ ] Documentation updated
- [ ] Performance targets met
- [ ] Security requirements validated
- [ ] Acceptance criteria verified

---

## 12. Glossary

| Term | Definition |
|------|------------|
| **Analyst Agent** | CLI tool for converting video/strategy content to TRDs |
| **ChromaDB** | Vector database for semantic search |
| **Gap** | Missing required information in TRD |
| **HITL** | Human-in-the-loop, requiring user input |
| **Knowledge Base (KB)** | Filtered collection of MQL5 articles |
| **LangChain** | Framework for LLM application development |
| **LangGraph** | State machine framework for complex workflows |
| **LLM** | Large Language Model (OpenAI GPT-4, Anthropic Claude) |
| **NPRD** | Non-Processing Research Data, video processing system |
| **Semantic Search** | Vector similarity search for meaning-based queries |
| **TRD** | Technical Requirements Document, structured strategy specification |
| **Vector Embedding** | Numerical representation of text for similarity search |

---

## Appendix A: Configuration Reference

### Default Configuration (.analyst_config.json)

```json
{
  "version": "1.0",
  "llm": {
    "provider": "openai",
    "model": "gpt-4o",
    "temperature": 0.1,
    "max_tokens": 2000,
    "retry_attempts": 3,
    "retry_delay": 1.0
  },
  "knowledge_base": {
    "chroma_path": "data/chromadb",
    "collection_name": "analyst_kb",
    "embedding_model": "all-MiniLM-L6-v2",
    "search_n_results": 5,
    "min_relevance_score": 0.6
  },
  "output": {
    "default_dir": "docs/trds",
    "filename_template": "{strategy_name}_{timestamp}.md",
    "timestamp_format": "%Y%m%d_%H%M%S"
  },
  "auto_mode": {
    "defaults": {
      "position_sizing": "risk_based",
      "max_risk_per_trade": "0.5%",
      "default_sl": "ATR 1.5",
      "default_tp": "1.5x risk"
    }
  },
  "logging": {
    "level": "INFO",
    "file": "logs/analyst_agent.log"
  }
}
```

---

## Appendix B: Troubleshooting Guide

### Common Issues

**Issue:** "E003: LLM_API_ERROR"
**Solution:**
- Check `OPENAI_API_KEY` is set in `.env`
- Verify API key has credits
- Check internet connection

**Issue:** "E005: KB_NOT_ACCESSIBLE"
**Solution:**
- Run `python scripts/create_analyst_kb.py`
- Check `data/chromadb/` directory exists
- Verify ChromaDB is installed

**Issue:** "E009: EXTRACTION_FAILED"
**Solution:**
- Run with `--verbose` to see details
- Check input file has valid content
- Try rephrasing content in strategy doc

---

## Appendix C: Migration Guide

### From v0 to v1.0

**Breaking Changes:** None (initial release)

**New Features:**
- Interactive mode
- Auto mode
- Knowledge base integration
- Gap detection

**Migration Steps:** None required for initial release

---

**End of Specification Document**

*This specification is version 1.0 and approved for implementation. All requirements are binding unless marked as optional or future enhancement.*

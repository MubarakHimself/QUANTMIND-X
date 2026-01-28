# System Architecture: Analyst Agent CLI

> **Component:** Analyst Agent CLI v1.0
> **Status:** Ready for Implementation
> **Last Updated:** 2026-01-27
> **Author:** System Architecture Designer

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Overview](#system-overview)
3. [Architecture Principles](#architecture-principities)
4. [Technology Stack](#technology-stack)
5. [System Architecture Diagram](#system-architecture-diagram)
6. [Deployment Architecture](#deployment-architecture)
7. [Quality Attributes](#quality-attributes)
8. [Architectural Decisions](#architectural-decisions)

---

## Executive Summary

### Purpose

The Analyst Agent CLI is a command-line tool that converts unstructured trading strategy content (from NPRD video outputs or strategy documents) into structured Technical Requirements Documents (TRDs). It leverages a filtered ChromaDB knowledge base and LangGraph for human-in-the-loop workflow orchestration.

### Key Characteristics

- **Input:** NPRD JSON outputs or strategy markdown documents
- **Processing:** LLM-based concept extraction, semantic search, gap identification
- **Output:** Structured TRD markdown files
- **Interaction:** Human-in-the-loop for missing information
- **Knowledge Base:** ChromaDB with ~330 filtered MQL5 articles

### Architectural Style

**Layered Architecture with State Machine Orchestration**

```
┌─────────────────────────────────────────────────────────┐
│                    CLI Layer                             │
│              (User Interaction)                          │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│              Orchestration Layer                         │
│         (LangGraph State Machine)                        │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│               Processing Layer                           │
│    (Extraction, Search, Generation Chains)               │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│                Data Layer                                │
│      (ChromaDB, File System, Configuration)              │
└─────────────────────────────────────────────────────────┘
```

---

## System Overview

### Core Functionalities

1. **Input Parsing**
   - NPRD JSON outputs with transcripts, OCR text, and keywords
   - Strategy documents in markdown format
   - Chunked file aggregation

2. **Concept Extraction**
   - Entry/exit conditions
   - Filters and time filters
   - Indicators and settings
   - Risk management parameters

3. **Knowledge Base Integration**
   - Semantic search over ChromaDB
   - Article relevance ranking
   - Reference extraction

4. **Gap Identification**
   - Required field validation
   - Missing information detection
   - Human-in-the-loop prompting

5. **TRD Generation**
   - Structured markdown output
   - YAML frontmatter
   - Knowledge base references

### System Boundaries

```
┌─────────────────────────────────────────────────────────┐
│                    Analyst Agent CLI                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   Input      │  │  Processing  │  │   Output     │ │
│  │  Handler     │  │   Engine     │  │  Generator   │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└────────┬──────────────────┬──────────────────┬─────────┘
         │                  │                  │
    ┌────▼────┐       ┌────▼────┐       ┌────▼────┐
    │  NPRD   │       │ChromaDB │       │   TRD   │
    │ Outputs │       │   KB    │       │  Files  │
    └─────────┘       └─────────┘       └─────────┘
```

### External Interfaces

| Interface | Type | Protocol | Direction |
|-----------|------|----------|-----------|
| NPRD JSON Input | File | JSON | Inbound |
| Strategy Doc Input | File | Markdown | Inbound |
| ChromaDB Query | Database | Native Python | Bidirectional |
| LLM API | Service | HTTP REST | Outbound |
| CLI Commands | Terminal | stdin/stdout | Bidirectional |
| TRD Output | File | Markdown | Outbound |

---

## Architecture Principles

### 1. Separation of Concerns

Each layer has a distinct responsibility:
- **CLI Layer:** User interaction and command routing
- **Orchestration Layer:** Workflow state management
- **Processing Layer:** Business logic and transformations
- **Data Layer:** Persistence and retrieval

### 2. State Machine Orchestration

LangGraph manages workflow state transitions:
- Explicit state definition
- Conditional routing based on gaps
- Human intervention checkpoints
- State persistence and recovery

### 3. Declarative Configuration

- YAML configuration files
- Environment variable overrides
- Click command-line options
- Prompt template externalization

### 4. Testability

- Dependency injection for LLM clients
- Mockable ChromaDB interface
- Isolated chain components
- Deterministic state transitions

### 5. Extensibility

- Plugin architecture for new input types
- Template-based TRD generation
- Modular chain components
- Configurable prompt templates

---

## Technology Stack

### Core Framework

| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.10+ | Primary language |
| **Click** | 8.1+ | CLI framework |
| **Pydantic** | 2.0+ | Data validation |

### Orchestration & AI

| Technology | Version | Purpose |
|------------|---------|---------|
| **LangChain** | 0.2+ | Chain composition |
| **LangGraph** | 0.2+ | State machine workflow |
| **OpenAI** | 1.0+ | LLM provider (primary) |
| **Anthropic** | 0.5+ | LLM provider (alternative) |

### Data & Storage

| Technology | Version | Purpose |
|------------|---------|---------|
| **ChromaDB** | 0.4+ | Vector database (local) |
| **YAML** | 6.0+ | Configuration |

### Utilities

| Technology | Version | Purpose |
|------------|---------|---------|
| **python-dotenv** | 1.0+ | Environment variables |
| **Rich** | 13.0+ | CLI formatting |
| **pyfiglet** | 1.0+ | ASCII art banners |

---

## System Architecture Diagram

### High-Level Architecture

```
┌───────────────────────────────────────────────────────────────────────┐
│                          Analyst Agent CLI                            │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                        CLI Layer                                │ │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐                │ │
│  │  │  Commands  │  │ Interface  │  │  Config    │                │ │
│  │  └────────────┘  └────────────┘  └────────────┘                │ │
│  └────────────────────────────┬────────────────────────────────────┘ │
│                               │                                       │
│  ┌────────────────────────────▼────────────────────────────────────┐ │
│  │                   Orchestration Layer                           │ │
│  │  ┌────────────────────────────────────────────────────────────┐ │ │
│  │  │              LangGraph State Machine                       │ │ │
│  │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │ │ │
│  │  │  │   Parse  │─►│ Extract  │─►│  Search  │─►│ Identify │  │ │ │
│  │  │  │  Input   │  │ Concepts │  │    KB    │  │   Gaps   │  │ │ │
│  │  │  └──────────┘  └──────────┘  └──────────┘  └────┬─────┘  │ │ │
│  │  │                                                  │         │ │ │
│  │  │                                      ┌───────────┴─────┐   │ │ │
│  │  │                                      │                 │   │ │ │
│  │  │                              ┌───────▼──────┐  ┌─────▼─────┐│ │ │
│  │  │                              │   Ask User   │  │ Generate  ││ │ │
│  │  │                              │   (HITL)     │  │    TRD    ││ │ │
│  │  │                              └───────┬──────┘  └─────┬─────┘│ │ │
│  │  │                                      └───────────┬─────┘   │ │ │
│  │  │                                                  │         │ │ │
│  │  │                                                  ▼         │ │ │
│  │  │                                          ┌───────────┐     │ │ │
│  │  │                                          │    END    │     │ │ │
│  │  │                                          └───────────┘     │ │ │
│  │  └────────────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────┬────────────────────────────────────┘ │
│                               │                                       │
│  ┌────────────────────────────▼────────────────────────────────────┐ │
│  │                     Processing Layer                             │ │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │ │
│  │  │  Extraction     │  │  Search         │  │  Generation     │ │ │
│  │  │  Chain          │  │  Chain          │  │  Chain          │ │ │
│  │  │  - Concepts     │  │  - Queries      │  │  - Template     │ │ │
│  │  │  - Entities     │  │  - Ranking      │  │  - Formatting   │ │ │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘ │ │
│  │  ┌─────────────────┐  ┌─────────────────┐                      │ │
│  │  │  Gap Detector   │  │  Prompt         │                      │ │
│  │  │  - Validation   │  │  Templates      │                      │ │
│  │  └─────────────────┘  └─────────────────┘                      │ │
│  └────────────────────────────┬────────────────────────────────────┘ │
│                               │                                       │
│  ┌────────────────────────────▼────────────────────────────────────┐ │
│  │                        Data Layer                                │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │ │
│  │  │  ChromaDB    │  │  File System │  │  Config      │         │ │
│  │  │  Client      │  │  - NPRD JSON │  │  - YAML       │         │ │
│  │  │  - analyst_kb│  │  - TRD MD    │  │  - .env       │         │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘         │ │
│  └─────────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────────┘
```

### Component Interaction Diagram

```
┌─────────┐
│  User   │
└────┬────┘
     │ click
     ▼
┌─────────────────────────────────────────────────────────────┐
│                         CLI                                   │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │ generate  │    │   list   │    │  stats   │              │
│  │  command  │    │ command  │    │ command  │              │
│  └─────┬────┘    └─────┬────┘    └─────┬────┘              │
└────────┼───────────────┼───────────────┼─────────────────────┘
         │               │               │
         ▼               │               │
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph Workflow                        │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│  │  Parse  │─►│ Extract │─►│  Search │─►│ Identify│        │
│  │  Node   │  │   Node  │  │   Node  │  │   Node  │        │
│  └─────────┘  └─────────┘  └─────────┘  └────┬────┘        │
│                                          │                   │
│                           ┌──────────────┴───────┐           │
│                           │                      │           │
│                    ┌──────▼──────┐         ┌────▼─────┐     │
│                    │   Ask User  │         │ Generate │     │
│                    │    Node     │         │   Node   │     │
│                    └──────┬──────┘         └────┬─────┘     │
│                           └──────────────┬───────┘           │
│                                          │                   │
└──────────────────────────────────────────┼───────────────────┘
                                           │
         ┌─────────────────────────────────┼───────────────────┐
         │                                 │                   │
         ▼                                 ▼                   ▼
┌─────────────────┐             ┌─────────────────┐   ┌───────────────┐
│  ChromaDB KB    │             │   LLM APIs      │   │ File System   │
│  - analyst_kb   │             │  - OpenAI       │   │ - NPRD JSON   │
│  - mql5_knowledge│            │  - Anthropic    │   │ - TRD MD      │
└─────────────────┘             └─────────────────┘   └───────────────┘
```

---

## Deployment Architecture

### Local Development Environment

```
┌─────────────────────────────────────────────────────────┐
│                   Developer Machine                      │
│                                                          │
│  ┌───────────────────────────────────────────────────┐ │
│  │            Virtual Environment (venv)             │ │
│  │                                                    │ │
│  │  tools/                                            │ │
│  │  ├── analyst_cli.py (entry point)                 │ │
│  │  └── analyst_agent/                                │ │
│  │      ├── cli/                                      │ │
│  │      ├── graph/                                    │ │
│  │      ├── chains/                                   │ │
│  │      ├── kb/                                       │ │
│  │      ├── prompts/                                  │ │
│  │      └── utils/                                    │ │
│  │                                                    │ │
│  │  data/                                             │ │
│  │  └── chromadb/                                     │ │
│  │      ├── chroma.sqlite3                           │ │
│  │      └── d253458f-085e-47e2-a077-1969e390fc84/   │ │
│  │                                                    │ │
│  │  outputs/                                          │ │
│  │  └── videos/ (NPRD JSON outputs)                   │ │
│  │                                                    │ │
│  │  docs/                                             │ │
│  │  └── trds/ (generated TRD files)                   │ │
│  └───────────────────────────────────────────────────┘ │
│                                                          │
│  External Services:                                     │
│  ┌──────────────┐        ┌──────────────┐              │
│  │ OpenAI API   │        │ Anthropic API │              │
│  │ (gpt-4o)     │        │ (claude-3.5)  │              │
│  └──────────────┘        └──────────────┘              │
└─────────────────────────────────────────────────────────┘
```

### Data Flow Architecture

```
┌──────────────┐
│ NPRD Output  │ (JSON file)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ File Parser  │ (Read & validate)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   LangGraph  │ (State machine)
│  Workflow    │
└──────┬───────┘
       │
       ├──────────────────────────────────────┐
       │                                      │
       ▼                                      ▼
┌──────────────┐                      ┌──────────────┐
│Extraction    │                      │   ChromaDB   │
│Chain (LLM)   │                      │   Query      │
└──────┬───────┘                      └──────┬───────┘
       │                                     │
       │                                     ▼
       │                            ┌──────────────┐
       │                            │  Reference   │
       │                            │  Ranking     │
       │                            └──────┬───────┘
       │                                     │
       └──────────────┬──────────────────────┘
                      ▼
              ┌──────────────┐
              │    Gap       │
              │  Detection   │
              └──────┬───────┘
                     │
         ┌───────────┴───────────┐
         │                       │
         ▼                       ▼
┌──────────────┐         ┌──────────────┐
│  HITL Prompt │         │    TRD       │
│  (Interactive│         │  Generation  │
│   or Auto)  │         └──────┬───────┘
└──────────────┘                │
                                ▼
                      ┌──────────────┐
                      │  TRD File    │
                      │  (Markdown)  │
                      └──────────────┘
```

---

## Quality Attributes

### Performance

| Attribute | Target | Strategy |
|-----------|--------|----------|
| CLI Startup | < 2 seconds | Lazy loading, async initialization |
| TRD Generation | < 30 seconds | Parallel chain execution |
| KB Search | < 500ms | ChromaDB indexing, result caching |
| Memory Usage | < 500MB | Streaming, chunk processing |

### Reliability

| Attribute | Target | Strategy |
|-----------|--------|----------|
| Error Handling | Graceful | Try-catch with helpful messages |
| State Recovery | Persisted | LangGraph checkpointing |
| Data Validation | Strict | Pydantic schemas |
| LLM Fallback | Automatic | Provider switching |

### Maintainability

| Attribute | Target | Strategy |
|-----------|--------|----------|
| Code Coverage | 70%+ | Unit & integration tests |
| Type Hints | 90%+ | mypy compliance |
| Documentation | 100% | Docstrings on all functions |
| Modularity | High | Single responsibility principle |

### Scalability

| Attribute | Target | Strategy |
|-----------|--------|----------|
| Concurrent Users | 1 (CLI) | Single-user design |
| KB Size | 1M+ articles | ChromaDB scalability |
| TRD Generation | Batch mode | Queue-based processing |
| Future Extensions | Plugin-ready | Abstract interfaces |

### Security

| Attribute | Target | Strategy |
|-----------|--------|----------|
| API Keys | Environment | .env file, gitignored |
| Data Privacy | Local-only | No external data transmission |
| Input Validation | Strict | Schema validation |
| Injection Prevention | Parameterized | No raw SQL/queries |

---

## Architectural Decisions (ADR)

### ADR-001: Use ChromaDB for Knowledge Base

**Status:** Accepted

**Context:**
- Existing ChromaDB installation with 1,805 MQL5 articles
- Need filtered collection (~330 articles) for analyst use
- Local deployment preferred

**Decision:**
- Use existing ChromaDB at `data/chromadb/`
- Create filtered `analyst_kb` collection
- No migration to Qdrant for v1.0

**Consequences:**
- + No data migration needed
- + Faster implementation
- - ChromaDB less scalable than Qdrant
- - Can mitigate in v2.0 if needed

### ADR-002: LangGraph for Workflow Orchestration

**Status:** Accepted

**Context:**
- Need state machine with conditional routing
- Human-in-the-loop integration required
- Checkpoint and recovery capability

**Decision:**
- Use LangGraph StateGraph for workflow
- Explicit state definition (TypedDict)
- Conditional edges for gap detection

**Consequences:**
- + Clear state transitions
- + Built-in checkpointing
- + Human intervention support
- - Learning curve for LangGraph
- - Debugging complexity

### ADR-003: Click Framework for CLI

**Status:** Accepted

**Context:**
- Need robust CLI interface
- Interactive and batch modes
- Command composition and groups

**Decision:**
- Use Click 8.1+ for CLI framework
- Command groups: generate, list, stats
- Rich integration for formatting

**Consequences:**
- + Declarative command definitions
- + Built-in help and validation
- + Easy testing
- - Additional dependency

### ADR-004: Template-Based TRD Generation

**Status:** Accepted

**Context:**
- Need consistent TRD structure
- User may want customization
- Multiple output formats in future

**Decision:**
- Jinja2-style template filling
- Template externalization
- YAML frontmatter for metadata

**Consequences:**
- + Easy to modify structure
- + Consistent formatting
- + Future format support
- - Template maintenance overhead

### ADR-005: Human-in-the-Loop by Default

**Status:** Accepted

**Context:**
- NPRD outputs often incomplete
- Need accurate TRDs for implementation
- User wants control over gaps

**Decision:**
- Interactive mode default
- Auto mode optional (--auto flag)
- Graceful degradation with defaults

**Consequences:**
- + Higher quality TRDs
- + User control
- - Slower workflow
- - Not suitable for automation

---

## Future Architecture Considerations

### v2.0 Enhancements

1. **Risk Management Integration**
   - RiskMind component validation
   - Prop firm rule checking
   - Position sizing module

2. **Multi-Strategy Batch Processing**
   - Queue-based architecture
   - Parallel TRD generation
   - Progress tracking

3. **TRD Comparison & Diff**
   - Version control integration
   - Strategy diff tool
   - Merge conflict resolution

### v3.0 IDE Integration

1. **VS Code Extension**
   - Language server for TRD files
   - Inline validation
   - Autocomplete from KB

2. **Web Dashboard**
   - TRD repository browser
   - Strategy comparison UI
   - Collaboration features

---

## Appendix: Architecture Viewpoints

### Stakeholder Viewpoint

| Stakeholder | Concern | Addressed By |
|-------------|---------|-------------|
| Mubarak (User) | Easy CLI usage | Click framework, interactive prompts |
| Developer | Maintainable code | Modular design, type hints |
| QA | Testability | Dependency injection, mocks |
| Future Dev | Extensibility | Plugin architecture, templates |

### Functional Viewpoint

**Main Functions:**
1. Parse NPRD/strategy inputs
2. Extract trading concepts
3. Search knowledge base
4. Identify missing information
5. Generate TRD documents

**Data Flows:**
- NPRD → Parser → LangGraph → LLM Chains → ChromaDB → TRD
- User → CLI → HITL → State Update → TRD

### Information Viewpoint

**Key Data Structures:**
- `AnalystState`: LangGraph state (TypedDict)
- `NPRDOutput`: Input schema (Pydantic)
- `TRDDocument`: Output schema (YAML + Markdown)
- `KBReference`: Search result schema

**Data Stores:**
- ChromaDB: Vector embeddings, metadata
- File System: NPRD JSON, TRD Markdown
- Config: YAML, environment variables

### Concurrency Viewpoint

**Single-threaded by design:**
- CLI runs synchronously
- LangGraph sequential nodes
- Future: Parallel KB searches

**Future Multi-threading:**
- Batch mode: Process multiple NPRDs
- Parallel chain execution
- Async LLM calls

### Deployment Viewpoint

**Current: Local CLI**
```bash
python tools/analyst_cli.py generate --input output.json
```

**Future: Containerized Service**
```yaml
services:
  analyst-agent:
    build: ./tools/analyst_agent
    volumes:
      - ./data:/app/data
      - ./outputs:/app/outputs
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
```

---

**Document Version:** 1.0
**Last Modified:** 2026-01-27
**Next Review:** Post-implementation

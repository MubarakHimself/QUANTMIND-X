# Data Flow: Analyst Agent CLI

> **Component:** Analyst Agent CLI v1.0
> **Status:** Ready for Implementation
> **Last Updated:** 2026-01-27
> **Author:** System Architecture Designer

---

## Table of Contents

1. [Data Flow Overview](#data-flow-overview)
2. [End-to-End Data Flow](#end-to-end-data-flow)
3. [State Transitions](#state-transitions)
4. [Data Transformations](#data-transformations)
5. [Integration Points](#integration-points)
6. [Error Handling Flows](#error-handling-flows)

---

## Data Flow Overview

### High-Level Flow

```
┌──────────────┐
│   User Input │ (NPRD JSON or Strategy MD)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   CLI Layer  │ (Commands & Interface)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  LangGraph   │ (State Machine)
│  Workflow    │
└──────┬───────┘
       │
       ├──────────────┬──────────────┬──────────────┐
       │              │              │              │
       ▼              ▼              ▼              ▼
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│   LLM    │  │ChromaDB  │  │  File    │  │   User   │
│   API    │  │   KB     │  │ System   │  │ (HITL)   │
└──────────┘  └──────────┘  └──────────┘  └──────────┘
       │              │              │              │
       └──────────────┴──────────────┴──────────────┘
                           │
                           ▼
                    ┌──────────┐
                    │   TRD    │ (Markdown File)
                    │  Output  │
                    └──────────┘
```

### Data Flow Types

| Flow Type | Source | Destination | Data Format | Volume |
|-----------|--------|-------------|-------------|--------|
| Input Flow | User/File | CLI | JSON/Markdown | Small (<1MB) |
| Extraction Flow | Transcript | LLM | Text → JSON | Medium |
| Search Flow | Concepts | ChromaDB | Text → Vectors | Small |
| Generation Flow | State | LLM | JSON → Markdown | Large |
| Output Flow | Generator | File | Markdown | Medium |
| HITL Flow | Gap Detector | User | Text prompts | Small |

---

## End-to-End Data Flow

### Scenario 1: NPRD JSON Input (Complete)

```
Step 1: User Input
┌─────────────────────────────────────────────────────┐
│ User runs:                                          │
│ analyst-cli generate --input outputs/videos/abc.json │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
Step 2: File Parsing
┌─────────────────────────────────────────────────────┐
│ File I/O: parse_nprd_output()                       │
│ - Read JSON file                                    │
│ - Validate schema                                   │
│ - Extract: video_id, video_title, chunks, summary   │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
Step 3: State Initialization
┌─────────────────────────────────────────────────────┐
│ Initial State:                                      │
│ {                                                   │
│   "input_path": "outputs/videos/abc.json",          │
│   "input_type": "nprd",                             │
│   "transcript": "full transcript text...",          │
│   "keywords": ["ORB", "breakout", ...],             │
│   "current_step": "start"                           │
│ }                                                   │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
Step 4: Concept Extraction (LLM)
┌─────────────────────────────────────────────────────┐
│ Extraction Chain:                                   │
│ Input: transcript text                              │
│ Process: Send to LLM with extraction prompt         │
│ Output: JSON with extracted concepts                │
│ {                                                   │
│   "strategy_name": "ORB Breakout Scalper",          │
│   "entry_conditions": [                             │
│     "price breaks opening range high",              │
│     "volume confirms breakout"                      │
│   ],                                                │
│   "exit_conditions": [                              │
│     "take profit at 1.5x risk",                     │
│     "stop loss at opening range low"                │
│   ],                                                │
│   "filters": ["london session only"],               │
│   "time_filters": ["first 2 hours of session"],     │
│   "indicators_used": ["Volume", "ATR"],             │
│   "mentioned_concepts": ["opening range break"]     │
│ }                                                   │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
Step 5: Knowledge Base Search
┌─────────────────────────────────────────────────────┐
│ Search Chain:                                       │
│ Input: extracted concepts                           │
│ Process:                                            │
│   1. Generate 10 search queries                     │
│   2. Query ChromaDB for each                        │
│   3. Rank by relevance score                        │
│   4. Deduplicate by title                           │
│ Output: Top 10 relevant articles                    │
│ {                                                   │
│   "kb_references": [                                │
│     {                                               │
│       "title": "Opening Range Breakout Strategy",   │
│       "relevance_score": 0.89,                      │
│       "categories": "Trading Systems",              │
│       "content_preview": "The opening range..."     │
│     },                                              │
│     ...                                             │
│   ]                                                 │
│ }                                                   │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
Step 6: Gap Detection
┌─────────────────────────────────────────────────────┐
│ Gap Detector:                                       │
│ Input: extracted concepts + KB references           │
│ Process: Check required fields against content      │
│ Output: List of missing information                 │
│ {                                                   │
│   "missing_info": [],  # Empty = no gaps            │
│   "needs_user_input": false                         │
│ }                                                   │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
Step 7: TRD Generation (Skip HITL - no gaps)
┌─────────────────────────────────────────────────────┐
│ Generation Chain:                                   │
│ Input: Complete state with all data                 │
│ Process:                                            │
│   1. Generate TRD sections via LLM                 │
│   2. Format with markdown template                  │
│   3. Add YAML frontmatter                           │
│ Output: Complete TRD markdown                       │
│ ---                                                 │
│ strategy_name: "ORB Breakout Scalper"               │
│ source: "NPRD: abc123"                              │
│ generated_at: "2026-01-27T10:30:00Z"                │
│ ---                                                 │
│ # Trading Strategy: ORB Breakout Scalper           │
│ ...                                                 │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
Step 8: File Output
┌─────────────────────────────────────────────────────┐
│ File I/O: save_trd()                                │
│ Input: TRD markdown content                         │
│ Process:                                            │
│   1. Generate filename: orb_breakout_scalper_...md  │
│   2. Create docs/trds/ directory                    │
│   3. Write file with UTF-8 encoding                 │
│ Output: File saved successfully                     │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
Step 9: Completion
┌─────────────────────────────────────────────────────┐
│ CLI Output:                                         │
│ ✅ Generating TRD...                                │
│ ✅ Saved to: docs/trds/orb_breakout_scalper_...md   │
│                                                     │
│ Next steps:                                         │
│ - Review the TRD file                               │
│ - Run: analyst-cli --complete [file] (if needed)    │
└─────────────────────────────────────────────────────┘
```

### Scenario 2: NPRD JSON Input (Incomplete - HITL)

```
Step 1-5: Same as Scenario 1
...
                     │
                     ▼
Step 6: Gap Detection (Gaps Found)
┌─────────────────────────────────────────────────────┐
│ Gap Detector:                                       │
│ Output:                                             │
│ {                                                   │
│   "missing_info": [                                 │
│     "exit_logic: stop loss placement",              │
│     "risk_management: position sizing method",      │
│     "indicators: indicator settings"                │
│   ],                                                │
│   "needs_user_input": true                          │
│ }                                                   │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
Step 7: Human-in-the-Loop
┌─────────────────────────────────────────────────────┐
│ Interactive Prompts:                                │
│                                                     │
│ ⚠️  Missing information detected:                  │
│                                                     │
│ 1. Stop loss placement                              │
│    Enter value (or 'skip'): ATR 1.5                 │
│    ✓ Saved: ATR 1.5                                │
│                                                     │
│ 2. Position sizing method                           │
│    Enter value (or 'skip'): risk_based              │
│    ✓ Saved: risk_based                             │
│                                                     │
│ 3. Indicator settings                               │
│    Enter value (or 'skip'): skip                    │
│    ⏭️  Skipped                                      │
│                                                     │
│ {                                                   │
│   "user_answers": {                                 │
│     "exit_logic: stop loss placement": "ATR 1.5",   │
│     "risk_management: position sizing": "risk_based"│
│   }                                                 │
│ }                                                   │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
Step 8: TRD Generation (with user answers)
┌─────────────────────────────────────────────────────┐
│ Generation Chain:                                   │
│ Input: State + user_answers                         │
│ Process: Include user answers in TRD                │
│ Output: TRD with filled gaps                        │
│                                                     │
│ ## Exit Logic                                       │
│ ...                                                 │
│ ### Stop Loss                                       │
│ - **Strategy:** ATR-based                           │
│ - **Setting:** 1.5x ATR from entry                  │
│                                                     │
│ ## Position Sizing & Risk Management               │
│ - **Method:** Risk-based (user specified)           │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
Step 9: File Output & Completion
(Same as Scenario 1)
```

### Scenario 3: Strategy Document Input

```
Step 1: User Input
┌─────────────────────────────────────────────────────┐
│ User runs:                                          │
│ analyst-cli generate \                              │
│   --input strategy_notes.md \                       │
│   --type strategy_doc                               │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
Step 2: File Parsing
┌─────────────────────────────────────────────────────┐
│ File I/O: parse_strategy_doc()                      │
│ - Read markdown file                                │
│ - Return full text content                          │
│ Output:                                             │
│ {                                                   │
│   "transcript": "# RSI Oversold Strategy\n\n...",   │
│   "raw_content": {"content": "..."}                 │
│ }                                                   │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
Step 3-9: Same as Scenario 1
(Extraction, Search, Gap Detection, etc.)
```

---

## State Transitions

### LangGraph State Machine

```
                    ┌─────────────┐
                    │   START     │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  PARSE      │  parse_input_node()
                    │  INPUT      │  state["current_step"] = "parsed"
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  EXTRACT    │  extract_concepts_node()
                    │  CONCEPTS   │  state["current_step"] = "extracted"
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  SEARCH KB  │  search_kb_node()
                    │             │  state["current_step"] = "searched"
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │ IDENTIFY    │  identify_gaps_node()
                    │    GAPS     │  state["current_step"] = "gaps_identified"
                    └──────┬──────┘
                           │
                  ┌────────┴────────┐
                  │                 │
            has_gaps()         has_gaps()
            returns True       returns False
                  │                 │
                  ▼                 ▼
           ┌─────────────┐   ┌─────────────┐
           │   ASK USER  │   │  GENERATE   │
           │   (HITL)    │   │    TRD      │
           └──────┬──────┘   └──────┬──────┘
                  │                 │
                  │     ┌───────────┘
                  │     │
                  ▼     ▼
           ┌─────────────┐
           │  GENERATE   │  generate_trd_node()
           │    TRD      │  state["completed"] = True
           └──────┬──────┘
                  │
                  ▼
           ┌─────────────┐
           │     END     │
           └─────────────┘
```

### State Evolution Table

| Step | Node | State Fields Updated | Key Changes |
|------|------|---------------------|-------------|
| Start | Initializer | `input_path`, `input_type` | Load input configuration |
| 1 | parse_input_node | `raw_content`, `transcript`, `keywords` | File loaded and parsed |
| 2 | extract_concepts_node | `strategy_name`, `entry_conditions`, `exit_conditions`, `filters`, `indicators_used` | Concepts extracted via LLM |
| 3 | search_kb_node | `kb_references`, `relevant_articles` | KB searched and ranked |
| 4 | identify_gaps_node | `missing_info`, `needs_user_input` | Gaps detected |
| 5a | ask_user_node | `user_answers` | User provided missing info (conditional) |
| 5b | generate_trd_node | `trd_content`, `trd_path`, `completed` | TRD generated and saved |

### State Schema Evolution

```python
# Initial State
{
    "input_path": "outputs/videos/abc.json",
    "input_type": "nprd",
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

# After parse_input_node
{
    "raw_content": {...},  # Added
    "transcript": "Full text...",
    "keywords": ["ORB", "breakout"],
    "current_step": "parsed"
}

# After extract_concepts_node
{
    "strategy_name": "ORB Breakout",  # Added
    "entry_conditions": ["breakout confirmed"],  # Added
    "exit_conditions": ["1.5x risk"],  # Added
    "filters": ["london session"],  # Added
    "time_filters": ["first 2 hours"],  # Added
    "indicators_used": ["ATR"],  # Added
    "mentioned_concepts": ["opening range"],  # Added
    "current_step": "extracted"
}

# After search_kb_node
{
    "kb_references": [{...}, {...}],  # Added
    "relevant_articles": ["Article 1", "Article 2"],  # Added
    "current_step": "searched"
}

# After identify_gaps_node
{
    "missing_info": ["stop loss placement"],  # Added
    "needs_user_input": True,  # Changed
    "current_step": "gaps_identified"
}

# After ask_user_node
{
    "user_answers": {"stop loss": "ATR 1.5"},  # Added
    "current_step": "user_input_received"
}

# After generate_trd_node
{
    "trd_content": "---\nstrategy_name: ...",  # Added
    "trd_path": "docs/trds/orb_breakout_...md",  # Added
    "completed": True,  # Changed
    "current_step": "completed"
}
```

---

## Data Transformations

### Transformation 1: NPRD JSON → AnalystState

**Input Format:**
```json
{
  "video_id": "abc123",
  "video_title": "ORB Strategy Explained",
  "chunks": [
    {
      "transcript": "In this video...",
      "ocr_text": "RSI: 45.2",
      "keywords": ["ORB", "breakout"]
    }
  ],
  "summary": {
    "full_transcript": "Combined transcript...",
    "all_keywords": ["ORB", "breakout", "london"]
  }
}
```

**Transformation Logic:**
```python
def parse_input_node(state: AnalystState) -> AnalystState:
    # Extract from NPRD structure
    data = json.loads(Path(state["input_path"]).read_text())

    state["raw_content"] = data
    state["transcript"] = data["summary"]["full_transcript"]
    state["keywords"] = data["summary"]["all_keywords"]
    state["current_step"] = "parsed"

    return state
```

**Output:**
```python
{
    "transcript": "Combined transcript...",
    "keywords": ["ORB", "breakout", "london"],
    "raw_content": {...}
}
```

---

### Transformation 2: Transcript → Extracted Concepts

**Input:**
```python
{
    "transcript": "The ORB strategy works by waiting for the first
                  30 minutes of the London session to form a range.
                  When price breaks above the range high with volume
                  confirmation, we enter long..."
}
```

**Transformation (LLM Extraction):**
```python
# Send to LLM with extraction prompt
result = llm.invoke(
    extraction_prompt.format(content=transcript)
)
```

**Output:**
```python
{
    "strategy_name": "London ORB Breakout",
    "entry_conditions": [
        "Price breaks above 30-minute opening range high",
        "Volume confirms breakout"
    ],
    "exit_conditions": [
        "Take profit at 1.5x risk",
        "Stop loss at opening range low"
    ],
    "filters": ["London session only"],
    "time_filters": ["First 2 hours of London session"],
    "indicators_used": ["Volume", "Opening Range"],
    "mentioned_concepts": ["Opening Range Breakout", "breakout confirmation"]
}
```

---

### Transformation 3: Concepts → Search Queries

**Input:**
```python
{
    "strategy_name": "London ORB Breakout",
    "entry_conditions": ["breakout", "volume confirmation"],
    "exit_conditions": ["take profit", "stop loss"],
    "indicators_used": ["Volume"]
}
```

**Transformation:**
```python
def generate_search_queries(state: AnalystState) -> List[str]:
    queries = []

    # Strategy name
    queries.append(state["strategy_name"])

    # Entry concepts
    for concept in state["entry_conditions"][:3]:
        queries.append(f"entry {concept}")

    # Exit concepts
    for concept in state["exit_conditions"][:2]:
        queries.append(f"exit {concept}")

    # Indicators
    for indicator in state["indicators_used"][:3]:
        queries.append(f"{indicator} indicator")

    return queries[:10]
```

**Output:**
```python
[
    "London ORB Breakout",
    "entry breakout",
    "entry volume confirmation",
    "exit take profit",
    "exit stop loss",
    "Volume indicator"
]
```

---

### Transformation 4: Search Queries → KB References

**Input:**
```python
["London ORB Breakout", "entry breakout"]
```

**Transformation (ChromaDB Search):**
```python
results = collection.query(
    query_texts=queries,
    n_results=5
)
```

**Output:**
```python
[
    {
        "title": "Opening Range Breakout Strategy",
        "relevance_score": 0.89,
        "categories": "Trading Systems",
        "content_preview": "The opening range breakout strategy...",
        "file_path": "data/scraped_articles/orb_strategy.md"
    },
    {
        "title": "Volume Confirmation Techniques",
        "relevance_score": 0.76,
        "categories": "Trading",
        "content_preview": "Volume is crucial for confirming breakouts...",
        "file_path": "data/scraped_articles/volume_confirmation.md"
    }
]
```

---

### Transformation 5: State → TRD Markdown

**Input:**
```python
{
    "strategy_name": "London ORB Breakout",
    "entry_conditions": ["breakout", "volume"],
    "exit_conditions": ["1.5x risk", "range low"],
    "kb_references": [...],
    "user_answers": {}
}
```

**Transformation (Template Generation):**
```python
def generate_trd(state: AnalystState) -> str:
    frontmatter = f"""---
strategy_name: "{state['strategy_name']}"
generated_at: "{datetime.now().isoformat()}"
status: "draft"
version: "1.0"
---

# Trading Strategy: {state['strategy_name']}
"""

    overview = f"## Overview\n\n{state.get('overview', 'No overview')}\n"

    entry = "## Entry Logic\n\n" + "\n".join(
        f"- {c}" for c in state['entry_conditions']
    )

    # ... more sections ...

    return frontmatter + overview + entry + ...
```

**Output:**
```markdown
---
strategy_name: "London ORB Breakout"
generated_at: "2026-01-27T10:30:00Z"
status: "draft"
version: "1.0"
---

# Trading Strategy: London ORB Breakout

## Overview

No overview

## Entry Logic

- breakout
- volume

## Exit Logic

- 1.5x risk
- range low

## Knowledge Base References

### Relevant Articles Found

1. **Opening Range Breakout Strategy**
   - Relevance: 0.89
   - Categories: Trading Systems
   ...
```

---

## Integration Points

### IP1: ChromaDB Integration

**Location:** `kb/client.py`

**Interface:**
```python
class AnalystKBClient:
    def __init__(self, chroma_path: str, collection_name: str)
    def search(query: str, n_results: int) -> List[Dict]
    def get_stats() -> Dict
    def list_categories() -> List[str]
```

**Data Flow:**
```
LangGraph Node
    │
    ├─► AnalystKBClient.__init__()
    │       │
    │       └─► chromadb.PersistentClient(path)
    │
    ├─► AnalystKBClient.search(query)
    │       │
    │       ├─► collection.query(query_texts)
    │       ├─► Extract titles from YAML frontmatter
    │       ├─► Calculate relevance scores
    │       ├─► Deduplicate results
    │       └─► Return formatted results
    │
    └─► Update state["kb_references"]
```

**Error Handling:**
- Collection not found → ValueError with setup instructions
- Empty results → Return empty list, log warning
- Connection error → Raise ChromaDBError with retry advice

---

### IP2: LLM API Integration

**Location:** `chains/extraction.py`, `chains/generation.py`

**Interface:**
```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# Initialize
llm = ChatOpenAI(model="gpt-4o", temperature=0.1)

# Invoke
result = llm.invoke(prompt)
```

**Data Flow:**
```
LangGraph Node
    │
    ├─► Create chain (prompt | llm | parser)
    │
    ├─► chain.invoke(input_data)
    │       │
    │       ├─► Format prompt with input
    │       ├─► Send HTTP request to API
    │       ├─► Parse response
    │       └─► Return structured output
    │
    └─► Update state with extracted data
```

**Provider Switching:**
```python
def get_llm(provider: str) -> Union[ChatOpenAI, ChatAnthropic]:
    if provider == "openai":
        return ChatOpenAI(model="gpt-4o", temperature=0.1)
    elif provider == "anthropic":
        return ChatAnthropic(model="claude-3-5-sonnet-20241022")
    else:
        raise ValueError(f"Unknown provider: {provider}")
```

**Error Handling:**
- API key missing → Clear error message
- Rate limit → Exponential backoff retry
- Invalid response → Fallback to alternative provider
- Timeout → Increase timeout or use faster model

---

### IP3: File System Integration

**Location:** `utils/file_io.py`, CLI output

**Interface:**
```python
def parse_nprd_output(file_path: Path) -> Dict
def parse_strategy_doc(file_path: Path) -> str
def save_trd(trd_content: str, output_path: Path) -> None
```

**Data Flow:**
```
User Command
    │
    ├─► Validate file path exists
    │
    ├─► Determine file type (JSON vs MD)
    │
    ├─► Parse content
    │       │
    │       ├─► Read file (UTF-8)
    │       ├─► Parse JSON or Markdown
    │       └─► Validate schema
    │
    └─► Return parsed data to workflow
```

**Output Flow:**
```
TRD Generation
    │
    ├─► Generate filename: {strategy}_{timestamp}.md
    │
    ├─► Create directory: docs/trds/
    │
    ├─► Write file (UTF-8)
    │
    └─► Display success message with path
```

**Error Handling:**
- File not found → FileNotFoundError with full path
- Permission denied → Clear permission error
- Invalid JSON → Parse error with line number
- Disk full → IOError with cleanup advice

---

### IP4: Human-in-the-Loop Integration

**Location:** `graph/nodes.py` (ask_user_node), `cli/interface.py`

**Interface:**
```python
class AnalystInterface:
    def prompt_missing_info(missing_fields: List[str]) -> Dict[str, str]
    def confirm_action(message: str) -> bool
```

**Data Flow:**
```
Gap Detection
    │
    ├─► Identify missing fields
    │
    ├─► Check if needs_user_input = True
    │
    ├─► ask_user_node invoked
    │       │
    │       ├─► Display gap list to user
    │       ├─► Prompt for each gap
    │       ├─► Collect user answers
    │       ├─► Allow 'skip' for optional fields
    │       └─► Return answers dictionary
    │
    └─► Update state["user_answers"]
```

**Auto Mode:**
```python
if auto_mode:
    # Use defaults from config
    state["user_answers"] = load_default_answers()
else:
    # Interactive prompts
    state["user_answers"] = interface.prompt_missing_info(
        state["missing_info"]
    )
```

---

## Error Handling Flows

### EH1: File Not Found

```
User Command
    │
    ├─► Check if file exists
    │
    ├─► File not found
    │       │
    │       ├─► Log error: FileNotFoundError
    │       ├─► Display user-friendly message:
    │       │   "ERROR: Input file not found: {path}"
    │       ├─► Suggest: "Run --list to see available files"
    │       └─► Exit with code 1
```

### EH2: Invalid NPRD JSON

```
File Parsing
    │
    ├─► Parse JSON
    │
    ├─► JSON decode error
    │       │
    │       ├─► Log error: ValueError
    │       ├─► Display message:
    │       │   "ERROR: Invalid JSON in {file}"
    │       │   "Line {n}: {error}"
    │       ├─► Suggest: "Validate file with --validate"
    │       └─► Exit with code 1
```

### EH3: LLM API Failure

```
Extraction Chain
    │
    ├─► Invoke LLM
    │
    ├─► API error (rate limit / timeout)
    │       │
    │       ├─► Log error with details
    │       ├──► Retry (exponential backoff, max 3)
    │       │       │
    │       │       ├─► Success → Continue
    │       │       │
    │       │       └─► All retries failed
    │       │               │
    │       │               ├─► Try fallback provider
    │       │               │
    │       │               └─► All providers failed
    │       │                       │
    │       │                       ├─► Display error
    │       │                       ├─► Save partial state
    │       │                       └─► Exit with code 2
```

### EH4: ChromaDB Collection Not Found

```
KB Search
    │
    ├─► Initialize client
    │
    ├─► Collection "analyst_kb" not found
    │       │
    │       ├─► Log error: ValueError
    │       ├─► Display message:
    │       │   "ERROR: ChromaDB collection 'analyst_kb' not found"
    │       ├─► Provide instructions:
    │       │   "Run: python scripts/create_analyst_kb.py"
    │       └─► Exit with code 1
```

### EH5: Empty Knowledge Base Results

```
KB Search
    │
    ├─► Execute search
    │
    ├─► No results found
    │       │
    │       ├─► Log warning: "No KB results for query: {query}"
    │       ├─► Set kb_references = []
    │       ├─► Display warning to user:
    │       │   "⚠️  No relevant articles found in knowledge base"
    │       │   "   TRD will be generated without references"
    │       └─► Continue workflow (non-fatal)
```

### EH6: User Interrupts HITL

```
HITL Prompts
    │
    ├─► Prompt user for missing info
    │
    ├─► User presses Ctrl+C
    │       │
    │       ├─► Log: "User interrupted HITL prompts"
    │       ├─► Save partial state to checkpoint
    │       ├─► Display message:
    │       │   "Interrupted by user. Partial state saved."
    │       │   "Resume with: --resume {checkpoint_id}"
    │       └─► Exit with code 130 (SIGINT)
```

---

## Performance Considerations

### Data Volume

| Data Type | Typical Size | Max Size | Processing Time |
|-----------|-------------|----------|-----------------|
| NPRD JSON | 50-500 KB | 2 MB | < 1s |
| Strategy MD | 5-50 KB | 500 KB | < 0.5s |
| Transcript | 10-100 KB | 1 MB | < 1s |
| LLM Request | 1-10 KB | 50 KB | 2-10s |
| LLM Response | 1-5 KB | 20 KB | 1-5s |
| KB Query | < 1 KB | < 1 KB | < 0.5s |
| KB Results | 10-100 KB | 500 KB | < 1s |
| TRD Output | 20-200 KB | 1 MB | < 1s |

### Bottlenecks

1. **LLM API Calls** (2-10s each)
   - Mitigation: Use faster models for extraction (Haiku)
   - Cache results for repeated concepts

2. **KB Search** (< 0.5s per query)
   - Mitigation: Batch queries, limit to 10

3. **File I/O** (< 1s)
   - Mitigation: Lazy loading, streaming

### Optimization Strategies

```python
# Parallel KB searches
async def parallel_search(queries: List[str]) -> List[Dict]:
    tasks = [search_async(q) for q in queries]
    results = await asyncio.gather(*tasks)
    return deduplicate(results)

# Cached extractions
@lru_cache(maxsize=100)
def extract_concepts_cached(transcript_hash: str) -> Dict:
    return extract_concepts(transcript)

# Streaming TRD generation
def generate_trd_stream(state: AnalystState):
    for section in trd_sections:
        yield section
```

---

**Document Version:** 1.0
**Last Modified:** 2026-01-27
**Next Review:** Post-implementation

# Component Specifications: Analyst Agent CLI

> **Component:** Analyst Agent CLI v1.0
> **Status:** Ready for Implementation
> **Last Updated:** 2026-01-27
> **Author:** System Architecture Designer

---

## Table of Contents

1. [Component Overview](#component-overview)
2. [CLI Layer Components](#cli-layer-components)
3. [Orchestration Layer Components](#orchestration-layer-components)
4. [Processing Layer Components](#processing-layer-components)
5. [Data Layer Components](#data-layer-components)
6. [Component Interfaces](#component-interfaces)
7. [Component Dependencies](#component-dependencies)

---

## Component Overview

### Component Hierarchy

```
analyst_agent/
├── cli/                    # CLI Layer
│   ├── commands.py         # Click command definitions
│   └── interface.py        # Interactive UI components
├── graph/                  # Orchestration Layer
│   ├── workflow.py         # LangGraph StateGraph
│   ├── nodes.py            # Node implementations
│   └── state.py            # State definitions
├── chains/                 # Processing Layer
│   ├── extraction.py       # Concept extraction chain
│   ├── search.py           # KB search chain
│   └── generation.py       # TRD generation chain
├── kb/                     # Data Layer
│   └── client.py           # ChromaDB client wrapper
├── prompts/                # Processing Layer
│   └── templates.py        # Prompt templates
└── utils/                  # Utilities
    ├── file_io.py          # File parsing
    ├── trd_template.py     # TRD generation
    ├── gap_detector.py     # Gap detection
    └── search_queries.py   # Query generation
```

### Component Matrix

| Component | Layer | Type | Interface |
|-----------|-------|------|-----------|
| `commands.py` | CLI | Module | Click commands |
| `interface.py` | CLI | Module | Rich UI |
| `workflow.py` | Orchestration | Module | LangGraph |
| `nodes.py` | Orchestration | Module | Node functions |
| `state.py` | Orchestration | Module | TypedDict |
| `extraction.py` | Processing | Chain | LangChain |
| `search.py` | Processing | Chain | LangChain |
| `generation.py` | Processing | Chain | LangChain |
| `client.py` | Data | Class | ChromaDB wrapper |
| `templates.py` | Processing | Module | Prompt strings |
| `file_io.py` | Utility | Module | File parsers |
| `trd_template.py` | Utility | Module | TRD generator |
| `gap_detector.py` | Utility | Module | Validator |
| `search_queries.py` | Utility | Module | Query builder |

---

## CLI Layer Components

### C1: Commands Module (`cli/commands.py`)

**Purpose:** Define Click command structure and routing

**Responsibilities:**
- Command registration and grouping
- Argument parsing and validation
- Help text generation
- Error handling and user feedback

**Interface:**
```python
import click
from pathlib import Path
from typing import Optional

@click.group()
@click.version_option(version="1.0.0")
def cli():
    """Analyst Agent CLI - Convert video/strategy content to TRD files."""
    pass

@cli.command()
@click.option('--input', '-i', type=Path, required=True,
              help='Input file path (NPRD JSON or strategy MD)')
@click.option('--type', 'input_type',
              type=click.Choice(['nprd', 'strategy_doc'], case_sensitive=False),
              default='nprd', help='Input type')
@click.option('--auto', is_flag=True,
              help='Autonomous mode (no prompts, use defaults)')
@click.option('--output', '-o', type=Path,
              help='Output directory (default: docs/trds)')
@click.option('--config', type=Path,
              help='Config file path (default: .analyst_config.json)')
def generate(input: Path, input_type: str, auto: bool,
             output: Optional[Path], config: Optional[Path]):
    """Generate TRD from NPRD output or strategy document."""
    pass

@cli.command()
def list():
    """List available NPRD outputs."""
    pass

@cli.command()
def stats():
    """Show knowledge base statistics."""
    pass

@cli.command()
@click.option('--trd', type=Path, required=True,
              help='TRD file to complete')
@click.option('--interactive', is_flag=True, default=True,
              help='Interactive mode (default: True)')
def complete(trd: Path, interactive: bool):
    """Complete missing fields in existing TRD."""
    pass
```

**Dependencies:**
- Click 8.1+
- Rich 13.0+
- LangGraph workflow
- Configuration module

**Error Handling:**
- File not found → Clear error with path
- Invalid JSON → Parse error with line number
- LLM API error → Retry with fallback
- ChromaDB error → Graceful degradation

---

### C2: Interface Module (`cli/interface.py`)

**Purpose:** Interactive user interface components

**Responsibilities:**
- Rich console formatting
- Interactive prompts
- Progress bars
- Menu displays

**Interface:**
```python
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.progress import Progress
from typing import List, Dict, Any

class AnalystInterface:
    """Interactive UI for Analyst Agent CLI."""

    def __init__(self):
        self.console = Console()

    def show_banner(self):
        """Display ASCII art banner."""
        pass

    def show_menu(self, options: List[Dict[str, str]]) -> int:
        """Display interactive menu and return selection."""
        pass

    def prompt_missing_info(self, missing_fields: List[str]) -> Dict[str, str]:
        """Prompt user for missing information."""
        pass

    def show_progress(self, steps: List[str]):
        """Show progress bar for workflow steps."""
        pass

    def show_kb_results(self, results: List[Dict[str, Any]]):
        """Display knowledge base search results."""
        pass

    def show_trd_preview(self, trd_content: str):
        """Preview generated TRD."""
        pass

    def confirm_action(self, message: str) -> bool:
        """Get yes/no confirmation."""
        pass
```

**Dependencies:**
- Rich 13.0+
- Pyfiglet 1.0+ (for ASCII art)

---

## Orchestration Layer Components

### C3: State Definition (`graph/state.py`)

**Purpose:** Define LangGraph state schema

**Responsibilities:**
- State structure definition
- Type safety enforcement
- State validation
- Serialization support

**Interface:**
```python
from typing import TypedDict, List, Optional, Dict, Any

class AnalystState(TypedDict):
    """State for Analyst Agent workflow."""

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
    overview: str
    entry_conditions: List[str]
    exit_conditions: List[str]
    filters: List[str]
    time_filters: List[str]
    indicators_used: List[str]
    mentioned_concepts: List[str]

    # KB search results
    kb_references: List[Dict[str, Any]]
    relevant_articles: List[str]

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

**Dependencies:**
- Python typing (TypedDict)
- Pydantic (optional validation)

---

### C4: Workflow Graph (`graph/workflow.py`)

**Purpose:** Define LangGraph StateGraph structure

**Responsibilities:**
- Graph construction
- Node wiring
- Conditional routing
- Checkpoint configuration

**Interface:**
```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint import MemorySaver
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

def create_analyst_graph(checkpointer: Optional[MemorySaver] = None) -> StateGraph:
    """Create and compile the analyst workflow graph.

    Args:
        checkpointer: Optional state checkpointer for persistence

    Returns:
        Compiled StateGraph ready for invocation
    """
    workflow = StateGraph(AnalystState)

    # Add nodes
    workflow.add_node("parse_input", parse_input_node)
    workflow.add_node("extract_concepts", extract_concepts_node)
    workflow.add_node("search_kb", search_kb_node)
    workflow.add_node("identify_gaps", identify_gaps_node)
    workflow.add_node("ask_user", ask_user_node)
    workflow.add_node("generate_trd", generate_trd_node)

    # Define edges
    workflow.set_entry_point("parse_input")
    workflow.add_edge("parse_input", "extract_concepts")
    workflow.add_edge("extract_concepts", "search_kb")
    workflow.add_edge("search_kb", "identify_gaps")

    # Conditional routing based on gaps
    workflow.add_conditional_edges(
        "identify_gaps",
        has_gaps,
        {
            "ask_user": "ask_user",
            "generate_trd": "generate_trd"
        }
    )

    workflow.add_edge("ask_user", "generate_trd")
    workflow.add_edge("generate_trd", END)

    # Compile with optional checkpointer
    return workflow.compile(checkpointer=checkpointer)

def run_analyst_workflow(
    initial_state: AnalystState,
    auto_mode: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> AnalystState:
    """Run the analyst workflow.

    Args:
        initial_state: Initial state with input_path and input_type
        auto_mode: If True, skip user prompts and use defaults
        config: Optional configuration for graph execution

    Returns:
        Final state with generated TRD
    """
    graph = create_analyst_graph()

    # Configure execution
    exec_config = {
        "recursion_limit": 20,
        "auto_mode": auto_mode,
        **(config or {})
    }

    # Run workflow
    result = graph.invoke(initial_state, exec_config)

    return result
```

**Dependencies:**
- LangGraph 0.2+
- Node implementations
- State definition

---

### C5: Node Implementations (`graph/nodes.py`)

**Purpose:** Implement individual workflow nodes

**Responsibilities:**
- State transformation logic
- Chain invocation
- Error handling
- Progress updates

**Interface:**
```python
from typing import Dict, Any, Callable
from .state import AnalystState

def parse_input_node(state: AnalystState) -> AnalystState:
    """Parse input file (NPRD JSON or strategy doc).

    Updates:
        - raw_content: Parsed file content
        - transcript: Full text content
        - keywords: Extracted keywords
        - current_step: Set to "parsed"
    """
    pass

def extract_concepts_node(state: AnalystState) -> AnalystState:
    """Extract trading concepts using LLM chain.

    Updates:
        - strategy_name: Extracted name
        - entry_conditions: Entry logic
        - exit_conditions: Exit logic
        - filters: Market filters
        - indicators_used: Technical indicators
        - current_step: Set to "extracted"
    """
    pass

def search_kb_node(state: AnalystState) -> AnalystState:
    """Search ChromaDB for relevant articles.

    Updates:
        - kb_references: Top 10 relevant articles
        - relevant_articles: Article titles
        - current_step: Set to "searched"
    """
    pass

def identify_gaps_node(state: AnalystState) -> AnalystState:
    """Identify missing required information.

    Updates:
        - missing_info: List of missing fields
        - needs_user_input: Boolean flag
        - current_step: Set to "gaps_identified"
    """
    pass

def ask_user_node(state: AnalystState) -> AnalystState:
    """Prompt user for missing information (interactive).

    Updates:
        - user_answers: Dict of field -> value
        - current_step: Set to "user_input_received"
    """
    pass

def generate_trd_node(state: AnalystState) -> AnalystState:
    """Generate final TRD markdown file.

    Updates:
        - trd_content: Generated markdown
        - trd_path: Output file path
        - completed: Set to True
        - current_step: Set to "completed"
    """
    pass

def has_gaps(state: AnalystState) -> str:
    """Conditional edge function.

    Returns:
        "ask_user" if gaps exist, "generate_trd" otherwise
    """
    return "ask_user" if state.get('needs_user_input', False) else "generate_trd"
```

**Dependencies:**
- LangChain chains
- ChromaDB client
- Gap detector
- TRD template
- CLI interface (for interactive prompts)

---

## Processing Layer Components

### C6: Extraction Chain (`chains/extraction.py`)

**Purpose:** Extract structured trading concepts from unstructured text

**Responsibilities:**
- LLM-based concept extraction
- JSON parsing and validation
- Error recovery
- Result formatting

**Interface:**
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List

class ExtractedConcepts(BaseModel):
    """Schema for extracted trading concepts."""
    strategy_name: str = Field(description="Name of the trading strategy")
    overview: str = Field(description="Brief strategy description")
    entry_conditions: List[str] = Field(description="Entry trigger conditions")
    exit_conditions: List[str] = Field(description="Exit conditions (TP, SL, trailing)")
    filters: List[str] = Field(description="Market condition filters")
    time_filters: List[str] = Field(description="Time-based filters")
    indicators_used: List[str] = Field(description="Technical indicators mentioned")
    mentioned_concepts: List[str] = Field(description="Other trading concepts")

def create_extraction_chain(
    llm: ChatOpenAI,
    temperature: float = 0.1
):
    """Create concept extraction chain.

    Args:
        llm: Language model instance
        temperature: Sampling temperature (lower = more deterministic)

    Returns:
        Compiled LangChain
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a trading strategy analyst. Extract structured information from trading content.

Return ONLY valid JSON with this structure:
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

Return ONLY valid JSON, no markdown, no explanations."""),
        ("human", "Content:\n{content}")
    ])

    parser = JsonOutputParser(pydantic_object=ExtractedConcepts)

    chain = prompt | llm | parser
    return chain

def extract_concepts(content: str, llm: ChatOpenAI) -> ExtractedConcepts:
    """Extract concepts from content.

    Args:
        content: Transcript or document text
        llm: Language model instance

    Returns:
        ExtractedConcepts object

    Raises:
        ValueError: If extraction fails or JSON is invalid
    """
    chain = create_extraction_chain(llm)
    result = chain.invoke({"content": content})
    return ExtractedConcepts(**result)
```

**Dependencies:**
- LangChain 0.2+
- OpenAI/Anthropic SDKs
- Pydantic 2.0+

---

### C7: Search Chain (`chains/search.py`)

**Purpose:** Search ChromaDB knowledge base for relevant articles

**Responsibilities:**
- Generate search queries
- Execute semantic search
- Rank and deduplicate results
- Format references

**Interface:**
```python
from typing import List, Dict, Any
from ..kb.client import AnalystKBClient
from ..utils.search_queries import generate_search_queries

def search_knowledge_base(
    state: AnalystState,
    kb_client: AnalystKBClient,
    n_results: int = 10
) -> AnalystState:
    """Search ChromaDB for relevant articles.

    Args:
        state: Current workflow state
        kb_client: ChromaDB client instance
        n_results: Number of results to return

    Updates:
        - kb_references: List of search results
        - relevant_articles: List of article titles

    Returns:
        Updated state
    """
    # Generate search queries from extracted concepts
    queries = generate_search_queries(state)

    # Search and collect results
    all_results = []
    for query in queries:
        results = kb_client.search(query, n_results=5)
        all_results.extend(results)

    # Deduplicate by title
    seen = set()
    unique_results = []
    for result in all_results:
        title = result.get('title', '')
        if title and title not in seen:
            seen.add(title)
            unique_results.append(result)

    # Sort by relevance score
    unique_results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)

    # Update state
    state['kb_references'] = unique_results[:n_results]
    state['relevant_articles'] = [r['title'] for r in unique_results[:n_results]]

    return state
```

**Dependencies:**
- ChromaDB client
- Query generator utility
- State definition

---

### C8: Generation Chain (`chains/generation.py`)

**Purpose:** Generate TRD markdown from extracted data

**Responsibilities:**
- Template filling
- Section generation
- Reference formatting
- Markdown validation

**Interface:**
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from typing import Dict, Any

def create_generation_chain(
    llm: ChatOpenAI,
    temperature: float = 0.3
):
    """Create TRD generation chain.

    Args:
        llm: Language model instance
        temperature: Sampling temperature

    Returns:
        Compiled LangChain
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a technical writer specializing in trading strategy documentation.

Generate a Technical Requirements Document (TRD) in Markdown format based on the provided information.

Structure:
1. Overview section with strategy description
2. Entry Logic section with detailed conditions
3. Exit Logic section with TP/SL/trailing
4. Filters section (time and market conditions)
5. Indicators & Settings section
6. Position Sizing & Risk Management section
7. Knowledge Base References section with relevant articles
8. Missing Information section for incomplete data

Use clear, professional language. Include code examples for logic where appropriate.
Format as valid Markdown with proper headers, tables, and code blocks."""),
        ("human", """Generate TRD for the following strategy:

Strategy Name: {strategy_name}
Overview: {overview}
Entry Conditions: {entry_conditions}
Exit Conditions: {exit_conditions}
Filters: {filters}
Time Filters: {time_filters}
Indicators: {indicators_used}
User Answers: {user_answers}
KB References: {kb_references}

Generate complete TRD markdown.""")
    ])

    chain = prompt | llm
    return chain

def generate_trd_content(
    state: AnalystState,
    llm: ChatOpenAI
) -> str:
    """Generate TRD markdown content.

    Args:
        state: Current workflow state
        llm: Language model instance

    Returns:
        TRD markdown string
    """
    chain = create_generation_chain(llm)
    result = chain.invoke(state)
    return result.content
```

**Dependencies:**
- LangChain 0.2+
- OpenAI/Anthropic SDKs
- State definition

---

## Data Layer Components

### C9: ChromaDB Client (`kb/client.py`)

**Purpose:** Wrapper for ChromaDB knowledge base operations

**Responsibilities:**
- Connection management
- Query execution
- Result formatting
- Statistics retrieval

**Interface:**
```python
import chromadb
from pathlib import Path
from typing import List, Dict, Optional

class AnalystKBClient:
    """Client for querying analyst_kb ChromaDB collection."""

    def __init__(
        self,
        chroma_path: Optional[str] = None,
        collection_name: str = "analyst_kb"
    ):
        """Initialize ChromaDB client.

        Args:
            chroma_path: Path to ChromaDB storage (default: data/chromadb)
            collection_name: Name of collection to query
        """
        self.chroma_path = chroma_path or str(Path("data/chromadb"))
        self.collection_name = collection_name

        # Initialize persistent client
        self.client = chromadb.PersistentClient(path=self.chroma_path)

        # Get or create collection
        try:
            self.collection = self.client.get_collection(collection_name)
        except Exception:
            raise ValueError(
                f"Collection '{collection_name}' not found. "
                f"Run scripts/create_analyst_kb.py first."
            )

    def search(
        self,
        query: str,
        n_results: int = 5,
        category_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search knowledge base.

        Args:
            query: Natural language search query
            n_results: Number of results to return
            category_filter: Optional category filter

        Returns:
            List of search results with metadata
        """
        # Execute search
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results * 2  # Get more for filtering
        )

        # Process results
        formatted = []
        seen_titles = set()

        if results['documents'] and len(results['documents'][0]) > 0:
            for i, doc in enumerate(results['documents'][0]):
                meta = results['metadatas'][0][i]
                distance = results['distances'][0][i] if 'distances' in results else 0

                # Extract title
                title = self._extract_title(doc)
                if not title or title in seen_titles:
                    continue

                seen_titles.add(title)

                # Category filter
                if category_filter:
                    categories = meta.get('categories', '')
                    if category_filter.lower() not in categories.lower():
                        continue

                # Calculate relevance score
                relevance_score = 1 - distance

                # Get preview
                preview = self._get_preview(doc, max_length=300)

                formatted.append({
                    'title': title,
                    'file_path': meta.get('file_path', ''),
                    'categories': meta.get('categories', ''),
                    'content_preview': preview,
                    'relevance_score': round(relevance_score, 3)
                })

                if len(formatted) >= n_results:
                    break

        return formatted

    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics.

        Returns:
            Dictionary with collection stats
        """
        count = self.collection.count()

        return {
            'collection': self.collection_name,
            'total_articles': count,
            'storage_path': self.chroma_path
        }

    def list_categories(self) -> List[str]:
        """List all categories in collection.

        Returns:
            Sorted list of unique categories
        """
        results = self.collection.get(include=["metadatas"])
        categories = set()

        if results['metadatas']:
            for meta in results['metadatas']:
                cat = meta.get('categories', '')
                if cat:
                    categories.add(cat)

        return sorted(list(categories))

    def _extract_title(self, doc: str) -> str:
        """Extract title from YAML frontmatter or heading."""
        # Try YAML frontmatter
        if doc.startswith('---'):
            lines = doc.split('\n')
            for line in lines[1:10]:
                if line.startswith('title:'):
                    return line.split(':', 1)[1].strip()
                if line == '---':
                    break

        # Try first # heading
        import re
        match = re.search(r'^#\s+(.+)$', doc, re.MULTILINE)
        if match:
            return match.group(1).strip()

        return "Untitled"

    def _get_preview(self, doc: str, max_length: int = 300) -> str:
        """Get content preview skipping frontmatter."""
        lines = doc.split('\n')
        preview_start = 0

        # Skip YAML frontmatter
        for i, line in enumerate(lines):
            if i > 0 and line == '---':
                preview_start = i + 1
                break

        preview = '\n'.join(lines[preview_start:preview_start + 10]).strip()

        if len(preview) > max_length:
            preview = preview[:max_length] + "..."

        return preview
```

**Dependencies:**
- ChromaDB 0.4+
- Python pathlib

---

## Processing Layer Components (Utilities)

### C10: Prompt Templates (`prompts/templates.py`)

**Purpose:** Centralized prompt template management

**Responsibilities:**
- Template storage
- Variable interpolation
- Version tracking

**Interface:**
```python
EXTRACTION_SYSTEM_PROMPT = """You are a trading strategy analyst. Extract structured information from trading content.

Return ONLY valid JSON with this structure:
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

Return ONLY valid JSON, no markdown, no explanations."""

GENERATION_SYSTEM_PROMPT = """You are a technical writer specializing in trading strategy documentation.

Generate a Technical Requirements Document (TRD) in Markdown format based on the provided information.

Structure:
1. Overview section with strategy description
2. Entry Logic section with detailed conditions
3. Exit Logic section with TP/SL/trailing
4. Filters section (time and market conditions)
5. Indicators & Settings section
6. Position Sizing & Risk Management section
7. Knowledge Base References section with relevant articles
8. Missing Information section for incomplete data

Use clear, professional language. Include code examples for logic where appropriate.
Format as valid Markdown with proper headers, tables, and code blocks."""

GAP_DETECTION_PROMPTS = {
    "entry_logic": [
        "entry trigger",
        "entry confirmation",
        "entry timeframe"
    ],
    "exit_logic": [
        "take profit strategy",
        "stop loss placement",
        "trailing stop method"
    ],
    "risk_management": [
        "position sizing method",
        "max risk per trade",
        "daily loss limit"
    ],
    "filters": [
        "trading session filters",
        "market condition filters",
        "volatility filters"
    ],
    "indicators": [
        "indicator settings",
        "indicator timeframes",
        "indicator parameters"
    ]
}
```

**Dependencies:**
- None (constant strings)

---

### C11: File I/O Utilities (`utils/file_io.py`)

**Purpose:** Parse input files (NPRD JSON and strategy docs)

**Responsibilities:**
- File validation
- JSON parsing
- Markdown parsing
- Error handling

**Interface:**
```python
from pathlib import Path
from typing import Dict, Any, Optional
import json

def parse_nprd_output(file_path: Path) -> Dict[str, Any]:
    """Parse NPRD JSON output file.

    Args:
        file_path: Path to NPRD JSON file

    Returns:
        Parsed NPRD data dictionary

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If JSON is invalid or schema doesn't match
    """
    if not file_path.exists():
        raise FileNotFoundError(f"NPRD output not found: {file_path}")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {file_path}: {e}")

    # Validate schema
    required_fields = ['video_id', 'video_title', 'chunks', 'summary']
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")

    return data

def parse_strategy_doc(file_path: Path) -> str:
    """Parse strategy markdown document.

    Args:
        file_path: Path to strategy markdown file

    Returns:
        Document content as string

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Strategy document not found: {file_path}")

    return file_path.read_text(encoding='utf-8')

def aggregate_chunks(chunk_dir: Path, pattern: str = "*.json") -> Dict[str, Any]:
    """Aggregate multiple NPRD chunk files.

    Args:
        chunk_dir: Directory containing chunk files
        pattern: Glob pattern for chunk files

    Returns:
        Aggregated NPRD data

    Raises:
        ValueError: If no chunks found or incompatible schemas
    """
    chunk_files = list(chunk_dir.glob(pattern))

    if not chunk_files:
        raise ValueError(f"No chunks found matching {pattern} in {chunk_dir}")

    # Load first chunk as base
    base_data = parse_nprd_output(chunk_files[0])

    # Aggregate chunks
    all_chunks = base_data.get('chunks', [])
    all_keywords = base_data.get('summary', {}).get('all_keywords', [])

    for chunk_file in chunk_files[1:]:
        chunk_data = parse_nprd_output(chunk_file)
        all_chunks.extend(chunk_data.get('chunks', []))

        keywords = chunk_data.get('summary', {}).get('all_keywords', [])
        all_keywords.extend(keywords)

    # Update base data
    base_data['chunks'] = all_chunks
    base_data['summary']['all_keywords'] = list(set(all_keywords))

    return base_data

def validate_input_file(file_path: Path) -> Optional[str]:
    """Validate input file and return type.

    Args:
        file_path: Path to input file

    Returns:
        "nprd" if NPRD JSON, "strategy_doc" if markdown, None if invalid

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    # Check by extension and content
    if file_path.suffix == '.json':
        try:
            data = json.loads(file_path.read_text())
            if 'video_id' in data and 'chunks' in data:
                return "nprd"
        except:
            pass

    if file_path.suffix in ['.md', '.markdown']:
        return "strategy_doc"

    return None
```

**Dependencies:**
- Python json, pathlib
- Pydantic (optional validation)

---

### C12: Gap Detector (`utils/gap_detector.py`)

**Purpose:** Identify missing required information

**Responsibilities:**
- Required field validation
- Gap detection
- Prioritization
- User prompt generation

**Interface:**
```python
from typing import List, Dict, Set
from ..graph.state import AnalystState

REQUIRED_TRD_FIELDS = {
    "entry_logic": [
        "entry trigger",
        "entry confirmation",
        "entry timeframe"
    ],
    "exit_logic": [
        "take profit",
        "stop loss",
        "exit strategy"
    ],
    "risk_management": [
        "position sizing",
        "max risk"
    ],
    "filters": [
        "time filters",
        "market condition filters"
    ],
    "indicators": [
        "indicator settings",
        "timeframes"
    ]
}

def detect_gaps(state: AnalystState) -> List[str]:
    """Detect missing required information.

    Args:
        state: Current workflow state

    Returns:
        List of missing field descriptions
    """
    missing = []

    # Combine all text content for searching
    content = " ".join([
        state.get('transcript', ''),
        " ".join(state.get('entry_conditions', [])),
        " ".join(state.get('exit_conditions', [])),
        " ".join(state.get('filters', [])),
        " ".join(state.get('time_filters', [])),
        " ".join(state.get('indicators_used', [])),
        " ".join(state.get('mentioned_concepts', []))
    ]).lower()

    # Check each required category
    for category, fields in REQUIRED_TRD_FIELDS.items():
        for field in fields:
            if field.lower() not in content:
                missing.append(f"{category}: {field}")

    return missing

def prioritize_gaps(gaps: List[str]) -> Dict[str, List[str]]:
    """Prioritize gaps by importance.

    Args:
        gaps: List of gap descriptions

    Returns:
        Dictionary with priority levels
    """
    high_priority = []
    medium_priority = []
    low_priority = []

    critical_terms = ['entry', 'exit', 'stop loss', 'take profit']
    important_terms = ['position sizing', 'risk', 'filters']
    nice_to_have = ['trailing', 'timeframes', 'settings']

    for gap in gaps:
        gap_lower = gap.lower()

        if any(term in gap_lower for term in critical_terms):
            high_priority.append(gap)
        elif any(term in gap_lower for term in important_terms):
            medium_priority.append(gap)
        else:
            low_priority.append(gap)

    return {
        "high": high_priority,
        "medium": medium_priority,
        "low": low_priority
    }
```

**Dependencies:**
- State definition
- Typing

---

### C13: TRD Template (`utils/trd_template.py`)

**Purpose:** Generate TRD markdown from state

**Responsibilities:**
- Template rendering
- Section organization
- YAML frontmatter generation
- Reference formatting

**Interface:**
```python
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path

def generate_trd(state: AnalystState) -> str:
    """Generate TRD markdown from workflow state.

    Args:
        state: Final workflow state

    Returns:
        Complete TRD markdown string
    """
    # Extract data
    strategy_name = state.get('strategy_name', 'Unnamed Strategy')
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Generate YAML frontmatter
    frontmatter = generate_frontmatter(state, timestamp)

    # Generate sections
    sections = []
    sections.append(generate_overview_section(state))
    sections.append(generate_entry_section(state))
    sections.append(generate_exit_section(state))
    sections.append(generate_filters_section(state))
    sections.append(generate_indicators_section(state))
    sections.append(generate_risk_section(state))
    sections.append(generate_references_section(state))

    if state.get('missing_info'):
        sections.append(generate_missing_section(state))

    # Combine
    trd = f"{frontmatter}\n\n" + "\n\n---\n\n".join(sections)

    return trd

def generate_frontmatter(state: AnalystState, timestamp: str) -> str:
    """Generate YAML frontmatter."""
    return f"""---
strategy_name: "{state.get('strategy_name', 'Unnamed')}"
source: "NPRD: {state.get('input_path', 'unknown')}"
generated_at: "{timestamp}"
status: "draft"
version: "1.0"
---

# Trading Strategy: {state.get('strategy_name', 'Unnamed')}
"""

def generate_overview_section(state: AnalystState) -> str:
    """Generate overview section."""
    return f"""## Overview

{state.get('overview', 'No overview available.')}

**Source:** {state.get('input_path', 'Unknown')}
**Analyst:** Analyst Agent v1.0
**Generated:** {datetime.now().strftime("%Y-%m-%d")}
"""

def generate_entry_section(state: AnalystState) -> str:
    """Generate entry logic section."""
    conditions = state.get('entry_conditions', [])

    if not conditions:
        return "## Entry Logic\n\n*No entry conditions specified.*\n"

    conditions_md = "\n".join(f"- {c}" for c in conditions)

    return f"""## Entry Logic

### Entry Conditions

{conditions_md}

### Entry Example

```
IF {" AND ".join(conditions[:3])}
THEN enter [LONG/SHORT]
```
"""

def generate_exit_section(state: AnalystState) -> str:
    """Generate exit logic section."""
    exits = state.get('exit_conditions', [])

    if not exits:
        return "## Exit Logic\n\n*No exit conditions specified.*\n"

    exits_md = "\n".join(f"- {e}" for e in exits)

    return f"""## Exit Logic

### Exit Conditions

{exits_md}
"""

def generate_filters_section(state: AnalystState) -> str:
    """Generate filters section."""
    time_filters = state.get('time_filters', [])
    market_filters = state.get('filters', [])

    sections = []
    if time_filters:
        time_md = "\n".join(f"- {f}" for f in time_filters)
        sections.append(f"### Time Filters\n\n{time_md}")

    if market_filters:
        market_md = "\n".join(f"- {f}" for f in market_filters)
        sections.append(f"### Market Condition Filters\n\n{market_md}")

    if not sections:
        return "## Filters\n\n*No filters specified.*\n"

    return "## Filters\n\n" + "\n\n".join(sections)

def generate_indicators_section(state: AnalystState) -> str:
    """Generate indicators section."""
    indicators = state.get('indicators_used', [])

    if not indicators:
        return "## Indicators & Settings\n\n*No indicators specified.*\n"

    table = "| Indicator | Settings | Purpose |\n|-----------|----------|---------|\n"
    for ind in indicators:
        table += f"| {ind} | TODO | TODO |\n"

    return f"## Indicators & Settings\n\n{table}"

def generate_risk_section(state: AnalystState) -> str:
    """Generate risk management section."""
    return """## Position Sizing & Risk Management

> **Note:** This section will be enhanced in v2.0 with dynamic risk management system.

### Risk Per Trade

- [Mentioned in source or TODO - specify risk amount]

### Position Sizing

- [Method: TODO - implement position sizing]

### Max Drawdown Limit

- [Daily limit if mentioned or TODO - specify drawdown limit]
"""

def generate_references_section(state: AnalystState) -> str:
    """Generate knowledge base references section."""
    refs = state.get('kb_references', [])

    if not refs:
        return "## Knowledge Base References\n\n*No references found.*\n"

    ref_md = "### Relevant Articles Found\n\n"
    for i, ref in enumerate(refs[:5], 1):
        ref_md += f"""{i}. **{ref['title']}**
   - Relevance: {ref.get('relevance_score', 0):.2f}
   - Categories: {ref.get('categories', 'N/A')}
   - Preview: {ref.get('content_preview', 'N/A')[:200]}...

"""

    return f"## Knowledge Base References\n\n{ref_md}"

def generate_missing_section(state: AnalystState) -> str:
    """Generate missing information section."""
    missing = state.get('missing_info', [])

    missing_md = "\n".join(f"- [ ] {m}" for m in missing)

    return f"""## Missing Information (Requires Input)

The following information was not found in the source and needs user input:

{missing_md}

**To complete this TRD, run:**
```bash
python tools/analyst_cli.py --trd docs/trds/{{this_file}}.md --complete
```
"""

def save_trd(trd_content: str, output_path: Path) -> None:
    """Save TRD to file.

    Args:
        trd_content: Generated TRD markdown
        output_path: Output file path

    Raises:
        IOError: If file cannot be written
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(trd_content, encoding='utf-8')
```

**Dependencies:**
- Python datetime, pathlib
- State definition

---

### C14: Query Generator (`utils/search_queries.py`)

**Purpose:** Generate search queries from extracted concepts

**Responsibilities:**
- Query construction
- Concept extraction
- Query expansion
- Synonym generation

**Interface:**
```python
from typing import List
from ..graph.state import AnalystState

def generate_search_queries(state: AnalystState) -> List[str]:
    """Generate search queries from extracted concepts.

    Args:
        state: Current workflow state

    Returns:
        List of search query strings
    """
    queries = []

    # Strategy name
    if state.get('strategy_name'):
        queries.append(state['strategy_name'])

    # Entry concepts
    for concept in state.get('entry_conditions', [])[:3]:
        queries.append(f"entry {concept}")

    # Exit concepts
    for concept in state.get('exit_conditions', [])[:2]:
        queries.append(f"exit {concept}")

    # Indicators
    for indicator in state.get('indicators_used', [])[:3]:
        queries.append(f"{indicator} indicator")

    # Filters
    for filter_term in state.get('filters', [])[:2]:
        queries.append(f"{filter_term} filter")

    # Time filters
    for time_filter in state.get('time_filters', [])[:2]:
        queries.append(f"{time_filter} trading")

    return queries[:10]  # Max 10 queries
```

**Dependencies:**
- State definition

---

## Component Interfaces

### Interface Summary

| Component | Public Interface | Internal Interface |
|-----------|-----------------|-------------------|
| CLI Commands | Click decorators | LangGraph workflow |
| Interface | Rich UI methods | None |
| Workflow | `create_analyst_graph()`, `run_workflow()` | Node functions |
| Nodes | Node functions | Chains, utilities |
| Extraction | `create_extraction_chain()`, `extract_concepts()` | LLM API |
| Search | `search_knowledge_base()` | ChromaDB client |
| Generation | `create_generation_chain()`, `generate_trd_content()` | LLM API |
| KB Client | `search()`, `get_stats()`, `list_categories()` | ChromaDB |
| Templates | Prompt constants | None |
| File I/O | `parse_nprd_output()`, `parse_strategy_doc()` | None |
| Gap Detector | `detect_gaps()`, `prioritize_gaps()` | None |
| TRD Template | `generate_trd()`, `save_trd()` | None |
| Query Generator | `generate_search_queries()` | None |

---

## Component Dependencies

### Dependency Graph

```
CLI Commands
    ├─► Interface (Rich UI)
    ├─► Workflow (LangGraph)
    │       ├─► Nodes
    │       │       ├─► Extraction Chain
    │       │       │       └─► LLM API
    │       │       ├─► Search Chain
    │       │       │       ├─► KB Client
    │       │       │       │       └─► ChromaDB
    │       │       │       └─► Query Generator
    │       │       ├─► Gap Detector
    │       │       ├─► Interface (for prompts)
    │       │       └─► Generation Chain
    │       │               └─► LLM API
    │       └─► State
    └─► File I/O
```

### External Dependencies

```
analyst_agent
    ├─► click (CLI framework)
    ├─► rich (CLI formatting)
    ├─► langchain (chain composition)
    ├─► langgraph (workflow orchestration)
    ├─► openai (LLM API)
    ├─► anthropic (LLM API)
    ├─► chromadb (vector database)
    ├─► pydantic (validation)
    ├─► pyyaml (configuration)
    └─► python-dotenv (environment)
```

---

**Document Version:** 1.0
**Last Modified:** 2026-01-27
**Next Review:** Post-implementation

# Test Specification: Analyst Agent CLI v1.0

> **Project:** QuantMindX Analyst Agent
> **Component:** Analyst Agent CLI
> **Version:** 1.0
> **Testing Approach:** Test-Driven Development (TDD)
> **Coverage Target:** 70% minimum, 80% ideal
> **Generated:** 2026-01-27

---

## Table of Contents

1. [Testing Strategy Overview](#1-testing-strategy-overview)
2. [Unit Test Requirements](#2-unit-test-requirements)
3. [Integration Test Requirements](#3-integration-test-requirements)
4. [End-to-End Test Scenarios](#4-end-to-end-test-scenarios)
5. [Manual Testing Procedures](#5-manual-testing-procedures)
6. [Test Data Requirements](#6-test-data-requirements)
7. [Performance Testing](#7-performance-testing)
8. [Security Testing](#8-security-testing)
9. [Test Environment Setup](#9-test-environment-setup)
10. [Coverage Metrics & Reporting](#10-coverage-metrics--reporting)
11. [Continuous Integration](#11-continuous-integration)
12. [Test Execution Guidelines](#12-test-execution-guidelines)

---

## 1. Testing Strategy Overview

### 1.1 Testing Pyramid

```
                    ┌─────────────────────┐
                    │   Manual E2E        │  <- 5 scenarios (User validation)
                    │   (Human-in-the-Loop)│
                    └─────────────────────┘
                              │
                    ┌─────────────────────┐
                    │   Integration       │  <- 12 test scenarios
                    │   Tests             │
                    └─────────────────────┘
                              │
                    ┌─────────────────────┐
                    │   Unit Tests        │  <- 50+ test cases
                    │   (pytest)          │
                    └─────────────────────┘
```

### 1.2 Test-Driven Development (TDD) Cycle

**RED-GREEN-REFACTOR** approach for all development:

1. **RED** - Write failing test first
2. **GREEN** - Implement minimal code to pass
3. **REFACTOR** - Improve while keeping tests green

### 1.3 Coverage Targets

| Metric Type | Minimum Target | Ideal Target |
|-------------|---------------|--------------|
| **Statement Coverage** | 70% | 85% |
| **Branch Coverage** | 65% | 80% |
| **Function Coverage** | 75% | 90% |
| **Line Coverage** | 70% | 85% |

### 1.4 Testing Philosophy

- **Fast Feedback**: Unit tests should run in < 100ms each
- **Isolation**: No dependencies between tests
- **Repeatable**: Same result every time
- **Self-Validating**: Clear pass/fail criteria
- **Timely**: Written with or before code

---

## 2. Unit Test Requirements

### 2.1 KB Client Module (`analyst_agent/kb/client.py`)

**Purpose:** Test ChromaDB client wrapper functionality

#### Test Cases

##### TC-KB-001: Client Initialization
```python
def test_client_initialization_success():
    """Test successful ChromaDB client initialization."""
    # Arrange
    chroma_path = "data/chromadb"

    # Act
    client = AnalystKBClient(chroma_path=chroma_path)

    # Assert
    assert client.chroma_path == chroma_path
    assert client.collection is not None
    assert client.collection.name == "analyst_kb"
```

##### TC-KB-002: Search with Results
```python
def test_search_returns_results():
    """Test search returns relevant articles."""
    # Arrange
    client = AnalystKBClient()
    query = "RSI trading strategy"

    # Act
    results = client.search(query, n_results=5)

    # Assert
    assert len(results) > 0
    assert all('title' in r for r in results)
    assert all('content' in r for r in results)
    assert all('distance' in r for r in results)
```

##### TC-KB-003: Search with No Results
```python
def test_search_no_results():
    """Test search with query that has no matches."""
    # Arrange
    client = AnalystKBClient()
    query = "xyzabc123nonexistent"

    # Act
    results = client.search(query, n_results=5)

    # Assert
    assert len(results) == 0
```

##### TC-KB-004: Search with Category Filter
```python
def test_search_with_category_filter():
    """Test search with category filtering."""
    # Arrange
    client = AnalystKBClient()
    query = "moving average"
    category = "Indicators"

    # Act
    results = client.search(query, n_results=5, category_filter=category)

    # Assert
    assert len(results) >= 0
    for result in results:
        assert category.lower() in result['categories'].lower()
```

##### TC-KB-005: Get Collection Stats
```python
def test_get_stats():
    """Test retrieving collection statistics."""
    # Arrange
    client = AnalystKBClient()

    # Act
    stats = client.get_stats()

    # Assert
    assert 'collection' in stats
    assert 'total_articles' in stats
    assert stats['total_articles'] > 0
    assert stats['collection'] == 'analyst_kb'
```

##### TC-KB-006: Deduplication of Results
```python
def test_search_deduplicates_results():
    """Test that search removes duplicate articles."""
    # Arrange
    client = AnalystKBClient()
    query = "breakout strategy"  # Common term

    # Act
    results = client.search(query, n_results=10)

    # Assert
    titles = [r['title'] for r in results]
    assert len(titles) == len(set(titles)), "Results should be deduplicated"
```

---

### 2.2 Extraction Chain Module (`analyst_agent/chains/extraction.py`)

**Purpose:** Test concept extraction from unstructured text

#### Test Cases

##### TC-EXT-001: Valid Input Extraction
```python
def test_extraction_valid_input():
    """Test extraction with valid strategy description."""
    # Arrange
    content = """
    Buy when RSI is below 30 and price crosses above EMA 20.
    Take profit at 1.5x risk, stop loss at ATR * 1.5.
    Only trade during London session.
    """

    # Act
    result = extraction_chain.invoke({"content": content})

    # Assert
    assert 'entry_conditions' in result
    assert 'exit_conditions' in result
    assert 'filters' in result
    assert len(result['entry_conditions']) > 0
    assert len(result['exit_conditions']) > 0
```

##### TC-EXT-002: Empty Input Handling
```python
def test_extraction_empty_input():
    """Test extraction with empty content."""
    # Arrange
    content = ""

    # Act & Assert
    with pytest.raises(ValueError):
        extraction_chain.invoke({"content": content})
```

##### TC-EXT-003: Partial Information Extraction
```python
def test_extraction_partial_info():
    """Test extraction when only some fields are present."""
    # Arrange
    content = "Buy when price breaks above resistance."

    # Act
    result = extraction_chain.invoke({"content": content})

    # Assert
    assert 'entry_conditions' in result
    # Exit conditions should be empty list, not None
    assert 'exit_conditions' in result
    assert result.get('exit_conditions', []) == [] or result.get('exit_conditions') is not None
```

##### TC-EXT-004: Indicator Recognition
```python
def test_extraction_indicators():
    """Test that common indicators are recognized."""
    # Arrange
    content = "Uses RSI 14, MACD 12 26 9, and ATR 14."

    # Act
    result = extraction_chain.invoke({"content": content})

    # Assert
    assert 'indicators_used' in result
    indicators = result['indicators_used']
    assert any('RSI' in ind for ind in indicators)
    assert any('MACD' in ind for ind in indicators)
```

##### TC-EXT-005: Time Filter Extraction
```python
def test_extraction_time_filters():
    """Test extraction of time-based filters."""
    # Arrange
    content = "Trade only during London and New York sessions. Avoid news hours."

    # Act
    result = extraction_chain.invoke({"content": content})

    # Assert
    assert 'time_filters' in result
    assert len(result['time_filters']) > 0
```

---

### 2.3 Search Chain Module (`analyst_agent/chains/search.py`)

**Purpose:** Test KB search chain functionality

#### Test Cases

##### TC-SRC-001: Query Generation
```python
def test_generate_search_queries():
    """Test generating search queries from concepts."""
    # Arrange
    state = {
        'entry_conditions': ['RSI oversold', 'price crosses EMA'],
        'exit_conditions': ['take profit 1:2', 'ATR stop loss'],
        'indicators': ['RSI', 'EMA', 'ATR']
    }

    # Act
    queries = generate_search_queries(state)

    # Assert
    assert len(queries) >= 3
    assert len(queries) <= 5  # Reasonable limit
    assert all(isinstance(q, str) for q in queries)
```

##### TC-SRC-002: Result Ranking
```python
def test_search_results_ranking():
    """Test that search results are ranked by relevance."""
    # Arrange
    state = {
        'entry_conditions': ['breakout'],
        'exit_conditions': [],
        'filters': []
    }

    # Act
    results = search_kb_chain.invoke(state)

    # Assert
    if results['kb_references']:
        distances = [r['distance'] for r in results['kb_references']]
        # Check sorted (lower distance = more relevant)
        assert distances == sorted(distances)
```

##### TC-SRC-003: Empty State Handling
```python
def test_search_empty_state():
    """Test search with minimal state information."""
    # Arrange
    state = {
        'entry_conditions': [],
        'exit_conditions': [],
        'filters': []
    }

    # Act
    results = search_kb_chain.invoke(state)

    # Assert
    assert 'kb_references' in results
    # Should return empty or generic results
    assert isinstance(results['kb_references'], list)
```

---

### 2.4 Graph State Module (`analyst_agent/graph/state.py`)

**Purpose:** Test TypedDict state definition and validation

#### Test Cases

##### TC-STA-001: State Initialization
```python
def test_state_initialization():
    """Test proper state initialization."""
    # Arrange & Act
    state = AnalystState(
        input_path="test.json",
        input_type="nprd",
        transcript="",
        keywords=[],
        entry_conditions=[],
        exit_conditions=[],
        filters=[],
        time_filters=[],
        kb_references=[],
        missing_info=[],
        user_answers={},
        trd_content="",
        trd_path="",
        current_step="start",
        needs_user_input=False,
        completed=False
    )

    # Assert
    assert state['input_path'] == "test.json"
    assert state['current_step'] == "start"
    assert state['completed'] is False
```

##### TC-STA-002: State Transition
```python
def test_state_transition():
    """Test state updates through workflow."""
    # Arrange
    state = AnalystState(
        input_path="test.json",
        input_type="nprd",
        transcript="",
        keywords=[],
        entry_conditions=[],
        exit_conditions=[],
        filters=[],
        time_filters=[],
        kb_references=[],
        missing_info=[],
        user_answers={},
        trd_content="",
        trd_path="",
        current_step="start",
        needs_user_input=False,
        completed=False
    )

    # Act
    state['current_step'] = "parsed"
    state['transcript'] = "Sample transcript"
    state['keywords'] = ["RSI", "EMA"]

    # Assert
    assert state['current_step'] == "parsed"
    assert len(state['keywords']) == 2
```

---

### 2.5 Graph Nodes Module (`analyst_agent/graph/nodes.py`)

**Purpose:** Test individual workflow nodes

#### Test Cases

##### TC-NOD-001: Parse Input Node (NPRD)
```python
def test_parse_input_node_nprd():
    """Test parsing NPRD JSON input."""
    # Arrange
    state = AnalystState(
        input_path="fixtures/test_nprd.json",
        input_type="nprd",
        transcript="",
        keywords=[],
        entry_conditions=[],
        exit_conditions=[],
        filters=[],
        time_filters=[],
        kb_references=[],
        missing_info=[],
        user_answers={},
        trd_content="",
        trd_path="",
        current_step="start",
        needs_user_input=False,
        completed=False
    )

    # Act
    result = parse_input_node(state)

    # Assert
    assert result['current_step'] == "parsed"
    assert len(result['transcript']) > 0
    assert len(result['keywords']) > 0
```

##### TC-NOD-002: Parse Input Node (Strategy Doc)
```python
def test_parse_input_node_strategy_doc():
    """Test parsing markdown strategy document."""
    # Arrange
    state = AnalystState(
        input_path="fixtures/test_strategy.md",
        input_type="strategy_doc",
        transcript="",
        keywords=[],
        entry_conditions=[],
        exit_conditions=[],
        filters=[],
        time_filters=[],
        kb_references=[],
        missing_info=[],
        user_answers={},
        trd_content="",
        trd_path="",
        current_step="start",
        needs_user_input=False,
        completed=False
    )

    # Act
    result = parse_input_node(state)

    # Assert
    assert result['current_step'] == "parsed"
    assert len(result['transcript']) > 0
```

##### TC-NOD-003: Extract Concepts Node
```python
@pytest.mark.integration
def test_extract_concepts_node():
    """Test concept extraction node."""
    # Arrange
    state = AnalystState(
        input_path="test.json",
        input_type="nprd",
        transcript="Buy when RSI < 30. Sell when RSI > 70.",
        keywords=["RSI"],
        entry_conditions=[],
        exit_conditions=[],
        filters=[],
        time_filters=[],
        kb_references=[],
        missing_info=[],
        user_answers={},
        trd_content="",
        trd_path="",
        current_step="parsed",
        needs_user_input=False,
        completed=False
    )

    # Act
    result = extract_concepts_node(state)

    # Assert
    assert result['current_step'] == "extracted"
    assert len(result['entry_conditions']) > 0
    assert len(result['exit_conditions']) > 0
```

##### TC-NOD-004: Identify Gaps Node
```python
def test_identify_gaps_node_with_gaps():
    """Test gap identification with missing info."""
    # Arrange
    state = AnalystState(
        input_path="test.json",
        input_type="nprd",
        transcript="Buy when RSI low.",
        keywords=[],
        entry_conditions=["RSI < 30"],
        exit_conditions=[],  # Missing exit conditions
        filters=[],
        time_filters=[],
        kb_references=[],
        missing_info=[],
        user_answers={},
        trd_content="",
        trd_path="",
        current_step="extracted",
        needs_user_input=False,
        completed=False
    )

    # Act
    result = identify_gaps_node(state)

    # Assert
    assert result['current_step'] == "gaps_identified"
    assert len(result['missing_info']) > 0
    assert result['needs_user_input'] is True
```

##### TC-NOD-005: Generate TRD Node
```python
def test_generate_trd_node():
    """Test TRD generation node."""
    # Arrange
    state = AnalystState(
        input_path="test.json",
        input_type="nprd",
        transcript="Test strategy",
        keywords=[],
        entry_conditions=["RSI < 30"],
        exit_conditions=["RSI > 70"],
        filters=["London session"],
        time_filters=[],
        kb_references=[],
        missing_info=[],
        user_answers={},
        trd_content="",
        trd_path="",
        current_step="gaps_identified",
        needs_user_input=False,
        completed=False
    )

    # Act
    result = generate_trd_node(state)

    # Assert
    assert result['current_step'] == "generated"
    assert len(result['trd_content']) > 0
    assert result['trd_path'] is not None
    assert result['completed'] is True
    assert result['trd_path'].endswith('.md')
```

##### TC-NOD-006: Has Gaps Conditional Edge
```python
def test_has_gaps_conditional():
    """Test conditional edge for gap detection."""
    # Test with gaps
    state_with_gaps = {
        'needs_user_input': True,
        'missing_info': ['position sizing']
    }
    assert has_gaps(state_with_gaps) == "ask_user"

    # Test without gaps
    state_without_gaps = {
        'needs_user_input': False,
        'missing_info': []
    }
    assert has_gaps(state_without_gaps) == "generate"
```

---

### 2.6 File I/O Utilities (`analyst_agent/utils/file_io.py`)

**Purpose:** Test file parsing utilities

#### Test Cases

##### TC-IO-001: Parse Valid NPRD JSON
```python
def test_parse_valid_nprd():
    """Test parsing valid NPRD JSON file."""
    # Arrange
    test_file = Path("fixtures/valid_nprd.json")

    # Act
    result = parse_nprd_file(test_file)

    # Assert
    assert 'video_id' in result
    assert 'transcript' in result
    assert 'keywords' in result
    assert len(result['transcript']) > 0
```

##### TC-IO-002: Parse Malformed JSON
```python
def test_parse_malformed_json():
    """Test handling of malformed JSON."""
    # Arrange
    test_file = Path("fixtures/malformed.json")

    # Act & Assert
    with pytest.raises(ValueError, match="Invalid JSON"):
        parse_nprd_file(test_file)
```

##### TC-IO-003: File Not Found
```python
def test_file_not_found():
    """Test handling of missing file."""
    # Arrange
    test_file = Path("nonexistent.json")

    # Act & Assert
    with pytest.raises(FileNotFoundError):
        parse_nprd_file(test_file)
```

##### TC-IO-004: Parse Strategy Document
```python
def test_parse_strategy_document():
    """Test parsing markdown strategy document."""
    # Arrange
    test_file = Path("fixtures/strategy.md")

    # Act
    result = parse_strategy_document(test_file)

    # Assert
    assert len(result) > 0
    assert 'content' in result or isinstance(result, str)
```

---

### 2.7 Gap Detector Utilities (`analyst_agent/utils/gap_detector.py`)

**Purpose:** Test missing information detection

#### Test Cases

##### TC-GAP-001: Detect Missing Exit Conditions
```python
def test_detect_missing_exit_conditions():
    """Test detection of missing exit conditions."""
    # Arrange
    state = {
        'entry_conditions': ['RSI < 30'],
        'exit_conditions': [],  # Missing
        'filters': ['London session'],
        'indicators': ['RSI']
    }

    # Act
    gaps = detect_gaps(state)

    # Assert
    assert any('exit' in gap.lower() for gap in gaps)
```

##### TC-GAP-002: Detect Missing Position Sizing
```python
def test_detect_missing_position_sizing():
    """Test detection of missing position sizing."""
    # Arrange
    state = {
        'entry_conditions': ['Buy signal'],
        'exit_conditions': ['Sell signal'],
        'position_sizing': None,  # Missing
        'risk_per_trade': None
    }

    # Act
    gaps = detect_gaps(state)

    # Assert
    assert any('position' in gap.lower() or 'sizing' in gap.lower() for gap in gaps)
```

##### TC-GAP-003: Complete State No Gaps
```python
def test_complete_state_no_gaps():
    """Test that complete state has no gaps."""
    # Arrange
    state = {
        'entry_conditions': ['RSI < 30'],
        'exit_conditions': ['TP at 1:2', 'SL at ATR 1.5'],
        'filters': ['London session'],
        'indicators': ['RSI', 'ATR'],
        'position_sizing': 'risk_based',
        'risk_per_trade': '1%'
    }

    # Act
    gaps = detect_gaps(state)

    # Assert
    assert len(gaps) == 0
```

---

### 2.8 TRD Template Utilities (`analyst_agent/utils/trd_template.py`)

**Purpose:** Test TRD markdown generation

#### Test Cases

##### TC-TRD-001: Generate Complete TRD
```python
def test_generate_complete_trd():
    """Test generating complete TRD markdown."""
    # Arrange
    state = {
        'strategy_name': 'RSI Scalper',
        'input_path': 'test.json',
        'entry_conditions': ['RSI < 30'],
        'exit_conditions': ['TP 1:2', 'SL ATR 1.5'],
        'filters': ['London session'],
        'time_filters': ['8am-12pm GMT'],
        'indicators': ['RSI', 'ATR'],
        'kb_references': [
            {'title': 'RSI Strategies', 'content': 'Best practices...'}
        ],
        'missing_info': [],
        'user_answers': {}
    }

    # Act
    trd = generate_trd(state)

    # Assert
    assert '# Trading Strategy:' in trd
    assert '## Entry Logic' in trd
    assert '## Exit Logic' in trd
    assert '## Filters' in trd
    assert '## Indicators' in trd
    assert '## Knowledge Base References' in trd
    assert 'RSI Scalper' in trd
```

##### TC-TRD-002: Generate TRD with Missing Info
```python
def test_generate_trd_with_missing_info():
    """Test TRD generation includes missing info section."""
    # Arrange
    state = {
        'strategy_name': 'Incomplete Strategy',
        'input_path': 'test.json',
        'entry_conditions': ['Buy signal'],
        'exit_conditions': [],  # Missing
        'filters': [],
        'time_filters': [],
        'indicators': [],
        'kb_references': [],
        'missing_info': ['exit_logic: take profit', 'exit_logic: stop loss'],
        'user_answers': {}
    }

    # Act
    trd = generate_trd(state)

    # Assert
    assert '## Missing Information' in trd
    assert 'take profit' in trd
    assert 'stop loss' in trd
```

##### TC-TRD-003: YAML Frontmatter Generation
```python
def test_yaml_frontmatter():
    """Test YAML frontmatter is properly formatted."""
    # Arrange
    state = {
        'strategy_name': 'Test Strategy',
        'input_path': 'test.json',
        'entry_conditions': [],
        'exit_conditions': [],
        'filters': [],
        'time_filters': [],
        'indicators': [],
        'kb_references': [],
        'missing_info': [],
        'user_answers': {}
    }

    # Act
    trd = generate_trd(state)

    # Assert
    assert trd.startswith('---')
    lines = trd.split('\n')
    assert lines[0] == '---'
    assert any('strategy_name:' in line for line in lines[:10])
    assert any('generated_at:' in line for line in lines[:10])
    assert '---' in lines[1:15]  # End of frontmatter
```

---

## 3. Integration Test Requirements

### 3.1 End-to-End Workflow Tests

#### TC-E2E-001: Complete TRD Generation
```python
@pytest.mark.integration
def test_complete_trd_generation():
    """Test end-to-end TRD generation from NPRD input."""
    # Arrange
    input_file = "fixtures/complete_nprd.json"
    output_dir = tmp_path / "trds"

    # Act
    result = run_analyst_workflow(
        input_path=input_file,
        input_type="nprd",
        auto_mode=True
    )

    # Assert
    assert result['completed'] is True
    assert Path(result['trd_path']).exists()

    # Verify TRD content
    trd_content = Path(result['trd_path']).read_text()
    assert '## Entry Logic' in trd_content
    assert '## Exit Logic' in trd_content
    assert '## Knowledge Base References' in trd_content
```

#### TC-E2E-002: Human-in-the-Loop Workflow
```python
@pytest.mark.integration
def test_hitl_workflow():
    """Test workflow with user input for missing information."""
    # Arrange
    input_file = "fixtures/incomplete_nprd.json"
    user_inputs = {
        'position sizing': 'risk_based',
        'take profit': '1.5x risk',
        'stop loss': 'ATR 1.5'
    }

    # Act - Mock user input
    with mock.patch('builtins.input', side_effect=user_inputs.values()):
        result = run_analyst_workflow(
            input_path=input_file,
            input_type="nprd",
            auto_mode=False
        )

    # Assert
    assert result['completed'] is True
    assert len(result['user_answers']) == 3
    assert result['user_answers']['position sizing'] == 'risk_based'
```

#### TC-E2E-003: Chunked NPRD Processing
```python
@pytest.mark.integration
def test_chunked_nprd_processing():
    """Test processing of NPRD split across multiple chunks."""
    # Arrange
    chunk_dir = "fixtures/chunked_nprd/"

    # Act
    result = run_analyst_workflow(
        input_path=chunk_dir,
        input_type="nprd",
        auto_mode=True
    )

    # Assert
    assert result['completed'] is True
    # Verify chunks were aggregated
    assert len(result['transcript']) > 1000  # Reasonable length
```

### 3.2 Knowledge Base Integration Tests

#### TC-KB-INT-001: KB Search Integration
```python
@pytest.mark.integration
def test_kb_search_integration():
    """Test KB search with real ChromaDB."""
    # Arrange
    client = AnalystKBClient()

    # Act
    results = client.search("trailing stop implementation", n_results=5)

    # Assert
    assert len(results) > 0
    assert all('title' in r for r in results)
    assert all('content' in r for r in results)
```

#### TC-KB-INT-002: KB Query Generation
```python
@pytest.mark.integration
def test_kb_query_generation():
    """Test that generated queries return relevant results."""
    # Arrange
    state = {
        'entry_conditions': ['breakout', 'volume confirmation'],
        'exit_conditions': ['trailing stop', 'partial close'],
        'indicators': ['VWAP', 'Volume']
    }

    # Act
    queries = generate_search_queries(state)
    client = AnalystKBClient()
    all_results = []

    for query in queries:
        results = client.search(query, n_results=3)
        all_results.extend(results)

    # Assert
    assert len(queries) >= 3
    assert len(all_results) > 0  # At least some matches
```

### 3.3 LLM Integration Tests

#### TC-LLM-001: Extraction Chain with LLM
```python
@pytest.mark.integration
@pytest.mark.llm
def test_extraction_chain_with_llm():
    """Test extraction chain with real LLM."""
    # Arrange
    content = """
    Strategy: RSI Mean Reversion
    Entry: RSI < 30, price crosses above EMA 20
    Exit: RSI > 70 or 1:2 risk-reward
    Stop Loss: ATR * 1.5
    Filters: London session only, avoid news
    """

    # Act
    result = extraction_chain.invoke({"content": content})

    # Assert
    assert 'entry_conditions' in result
    assert 'exit_conditions' in result
    assert 'indicators_used' in result
    assert len(result['entry_conditions']) > 0
```

#### TC-LLM-002: Generation Chain with LLM
```python
@pytest.mark.integration
@pytest.mark.llm
def test_generation_chain_with_llm():
    """Test TRD generation with real LLM."""
    # Arrange
    state = {
        'strategy_name': 'Test Strategy',
        'entry_conditions': ['RSI < 30'],
        'exit_conditions': ['RSI > 70'],
        'indicators': ['RSI'],
        'kb_references': [],
        'missing_info': []
    }

    # Act
    trd_content = generation_chain.invoke(state)

    # Assert
    assert len(trd_content) > 500  # Reasonable length
    assert 'Entry Logic' in trd_content
    assert 'Exit Logic' in trd_content
```

---

## 4. End-to-End Test Scenarios

### 4.1 Scenario-Based Testing

#### SCENARIO-001: Simple Strategy (Complete Information)
```bash
# Given: A complete NPRD output with all required fields
# When: Run analyst-cli generate
# Then: Valid TRD generated without user prompts

analyst-cli generate \
  --input fixtures/simple_strategy_nprd.json \
  --output test_outputs/simple_strategy.md

# Validation:
- [ ] TRD file created
- [ ] All sections populated
- [ ] No "TODO" placeholders
- [ ] KB references present
```

#### SCENARIO-002: Complex Multi-Indicator Strategy
```bash
# Given: Strategy with 5+ indicators
# When: Generate TRD
# Then: All indicators listed correctly

analyst-cli generate \
  --input fixtures/multi_indicator_nprd.json \
  --output test_outputs/multi_indicator.md

# Validation:
- [ ] All 5+ indicators in Indicators table
- [ ] Entry conditions mention all relevant indicators
- [ ] KB references include indicator-specific articles
```

#### SCENARIO-003: Incomplete Strategy (HITL Required)
```bash
# Given: NPRD missing stop loss strategy
# When: Run in interactive mode
# Then: User prompted for missing info

analyst-cli generate \
  --input fixtures/incomplete_nprd.json \
  --interactive

# Expected prompts:
# 1. "What is the stop loss strategy?"
# 2. "What is the take profit target?"

# Validation:
- [ ] Prompts appear sequentially
- [ ] User input incorporated into TRD
- [ ] Missing info section updated
```

#### SCENARIO-004: Auto Mode with Defaults
```bash
# Given: Incomplete NPRD
# When: Run in auto mode
# Then: Defaults used for missing info

analyst-cli generate \
  --input fixtures/incomplete_nprd.json \
  --auto

# Validation:
- [ ] No interactive prompts
- [ ] Default values from config used
- [ ] TRD generated successfully
- [ ] Missing info section shows defaults
```

#### SCENARIO-005: Strategy Document Input (Non-NPRD)
```bash
# Given: Markdown strategy document
# When: Generate TRD
# Then: TRD created from strategy doc

analyst-cli generate \
  --input fixtures/manual_strategy.md \
  --type strategy_doc \
  --output test_outputs/from_manual.md

# Validation:
- [ ] Markdown parsed correctly
- [ ] Content extracted and structured
- [ ] TRD sections populated
```

---

## 5. Manual Testing Procedures

### 5.1 Pre-Test Setup

```bash
# 1. Environment Setup
cd /home/mubarkahimself/Desktop/QUANTMINDX
source venv/bin/activate  # or: python -m venv venv && source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
pip install pytest pytest-cov pytest-mock

# 3. Verify ChromaDB exists
ls -la data/chromadb/chroma.sqlite3

# 4. Run health check
python -m analyst_agent.cli health
```

### 5.2 Manual Test Cases

#### MT-001: CLI Help Display
```bash
# Test: CLI help is accessible and complete
python tools/analyst_cli.py --help

# Expected:
# - All commands listed
# - Options documented
# - Usage examples shown
```

#### MT-002: List Available NPRD Outputs
```bash
# Test: List command shows available NPRD files
python tools/analyst_cli.py list

# Expected:
# - List of JSON files in outputs/videos/
# - File sizes and timestamps
# - At least 1 file listed
```

#### MT-003: KB Statistics Display
```bash
# Test: Stats command shows KB info
python tools/analyst_cli.py stats

# Expected:
# - Collection name: analyst_kb
# - Article count: ~330
# - Storage path: data/chromadb
```

#### MT-004: Interactive TRD Generation
```bash
# Test: Generate TRD with user interaction
echo "risk_based
1.5x risk
ATR 1.5" | python tools/analyst_cli.py \
  --input fixtures/incomplete_nprd.json \
  --interactive

# Expected:
# - User prompted 3 times
# - Inputs accepted
# - TRD file created
# - Success message shown
```

#### MT-005: Auto Mode TRD Generation
```bash
# Test: Generate TRD without prompts
python tools/analyst_cli.py \
  --input fixtures/test_nprd.json \
  --auto

# Expected:
# - No interactive prompts
# - Processing messages shown
# - TRD file created
# - Path to TRD displayed
```

### 5.3 TRD Quality Validation Checklist

For each generated TRD, manually verify:

```markdown
## TRD Quality Checklist

### Structure Validation
- [ ] File exists at specified path
- [ ] YAML frontmatter present and valid
- [ ] All required sections exist:
  - [ ] # Trading Strategy: [Name]
  - [ ] ## Overview
  - [ ] ## Entry Logic
  - [ ] ## Exit Logic
  - [ ] ## Filters
  - [ ] ## Indicators & Settings
  - [ ] ## Position Sizing & Risk Management
  - [ ] ## Knowledge Base References
  - [ ] ## Next Steps

### Content Quality
- [ ] No "TODO" placeholders (except in approved sections)
- [ ] No "N/A" in main content
- [ ] Entry conditions are clear and actionable
- [ ] Exit conditions are complete (TP, SL, trailing)
- [ ] Filters are specific (time, volatility, etc.)
- [ ] Indicators listed with settings
- [ ] KB references are relevant
- [ ] Markdown formatting is correct

### Completeness (if info in source)
- [ ] Strategy name extracted
- [ ] Entry trigger specified
- [ ] Entry confirmation (if in source)
- [ ] Take profit strategy
- [ ] Stop loss strategy
- [ ] Trailing stop (if mentioned)
- [ ] Time filters
- [ ] Market condition filters
- [ ] Indicator settings
- [ ] Position sizing (if mentioned)
- [ ] Risk per trade (if mentioned)

### Readability
- [ ] Headers use correct # level
- [ ] Code blocks have language identifiers
- [ ] Tables properly formatted
- [ ] Lists use correct syntax (- or 1.)
- [ ] No broken links or references
- [ ] Technical terminology correct
```

---

## 6. Test Data Requirements

### 6.1 Test Fixtures Directory Structure

```
tools/analyst_agent/tests/
├── fixtures/
│   ├── nprd/
│   │   ├── valid_nprd.json              # Valid NPRD output
│   │   ├── incomplete_nprd.json         # Missing exit conditions
│   │   ├── multi_indicator_nprd.json    # 5+ indicators
│   │   ├── chunked_nprd/                # Multiple chunks
│   │   │   ├── part_1.json
│   │   │   ├── part_2.json
│   │   │   └── part_3.json
│   │   └── malformed.json               # Invalid JSON
│   ├── strategy_docs/
│   │   ├── simple_strategy.md           # Basic strategy
│   │   ├── complex_strategy.md          # Multi-indicator
│   │   └── incomplete_strategy.md       # Missing sections
│   └── expected_trds/
│       ├── simple_strategy_expected.md  # Expected output
│       └── complex_strategy_expected.md
└── tmp/                                  # Temporary test outputs
```

### 6.2 Sample NPRD Fixture (`valid_nprd.json`)

```json
{
  "video_id": "test_abc123",
  "video_title": "RSI Scalping Strategy",
  "video_url": "https://youtube.com/watch?v=test",
  "processed_at": "2026-01-27T10:00:00Z",
  "chunks": [
    {
      "chunk_id": 1,
      "start_time": "00:00",
      "end_time": "05:00",
      "transcript": "In this video I'll show you a simple RSI scalping strategy for EURUSD.",
      "visual_description": "Chart showing EURUSD 5-minute",
      "ocr_text": "RSI: 28.5, EMA: 1.0850",
      "keywords": ["RSI", "scalping", "EURUSD"]
    },
    {
      "chunk_id": 2,
      "start_time": "05:00",
      "end_time": "10:00",
      "transcript": "Enter when RSI goes below 30 and price crosses above the 20 EMA. Set stop loss at ATR times 1.5 and take profit at 1.5 times your risk.",
      "visual_description": "Entry signal shown on chart",
      "ocr_text": "SL: 1.5*ATR, TP: 1:2RR",
      "keywords": ["entry", "stop loss", "take profit"]
    }
  ],
  "summary": {
    "full_transcript": "In this video I'll show you a simple RSI scalping strategy for EURUSD. Enter when RSI goes below 30 and price crosses above the 20 EMA. Set stop loss at ATR times 1.5 and take profit at 1.5 times your risk.",
    "all_keywords": ["RSI", "scalping", "EURUSD", "entry", "stop loss", "take profit", "EMA", "ATR"],
    "visual_elements": ["charts", "indicators"]
  }
}
```

### 6.3 Sample Incomplete NPRD Fixture (`incomplete_nprd.json`)

```json
{
  "video_id": "test_incomplete",
  "video_title": "Partial Strategy",
  "processed_at": "2026-01-27T10:00:00Z",
  "chunks": [
    {
      "chunk_id": 1,
      "transcript": "Buy when RSI is low. Sell when RSI is high.",
      "keywords": ["RSI", "buy", "sell"]
    }
  ],
  "summary": {
    "full_transcript": "Buy when RSI is low. Sell when RSI is high.",
    "all_keywords": ["RSI", "buy", "sell"],
    "visual_elements": []
  }
}
```

### 6.4 Sample Strategy Document Fixture (`simple_strategy.md`)

```markdown
# RSI Mean Reversion Strategy

## Entry Rules

- RSI must be below 30 (oversold)
- Price must cross above EMA 20
- Enter long at close of candle

## Exit Rules

- Take profit at 1:2 risk-reward
- Stop loss at ATR * 1.5 below entry

## Filters

- Only trade London session (8am-12pm GMT)
- Avoid high-impact news events
- Minimum ATR of 0.0010

## Indicators

- RSI: Period 14
- EMA: Period 20
- ATR: Period 14
```

---

## 7. Performance Testing

### 7.1 Performance Targets

| Operation | Target | Maximum |
|-----------|--------|---------|
| CLI startup | < 1s | 2s |
| NPRD parsing | < 100ms | 500ms |
| Concept extraction | < 5s | 10s |
| KB search | < 500ms | 1s |
| TRD generation | < 10s | 30s |
| End-to-end workflow | < 20s | 60s |

### 7.2 Performance Test Cases

#### TC-PERF-001: CLI Startup Time
```python
def test_cli_startup_time():
    """Test CLI starts within target time."""
    # Act
    start = time.time()
    result = subprocess.run(['python', 'tools/analyst_cli.py', '--help'])
    duration = time.time() - start

    # Assert
    assert duration < 2.0, f"CLI startup took {duration:.2f}s (target: <2s)"
```

#### TC-PERF-002: KB Search Latency
```python
@pytest.mark.benchmark
def test_kb_search_latency():
    """Test KB search performance."""
    # Arrange
    client = AnalystKBClient()
    iterations = 10

    # Act
    start = time.time()
    for _ in range(iterations):
        client.search("RSI strategy", n_results=5)
    avg_duration = (time.time() - start) / iterations

    # Assert
    assert avg_duration < 1.0, f"Average search took {avg_duration:.2f}s (target: <1s)"
```

#### TC-PERF-003: End-to-End Workflow Time
```python
@pytest.mark.benchmark
def test_e2e_workflow_duration():
    """Test complete workflow duration."""
    # Arrange
    input_file = "fixtures/valid_nprd.json"

    # Act
    start = time.time()
    result = run_analyst_workflow(input_file, "nprd", auto_mode=True)
    duration = time.time() - start

    # Assert
    assert result['completed'] is True
    assert duration < 60, f"E2E workflow took {duration:.2f}s (target: <60s)"
```

### 7.3 Load Testing

#### TC-LOAD-001: Concurrent Workflows
```python
@pytest.mark.load
def test_concurrent_workflows():
    """Test multiple concurrent workflow executions."""
    # Arrange
    input_files = [
        "fixtures/valid_nprd.json",
        "fixtures/multi_indicator_nprd.json",
        "fixtures/simple_strategy.md"
    ]

    # Act
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(run_analyst_workflow, f, "nprd", True)
            for f in input_files
        ]
        results = [f.result() for f in futures]

    # Assert
    assert all(r['completed'] for r in results)
```

---

## 8. Security Testing

### 8.1 Security Test Cases

#### TC-SEC-001: Path Traversal Prevention
```python
def test_path_traversal_prevention():
    """Test that path traversal attacks are prevented."""
    # Arrange & Act & Assert
    with pytest.raises(ValueError):
        parse_nprd_file(Path("../../../etc/passwd"))
```

#### TC-SEC-002: Malicious JSON Handling
```python
def test_malicious_json_handling():
    """Test handling of malicious JSON payload."""
    # Arrange
    malicious_json = '{"__proto__": "polluted"}'
    test_file = Path("fixtures/malicious.json")
    test_file.write_text(malicious_json)

    # Act & Assert
    with pytest.raises(ValueError):
        parse_nprd_file(test_file)
```

#### TC-SEC-003: API Key Protection
```python
def test_api_key_not_logged():
    """Test that API keys are not logged."""
    # Arrange
    os.environ['OPENAI_API_KEY'] = 'sk-test-secret-key'

    # Act
    with mock.patch('builtins.print') as mock_print:
        run_analyst_workflow("fixtures/valid_nprd.json", "nprd", True)

    # Assert - API key should not appear in any output
    for call in mock_print.call_args_list:
        assert 'sk-test-secret-key' not in str(call)
```

---

## 9. Test Environment Setup

### 9.1 Initial Setup

```bash
# 1. Create test directories
mkdir -p tools/analyst_agent/tests/{fixtures,strategy_docs,expected_trds,tmp}
mkdir -p docs/testing/fixtures

# 2. Install test dependencies
pip install pytest pytest-cov pytest-mock pytest-benchmark pytest-asyncio

# 3. Create pytest configuration
cat > tools/analyst_agent/tests/pytest.ini << 'EOF'
[pytest]
testpaths = .
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts =
    -v
    --tb=short
    --cov=analyst_agent
    --cov-report=html
    --cov-report=term-missing
    --cov-fail-under=70
markers =
    unit: Unit tests
    integration: Integration tests
    e2e: End-to-end tests
    llm: Tests requiring LLM API
    benchmark: Performance tests
    slow: Slow-running tests
EOF
```

### 9.2 Test Configuration

#### `tools/analyst_agent/tests/conftest.py`

```python
"""Pytest configuration and fixtures."""

import pytest
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

@pytest.fixture
def mock_llm():
    """Mock LLM for unit tests."""
    from unittest.mock import Mock
    mock = Mock()
    mock.invoke.return_value = {
        'entry_conditions': ['Test condition'],
        'exit_conditions': ['Test exit'],
        'indicators_used': ['RSI']
    }
    return mock

@pytest.fixture
def sample_state():
    """Sample AnalystState for testing."""
    return {
        'input_path': 'test.json',
        'input_type': 'nprd',
        'transcript': 'Test transcript',
        'keywords': ['test'],
        'entry_conditions': [],
        'exit_conditions': [],
        'filters': [],
        'time_filters': [],
        'kb_references': [],
        'missing_info': [],
        'user_answers': {},
        'trd_content': '',
        'trd_path': '',
        'current_step': 'start',
        'needs_user_input': False,
        'completed': False
    }

@pytest.fixture
def tmp_path(tmp_path_factory):
    """Temporary path for test outputs."""
    return tmp_path_factory.mktemp('test_outputs')

@pytest.fixture(scope="session")
def test_env():
    """Set up test environment variables."""
    os.environ['OPENAI_API_KEY'] = 'sk-test-key'
    os.environ['CHROMA_PATH'] = 'data/chromadb'
    yield
    # Cleanup
    del os.environ['OPENAI_API_KEY']
```

---

## 10. Coverage Metrics & Reporting

### 10.1 Coverage Commands

```bash
# Run all tests with coverage
pytest tools/analyst_agent/tests/ --cov=analyst_agent --cov-report=html

# Generate coverage report
pytest tools/analyst_agent/tests/ --cov=analyst_agent --cov-report=term-missing

# Generate HTML report
pytest tools/analyst_agent/tests/ --cov=analyst_agent --cov-report=html:htmlcov

# Check coverage threshold (fail if below 70%)
pytest tools/analyst_agent/tests/ --cov=analyst_agent --cov-fail-under=70

# Generate XML report for CI
pytest tools/analyst_agent/tests/ --cov=analyst_agent --cov-report=xml
```

### 10.2 Coverage Targets by Module

| Module | Statement Target | Branch Target | Function Target |
|--------|-----------------|---------------|-----------------|
| `kb/client.py` | 80% | 75% | 85% |
| `chains/extraction.py` | 75% | 70% | 80% |
| `chains/search.py` | 75% | 70% | 80% |
| `chains/generation.py` | 75% | 70% | 80% |
| `graph/workflow.py` | 80% | 75% | 85% |
| `graph/nodes.py` | 80% | 75% | 85% |
| `utils/file_io.py` | 85% | 80% | 90% |
| `utils/gap_detector.py` | 85% | 80% | 90% |
| `utils/trd_template.py` | 85% | 80% | 90% |
| `cli/commands.py` | 70% | 65% | 75% |

### 10.3 Interpreting Coverage Reports

```
Name                                  Stmts   Miss  Cover   Missing
-----------------------------------------------------------------------
analyst_agent/kb/client.py              80      12    85%   23-27, 45-49
analyst_agent/chains/extraction.py      60      15    75%   34-38, 52-55
analyst_agent/graph/nodes.py           120      30    75%   67-72, 89-95
-----------------------------------------------------------------------
TOTAL                                  500     150    70%
```

- **Stmts**: Total statements
- **Miss**: Statements not executed
- **Cover**: Coverage percentage
- **Missing**: Line numbers not covered

---

## 11. Continuous Integration

### 11.1 CI Pipeline Configuration

#### `.github/workflows/test.yml`

```yaml
name: Analyst Agent Tests

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov pytest-mock

    - name: Run unit tests
      run: |
        pytest tools/analyst_agent/tests/unit/ -v \
          --cov=analyst_agent \
          --cov-report=xml \
          --cov-fail-under=70

    - name: Run integration tests
      run: |
        pytest tools/analyst_agent/tests/integration/ -v -m "not llm"

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        files: ./coverage.xml
        fail_ci_if_error: false
```

### 11.2 Pre-Commit Hooks

#### `.pre-commit-config.yaml`

```yaml
repos:
  - repo: local
    hooks:
      - id: pytest-unit
        name: Run unit tests
        entry: pytest tools/analyst_agent/tests/unit/ -v
        language: system
        pass_filenames: false
        always_run: true

      - id: pytest-coverage
        name: Check coverage
        entry: pytest tools/analyst_agent/tests/unit/ --cov=analyst_agent --cov-fail-under=70
        language: system
        pass_filenames: false
        always_run: true
```

---

## 12. Test Execution Guidelines

### 12.1 Running Tests

```bash
# Run all tests
pytest tools/analyst_agent/tests/ -v

# Run only unit tests
pytest tools/analyst_agent/tests/unit/ -v

# Run only integration tests
pytest tools/analyst_agent/tests/integration/ -v

# Run tests for specific module
pytest tools/analyst_agent/tests/test_kb_client.py -v

# Run tests matching pattern
pytest tools/analyst_agent/tests/ -k "search" -v

# Run tests without LLM calls (faster)
pytest tools/analyst_agent/tests/ -m "not llm" -v

# Run only slow tests
pytest tools/analyst_agent/tests/ -m "slow" -v

# Run with verbose output
pytest tools/analyst_agent/tests/ -vv

# Stop on first failure
pytest tools/analyst_agent/tests/ -x

# Run failed tests only
pytest tools/analyst_agent/tests/ --lf
```

### 12.2 Test Organization

```
tools/analyst_agent/tests/
├── unit/                          # Fast, isolated tests
│   ├── test_kb_client.py
│   ├── test_extraction_chain.py
│   ├── test_search_chain.py
│   ├── test_graph_nodes.py
│   ├── test_file_io.py
│   ├── test_gap_detector.py
│   └── test_trd_template.py
│
├── integration/                   # Component interaction tests
│   ├── test_workflow_integration.py
│   ├── test_kb_integration.py
│   └── test_llm_integration.py
│
├── e2e/                          # Full workflow tests
│   ├── test_complete_trd_generation.py
│   └── test_hitl_workflow.py
│
├── performance/                  # Performance benchmarks
│   ├── test_cli_startup.py
│   ├── test_kb_search_latency.py
│   └── test_e2e_duration.py
│
├── security/                     # Security tests
│   ├── test_path_traversal.py
│   └── test_input_validation.py
│
└── fixtures/                     # Test data
    ├── nprd/
    ├── strategy_docs/
    └── expected_trds/
```

### 12.3 Test Markers

```python
@pytest.mark.unit           # Unit tests
@pytest.mark.integration    # Integration tests
@pytest.mark.e2e           # End-to-end tests
@pytest.mark.llm           # Tests requiring LLM API
@pytest.mark.benchmark     # Performance tests
@pytest.mark.slow          # Slow-running tests
```

### 12.4 Test Execution Best Practices

1. **Before committing**: Run unit tests + coverage check
   ```bash
   pytest tools/analyst_agent/tests/unit/ --cov=analyst_agent --cov-fail-under=70
   ```

2. **Before pushing**: Run all tests
   ```bash
   pytest tools/analyst_agent/tests/ -v
   ```

3. **Before merging**: Run full suite + integration
   ```bash
   pytest tools/analyst_agent/tests/ -v -m "not llm"
   pytest tools/analyst_agent/tests/integration/ -v -m "llm"
   ```

4. **Continuous monitoring**: Watch for coverage drops
   ```bash
   pytest tools/analyst_agent/tests/ --cov=analyst_agent --cov-report=term-missing
   ```

---

## Appendices

### Appendix A: Test Data Creation Scripts

#### Script: Generate Test NPRD Fixtures

```bash
#!/bin/bash
# scripts/generate_test_fixtures.sh

mkdir -p tools/analyst_agent/tests/fixtures/nprd

# Valid NPRD
cat > tools/analyst_agent/tests/fixtures/nprd/valid_nprd.json << 'EOF'
{
  "video_id": "test_valid_001",
  "video_title": "Complete RSI Strategy",
  "video_url": "https://test.com/video1",
  "processed_at": "2026-01-27T10:00:00Z",
  "chunks": [
    {
      "chunk_id": 1,
      "start_time": "00:00",
      "end_time": "05:00",
      "transcript": "Enter when RSI below 30 and price above EMA.",
      "keywords": ["RSI", "entry", "EMA"]
    },
    {
      "chunk_id": 2,
      "start_time": "05:00",
      "end_time": "10:00",
      "transcript": "Exit at 1:2 risk-reward or ATR 1.5 stop loss.",
      "keywords": ["exit", "stop loss", "take profit"]
    }
  ],
  "summary": {
    "full_transcript": "Enter when RSI below 30 and price above EMA. Exit at 1:2 risk-reward or ATR 1.5 stop loss.",
    "all_keywords": ["RSI", "entry", "EMA", "exit", "stop loss", "take profit"],
    "visual_elements": []
  }
}
EOF

echo "Test fixtures generated successfully"
```

### Appendix B: Troubleshooting Test Failures

| Issue | Common Cause | Solution |
|-------|--------------|----------|
| Import errors | Path not set | Add parent dir to sys.path in conftest.py |
| ChromaDB connection | DB not initialized | Run `python scripts/create_analyst_kb.py` |
| LLM API errors | Missing API key | Set `OPENAI_API_KEY` in `.env` |
| Timeout errors | Slow LLM response | Increase timeout in test config |
| Coverage too low | Missing tests | Add tests for uncovered lines |

### Appendix C: Test Metrics Dashboard

Track these metrics to ensure quality:

```bash
# Total tests
pytest tools/analyst_agent/tests/ --collect-only | grep "test session starts" -A 1

# Pass rate
pytest tools/analyst_agent/tests/ -v | grep "passed"

# Average test duration
pytest tools/analyst_agent/tests/ --durations=10

# Coverage trends
pytest tools/analyst_agent/tests/ --cov=analyst_agent --cov-report=html
# Compare htmlcov/index.html over time
```

---

**Document Version:** 1.0
**Last Updated:** 2026-01-27
**Maintained By:** Test Specialist Agent
**Status:** Ready for Implementation

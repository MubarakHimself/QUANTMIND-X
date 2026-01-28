# Analyst Agent Test Suite

This directory contains comprehensive tests for the Analyst Agent functionality.

## Test Structure

The test suite is organized into the following files:

1. `test_kb_client.py` - Tests for ChromaKBClient
   - Search functionality with mock results
   - Collection stats
   - create_analyst_kb filtering
   - Error handling and edge cases

2. `test_chains.py` - Tests for LangChain chains
   - extraction_chain with mock NPRD data
   - search_chain with mock KB
   - generation_chain with mock inputs
   - Chain error handling and integration

3. `test_graph.py` - Tests for LangGraph workflow
   - State transitions
   - auto_mode bypasses human_input
   - interactive mode includes HITL
   - Workflow error handling

4. `test_cli.py` - Tests for CLI commands
   - generate command
   - list command
   - complete command
   - CLI error handling and output formatting

5. `conftest.py` - pytest configuration and fixtures
   - Test data and mock fixtures
   - Custom markers

6. `requirements.txt` - Test dependencies
   - pytest and related packages

7. `run_tests.py` - Test runner script
   - Command line interface for running tests

## Running Tests

### Basic Test Execution

```bash
# Run all tests
python run_tests.py

# Run with verbose output
python run_tests.py -v

# Run specific test with keyword
python run_tests.py -k "search"
```

### Using pytest directly

```bash
# Run all tests
pytest

# Run specific test file
pytest test_kb_client.py

# Run tests with markers
pytest -m kb
pytest -m chains
pytest -m graph
pytest -m cli
pytest -m integration
```

### Coverage Reporting

```bash
# Generate HTML coverage report
python run_tests.py --cov=tools/analyst_agent --html

# Generate XML coverage report
python run_tests.py --cov=tools/analyst_agent --xml

# Run specific tests with coverage
pytest --cov=tools/analyst_agent test_kb_client.py
```

## Test Fixtures

The test suite provides several fixtures:

- `test_data_dir`: Temporary directory for test data
- `sample_nprd_data`: Sample NPRD data for testing
- `sample_trd_data`: Sample TRD data for testing
- `mock_chroma_db`: Mock ChromaDB client
- `mock_langchain`: Mock LangChain components
- `mock_workflow`: Mock workflow components
- `mock_cli`: Mock CLI runner

## Test Markers

The test suite uses the following markers:

- `kb`: Knowledge base related tests
- `chains`: LangChain chain related tests
- `graph`: LangGraph workflow related tests
- `cli`: CLI related tests
- `integration`: Integration tests

Example usage:
```bash
pytest -m kb  # Run only knowledge base tests
pytest -m chains  # Run only chain tests
pytest -m "kb or chains"  # Run knowledge base or chain tests
```

## Test Categories

### Knowledge Base Tests (`test_kb_client.py`)
- Search functionality with mock results
- Collection statistics
- Filtering logic
- Error handling
- Edge cases and boundary conditions

### Chain Tests (`test_chains.py`)
- Extraction chain functionality
- Search chain integration
- Generation chain output
- Chain error handling
- Integration testing

### Workflow Tests (`test_graph.py`)
- State transitions
- Auto mode behavior
- Interactive mode handling
- Error recovery
- Workflow visualization

### CLI Tests (`test_cli.py`)
- Command parsing
- Argument validation
- Output formatting
- Error handling
- Progress display

## Requirements

The test suite requires:

- Python 3.8+
- pytest>=7.0.0
- pytest-asyncio>=0.21.0
- pytest-mock>=3.10.0
- rich>=13.0.0
- chromadb (for actual tests)

Install test dependencies:
```bash
pip install -r requirements.txt
```

## Test Coverage

The test suite aims for comprehensive coverage including:

- Unit tests for individual components
- Integration tests for chain workflows
- End-to-end workflow testing
- Error handling scenarios
- Edge cases and boundary conditions
- Performance testing (where applicable)

## Contributing

To add new tests:

1. Create test functions in the appropriate test file
2. Use pytest fixtures for common setup/teardown
3. Add appropriate markers
4. Ensure tests are isolated and repeatable
5. Include both success and error scenarios

For more information on pytest, see the [pytest documentation](https://docs.pytest.org/).
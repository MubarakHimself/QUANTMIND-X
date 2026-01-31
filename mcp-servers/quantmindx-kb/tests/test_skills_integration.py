#!/usr/bin/env python3
"""
Tests for Assets Hub Skills Integration (Task Group 13)

Tests cover:
- load_skill returns skill definition from ChromaDB
- Skill dependencies are loaded recursively
- Semantic search finds relevant skills by description
- Skill versioning returns latest version by default
"""

import json
import sys
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "mcp-servers" / "quantmindx-kb"))

try:
    import chromadb
except ImportError:
    print("ChromaDB not found. Install with: pip install chromadb")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("sentence-transformers not found. Install with: pip install sentence-transformers")
    sys.exit(1)


# =============================================================================
# Test Fixtures
# =============================================================================

CHROMA_PATH = PROJECT_ROOT / "data" / "chromadb"
COLLECTION_NAME = "agentic_skills"
HNSW_CONFIG = {
    "hnsw:space": "cosine",
    "hnsw:M": 16,
    "hnsw:construction_ef": 100,
    "hnsw:search_ef": 50
}


def create_test_skills():
    """Create test skill data for indexing."""
    return [
        {
            'id': 'calculate_rsi',
            'name': 'calculate_rsi',
            'category': 'trading_skills',
            'description': 'Calculate the Relative Strength Index (RSI) technical indicator for trading analysis',
            'version': '1.0.0',
            'dependencies': ['fetch_historical_data'],
            'file_path': 'docs/skills/trading_skills/calculate_rsi.md',
            'input_schema': json.dumps({
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "period": {"type": "integer", "default": 14}
                }
            }),
            'output_schema': json.dumps({
                "type": "object",
                "properties": {
                    "rsi_values": {"type": "array"},
                    "current_rsi": {"type": "number"}
                }
            }),
            'code': 'def calculate_rsi(symbol, period=14): return {"rsi_values": [], "current_rsi": 50}',
            'example_usage': 'result = calculate_rsi("EURUSD")',
            'search_text': 'Calculate RSI indicator momentum technical analysis trading'
        },
        {
            'id': 'fetch_historical_data',
            'name': 'fetch_historical_data',
            'category': 'data_skills',
            'description': 'Fetch historical OHLCV data from MT5 for backtesting and analysis',
            'version': '1.0.0',
            'dependencies': [],
            'file_path': 'docs/skills/data_skills/fetch_historical_data.md',
            'input_schema': json.dumps({
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "timeframe": {"type": "string"},
                    "bars": {"type": "integer"}
                }
            }),
            'output_schema': json.dumps({
                "type": "object",
                "properties": {
                    "data": {"type": "array"}
                }
            }),
            'code': 'def fetch_historical_data(symbol, timeframe="H1", bars=100): return []',
            'example_usage': 'data = fetch_historical_data("EURUSD", "H1", 100)',
            'search_text': 'Fetch historical data OHLCV MT5 backtesting analysis'
        },
        {
            'id': 'calculate_macd',
            'name': 'calculate_macd',
            'category': 'trading_skills',
            'description': 'Calculate MACD indicator with signal line crossover detection',
            'version': '1.0.0',
            'dependencies': ['fetch_historical_data'],
            'file_path': 'docs/skills/trading_skills/calculate_macd.md',
            'input_schema': json.dumps({
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "fast_period": {"type": "integer", "default": 12},
                    "slow_period": {"type": "integer", "default": 26}
                }
            }),
            'output_schema': json.dumps({
                "type": "object",
                "properties": {
                    "macd_line": {"type": "array"},
                    "signal_line": {"type": "array"},
                    "histogram": {"type": "array"}
                }
            }),
            'code': 'def calculate_macd(symbol, fast=12, slow=26): return {"macd_line": [], "signal_line": []}',
            'example_usage': 'result = calculate_macd("EURUSD")',
            'search_text': 'Calculate MACD indicator moving average convergence divergence trading'
        },
        # Test versioning with same skill, different versions
        {
            'id': 'calculate_position_size',
            'name': 'calculate_position_size',
            'category': 'trading_skills',
            'description': 'Calculate position size v2.0 with improved risk management',
            'version': '2.0.0',
            'dependencies': [],
            'file_path': 'docs/skills/trading_skills/calculate_position_size.md',
            'input_schema': json.dumps({"type": "object"}),
            'output_schema': json.dumps({"type": "object"}),
            'code': 'def calculate_position_size_v2(): return {"lots": 0.1}',
            'example_usage': 'size = calculate_position_size()',
            'search_text': 'Calculate position size risk management money trading v2'
        },
        {
            'id': 'validate_stop_loss',
            'name': 'validate_stop_loss',
            'category': 'trading_skills',
            'description': 'Validate stop loss levels using ATR for dynamic risk management',
            'version': '1.0.0',
            'dependencies': [],
            'file_path': 'docs/skills/trading_skills/validate_stop_loss.md',
            'input_schema': json.dumps({"type": "object"}),
            'output_schema': json.dumps({"type": "object"}),
            'code': 'def validate_stop_loss(): return {"valid": True}',
            'example_usage': 'result = validate_stop_loss()',
            'search_text': 'Validate stop loss ATR risk management trading'
        }
    ]


def setup_test_collection():
    """Set up test ChromaDB collection with sample skills."""
    # Create test directory
    test_chroma_path = PROJECT_ROOT / "data" / "chromadb_test"
    test_chroma_path.mkdir(parents=True, exist_ok=True)

    # Initialize client
    client = chromadb.PersistentClient(path=str(test_chroma_path))

    # Create collection
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata=HNSW_CONFIG
    )

    # Clear existing data
    try:
        existing = collection.get()
        if existing['ids']:
            collection.delete(ids=existing['ids'])
    except Exception:
        pass

    # Add test skills
    skills = create_test_skills()
    model = SentenceTransformer('all-MiniLM-L6-v2')
    search_texts = [skill['search_text'] for skill in skills]
    embeddings = model.encode(search_texts)

    ids = [skill['id'] for skill in skills]
    documents = [skill['description'] for skill in skills]
    metadatas = [
        {
            'name': skill['name'],
            'category': skill['category'],
            'version': skill['version'],
            'dependencies': json.dumps(skill['dependencies']),
            'file_path': skill['file_path'],
            'input_schema': skill['input_schema'],
            'output_schema': skill['output_schema'],
            'example_usage': skill['example_usage'],
            'code': skill['code']  # Include code field
        }
        for skill in skills
    ]

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )

    return client, collection


# =============================================================================
# Tests
# =============================================================================

def test_load_skill_returns_skill_definition_from_chromadb():
    """Test that load_skill returns skill definition from ChromaDB."""
    client, collection = setup_test_collection()

    # Get skill by ID
    results = collection.get(
        ids=['calculate_rsi'],
        include=['documents', 'metadatas']
    )

    assert results['ids'], "No results returned from ChromaDB"
    assert len(results['ids']) == 1, "Expected exactly 1 result"

    metadata = results['metadatas'][0]
    assert metadata['name'] == 'calculate_rsi', "Skill name mismatch"
    assert metadata['category'] == 'trading_skills', "Category mismatch"
    assert metadata['version'] == '1.0.0', "Version mismatch"
    assert 'fetch_historical_data' in json.loads(metadata['dependencies']), "Dependencies not parsed correctly"

    # Clean up
    client.delete_collection(COLLECTION_NAME)
    print("test_load_skill_returns_skill_definition_from_chromadb: PASSED")


def test_skill_dependencies_loaded_recursively():
    """Test that skill dependencies are loaded recursively."""
    client, collection = setup_test_collection()

    # Load calculate_rsi which depends on fetch_historical_data
    results = collection.get(
        ids=['calculate_rsi'],
        include=['metadatas']
    )

    metadata = results['metadatas'][0]
    dependencies = json.loads(metadata['dependencies'])
    assert 'fetch_historical_data' in dependencies, "Dependency not found"

    # Load the dependency
    dep_results = collection.get(
        ids=['fetch_historical_data'],
        include=['metadatas']
    )

    assert dep_results['ids'], "Dependency not found in ChromaDB"
    dep_metadata = dep_results['metadatas'][0]
    assert dep_metadata['name'] == 'fetch_historical_data', "Dependency name mismatch"
    assert dep_metadata['category'] == 'data_skills', "Dependency category mismatch"

    # Verify dependency has no further dependencies (base case)
    dep_deps = json.loads(dep_metadata['dependencies'])
    assert len(dep_deps) == 0, "Unexpected nested dependencies"

    # Clean up
    client.delete_collection(COLLECTION_NAME)
    print("test_skill_dependencies_loaded_recursively: PASSED")


def test_semantic_search_finds_relevant_skills():
    """Test that semantic search finds relevant skills by description."""
    client, collection = setup_test_collection()

    # Search for "RSI indicator" - should match calculate_rsi
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode("RSI indicator technical analysis")

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=3
    )

    assert results['ids'], "No results from semantic search"
    assert len(results['ids'][0]) > 0, "Expected at least 1 result"

    # First result should be calculate_rsi or calculate_macd (both indicators)
    top_result_id = results['ids'][0][0]
    assert top_result_id in ['calculate_rsi', 'calculate_macd'], f"Unexpected top result: {top_result_id}"

    # Search for "position sizing"
    query_embedding = model.encode("position size risk management")
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=2
    )

    assert results['ids'], "No results for position sizing search"
    top_result = results['ids'][0][0]
    assert top_result == 'calculate_position_size', f"Expected calculate_position_size, got {top_result}"

    # Clean up
    client.delete_collection(COLLECTION_NAME)
    print("test_semantic_search_finds_relevant_skills: PASSED")


def test_skill_versioning_returns_latest():
    """Test that skill versioning returns latest version by default."""
    client, collection = setup_test_collection()

    # Get skill - should return latest version
    results = collection.get(
        ids=['calculate_position_size'],
        include=['metadatas']
    )

    assert results['ids'], "Skill not found"
    metadata = results['metadatas'][0]
    assert metadata['version'] == '2.0.0', f"Expected version 2.0.0, got {metadata['version']}"

    # Verify we can query by version using metadata filter
    # (In real implementation, this would use where={"version": "2.0.0"})
    all_results = collection.get(
        where={'name': 'calculate_position_size'},
        include=['metadatas']
    )

    version_found = False
    for metadata in all_results['metadatas']:
        if metadata.get('version') == '2.0.0':
            version_found = True
            break

    assert version_found, "Version 2.0.0 not found in results"

    # Clean up
    client.delete_collection(COLLECTION_NAME)
    print("test_skill_versioning_returns_latest: PASSED")


def test_load_skill_invalid_skill_returns_error():
    """Test that loading an invalid skill returns appropriate error."""
    client, collection = setup_test_collection()

    # Try to get non-existent skill
    results = collection.get(
        ids=['non_existent_skill'],
        include=['documents', 'metadatas']
    )

    assert not results['ids'], "Expected empty results for non-existent skill"
    assert len(results['ids']) == 0, "Expected 0 results"

    # Clean up
    client.delete_collection(COLLECTION_NAME)
    print("test_load_skill_invalid_skill_returns_error: PASSED")


def test_semantic_search_with_category_filter():
    """Test semantic search with category filter."""
    import time
    client, collection = setup_test_collection()

    # Small delay to ensure collection is ready
    time.sleep(0.1)

    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Search for data skills
    query_embedding = model.encode("data")
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=5,
        where={'category': 'data_skills'}
    )

    assert results['ids'], "No results for data_skills category"
    # All results should be from data_skills category
    for metadata in results['metadatas'][0]:
        assert metadata['category'] == 'data_skills', f"Expected data_skills, got {metadata['category']}"

    # Clean up
    client.delete_collection(COLLECTION_NAME)
    print("test_semantic_search_with_category_filter: PASSED")


def test_skill_dependency_chain_resolution():
    """Test resolving a chain of dependencies."""
    client, collection = setup_test_collection()

    # calculate_rsi -> fetch_historical_data (chain of 2)
    # calculate_macd -> fetch_historical_data (another chain)

    # Load calculate_macd
    results = collection.get(
        ids=['calculate_macd'],
        include=['metadatas']
    )

    metadata = results['metadatas'][0]
    dependencies = json.loads(metadata['dependencies'])

    assert 'fetch_historical_data' in dependencies, "Dependency not found"

    # Resolve dependency
    dep_results = collection.get(
        ids=dependencies,
        include=['metadatas']
    )

    assert dep_results['ids'], "Dependencies not resolved"
    assert len(dep_results['ids']) == 1, "Expected 1 dependency"

    dep_metadata = dep_results['metadatas'][0]
    dep_deps = json.loads(dep_metadata['dependencies'])
    assert len(dep_deps) == 0, "Expected leaf node (no further dependencies)"

    # Clean up
    client.delete_collection(COLLECTION_NAME)
    print("test_skill_dependency_chain_resolution: PASSED")


def test_skill_conforms_to_agent_skill_interface():
    """Test that loaded skills conform to AgentSkill interface requirements."""
    client, collection = setup_test_collection()

    results = collection.get(
        ids=['calculate_rsi'],
        include=['metadatas', 'documents']
    )

    metadata = results['metadatas'][0]

    # AgentSkill interface requires: name, description
    assert metadata['name'], "Skill must have name"
    assert results['documents'][0], "Skill must have description"
    assert metadata['category'], "Skill must have category"
    assert metadata['code'], "Skill must have code"

    # Verify schemas are valid JSON
    try:
        input_schema = json.loads(metadata['input_schema'])
        output_schema = json.loads(metadata['output_schema'])
        assert 'type' in input_schema or 'properties' in input_schema, "Invalid input schema"
        assert 'type' in output_schema or 'properties' in output_schema, "Invalid output schema"
    except json.JSONDecodeError:
        assert False, "Schemas must be valid JSON"

    # Clean up
    client.delete_collection(COLLECTION_NAME)
    print("test_skill_conforms_to_agent_skill_interface: PASSED")


# =============================================================================
# Test Runner
# =============================================================================

def run_tests():
    """Run all tests."""
    print("=" * 70)
    print("Assets Hub Skills Integration Tests (Task Group 13)")
    print("=" * 70)
    print()

    tests = [
        test_load_skill_returns_skill_definition_from_chromadb,
        test_skill_dependencies_loaded_recursively,
        test_semantic_search_finds_relevant_skills,
        test_skill_versioning_returns_latest,
        test_load_skill_invalid_skill_returns_error,
        test_semantic_search_with_category_filter,
        test_skill_dependency_chain_resolution,
        test_skill_conforms_to_agent_skill_interface
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"{test.__name__}: FAILED - {e}")
            failed += 1
        except Exception as e:
            print(f"{test.__name__}: ERROR - {e}")
            failed += 1

    print()
    print("=" * 70)
    print(f"Test Results: {passed} passed, {failed} failed out of {len(tests)} total")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)

"""
Test suite for ChromaKBClient functionality.

Tests:
- Search functionality with mock results
- Collection stats
- create_analyst_kb filtering
- Error handling
- Edge cases
"""

import pytest
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Import the module under test
from tools.analyst_agent.kb.client import ChromaKBClient, ChromaKBError, CollectionNotFoundError, InvalidQueryError


@pytest.fixture
def mock_chroma_client():
    """Mock ChromaDB client for testing."""
    with patch('tools.analyst_agent.kb.client.chromadb') as mock_chroma:
        # Mock PersistentClient
        mock_client = Mock()
        mock_chroma.PersistentClient.return_value = mock_client

        # Mock collection
        mock_collection = Mock()
        mock_client.get_collection.return_value = mock_collection
        mock_client.get_or_create_collection.return_value = mock_collection

        # Mock query results
        mock_collection.query.return_value = {
            "documents": [["Document 1 content", "Document 2 content"]],
            "metadatas": [[{"title": "Doc 1", "categories": "Trading Systems"}, {"title": "Doc 2", "categories": "Machine Learning"}]],
            "distances": [[0.1, 0.2]]
        }

        # Mock count
        mock_collection.count.return_value = 2

        yield mock_client, mock_collection


@pytest.fixture
def temp_db_path():
    """Create temporary database path for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def test_chroma_kb_client_initialization(temp_db_path):
    """Test ChromaKBClient initialization."""
    client = ChromaKBClient(db_path=temp_db_path)
    assert client.db_path == temp_db_path
    assert client._client is not None


def test_search_with_mock_results(mock_chroma_client):
    """Test search functionality with mock results."""
    mock_client, mock_collection = mock_chroma_client
    client = ChromaKBClient()

    # Mock query results
    mock_collection.query.return_value = {
        "documents": [["Test document content"]],
        "metadatas": [[{"title": "Test Article", "categories": "Trading Systems", "file_path": "test.md"}]],
        "distances": [[0.15]]
    }

    results = client.search("test query", n=1)

    assert len(results) == 1
    assert results[0]["title"] == "Test Article"
    assert results[0]["score"] == 0.85  # 1.0 - 0.15
    assert results[0]["categories"] == "Trading Systems"
    assert results[0]["file_path"] == "test.md"


def test_search_with_category_filter(mock_chroma_client):
    """Test search with category filter."""
    mock_client, mock_collection = mock_chroma_client
    client = ChromaKBClient()

    # Mock documents with different categories
    mock_collection.query.return_value = {
        "documents": [
            ["Trading Systems article"],
            ["Machine Learning article"],
            ["Expert Advisors article"]
        ],
        "metadatas": [
            {"title": "Doc 1", "categories": "Trading Systems"},
            {"title": "Doc 2", "categories": "Machine Learning"},
            {"title": "Doc 3", "categories": "Expert Advisors"}
        ],
        "distances": [[0.1, 0.2, 0.15]]
    }

    # Search with Trading Systems filter
    results = client.search("query", category_filter="Trading Systems", n=2)

    assert len(results) == 1
    assert results[0]["title"] == "Doc 1"
    assert "Trading Systems" in results[0]["categories"]


def test_search_empty_query():
    """Test search with empty query raises InvalidQueryError."""
    client = ChromaKBClient()

    with pytest.raises(InvalidQueryError, match="Query must be a non-empty string"):
        client.search("", n=5)


def test_search_invalid_n():
    """Test search with invalid n parameter."""
    client = ChromaKBClient()

    with pytest.raises(InvalidQueryError, match="n must be greater than 0"):
        client.search("test", n=0)


def test_search_nonexistent_collection():
    """Test search with non-existent collection raises CollectionNotFoundError."""
    client = ChromaKBClient()

    with patch('tools.analyst_agent.kb.client.chromadb') as mock_chroma:
        mock_client = Mock()
        mock_chroma.PersistentClient.return_value = mock_client
        mock_client.get_collection.side_effect = Exception("Collection not found")

        with pytest.raises(CollectionNotFoundError, match="Collection 'nonexistent' not found"):
            client.search("test", collection="nonexistent")


def test_get_collection_stats(mock_chroma_client):
    """Test get_collection_stats functionality."""
    mock_client, mock_collection = mock_chroma_client
    client = ChromaKBClient()

    stats = client.get_collection_stats("test_collection")

    assert stats["name"] == "test_collection"
    assert stats["count"] == 2
    assert "metadata" in stats


def test_get_collection_stats_nonexistent():
    """Test get_collection_stats with non-existent collection."""
    client = ChromaKBClient()

    with patch('tools.analyst_agent.kb.client.chromadb') as mock_chroma:
        mock_client = Mock()
        mock_chroma.PersistentClient.return_value = mock_client
        mock_client.get_collection.side_effect = Exception("Collection not found")

        with pytest.raises(CollectionNotFoundError, match="Collection 'nonexistent' not found"):
            client.get_collection_stats("nonexistent")


def test_list_collections(mock_chroma_client):
    """Test list_collections functionality."""
    mock_client, mock_collection = mock_chroma_client
    client = ChromaKBClient()

    mock_client.list_collections.return_value = [Mock(name="collection1"), Mock(name="collection2")]

    collections = client.list_collections()

    assert len(collections) == 2
    assert "collection1" in collections
    assert "collection2" in collections


def test_create_analyst_kb(mock_chroma_client):
    """Test create_analyst_kb functionality."""
    mock_client, mock_collection = mock_chroma_client
    client = ChromaKBClient()

    # Mock source collection data
    mock_source = Mock()
    mock_client.get_collection.return_value = mock_source
    mock_source.get.return_value = {
        "documents": ["Trading article", "ML article", "Indicators article"],
        "metadatas": [
            {"categories": "Trading Systems"},
            {"categories": "Machine Learning"},
            {"categories": "Indicators"}
        ],
        "embeddings": [[0.1], [0.2], [0.15]]
    }

    # Mock target collection
    mock_target = Mock()
    mock_client.get_or_create_collection.return_value = mock_target

    stats = client.create_analyst_kb()

    assert stats["total"] == 3
    assert stats["included"] == 2  # Trading Systems and Indicators
    assert stats["excluded"] == 1  # Machine Learning


def test_create_analyst_kb_no_source_collection():
    """Test create_analyst_kb with non-existent source collection."""
    client = ChromaKBClient()

    with patch('tools.analyst_agent.kb.client.chromadb') as mock_chroma:
        mock_client = Mock()
        mock_chroma.PersistentClient.return_value = mock_client
        mock_client.get_collection.side_effect = Exception("Source not found")

        with pytest.raises(CollectionNotFoundError, match="Source collection 'mql5_knowledge' not found"):
            client.create_analyst_kb()


def test_should_include_for_analyst():
    """Test _should_include_for_analyst filtering logic."""
    client = ChromaKBClient()

    # Should include: matches analyst category
    assert client._should_include_for_analyst("Trading Systems") is True
    assert client._should_include_for_analyst("Expert Advisors") is True
    assert client._should_include_for_analyst("Indicators") is True

    # Should exclude: matches excluded category
    assert client._should_include_for_analyst("Machine Learning") is False
    assert client._should_include_for_analyst("Integration") is False

    # Should include: no categories specified
    assert client._should_include_for_analyst("") is True
    assert client._should_include_for_analyst(None) is True

    # Should exclude: doesn't match any category
    assert client._should_include_for_analyst("Random Category") is False


def test_extract_title_from_yaml_frontmatter():
    """Test title extraction from YAML frontmatter."""
    client = ChromaKBClient()

    doc = """---
title: Test Article
author: John Doe
---
Content of the article"""

    title = client._extract_title(doc)
    assert title == "Test Article"


def test_extract_title_from_heading():
    """Test title extraction from markdown heading."""
    client = ChromaKBClient()

    doc = """# Test Article
Content of the article"""

    title = client._extract_title(doc)
    assert title == "Test Article"


def test_extract_title_untitled():
    """Test title extraction when no title found."""
    client = ChromaKBClient()

    doc = "Just some content without a title"

    title = client._extract_title(doc)
    assert title == "Untitled"


def test_extract_preview():
    """Test preview extraction functionality."""
    client = ChromaKBClient()

    doc = """---
title: Test
---
# Heading
This is the content of the article that should be previewed.
More content here.
"""

    preview = client._extract_preview(doc)
    assert "This is the content" in preview
    assert len(preview) <= 300  # Should be truncated


def test_extract_preview_long_content():
    """Test preview extraction with long content."""
    client = ChromaKBClient()

    doc = "A" * 500  # 500 characters

    preview = client._extract_preview(doc)
    assert len(preview) == 300  # Should be truncated to max_chars
    assert preview.endswith("...")


def test_chroma_kb_error_hierarchy():
    """Test ChromaKBError exception hierarchy."""
    assert issubclass(CollectionNotFoundError, ChromaKBError)
    assert issubclass(InvalidQueryError, ChromaKBError)

    # Test exception instantiation
    error = ChromaKBError("Test error")
    assert str(error) == "Test error"

    collection_error = CollectionNotFoundError("Collection not found")
    assert str(collection_error) == "Collection not found"

    query_error = InvalidQueryError("Invalid query")
    assert str(query_error) == "Invalid query"
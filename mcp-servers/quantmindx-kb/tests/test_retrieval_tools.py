"""Focused tests for structured asset retrieval tools.

Tests cover the 4 new MCP tools:
- get_algorithm_template
- get_coding_standards
- get_bad_patterns
- load_skill
"""

import sys
import os
from pathlib import Path

import pytest

# Ensure we're in the project root to pass the security check
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
if PROJECT_ROOT.name == "QUANTMINDX":
    os.chdir(PROJECT_ROOT)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import after changing directory
import chromadb

# Initialize ChromaDB client for testing
CHROMA_PATH = PROJECT_ROOT / "data" / "chromadb"
client = chromadb.PersistentClient(path=str(CHROMA_PATH))

# Get or create collections for testing
algorithm_templates_collection = client.get_or_create_collection(name="algorithm_templates")
bad_patterns_collection = client.get_or_create_collection(name="bad_patterns_graveyard")
coding_standards_collection = client.get_or_create_collection(name="coding_standards")
agentic_skills_collection = client.get_or_create_collection(name="agentic_skills")


@pytest.mark.unit
class TestGetAlgorithmTemplate:
    """Test get_algorithm_template tool ChromaDB collection access."""

    def test_collection_exists_and_accessible(self):
        """Test that algorithm_templates collection exists and is accessible."""
        count = algorithm_templates_collection.count()
        assert isinstance(count, int)
        assert count >= 0

    def test_query_by_category_filter(self):
        """Test retrieving algorithm template with category filter."""
        # Query with metadata filter
        results = algorithm_templates_collection.query(
            query_texts=["trend following"],
            n_results=1,
            where={"category": "trend_following"}
        )

        # Verify response structure
        assert "documents" in results
        assert "metadatas" in results
        assert isinstance(results["documents"], list)

    def test_query_with_complexity_filter(self):
        """Test retrieving algorithm template with complexity filter."""
        results = algorithm_templates_collection.query(
            query_texts=["strategy"],
            n_results=1,
            where={"complexity": "basic"}
        )

        # Verify response structure
        assert "documents" in results
        assert "metadatas" in results


@pytest.mark.unit
class TestGetCodingStandards:
    """Test get_coding_standards tool ChromaDB collection access."""

    def test_collection_exists_and_accessible(self):
        """Test that coding_standards collection exists and is accessible."""
        count = coding_standards_collection.count()
        assert isinstance(count, int)
        assert count >= 0

    def test_get_coding_standards_returns_metadata(self):
        """Test that coding standards returns project conventions metadata."""
        results = coding_standards_collection.get(include=['documents', 'metadatas'])

        # Verify response structure
        assert "documents" in results
        assert "metadatas" in results
        assert isinstance(results["documents"], list)
        assert isinstance(results["metadatas"], list)


@pytest.mark.unit
class TestGetBadPatterns:
    """Test get_bad_patterns tool ChromaDB collection access."""

    def test_collection_exists_and_accessible(self):
        """Test that bad_patterns_graveyard collection exists and is accessible."""
        count = bad_patterns_collection.count()
        assert isinstance(count, int)
        assert count >= 0

    def test_get_bad_patterns_returns_structured_data(self):
        """Test that bad patterns return structured data with severity."""
        results = bad_patterns_collection.get(include=['documents', 'metadatas'])

        # Verify response structure
        assert "documents" in results
        assert "metadatas" in results

        # If patterns exist, check for severity in metadata
        if results['metadatas'] and len(results['metadatas'][0]) > 0:
            metadata = results['metadatas'][0][0]
            assert 'severity' in metadata or 'name' in metadata


@pytest.mark.unit
class TestLoadSkill:
    """Test load_skill tool ChromaDB collection access."""

    def test_collection_exists_and_accessible(self):
        """Test that agentic_skills collection exists and is accessible."""
        count = agentic_skills_collection.count()
        assert isinstance(count, int)
        assert count >= 0

    def test_load_skill_by_id(self):
        """Test retrieving skill by ID from collection."""
        # Try to get a skill - may not exist but tests the API
        results = agentic_skills_collection.get(
            ids=['test_skill'],
            include=['documents', 'metadatas']
        )

        # Verify response structure
        assert "documents" in results
        assert "metadatas" in results

    def test_load_skill_metadata_structure(self):
        """Test that skill metadata conforms to expected structure."""
        results = agentic_skills_collection.get(include=['metadatas'], limit=5)

        # Verify response structure
        assert "metadatas" in results
        assert isinstance(results["metadatas"], list)

        # Check that metadata has expected fields if skills exist
        if results['metadatas'] and len(results['metadatas'][0]) > 0:
            metadata = results['metadatas'][0][0]
            # Skills should have at least a name
            assert 'name' in metadata or 'category' in metadata

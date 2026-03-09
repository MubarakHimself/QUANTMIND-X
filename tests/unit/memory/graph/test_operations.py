"""Tests for graph memory operations."""
import tempfile
from pathlib import Path

import pytest

from src.memory.graph.operations import MemoryOperations
from src.memory.graph.store import GraphMemoryStore
from src.memory.graph.types import MemoryCategory, MemoryNodeType, RelationType


@pytest.fixture
def ops():
    """Create a MemoryOperations instance with a temporary store."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = GraphMemoryStore(Path(tmpdir) / "test.db")
        yield MemoryOperations(store)
        store.close()


class TestRetain:
    """Tests for RETAIN operation."""

    def test_retain_extracts_facts(self, ops):
        """Test RETAIN extracts factual information."""
        content = "User prefers to trade EURUSD on M15 timeframe. Maximum risk per trade is 2%."
        node = ops.retain(
            content=content,
            source="user_preference",
            department="trading",
        )
        assert node.node_type == MemoryNodeType.OPINION
        assert node.category == MemoryCategory.SUBJECTIVE

    def test_retain_categorizes_experiential(self, ops):
        """Test RETAIN categorizes experiential content."""
        content = "Executed buy order for EURUSD at 1.0850"
        node = ops.retain(
            content=content,
            source="trade_execution",
        )
        assert node.category == MemoryCategory.EXPERIENTIAL

    def test_retain_categorizes_factual(self, ops):
        """Test RETAIN categorizes factual content."""
        content = "Office hours are 9-5 GMT."
        node = ops.retain(content=content, source="general_info")
        assert node.category == MemoryCategory.FACTUAL

    def test_retain_extracts_title(self, ops):
        """Test RETAIN extracts title from content."""
        content = "This is the first sentence. This is the second sentence."
        node = ops.retain(content=content, source="test")
        assert node.title == "This is the first sentence"

    def test_retain_stores_metadata(self, ops):
        """Test RETAIN stores all provided metadata."""
        node = ops.retain(
            content="Test content",
            source="test",
            department="trading",
            agent_id="agent-001",
            session_id="session-123",
            importance=0.8,
            tags=["tag1", "tag2"],
        )
        assert node.department == "trading"
        assert node.agent_id == "agent-001"
        assert node.session_id == "session-123"
        assert node.importance == 0.8
        assert "tag1" in node.tags
        assert "tag2" in node.tags

    def test_retain_with_explicit_category(self, ops):
        """Test RETAIN uses explicit category when provided."""
        content = "Some content"
        node = ops.retain(
            content=content,
            source="test",
            category=MemoryCategory.FACTUAL,
        )
        assert node.category == MemoryCategory.FACTUAL

    def test_retain_links_related_nodes(self, ops):
        """Test RETAIN creates edges to related nodes."""
        # Create first node
        node1 = ops.retain(
            content="First memory",
            source="test",
        )
        # Create second node related to first
        node2 = ops.retain(
            content="Second memory",
            source="test",
            related_to=[str(node1.id)],
        )
        # Check that edge was created
        edges = ops.store.get_edges(str(node2.id))
        assert len(edges) == 1
        assert str(edges[0].target_id) == str(node1.id)


class TestRecall:
    """Tests for RECALL operation."""

    def test_recall_by_department(self, ops):
        """Test RECALL retrieves by department."""
        ops.retain(
            content="Risk per trade is 2%",
            department="trading",
            importance=0.9,
        )
        ops.retain(
            content="Preferred timeframe is M15",
            department="trading",
            importance=0.8,
        )
        ops.retain(
            content="Office hours are 9-5",
            department="general",
            importance=0.5,
        )

        results = ops.recall(department="trading")
        assert len(results) == 2

    def test_recall_by_min_importance(self, ops):
        """Test RECALL filters by minimum importance."""
        ops.retain(content="High importance", importance=0.9)
        ops.retain(content="Medium importance", importance=0.5)
        ops.retain(content="Low importance", importance=0.2)

        results = ops.recall(min_importance=0.6)
        assert len(results) == 1
        assert results[0].importance == 0.9

    def test_recall_by_query(self, ops):
        """Test RECALL searches by query text."""
        ops.retain(content="Risk management is important")
        ops.retain(content="Timeframe preferences")
        ops.retain(content="Office location")

        results = ops.recall(query="risk")
        assert len(results) == 1
        assert "risk" in results[0].content.lower()

    def test_recall_limits_results(self, ops):
        """Test RECALL respects limit parameter."""
        for i in range(20):
            ops.retain(content=f"Memory {i}", importance=0.5)

        results = ops.recall(limit=5)
        assert len(results) == 5

    def test_recall_updates_access_count(self, ops):
        """Test RECALL updates access statistics."""
        node = ops.retain(content="Test memory")
        assert node.access_count == 0

        results = ops.recall()
        assert results[0].access_count == 1

    def test_recall_by_filters(self, ops):
        """Test RECALL retrieves by filters."""
        ops.retain(
            content="Risk per trade is 2%",
            department="trading",
            importance=0.9,
        )
        ops.retain(
            content="Preferred timeframe is M15",
            department="trading",
            importance=0.8,
        )
        ops.retain(
            content="Office hours are 9-5",
            department="general",
            importance=0.5,
        )

        results = ops.recall(department="trading", min_importance=0.7)
        assert len(results) == 2


class TestReflect:
    """Tests for REFLECT operation."""

    def test_reflect_synthesizes(self, ops):
        """Test REFLECT synthesizes answer."""
        ops.retain(
            content="User prefers EURUSD",
            department="trading",
        )
        ops.retain(
            content="User trades M15",
            department="trading",
        )

        result = ops.reflect(
            query="What does user prefer?",
            department="trading",
        )
        assert result["synthesized"] is True
        assert "EURUSD" in result["answer"]
        assert len(result["sources"]) == 2

    def test_reflect_returns_sources(self, ops):
        """Test REFLECT returns source node IDs."""
        node1 = ops.retain(content="First fact", department="trading")
        node2 = ops.retain(content="Second fact", department="trading")

        result = ops.reflect(query="facts", department="trading")
        assert str(node1.id) in result["sources"]
        assert str(node2.id) in result["sources"]

    def test_reflect_with_no_results(self, ops):
        """Test REFLECT handles no results gracefully."""
        result = ops.reflect(query="nonexistent", department="trading")
        assert result["synthesized"] is False
        assert "No relevant memories found" in result["answer"]

    def test_reflect_respects_department_filter(self, ops):
        """Test REFLECT filters by department."""
        ops.retain(content="Trading fact", department="trading")
        ops.retain(content="General fact", department="general")

        result = ops.reflect(query="fact", department="trading")
        assert "Trading fact" in result["answer"]
        assert "General fact" not in result["answer"]


class TestLinkNodes:
    """Tests for link_nodes method."""

    def test_link_creates_edge(self, ops):
        """Test link_nodes creates edges between nodes."""
        node1 = ops.retain(content="Node 1")
        node2 = ops.retain(content="Node 2")

        edges = ops.link_nodes(
            source_id=str(node1.id),
            target_ids=[str(node2.id)],
            relation_type=RelationType.RELATED_TO,
        )

        assert len(edges) == 1
        assert str(edges[0].source_id) == str(node1.id)
        assert str(edges[0].target_id) == str(node2.id)


class TestCategorizeContent:
    """Tests for content categorization."""

    def test_categorize_experiential(self, ops):
        """Test categorization of experiential content."""
        content = "I executed a buy order for EURUSD"
        category = ops._categorize_content(content)
        assert category == MemoryCategory.EXPERIENTIAL

    def test_categorize_subjective(self, ops):
        """Test categorization of subjective content."""
        content = "I think EURUSD is a good pair"
        category = ops._categorize_content(content)
        assert category == MemoryCategory.SUBJECTIVE

    def test_categorize_factual_default(self, ops):
        """Test factual is default category."""
        content = "The sky is blue"
        category = ops._categorize_content(content)
        assert category == MemoryCategory.FACTUAL


class TestExtractTitle:
    """Tests for title extraction."""

    def test_extract_title_from_sentence(self, ops):
        """Test extracting title from first sentence."""
        content = "This is the first sentence. This is the second."
        title = ops._extract_title(content)
        assert title == "This is the first sentence"

    def test_extract_title_truncates_long(self, ops):
        """Test truncating very long titles."""
        content = "A" * 200
        title = ops._extract_title(content)
        assert len(title) == 100
        assert title.endswith("...")

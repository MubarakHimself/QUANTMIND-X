"""Tests for ContextCompactionTrigger."""
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.memory.graph.compaction import ContextCompactionTrigger
from src.memory.graph.store import GraphMemoryStore
from src.memory.graph.types import (
    MemoryCategory,
    MemoryEdge,
    MemoryNode,
    MemoryNodeType,
    MemoryTier,
    RelationType,
)


@pytest.fixture
def trigger():
    """Create a compaction trigger with a temporary database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = GraphMemoryStore(Path(tmpdir) / "test.db")
        yield ContextCompactionTrigger(store)


class TestShouldCompact:
    """Tests for should_compact method."""

    def test_should_compact_below_threshold(self, trigger):
        """Test that compaction is not triggered below threshold."""
        assert trigger.should_compact(40) is False
        assert trigger.should_compact(0) is False
        assert trigger.should_compact(49.9) is False

    def test_should_compact_at_threshold(self, trigger):
        """Test that compaction is triggered at threshold."""
        assert trigger.should_compact(50) is True

    def test_should_compact_above_threshold(self, trigger):
        """Test that compaction is triggered above threshold."""
        assert trigger.should_compact(55) is True
        assert trigger.should_compact(75) is True
        assert trigger.should_compact(100) is True


class TestDefaultConstants:
    """Tests for default constants."""

    def test_default_compaction_threshold(self, trigger):
        """Test DEFAULT_COMPACTION_THRESHOLD constant."""
        assert trigger.DEFAULT_COMPACTION_THRESHOLD == 50.0

    def test_default_summary_max_tokens(self, trigger):
        """Test DEFAULT_SUMMARY_MAX_TOKENS constant."""
        assert trigger.DEFAULT_SUMMARY_MAX_TOKENS == 500

    def test_custom_threshold(self):
        """Test custom threshold initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GraphMemoryStore(Path(tmpdir) / "test.db")
            trigger = ContextCompactionTrigger(store, threshold=30.0)
            assert trigger.threshold == 30.0
            assert trigger.should_compact(35) is True
            assert trigger.should_compact(25) is False


class TestGetCompactionCandidates:
    """Tests for get_compaction_candidates method."""

    def test_no_candidates_empty_store(self, trigger):
        """Test no candidates when store is empty."""
        candidates = trigger.get_compaction_candidates()
        assert len(candidates) == 0

    def test_excludes_high_importance_nodes(self, trigger):
        """Test that high importance nodes are excluded."""
        # Create high importance node
        node = MemoryNode(
            title="High Importance",
            content="content",
            node_type=MemoryNodeType.WORLD,
            category=MemoryCategory.FACTUAL,
            importance=0.9,
            tier=MemoryTier.WARM,
        )
        trigger.store.create_node(node)

        candidates = trigger.get_compaction_candidates(max_importance=0.5)
        assert len(candidates) == 0

    def test_includes_low_importance_nodes(self, trigger):
        """Test that low importance nodes are included."""
        node = MemoryNode(
            title="Low Importance",
            content="content",
            node_type=MemoryNodeType.WORLD,
            category=MemoryCategory.FACTUAL,
            importance=0.3,
            tier=MemoryTier.WARM,
        )
        trigger.store.create_node(node)

        candidates = trigger.get_compaction_candidates(max_importance=0.5)
        assert len(candidates) == 1

    def test_excludes_recently_accessed_nodes(self, trigger):
        """Test that recently accessed nodes are excluded."""
        node = MemoryNode(
            title="Recent Node",
            content="content",
            node_type=MemoryNodeType.WORLD,
            category=MemoryCategory.FACTUAL,
            importance=0.3,
            tier=MemoryTier.WARM,
            last_accessed=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        trigger.store.create_node(node)

        candidates = trigger.get_compaction_candidates(
            max_importance=0.5,
            exclude_recent_hours=24
        )
        assert len(candidates) == 0

    def test_excludes_already_compacted_nodes(self, trigger):
        """Test that already compacted nodes are excluded."""
        node = MemoryNode(
            title="Already Compacted",
            content="content",
            node_type=MemoryNodeType.WORLD,
            category=MemoryCategory.FACTUAL,
            importance=0.3,
            compaction_level=2,
            tier=MemoryTier.COLD,
        )
        trigger.store.create_node(node)

        candidates = trigger.get_compaction_candidates(max_importance=0.5)
        assert len(candidates) == 0


class TestCreateMemoryAnchor:
    """Tests for create_memory_anchor method."""

    def test_create_memory_anchor(self, trigger):
        """Test creating a memory anchor from original node."""
        # Create original node
        node = MemoryNode(
            title="Long Memory",
            content="Very long content " * 100,
            node_type=MemoryNodeType.WORLD,
            category=MemoryCategory.FACTUAL,
            importance=0.8,
            tier=MemoryTier.WARM,
        )
        trigger.store.create_node(node)

        # Create anchor
        anchor = trigger.create_memory_anchor(str(node.id))

        assert anchor is not None
        assert anchor.original_id == node.id
        assert anchor.compaction_level == 1
        assert anchor.importance == 0.8
        assert "[ANCHOR]" in anchor.title
        assert anchor.summary is not None
        assert len(anchor.summary) > 0

    def test_create_memory_anchor_with_custom_summary(self, trigger):
        """Test creating anchor with custom summary."""
        node = MemoryNode(
            title="Test",
            content="content",
            node_type=MemoryNodeType.WORLD,
            category=MemoryCategory.FACTUAL,
            importance=0.5,
        )
        trigger.store.create_node(node)

        custom_summary = "Custom summary"
        anchor = trigger.create_memory_anchor(str(node.id), summary=custom_summary)

        assert anchor.summary == custom_summary

    def test_create_memory_anchor_preserves_edges(self, trigger):
        """Test that edges are transferred from original to anchor."""
        # Create two nodes
        node1 = MemoryNode(
            title="Node 1",
            content="content1",
            node_type=MemoryNodeType.WORLD,
            category=MemoryCategory.FACTUAL,
            importance=0.5,
        )
        node2 = MemoryNode(
            title="Node 2",
            content="content2",
            node_type=MemoryNodeType.WORLD,
            category=MemoryCategory.FACTUAL,
            importance=0.5,
        )
        trigger.store.create_node(node1)
        trigger.store.create_node(node2)

        # Create edge between them
        edge = MemoryEdge(
            relation_type=RelationType.RELATED_TO,
            source_id=node1.id,
            target_id=node2.id,
            strength=0.8,
        )
        trigger.store.create_edge(edge)

        # Create anchor for node1
        anchor = trigger.create_memory_anchor(str(node1.id))

        # Check that anchor has edges to node2
        anchor_edges = trigger.store.get_edges(str(anchor.id))
        assert len(anchor_edges) == 1

    def test_create_memory_anchor_nonexistent_node(self, trigger):
        """Test creating anchor from nonexistent node returns None."""
        anchor = trigger.create_memory_anchor("nonexistent-id")
        assert anchor is None


class TestCompactAndAnchor:
    """Tests for compact_and_anchor method."""

    def test_compact_and_anchor_creates_anchors(self, trigger):
        """Test that compact_and_anchor creates anchor nodes."""
        # Create multiple low importance nodes
        for i in range(5):
            node = MemoryNode(
                title=f"Mem {i}",
                content=f"Content {i}",
                node_type=MemoryNodeType.WORLD,
                category=MemoryCategory.FACTUAL,
                importance=0.5 - i * 0.1,
                tier=MemoryTier.WARM,
            )
            trigger.store.create_node(node)

        result = trigger.compact_and_anchor(max_importance=0.4, max_nodes=10)

        assert result["compacted_count"] > 0
        assert result["anchors_created"] > 0

    def test_compact_and_anchor_marks_originals_compacted(self, trigger):
        """Test that originals are marked as compacted."""
        node = MemoryNode(
            title="Test",
            content="content",
            node_type=MemoryNodeType.WORLD,
            category=MemoryCategory.FACTUAL,
            importance=0.3,
            tier=MemoryTier.WARM,
        )
        trigger.store.create_node(node)

        result = trigger.compact_and_anchor(max_importance=0.4)

        # Check original is marked as compacted
        updated_node = trigger.store.get_node(str(node.id))
        assert updated_node.compaction_level == 2

    def test_compact_and_anchor_without_anchors(self, trigger):
        """Test compaction without creating anchors."""
        node = MemoryNode(
            title="Test",
            content="content",
            node_type=MemoryNodeType.WORLD,
            category=MemoryCategory.FACTUAL,
            importance=0.3,
            tier=MemoryTier.WARM,
        )
        trigger.store.create_node(node)

        result = trigger.compact_and_anchor(max_importance=0.4, create_anchors=False)

        assert result["compacted_count"] > 0
        assert result["anchors_created"] == 0

    def test_compact_and_anchor_respects_max_nodes(self, trigger):
        """Test that max_nodes limit is respected."""
        # Create many nodes
        for i in range(20):
            node = MemoryNode(
                title=f"Mem {i}",
                content=f"Content {i}",
                node_type=MemoryNodeType.WORLD,
                category=MemoryCategory.FACTUAL,
                importance=0.2,
                tier=MemoryTier.WARM,
            )
            trigger.store.create_node(node)

        result = trigger.compact_and_anchor(max_importance=0.5, max_nodes=5)

        assert result["compacted_count"] <= 5


class TestGenerateSummary:
    """Tests for _generate_summary method."""

    def test_generate_summary_short_content(self, trigger):
        """Test summary of short content."""
        content = "This is short content."
        summary = trigger._generate_summary(content)
        assert "This is short content" in summary

    def test_generate_summary_long_content(self, trigger):
        """Test summary of long content takes first two sentences."""
        content = "First sentence here. Second sentence here. Third sentence here."
        summary = trigger._generate_summary(content)
        # Should contain first two sentences
        assert "First sentence" in summary
        assert "Second sentence" in summary
        assert "Third sentence" not in summary

    def test_generate_summary_max_length(self, trigger):
        """Test that summary is capped at 200 characters."""
        content = "A" * 300 + ". This is another sentence."
        summary = trigger._generate_summary(content)
        assert len(summary) <= 200

    def test_generate_summary_empty_content(self, trigger):
        """Test summary of empty content."""
        summary = trigger._generate_summary("")
        assert summary == ""

    def test_generate_summary_no_sentences(self, trigger):
        """Test summary of content without clear sentence breaks."""
        content = "No clear sentences here"
        summary = trigger._generate_summary(content)
        assert len(summary) > 0

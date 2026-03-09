"""Tests for GraphMemoryFacade."""
import tempfile
from pathlib import Path

import pytest

from src.memory.graph.facade import GraphMemoryFacade, get_graph_memory


@pytest.fixture
def facade():
    """Create a GraphMemoryFacade instance with a temporary database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield GraphMemoryFacade(db_path=Path(tmpdir) / "memory.db")


class TestFacadeRetainRecall:
    """Tests for retain and recall operations."""

    def test_facade_retain_and_recall(self, facade):
        """Test that retain stores and recall retrieves memories."""
        node_id = facade.retain(content="Test memory", department="trading")
        assert node_id is not None

        results = facade.recall(department="trading")
        assert len(results) > 0
        assert any("Test memory" in n.content for n in results)

    def test_retain_returns_node_id(self, facade):
        """Test that retain returns the node ID."""
        node_id = facade.retain(content="Test content", department="trading")
        assert node_id is not None
        # Should be a string or UUID
        assert str(node_id)

    def test_recall_by_query(self, facade):
        """Test recall with text query."""
        facade.retain(content="Risk management is important", department="trading")
        facade.retain(content="Timeframe preferences", department="trading")

        results = facade.recall(query="risk")
        assert len(results) > 0
        assert "risk" in results[0].content.lower()


class TestFacadeTierManagement:
    """Tests for tier management operations."""

    def test_facade_tier_movement(self, facade):
        """Test moving nodes between tiers."""
        node_id = facade.retain(content="Test tier movement")

        # Move to cold tier
        result = facade.move_to_cold(node_id)
        assert result is not None

        # Verify node is in cold tier
        cold_nodes = facade.get_cold_nodes()
        assert any(str(n.id) == str(node_id) for n in cold_nodes)

    def test_get_hot_nodes(self, facade):
        """Test retrieving hot nodes."""
        node_id = facade.retain(content="Hot node")
        facade.move_to_hot(node_id)

        hot_nodes = facade.get_hot_nodes()
        assert len(hot_nodes) > 0

    def test_get_warm_nodes(self, facade):
        """Test retrieving warm nodes."""
        node_id = facade.retain(content="Warm node")
        facade.move_to_warm(node_id)

        warm_nodes = facade.get_warm_nodes()
        assert len(warm_nodes) > 0

    def test_promote_to_hot(self, facade):
        """Test promoting cold node to hot."""
        node_id = facade.retain(content="Promote test")
        facade.move_to_cold(node_id)

        result = facade.promote_to_hot(node_id)
        assert result is not None

        hot_nodes = facade.get_hot_nodes()
        assert any(str(n.id) == str(node_id) for n in hot_nodes)


class TestFacadeCompaction:
    """Tests for compaction operations."""

    def test_facade_should_compact(self, facade):
        """Test should_compact returns correct value."""
        # Should not compact at 40%
        assert facade.should_compact(40.0) is False
        # Should compact at 55%
        assert facade.should_compact(55.0) is True

    def test_facade_compact_trigger(self, facade):
        """Test check_and_compact triggers correctly."""
        # Create nodes with low importance
        for i in range(10):
            facade.retain(content=f"Memory {i}", importance=0.3)

        # Check with context_percent above threshold
        result = facade.check_and_compact(context_percent=55)
        assert result["triggered"] is True

    def test_compact_old_nodes(self, facade):
        """Test compact_old_nodes method."""
        result = facade.compact_old_nodes()
        assert "nodes_compacted" in result


class TestFacadeStats:
    """Tests for statistics."""

    def test_stats(self, facade):
        """Test get_stats returns node counts."""
        facade.retain(content="Test1", department="trading")
        facade.retain(content="Test2", department="trading")

        stats = facade.get_stats()
        assert stats["total_nodes"] >= 2

    def test_stats_includes_tiers(self, facade):
        """Test stats include tier information."""
        facade.retain(content="Hot node")
        facade.retain(content="Cold node")

        stats = facade.get_stats()
        assert "hot" in stats
        assert "warm" in stats
        assert "cold" in stats


class TestFacadeReflect:
    """Tests for reflect operation."""

    def test_reflect(self, facade):
        """Test reflect synthesizes answer."""
        facade.retain(content="User prefers EURUSD", department="trading")
        facade.retain(content="User trades on M15", department="trading")

        result = facade.reflect(query="What does user prefer?", department="trading")
        assert result["synthesized"] is True
        assert "EURUSD" in result["answer"]


class TestFacadeLink:
    """Tests for link operation."""

    def test_link(self, facade):
        """Test linking nodes together."""
        node1_id = facade.retain(content="First node")
        node2_id = facade.retain(content="Second node")

        result = facade.link(node1_id, [node2_id])
        assert len(result) > 0


class TestGraphMemorySingleton:
    """Tests for the get_graph_memory singleton."""

    def test_get_graph_memory_returns_facade(self):
        """Test that get_graph_memory returns a GraphMemoryFacade."""
        with tempfile.TemporaryDirectory() as tmpdir:
            facade = get_graph_memory(db_path=Path(tmpdir) / "singleton.db")
            assert isinstance(facade, GraphMemoryFacade)

    def test_get_graph_memory_singleton(self):
        """Test that get_graph_memory returns same instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            facade1 = get_graph_memory(db_path=Path(tmpdir) / "singleton.db")
            facade2 = get_graph_memory(db_path=Path(tmpdir) / "singleton.db")
            # Both should be the same instance (singleton)
            assert facade1 is facade2

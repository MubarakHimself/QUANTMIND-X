"""Tests for MemoryTierManager."""
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.memory.graph.store import GraphMemoryStore
from src.memory.graph.tier_manager import MemoryTierManager
from src.memory.graph.types import MemoryCategory, MemoryNode, MemoryNodeType, MemoryTier


@pytest.fixture
def tier_mgr():
    """Create a tier manager with a temporary database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = GraphMemoryStore(Path(tmpdir) / "test.db")
        yield MemoryTierManager(store)


class TestTierAssignmentOnCreate:
    """Tests for tier assignment when creating nodes."""

    def test_tier_assignment_on_create(self, tier_mgr):
        """Test that tier is correctly assigned when creating a node."""
        node = MemoryNode(
            title="Hot",
            content="content",
            node_type=MemoryNodeType.WORLD,
            category=MemoryCategory.FACTUAL,
            tier=MemoryTier.HOT,
        )
        tier_mgr.store.create_node(node)
        retrieved = tier_mgr.store.get_node(str(node.id))
        assert retrieved.tier == MemoryTier.HOT


class TestMoveToCold:
    """Tests for moving nodes to cold tier."""

    def test_move_to_cold(self, tier_mgr):
        """Test moving a node to cold tier."""
        node = MemoryNode(
            title="Test",
            content="content",
            node_type=MemoryNodeType.WORLD,
            category=MemoryCategory.FACTUAL,
        )
        tier_mgr.store.create_node(node)
        tier_mgr.move_to_cold(node.id)
        updated = tier_mgr.store.get_node(str(node.id))
        assert updated.tier == MemoryTier.COLD


class TestGetHotNodes:
    """Tests for getting hot tier nodes."""

    def test_get_hot_nodes(self, tier_mgr):
        """Test retrieving hot tier nodes."""
        now = datetime.utcnow()
        for i in range(3):
            node = MemoryNode(
                title=f"Hot {i}",
                content="content",
                node_type=MemoryNodeType.WORLD,
                category=MemoryCategory.FACTUAL,
                tier=MemoryTier.HOT,
                last_accessed=now - timedelta(minutes=i * 10),
            )
            tier_mgr.store.create_node(node)
        hot = tier_mgr.get_hot_nodes(limit=10)
        assert len(hot) == 3


class TestCompactOldNodes:
    """Tests for compacting old nodes to cold tier."""

    def test_compact_old_nodes(self, tier_mgr):
        """Test compacting old low-importance nodes to cold tier."""
        old_date = datetime(2025, 1, 1, tzinfo=timezone.utc)  # Use timezone-aware datetime
        for i in range(3):
            node = MemoryNode(
                title=f"Old {i}",
                content=f"Content {i}",
                node_type=MemoryNodeType.WORLD,
                category=MemoryCategory.FACTUAL,
                created_at=old_date,
                importance=0.2,  # Less than min_importance=0.3
                tier=MemoryTier.WARM,
            )
            tier_mgr.store.create_node(node)
        result = tier_mgr.compact_old_nodes(days_old=30, max_nodes=10, min_importance=0.3)
        assert result["nodes_compacted"] > 0


class TestGetTierStats:
    """Tests for getting tier statistics."""

    def test_get_tier_stats(self, tier_mgr):
        """Test retrieving tier statistics."""
        for tier in [MemoryTier.HOT, MemoryTier.WARM, MemoryTier.COLD, MemoryTier.WARM]:
            node = MemoryNode(
                title="Node",
                content="content",
                node_type=MemoryNodeType.WORLD,
                category=MemoryCategory.FACTUAL,
                tier=tier,
            )
            tier_mgr.store.create_node(node)
        stats = tier_mgr.get_tier_stats()
        assert stats[MemoryTier.WARM] == 2


class TestMoveToHot:
    """Tests for moving nodes to hot tier."""

    def test_move_to_hot(self, tier_mgr):
        """Test moving a node to hot tier."""
        node = MemoryNode(
            title="Test",
            content="content",
            node_type=MemoryNodeType.WORLD,
            category=MemoryCategory.FACTUAL,
            tier=MemoryTier.COLD,
        )
        tier_mgr.store.create_node(node)
        tier_mgr.move_to_hot(node.id)
        updated = tier_mgr.store.get_node(str(node.id))
        assert updated.tier == MemoryTier.HOT


class TestMoveToWarm:
    """Tests for moving nodes to warm tier."""

    def test_move_to_warm(self, tier_mgr):
        """Test moving a node to warm tier."""
        node = MemoryNode(
            title="Test",
            content="content",
            node_type=MemoryNodeType.WORLD,
            category=MemoryCategory.FACTUAL,
            tier=MemoryTier.HOT,
        )
        tier_mgr.store.create_node(node)
        tier_mgr.move_to_warm(node.id)
        updated = tier_mgr.store.get_node(str(node.id))
        assert updated.tier == MemoryTier.WARM


class TestPromoteHot:
    """Tests for promoting nodes to hot tier."""

    def test_promote_hot_from_cold(self, tier_mgr):
        """Test promoting a cold node to hot tier."""
        node = MemoryNode(
            title="Test",
            content="content",
            node_type=MemoryNodeType.WORLD,
            category=MemoryCategory.FACTUAL,
            tier=MemoryTier.COLD,
            last_accessed=None,
        )
        tier_mgr.store.create_node(node)
        tier_mgr.promote_hot(node.id)
        updated = tier_mgr.store.get_node(str(node.id))
        assert updated.tier == MemoryTier.HOT
        assert updated.last_accessed is not None


class TestGetNodesByTier:
    """Tests for getting nodes by specific tier."""

    def test_get_warm_nodes(self, tier_mgr):
        """Test retrieving warm tier nodes."""
        for i in range(2):
            node = MemoryNode(
                title=f"Warm {i}",
                content="content",
                node_type=MemoryNodeType.WORLD,
                category=MemoryCategory.FACTUAL,
                tier=MemoryTier.WARM,
            )
            tier_mgr.store.create_node(node)
        warm = tier_mgr.get_nodes_by_tier(MemoryTier.WARM, limit=10)
        assert len(warm) == 2

    def test_get_cold_nodes(self, tier_mgr):
        """Test retrieving cold tier nodes."""
        for i in range(2):
            node = MemoryNode(
                title=f"Cold {i}",
                content="content",
                node_type=MemoryNodeType.WORLD,
                category=MemoryCategory.FACTUAL,
                tier=MemoryTier.COLD,
            )
            tier_mgr.store.create_node(node)
        cold = tier_mgr.get_nodes_by_tier(MemoryTier.COLD, limit=10)
        assert len(cold) == 2


class TestTierManagerConstants:
    """Tests for tier manager constants."""

    def test_hot_threshold_hours(self, tier_mgr):
        """Test HOT_THRESHOLD_HOURS constant."""
        assert tier_mgr.HOT_THRESHOLD_HOURS == 1

    def test_warm_threshold_days(self, tier_mgr):
        """Test WARM_THRESHOLD_DAYS constant."""
        assert tier_mgr.WARM_THRESHOLD_DAYS == 30

"""Tests for Migration Path & Wrapper."""
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from pathlib import Path

from src.memory.graph.migration import (
    MemoryMigrator,
    DualWriteWrapper,
)


class TestMemoryMigrator:
    """Tests for MemoryMigrator class."""

    def test_initialization(self):
        """Test initialization."""
        source_facade = MagicMock()
        target_facade = MagicMock()

        migrator = MemoryMigrator(source_facade, target_facade)

        assert migrator.source == source_facade
        assert migrator.target == target_facade

    def test_migrate_entries_basic(self):
        """Test basic migration of entries."""
        source_facade = MagicMock()
        target_facade = MagicMock()

        # Mock source recall
        mock_node1 = MagicMock()
        mock_node1.id = "node-1"
        mock_node1.content = "Content 1"
        mock_node1.title = "Title 1"
        mock_node1.importance = 0.8
        mock_node1.tags = ["tag1"]
        mock_node1.department = "research"
        mock_node1.agent_id = "agent-001"
        mock_node1.session_id = "session-001"
        mock_node1.role = None

        mock_node2 = MagicMock()
        mock_node2.id = "node-2"
        mock_node2.content = "Content 2"
        mock_node2.title = "Title 2"
        mock_node2.importance = 0.6
        mock_node2.tags = ["tag2"]
        mock_node2.department = "development"
        mock_node2.agent_id = "agent-002"
        mock_node2.session_id = "session-001"
        mock_node2.role = None

        source_facade.recall = MagicMock(return_value=[mock_node1, mock_node2])
        target_facade.retain = MagicMock(return_value="new-node-1")

        migrator = MemoryMigrator(source_facade, target_facade)

        result = migrator.migrate_entries(limit=100)

        assert result["migrated"] == 2
        assert result["failed"] == 0
        assert target_facade.retain.call_count == 2

    def test_migrate_entries_with_department_filter(self):
        """Test migration with department filter."""
        source_facade = MagicMock()
        target_facade = MagicMock()

        source_facade.recall = MagicMock(return_value=[])
        target_facade.retain = MagicMock(return_value="new-node-1")

        migrator = MemoryMigrator(source_facade, target_facade)

        result = migrator.migrate_entries(department="research")

        source_facade.recall.assert_called_once()
        call_kwargs = source_facade.recall.call_args.kwargs

        assert call_kwargs["department"] == "research"

    def test_verify_migration(self):
        """Test migration verification."""
        source_facade = MagicMock()
        target_facade = MagicMock()

        # Mock source has 10 entries
        source_facade.get_stats = MagicMock(return_value={
            "total_nodes": 10,
            "hot": 5,
            "warm": 3,
            "cold": 2,
        })

        # Mock target has 10 entries
        target_facade.get_stats = MagicMock(return_value={
            "total_nodes": 10,
            "hot": 5,
            "warm": 3,
            "cold": 2,
        })

        migrator = MemoryMigrator(source_facade, target_facade)

        result = migrator.verify_migration()

        assert result["verified"] is True
        assert result["source_total"] == 10
        assert result["target_total"] == 10


class TestDualWriteWrapper:
    """Tests for DualWriteWrapper class."""

    def test_initialization(self):
        """Test initialization."""
        legacy_facade = MagicMock()
        graph_facade = MagicMock()

        wrapper = DualWriteWrapper(legacy_facade, graph_facade)

        assert wrapper.legacy == legacy_facade
        assert wrapper.graph == graph_facade

    def test_add_memory_writes_to_both(self):
        """Test add_memory writes to both systems."""
        legacy_facade = MagicMock()
        graph_facade = MagicMock()

        legacy_facade.add_memory = MagicMock(return_value="legacy-id")
        graph_facade.retain = MagicMock(return_value="graph-id")

        wrapper = DualWriteWrapper(legacy_facade, graph_facade)

        result = wrapper.add_memory(
            key="test_key",
            value="test_value",
            namespace="test",
        )

        # Both should be called
        legacy_facade.add_memory.assert_called_once()
        graph_facade.retain.assert_called_once()

    def test_search_queries_graph_first(self):
        """Test search queries graph first, falls back to legacy."""
        legacy_facade = MagicMock()
        graph_facade = MagicMock()

        # Mock graph returns results
        mock_result = MagicMock()
        mock_result.id = "graph-node-1"
        graph_facade.recall = MagicMock(return_value=[mock_result])

        wrapper = DualWriteWrapper(legacy_facade, graph_facade)

        result = wrapper.search("test query")

        # Graph should be queried first
        graph_facade.recall.assert_called_once()

    def test_search_falls_back_to_legacy(self):
        """Test search falls back to legacy when graph returns nothing."""
        legacy_facade = MagicMock()
        graph_facade = MagicMock()

        # Mock graph returns empty
        graph_facade.recall = MagicMock(return_value=[])

        # Mock legacy returns results
        mock_legacy_result = MagicMock()
        mock_legacy_result.key = "legacy-key"
        legacy_facade.search = MagicMock(return_value=MagicMock(entries=[mock_legacy_result]))

        wrapper = DualWriteWrapper(legacy_facade, graph_facade)

        result = wrapper.search("test query")

        # Legacy should be called as fallback
        legacy_facade.search.assert_called_once()

    def test_get_stats_combines(self):
        """Test get_stats combines both systems."""
        legacy_facade = MagicMock()
        graph_facade = MagicMock()

        legacy_facade.get_stats = MagicMock(return_value={
            "total_entries": 50,
            "agent_memory": {},
        })

        graph_facade.get_stats = MagicMock(return_value={
            "total_nodes": 30,
            "hot": 10,
            "warm": 10,
            "cold": 10,
        })

        wrapper = DualWriteWrapper(legacy_facade, graph_facade)

        result = wrapper.get_stats()

        assert result["legacy_total"] == 50
        assert result["graph_total"] == 30
        assert result["combined_total"] == 80

"""Tests for GraphMemoryTools - Agent Tool Interface."""
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

from src.memory.graph.tools import GraphMemoryTools
from src.memory.graph.facade import GraphMemoryFacade
from pathlib import Path


@pytest.fixture
def mock_facade():
    """Create a mock GraphMemoryFacade."""
    facade = MagicMock()
    facade.retain = MagicMock(return_value="node-123")
    facade.recall = MagicMock(return_value=[])
    facade.reflect = MagicMock(return_value={"answer": "test", "sources": [], "synthesized": True})
    facade.link = MagicMock(return_value=[])
    facade.should_compact = MagicMock(return_value=False)
    facade.check_and_compact = MagicMock(return_value={"triggered": False})
    facade.get_stats = MagicMock(return_value={"total_nodes": 100, "hot": 50, "warm": 30, "cold": 20})
    return facade


class TestGraphMemoryToolsRetain:
    """Tests for retain method."""

    def test_retain_basic(self, mock_facade):
        """Test basic retain operation."""
        tools = GraphMemoryTools(mock_facade)

        result = tools.retain(content="Test content", importance=0.8)

        mock_facade.retain.assert_called_once()
        call_kwargs = mock_facade.retain.call_args.kwargs

        assert call_kwargs["content"] == "Test content"
        assert call_kwargs["importance"] == 0.8

        # Result should be a dict with node_id
        assert result["node_id"] == "node-123"

    def test_retain_with_all_params(self, mock_facade):
        """Test retain with all parameters."""
        tools = GraphMemoryTools(mock_facade)

        result = tools.retain(
            content="Test content",
            importance=0.7,
            tags=["tag1", "tag2"],
            department="research",
            agent_id="agent-001",
            session_id="session-001",
            related_to=["node-1", "node-2"],
        )

        call_kwargs = mock_facade.retain.call_args.kwargs

        assert call_kwargs["content"] == "Test content"
        assert call_kwargs["importance"] == 0.7
        assert call_kwargs["tags"] == ["tag1", "tag2"]
        assert call_kwargs["department"] == "research"
        assert call_kwargs["agent_id"] == "agent-001"
        assert call_kwargs["session_id"] == "session-001"
        assert call_kwargs["related_to"] == ["node-1", "node-2"]

    def test_retain_defaults_importance(self, mock_facade):
        """Test retain uses default importance."""
        tools = GraphMemoryTools(mock_facade)

        tools.retain(content="Test")

        call_kwargs = mock_facade.retain.call_args.kwargs
        assert call_kwargs["importance"] == 0.5


class TestGraphMemoryToolsRecall:
    """Tests for recall method."""

    def test_recall_basic(self, mock_facade):
        """Test basic recall operation."""
        mock_node = MagicMock()
        mock_node.id = "node-123"
        mock_node.content = "Test content"
        mock_node.title = "Test"
        mock_node.importance = 0.8
        mock_node.tags = ["tag1"]
        mock_node.department = "research"
        mock_node.agent_id = "agent-001"
        mock_node.session_id = "session-001"
        mock_node.created_at = datetime.now(timezone.utc)
        mock_facade.recall.return_value = [mock_node]

        tools = GraphMemoryTools(mock_facade)

        result = tools.recall(query="test")

        mock_facade.recall.assert_called_once()
        call_kwargs = mock_facade.recall.call_args.kwargs
        assert call_kwargs["query"] == "test"

        # Result should be a list of dicts
        assert len(result) == 1
        assert result[0]["id"] == "node-123"

    def test_recall_with_filters(self, mock_facade):
        """Test recall with various filters."""
        tools = GraphMemoryTools(mock_facade)

        tools.recall(
            query="test",
            department="research",
            agent_id="agent-001",
            session_id="session-001",
            role="analyst",
            tags=["important"],
            min_importance=0.7,
            limit=20,
            traverse=True,
        )

        call_kwargs = mock_facade.recall.call_args.kwargs

        assert call_kwargs["query"] == "test"
        assert call_kwargs["department"] == "research"
        assert call_kwargs["agent_id"] == "agent-001"
        assert call_kwargs["session_id"] == "session-001"
        assert call_kwargs["role"] == "analyst"
        assert call_kwargs["tags"] == ["important"]
        assert call_kwargs["min_importance"] == 0.7
        assert call_kwargs["limit"] == 20

    def test_recall_returns_serialized_nodes(self, mock_facade):
        """Test recall returns properly serialized nodes."""
        mock_node = MagicMock()
        mock_node.id = "node-123"
        mock_node.content = "Test content"
        mock_node.title = "Test"
        mock_node.importance = 0.8
        mock_node.tags = ["tag1"]
        mock_node.department = "research"
        mock_node.agent_id = "agent-001"
        mock_node.session_id = "session-001"
        mock_node.created_at = datetime.now(timezone.utc)
        mock_node.updated_at = datetime.now(timezone.utc)
        mock_node.access_count = 5
        mock_node.last_accessed = datetime.now(timezone.utc)
        mock_facade.recall.return_value = [mock_node]

        tools = GraphMemoryTools(mock_facade)

        result = tools.recall(query="test")

        # Check serialized fields
        node_dict = result[0]
        assert "id" in node_dict
        assert "content" in node_dict
        assert "title" in node_dict
        assert "importance" in node_dict


class TestGraphMemoryToolsReflect:
    """Tests for reflect method."""

    def test_reflect_basic(self, mock_facade):
        """Test basic reflect operation."""
        tools = GraphMemoryTools(mock_facade)

        result = tools.reflect(query="What did I learn?")

        mock_facade.reflect.assert_called_once()
        call_kwargs = mock_facade.reflect.call_args.kwargs

        assert call_kwargs["query"] == "What did I learn?"

        # Result should be a dict
        assert "answer" in result

    def test_reflect_with_context(self, mock_facade):
        """Test reflect with context."""
        tools = GraphMemoryTools(mock_facade)

        tools.reflect(query="What did I learn?", context="recent trading")

        call_kwargs = mock_facade.reflect.call_args.kwargs

        assert call_kwargs["query"] == "What did I learn?"
        assert call_kwargs["context"] == "recent trading"


class TestGraphMemoryToolsLink:
    """Tests for link method."""

    def test_link_basic(self, mock_facade):
        """Test basic link operation."""
        tools = GraphMemoryTools(mock_facade)

        result = tools.link(
            source_id="node-1",
            target_id="node-2",
            relation="related_to",
        )

        mock_facade.link.assert_called_once()
        call_kwargs = mock_facade.link.call_args.kwargs

        assert call_kwargs["source_id"] == "node-1"
        assert call_kwargs["target_ids"] == ["node-2"]
        assert call_kwargs["relation_type"] == "related_to"

    def test_link_with_strength(self, mock_facade):
        """Test link with strength."""
        tools = GraphMemoryTools(mock_facade)

        tools.link(
            source_id="node-1",
            target_id="node-2",
            relation="related_to",
            strength=0.9,
        )

        call_kwargs = mock_facade.link.call_args.kwargs
        assert call_kwargs["strength"] == 0.9


class TestGraphMemoryToolsCheckCompaction:
    """Tests for check_compaction method."""

    def test_check_compaction_returns_dict(self, mock_facade):
        """Test check_compaction returns proper dict."""
        mock_facade.check_and_compact.return_value = {
            "triggered": True,
            "compacted_count": 10,
            "anchored_count": 5,
        }

        tools = GraphMemoryTools(mock_facade)

        result = tools.check_compaction(context_percent=80.0)

        mock_facade.check_and_compact.assert_called_once()
        call_kwargs = mock_facade.check_and_compact.call_args.kwargs

        assert call_kwargs["context_percent"] == 80.0

        assert "triggered" in result


class TestGraphMemoryToolsGetStats:
    """Tests for get_stats method."""

    def test_get_stats_returns_dict(self, mock_facade):
        """Test get_stats returns proper dict."""
        tools = GraphMemoryTools(mock_facade)

        result = tools.get_stats()

        mock_facade.get_stats.assert_called_once()

        assert "total_nodes" in result
        assert result["total_nodes"] == 100


class TestGraphMemoryToolsIntegration:
    """Integration tests with real facade."""

    def test_with_real_facade(self):
        """Test with a real (non-mocked) facade."""
        facade = GraphMemoryFacade(db_path=Path(":memory:"))
        tools = GraphMemoryTools(facade)

        # Test retain
        result = tools.retain(content="Test content", importance=0.7)
        assert "node_id" in result

        # Test get_stats
        stats = tools.get_stats()
        assert "total_nodes" in stats

        facade.close()

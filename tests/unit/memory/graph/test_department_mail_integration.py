"""Tests for Department Mail Integration."""
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

from src.memory.graph.integrations import DepartmentMailIntegration
from src.agents.departments.department_mail import (
    DepartmentMessage,
    MessageType,
    Priority,
)


@pytest.fixture
def mock_facade():
    """Create a mock GraphMemoryFacade."""
    facade = MagicMock()
    facade.retain = MagicMock(return_value="node-123")
    facade.recall = MagicMock(return_value=[])
    facade.link = MagicMock(return_value=[])
    return facade


@pytest.fixture
def sample_message():
    """Create a sample DepartmentMessage."""
    return DepartmentMessage(
        id="msg-001",
        from_dept="research",
        to_dept="development",
        type=MessageType.STRATEGY_DISPATCH,
        subject="New trading strategy",
        body="We have developed a new momentum strategy for review.",
        priority=Priority.HIGH,
        timestamp=datetime.now(timezone.utc),
        read=False,
        workflow_id="wf-001",
    )


@pytest.fixture
def sample_thread():
    """Create a sample message thread."""
    base_time = datetime.now(timezone.utc)
    return [
        DepartmentMessage(
            id="msg-001",
            from_dept="research",
            to_dept="development",
            type=MessageType.STATUS,
            subject="Initial request",
            body="Please implement this strategy.",
            priority=Priority.NORMAL,
            timestamp=base_time,
            read=True,
            workflow_id="wf-001",
        ),
        DepartmentMessage(
            id="msg-002",
            from_dept="development",
            to_dept="research",
            type=MessageType.RESULT,
            subject="Implementation complete",
            body="Strategy has been implemented.",
            priority=Priority.NORMAL,
            timestamp=base_time,
            read=True,
            workflow_id="wf-001",
        ),
        DepartmentMessage(
            id="msg-003",
            from_dept="research",
            to_dept="development",
            type=MessageType.QUESTION,
            subject="Question about implementation",
            body="How does the position sizing work?",
            priority=Priority.NORMAL,
            timestamp=base_time,
            read=False,
            workflow_id="wf-001",
        ),
    ]


class TestConvertMessageToMemory:
    """Tests for convert_message_to_memory method."""

    def test_convert_basic_message(self, mock_facade, sample_message):
        """Test converting a basic message to memory."""
        integration = DepartmentMailIntegration(mock_facade)

        node_id = integration.convert_message_to_memory(
            sample_message, importance=0.7
        )

        # Verify retain was called
        mock_facade.retain.assert_called_once()
        call_args = mock_facade.retain.call_args

        # Check content contains message info
        content = call_args.kwargs["content"]
        assert "New trading strategy" in content
        assert "research" in content
        assert "development" in content

        # Check metadata
        assert call_args.kwargs["importance"] == 0.7
        assert "department_mail" in call_args.kwargs["tags"]
        assert "msg-msg-001" in call_args.kwargs["tags"]
        assert call_args.kwargs["department"] == "research"

        assert node_id == "node-123"

    def test_convert_message_with_workflow(self, mock_facade, sample_message):
        """Test converting message with workflow ID."""
        integration = DepartmentMailIntegration(mock_facade)

        integration.convert_message_to_memory(sample_message, importance=0.5)

        call_args = mock_facade.retain.call_args
        tags = call_args.kwargs["tags"]

        assert "workflow:wf-001" in tags


class TestConvertThreadToMemory:
    """Tests for convert_thread_to_memory method."""

    def test_convert_thread(self, mock_facade, sample_thread):
        """Test converting a message thread."""
        integration = DepartmentMailIntegration(mock_facade)

        node_ids = integration.convert_thread_to_memory(
            sample_thread, thread_id="thread-001"
        )

        # Should create 3 nodes
        assert len(node_ids) == 3

        # Verify retain was called 3 times
        assert mock_facade.retain.call_count == 3

    def test_thread_creates_links(self, mock_facade, sample_thread):
        """Test that thread conversion creates links between messages."""
        mock_facade.link = MagicMock(return_value=[])
        integration = DepartmentMailIntegration(mock_facade)

        node_ids = ["node-1", "node-2", "node-3"]
        mock_facade.retain = MagicMock(side_effect=node_ids)

        integration.convert_thread_to_memory(sample_thread, thread_id="thread-001")

        # Should create 2 links (msg-001->msg-002, msg-002->msg-003)
        assert mock_facade.link.call_count == 2


class TestLinkMessages:
    """Tests for link_messages method."""

    def test_link_two_messages(self, mock_facade):
        """Test linking two message nodes."""
        integration = DepartmentMailIntegration(mock_facade)

        integration.link_messages(
            source_id="node-1",
            target_id="node-2",
            relation="replies_to",
            context="Response to original message",
        )

        mock_facade.link.assert_called_once()
        call_args = mock_facade.link.call_args

        assert call_args.kwargs["source_id"] == "node-1"
        assert call_args.kwargs["target_ids"] == ["node-2"]
        assert call_args.kwargs["relation_type"] == "replies_to"


class TestSearchMailMemory:
    """Tests for search_mail_memory method."""

    def test_search_by_query(self, mock_facade):
        """Test searching by query."""
        mock_node = MagicMock()
        mock_node.id = "node-123"
        mock_node.content = "Strategy implementation complete"
        mock_facade.recall.return_value = [mock_node]

        integration = DepartmentMailIntegration(mock_facade)

        results = integration.search_mail_memory(query="strategy")

        mock_facade.recall.assert_called_once()
        call_args = mock_facade.recall.call_args

        assert call_args.kwargs["query"] == "strategy"
        assert len(results) == 1

    def test_search_by_department(self, mock_facade):
        """Test searching by department."""
        integration = DepartmentMailIntegration(mock_facade)

        integration.search_mail_memory(department="research")

        call_args = mock_facade.recall.call_args
        assert call_args.kwargs["department"] == "research"

    def test_search_by_message_type(self, mock_facade):
        """Test searching by message type."""
        integration = DepartmentMailIntegration(mock_facade)

        integration.search_mail_memory(message_type=MessageType.STATUS)

        call_args = mock_facade.recall.call_args
        assert "status" in call_args.kwargs["tags"]

    def test_combined_filters(self, mock_facade):
        """Test combining multiple filters."""
        integration = DepartmentMailIntegration(mock_facade)

        integration.search_mail_memory(
            query="strategy",
            department="research",
            message_type=MessageType.DISPATCH,
        )

        call_args = mock_facade.recall.call_args

        assert call_args.kwargs["query"] == "strategy"
        assert call_args.kwargs["department"] == "research"
        assert "dispatch" in call_args.kwargs["tags"]


class TestDepartmentMailIntegration:
    """Integration class tests."""

    def test_initialization(self, mock_facade):
        """Test initialization."""
        integration = DepartmentMailIntegration(mock_facade)

        assert integration.facade == mock_facade

    def test_with_real_facade(self):
        """Test with a real (non-mocked) facade."""
        from src.memory.graph.facade import GraphMemoryFacade
        from pathlib import Path

        facade = GraphMemoryFacade(db_path=Path(":memory:"))
        integration = DepartmentMailIntegration(facade)

        assert integration.facade == facade

        facade.close()

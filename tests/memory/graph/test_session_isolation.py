"""Tests for session workspace isolation (Story 7.5).

These tests verify:
- Session isolation between concurrent sessions
- Commit workflow and visibility
- Conflict detection and resolution
"""
import pytest
from datetime import datetime, timezone
from pathlib import Path
import tempfile

from src.memory.graph.store import GraphMemoryStore
from src.memory.graph.facade import GraphMemoryFacade
from src.memory.graph.types import (
    MemoryNode,
    MemoryNodeType,
    MemoryCategory,
    SessionStatus,
)


@pytest.fixture
def store():
    """Create an in-memory store for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    store = GraphMemoryStore(db_path)
    yield store
    store.close()
    db_path.unlink(missing_ok=True)


@pytest.fixture
def facade():
    """Create an in-memory facade for testing."""
    facade = GraphMemoryFacade(db_path=Path(":memory:"))
    yield facade
    facade.close()


class TestSessionIsolation:
    """Test session workspace isolation."""

    def test_draft_nodes_isolated_by_session(self, store):
        """Test that draft nodes are isolated by session_id (AC #1)."""
        # Create nodes for session A
        node_a1 = MemoryNode(
            node_type=MemoryNodeType.OBSERVATION,
            category=MemoryCategory.EXPERIENTIAL,
            title="Session A node 1",
            content="Content from session A",
            session_id="session-a",
            session_status=SessionStatus.DRAFT,
            importance=0.5,
        )
        node_a2 = MemoryNode(
            node_type=MemoryNodeType.OBSERVATION,
            category=MemoryCategory.EXPERIENTIAL,
            title="Session A node 2",
            content="Another from session A",
            session_id="session-a",
            session_status=SessionStatus.DRAFT,
            importance=0.6,
        )

        # Create nodes for session B
        node_b1 = MemoryNode(
            node_type=MemoryNodeType.OBSERVATION,
            category=MemoryCategory.EXPERIENTIAL,
            title="Session B node 1",
            content="Content from session B",
            session_id="session-b",
            session_status=SessionStatus.DRAFT,
            importance=0.7,
        )

        store.create_node(node_a1)
        store.create_node(node_a2)
        store.create_node(node_b1)

        # Query draft nodes for session A - should only see A's nodes
        draft_a = store.get_nodes_by_session_status(SessionStatus.DRAFT, session_id="session-a")
        assert len(draft_a) == 2
        assert all(n.session_id == "session-a" for n in draft_a)

        # Query draft nodes for session B - should only see B's nodes
        draft_b = store.get_nodes_by_session_status(SessionStatus.DRAFT, session_id="session-b")
        assert len(draft_b) == 1
        assert all(n.session_id == "session-b" for n in draft_b)

    def test_committed_nodes_visible_to_all(self, store):
        """Test that committed nodes are visible to all sessions (AC #2)."""
        # Create nodes for different sessions with different statuses
        node_draft = MemoryNode(
            node_type=MemoryNodeType.OBSERVATION,
            category=MemoryCategory.EXPERIENTIAL,
            title="Draft node",
            content="Draft content",
            session_id="session-a",
            session_status=SessionStatus.DRAFT,
            importance=0.5,
        )
        node_committed = MemoryNode(
            node_type=MemoryNodeType.OBSERVATION,
            category=MemoryCategory.EXPERIENTIAL,
            title="Committed node",
            content="Committed content",
            session_id="session-b",
            session_status=SessionStatus.COMMITTED,
            importance=0.6,
        )

        store.create_node(node_draft)
        store.create_node(node_committed)

        # Query committed nodes - should see all committed
        committed = store.get_nodes_by_session_status(SessionStatus.COMMITTED)
        assert len(committed) == 1
        assert committed[0].title == "Committed node"

    def test_session_query_filters_draft_by_session(self, store):
        """Test that query filters draft nodes by session_id (Subtask 1.2)."""
        # Create nodes
        for i, (session, status) in enumerate([
            ("session-1", SessionStatus.DRAFT),
            ("session-1", SessionStatus.DRAFT),
            ("session-2", SessionStatus.DRAFT),
            ("session-1", SessionStatus.COMMITTED),
        ]):
            node = MemoryNode(
                node_type=MemoryNodeType.OBSERVATION,
                category=MemoryCategory.FACTUAL,
                title=f"Node {i}",
                content=f"Content {i}",
                session_id=session,
                session_status=status,
                importance=0.5,
            )
            store.create_node(node)

        # Query session-1 draft only
        results = store.query_nodes(session_id="session-1", session_status=SessionStatus.DRAFT)
        assert len(results) == 2
        assert all(n.session_id == "session-1" for n in results)

        # Query session-2 draft only
        results = store.query_nodes(session_id="session-2", session_status=SessionStatus.DRAFT)
        assert len(results) == 1
        assert results[0].session_id == "session-2"


class TestSessionCommit:
    """Test session commit workflow."""

    def test_commit_session_updates_status(self, store):
        """Test that committing a session changes status to committed (AC #2)."""
        # Create draft nodes
        node = MemoryNode(
            node_type=MemoryNodeType.OPINION,
            category=MemoryCategory.SUBJECTIVE,
            title="Draft opinion",
            content="My opinion",
            session_id="session-123",
            session_status=SessionStatus.DRAFT,
            department="research",
            importance=0.7,
        )
        store.create_node(node)

        # Commit the session
        result = store.commit_session(session_id="session-123")

        assert result["node_count"] == 1
        assert result["session_id"] == "session-123"

        # Verify node is now committed
        updated = store.get_node(str(node.id))
        assert updated.session_status == SessionStatus.COMMITTED

    def test_commit_log_records_metadata(self, store):
        """Test that commit log records session metadata (Subtask 2.2)."""
        # Create and commit nodes
        for i in range(3):
            node = MemoryNode(
                node_type=MemoryNodeType.OBSERVATION,
                category=MemoryCategory.EXPERIENTIAL,
                title=f"Node {i}",
                content=f"Content {i}",
                session_id="session-abc",
                session_status=SessionStatus.DRAFT,
                department="development",
                importance=0.5,
            )
            store.create_node(node)

        result = store.commit_session(session_id="session-abc", department="development")

        # Check commit log
        log = store.get_commit_log(session_id="session-abc")
        assert len(log) == 3
        assert all(entry["session_id"] == "session-abc" for entry in log)

    def test_commit_with_department_filter(self, store):
        """Test commit respects department filter."""
        # Create nodes for different departments
        for dept in ["research", "development"]:
            for i in range(2):
                node = MemoryNode(
                    node_type=MemoryNodeType.OBSERVATION,
                    category=MemoryCategory.FACTUAL,
                    title=f"{dept} node {i}",
                    content=f"Content {i}",
                    session_id="session-xyz",
                    session_status=SessionStatus.DRAFT,
                    department=dept,
                    importance=0.5,
                )
                store.create_node(node)

        # Commit only research department
        result = store.commit_session(session_id="session-xyz", department="research")

        assert result["node_count"] == 2
        assert result["department"] == "research"

        # Verify research is committed, development is not
        research_nodes = store.query_nodes(
            session_id="session-xyz",
            department="research",
            session_status=SessionStatus.COMMITTED,
        )
        dev_nodes = store.query_nodes(
            session_id="session-xyz",
            department="development",
            session_status=SessionStatus.DRAFT,
        )

        assert len(research_nodes) == 2
        assert len(dev_nodes) == 2


class TestConflictDetection:
    """Test concurrent conflict detection."""

    def test_detect_conflicts_same_strategy(self, store):
        """Test detection of concurrent writes to same strategy (AC #3)."""
        # Session A writes to strategy-1
        node_a = MemoryNode(
            node_type=MemoryNodeType.DECISION,
            category=MemoryCategory.SUBJECTIVE,
            title="Decision from session A",
            content="We should buy",
            session_id="session-a",
            session_status=SessionStatus.DRAFT,
            entity_id="strategy-1",  # Same strategy
            importance=0.8,
        )
        store.create_node(node_a)

        # Session B writes to same strategy
        node_b = MemoryNode(
            node_type=MemoryNodeType.DECISION,
            category=MemoryCategory.SUBJECTIVE,
            title="Decision from session B",
            content="We should sell",
            session_id="session-b",
            session_status=SessionStatus.DRAFT,
            entity_id="strategy-1",  # Same strategy
            importance=0.9,
        )
        store.create_node(node_b)

        # Detect conflicts - session B checking against session A
        conflicts = store.detect_conflicts(strategy_id="strategy-1", exclude_session_id="session-b")

        assert len(conflicts) == 1
        assert conflicts[0].session_id == "session-a"

    def test_no_conflict_different_strategy(self, store):
        """Test no conflicts when sessions work on different strategies."""
        # Session A writes to strategy-1
        node_a = MemoryNode(
            node_type=MemoryNodeType.DECISION,
            category=MemoryCategory.SUBJECTIVE,
            title="Decision A",
            content="Content A",
            session_id="session-a",
            session_status=SessionStatus.DRAFT,
            entity_id="strategy-1",
            importance=0.8,
        )
        store.create_node(node_a)

        # Session B writes to different strategy
        node_b = MemoryNode(
            node_type=MemoryNodeType.DECISION,
            category=MemoryCategory.SUBJECTIVE,
            title="Decision B",
            content="Content B",
            session_id="session-b",
            session_status=SessionStatus.DRAFT,
            entity_id="strategy-2",
            importance=0.9,
        )
        store.create_node(node_b)

        # No conflicts expected
        conflicts = store.detect_conflicts(strategy_id="strategy-2", exclude_session_id="session-b")
        assert len(conflicts) == 0


class TestFacadeIntegration:
    """Integration tests using the facade."""

    def test_query_session_isolated(self, facade):
        """Test query_session_isolated returns correct nodes."""
        # Create nodes for session-1 (draft)
        facade.retain(
            content="Draft node 1",
            session_id="session-1",
            importance=0.6,
            tags=["test"],
        )
        facade.retain(
            content="Draft node 2",
            session_id="session-1",
            importance=0.7,
            tags=["test"],
        )

        # Create committed node
        node = MemoryNode(
            node_type=MemoryNodeType.WORLD,
            category=MemoryCategory.FACTUAL,
            title="Committed fact",
            content="Important fact",
            session_id="shared",
            session_status=SessionStatus.COMMITTED,
            importance=0.9,
        )
        facade.store.create_node(node)

        # Query with session isolation
        results = facade.query_session_isolated(
            session_id="session-1",
            include_committed=True,
        )

        # Should get 2 draft nodes + 1 committed = 3
        assert len(results) >= 2  # At least the draft nodes
        assert any(n.session_status == SessionStatus.COMMITTED for n in results)

    def test_commit_session_via_facade(self, facade):
        """Test commit_session via facade."""
        # Create draft nodes
        facade.retain(
            content="Node to commit",
            session_id="test-session",
            department="research",
            importance=0.5,
        )

        # Commit
        result = facade.commit_session(session_id="test-session", department="research")

        assert result["node_count"] >= 1
        assert result["session_id"] == "test-session"

        # Verify they're now committed
        committed = facade.get_committed_nodes(session_id="test-session")
        assert len(committed) >= 1
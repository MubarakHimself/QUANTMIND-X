"""P0 Tests: Session Recovery Returns Correct Committed State (Epic 5 - R-009)

These tests verify that session recovery correctly loads committed
memories and handles edge cases like no-committed-state.

Test Design Reference: _bmad-output/test-artifacts/test-design-epic-5.md
Risk: R-009 (Score 4) - Session recovery returns stale or missing state
Priority: P0 - Critical for session continuity
"""

import pytest
import tempfile

from src.memory.graph.facade import GraphMemoryFacade
from src.memory.graph.reflection_executor import ReflectionExecutor
from src.memory.graph.types import (
    MemoryNode,
    MemoryNodeType,
    MemoryCategory,
    SessionStatus,
)


class TestSessionRecovery:
    """Test R-009: Session recovery returns correct committed state."""

    def test_recovery_loads_committed_state_correctly(self):
        """P0: Session recovery MUST return all committed nodes for a session.

        When recovering a session, the system must return exactly the nodes
        that were committed, not draft nodes or nodes from other sessions.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/test_recovery.db"
            facade = GraphMemoryFacade(db_path=db_path)

            # Create multiple nodes in session
            node1_id = facade.retain(
                content="First committed analysis for EURUSD",
                source="test",
                session_id="session-1",
                department="research",
                importance=0.8,
                tags=["analysis"],
            )

            node2_id = facade.retain(
                content="Second committed observation about GBPUSD",
                source="test",
                session_id="session-1",
                department="research",
                importance=0.7,
                tags=["observation"],
            )

            # Keep one node as draft (importance < 0.3 so it fails validation and stays draft)
            draft_id = facade.retain(
                content="This is a draft and should not be recovered",
                source="test",
                session_id="session-1",
                importance=0.1,
                tags=["draft"],
            )

            # Commit the first two nodes (the draft stays uncommitted due to low importance)
            executor = ReflectionExecutor(facade.store)
            executor.execute(session_id="session-1", force_commit=False)

            # Verify committed state
            result = facade.load_committed_state(session_id="session-1")

            # CRITICAL ASSERTION: Recovery must return committed nodes
            assert result["committed_nodes_count"] >= 2, (
                f"Recovery should return at least 2 committed nodes, got {result['committed_nodes_count']}"
            )

            committed_ids = [node["id"] for node in result["nodes"]]

            # Both committed nodes should be in recovery
            assert str(node1_id) in committed_ids, (
                f"Committed node {node1_id} was not in recovery result!"
            )
            assert str(node2_id) in committed_ids, (
                f"Committed node {node2_id} was not in recovery result!"
            )

            # Draft node should NOT be in recovery
            assert str(draft_id) not in committed_ids, (
                f"SECURITY VIOLATION: Draft node {draft_id} was returned in recovery! "
                f"Recovery must only return committed nodes."
            )

            facade.close()

    def test_recovery_handles_no_committed_state_edge_case(self):
        """P0: Session recovery MUST handle sessions with no committed state gracefully.

        A session that has only draft nodes (never committed) should return
        an empty result without error.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/test_empty_recovery.db"
            facade = GraphMemoryFacade(db_path=db_path)

            # Create only draft nodes, never commit
            draft_id = facade.retain(
                content="In-progress work that was never committed",
                source="test",
                session_id="session-never-committed",
                importance=0.5,
                tags=["draft"],
            )

            # Attempt recovery on session with no committed state
            result = facade.load_committed_state(session_id="session-never-committed")

            # CRITICAL: Should return empty, not error
            assert result["committed_nodes_count"] == 0, (
                f"Recovery of never-committed session should return 0, got {result['committed_nodes_count']}"
            )
            assert result["nodes"] == [], (
                f"Recovery of never-committed session should return empty list, got {result['nodes']}"
            )

            # Should not have an error field (success case)
            assert "error" not in result or result.get("error") is None, (
                f"Recovery should not have error for empty session, got: {result.get('error')}"
            )

            facade.close()

    def test_recovery_does_not_leak_nodes_from_other_sessions(self):
        """P0: Session recovery MUST NOT return nodes from other sessions.

        Each session must only see its own committed nodes during recovery.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/test_session_isolation_recovery.db"
            facade = GraphMemoryFacade(db_path=db_path)

            # Session A creates and commits a node
            session_a_node = facade.retain(
                content="Session A private committed work",
                source="test",
                session_id="session-A",
                importance=0.9,
                tags=["private-a"],
            )

            # Session B creates and commits a node
            session_b_node = facade.retain(
                content="Session B private committed work",
                source="test",
                session_id="session-B",
                importance=0.9,
                tags=["private-b"],
            )

            # Commit both
            executor = ReflectionExecutor(facade.store)
            executor.execute(session_id="session-A", force_commit=True)
            executor.execute(session_id="session-B", force_commit=True)

            # Recover session A
            result_a = facade.load_committed_state(session_id="session-A")

            committed_ids_a = [node["id"] for node in result_a["nodes"]]

            # Session A should see its own node
            assert str(session_a_node) in committed_ids_a, (
                f"Session A should see its own node {session_a_node}"
            )

            # Session A should NOT see Session B's node
            assert str(session_b_node) not in committed_ids_a, (
                f"SECURITY VIOLATION: Session A recovered Session B's node {session_b_node}! "
                f"Session isolation broken during recovery!"
            )

            # Recover session B
            result_b = facade.load_committed_state(session_id="session-B")

            committed_ids_b = [node["id"] for node in result_b["nodes"]]

            # Session B should see its own node
            assert str(session_b_node) in committed_ids_b, (
                f"Session B should see its own node {session_b_node}"
            )

            # Session B should NOT see Session A's node
            assert str(session_a_node) not in committed_ids_b, (
                f"SECURITY VIOLATION: Session B recovered Session A's node {session_a_node}! "
                f"Session isolation broken during recovery!"
            )

            facade.close()

    def test_recovery_includes_node_metadata(self):
        """P0: Recovered nodes MUST include required metadata fields.

        The recovery result must include node_type, title, category,
        importance, tags, etc. for proper reconstruction.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/test_recovery_metadata.db"
            facade = GraphMemoryFacade(db_path=db_path)

            # Create a node with known metadata
            node_id = facade.retain(
                content="Important analysis with full metadata",
                source="test",
                session_id="session-metadata",
                department="research",
                agent_id="research-agent-1",
                importance=0.85,
                tags=["important", "analysis", "eurusd"],
            )

            # Commit it
            executor = ReflectionExecutor(facade.store)
            executor.execute(session_id="session-metadata", force_commit=True)

            # Recover
            result = facade.load_committed_state(session_id="session-metadata")

            assert result["committed_nodes_count"] >= 1
            recovered_node = result["nodes"][0]

            # Verify metadata fields are present
            assert "id" in recovered_node, "Recovered node must have id"
            assert "node_type" in recovered_node, "Recovered node must have node_type"
            assert "title" in recovered_node, "Recovered node must have title"
            assert "category" in recovered_node, "Recovered node must have category"
            assert "importance" in recovered_node, "Recovered node must have importance"
            assert "tags" in recovered_node, "Recovered node must have tags"
            assert "created_at" in recovered_node, "Recovered node must have created_at"

            # Verify values
            assert recovered_node["id"] == str(node_id)
            assert recovered_node["importance"] == 0.85
            assert "important" in recovered_node["tags"]

            facade.close()

    def test_reflection_executor_recover_session_method(self):
        """P0: ReflectionExecutor MUST have recover_session method.

        The ReflectionExecutor class must implement the recover_session
        method as part of the session recovery flow.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/test_executor_recovery.db"
            facade = GraphMemoryFacade(db_path=db_path)

            executor = ReflectionExecutor(facade.store)

            # Verify the method exists and is callable
            assert hasattr(executor, "recover_session"), (
                "ReflectionExecutor must have recover_session method"
            )
            assert callable(executor.recover_session), (
                "recover_session must be callable"
            )

            # Create and commit a node
            node_id = facade.retain(
                content="Test content for executor recovery",
                source="test",
                session_id="session-executor",
                importance=0.8,
            )

            executor.execute(session_id="session-executor", force_commit=True)

            # Call recover_session
            result = executor.recover_session(session_id="session-executor")

            # Verify result structure
            assert "session_id" in result, "recover_session result must have session_id"
            assert result["session_id"] == "session-executor"
            assert "committed_nodes_count" in result, "Result must have committed_nodes_count"
            assert "committed_nodes" in result, "Result must have committed_nodes list"

            # Verify node was recovered
            assert result["committed_nodes_count"] >= 1, (
                f"Should have recovered at least 1 node, got {result['committed_nodes_count']}"
            )

            facade.close()

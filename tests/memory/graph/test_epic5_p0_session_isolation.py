"""P0 Tests: Graph Memory Session Isolation (Epic 5 - R-001)

These tests verify that draft nodes are invisible to other sessions
while committed nodes are visible to all sessions.

Test Design Reference: _bmad-output/test-artifacts/test-design-epic-5.md
Risk: R-001 (Score 9) - Graph memory session isolation failure
Priority: P0 - Critical path, no workaround
"""

import pytest
import tempfile
from datetime import datetime, timezone

from src.memory.graph.facade import GraphMemoryFacade
from src.memory.graph.types import (
    MemoryNode,
    MemoryNodeType,
    MemoryCategory,
    SessionStatus,
)


class TestSessionIsolationDraftNodes:
    """Test R-001: Graph memory session isolation for draft nodes."""

    def test_draft_nodes_invisible_to_other_session(self):
        """P0: Draft nodes from session A MUST NOT be visible to session B.

        This is a critical security/isolation requirement. Draft nodes contain
        in-progress work that should not be exposed to other sessions.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/test_session_isolation.db"
            facade = GraphMemoryFacade(db_path=db_path)

            # Create a draft node in session A (importance < 0.3 so it fails validation)
            session_a_node_id = facade.retain(
                content="This is secret draft work from session A",
                source="test",
                session_id="session-A",
                importance=0.1,
                tags=["draft", "secret"],
            )

            # Create a committed node in session A (for comparison)
            session_a_committed_id = facade.retain(
                content="This is committed work visible to all",
                source="test",
                session_id="session-A",
                importance=0.8,
                tags=["committed"],
            )

            # Commit the committed node
            facade.commit_session(session_id="session-A")

            # Query from session B - should NOT see draft node
            session_b_visible = facade.query_session_isolated(
                session_id="session-B",
                query="secret draft work",
                include_committed=True,
            )

            session_b_node_ids = [str(node.id) for node in session_b_visible]

            # CRITICAL ASSERTION: Draft node from session A MUST NOT be visible to session B
            assert session_a_node_id not in session_b_node_ids, (
                f"SECURITY VIOLATION: Draft node {session_a_node_id} from session-A "
                f"was visible to session-B! Draft nodes must be session-private."
            )

            # Verify committed node IS visible to session B
            assert session_a_committed_id in session_b_node_ids, (
                f"Committed node {session_a_committed_id} should be visible to session-B"
            )

            facade.close()

    def test_committed_nodes_visible_to_other_sessions(self):
        """P0: Committed nodes MUST be visible to all sessions.

        Once a node is committed, it becomes part of the shared knowledge
        base and must be accessible to all sessions.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/test_committed_visibility.db"
            facade = GraphMemoryFacade(db_path=db_path)

            # Create and commit a node in session A
            committed_node_id = facade.retain(
                content="Important committed knowledge about EURUSD",
                source="test",
                session_id="session-A",
                department="research",
                importance=0.9,
                tags=["committed", "important"],
            )

            # Commit the node
            result = facade.commit_session(session_id="session-A")
            assert result.get("status") != "error", f"Failed to commit: {result}"

            # Query from session B - MUST see the committed node
            session_b_results = facade.query_session_isolated(
                session_id="session-B",
                query="EURUSD",
                include_committed=True,
            )

            session_b_node_ids = [str(node.id) for node in session_b_results]

            # CRITICAL ASSERTION: Committed node MUST be visible to session B
            assert committed_node_id in session_b_node_ids, (
                f"Committed node {committed_node_id} was NOT visible to session-B! "
                f"Committed nodes must be globally visible."
            )

            facade.close()

    def test_draft_nodes_visible_within_same_session(self):
        """P0: Draft nodes MUST be visible within the same session that created them.

        The owning session should be able to see its own draft nodes for
        continued work and editing.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/test_same_session_visibility.db"
            facade = GraphMemoryFacade(db_path=db_path)

            # Create a draft node in session A
            draft_node_id = facade.retain(
                content="My in-progress draft work on GBPUSD strategy",
                source="test",
                session_id="session-A",
                importance=0.7,
                tags=["draft", "wip"],
            )

            # Query from the SAME session - MUST see the draft node
            same_session_results = facade.query_session_isolated(
                session_id="session-A",
                query="GBPUSD strategy",
                include_committed=True,
            )

            same_session_node_ids = [str(node.id) for node in same_session_results]

            # CRITICAL ASSERTION: Draft node MUST be visible to its own session
            assert draft_node_id in same_session_node_ids, (
                f"Draft node {draft_node_id} was NOT visible to its own session-A! "
                f"Draft nodes must be visible to the creating session."
            )

            facade.close()

    def test_query_session_isolated_excludes_draft_from_other_sessions(self):
        """P0: query_session_isolated MUST filter out draft nodes from other sessions.

        This is the core isolation enforcement method - it must properly
        implement the isolation boundary.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/test_query_isolated.db"
            facade = GraphMemoryFacade(db_path=db_path)

            # Session A creates draft node
            session_a_draft = facade.retain(
                content="Private session A research on NASDAQ",
                source="test",
                session_id="session-A",
                importance=0.8,
            )

            # Session B creates draft node
            session_b_draft = facade.retain(
                content="Private session B research on NYSE",
                source="test",
                session_id="session-B",
                importance=0.8,
            )

            # Session A queries - should see its own draft but NOT session B's
            session_a_visible = facade.query_session_isolated(
                session_id="session-A",
                include_committed=False,  # Only draft nodes
            )

            session_a_draft_ids = [str(node.id) for node in session_a_visible]

            # Session A should see its own draft
            assert session_a_draft in session_a_draft_ids, (
                "Session A should see its own draft nodes"
            )

            # Session A should NOT see Session B's draft
            assert session_b_draft not in session_a_draft_ids, (
                f"SECURITY VIOLATION: Session A could see Session B's draft {session_b_draft}! "
                f"Session isolation failed!"
            )

            facade.close()

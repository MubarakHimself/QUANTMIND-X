"""P0 Tests: ReflectionExecutor Draft-to-Commit Promotion (Epic 5 - R-008)

These tests verify that ReflectionExecutor correctly promotes valid
draft nodes to committed status while rejecting invalid ones.

Test Design Reference: _bmad-output/test-artifacts/test-design-epic-5.md
Risk: R-008 (Score 4) - ReflectionExecutor promotion logic bug
Priority: P0 - Critical path for memory persistence
"""

import pytest
import tempfile
import uuid

from src.memory.graph.facade import GraphMemoryFacade
from src.memory.graph.reflection_executor import ReflectionExecutor
from src.memory.graph.types import (
    MemoryNode,
    MemoryNodeType,
    MemoryCategory,
    SessionStatus,
    RelationType,
)


class TestReflectionExecutorDraftPromotion:
    """Test R-008: ReflectionExecutor draft-to-commit promotion."""

    def test_valid_draft_node_promoted_to_committed(self):
        """P0: Valid draft nodes MUST be promoted to committed status.

        A node is valid for promotion if it has:
        - Sufficient importance (>= 0.3)
        - Adequate content (>= 10 chars)
        - For OPINION nodes: action, reasoning, and SUPPORTED_BY edge
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/test_valid_promotion.db"
            facade = GraphMemoryFacade(db_path=db_path)

            # Create a valid draft node with sufficient content and importance
            valid_node_id = facade.retain(
                content="EURUSD showed strong momentum today with clear trend signals",
                source="test",
                session_id="session-test",
                importance=0.8,  # High importance
                department="research",
                tags=["analysis", "valid"],
            )

            # Verify it's in draft
            draft_nodes = facade.get_draft_nodes(session_id="session-test")
            draft_ids = [str(n.id) for n in draft_nodes]
            assert valid_node_id in draft_ids, "Node should start as draft"

            # Execute reflection to promote
            executor = ReflectionExecutor(facade.store)
            result = executor.execute(session_id="session-test", force_commit=False)

            # Check promotion
            committed_nodes = facade.get_committed_nodes(session_id="session-test")
            committed_ids = [str(n.id) for n in committed_nodes]

            # CRITICAL: Valid node MUST be promoted
            assert valid_node_id in committed_ids, (
                f"Valid draft node {valid_node_id} was NOT promoted to committed! "
                f"ReflectionExecutor is failing to promote valid memories. "
                f"Result: {result}"
            )

            facade.close()

    def test_invalid_draft_node_rejected_from_promotion(self):
        """P0: Invalid draft nodes MUST be rejected from promotion.

        A node is invalid if it has:
        - Low importance (< 0.3)
        - Insufficient content (< 10 chars)
        - Missing required fields for OPINION nodes
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/test_invalid_rejection.db"
            facade = GraphMemoryFacade(db_path=db_path)

            # Create a node with LOW importance (invalid)
            low_importance_id = facade.retain(
                content="Test",
                source="test",
                session_id="session-test",
                importance=0.1,  # Too low
                tags=["test"],
            )

            # Execute reflection
            executor = ReflectionExecutor(facade.store)
            result = executor.execute(session_id="session-test", force_commit=False)

            # Check that low importance node was NOT promoted
            committed_nodes = facade.get_committed_nodes(session_id="session-test")
            committed_ids = [str(n.id) for n in committed_nodes]

            assert low_importance_id not in committed_ids, (
                f"Invalid draft node {low_importance_id} with low importance WAS promoted! "
                f"ReflectionExecutor is promoting invalid nodes."
            )

            # Node should remain in draft
            draft_nodes = facade.get_draft_nodes(session_id="session-test")
            draft_ids = [str(n.id) for n in draft_nodes]
            assert low_importance_id in draft_ids, (
                "Invalid node should remain in draft"
            )

            facade.close()

    def test_opinion_node_without_action_rejected(self):
        """P0: OPINION node without action field MUST be rejected.

        Opinion nodes require the 'action' field to be valid for promotion.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/test_opinion_action.db"
            facade = GraphMemoryFacade(db_path=db_path)

            # Create OPINION node directly (to control all fields)
            opinion_no_action = MemoryNode(
                node_type=MemoryNodeType.OPINION,
                category=MemoryCategory.SUBJECTIVE,
                title="Missing action",
                content="I think we should consider different approaches",
                session_id="session-test",
                importance=0.8,
                reasoning="Market conditions suggest caution",  # Has reasoning
                # Missing action
                confidence=0.6,
                tags=["opinion"],
            )
            created = facade.store.create_node(opinion_no_action)

            # Execute reflection
            executor = ReflectionExecutor(facade.store)
            result = executor.execute(session_id="session-test", force_commit=False)

            # OPINION without action should NOT be promoted
            committed_nodes = facade.get_committed_nodes(session_id="session-test")
            committed_ids = [str(n.id) for n in committed_nodes]

            assert str(created.id) not in committed_ids, (
                f"OPINION node {created.id} without action WAS promoted! "
                f"OPINION nodes require action field."
            )

            facade.close()

    def test_opinion_node_without_reasoning_rejected(self):
        """P0: OPINION node without reasoning field MUST be rejected.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/test_opinion_reasoning.db"
            facade = GraphMemoryFacade(db_path=db_path)

            # Create OPINION without reasoning
            opinion_no_reasoning = MemoryNode(
                node_type=MemoryNodeType.OPINION,
                category=MemoryCategory.SUBJECTIVE,
                title="Missing reasoning",
                content="Close the position",
                session_id="session-test",
                importance=0.8,
                action="Close position",  # Has action
                # Missing reasoning
                confidence=0.6,
                tags=["opinion"],
            )
            created = facade.store.create_node(opinion_no_reasoning)

            # Execute reflection
            executor = ReflectionExecutor(facade.store)
            result = executor.execute(session_id="session-test", force_commit=False)

            # OPINION without reasoning should NOT be promoted
            committed_nodes = facade.get_committed_nodes(session_id="session-test")
            committed_ids = [str(n.id) for n in committed_nodes]

            assert str(created.id) not in committed_ids, (
                f"OPINION node {created.id} without reasoning WAS promoted! "
                f"OPINION nodes require reasoning field."
            )

            facade.close()

    def test_promotion_respects_force_commit_flag(self):
        """P0: force_commit=True MUST bypass validation and commit all nodes.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/test_force_commit.db"
            facade = GraphMemoryFacade(db_path=db_path)

            # Create invalid node (low importance)
            invalid_id = facade.retain(
                content="X",
                source="test",
                session_id="session-test",
                importance=0.1,  # Invalid
            )

            # Execute reflection WITHOUT force - should NOT promote
            executor = ReflectionExecutor(facade.store)
            result_normal = executor.execute(session_id="session-test", force_commit=False)

            committed_normal = facade.get_committed_nodes(session_id="session-test")
            assert invalid_id not in [str(n.id) for n in committed_normal], (
                "Invalid node should not be promoted normally"
            )

            # Execute reflection WITH force - SHOULD promote
            result_force = executor.execute(session_id="session-test", force_commit=True)

            committed_force = facade.get_committed_nodes(session_id="session-test")

            # With force=True, even invalid nodes should be promoted
            assert invalid_id in [str(n.id) for n in committed_force], (
                f"force_commit=True should have promoted invalid node {invalid_id}. "
                f"Force commit bypasses validation."
            )

            facade.close()

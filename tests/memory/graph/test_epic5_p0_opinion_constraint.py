"""P0 Tests: OPINION Node SUPPORTED_BY Edge Constraint (Epic 5 - R-002)

These tests verify that OPINION nodes require a SUPPORTED_BY edge
as a mandatory constraint - an OPINION without supporting evidence
should not be allowed to be promoted to committed status.

Test Design Reference: _bmad-output/test-artifacts/test-design-epic-5.md
Risk: R-002 (Score 9) - OPINION node orphaned from SUPPORTED_BY edge
Priority: P0 - Critical path, no workaround
"""

import pytest
import tempfile
import uuid
from datetime import datetime, timezone

from src.memory.graph.facade import GraphMemoryFacade
from src.memory.graph.types import (
    MemoryNode,
    MemoryNodeType,
    MemoryCategory,
    MemoryEdge,
    SessionStatus,
    RelationType,
)
from src.memory.graph.reflection_executor import ReflectionExecutor


class TestOpinionNodeSupportedByConstraint:
    """Test R-002: OPINION node mandatory SUPPORTED_BY edge constraint."""

    def _create_opinion_node(self, facade, content, session_id, action, reasoning,
                             confidence=0.8, supported_by_node_id=None):
        """Helper to create an OPINION node with required fields."""
        # Create the opinion node directly
        opinion = MemoryNode(
            node_type=MemoryNodeType.OPINION,
            category=MemoryCategory.SUBJECTIVE,
            title=content[:50] + "..." if len(content) > 50 else content,
            content=content,
            session_id=session_id,
            importance=0.8,
            action=action,
            reasoning=reasoning,
            confidence=confidence,
            tags=["opinion"],
        )
        created = facade.store.create_node(opinion)

        # If there's a supporting node, create the edge
        if supported_by_node_id:
            edge = MemoryEdge(
                relation_type=RelationType.SUPPORTED_BY,
                source_id=created.id,
                target_id=supported_by_node_id,
                strength=0.9,
            )
            facade.store.create_edge(edge)

        return created

    def _create_observation_node(self, facade, content, session_id):
        """Helper to create an OBSERVATION node."""
        node = MemoryNode(
            node_type=MemoryNodeType.OBSERVATION,
            category=MemoryCategory.EXPERIENTIAL,
            title=content[:50] + "..." if len(content) > 50 else content,
            content=content,
            session_id=session_id,
            importance=0.8,
            tags=["observation"],
        )
        return facade.store.create_node(node)

    def test_opinion_node_requires_supported_by_edge_for_promotion(self):
        """P0: OPINION node MUST have SUPPORTED_BY edge to be promoted to committed.

        An OPINION without supporting evidence (SUPPORTED_BY edge to an OBSERVATION
        or WORLD node) represents unverified opinion that should not be committed.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/test_opinion_constraint.db"
            facade = GraphMemoryFacade(db_path=db_path)

            # Create a supporting OBSERVATION node
            obs_node = self._create_observation_node(
                facade,
                "EURUSD has been in a downtrend for 5 consecutive days",
                "session-test"
            )

            # Create an OPINION node that references the observation
            opinion_node = self._create_opinion_node(
                facade,
                "We should close all EURUSD positions",
                "session-test",
                action="Close EURUSD positions",
                reasoning="EURUSD downtrend indicates weakening",
                confidence=0.85,
                supported_by_node_id=obs_node.id,
            )

            # Create ReflectionExecutor and attempt promotion
            executor = ReflectionExecutor(facade.store)

            # Execute reflection - this should promote the opinion since it has SUPPORTED_BY
            result = executor.execute(session_id="session-test", force_commit=False)

            # The opinion WITH SUPPORTED_BY edge should be promoted
            # Check if opinion was promoted (committed)
            committed_nodes = facade.get_committed_nodes(session_id="session-test")
            committed_ids = [str(node.id) for node in committed_nodes]

            assert str(opinion_node.id) in committed_ids, (
                f"OPINION node {opinion_node.id} with SUPPORTED_BY edge was NOT promoted! "
                f"ReflectionExecutor failed to promote valid opinion."
            )

            facade.close()

    def test_opinion_node_without_supported_by_edge_rejected(self):
        """P0: OPINION node WITHOUT SUPPORTED_BY edge MUST be rejected from promotion.

        An OPINION that lacks supporting evidence should remain in draft status
        and not be promoted to committed. This is the critical constraint
        that prevents unverified opinions from becoming "truth".
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/test_opinion_rejection.db"
            facade = GraphMemoryFacade(db_path=db_path)

            # Create an OPINION node WITHOUT any supporting evidence
            orphan_opinion = self._create_opinion_node(
                facade,
                "We should buy Apple stock",
                "session-test",
                action="Buy Apple stock",
                reasoning="Just because I feel like it",  # Not backed by evidence
                confidence=0.5,  # Low confidence
            )

            # Verify the opinion was created as draft
            draft_nodes = facade.get_draft_nodes(session_id="session-test")
            draft_ids = [str(node.id) for node in draft_nodes]
            assert str(orphan_opinion.id) in draft_ids, "Opinion should start as draft"

            # Create ReflectionExecutor and attempt promotion
            executor = ReflectionExecutor(facade.store)

            # Execute reflection WITHOUT force - should NOT promote orphan opinion
            result = executor.execute(session_id="session-test", force_commit=False)

            # Check the promotion result
            committed_count = result.get("committed_count", 0)

            # The orphan opinion should NOT have been promoted
            committed_nodes = facade.get_committed_nodes(session_id="session-test")
            committed_ids = [str(node.id) for node in committed_nodes]

            assert str(orphan_opinion.id) not in committed_ids, (
                f"SECURITY VIOLATION: Orphan OPINION node {orphan_opinion.id} "
                f"without SUPPORTED_BY edge WAS promoted to committed! "
                f"This violates the mandatory constraint - opinions require evidence!"
            )

            # Verify it's still in draft
            draft_nodes_after = facade.get_draft_nodes(session_id="session-test")
            draft_ids_after = [str(node.id) for node in draft_nodes_after]
            assert str(orphan_opinion.id) in draft_ids_after, (
                "Orphan opinion should remain in draft status"
            )

            facade.close()

    def test_opinion_supported_by_edge_validation(self):
        """P0: ReflectionExecutor._has_supported_by_edges MUST correctly validate.

        The internal validation method must properly check for SUPPORTED_BY edges
        on OPINION nodes before allowing promotion.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/test_edge_validation.db"
            facade = GraphMemoryFacade(db_path=db_path)

            # Create OBSERVATION node
            obs_node = self._create_observation_node(
                facade,
                "VIX spiked 20% today",
                "session-test"
            )

            # Create OPINION without edge
            opinion_no_edge = self._create_opinion_node(
                facade,
                "Market will crash tomorrow",
                "session-test",
                action="Exit all positions",
                reasoning="I predict a crash",
                confidence=0.6,
            )

            # Create OPINION with edge
            opinion_with_edge = self._create_opinion_node(
                facade,
                "Reduce exposure due to VIX",
                "session-test",
                action="Reduce exposure",
                reasoning="VIX spike indicates fear",
                confidence=0.8,
                supported_by_node_id=obs_node.id,
            )

            # Create executor and test edge validation
            executor = ReflectionExecutor(facade.store)

            # Test validation for opinion without edge
            has_edge_no_edge = executor._has_supported_by_edges(str(opinion_no_edge.id))
            assert has_edge_no_edge is False, (
                f"Opinion {opinion_no_edge.id} incorrectly reported as having SUPPORTED_BY edge"
            )

            # Test validation for opinion with edge
            has_edge_with_edge = executor._has_supported_by_edges(str(opinion_with_edge.id))
            assert has_edge_with_edge is True, (
                f"Opinion {opinion_with_edge.id} incorrectly reported as NOT having SUPPORTED_BY edge"
            )

            facade.close()

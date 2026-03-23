"""P1 Tests: Opinion node lifecycle edge cases."""

import pytest
import tempfile
from src.memory.graph.facade import GraphMemoryFacade
from src.memory.graph.reflection_executor import ReflectionExecutor
from src.memory.graph.types import (
    MemoryNode,
    MemoryNodeType,
    MemoryCategory,
    MemoryEdge,
    RelationType,
)


class TestOpinionNodeEdgeCases:
    """P1: Test OPINION node edge cases beyond basic constraint."""

    def _create_opinion_with_support(self, facade, content, reasoning, action,
                                     support_content, confidence=0.8):
        """Helper: Create opinion with supporting observation."""
        # Create supporting observation
        obs = MemoryNode(
            node_type=MemoryNodeType.OBSERVATION,
            category=MemoryCategory.EXPERIENTIAL,
            title=support_content[:50],
            content=support_content,
            session_id="test-session",
            importance=0.8,
            tags=["observation"],
        )
        obs_created = facade.store.create_node(obs)

        # Create opinion with SUPPORTED_BY edge
        opinion = MemoryNode(
            node_type=MemoryNodeType.OPINION,
            category=MemoryCategory.SUBJECTIVE,
            title=content[:50],
            content=content,
            session_id="test-session",
            importance=0.8,
            reasoning=reasoning,
            action=action,
            confidence=confidence,
            tags=["opinion"],
        )
        opinion_created = facade.store.create_node(opinion)

        edge = MemoryEdge(
            relation_type=RelationType.SUPPORTED_BY,
            source_id=opinion_created.id,
            target_id=obs_created.id,
            strength=0.9,
        )
        facade.store.create_edge(edge)

        return opinion_created

    def test_opinion_with_multiple_supported_by_edges(self):
        """[P1] OPINION with multiple SUPPORTED_BY edges should be promoted if all valid."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/test_multi_support.db"
            facade = GraphMemoryFacade(db_path=db_path)

            # Create two observations
            obs1 = MemoryNode(
                node_type=MemoryNodeType.OBSERVATION,
                category=MemoryCategory.EXPERIENTIAL,
                title="Obs1",
                content="EURUSD showing bearish divergence",
                session_id="test-session",
                importance=0.8,
            )
            obs1_created = facade.store.create_node(obs1)

            obs2 = MemoryNode(
                node_type=MemoryNodeType.OBSERVATION,
                category=MemoryCategory.EXPERIENTIAL,
                title="Obs2",
                content="Volume declining on down moves",
                session_id="test-session",
                importance=0.7,
            )
            obs2_created = facade.store.create_node(obs2)

            # Create opinion with two SUPPORTED_BY edges
            opinion = MemoryNode(
                node_type=MemoryNodeType.OPINION,
                category=MemoryCategory.SUBJECTIVE,
                title="Close positions",
                content="Close all EURUSD short positions",
                session_id="test-session",
                importance=0.8,
                reasoning="Bearish divergence + declining volume = reversal risk",
                action="Close short positions",
                confidence=0.85,
            )
            opinion_created = facade.store.create_node(opinion)

            # Add two SUPPORTED_BY edges
            for obs_id in [obs1_created.id, obs2_created.id]:
                edge = MemoryEdge(
                    relation_type=RelationType.SUPPORTED_BY,
                    source_id=opinion_created.id,
                    target_id=obs_id,
                    strength=0.8,
                )
                facade.store.create_edge(edge)

            # Execute reflection
            executor = ReflectionExecutor(facade.store)
            result = executor.execute(session_id="test-session", force_commit=False)

            # Opinion with multiple valid edges should be promoted
            committed = facade.get_committed_nodes(session_id="test-session")
            committed_ids = [str(n.id) for n in committed]

            assert str(opinion_created.id) in committed_ids, (
                f"OPINION with multiple SUPPORTED_BY edges was NOT promoted"
            )

            facade.close()

    def test_opinion_confidence_below_minimum_rejected(self):
        """[P1] OPINION with confidence < 0.3 should be rejected regardless of edges."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/test_low_confidence.db"
            facade = GraphMemoryFacade(db_path=db_path)

            # Create opinion with very low confidence
            opinion = self._create_opinion_with_support(
                facade,
                content="Market will crash tomorrow",
                reasoning="Just a feeling",
                action="Exit all trades",
                support_content="VIX up 5% today",
                confidence=0.1,  # Very low confidence
            )

            executor = ReflectionExecutor(facade.store)
            result = executor.execute(session_id="test-session", force_commit=False)

            committed = facade.get_committed_nodes(session_id="test-session")
            committed_ids = [str(n.id) for n in committed]

            assert str(opinion.id) not in committed_ids, (
                "OPINION with confidence < 0.3 was promoted despite low confidence"
            )

            facade.close()

    def test_opinion_missing_reasoning_rejected(self):
        """[P1] OPINION without reasoning field should be rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/test_missing_reasoning.db"
            facade = GraphMemoryFacade(db_path=db_path)

            # Create opinion without reasoning
            opinion = MemoryNode(
                node_type=MemoryNodeType.OPINION,
                category=MemoryCategory.SUBJECTIVE,
                title="Trade now",
                content="Buy EURUSD now",
                session_id="test-session",
                importance=0.8,
                action="Buy EURUSD",  # Has action
                # Missing reasoning
                confidence=0.8,
                tags=["opinion"],
            )
            opinion_created = facade.store.create_node(opinion)

            executor = ReflectionExecutor(facade.store)
            result = executor.execute(session_id="test-session", force_commit=False)

            committed = facade.get_committed_nodes(session_id="test-session")
            committed_ids = [str(n.id) for n in committed]

            assert str(opinion_created.id) not in committed_ids, (
                "OPINION without reasoning was promoted"
            )

            facade.close()

    def test_opinion_missing_action_rejected(self):
        """[P1] OPINION without action field should be rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/test_missing_action.db"
            facade = GraphMemoryFacade(db_path=db_path)

            # Create opinion without action
            opinion = MemoryNode(
                node_type=MemoryNodeType.OPINION,
                category=MemoryCategory.SUBJECTIVE,
                title="Analysis",
                content="EURUSD looks weak",
                session_id="test-session",
                importance=0.8,
                reasoning="Bearish signals everywhere",
                # Missing action
                confidence=0.8,
                tags=["opinion"],
            )
            opinion_created = facade.store.create_node(opinion)

            executor = ReflectionExecutor(facade.store)
            result = executor.execute(session_id="test-session", force_commit=False)

            committed = facade.get_committed_nodes(session_id="test-session")
            committed_ids = [str(n.id) for n in committed]

            assert str(opinion_created.id) not in committed_ids, (
                "OPINION without action was promoted"
            )

            facade.close()

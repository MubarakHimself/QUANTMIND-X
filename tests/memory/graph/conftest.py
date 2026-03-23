"""Graph memory test fixtures for Epic 5 P1/P2/P3 tests."""

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
    SessionStatus,
)


@pytest.fixture
def graph_memory_facade():
    """Factory fixture for GraphMemoryFacade with temp DB."""
    _facades = []

    def _create_facade():
        tmpdir = tempfile.mkdtemp()
        db_path = f"{tmpdir}/test.db"
        facade = GraphMemoryFacade(db_path=db_path)
        _facades.append(facade)
        return facade

    yield _create_facade

    # Cleanup
    for facade in _facades:
        try:
            facade.close()
        except Exception:
            pass


@pytest.fixture
def reflection_executor(graph_memory_facade):
    """Factory fixture for ReflectionExecutor."""
    facade = graph_memory_facade()
    return ReflectionExecutor(facade.store)


@pytest.fixture
def sample_observation_node(graph_memory_facade):
    """Factory fixture: creates a sample OBSERVATION node."""
    def _create(content="Test observation content", session_id="test-session"):
        facade = graph_memory_facade()
        node = MemoryNode(
            node_type=MemoryNodeType.OBSERVATION,
            category=MemoryCategory.EXPERIENTIAL,
            title=content[:50] if len(content) > 50 else content,
            content=content,
            session_id=session_id,
            importance=0.8,
            tags=["observation", "test"],
        )
        return facade.store.create_node(node)
    return _create


@pytest.fixture
def sample_opinion_node(graph_memory_facade):
    """Factory fixture: creates a sample OPINION node with SUPPORTED_BY edge."""
    def _create(
        content="Test opinion content",
        reasoning="Test reasoning",
        action="Test action",
        confidence=0.8,
        support_node=None,
        session_id="test-session",
    ):
        facade = graph_memory_facade()

        # Create support node if not provided
        if support_node is None:
            support_node = MemoryNode(
                node_type=MemoryNodeType.OBSERVATION,
                category=MemoryCategory.EXPERIENTIAL,
                title="Support observation",
                content="Supporting observation for opinion",
                session_id=session_id,
                importance=0.8,
            )
            support_node = facade.store.create_node(support_node)

        # Create opinion
        opinion = MemoryNode(
            node_type=MemoryNodeType.OPINION,
            category=MemoryCategory.SUBJECTIVE,
            title=content[:50] if len(content) > 50 else content,
            content=content,
            session_id=session_id,
            importance=0.8,
            reasoning=reasoning,
            action=action,
            confidence=confidence,
            tags=["opinion", "test"],
        )
        opinion_created = facade.store.create_node(opinion)

        # Create SUPPORTED_BY edge
        edge = MemoryEdge(
            relation_type=RelationType.SUPPORTED_BY,
            source_id=opinion_created.id,
            target_id=support_node.id,
            strength=0.9,
        )
        facade.store.create_edge(edge)

        return opinion_created
    return _create


@pytest.fixture
def session_checkpoint_service():
    """Factory fixture for SessionCheckpointService."""
    from src.agents.memory.session_checkpoint_service import SessionCheckpointService

    def _create(interval_minutes=5, stale_threshold_hours=24, milestone_enabled=True):
        return SessionCheckpointService(
            checkpoint_interval_minutes=interval_minutes,
            stale_draft_threshold_hours=stale_threshold_hours,
            checkpoint_on_milestone=milestone_enabled,
        )
    return _create

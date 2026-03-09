"""Tests for graph memory types."""
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from src.memory.graph.types import (
    MemoryCategory,
    MemoryEdge,
    MemoryNode,
    MemoryNodeType,
    MemoryTier,
    RelationType,
)


class TestMemoryNodeType:
    """Tests for MemoryNodeType enum."""

    def test_hindsight_network_types(self):
        """Test World, Bank, Observation, Opinion from Hindsight."""
        assert MemoryNodeType.WORLD is not None
        assert MemoryNodeType.BANK is not None
        assert MemoryNodeType.OBSERVATION is not None
        assert MemoryNodeType.OPINION is not None

    def test_coala_omem_types(self):
        """Test Working, Persona, Procedural, Episodic from CoALA/O-Mem."""
        assert MemoryNodeType.WORKING is not None
        assert MemoryNodeType.PERSONA is not None
        assert MemoryNodeType.PROCEDURAL is not None
        assert MemoryNodeType.EPISODIC is not None

    def test_additional_types(self):
        """Test Conversation, Message, Agent, Department, Task, Session, Decision."""
        assert MemoryNodeType.CONVERSATION is not None
        assert MemoryNodeType.MESSAGE is not None
        assert MemoryNodeType.AGENT is not None
        assert MemoryNodeType.DEPARTMENT is not None
        assert MemoryNodeType.TASK is not None
        assert MemoryNodeType.SESSION is not None
        assert MemoryNodeType.DECISION is not None


class TestMemoryCategory:
    """Tests for MemoryCategory enum."""

    def test_factual_category(self):
        """Test FACTUAL category from PREMem."""
        assert MemoryCategory.FACTUAL is not None

    def test_experiential_category(self):
        """Test EXPERIENTIAL category from PREMem."""
        assert MemoryCategory.EXPERIENTIAL is not None

    def test_subjective_category(self):
        """Test SUBJECTIVE category from PREMem."""
        assert MemoryCategory.SUBJECTIVE is not None


class TestMemoryTier:
    """Tests for MemoryTier enum."""

    def test_hot_tier(self):
        """Test HOT tier (< 1 hour)."""
        assert MemoryTier.HOT is not None

    def test_warm_tier(self):
        """Test WARM tier (< 30 days)."""
        assert MemoryTier.WARM is not None

    def test_cold_tier(self):
        """Test COLD tier (archived)."""
        assert MemoryTier.COLD is not None


class TestRelationType:
    """Tests for RelationType class with constants."""

    def test_structural_relations(self):
        """Test PART_OF, CONTAINS, RELATED_TO, SIMILAR_TO, DERIVED_FROM."""
        assert RelationType.PART_OF is not None
        assert RelationType.CONTAINS is not None
        assert RelationType.RELATED_TO is not None
        assert RelationType.SIMILAR_TO is not None
        assert RelationType.DERIVED_FROM is not None

    def test_causal_relations(self):
        """Test CAUSED_BY, ENABLES, LEADS_TO."""
        assert RelationType.CAUSED_BY is not None
        assert RelationType.ENABLES is not None
        assert RelationType.LEADS_TO is not None

    def test_temporal_relations(self):
        """Test FOLLOWS, BEFORE, CONCURRENT_WITH."""
        assert RelationType.FOLLOWS is not None
        assert RelationType.BEFORE is not None
        assert RelationType.CONCURRENT_WITH is not None

    def test_agency_relations(self):
        """Test DECIDED_BY, EXECUTED_BY, CREATED_BY, MENTIONED_IN."""
        assert RelationType.DECIDED_BY is not None
        assert RelationType.EXECUTED_BY is not None
        assert RelationType.CREATED_BY is not None
        assert RelationType.MENTIONED_IN is not None

    def test_semantic_relations(self):
        """Test RECALLS, REMINDS_OF, EXTENDS, TRANSFORMS, IMPLIES."""
        assert RelationType.RECALLS is not None
        assert RelationType.REMINDS_OF is not None
        assert RelationType.EXTENDS is not None
        assert RelationType.TRANSFORMS is not None
        assert RelationType.IMPLIES is not None


class TestMemoryNode:
    """Tests for MemoryNode dataclass."""

    def test_memory_node_creation(self):
        """Test creating a memory node with all fields."""
        node_id = uuid.uuid4()
        now = datetime.now(timezone.utc)
        expires = now + timedelta(hours=2)

        node = MemoryNode(
            id=node_id,
            node_type=MemoryNodeType.OBSERVATION,
            category=MemoryCategory.EXPERIENTIAL,
            title="Test Observation",
            content="This is test content for the observation.",
            summary="Test summary",
            evidence="Evidence string",
            department="trading",
            agent_id="agent-001",
            session_id="session-123",
            role="analyst",
            tags=["test", "observation", "trading"],
            importance=0.8,
            relevance_score=0.75,
            access_count=5,
            last_accessed=now,
            created_by="system",
            created_at=now,
            updated_at=now,
            expires_at=expires,
            event_time=now,
            tier=MemoryTier.HOT,
            compaction_level=0,
            original_id=None,
            entity_id=None,
        )

        assert node.id == node_id
        assert node.node_type == MemoryNodeType.OBSERVATION
        assert node.category == MemoryCategory.EXPERIENTIAL
        assert node.title == "Test Observation"
        assert node.content == "This is test content for the observation."
        assert node.summary == "Test summary"
        assert node.evidence == "Evidence string"
        assert node.department == "trading"
        assert node.agent_id == "agent-001"
        assert node.session_id == "session-123"
        assert node.role == "analyst"
        assert node.tags == ["test", "observation", "trading"]
        assert node.importance == 0.8
        assert node.relevance_score == 0.75
        assert node.access_count == 5
        assert node.last_accessed == now
        assert node.created_by == "system"
        assert node.created_at == now
        assert node.updated_at == now
        assert node.expires_at == expires
        assert node.event_time == now
        assert node.tier == MemoryTier.HOT
        assert node.compaction_level == 0
        assert node.original_id is None
        assert node.entity_id is None

    def test_memory_node_default_values(self):
        """Test that MemoryNode has correct default values."""
        node = MemoryNode(
            node_type=MemoryNodeType.WORKING,
            category=MemoryCategory.FACTUAL,
            title="Minimal Node",
            content="Content only",
        )

        assert node.id is not None
        assert isinstance(node.id, uuid.UUID)
        assert node.summary == ""
        assert node.evidence == ""
        assert node.department is None
        assert node.agent_id is None
        assert node.session_id is None
        assert node.role is None
        assert node.tags == []
        assert node.importance == 0.0
        assert node.relevance_score == 0.0
        assert node.access_count == 0
        assert node.last_accessed is None
        assert node.created_by is None
        assert node.created_at is not None
        assert node.updated_at is not None
        assert node.expires_at is None
        assert node.event_time is None
        assert node.tier == MemoryTier.HOT
        assert node.compaction_level == 0
        assert node.original_id is None
        assert node.entity_id is None

    def test_memory_node_compaction_tracking(self):
        """Test tracking original_id and compaction_level for compacted versions."""
        original_id = uuid.uuid4()
        now = datetime.now(timezone.utc)

        # Original node (level 0 - raw)
        original_node = MemoryNode(
            id=original_id,
            node_type=MemoryNodeType.CONVERSATION,
            category=MemoryCategory.EXPERIENTIAL,
            title="Original Conversation",
            content="Full conversation content here.",
            created_at=now,
            tier=MemoryTier.HOT,
            compaction_level=0,
        )

        # First compaction - summarized (level 1)
        summarized_id = uuid.uuid4()
        summarized_node = MemoryNode(
            id=summarized_id,
            node_type=MemoryNodeType.CONVERSATION,
            category=MemoryCategory.EXPERIENTIAL,
            title="Original Conversation",
            content="Summarized version.",
            original_id=original_id,
            created_at=now,
            tier=MemoryTier.WARM,
            compaction_level=1,
        )

        # Second compaction - compressed (level 2)
        compressed_id = uuid.uuid4()
        compressed_node = MemoryNode(
            id=compressed_id,
            node_type=MemoryNodeType.CONVERSATION,
            category=MemoryCategory.EXPERIENTIAL,
            title="Original Conversation",
            content="Compressed.",
            original_id=original_id,
            created_at=now,
            tier=MemoryTier.COLD,
            compaction_level=2,
        )

        # Verify original node
        assert original_node.compaction_level == 0
        assert original_node.original_id is None

        # Verify summarized node
        assert summarized_node.compaction_level == 1
        assert summarized_node.original_id == original_id

        # Verify compressed node
        assert compressed_node.compaction_level == 2
        assert compressed_node.original_id == original_id


class TestMemoryEdge:
    """Tests for MemoryEdge dataclass."""

    def test_memory_edge_creation(self):
        """Test creating a memory edge with relation types."""
        edge_id = uuid.uuid4()
        source_id = uuid.uuid4()
        target_id = uuid.uuid4()
        now = datetime.now(timezone.utc)

        edge = MemoryEdge(
            id=edge_id,
            relation_type=RelationType.RELATED_TO,
            source_id=source_id,
            target_id=target_id,
            strength=0.9,
            bidirectional=True,
            pattern="extension",
            context="These observations are related through shared context.",
            created_at=now,
            traversal_count=10,
            last_traversed=now,
        )

        assert edge.id == edge_id
        assert edge.relation_type == RelationType.RELATED_TO
        assert edge.source_id == source_id
        assert edge.target_id == target_id
        assert edge.strength == 0.9
        assert edge.bidirectional is True
        assert edge.pattern == "extension"
        assert edge.context == "These observations are related through shared context."
        assert edge.created_at == now
        assert edge.traversal_count == 10
        assert edge.last_traversed == now

    def test_memory_edge_default_values(self):
        """Test MemoryEdge has correct default values."""
        source_id = uuid.uuid4()
        target_id = uuid.uuid4()

        edge = MemoryEdge(
            relation_type=RelationType.CAUSED_BY,
            source_id=source_id,
            target_id=target_id,
        )

        assert edge.id is not None
        assert isinstance(edge.id, uuid.UUID)
        assert edge.strength == 1.0
        assert edge.bidirectional is False
        assert edge.pattern is None
        assert edge.context is None
        assert edge.created_at is not None
        assert edge.traversal_count == 0
        assert edge.last_traversed is None

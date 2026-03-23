"""Graph-based memory types for QuantMindX memory system.

This module defines the core types for the graph-based memory system:
- MemoryNodeType: Types based on Hindsight's Four Networks and CoALA/O-Mem
- MemoryCategory: PREMem categories (Factual, Experiential, Subjective)
- MemoryTier: Storage tiers (Hot, Warm, Cold)
- RelationType: Relationship types between memory nodes
- MemoryNode: Core memory entity
- MemoryEdge: Relationships between memory nodes
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
import uuid


class MemoryNodeType(Enum):
    """Types of memory nodes based on research.

    Hindsight's Four Networks:
    - WORLD: World model / knowledge base
    - BANK: Knowledge bank / semantic memory
    - OBSERVATION: Observations / episodic memory
    - OPINION: Opinions / preferences

    CoALA/O-Mem:
    - WORKING: Working memory (active processing)
    - PERSONA: Self-model / agent identity
    - PROCEDURAL: Procedural memory / skills
    - EPISODIC: Episodic memory / experiences

    Additional types:
    - CONVERSATION: Conversations / dialogues
    - MESSAGE: Individual messages
    - AGENT: Agent entities
    - DEPARTMENT: Department entities
    - TASK: Task entities
    - SESSION: Session entities
    - DECISION: Decision entities
    """

    # Hindsight's Four Networks
    WORLD = "world"
    BANK = "bank"
    OBSERVATION = "observation"
    OPINION = "opinion"

    # CoALA/O-Mem
    WORKING = "working"
    PERSONA = "persona"
    PROCEDURAL = "procedural"
    EPISODIC = "episodic"

    # Additional types
    CONVERSATION = "conversation"
    MESSAGE = "message"
    AGENT = "agent"
    DEPARTMENT = "department"
    TASK = "task"
    SESSION = "session"
    DECISION = "decision"


class MemoryCategory(Enum):
    """Memory categories based on PREMem research.

    - FACTUAL: Factual knowledge (world knowledge, semantic facts)
    - EXPERIENTIAL: Experiential memory (events, experiences)
    - SUBJECTIVE: Subjective memory (opinions, beliefs, preferences)
    """

    FACTUAL = "factual"
    EXPERIENTIAL = "experiential"
    SUBJECTIVE = "subjective"


class MemoryTier(Enum):
    """Storage tiers for memory based on access patterns.

    - HOT: Recent memory (< 1 hour), high-speed access
    - WARM: Recent memory (< 30 days), moderate access speed
    - COLD: Archived memory, long-term storage
    """

    HOT = "hot"
    WARM = "warm"
    COLD = "cold"


class RelationType:
    """Relationship types between memory nodes.

    Structural relations:
    - PART_OF: Node is part of another node
    - CONTAINS: Node contains another node
    - RELATED_TO: Nodes are generally related
    - SIMILAR_TO: Nodes are similar in content
    - DERIVED_FROM: Node was derived from another

    Causal relations:
    - CAUSED_BY: Node was caused by another
    - ENABLES: Node enables another
    - LEADS_TO: Node leads to another

    Temporal relations:
    - FOLLOWS: Node follows another in time
    - BEFORE: Node is before another in time
    - CONCURRENT_WITH: Node occurred concurrently with another

    Agency relations:
    - DECIDED_BY: Decision was made by an agent
    - EXECUTED_BY: Task was executed by an agent
    - CREATED_BY: Node was created by an agent
    - MENTIONED_IN: Node was mentioned in another

    Semantic relations (from PREMem):
    - RECALLS: Node recalls another memory
    - REMINDS_OF: Node reminds of another
    - EXTENDS: Node extends another
    - TRANSFORMS: Node transforms another
    - IMPLIES: Node implies another
    """

    # Structural
    PART_OF = "part_of"
    CONTAINS = "contains"
    RELATED_TO = "related_to"
    SIMILAR_TO = "similar_to"
    DERIVED_FROM = "derived_from"

    # Causal
    CAUSED_BY = "caused_by"
    ENABLES = "enables"
    LEADS_TO = "leads_to"

    # Temporal
    FOLLOWS = "follows"
    BEFORE = "before"
    CONCURRENT_WITH = "concurrent_with"

    # Agency
    DECIDED_BY = "decided_by"
    EXECUTED_BY = "executed_by"
    CREATED_BY = "created_by"
    MENTIONED_IN = "mentioned_in"

    # Semantic
    RECALLS = "recalls"
    REMINDS_OF = "reminds_of"
    EXTENDS = "extends"
    TRANSFORMS = "transforms"
    IMPLIES = "implies"

    # OPINION-specific
    SUPPORTED_BY = "supported_by"


class SessionStatus:
    """Session status for memory nodes (draft vs committed).

    - DRAFT: Memory is being written, invisible to other sessions
    - COMMITTED: Memory is finalized and visible to downstream agents
    """

    DRAFT = "draft"
    COMMITTED = "committed"


@dataclass
class MemoryNode:
    """Core memory entity in the graph-based memory system.

    Attributes:
        id: Unique identifier (auto-generated UUID)
        node_type: Type of memory node
        category: Memory category (factual, experiential, subjective)
        title: Short title for the memory
        content: Full content of the memory
        summary: Brief summary (used in compaction)
        evidence: Evidence supporting the memory
        department: Department associated with this memory
        agent_id: Agent that created/owns this memory
        session_id: Session where this memory was created
        role: Role associated with this memory
        tags: List of tags for categorization
        importance: Importance score (0-1)
        relevance_score: Relevance score (0-1)
        access_count: Number of times accessed
        last_accessed_utc: Last access timestamp (UTC)
        created_by: Creator identifier
        created_at_utc: Creation timestamp (UTC)
        updated_at_utc: Last update timestamp (UTC)
        expires_at_utc: Expiration timestamp (UTC)
        event_time_utc: Event timestamp (when the memory occurred, UTC)
        tier: Storage tier (hot, warm, cold)
        compaction_level: Compaction level (0=raw, 1=summarized, 2=compressed)
        original_id: ID of original node (for compacted versions)
        entity_id: Optional entity identifier for entity-based grouping
        session_status: Session status (draft | committed)
        embedding: Vector embedding for semantic search (bytes)
        # OPINION-specific fields
        action: What the agent did (for OPINION nodes)
        reasoning: Why they did it (for OPINION nodes)
        confidence: Confidence score 0.0-1.0 (for OPINION nodes)
        alternatives_considered: Options evaluated (for OPINION nodes)
        constraints_applied: Constraints that influenced the decision (for OPINION nodes)
        agent_role: Which agent/role took the action (for OPINION nodes)
    """

    node_type: MemoryNodeType
    category: MemoryCategory
    title: str
    content: str
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    summary: str = ""
    evidence: str = ""
    department: Optional[str] = None
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    role: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    importance: float = 0.0
    relevance_score: float = 0.0
    access_count: int = 0
    last_accessed_utc: Optional[datetime] = None
    created_by: Optional[str] = None
    created_at_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at_utc: Optional[datetime] = None
    event_time_utc: Optional[datetime] = None
    tier: MemoryTier = MemoryTier.HOT
    compaction_level: int = 0
    original_id: Optional[uuid.UUID] = None
    entity_id: Optional[str] = None
    session_status: str = SessionStatus.DRAFT
    embedding: Optional[bytes] = None
    # OPINION-specific fields
    action: Optional[str] = None
    reasoning: Optional[str] = None
    confidence: Optional[float] = None
    alternatives_considered: Optional[str] = None
    constraints_applied: Optional[str] = None
    agent_role: Optional[str] = None


@dataclass
class MemoryEdge:
    """Relationship between memory nodes in the graph.

    Attributes:
        id: Unique identifier (auto-generated UUID)
        relation_type: Type of relationship
        source_id: UUID of source node
        target_id: UUID of target node
        strength: Relationship strength (0-1)
        bidirectional: Whether relationship is bidirectional
        pattern: Extension, transformation, or implication (from PREMem)
        context: Why the relationship exists
        created_at_utc: Creation timestamp (UTC)
        traversal_count: Number of times traversed
        last_traversed_utc: Last traversal timestamp (UTC)
    """

    relation_type: str
    source_id: uuid.UUID
    target_id: uuid.UUID
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    strength: float = 1.0
    bidirectional: bool = False
    pattern: Optional[str] = None
    context: Optional[str] = None
    created_at_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    traversal_count: int = 0
    last_traversed_utc: Optional[datetime] = None

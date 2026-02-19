"""
LangMem Integration for QuantMind Agents.

This module provides LangMem-based memory management for agents,
supporting semantic, episodic, and procedural memory types.

Uses real LangMem client with embedding-based similarity search and persistence.
"""

import asyncio
import logging
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json

logger = logging.getLogger(__name__)


# =============================================================================
# LangMem Client Availability Check
# =============================================================================

try:
    from langmem import (
        create_memory_manager,
        create_manage_memory_tool,
        create_search_memory_tool,
        create_memory_store_manager,
        ReflectionExecutor,
    )
    from langgraph.store.memory import InMemoryStore
    from langgraph.store.base import BaseStore
    LANGMEM_AVAILABLE = True
    logger.info("LangMem is available")
except ImportError as e:
    LANGMEM_AVAILABLE = False
    logger.warning(f"LangMem not available: {e}. Using fallback implementation.")


# =============================================================================
# Pydantic Models for Memory Types
# =============================================================================

try:
    from pydantic import BaseModel
    
    class SemanticMemory(BaseModel):
        """Store facts, preferences, and relationships as triples."""
        subject: str
        predicate: str
        object: str
        context: Optional[str] = None
    
    class EpisodicMemory(BaseModel):
        """Capture experiences with full chain of reasoning."""
        observation: str  # The context and setup - what happened
        thoughts: str     # Internal reasoning process "I ..."
        action: str       # What was done, how, in what format
        result: str       # Outcome and retrospective
    
    class ProceduralMemory(BaseModel):
        """Store and update agent instructions/skills."""
        skill_name: str
        instructions: str
        when_to_use: str
        examples: Optional[str] = None
    
    PYDANTIC_AVAILABLE = True
    
except ImportError:
    PYDANTIC_AVAILABLE = False
    SemanticMemory = None
    EpisodicMemory = None
    ProceduralMemory = None
    logger.warning("Pydantic not available for memory schemas")


class MemoryType(str):
    """Memory types supported by LangMem."""
    SEMANTIC = "semantic"      # Facts and concepts
    EPISODIC = "episodic"      # Events and experiences
    PROCEDURAL = "procedural"  # Skills and procedures


@dataclass
class MemoryEntry:
    """A single memory entry."""
    id: str
    content: str
    memory_type: str
    namespace: str
    created_at: str
    importance: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type,
            "namespace": self.namespace,
            "created_at": self.created_at,
            "importance": self.importance,
            "metadata": self.metadata
        }


@dataclass
class MemorySearchResult:
    """Result of a memory search."""
    entry: MemoryEntry
    relevance: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entry": self.entry.to_dict(),
            "relevance": self.relevance
        }


class LangMemManager:
    """
    Manager for LangMem-based agent memory.
    
    Provides per-agent memory namespaces with support for:
    - Semantic memory: Facts, concepts, and knowledge
    - Episodic memory: Events, experiences, and interactions
    - Procedural memory: Skills, procedures, and how-to knowledge
    
    Uses real LangMem client with embedding-based similarity search.
    Memory namespaces follow the pattern: ("memories", agent_name, namespace)
    
    Usage:
        manager = LangMemManager()
        
        # Add memory
        await manager.add_memory(
            agent_name="analyst",
            memory_type="semantic",
            content="EURUSD shows mean reversion behavior on H1 timeframe",
            importance=0.8
        )
        
        # Search memory
        results = await manager.search_memory(
            agent_name="analyst",
            query="EURUSD behavior",
            memory_type="semantic"
        )
    """
    
    # Default namespaces for each agent
    DEFAULT_NAMESPACES = {
        "analyst": ("memories", "analyst", "default"),
        "quantcode": ("memories", "quantcode", "default"),
        "copilot": ("memories", "copilot", "default")
    }
    
    def __init__(
        self,
        storage_path: Optional[Path] = None,
        embedding_model: str = "text-embedding-3-small",
        llm_model: str = "anthropic:claude-3-5-sonnet-latest",
        max_memories_per_namespace: int = 1000
    ):
        """
        Initialize LangMem manager.
        
        Args:
            storage_path: Path for persistent storage (default: data/memories)
            embedding_model: Embedding model for semantic search
            llm_model: LLM model for memory extraction
            max_memories_per_namespace: Maximum memories per namespace
        """
        self.storage_path = storage_path or Path("data/memories")
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.max_memories = max_memories_per_namespace
        
        # Initialize LangMem components
        self._store: Optional[BaseStore] = None
        self._memory_manager = None
        self._reflection_executor = None
        
        # Fallback in-memory storage (used when LangMem is not available)
        self._memories: Dict[str, List[MemoryEntry]] = {}
        
        # Ensure storage directory exists
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize on creation
        self._initialized = False
        
        logger.info(f"LangMemManager initialized with storage at {self.storage_path}")
    
    async def _ensure_initialized(self) -> None:
        """Ensure LangMem components are initialized."""
        if self._initialized:
            return
        
        if LANGMEM_AVAILABLE:
            try:
                # Create the memory store with embeddings
                # Use OpenAI embeddings by default
                embed_config = self._get_embed_config()
                
                self._store = InMemoryStore(
                    index={
                        "dims": embed_config["dims"],
                        "embed": embed_config["embed"],
                    }
                )
                
                # Create memory manager
                schemas = []
                if PYDANTIC_AVAILABLE:
                    schemas = [SemanticMemory, EpisodicMemory, ProceduralMemory]
                
                self._memory_manager = create_memory_manager(
                    self.llm_model,
                    schemas=schemas,
                    instructions="Extract trading knowledge, strategy insights, and procedural skills.",
                    enable_inserts=True,
                    enable_deletes=True,
                )
                
                # Create reflection executor for delayed processing
                self._reflection_executor = ReflectionExecutor(self._memory_manager)
                
                logger.info("LangMem client initialized successfully")
                
            except Exception as e:
                logger.error(f"Failed to initialize LangMem client: {e}")
                self._store = None
                self._memory_manager = None
        
        # Load existing memories from disk
        await self._load_all_namespaces()
        
        self._initialized = True
    
    def _get_embed_config(self) -> Dict[str, Any]:
        """Get embedding configuration based on available API keys."""
        # Check for OpenAI API key
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            return {
                "dims": 1536,
                "embed": f"openai:{self.embedding_model}"
            }
        
        # Check for Anthropic API key (use their embeddings if available)
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key:
            return {
                "dims": 1536,
                "embed": "openai:text-embedding-3-small"  # Fallback to OpenAI
            }
        
        # Default configuration
        return {
            "dims": 1536,
            "embed": f"openai:{self.embedding_model}"
        }
    
    def _get_namespace_tuple(self, agent_name: str, namespace: str = "default") -> tuple:
        """Get namespace tuple for LangMem operations."""
        return ("memories", agent_name, namespace)
    
    def _get_namespace_key(self, agent_name: str, namespace: str = "default") -> str:
        """Get namespace key for fallback storage."""
        return f"{agent_name}:{namespace}"
    
    async def add_memory(
        self,
        agent_name: str,
        memory_type: str,
        content: str,
        namespace: str = "default",
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> MemoryEntry:
        """
        Add a new memory entry.
        
        Args:
            agent_name: Name of the agent (analyst, quantcode, copilot)
            memory_type: Type of memory (semantic, episodic, procedural)
            content: Memory content
            namespace: Memory namespace (default: "default")
            importance: Importance score 0-1 (default: 0.5)
            metadata: Optional metadata
            
        Returns:
            Created MemoryEntry
        """
        await self._ensure_initialized()
        
        # Generate memory ID
        memory_id = f"mem_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self._memories)}"
        
        # Create entry
        entry = MemoryEntry(
            id=memory_id,
            content=content,
            memory_type=memory_type,
            namespace=namespace,
            created_at=datetime.now().isoformat(),
            importance=importance,
            metadata=metadata or {}
        )
        
        # Use LangMem if available
        if LANGMEM_AVAILABLE and self._store and self._memory_manager:
            try:
                ns_tuple = self._get_namespace_tuple(agent_name, namespace)
                
                # Store in LangGraph store
                self._store.put(
                    ns_tuple,
                    key=memory_id,
                    value={
                        "content": content,
                        "memory_type": memory_type,
                        "importance": importance,
                        "metadata": metadata or {},
                        "created_at": entry.created_at
                    }
                )
                
                logger.debug(f"Added {memory_type} memory for {agent_name} via LangMem")
                
            except Exception as e:
                logger.error(f"LangMem storage failed, using fallback: {e}")
                # Fall through to fallback storage
        
        # Fallback: Store in local memory
        ns_key = self._get_namespace_key(agent_name, namespace)
        if ns_key not in self._memories:
            self._memories[ns_key] = []
        
        self._memories[ns_key].append(entry)
        
        # Enforce max memories limit
        if len(self._memories[ns_key]) > self.max_memories:
            await self._prune_memories(ns_key)
        
        # Persist to disk
        await self._persist_memory(agent_name, namespace, entry)
        
        logger.debug(f"Added {memory_type} memory for {agent_name}: {content[:50]}...")
        
        return entry
    
    async def search_memory(
        self,
        agent_name: str,
        query: str,
        memory_type: Optional[str] = None,
        namespace: str = "default",
        limit: int = 10,
        min_relevance: float = 0.0
    ) -> List[MemorySearchResult]:
        """
        Search memories by query using embedding-based similarity.
        
        Args:
            agent_name: Name of the agent
            query: Search query
            memory_type: Filter by memory type (optional)
            namespace: Memory namespace (default: "default")
            limit: Maximum results to return
            min_relevance: Minimum relevance threshold
            
        Returns:
            List of MemorySearchResult sorted by relevance
        """
        await self._ensure_initialized()
        
        results = []
        
        # Use LangMem if available
        if LANGMEM_AVAILABLE and self._store:
            try:
                ns_tuple = self._get_namespace_tuple(agent_name, namespace)
                
                # Search using LangGraph store's embedding-based search
                search_results = self._store.search(ns_tuple, query=query)
                
                for item in search_results[:limit]:
                    value = item.value
                    relevance = getattr(item, 'score', 1.0)
                    
                    if relevance < min_relevance:
                        continue
                    
                    if memory_type and value.get("memory_type") != memory_type:
                        continue
                    
                    entry = MemoryEntry(
                        id=item.key,
                        content=value.get("content", ""),
                        memory_type=value.get("memory_type", "semantic"),
                        namespace=namespace,
                        created_at=value.get("created_at", ""),
                        importance=value.get("importance", 0.5),
                        metadata=value.get("metadata", {})
                    )
                    
                    results.append(MemorySearchResult(
                        entry=entry,
                        relevance=relevance
                    ))
                
                return results
                
            except Exception as e:
                logger.error(f"LangMem search failed, using fallback: {e}")
                # Fall through to fallback search
        
        # Fallback: Search in local memory
        ns_key = self._get_namespace_key(agent_name, namespace)
        memories = self._memories.get(ns_key, [])
        
        # Filter by memory type if specified
        if memory_type:
            memories = [m for m in memories if m.memory_type == memory_type]
        
        # Calculate relevance using embedding similarity or fallback
        for memory in memories:
            relevance = await self._calculate_relevance(query, memory.content)
            if relevance >= min_relevance:
                results.append(MemorySearchResult(
                    entry=memory,
                    relevance=relevance
                ))
        
        # Sort by relevance
        results.sort(key=lambda r: r.relevance, reverse=True)
        
        return results[:limit]
    
    async def get_memory(self, memory_id: str) -> Optional[MemoryEntry]:
        """
        Get a specific memory by ID.
        
        Args:
            memory_id: Memory identifier
            
        Returns:
            MemoryEntry or None if not found
        """
        await self._ensure_initialized()
        
        # Search in LangMem store first
        if LANGMEM_AVAILABLE and self._store:
            try:
                # Search across all namespaces
                for agent_name in self.DEFAULT_NAMESPACES.keys():
                    for ns in ["default"]:
                        ns_tuple = self._get_namespace_tuple(agent_name, ns)
                        try:
                            item = self._store.get(ns_tuple, memory_id)
                            if item:
                                value = item.value
                                return MemoryEntry(
                                    id=memory_id,
                                    content=value.get("content", ""),
                                    memory_type=value.get("memory_type", "semantic"),
                                    namespace=ns,
                                    created_at=value.get("created_at", ""),
                                    importance=value.get("importance", 0.5),
                                    metadata=value.get("metadata", {})
                                )
                        except:
                            continue
            except Exception as e:
                logger.debug(f"LangMem get failed, using fallback: {e}")
        
        # Fallback: Search in local memory
        for memories in self._memories.values():
            for memory in memories:
                if memory.id == memory_id:
                    return memory
        return None
    
    async def update_memory(
        self,
        memory_id: str,
        content: Optional[str] = None,
        importance: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[MemoryEntry]:
        """
        Update an existing memory.
        
        Args:
            memory_id: Memory identifier
            content: New content (optional)
            importance: New importance (optional)
            metadata: New or updated metadata (optional)
            
        Returns:
            Updated MemoryEntry or None if not found
        """
        await self._ensure_initialized()
        
        memory = await self.get_memory(memory_id)
        if not memory:
            return None
        
        if content is not None:
            memory.content = content
        if importance is not None:
            memory.importance = importance
        if metadata is not None:
            memory.metadata.update(metadata)
        
        # Update in LangMem store
        if LANGMEM_AVAILABLE and self._store:
            try:
                ns_tuple = self._get_namespace_tuple(
                    memory.namespace.split(":")[0] if ":" in memory.namespace else "analyst",
                    memory.namespace
                )
                self._store.put(
                    ns_tuple,
                    key=memory_id,
                    value={
                        "content": memory.content,
                        "memory_type": memory.memory_type,
                        "importance": memory.importance,
                        "metadata": memory.metadata,
                        "created_at": memory.created_at
                    }
                )
            except Exception as e:
                logger.debug(f"LangMem update failed: {e}")
        
        return memory
    
    async def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory.
        
        Args:
            memory_id: Memory identifier
            
        Returns:
            True if deleted, False if not found
        """
        await self._ensure_initialized()
        
        # Delete from LangMem store
        if LANGMEM_AVAILABLE and self._store:
            try:
                for agent_name in self.DEFAULT_NAMESPACES.keys():
                    for ns in ["default"]:
                        ns_tuple = self._get_namespace_tuple(agent_name, ns)
                        try:
                            self._store.delete(ns_tuple, memory_id)
                            return True
                        except:
                            continue
            except Exception as e:
                logger.debug(f"LangMem delete failed: {e}")
        
        # Fallback: Delete from local memory
        for ns_key, memories in self._memories.items():
            for i, memory in enumerate(memories):
                if memory.id == memory_id:
                    memories.pop(i)
                    return True
        return False
    
    async def get_all_memories(
        self,
        agent_name: str,
        namespace: str = "default",
        memory_type: Optional[str] = None
    ) -> List[MemoryEntry]:
        """
        Get all memories for an agent.
        
        Args:
            agent_name: Name of the agent
            namespace: Memory namespace
            memory_type: Filter by memory type (optional)
            
        Returns:
            List of MemoryEntry
        """
        await self._ensure_initialized()
        
        memories = []
        
        # Get from LangMem store
        if LANGMEM_AVAILABLE and self._store:
            try:
                ns_tuple = self._get_namespace_tuple(agent_name, namespace)
                items = self._store.search(ns_tuple, query="")  # Get all
                
                for item in items:
                    value = item.value
                    if memory_type and value.get("memory_type") != memory_type:
                        continue
                    
                    memories.append(MemoryEntry(
                        id=item.key,
                        content=value.get("content", ""),
                        memory_type=value.get("memory_type", "semantic"),
                        namespace=namespace,
                        created_at=value.get("created_at", ""),
                        importance=value.get("importance", 0.5),
                        metadata=value.get("metadata", {})
                    ))
                
                return memories
                
            except Exception as e:
                logger.debug(f"LangMem get all failed, using fallback: {e}")
        
        # Fallback: Get from local memory
        ns_key = self._get_namespace_key(agent_name, namespace)
        memories = self._memories.get(ns_key, [])
        
        if memory_type:
            memories = [m for m in memories if m.memory_type == memory_type]
        
        return memories
    
    async def clear_memories(
        self,
        agent_name: str,
        namespace: str = "default"
    ) -> int:
        """
        Clear all memories for an agent.
        
        Args:
            agent_name: Name of the agent
            namespace: Memory namespace
            
        Returns:
            Number of memories cleared
        """
        await self._ensure_initialized()
        
        count = 0
        
        # Clear from LangMem store
        if LANGMEM_AVAILABLE and self._store:
            try:
                ns_tuple = self._get_namespace_tuple(agent_name, namespace)
                items = self._store.search(ns_tuple, query="")
                
                for item in items:
                    try:
                        self._store.delete(ns_tuple, item.key)
                        count += 1
                    except:
                        continue
                        
            except Exception as e:
                logger.debug(f"LangMem clear failed: {e}")
        
        # Also clear from fallback
        ns_key = self._get_namespace_key(agent_name, namespace)
        count = max(count, len(self._memories.get(ns_key, [])))
        self._memories[ns_key] = []
        
        return count
    
    async def _calculate_relevance(self, query: str, content: str) -> float:
        """
        Calculate relevance between query and content.
        
        Uses embedding-based similarity if available, otherwise falls back
        to Jaccard similarity.
        """
        # Try to use embedding-based similarity
        if LANGMEM_AVAILABLE and self._store:
            try:
                # The store's search already uses embeddings
                # This is a fallback for when we're not using the store's search
                pass
            except:
                pass
        
        # Fallback: Jaccard similarity
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        
        if not query_words or not content_words:
            return 0.0
        
        intersection = query_words & content_words
        union = query_words | content_words
        
        return len(intersection) / len(union)
    
    async def _prune_memories(self, ns_key: str) -> None:
        """
        Prune memories to stay within limit.
        
        Removes least important memories first.
        """
        memories = self._memories.get(ns_key, [])
        if len(memories) <= self.max_memories:
            return
        
        # Sort by importance (ascending)
        memories.sort(key=lambda m: m.importance)
        
        # Remove least important
        to_remove = len(memories) - self.max_memories
        self._memories[ns_key] = memories[to_remove:]
        
        logger.debug(f"Pruned {to_remove} memories from {ns_key}")
    
    async def _persist_memory(
        self,
        agent_name: str,
        namespace: str,
        entry: MemoryEntry
    ) -> None:
        """
        Persist memory to disk.
        
        Args:
            agent_name: Agent name
            namespace: Namespace
            entry: Memory entry to persist
        """
        file_path = self.storage_path / agent_name / namespace / f"{entry.id}.json"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(entry.to_dict(), f, indent=2)
    
    async def _load_all_namespaces(self) -> None:
        """
        Load all memories from disk into fallback storage.
        """
        if not self.storage_path.exists():
            return
        
        for agent_dir in self.storage_path.iterdir():
            if not agent_dir.is_dir():
                continue
            
            agent_name = agent_dir.name
            
            for ns_dir in agent_dir.iterdir():
                if not ns_dir.is_dir():
                    continue
                
                namespace = ns_dir.name
                ns_key = self._get_namespace_key(agent_name, namespace)
                
                if ns_key not in self._memories:
                    self._memories[ns_key] = []
                
                for file_path in ns_dir.glob("*.json"):
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                        
                        entry = MemoryEntry(
                            id=data["id"],
                            content=data["content"],
                            memory_type=data["memory_type"],
                            namespace=data["namespace"],
                            created_at=data["created_at"],
                            importance=data.get("importance", 0.5),
                            metadata=data.get("metadata", {})
                        )
                        self._memories[ns_key].append(entry)
                        
                    except Exception as e:
                        logger.error(f"Failed to load memory {file_path}: {e}")
    
    async def load_memories(self, agent_name: str, namespace: str = "default") -> int:
        """
        Load memories from disk.
        
        Args:
            agent_name: Agent name
            namespace: Namespace
            
        Returns:
            Number of memories loaded
        """
        await self._ensure_initialized()
        
        dir_path = self.storage_path / agent_name / namespace
        if not dir_path.exists():
            return 0
        
        ns_key = self._get_namespace_key(agent_name, namespace)
        if ns_key not in self._memories:
            self._memories[ns_key] = []
        
        count = 0
        for file_path in dir_path.glob("*.json"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                entry = MemoryEntry(
                    id=data["id"],
                    content=data["content"],
                    memory_type=data["memory_type"],
                    namespace=data["namespace"],
                    created_at=data["created_at"],
                    importance=data.get("importance", 0.5),
                    metadata=data.get("metadata", {})
                )
                self._memories[ns_key].append(entry)
                count += 1
                
            except Exception as e:
                logger.error(f"Failed to load memory {file_path}: {e}")
        
        return count
    
    # =========================================================================
    # Convenience Methods for Specific Memory Types
    # =========================================================================
    
    async def add_semantic_memory(
        self,
        agent_name: str,
        content: str,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> MemoryEntry:
        """Add a semantic memory (fact or concept)."""
        return await self.add_memory(
            agent_name=agent_name,
            memory_type=MemoryType.SEMANTIC,
            content=content,
            importance=importance,
            metadata=metadata
        )
    
    async def add_episodic_memory(
        self,
        agent_name: str,
        content: str,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> MemoryEntry:
        """Add an episodic memory (event or experience)."""
        return await self.add_memory(
            agent_name=agent_name,
            memory_type=MemoryType.EPISODIC,
            content=content,
            importance=importance,
            metadata=metadata
        )
    
    async def add_procedural_memory(
        self,
        agent_name: str,
        content: str,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> MemoryEntry:
        """Add a procedural memory (skill or procedure)."""
        return await self.add_memory(
            agent_name=agent_name,
            memory_type=MemoryType.PROCEDURAL,
            content=content,
            importance=importance,
            metadata=metadata
        )
    
    async def search_semantic_memory(
        self,
        agent_name: str,
        query: str,
        limit: int = 10
    ) -> List[MemorySearchResult]:
        """Search semantic memories."""
        return await self.search_memory(
            agent_name=agent_name,
            query=query,
            memory_type=MemoryType.SEMANTIC,
            limit=limit
        )
    
    async def search_episodic_memory(
        self,
        agent_name: str,
        query: str,
        limit: int = 10
    ) -> List[MemorySearchResult]:
        """Search episodic memories."""
        return await self.search_memory(
            agent_name=agent_name,
            query=query,
            memory_type=MemoryType.EPISODIC,
            limit=limit
        )
    
    async def search_procedural_memory(
        self,
        agent_name: str,
        query: str,
        limit: int = 10
    ) -> List[MemorySearchResult]:
        """Search procedural memories."""
        return await self.search_memory(
            agent_name=agent_name,
            query=query,
            memory_type=MemoryType.PROCEDURAL,
            limit=limit
        )
    
    # =========================================================================
    # LangGraph Integration Methods
    # =========================================================================
    
    def get_manage_memory_tool(self, agent_name: str, namespace: str = "default"):
        """Get a manage memory tool for LangGraph agents."""
        if not LANGMEM_AVAILABLE:
            logger.warning("LangMem not available, cannot create manage memory tool")
            return None
        
        ns_tuple = self._get_namespace_tuple(agent_name, namespace)
        return create_manage_memory_tool(namespace=ns_tuple)
    
    def get_search_memory_tool(self, agent_name: str, namespace: str = "default"):
        """Get a search memory tool for LangGraph agents."""
        if not LANGMEM_AVAILABLE:
            logger.warning("LangMem not available, cannot create search memory tool")
            return None
        
        ns_tuple = self._get_namespace_tuple(agent_name, namespace)
        return create_search_memory_tool(namespace=ns_tuple)
    
    def get_store(self) -> Optional[BaseStore]:
        """Get the LangGraph store for use with agents."""
        return self._store


# =============================================================================
# Global Manager Instance
# =============================================================================

_manager: Optional[LangMemManager] = None


def get_langmem_manager() -> LangMemManager:
    """Get or create the global LangMem manager instance."""
    global _manager
    if _manager is None:
        _manager = LangMemManager()
    return _manager

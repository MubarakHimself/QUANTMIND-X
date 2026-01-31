"""
LangMem Memory Management Integration

Implements semantic, episodic, and procedural memory using LangMem framework.
Integrates with ChromaDB for vector-based memory search.

**Validates: Property 17: Memory Namespace Hierarchy**
**Validates: Property 18: Memory Consolidation Timing**
"""

import logging
from typing import Tuple, List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ============================================================================
# Memory Data Classes
# ============================================================================

@dataclass
class Triple:
    """Semantic memory triple (subject-predicate-object)."""
    subject: str
    predicate: str
    object: str
    context: str
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


@dataclass
class Episode:
    """Episodic memory episode."""
    observation: str
    thoughts: str
    action: str
    result: str
    timestamp: datetime = None
    agent_type: str = "unknown"
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


@dataclass
class Instruction:
    """Procedural memory instruction."""
    task: str
    steps: List[str]
    conditions: Dict[str, Any]
    expected_outcome: str
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


# ============================================================================
# Semantic Memory
# ============================================================================

class SemanticMemory:
    """
    Stores facts and relationships as triples.
    
    **Validates: Requirements 10.1**
    """
    
    def __init__(self, namespace: Tuple[str, ...], db_manager=None):
        """
        Initialize semantic memory.
        
        Args:
            namespace: Hierarchical namespace (e.g., ("memories", "user_123", "project_456"))
            db_manager: DatabaseManager instance for ChromaDB access
        """
        self.namespace = namespace
        self._validate_namespace()
        
        if db_manager is None:
            from src.database.manager import DatabaseManager
            self.db_manager = DatabaseManager()
        else:
            self.db_manager = db_manager
        
        logger.info(f"SemanticMemory initialized with namespace: {namespace}")
    
    def _validate_namespace(self):
        """
        Validate namespace hierarchy.
        
        **Validates: Property 17: Memory Namespace Hierarchy**
        """
        if not isinstance(self.namespace, tuple):
            raise ValueError("Namespace must be a tuple")
        
        if len(self.namespace) < 2:
            raise ValueError("Namespace must have at least 2 levels")
        
        if self.namespace[0] != "memories":
            raise ValueError("Namespace must start with 'memories'")
    
    def store_triple(
        self,
        subject: str,
        predicate: str,
        obj: str,
        context: str
    ) -> str:
        """
        Store semantic triple.
        
        Args:
            subject: Subject of the triple
            predicate: Relationship/predicate
            obj: Object of the triple
            context: Context in which triple was created
            
        Returns:
            Memory ID
        """
        triple = Triple(
            subject=subject,
            predicate=predicate,
            object=obj,
            context=context
        )
        
        # Store in ChromaDB
        memory_id = f"triple_{triple.timestamp.timestamp()}"
        content = f"{subject} {predicate} {obj}"
        
        self.db_manager.add_agent_memory(
            memory_id=memory_id,
            content=content,
            agent_type=self.namespace[1] if len(self.namespace) > 1 else "unknown",
            memory_type="semantic",
            context=context
        )
        
        logger.debug(f"Stored triple: {content}")
        return memory_id
    
    def search_triples(
        self,
        query: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search semantic triples by query.
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            List of matching triples
        """
        results = self.db_manager.search_agent_memory(
            query=query,
            memory_type="semantic",
            limit=limit
        )
        
        return results


# ============================================================================
# Episodic Memory
# ============================================================================

class EpisodicMemory:
    """
    Stores agent experiences and learning episodes.
    
    **Validates: Requirements 10.2**
    """
    
    def __init__(self, namespace: Tuple[str, ...], db_manager=None):
        """
        Initialize episodic memory.
        
        Args:
            namespace: Hierarchical namespace
            db_manager: DatabaseManager instance
        """
        self.namespace = namespace
        self._validate_namespace()
        
        if db_manager is None:
            from src.database.manager import DatabaseManager
            self.db_manager = DatabaseManager()
        else:
            self.db_manager = db_manager
        
        logger.info(f"EpisodicMemory initialized with namespace: {namespace}")
    
    def _validate_namespace(self):
        """Validate namespace hierarchy."""
        if not isinstance(self.namespace, tuple):
            raise ValueError("Namespace must be a tuple")
        
        if len(self.namespace) < 2:
            raise ValueError("Namespace must have at least 2 levels")
        
        if self.namespace[0] != "memories":
            raise ValueError("Namespace must start with 'memories'")
    
    def store_episode(
        self,
        observation: str,
        thoughts: str,
        action: str,
        result: str,
        agent_type: str = "unknown"
    ) -> str:
        """
        Store agent episode.
        
        Args:
            observation: What the agent observed
            thoughts: Agent's reasoning
            action: Action taken
            result: Outcome of action
            agent_type: Type of agent
            
        Returns:
            Memory ID
        """
        episode = Episode(
            observation=observation,
            thoughts=thoughts,
            action=action,
            result=result,
            agent_type=agent_type
        )
        
        # Store in ChromaDB
        memory_id = f"episode_{episode.timestamp.timestamp()}"
        content = f"Observation: {observation}\nThoughts: {thoughts}\nAction: {action}\nResult: {result}"
        
        self.db_manager.add_agent_memory(
            memory_id=memory_id,
            content=content,
            agent_type=agent_type,
            memory_type="episodic",
            context=f"episode_{agent_type}"
        )
        
        logger.debug(f"Stored episode for {agent_type}")
        return memory_id
    
    def search_episodes(
        self,
        query: str,
        agent_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search episodic memories.
        
        Args:
            query: Search query
            agent_type: Filter by agent type
            limit: Maximum results
            
        Returns:
            List of matching episodes
        """
        results = self.db_manager.search_agent_memory(
            query=query,
            agent_type=agent_type,
            memory_type="episodic",
            limit=limit
        )
        
        return results


# ============================================================================
# Procedural Memory
# ============================================================================

class ProceduralMemory:
    """
    Stores instructions and procedures.
    
    **Validates: Requirements 10.3**
    """
    
    def __init__(self, namespace: Tuple[str, ...], db_manager=None):
        """
        Initialize procedural memory.
        
        Args:
            namespace: Hierarchical namespace
            db_manager: DatabaseManager instance
        """
        self.namespace = namespace
        self._validate_namespace()
        
        if db_manager is None:
            from src.database.manager import DatabaseManager
            self.db_manager = DatabaseManager()
        else:
            self.db_manager = db_manager
        
        logger.info(f"ProceduralMemory initialized with namespace: {namespace}")
    
    def _validate_namespace(self):
        """Validate namespace hierarchy."""
        if not isinstance(self.namespace, tuple):
            raise ValueError("Namespace must be a tuple")
        
        if len(self.namespace) < 2:
            raise ValueError("Namespace must have at least 2 levels")
        
        if self.namespace[0] != "memories":
            raise ValueError("Namespace must start with 'memories'")
    
    def store_instruction(
        self,
        task: str,
        steps: List[str],
        conditions: Dict[str, Any],
        expected_outcome: str
    ) -> str:
        """
        Store procedural instruction.
        
        Args:
            task: Task description
            steps: List of steps to perform
            conditions: Conditions for execution
            expected_outcome: Expected result
            
        Returns:
            Memory ID
        """
        instruction = Instruction(
            task=task,
            steps=steps,
            conditions=conditions,
            expected_outcome=expected_outcome
        )
        
        # Store in ChromaDB
        memory_id = f"instruction_{instruction.timestamp.timestamp()}"
        content = f"Task: {task}\nSteps: {', '.join(steps)}\nOutcome: {expected_outcome}"
        
        self.db_manager.add_agent_memory(
            memory_id=memory_id,
            content=content,
            agent_type=self.namespace[1] if len(self.namespace) > 1 else "unknown",
            memory_type="procedural",
            context=f"instruction_{task}"
        )
        
        logger.debug(f"Stored instruction: {task}")
        return memory_id
    
    def search_instructions(
        self,
        query: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search procedural instructions.
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            List of matching instructions
        """
        results = self.db_manager.search_agent_memory(
            query=query,
            memory_type="procedural",
            limit=limit
        )
        
        return results


# ============================================================================
# Reflection Executor
# ============================================================================

class ReflectionExecutor:
    """
    Handles deferred memory processing and consolidation.
    
    **Validates: Property 18: Memory Consolidation Timing**
    """
    
    def __init__(self, consolidation_delay: int = 30):
        """
        Initialize reflection executor.
        
        Args:
            consolidation_delay: Delay in minutes before consolidation (default: 30)
        """
        self.consolidation_delay = consolidation_delay
        self.pending_memories = []
        
        logger.info(f"ReflectionExecutor initialized with {consolidation_delay}min delay")
    
    def queue_memory(self, memory_data: Dict[str, Any]):
        """
        Queue memory for deferred processing.
        
        Args:
            memory_data: Memory data to process
        """
        memory_data['queued_at'] = datetime.utcnow()
        self.pending_memories.append(memory_data)
        
        logger.debug(f"Queued memory for consolidation: {memory_data.get('type', 'unknown')}")
    
    def process_pending_memories(self):
        """
        Process memories that have passed the consolidation delay.
        
        **Validates: Property 18: Memory Consolidation Timing**
        """
        current_time = datetime.utcnow()
        processed = []
        
        for memory in self.pending_memories:
            queued_at = memory.get('queued_at')
            if not queued_at:
                continue
            
            # Check if consolidation delay has passed
            time_elapsed = (current_time - queued_at).total_seconds() / 60
            
            if time_elapsed >= self.consolidation_delay:
                # Process memory
                self._consolidate_memory(memory)
                processed.append(memory)
        
        # Remove processed memories
        for memory in processed:
            self.pending_memories.remove(memory)
        
        logger.info(f"Processed {len(processed)} memories after consolidation delay")
        
        return len(processed)
    
    def _consolidate_memory(self, memory_data: Dict[str, Any]):
        """
        Consolidate memory after delay.
        
        Args:
            memory_data: Memory to consolidate
        """
        # In production, implement actual consolidation logic
        # For now, just log
        logger.debug(f"Consolidated memory: {memory_data.get('type', 'unknown')}")


# ============================================================================
# Memory Tools for Agent Access
# ============================================================================

def create_manage_memory_tool(namespace: Tuple[str, ...]):
    """
    Create memory management tool for agent access.
    
    **Validates: Requirements 10.8**
    
    Args:
        namespace: Memory namespace
        
    Returns:
        Memory management function
    """
    semantic = SemanticMemory(namespace)
    episodic = EpisodicMemory(namespace)
    procedural = ProceduralMemory(namespace)
    
    def manage_memory(
        memory_type: str,
        action: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Manage agent memory.
        
        Args:
            memory_type: Type of memory (semantic, episodic, procedural)
            action: Action to perform (store, search)
            **kwargs: Additional arguments
            
        Returns:
            Operation result
        """
        try:
            if memory_type == "semantic":
                if action == "store":
                    memory_id = semantic.store_triple(**kwargs)
                    return {"success": True, "memory_id": memory_id}
                elif action == "search":
                    results = semantic.search_triples(**kwargs)
                    return {"success": True, "results": results}
            
            elif memory_type == "episodic":
                if action == "store":
                    memory_id = episodic.store_episode(**kwargs)
                    return {"success": True, "memory_id": memory_id}
                elif action == "search":
                    results = episodic.search_episodes(**kwargs)
                    return {"success": True, "results": results}
            
            elif memory_type == "procedural":
                if action == "store":
                    memory_id = procedural.store_instruction(**kwargs)
                    return {"success": True, "memory_id": memory_id}
                elif action == "search":
                    results = procedural.search_instructions(**kwargs)
                    return {"success": True, "results": results}
            
            return {"success": False, "error": "Invalid memory type or action"}
            
        except Exception as e:
            logger.error(f"Memory management failed: {e}")
            return {"success": False, "error": str(e)}
    
    return manage_memory


def create_search_memory_tool(namespace: Tuple[str, ...]):
    """
    Create memory search tool for agent retrieval.
    
    **Validates: Requirements 10.9**
    
    Args:
        namespace: Memory namespace
        
    Returns:
        Memory search function
    """
    semantic = SemanticMemory(namespace)
    episodic = EpisodicMemory(namespace)
    procedural = ProceduralMemory(namespace)
    
    def search_memory(
        query: str,
        memory_types: List[str] = None,
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Search across memory types.
        
        Args:
            query: Search query
            memory_types: Types to search (default: all)
            limit: Maximum results per type
            
        Returns:
            Search results
        """
        if memory_types is None:
            memory_types = ["semantic", "episodic", "procedural"]
        
        results = {}
        
        try:
            if "semantic" in memory_types:
                results["semantic"] = semantic.search_triples(query, limit)
            
            if "episodic" in memory_types:
                results["episodic"] = episodic.search_episodes(query, limit=limit)
            
            if "procedural" in memory_types:
                results["procedural"] = procedural.search_instructions(query, limit)
            
            return {"success": True, "results": results}
            
        except Exception as e:
            logger.error(f"Memory search failed: {e}")
            return {"success": False, "error": str(e)}
    
    return search_memory

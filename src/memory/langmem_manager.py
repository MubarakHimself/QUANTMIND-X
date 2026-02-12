"""
LangMem Memory Integration for QuantMind Agents.

Implements semantic, episodic, and procedural memory for agents using LangMem SDK.
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

logger = logging.getLogger(__name__)

try:
    from langmem import (
        create_memory_manager,
        create_memory_store_manager,
        create_manage_memory_tool,
        create_search_memory_tool,
        ReflectionExecutor
    )
    from pydantic import BaseModel
    LANGMEM_AVAILABLE = True
except ImportError:
    logger.warning("LangMem not available. Install with: pip install langmem")
    LANGMEM_AVAILABLE = False


# Memory Schemas
class Triple(BaseModel):
    """Semantic memory: Facts, preferences, and relationships as triples."""
    subject: str
    predicate: str
    object: str
    context: Optional[str] = None


class Episode(BaseModel):
    """Episodic memory: Experiences with chain of reasoning."""
    observation: str  # What happened
    thoughts: str     # Internal reasoning
    action: str       # What was done
    result: str       # Outcome and retrospective


class MemoryManager:
    """
    Manages agent memory using LangMem.
    
    Features:
    - Semantic memory (facts, preferences)
    - Episodic memory (experiences, learning)
    - Procedural memory (skill improvement)
    - Per-agent namespaces
    - Delayed processing with ReflectionExecutor
    """
    
    def __init__(
        self,
        model: str = "anthropic:claude-3-5-sonnet-latest",
        storage_path: str = "data/memory"
    ):
        """
        Initialize memory manager.
        
        Args:
            model: LLM model for memory extraction
            storage_path: Path to store memory data
        """
        if not LANGMEM_AVAILABLE:
            raise ImportError("LangMem not available")
        
        self.model = model
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Create memory managers for each type
        self.semantic_manager = self._create_semantic_manager()
        self.episodic_manager = self._create_episodic_manager()
        
        # Reflection executor for delayed processing
        self.executor = ReflectionExecutor(self.semantic_manager)
        
        logger.info("LangMem memory manager initialized")
    
    def _create_semantic_manager(self):
        """Create memory manager for semantic memory (facts)."""
        return create_memory_store_manager(
            self.model,
            namespace=("quantmind", "semantic"),
            schemas=[Triple],
            instructions="Extract user preferences, strategy facts, and system knowledge as triples.",
            enable_inserts=True,
            enable_deletes=True
        )
    
    def _create_episodic_manager(self):
        """Create memory manager for episodic memory (experiences)."""
        return create_memory_store_manager(
            self.model,
            namespace=("quantmind", "episodic"),
            schemas=[Episode],
            instructions="Capture successful explanations and problem-solving experiences with full reasoning chain.",
            enable_inserts=True
        )
    
    def get_tools_for_agent(self, agent_name: str) -> List:
        """
        Get memory management tools for specific agent.
        
        Args:
            agent_name: Agent identifier (analyst, quantcode, copilot)
        
        Returns:
            List of LangChain tools for memory management
        """
        namespace = ("quantmind", agent_name)
        
        return [
            create_manage_memory_tool(namespace=namespace),
            create_search_memory_tool(namespace=namespace)
        ]
    
    async def process_conversation(
        self,
        messages: List[Dict[str, str]],
        agent_name: str,
        delay_seconds: int = 1800
    ):
        """
        Queue conversation for memory extraction (delayed processing).
        
        Args:
            messages: Conversation messages
            agent_name: Agent that had the conversation
            delay_seconds: Delay before processing (default 30 min)
        """
        to_process = {
            "messages": messages,
            "agent": agent_name
        }
        
        # Submit for delayed processing (cancels if new messages arrive)
        self.executor.submit(to_process, after_seconds=delay_seconds)
        logger.debug(f"Queued conversation for memory extraction (agent: {agent_name})")
    
    def search_memory(
        self,
        query: str,
        agent_name: str = None,
        memory_type: str = "semantic"
    ) -> List[Dict[str, Any]]:
        """
        Search agent memory.
        
        Args:
            query: Search query
            agent_name: Optional agent filter
            memory_type: "semantic" or "episodic"
        
        Returns:
            List of memory entries
        """
        # TODO: Implement actual LangGraph store search
        logger.info(f"Searching {memory_type} memory: {query}")
        return []
    
    def clear_agent_memory(self, agent_name: str):
        """
        Clear all memory for specific agent.
        
        Args:
            agent_name: Agent to clear memory for
        """
        # TODO: Implement memory clearing
        logger.warning(f"Clearing memory for agent: {agent_name}")


# Global memory manager instance
_memory_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    """Get or create global memory manager instance."""
    global _memory_manager
    if _memory_manager is None:
        try:
            _memory_manager = MemoryManager()
        except ImportError:
           logger.error("LangMem not available, memory features disabled")
            return None
    return _memory_manager


__all__ = [
    'MemoryManager',
    'get_memory_manager',
    'Triple',
    'Episode'
]

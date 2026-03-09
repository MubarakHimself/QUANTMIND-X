"""
Agent Registry - DEPRECATED

Use floor_manager /api/floor-manager endpoints instead.
This module used the factory-based agent system which has been removed.

**Validates: Phase 3.1 - Agent Registry**
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import threading

logger = logging.getLogger(__name__)

logger.warning(
    "registry.py is deprecated. "
    "Use /api/floor-manager endpoints instead."
)


class CompiledAgent:
    """Deprecated - use floor_manager instead."""
    pass

logger = logging.getLogger(__name__)


@dataclass
class AgentInfo:
    """Information about a registered agent."""
    agent_id: str
    agent_type: str
    name: str
    registered_at: datetime
    last_invoked: Optional[datetime] = None
    invocation_count: int = 0


class AgentRegistry:
    """
    Singleton registry for managing factory-created agents.
    
    Provides agent registration, retrieval, and lifecycle management.
    """
    
    _instance: Optional["AgentRegistry"] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._agents: Dict[str, CompiledAgent] = {}
            self._agent_info: Dict[str, AgentInfo] = {}
            self._lock = threading.Lock()
            self._initialized = True
            logger.info("AgentRegistry initialized")
    
    def register(self, agent: CompiledAgent) -> None:
        """
        Register an agent in the registry.
        
        Args:
            agent: CompiledAgent to register
            
        Raises:
            ValueError: If agent is already registered
        """
        agent_id = agent.agent_id
        
        with self._lock:
            if agent_id in self._agents:
                raise ValueError(f"Agent already registered: {agent_id}")
            
            self._agents[agent_id] = agent
            self._agent_info[agent_id] = AgentInfo(
                agent_id=agent_id,
                agent_type=agent.agent_type,
                name=agent.name,
                registered_at=datetime.utcnow(),
            )
            
            logger.info(f"Agent registered: {agent_id}")
    
    def unregister(self, agent_id: str) -> bool:
        """
        Unregister an agent from the registry.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            True if agent was unregistered, False if not found
        """
        with self._lock:
            if agent_id not in self._agents:
                logger.warning(f"Agent not found for unregister: {agent_id}")
                return False
            
            # Clean up the agent
            agent = self._agents[agent_id]
            try:
                agent.cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up agent {agent_id}: {e}")
            
            # Remove from registry
            del self._agents[agent_id]
            del self._agent_info[agent_id]
            
            logger.info(f"Agent unregistered: {agent_id}")
            return True
    
    def get(self, agent_id: str) -> Optional[CompiledAgent]:
        """
        Get an agent by ID.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            CompiledAgent or None if not found
        """
        return self._agents.get(agent_id)
    
    def list_agents(self) -> List[str]:
        """
        List all registered agent IDs.
        
        Returns:
            List of agent IDs
        """
        return list(self._agents.keys())
    
    def list_agent_info(self) -> List[AgentInfo]:
        """
        Get information about all registered agents.
        
        Returns:
            List of AgentInfo objects
        """
        return list(self._agent_info.values())
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics for all agents.
        
        Returns:
            Dictionary of agent statistics
        """
        stats = {}
        
        for agent_id, agent in self._agents.items():
            try:
                agent_stats = agent.get_stats()
                stats[agent_id] = agent_stats
            except Exception as e:
                logger.warning(f"Error getting stats for {agent_id}: {e}")
                stats[agent_id] = {"error": str(e)}
        
        return stats
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all agents.
        
        Returns:
            Summary dictionary
        """
        total_agents = len(self._agents)
        total_invocations = 0
        
        agent_types: Dict[str, int] = {}
        
        for agent_id, agent in self._agents.items():
            # Count by type
            agent_type = agent.agent_type
            agent_types[agent_type] = agent_types.get(agent_type, 0) + 1
            
            # Get stats
            try:
                stats = agent.get_stats()
                total_invocations += stats.get("total_invocations", 0)
            except Exception:
                pass
        
        return {
            "total_agents": total_agents,
            "total_invocations": total_invocations,
            "agent_types": agent_types,
            "agents": [
                {
                    "agent_id": info.agent_id,
                    "agent_type": info.agent_type,
                    "name": info.name,
                    "registered_at": info.registered_at.isoformat(),
                    "last_invoked": info.last_invoked.isoformat() if info.last_invoked else None,
                    "invocation_count": info.invocation_count,
                }
                for info in self._agent_info.values()
            ]
        }
    
    def cleanup_all(self) -> None:
        """Clean up all registered agents."""
        logger.info("Cleaning up all agents in registry")
        
        agent_ids = list(self._agents.keys())
        
        for agent_id in agent_ids:
            try:
                self.unregister(agent_id)
            except Exception as e:
                logger.error(f"Error cleaning up agent {agent_id}: {e}")
        
        logger.info("All agents cleaned up")
    
    def is_registered(self, agent_id: str) -> bool:
        """
        Check if an agent is registered.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            True if registered, False otherwise
        """
        return agent_id in self._agents
    
    def __len__(self) -> int:
        """Get the number of registered agents."""
        return len(self._agents)
    
    def __contains__(self, agent_id: str) -> bool:
        """Check if an agent is registered."""
        return agent_id in self._agents


# Convenience functions
_registry: Optional[AgentRegistry] = None


def get_registry() -> AgentRegistry:
    """
    Get the global agent registry.
    
    Returns:
        AgentRegistry instance
    """
    global _registry
    if _registry is None:
        _registry = AgentRegistry()
    return _registry


def register_agent(agent: CompiledAgent) -> None:
    """
    Register an agent in the global registry.
    
    Args:
        agent: CompiledAgent to register
    """
    get_registry().register(agent)


def get_agent(agent_id: str) -> Optional[CompiledAgent]:
    """
    Get an agent from the global registry.
    
    Args:
        agent_id: Agent identifier
        
    Returns:
        CompiledAgent or None
    """
    return get_registry().get(agent_id)


def unregister_agent(agent_id: str) -> bool:
    """
    Unregister an agent from the global registry.
    
    Args:
        agent_id: Agent identifier
        
    Returns:
        True if unregistered, False if not found
    """
    return get_registry().unregister(agent_id)

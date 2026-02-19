"""
Dependency Injection Container for Agent Factory

Provides centralized dependency management for factory-created agents,
including LLM providers, tool registries, checkpointers, and metrics.

**Validates: Phase 1.3 - Dependency Injection Container**
"""

import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field

from src.agents.config import AgentConfig
from src.agents.llm_provider import (
    get_llm_for_agent,
    ProviderType,
    AGENT_MODELS,
)
from src.agents.tool_registry import ToolRegistry, global_tool_registry

logger = logging.getLogger(__name__)


# Import agent-specific tools
def _get_analyst_tools() -> List[Any]:
    """Get analyst agent tools."""
    from src.agents.analyst_v2 import ANALYST_TOOLS
    return ANALYST_TOOLS


def _get_quantcode_tools() -> List[Any]:
    """Get quantcode agent tools."""
    from src.agents.quantcode_v2 import QUANTCODE_TOOLS
    return QUANTCODE_TOOLS


def _get_copilot_tools() -> List[Any]:
    """Get copilot agent tools."""
    from src.agents.copilot_v2 import COPILOT_TOOLS
    return COPILOT_TOOLS


# Tool loader mapping
TOOL_LOADERS = {
    "analyst": _get_analyst_tools,
    "quantcode": _get_quantcode_tools,
    "copilot": _get_copilot_tools,
    "router": lambda: [],  # Router uses classification, not tools
    "executor": lambda: [],  # Executor tools loaded separately
}


@dataclass
class ContainerResources:
    """Container for pooled resources."""
    db_pool: Optional[Any] = None
    redis_pool: Optional[Any] = None


class DependencyContainer:
    """
    Dependency injection container for agent factory.
    
    Manages shared resources like LLM providers, tool registries,
    checkpointers, metrics collectors, and observers.
    """
    
    def __init__(self):
        """Initialize the dependency container."""
        # Cached instances
        self._llm_providers: Dict[str, Any] = {}
        self._tool_registries: Dict[str, ToolRegistry] = {}
        self._checkpointers: Dict[str, Any] = {}
        self._metrics_collectors: Dict[str, Any] = {}
        self._observers: List[Any] = []
        
        # Resource pools
        self._resources = ContainerResources()
        
        # Configuration
        self._config = {}
        
        logger.info("DependencyContainer initialized")
    
    # =========================================================================
    # LLM Provider Management
    # =========================================================================
    
    def get_llm_provider(self, config: AgentConfig) -> Any:
        """
        Get or create an LLM provider for the given config.
        
        Args:
            config: Agent configuration
            
        Returns:
            Configured LLM instance
        """
        # Create cache key based on config
        cache_key = f"{config.llm_provider}:{config.llm_model}:{config.agent_type}"
        
        if cache_key in self._llm_providers:
            logger.debug(f"Reusing cached LLM provider: {cache_key}")
            return self._llm_providers[cache_key]
        
        # Get tools for binding
        tools = self._get_tools_for_config(config)
        
        # Create new LLM provider
        try:
            # Use the provider config from llm_provider.py
            llm = get_llm_for_agent(
                agent_type=config.agent_type,
                tools=tools if tools else None,
                use_fallback=False
            )
            
            # Override with config values if specified
            if config.llm_model:
                llm.model = config.llm_model
            
            self._llm_providers[cache_key] = llm
            logger.info(f"Created LLM provider: {cache_key}")
            
            return llm
            
        except Exception as e:
            logger.error(f"Failed to create LLM provider: {e}")
            raise
    
    def _get_tools_for_config(self, config: AgentConfig) -> List[Any]:
        """
        Get tools for the given config.
        
        Args:
            config: Agent configuration
            
        Returns:
            List of tools
        """
        tool_names = config.get_tools_for_agent_type()
        
        if not tool_names:
            return []
        
        # Get tool loader for agent type
        tool_loader = TOOL_LOADERS.get(config.agent_type)
        
        if tool_loader:
            try:
                all_tools = tool_loader()
                # Filter to requested tools
                tool_dict = {t.name if hasattr(t, 'name') else t.__name__: t for t in all_tools}
                selected_tools = [tool_dict[name] for name in tool_names if name in tool_dict]
                return selected_tools
            except Exception as e:
                logger.warning(f"Failed to load tools for {config.agent_type}: {e}")
        
        return []
    
    # =========================================================================
    # Tool Registry Management
    # =========================================================================
    
    def get_tool_registry(self, config: AgentConfig) -> ToolRegistry:
        """
        Get or create a tool registry for the agent type.
        
        Args:
            config: Agent configuration
            
        Returns:
            ToolRegistry instance
        """
        if config.agent_type in self._tool_registries:
            return self._tool_registries[config.agent_type]
        
        # Get global registry for agent type
        registry = global_tool_registry.get_registry(config.agent_type)
        
        # Load tools if not already loaded
        if registry.tool_count == 0:
            tools = self._get_tools_for_config(config)
            for tool in tools:
                tool_name = tool.name if hasattr(tool, 'name') else tool.__name__
                try:
                    registry.register(
                        tool_name,
                        tool,
                        metadata={
                            "description": tool.description if hasattr(tool, 'description') else "",
                            "agent_type": config.agent_type,
                            "category": "agent",
                        }
                    )
                except ValueError:
                    pass  # Already registered
        
        self._tool_registries[config.agent_type] = registry
        return registry
    
    # =========================================================================
    # Checkpointer Management
    # =========================================================================
    
    def get_checkpointer(self, config: AgentConfig) -> Any:
        """
        Get or create a checkpointer for state persistence.
        
        Args:
            config: Agent configuration
            
        Returns:
            Checkpointer instance
        """
        checkpointer_key = f"{config.checkpointer_type}:{config.agent_id}"
        
        if checkpointer_key in self._checkpointers:
            return self._checkpointers[checkpointer_key]
        
        checkpointer = self._create_checkpointer(config)
        self._checkpointers[checkpointer_key] = checkpointer
        
        logger.info(f"Created checkpointer: {checkpointer_key}")
        return checkpointer
    
    def _create_checkpointer(self, config: AgentConfig) -> Any:
        """
        Create a checkpointer based on config.
        
        Args:
            config: Agent configuration
            
        Returns:
            Checkpointer instance
        """
        from langgraph.checkpoint.memory import MemorySaver
        
        checkpointer_type = config.checkpointer_type
        
        if checkpointer_type == "memory":
            return MemorySaver()
        
        elif checkpointer_type == "postgres":
            return self._create_postgres_checkpointer(config)
        
        elif checkpointer_type == "redis":
            return self._create_redis_checkpointer(config)
        
        # Default to memory
        return MemorySaver()
    
    def _create_postgres_checkpointer(self, config: AgentConfig) -> Any:
        """Create a Postgres checkpointer."""
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
            import psycopg2
            
            # Get connection details from config or environment
            conn_config = config.checkpointer_config or {}
            db_url = conn_config.get(
                "database_url",
                f"postgresql://{config.custom.get('db_user', 'postgres')}:"
                f"{config.custom.get('db_password', 'postgres')}@"
                f"{config.custom.get('db_host', 'localhost')}:"
                f"{config.custom.get('db_port', 5432)}/"
                f"{config.custom.get('db_name', 'quantmind')}"
            )
            
            # Create connection pool
            conn = psycopg2.connect(db_url)
            return PostgresSaver(conn)
            
        except Exception as e:
            logger.warning(f"Failed to create Postgres checkpointer: {e}, using MemorySaver")
            return MemorySaver()
    
    def _create_redis_checkpointer(self, config: AgentConfig) -> Any:
        """Create a Redis checkpointer."""
        try:
            from langgraph.checkpoint.redis import RedisSaver
            import redis
            
            # Get connection details from config or environment
            conn_config = config.checkpointer_config or {}
            redis_url = conn_config.get(
                "redis_url",
                f"redis://{config.custom.get('redis_host', 'localhost')}:"
                f"{config.custom.get('redis_port', 6379)}/0"
            )
            
            # Create Redis client
            client = redis.from_url(redis_url)
            return RedisSaver(client)
            
        except Exception as e:
            logger.warning(f"Failed to create Redis checkpointer: {e}, using MemorySaver")
            return MemorySaver()
    
    # =========================================================================
    # Metrics Collector Management
    # =========================================================================
    
    def get_metrics_collector(self, config: AgentConfig) -> Any:
        """
        Get or create a metrics collector for the agent.
        
        Args:
            config: Agent configuration
            
        Returns:
            MetricsCollector instance
        """
        if config.agent_id in self._metrics_collectors:
            return self._metrics_collectors[config.agent_id]
        
        # Import here to avoid circular dependencies
        from src.agents.metrics_collector import MetricsCollector
        
        collector = MetricsCollector(
            agent_id=config.agent_id,
            agent_type=config.agent_type
        )
        
        self._metrics_collectors[config.agent_id] = collector
        logger.info(f"Created metrics collector for: {config.agent_id}")
        
        return collector
    
    # =========================================================================
    # Observer Management
    # =========================================================================
    
    def add_observer(self, observer: Any) -> None:
        """
        Add an observer to the container.
        
        Args:
            observer: AgentObserver instance
        """
        if observer not in self._observers:
            self._observers.append(observer)
            logger.debug(f"Added observer: {observer.__class__.__name__}")
    
    def remove_observer(self, observer: Any) -> None:
        """
        Remove an observer from the container.
        
        Args:
            observer: AgentObserver instance
        """
        if observer in self._observers:
            self._observers.remove(observer)
            logger.debug(f"Removed observer: {observer.__class__.__name__}")
    
    def get_observers(self) -> List[Any]:
        """
        Get all registered observers.
        
        Returns:
            List of observers
        """
        return self._observers.copy()
    
    # =========================================================================
    # Resource Management
    # =========================================================================
    
    def set_db_pool(self, pool: Any) -> None:
        """Set database connection pool."""
        self._resources.db_pool = pool
    
    def set_redis_pool(self, pool: Any) -> None:
        """Set Redis connection pool."""
        self._resources.redis_pool = pool
    
    def cleanup(self) -> None:
        """Clean up all resources."""
        # Clean up checkpointers
        for checkpointer in self._checkpointers.values():
            if hasattr(checkpointer, 'close'):
                try:
                    checkpointer.close()
                except Exception as e:
                    logger.warning(f"Error closing checkpointer: {e}")
        
        self._checkpointers.clear()
        
        # Clear caches
        self._llm_providers.clear()
        self._tool_registries.clear()
        self._metrics_collectors.clear()
        
        logger.info("DependencyContainer cleaned up")
    
    # =========================================================================
    # Configuration
    # =========================================================================
    
    def set_config(self, key: str, value: Any) -> None:
        """Set a configuration value."""
        self._config[key] = value
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return self._config.get(key, default)


# Global container instance
_container: Optional[DependencyContainer] = None


def get_container() -> DependencyContainer:
    """
    Get the global dependency container.
    
    Returns:
        DependencyContainer instance
    """
    global _container
    if _container is None:
        _container = DependencyContainer()
    return _container


def set_container(container: DependencyContainer) -> None:
    """
    Set the global dependency container.
    
    Args:
        container: DependencyContainer instance
    """
    global _container
    _container = container

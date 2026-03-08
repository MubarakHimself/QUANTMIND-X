"""
DEPRECATED: This module is deprecated.

Use src.agents.departments.tool_registry instead.
The department tool registry provides better integration with the department system.

---

Tool Registry for Agent Tools

Provides centralized tool registration, retrieval, and metadata management
for factory-created agents.

**Validates: Phase 1.2 - Tool Registry**
"""

import logging
from typing import Dict, Any, List, Callable, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ToolMetadata:
    """Metadata for a registered tool."""
    name: str
    description: str
    agent_type: str
    category: str = "general"
    parameters: Dict[str, Any] = field(default_factory=dict)
    is_async: bool = False
    requires_auth: bool = False
    tags: List[str] = field(default_factory=list)
    registered_at: datetime = field(default_factory=datetime.utcnow)


class ToolRegistry:
    """
    Central registry for agent tools.
    
    Provides tool registration, retrieval, and metadata management.
    Supports lazy loading and caching.
    """
    
    def __init__(self, agent_type: str = None):
        """
        Initialize the tool registry.
        
        Args:
            agent_type: Optional agent type to scope this registry
        """
        self._tools: Dict[str, Callable] = {}
        self._tool_metadata: Dict[str, ToolMetadata] = {}
        self._agent_type = agent_type
        self._lazy_loaders: Dict[str, Callable] = {}
        self._is_initialized: bool = False
    
    def register(
        self,
        name: str,
        tool: Callable,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register a tool with optional metadata.
        
        Args:
            name: Tool name (must be unique in this registry)
            tool: Tool function/callable
            metadata: Optional metadata dictionary
            
        Raises:
            ValueError: If tool name is already registered
        """
        if name in self._tools:
            raise ValueError(f"Tool '{name}' already registered")
        
        self._tools[name] = tool
        
        # Create metadata if provided
        if metadata:
            self._tool_metadata[name] = ToolMetadata(
                name=name,
                description=metadata.get("description", ""),
                agent_type=metadata.get("agent_type", self._agent_type or "unknown"),
                category=metadata.get("category", "general"),
                parameters=metadata.get("parameters", {}),
                is_async=metadata.get("is_async", False),
                requires_auth=metadata.get("requires_auth", False),
                tags=metadata.get("tags", []),
            )
        
        logger.debug(f"Registered tool: {name}")
    
    def register_lazy(self, name: str, loader: Callable) -> None:
        """
        Register a lazy loader for a tool.
        
        The tool will only be loaded when first accessed.
        
        Args:
            name: Tool name
            loader: Function that returns the tool when called
        """
        self._lazy_loaders[name] = loader
        logger.debug(f"Registered lazy loader for tool: {name}")
    
    def get(self, name: str) -> Optional[Callable]:
        """
        Retrieve a tool by name.
        
        Args:
            name: Tool name
            
        Returns:
            Tool callable or None if not found
        """
        # Check if already loaded
        if name in self._tools:
            return self._tools[name]
        
        # Check if lazy loader exists
        if name in self._lazy_loaders:
            tool = self._lazy_loaders[name]()
            self._tools[name] = tool
            del self._lazy_loaders[name]
            logger.debug(f"Lazy loaded tool: {name}")
            return tool
        
        return None
    
    def get_all(self) -> List[Callable]:
        """
        Get all registered tools.
        
        Returns:
            List of all tool callables
        """
        # Load any remaining lazy tools
        for name in list(self._lazy_loaders.keys()):
            self.get(name)
        
        return list(self._tools.values())
    
    def get_metadata(self, name: str) -> Optional[ToolMetadata]:
        """
        Get metadata for a tool.
        
        Args:
            name: Tool name
            
        Returns:
            ToolMetadata or None if not found
        """
        return self._tool_metadata.get(name)
    
    def list_tools(self) -> List[str]:
        """
        List all registered tool names.
        
        Returns:
            List of tool names
        """
        # Include lazy loaders in the list
        return list(self._tools.keys()) + list(self._lazy_loaders.keys())
    
    def list_tools_by_category(self, category: str) -> List[str]:
        """
        List tools filtered by category.
        
        Args:
            category: Category to filter by
            
        Returns:
            List of tool names in the category
        """
        return [
            name for name, meta in self._tool_metadata.items()
            if meta.category == category
        ]
    
    def list_tools_by_tag(self, tag: str) -> List[str]:
        """
        List tools filtered by tag.
        
        Args:
            tag: Tag to filter by
            
        Returns:
            List of tool names with the tag
        """
        return [
            name for name, meta in self._tool_metadata.items()
            if tag in meta.tags
        ]
    
    def unregister(self, name: str) -> bool:
        """
        Unregister a tool.
        
        Args:
            name: Tool name
            
        Returns:
            True if tool was unregistered, False if not found
        """
        if name in self._tools:
            del self._tools[name]
            if name in self._tool_metadata:
                del self._tool_metadata[name]
            logger.debug(f"Unregistered tool: {name}")
            return True
        
        if name in self._lazy_loaders:
            del self._lazy_loaders[name]
            logger.debug(f"Removed lazy loader for tool: {name}")
            return True
        
        return False
    
    def clear(self) -> None:
        """Clear all registered tools and metadata."""
        self._tools.clear()
        self._tool_metadata.clear()
        self._lazy_loaders.clear()
        self._is_initialized = False
        logger.debug("Cleared tool registry")
    
    @property
    def tool_count(self) -> int:
        """Get the count of registered tools."""
        return len(self._tools) + len(self._lazy_loaders)
    
    def get_tools_dict(self) -> Dict[str, Callable]:
        """
        Get all tools as a dictionary.
        
        Returns:
            Dictionary of tool name to tool callable
        """
        # Ensure lazy tools are loaded
        self.get_all()
        return self._tools.copy()


class GlobalToolRegistry:
    """
    Global registry that manages multiple agent-specific registries.
    
    Provides a singleton-like interface for tool management.
    """
    
    _instance: Optional["GlobalToolRegistry"] = None
    _registries: Dict[str, ToolRegistry] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._registries = {}
            self._initialized = True
    
    def get_registry(self, agent_type: str = None) -> ToolRegistry:
        """
        Get or create a registry for an agent type.
        
        Args:
            agent_type: Agent type (analyst, quantcode, copilot, router)
            
        Returns:
            ToolRegistry for the agent type
        """
        key = agent_type or "default"
        
        if key not in self._registries:
            self._registries[key] = ToolRegistry(agent_type=key)
            logger.info(f"Created new tool registry for: {key}")
        
        return self._registries[key]
    
    def register_tool(
        self,
        agent_type: str,
        name: str,
        tool: Callable,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register a tool in a specific agent's registry.
        
        Args:
            agent_type: Agent type
            name: Tool name
            tool: Tool callable
            metadata: Optional metadata
        """
        registry = self.get_registry(agent_type)
        registry.register(name, tool, metadata)
    
    def get_tools_for_agent(self, agent_type: str) -> List[Callable]:
        """
        Get all tools for a specific agent type.
        
        Args:
            agent_type: Agent type
            
        Returns:
            List of tool callables
        """
        registry = self.get_registry(agent_type)
        return registry.get_all()
    
    def list_all_tools(self) -> Dict[str, List[str]]:
        """
        List all tools across all registries.
        
        Returns:
            Dictionary mapping agent type to list of tool names
        """
        return {
            agent_type: registry.list_tools()
            for agent_type, registry in self._registries.items()
        }
    
    def clear_all(self) -> None:
        """Clear all registries."""
        for registry in self._registries.values():
            registry.clear()
        self._registries.clear()
        logger.info("Cleared all tool registries")


# Global instance
global_tool_registry = GlobalToolRegistry()


def get_tool_registry(agent_type: str = None) -> ToolRegistry:
    """
    Convenience function to get a tool registry.
    
    Args:
        agent_type: Optional agent type
        
    Returns:
        ToolRegistry instance
    """
    return global_tool_registry.get_registry(agent_type)

"""
Unified Tool Registry for QuantMind agents.

This module provides a centralized registry for all tools,
mirroring the frontend skill registry system with:
- Tool registration and discovery
- Categorization by agent type
- Enable/disable functionality
- Metadata and schema management
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union

from .base import QuantMindTool, ToolCategory, ToolPriority


logger = logging.getLogger(__name__)


class AgentType(str, Enum):
    """Types of agents in the QuantMind system."""
    COPILOT = "copilot"
    ANALYST = "analyst"
    QUANTCODE = "quantcode"
    ALL = "all"


@dataclass
class ToolMetadata:
    """Metadata for a registered tool."""
    name: str
    description: str
    category: ToolCategory
    priority: ToolPriority = ToolPriority.NORMAL
    version: str = "1.0.0"
    author: str = "QuantMind"
    tags: List[str] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    requirements: List[str] = field(default_factory=list)
    agent_types: List[AgentType] = field(default_factory=lambda: [AgentType.ALL])
    enabled: bool = True
    tool_class: Optional[Type[QuantMindTool]] = None
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "priority": self.priority.value,
            "version": self.version,
            "author": self.author,
            "tags": self.tags,
            "examples": self.examples,
            "requirements": self.requirements,
            "agent_types": [a.value for a in self.agent_types],
            "enabled": self.enabled,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
        }


class ToolRegistry:
    """
    Centralized registry for all QuantMind tools.

    Features:
    - Register tools with metadata
    - Discover tools by category, agent type, or tags
    - Enable/disable tools dynamically
    - Export tool schemas for frontend integration
    """

    def __init__(self):
        self._tools: Dict[str, QuantMindTool] = {}
        self._metadata: Dict[str, ToolMetadata] = {}
        self._categories: Dict[ToolCategory, Set[str]] = {
            cat: set() for cat in ToolCategory
        }
        self._agent_tools: Dict[AgentType, Set[str]] = {
            agent: set() for agent in AgentType
        }
        self._tags: Dict[str, Set[str]] = {}

    def register(
        self,
        tool: Union[QuantMindTool, Type[QuantMindTool]],
        agent_types: Optional[List[Union[AgentType, str]]] = None,
        tags: Optional[List[str]] = None,
        examples: Optional[List[Dict[str, Any]]] = None,
        requirements: Optional[List[str]] = None,
        enabled: bool = True,
    ) -> None:
        """
        Register a tool in the registry.

        Args:
            tool: Tool instance or class
            agent_types: Which agents can use this tool (default: all)
            tags: Additional tags for discovery
            examples: Usage examples
            requirements: Dependencies required by this tool
            enabled: Whether tool is enabled by default
        """
        # Instantiate if class provided
        if isinstance(tool, type):
            tool = tool()

        name = tool.name

        if name in self._tools:
            logger.warning(f"Tool '{name}' already registered, overwriting")

        # Parse agent types
        parsed_agent_types = []
        if agent_types:
            for at in agent_types:
                if isinstance(at, str):
                    parsed_agent_types.append(AgentType(at.lower()))
                else:
                    parsed_agent_types.append(at)
        else:
            parsed_agent_types = [AgentType.ALL]

        # Create metadata
        metadata = ToolMetadata(
            name=name,
            description=tool.description,
            category=tool.category,
            priority=tool.priority,
            version=tool.version,
            author=tool.author,
            tags=tags or tool.tags,
            examples=examples or tool.examples,
            requirements=requirements or tool.requirements,
            agent_types=parsed_agent_types,
            enabled=enabled,
            tool_class=type(tool),
            input_schema=self._extract_schema(tool),
        )

        # Register
        self._tools[name] = tool
        self._metadata[name] = metadata
        self._categories[tool.category].add(name)

        # Index by agent type
        for agent_type in parsed_agent_types:
            self._agent_tools[agent_type].add(name)

        # Index by tags
        for tag in metadata.tags:
            if tag not in self._tags:
                self._tags[tag] = set()
            self._tags[tag].add(name)

        logger.info(f"Registered tool '{name}' in category '{tool.category.value}'")

    def unregister(self, name: str) -> bool:
        """
        Unregister a tool from the registry.

        Args:
            name: Tool name

        Returns:
            True if tool was unregistered, False if not found
        """
        if name not in self._tools:
            return False

        metadata = self._metadata[name]

        # Remove from all indices
        del self._tools[name]
        del self._metadata[name]
        self._categories[metadata.category].discard(name)

        for agent_type in metadata.agent_types:
            self._agent_tools[agent_type].discard(name)

        for tag in metadata.tags:
            if tag in self._tags:
                self._tags[tag].discard(name)

        logger.info(f"Unregistered tool '{name}'")
        return True

    def get(self, name: str) -> Optional[QuantMindTool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def get_metadata(self, name: str) -> Optional[ToolMetadata]:
        """Get tool metadata by name."""
        return self._metadata.get(name)

    def get_all(self) -> List[QuantMindTool]:
        """Get all registered tools."""
        return list(self._tools.values())

    def get_by_category(self, category: Union[ToolCategory, str]) -> List[QuantMindTool]:
        """Get all tools in a category."""
        if isinstance(category, str):
            category = ToolCategory(category)

        return [
            self._tools[name]
            for name in self._categories.get(category, set())
            if name in self._tools
        ]

    def get_by_agent(self, agent_type: Union[AgentType, str]) -> List[QuantMindTool]:
        """
        Get all tools available to an agent type.

        Includes both agent-specific tools and tools marked for ALL agents.
        """
        if isinstance(agent_type, str):
            agent_type = AgentType(agent_type.lower())

        tool_names = (
            self._agent_tools.get(agent_type, set()) |
            self._agent_tools.get(AgentType.ALL, set())
        )

        return [
            self._tools[name]
            for name in tool_names
            if name in self._tools and self._metadata[name].enabled
        ]

    def get_by_tags(self, tags: List[str], match_all: bool = False) -> List[QuantMindTool]:
        """
        Get tools by tags.

        Args:
            tags: List of tags to match
            match_all: If True, tool must have all tags; if False, any tag matches

        Returns:
            List of matching tools
        """
        if not tags:
            return []

        matching_sets = [
            self._tags.get(tag, set())
            for tag in tags
        ]

        if match_all:
            tool_names = set.intersection(*matching_sets) if matching_sets else set()
        else:
            tool_names = set.union(*matching_sets) if matching_sets else set()

        return [
            self._tools[name]
            for name in tool_names
            if name in self._tools
        ]

    def search(self, query: str) -> List[QuantMindTool]:
        """
        Search tools by name, description, or tags.

        Args:
            query: Search query (case-insensitive)

        Returns:
            List of matching tools
        """
        query_lower = query.lower()
        matches = []

        for name, metadata in self._metadata.items():
            # Search in name
            if query_lower in name.lower():
                matches.append(self._tools[name])
                continue

            # Search in description
            if query_lower in metadata.description.lower():
                matches.append(self._tools[name])
                continue

            # Search in tags
            if any(query_lower in tag.lower() for tag in metadata.tags):
                matches.append(self._tools[name])
                continue

        return matches

    def enable(self, name: str) -> bool:
        """Enable a tool."""
        if name not in self._metadata:
            return False
        self._metadata[name].enabled = True
        logger.info(f"Enabled tool '{name}'")
        return True

    def disable(self, name: str) -> bool:
        """Disable a tool."""
        if name not in self._metadata:
            return False
        self._metadata[name].enabled = False
        logger.info(f"Disabled tool '{name}'")
        return True

    def is_enabled(self, name: str) -> bool:
        """Check if a tool is enabled."""
        metadata = self._metadata.get(name)
        return metadata.enabled if metadata else False

    def get_langchain_tools(
        self,
        agent_type: Optional[Union[AgentType, str]] = None,
        categories: Optional[List[Union[ToolCategory, str]]] = None,
        enabled_only: bool = True,
    ) -> List[QuantMindTool]:
        """
        Get tools formatted for LangChain ToolNode.

        Args:
            agent_type: Filter by agent type
            categories: Filter by categories
            enabled_only: Only return enabled tools

        Returns:
            List of LangChain-compatible tools
        """
        tools = []

        # Get base tool list
        if agent_type:
            tools = self.get_by_agent(agent_type)
        elif categories:
            tools = []
            for cat in categories:
                tools.extend(self.get_by_category(cat))
        else:
            tools = self.get_all()

        # Filter by enabled status
        if enabled_only:
            tools = [t for t in tools if self.is_enabled(t.name)]

        return tools

    def export_schema(self) -> Dict[str, Any]:
        """
        Export all tool schemas for frontend integration.

        Returns:
            Dictionary with all tool metadata and schemas
        """
        return {
            "tools": [m.to_dict() for m in self._metadata.values()],
            "categories": {cat.value: list(names) for cat, names in self._categories.items()},
            "tags": dict(self._tags),
            "version": "1.0.0",
        }

    def import_schema(self, schema: Dict[str, Any]) -> None:
        """
        Import tool schemas (for loading from config).

        Args:
            schema: Schema dictionary from export_schema()
        """
        # This would be used to load tool configurations from a file
        # Actual tool instances would still need to be registered via register()
        logger.info(f"Importing schema with {len(schema.get('tools', []))} tool definitions")

    def _extract_schema(self, tool: QuantMindTool) -> Optional[Dict[str, Any]]:
        """Extract JSON schema from tool's args_schema."""
        if hasattr(tool, "args_schema") and tool.args_schema:
            return tool.args_schema.model_json_schema()
        return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics."""
        total_tools = len(self._tools)
        enabled_tools = sum(1 for m in self._metadata.values() if m.enabled)

        category_counts = {
            cat.value: len(names)
            for cat, names in self._categories.items()
            if names
        }

        agent_counts = {
            agent.value: len(names)
            for agent, names in self._agent_tools.items()
            if names and agent != AgentType.ALL
        }

        total_executions = sum(
            t._execution_count
            for t in self._tools.values()
        )

        return {
            "total_tools": total_tools,
            "enabled_tools": enabled_tools,
            "disabled_tools": total_tools - enabled_tools,
            "categories": category_counts,
            "agent_distribution": agent_counts,
            "total_executions": total_executions,
        }


# Global registry instance
tool_registry = ToolRegistry()


def register_tool(
    agent_types: Optional[List[Union[AgentType, str]]] = None,
    tags: Optional[List[str]] = None,
    **kwargs
) -> Callable:
    """
    Decorator to register a tool class.

    Usage:
        @register_tool(agent_types=["copilot"], tags=["file", "read"])
        class ReadFileTool(QuantMindTool):
            ...

    Args:
        agent_types: Which agents can use this tool
        tags: Tags for discovery
        **kwargs: Additional metadata

    Returns:
        Decorator function
    """
    def decorator(cls: Type[QuantMindTool]) -> Type[QuantMindTool]:
        tool_registry.register(cls, agent_types=agent_types, tags=tags, **kwargs)
        return cls

    return decorator

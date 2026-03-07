"""
Tool Registry for department-based tool access.

Central system that manages which tools are available to each department
based on their permissions from tool_access.py.
"""

from typing import Dict, List, Optional, Any
from logging import getLogger

from .types import Department
from .tool_access import ToolAccessController, ToolPermission

logger = getLogger(__name__)


class ToolRegistry:
    """
    Central registry for managing tool access by department.

    Enforces permissions from tool_access.py and provides
    filtered tool lists to departments.
    """

    # Map tool names to their module paths
    TOOL_MODULES = {
        "memory_tools": "src.agents.tools.memory_tools",
        "knowledge_tools": "src.agents.tools.knowledge_tools",
        "pinescript_tools": "src.agents.tools.pinescript_tools",
        "mql5_tools": "src.agents.tools.mql5_tools",
        "backtest_tools": "src.agents.tools.backtest_tools",
        "strategy_router": "src.agents.tools.strategy_router",
        "risk_tools": "src.agents.tools.risk_tools",
        "broker_tools": "src.agents.tools.broker_tools",
        "ea_lifecycle": "src.agents.tools.ea_lifecycle",
        "strategy_extraction": "src.agents.tools.strategy_extraction",
        "gemini_cli": "src.agents.tools.gemini_cli_tool",
        "mail": "src.agents.departments.department_mail",
        "memory_all_depts": "src.agents.departments.memory_access",
    }

    # Cached tool instances
    _tool_instances: Dict[str, Any] = {}

    @classmethod
    def get_tools_for_department(
        cls,
        department: Department,
    ) -> Dict[str, Any]:
        """
        Get all available tools for a department.

        Args:
            department: Department to get tools for

        Returns:
            Dictionary of tool_name to tool_instance
        """
        access_controller = ToolAccessController(department)
        available_tools = access_controller.get_available_tools()

        tools = {}
        for tool_name in available_tools:
            # Only include tools the department can access
            if access_controller.has_read_access(tool_name):
                tool_instance = cls._get_tool_instance(tool_name)
                if tool_instance:
                    tools[tool_name] = tool_instance

        logger.info(f"Provided {len(tools)} tools to {department.value}")
        return tools

    @classmethod
    def get_tool(
        cls,
        tool_name: str,
        department: Department,
    ) -> Optional[Any]:
        """
        Get a specific tool for a department.

        Args:
            tool_name: Name of the tool
            department: Department requesting the tool

        Returns:
            Tool instance or None if not permitted
        """
        access_controller = ToolAccessController(department)

        if not access_controller.has_read_access(tool_name):
            logger.warning(
                f"{department.value} denied access to {tool_name}"
            )
            return None

        return cls._get_tool_instance(tool_name)

    @classmethod
    def _get_tool_instance(cls, tool_name: str) -> Optional[Any]:
        """
        Get or create tool instance.

        Args:
            tool_name: Name of the tool

        Returns:
            Tool instance or None if module not found
        """
        # Return cached instance if available
        if tool_name in cls._tool_instances:
            return cls._tool_instances[tool_name]

        # Get module path
        module_path = cls.TOOL_MODULES.get(tool_name)
        if not module_path:
            logger.warning(f"Tool module not found: {tool_name}")
            return None

        try:
            # Handle special cases with custom module imports
            import importlib

            if tool_name == "gemini_cli":
                # Special case: import the full module path directly
                module = importlib.import_module(module_path)
                class_name = "GeminiCLITool"
            elif tool_name == "mail":
                # Special case: import full module for mail
                module = importlib.import_module(module_path)
                class_name = "DepartmentMailService"
            else:
                # Standard import using namespace package approach
                parts = module_path.split(".")
                module_name = ".".join(parts[:-1])
                module = importlib.import_module(module_name)

                # Get main class from module (convention: class name = CamelCase of tool_name)
                class_name = "".join(
                    word.capitalize() for word in tool_name.split("_")
                )
                # Handle special cases
                if tool_name == "ea_lifecycle":
                    class_name = "EALifecycleTools"

            tool_class = getattr(module, class_name, None)
            if not tool_class:
                logger.warning(f"Tool class not found: {class_name} in {module_path}")
                return None

            # Create and cache instance
            instance = tool_class()
            cls._tool_instances[tool_name] = instance

            logger.info(f"Initialized tool: {tool_name}")
            return instance

        except Exception as e:
            logger.error(f"Failed to load tool {tool_name}: {e}")
            return None

    @classmethod
    def can_use_tool(
        cls,
        tool_name: str,
        department: Department,
        permission: ToolPermission = ToolPermission.READ,
    ) -> bool:
        """
        Check if department can use a tool with specific permission.

        Args:
            tool_name: Name of the tool
            department: Department requesting access
            permission: Permission level required

        Returns:
            True if access is granted
        """
        access_controller = ToolAccessController(department)
        return access_controller.can_access(tool_name, permission)

    @classmethod
    def get_tool_info(cls, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Tool information dictionary
        """
        module_path = cls.TOOL_MODULES.get(tool_name)
        if not module_path:
            return None

        return {
            "name": tool_name,
            "module": module_path,
            "available": module_path is not None,
        }

    @classmethod
    def list_all_tools(cls) -> List[str]:
        """
        List all registered tools.

        Returns:
            List of tool names
        """
        return list(cls.TOOL_MODULES.keys())

    @classmethod
    def clear_cache(cls):
        """Clear cached tool instances."""
        cls._tool_instances.clear()
        logger.info("Tool cache cleared")


def get_tool_registry() -> ToolRegistry:
    """
    Get the singleton tool registry instance.

    Returns:
        ToolRegistry instance
    """
    return ToolRegistry()

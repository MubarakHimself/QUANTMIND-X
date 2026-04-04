"""
Skill-to-Tool Bridge — Wires department skills as agent tools.

Issue #17: Skills registered but NOT wired to agents.

Converts skill definitions from department_skills.py into Claude-compatible
tool definitions that agents can invoke. Each skill becomes a tool that:
1. Loads the skill.md file JIT for context
2. Executes the skill's handler function if available
3. Falls back to returning the skill prompt for the agent to process

Architecture reference: §16.3 Tool Tiers — Department tier
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _build_skill_tool_schema(skill: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a skill definition's parameters to JSON Schema for Claude."""
    properties = {}
    required = []

    for param in skill.get("parameters", []):
        if isinstance(param, str):
            properties[param] = {
                "type": "string",
                "description": f"Parameter: {param}",
            }
        elif isinstance(param, dict):
            properties[param.get("name", "value")] = {
                "type": param.get("type", "string"),
                "description": param.get("description", ""),
            }

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def _create_skill_handler(skill_name: str, skill_module: str):
    """Create a handler function for a skill tool."""
    def handler(args: Dict[str, Any]) -> Dict[str, Any]:
        # Try to call the actual skill function
        try:
            import importlib
            mod = importlib.import_module(skill_module)
            skill_fn = getattr(mod, skill_name, None)
            if skill_fn and callable(skill_fn):
                result = skill_fn(**args)
                return {
                    "skill": skill_name,
                    "status": "executed",
                    "result": result,
                }
        except Exception as e:
            logger.debug(f"Skill module execution failed for {skill_name}: {e}")

        # Fallback: return skill definition for agent to process
        return {
            "skill": skill_name,
            "status": "loaded",
            "message": (
                f"Skill '{skill_name}' loaded. Process the request using "
                f"your knowledge and the provided parameters."
            ),
            "parameters_received": args,
        }

    return handler


def get_department_skill_tools(department: str) -> List[Dict[str, Any]]:
    """
    Get Claude-compatible tool definitions for a department's skills.

    Converts skills from department_skills.py into tool dicts that can be
    registered with BaseAgent or DepartmentHead.

    Args:
        department: Department name (research, development, risk, trading, portfolio)

    Returns:
        List of tool definition dicts with name, description, parameters, handler
    """
    try:
        from src.agents.skills.department_skills import get_department_skills
        skills = get_department_skills(department)
    except ImportError:
        logger.debug(f"Department skills not available for {department}")
        return []

    tools = []
    for skill in skills:
        skill_name = skill.get("name", "")
        if not skill_name:
            continue

        tool_def = {
            "name": f"skill_{skill_name}",
            "description": skill.get("description", f"Skill: {skill_name}"),
            "parameters": _build_skill_tool_schema(skill),
            "handler": _create_skill_handler(
                skill_name, skill.get("module", ""),
            ),
            "read_only": False,
            "category": "department_skill",
            "slash_command": skill.get("slash_command", ""),
        }
        tools.append(tool_def)

    logger.debug(
        f"Built {len(tools)} skill tools for {department}: "
        f"{[t['name'] for t in tools]}"
    )
    return tools


def register_skill_tools_on_agent(agent, department: str) -> int:
    """
    Register department skill tools directly on a BaseAgent instance.

    Args:
        agent: BaseAgent or DepartmentHead instance
        department: Department name

    Returns:
        Number of tools registered
    """
    tools = get_department_skill_tools(department)
    registered = 0

    for tool_def in tools:
        try:
            # Import ToolDefinition if available
            from src.agents.core.base_agent import ToolDefinition
            td = ToolDefinition(
                name=tool_def["name"],
                description=tool_def["description"],
                parameters=tool_def["parameters"],
                handler=tool_def["handler"],
                read_only=tool_def.get("read_only", False),
            )
            if hasattr(agent, 'tools') and isinstance(agent.tools, dict):
                if td.name not in agent.tools:
                    agent.tools[td.name] = td
                    registered += 1
            elif hasattr(agent, 'register_tool'):
                agent.register_tool(td)
                registered += 1
        except Exception as e:
            logger.debug(f"Skill tool registration failed: {tool_def['name']}: {e}")

    if registered > 0:
        logger.info(
            f"Registered {registered} skill tools for {department} "
            f"on agent {getattr(agent, 'name', 'unknown')}"
        )

    return registered


def get_skill_tool_definitions_for_claude(department: str) -> List[Dict[str, Any]]:
    """
    Get Claude API-compatible tool definitions (without handlers) for a department.

    These are suitable for passing to the Claude Messages API tools parameter.

    Args:
        department: Department name

    Returns:
        List of {name, description, input_schema} dicts for Claude API
    """
    tools = get_department_skill_tools(department)
    claude_tools = []

    for tool in tools:
        claude_tools.append({
            "name": tool["name"],
            "description": tool["description"],
            "input_schema": tool["parameters"],
        })

    return claude_tools

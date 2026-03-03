"""
Sequential Thinking MCP Tools Module.

Provides tools for task decomposition and error analysis via Sequential Thinking MCP server.
"""

import logging
from typing import Dict, Any, Optional, List

from src.agents.tools.mcp.manager import get_mcp_manager

logger = logging.getLogger(__name__)


async def sequential_thinking(
    task: str,
    context: Optional[str] = None,
    max_steps: int = 10
) -> Dict[str, Any]:
    """
    Decompose a complex task into sequential steps.

    This tool uses the Sequential Thinking MCP server to break down
    complex tasks into manageable steps with reasoning.

    Args:
        task: Task description to decompose
        context: Optional context for the task
        max_steps: Maximum number of steps to generate

    Returns:
        Dictionary containing:
        - steps: List of sequential steps
        - reasoning: Overall reasoning
        - estimated_complexity: Complexity assessment
    """
    logger.info(f"Sequential thinking for task: {task[:100]}...")

    manager = await get_mcp_manager()

    try:
        result = await manager.call_tool(
            "sequential_thinking",
            "decompose",
            {
                "task": task,
                "context": context,
                "max_steps": max_steps
            }
        )

        if isinstance(result, dict):
            return {
                "steps": result.get("steps", []),
                "reasoning": result.get("reasoning", ""),
                "estimated_complexity": result.get("complexity", "medium"),
                "total_steps": len(result.get("steps", []))
            }
        return {
            "steps": [],
            "reasoning": "",
            "estimated_complexity": "unknown",
            "total_steps": 0
        }

    except Exception as e:
        logger.error(f"Sequential thinking failed: {e}")
        raise RuntimeError(f"Failed to decompose task: {e}")


async def analyze_errors(
    errors: List[str],
    code: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze errors and suggest fixes using sequential thinking.

    Args:
        errors: List of error messages
        code: Optional code that caused the errors

    Returns:
        Dictionary containing error analysis and suggested fixes
    """
    logger.info(f"Analyzing {len(errors)} errors")

    manager = await get_mcp_manager()

    try:
        result = await manager.call_tool(
            "sequential_thinking",
            "analyze-errors",
            {
                "errors": errors,
                "code": code
            }
        )

        if isinstance(result, dict):
            return {
                "analysis": result.get("analysis", []),
                "overall_assessment": result.get("assessment", "Errors require attention"),
                "suggested_approach": result.get("approach", "Fix errors in order of priority")
            }
        return {
            "analysis": [],
            "overall_assessment": "Analysis failed",
            "suggested_approach": "Manual review required"
        }

    except Exception as e:
        logger.error(f"Error analysis failed: {e}")
        raise RuntimeError(f"Failed to analyze errors: {e}")

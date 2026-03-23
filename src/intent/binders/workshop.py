"""Workshop canvas binder.

Story 5.7: NL System Commands & Context-Aware Canvas Binding
"""
import logging
from typing import Any, Dict, List

from src.intent.binders.base import CanvasBinder

logger = logging.getLogger(__name__)


class WorkshopBinder(CanvasBinder):
    """Binder for Workshop canvas context."""

    async def bind_context(
        self,
        query: str,
        canvas_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Bind workshop context to query.

        Returns agent tasks, skills, memory info.
        """
        try:
            tasks = await self._get_tasks()
            skills = await self._get_skills()
            memory = await self._get_memory()

            return {
                "canvas_type": "workshop",
                "tasks": tasks,
                "skills": skills,
                "memory": memory,
                "query": query,
            }
        except Exception as e:
            logger.warning(f"Failed to bind workshop context: {e}")
            return {"canvas_type": "workshop", "query": query}

    async def get_positions(self) -> List[Dict[str, Any]]:
        """Workshop canvas doesn't track positions."""
        return []

    async def _get_tasks(self) -> List[Dict[str, Any]]:
        """Get active agent tasks."""
        # TODO: Implement actual API call
        logger.debug("Fetching agent tasks")
        return []

    async def _get_skills(self) -> List[Dict[str, Any]]:
        """Get available skills."""
        # TODO: Implement actual API call
        logger.debug("Fetching skills")
        return []

    async def _get_memory(self) -> Dict[str, Any]:
        """Get memory/graph info."""
        # TODO: Implement actual API call
        logger.debug("Fetching memory info")
        return {"namespaces": [], "entries": 0}

    async def get_regime(self) -> str:
        """Workshop doesn't have regime."""
        return "N/A"

    async def get_account_info(self) -> Dict[str, Any]:
        """Workshop doesn't have account info."""
        return {}

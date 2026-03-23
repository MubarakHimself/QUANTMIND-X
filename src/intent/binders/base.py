"""Base canvas binder class.

Story 5.7: NL System Commands & Context-Aware Canvas Binding
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class CanvasBinder(ABC):
    """
    Base class for canvas-specific context binding.

    Each canvas (Live Trading, Risk, Portfolio, Workshop) has a binder
    that fetches current data to enhance query understanding.
    """

    @abstractmethod
    async def bind_context(
        self,
        query: str,
        canvas_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Bind canvas-specific data to a query.

        Args:
            query: User query
            canvas_context: Canvas context dictionary

        Returns:
            Dictionary with canvas-specific context data
        """
        pass

    @abstractmethod
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions for this canvas."""
        pass

    @abstractmethod
    async def get_regime(self) -> str:
        """Get current market regime."""
        pass

    @abstractmethod
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information."""
        pass

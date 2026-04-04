"""
Base Indicator Class

Abstract base class for all SVSS indicators.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class IndicatorResult:
    """
    Result from an indicator computation.

    Attributes:
        name: Indicator name (e.g., 'vwap', 'rvvol')
        value: Computed indicator value
        timestamp: When the value was computed
        session_id: Current session identifier
        metadata: Additional indicator-specific data
    """

    name: str
    value: float
    timestamp: datetime
    session_id: str
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> dict:
        """Convert to dictionary for Redis publishing."""
        return {
            "symbol": self.metadata.get("symbol", ""),
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "session_id": self.session_id,
        }


class BaseIndicator(ABC):
    """
    Abstract base class for all SVSS indicators.

    All indicators must implement:
    - compute(): Calculate indicator value from tick data
    - reset(): Reset accumulators on session boundary
    - name: Indicator name for channel naming
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return indicator name (e.g., 'vwap', 'rvvol')."""
        pass

    @abstractmethod
    def compute(self, tick) -> IndicatorResult:
        """
        Compute indicator value from tick data.

        Args:
            tick: TickData instance

        Returns:
            IndicatorResult with computed value
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset indicator accumulators (called on session boundary)."""
        pass

    def get_value(self) -> Optional[float]:
        """
        Get current indicator value without computing.

        Returns:
            Current value if available, None otherwise.
        """
        return None

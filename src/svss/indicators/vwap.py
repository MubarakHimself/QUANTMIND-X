"""
VWAP Indicator Module

Volume-Weighted Average Price computation for the current session.

VWAP = sum(price_i * volume_i for i in session) / sum(volume_i for i in session)
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from svss.indicators.base import BaseIndicator, IndicatorResult

logger = logging.getLogger(__name__)


class VWAPIndicator(BaseIndicator):
    """
    VWAP (Volume-Weighted Average Price) indicator.

    Computes the volume-weighted average price for the current trading session.
    Reset on session boundary.
    """

    def __init__(self, symbol: str, session_id: str):
        """
        Initialize VWAP indicator.

        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            session_id: Current session identifier
        """
        self._symbol = symbol
        self._session_id = session_id
        self._cumulative_price_volume: float = 0.0
        self._cumulative_volume: float = 0.0
        self._last_value: Optional[float] = None

    @property
    def name(self) -> str:
        return "vwap"

    def compute(self, tick) -> IndicatorResult:
        """
        Compute VWAP from tick data.

        Args:
            tick: TickData instance with bid, ask, last, volume

        Returns:
            IndicatorResult with VWAP value
        """
        # Use last price as the price for VWAP calculation
        price = tick.last
        volume = tick.volume

        if volume > 0:
            self._cumulative_price_volume += price * volume
            self._cumulative_volume += volume

            if self._cumulative_volume > 0:
                self._last_value = self._cumulative_price_volume / self._cumulative_volume
            else:
                self._last_value = price  # Fallback to price if no volume

        return IndicatorResult(
            name=self.name,
            value=self._last_value or 0.0,
            timestamp=tick.timestamp,
            session_id=self._session_id,
            metadata={"symbol": self._symbol},
        )

    def reset(self) -> None:
        """Reset VWAP accumulators on session boundary."""
        self._cumulative_price_volume = 0.0
        self._cumulative_volume = 0.0
        self._last_value = None
        logger.debug(f"VWAP reset for {self._symbol}")

    def get_value(self) -> Optional[float]:
        """Get current VWAP value."""
        return self._last_value

    @property
    def cumulative_price_volume(self) -> float:
        """Get cumulative price * volume (for debugging/testing)."""
        return self._cumulative_price_volume

    @property
    def cumulative_volume(self) -> float:
        """Get cumulative volume (for debugging/testing)."""
        return self._cumulative_volume

"""
RVOL Indicator Module

Relative Volume indicator: current bar volume / 20-session rolling average volume at this time-of-day.

RVOL = current_bar_volume / rolling_avg_volume_at_time_of_day

Where rolling_avg_volume_at_time_of_day =
    average volume at same minute-of-day across last 20 sessions
"""

import logging
from datetime import datetime, timezone
from typing import Optional, Dict

from svss.indicators.base import BaseIndicator, IndicatorResult

logger = logging.getLogger(__name__)

# Default RVOL when no historical data available
DEFAULT_RVOL = 1.0


class RVOLIndicator(BaseIndicator):
    """
    RVOL (Relative Volume) indicator.

    Computes current bar volume relative to the rolling 20-session average
    volume at the same time of day.
    """

    def __init__(self, symbol: str, session_id: str, rolling_avg_volume_profile: Dict[int, float] = None):
        """
        Initialize RVOL indicator.

        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            session_id: Current session identifier
            rolling_avg_volume_profile: Dict mapping minute-of-day (0-1439) to average volume
        """
        self._symbol = symbol
        self._session_id = session_id
        self._rolling_avg_profile: Dict[int, float] = rolling_avg_volume_profile or {}
        self._current_bar_volume: float = 0.0
        self._last_value: Optional[float] = None
        self._current_minute: Optional[int] = None  # Track current minute for bar detection

    @property
    def name(self) -> str:
        return "rvvol"

    def set_rolling_avg_profile(self, profile: Dict[int, float]) -> None:
        """
        Set the rolling average volume profile.

        Args:
            profile: Dict mapping minute-of-day (0-1439) to average volume
        """
        self._rolling_avg_profile = profile
        logger.debug(f"RVOL rolling avg profile updated with {len(profile)} buckets")

    def _get_minute_of_day(self, timestamp: datetime) -> int:
        """Get minute of day (0-1439) from timestamp."""
        return timestamp.hour * 60 + timestamp.minute

    def compute(self, tick) -> IndicatorResult:
        """
        Compute RVOL from tick data.

        Args:
            tick: TickData instance with volume

        Returns:
            IndicatorResult with RVOL value
        """
        minute_of_day = self._get_minute_of_day(tick.timestamp)

        # Detect bar boundary: reset volume when minute changes
        if self._current_minute is not None and self._current_minute != minute_of_day:
            # New bar started - reset volume accumulation
            self._current_bar_volume = 0.0
            logger.debug(f"RVOL bar reset for {self._symbol} at minute {minute_of_day}")

        self._current_minute = minute_of_day

        # Accumulate volume for current bar
        self._current_bar_volume += tick.volume

        # Get rolling average for this minute of day
        rolling_avg = self._rolling_avg_profile.get(minute_of_day, 0)

        if rolling_avg > 0 and self._current_bar_volume > 0:
            self._last_value = self._current_bar_volume / rolling_avg
        else:
            # No historical data or zero volume - use default
            self._last_value = DEFAULT_RVOL

        return IndicatorResult(
            name=self.name,
            value=self._last_value,
            timestamp=tick.timestamp,
            session_id=self._session_id,
            metadata={
                "symbol": self._symbol,
                "current_volume": self._current_bar_volume,
                "rolling_avg": rolling_avg,
            },
        )

    def reset(self, new_session_id: str = None) -> None:
        """
        Reset RVOL accumulators on session/bar boundary.

        Args:
            new_session_id: New session ID if session changed
        """
        if new_session_id:
            self._session_id = new_session_id
        self._current_bar_volume = 0.0
        self._current_minute = None
        self._last_value = None
        logger.debug(f"RVOL reset for {self._symbol}, new session: {new_session_id}")

    def get_value(self) -> Optional[float]:
        """Get current RVOL value."""
        return self._last_value

    @property
    def current_bar_volume(self) -> float:
        """Get current bar volume (for debugging/testing)."""
        return self._current_bar_volume

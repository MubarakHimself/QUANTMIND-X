"""
Volume Profile Indicator Module

Tracks price distribution by volume for the session and identifies POC (Point of Control).

Volume Profile buckets tick prices into price levels and accumulates volume per level.
POC is the price level with the highest volume.
"""

import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional, Dict, Tuple

from svss.indicators.base import BaseIndicator, IndicatorResult

logger = logging.getLogger(__name__)

# Default price bucket size in points (e.g., 10 pips for forex)
DEFAULT_BUCKET_SIZE = 0.001  # 10 pips for 5-digit forex


class VolumeProfileIndicator(BaseIndicator):
    """
    Volume Profile indicator.

    Tracks volume distribution across price levels for the session.
    Identifies POC (Point of Control) - the price level with highest volume.
    """

    def __init__(self, symbol: str, session_id: str, bucket_size: float = DEFAULT_BUCKET_SIZE):
        """
        Initialize Volume Profile indicator.

        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            session_id: Current session identifier
            bucket_size: Price bucket size for volume distribution
        """
        self._symbol = symbol
        self._session_id = session_id
        self._bucket_size = bucket_size
        self._profile: Dict[float, float] = defaultdict(float)  # price_level -> volume
        self._last_value: Optional[float] = None  # POC price
        self._poc: Optional[float] = None
        self._update_poc()

    @property
    def name(self) -> str:
        return "volume_profile"

    def _bucket_price(self, price: float) -> float:
        """Bucket price to nearest level."""
        return round(price / self._bucket_size) * self._bucket_size

    def _update_poc(self) -> None:
        """Update Point of Control (POC) from current profile."""
        if self._profile:
            self._poc = max(self._profile.keys(), key=lambda p: self._profile[p])
        else:
            self._poc = None

    def compute(self, tick) -> IndicatorResult:
        """
        Compute/Update Volume Profile from tick data.

        Args:
            tick: TickData instance with last price and volume

        Returns:
            IndicatorResult with Volume Profile data
        """
        price_level = self._bucket_price(tick.last)
        volume = tick.volume

        if volume > 0:
            self._profile[price_level] += volume
            self._update_poc()
            self._last_value = self._poc

        # Serialize profile for publishing
        profile_data = {
            "poc": self._poc,
            "levels": {str(k): v for k, v in self._profile.items()},
            "total_levels": len(self._profile),
        }

        return IndicatorResult(
            name=self.name,
            value=self._poc or 0.0,  # Use POC as primary value
            timestamp=tick.timestamp,
            session_id=self._session_id,
            metadata={
                "symbol": self._symbol,
                "profile": profile_data,
            },
        )

    def reset(self) -> None:
        """Reset Volume Profile accumulators on session boundary."""
        self._profile.clear()
        self._last_value = None
        self._poc = None
        logger.debug(f"Volume Profile reset for {self._symbol}")

    def get_value(self) -> Optional[float]:
        """Get current POC value."""
        return self._poc

    def get_profile(self) -> Dict[float, float]:
        """Get current volume profile (for debugging/testing)."""
        return dict(self._profile)

    @property
    def poc(self) -> Optional[float]:
        """Get Point of Control price level."""
        return self._poc

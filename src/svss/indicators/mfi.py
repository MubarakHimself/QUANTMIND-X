"""
MFI Indicator Module

Money Flow Index (MFI) - 14-period typical price * volume indicator.

MFI = 100 - (100 / (1 + money_flow_ratio))

Where:
- Typical Price = (High + Low + Close) / 3
- Money Flow = Typical Price * Volume
- Positive Money Flow = sum of Money Flow where Typical Price > Previous Typical Price
- Negative Money Flow = sum of Money Flow where Typical Price < Previous Typical Price
- Money Flow Ratio = Positive Money Flow / Negative Money Flow
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from svss.indicators.base import BaseIndicator, IndicatorResult

logger = logging.getLogger(__name__)

# MFI period as specified in requirements
MFI_PERIOD = 14


class MFIIndicator(BaseIndicator):
    """
    MFI (Money Flow Index) indicator.

    14-period money flow index measuring trading pressure.
    Values range from 0-100:
    - MFI > 80: Overbought
    - MFI < 20: Oversold
    """

    def __init__(self, symbol: str, session_id: str, period: int = MFI_PERIOD):
        """
        Initialize MFI indicator.

        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            session_id: Current session identifier
            period: MFI period (default: 14)
        """
        self._symbol = symbol
        self._session_id = session_id
        self._period = period
        self._typical_prices: list[float] = []
        self._money_flows: list[float] = []
        self._positive_flow: float = 0.0
        self._negative_flow: float = 0.0
        self._last_value: Optional[float] = None
        self._prev_typical_price: Optional[float] = None

    @property
    def name(self) -> str:
        return "mfi"

    def _compute_typical_price(self, tick) -> float:
        """
        Compute typical price from tick.

        For tick data, we use last as close, and estimate high/low from bid/ask.
        """
        high = tick.ask  # Use ask as high estimate
        low = tick.bid   # Use bid as low estimate
        close = tick.last
        return (high + low + close) / 3

    def compute(self, tick) -> IndicatorResult:
        """
        Compute MFI from tick data.

        Args:
            tick: TickData instance with bid, ask, last, volume

        Returns:
            IndicatorResult with MFI value
        """
        typical_price = self._compute_typical_price(tick)
        volume = tick.volume

        if volume > 0:
            money_flow = typical_price * volume
            self._money_flows.append(money_flow)

            # Determine if positive or negative flow
            if self._prev_typical_price is not None:
                if typical_price > self._prev_typical_price:
                    self._positive_flow += money_flow
                elif typical_price < self._prev_typical_price:
                    self._negative_flow += money_flow
                # If equal, no flow recorded

            self._prev_typical_price = typical_price
            self._typical_prices.append(typical_price)

            # Keep only last 'period' values
            if len(self._typical_prices) > self._period:
                # Remove oldest values and adjust flows accordingly
                removed_tp = self._typical_prices.pop(0)
                removed_mf = self._money_flows.pop(0)

                # Recalculate flows based on what was removed
                # Find the previous typical price to compare
                if self._typical_prices:
                    prev_tp = self._typical_prices[0] if len(self._typical_prices) > 0 else None
                    if prev_tp is not None:
                        if removed_tp > prev_tp:
                            self._positive_flow -= removed_mf
                        elif removed_tp < prev_tp:
                            self._negative_flow -= removed_mf

            # Compute MFI if we have enough data
            if len(self._typical_prices) >= self._period:
                self._last_value = self._compute_mfi()

        return IndicatorResult(
            name=self.name,
            value=self._last_value or 50.0,  # Default to neutral
            timestamp=tick.timestamp,
            session_id=self._session_id,
            metadata={
                "symbol": self._symbol,
                "typical_price": typical_price,
                "money_flow": money_flow if volume > 0 else 0,
            },
        )

    def _compute_mfi(self) -> float:
        """
        Compute MFI from accumulated flows.

        Returns:
            MFI value (0-100)
        """
        if self._negative_flow == 0:
            # No negative flow - could be overbought
            return 100.0

        money_flow_ratio = self._positive_flow / self._negative_flow
        mfi = 100 - (100 / (1 + money_flow_ratio))

        # Clamp to valid range
        return max(0.0, min(100.0, mfi))

    def reset(self) -> None:
        """Reset MFI accumulators on session boundary."""
        self._typical_prices.clear()
        self._money_flows.clear()
        self._positive_flow = 0.0
        self._negative_flow = 0.0
        self._last_value = None
        self._prev_typical_price = None
        logger.debug(f"MFI reset for {self._symbol}")

    def get_value(self) -> Optional[float]:
        """Get current MFI value."""
        return self._last_value

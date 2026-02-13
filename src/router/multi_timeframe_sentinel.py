"""
Multi-Timeframe Regime Detection System

This module implements multi-timeframe regime detection by aggregating ticks into OHLC bars
and maintaining separate Sentinel instances per timeframe. It enables timeframe-aware bot
selection and multi-timeframe alignment checks in the Commander.
"""

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class Timeframe(Enum):
    """Timeframe enum with conversion methods for multi-timeframe support."""

    M1 = 60      # 1 minute
    M5 = 300     # 5 minutes
    M15 = 900    # 15 minutes
    H1 = 3600    # 1 hour
    H4 = 14400   # 4 hours
    D1 = 86400   # 1 day

    @property
    def seconds(self) -> int:
        """Return the timeframe duration in seconds."""
        return self.value

    @classmethod
    def from_mql5_timeframe(cls, mql5_tf: int) -> 'Timeframe':
        """Convert MQL5 timeframe constant to Timeframe enum."""
        mapping = {
            1: cls.M1,      # PERIOD_M1
            5: cls.M5,      # PERIOD_M5
            15: cls.M15,    # PERIOD_M15
            60: cls.H1,     # PERIOD_H1
            240: cls.H4,    # PERIOD_H4
            1440: cls.D1,   # PERIOD_D1
        }
        return mapping.get(mql5_tf, cls.H1)  # Default to H1

    def to_pandas_freq(self) -> str:
        """Convert to pandas frequency string for compatibility."""
        mapping = {
            Timeframe.M1: '1min',
            Timeframe.M5: '5min',
            Timeframe.M15: '15min',
            Timeframe.H1: '1H',
            Timeframe.H4: '4H',
            Timeframe.D1: '1D',
        }
        return mapping[self]


@dataclass
class OHLCBar:
    """OHLC bar data structure for timeframe aggregation."""

    open: float
    high: float
    low: float
    close: float
    timestamp: datetime  # Start time of the bar
    timeframe: Timeframe
    volume: int = 0

    def __post_init__(self):
        """Ensure timestamp is timezone-aware."""
        if self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=timezone.utc)


class TickAggregator:
    """Aggregates ticks into OHLC bars for a specific timeframe."""

    def __init__(self, timeframe: Timeframe):
        self.timeframe = timeframe
        self.current_bar: Optional[OHLCBar] = None
        self.current_bar_start: Optional[datetime] = None

    def _floor_to_timeframe(self, timestamp: datetime) -> datetime:
        """Floor timestamp to the nearest timeframe boundary."""
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        # Calculate seconds since epoch
        epoch_seconds = int(timestamp.timestamp())

        # Floor to timeframe boundary
        floored_seconds = (epoch_seconds // self.timeframe.seconds) * self.timeframe.seconds

        # Convert back to datetime
        return datetime.fromtimestamp(floored_seconds, tz=timezone.utc)

    def on_tick(self, price: float, timestamp: datetime) -> Optional[OHLCBar]:
        """
        Process a tick and return completed OHLC bar if timeframe boundary crossed.

        Args:
            price: Current price
            timestamp: Tick timestamp

        Returns:
            Completed OHLCBar if timeframe boundary crossed, None otherwise
        """
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        bar_start = self._floor_to_timeframe(timestamp)

        # If this is the first tick or we've crossed a timeframe boundary
        if self.current_bar is None or bar_start != self.current_bar_start:
            # Return the completed bar if we have one
            completed_bar = self.current_bar

            # Start a new bar
            self.current_bar = OHLCBar(
                open=price,
                high=price,
                low=price,
                close=price,
                timestamp=bar_start,
                timeframe=self.timeframe,
                volume=1
            )
            self.current_bar_start = bar_start

            return completed_bar
        else:
            # Update the current bar
            self.current_bar.high = max(self.current_bar.high, price)
            self.current_bar.low = min(self.current_bar.low, price)
            self.current_bar.close = price
            self.current_bar.volume += 1

            return None

    def get_current_bar(self) -> Optional[OHLCBar]:
        """Get the current (potentially incomplete) bar."""
        return self.current_bar


class MultiTimeframeSentinel:
    """
    Multi-timeframe regime detection system that maintains separate Sentinel
    instances per timeframe and aggregates their regime reports.
    """

    def __init__(self, timeframes: List[Timeframe] = None):
        if timeframes is None:
            timeframes = [Timeframe.M5, Timeframe.H1, Timeframe.H4]

        self.timeframes = timeframes

        # Import here to avoid circular imports
        from src.router.sentinel import Sentinel

        # Create aggregators and sentinels for each timeframe
        self.aggregators: Dict[Timeframe, TickAggregator] = {}
        self.sentinels: Dict[Timeframe, Sentinel] = {}
        self.regime_reports: Dict[Timeframe, 'RegimeReport'] = {}

        for timeframe in timeframes:
            self.aggregators[timeframe] = TickAggregator(timeframe)
            self.sentinels[timeframe] = Sentinel()

    def on_tick(self, symbol: str, price: float, timestamp: datetime) -> Dict[Timeframe, 'RegimeReport']:
        """
        Process a tick and update regime detection across all timeframes.

        Args:
            symbol: Trading symbol
            price: Current price
            timestamp: Tick timestamp

        Returns:
            Dictionary of updated regime reports by timeframe
        """
        updated_regimes = {}

        for timeframe in self.timeframes:
            aggregator = self.aggregators[timeframe]
            completed_bar = aggregator.on_tick(price, timestamp)

            if completed_bar is not None:
                # Bar completed, update the sentinel with the close price
                sentinel = self.sentinels[timeframe]
                # Comment 1 fix: Call sentinel.on_tick without timestamp (signature is symbol, price only)
                sentinel.on_tick(symbol, completed_bar.close)

                # Store the updated regime report
                self.regime_reports[timeframe] = sentinel.current_report
                updated_regimes[timeframe] = sentinel.current_report

                logger.debug(f"Bar completed for {timeframe.name}: {completed_bar}")

        return updated_regimes

    def get_regime(self, timeframe: Timeframe) -> Optional['RegimeReport']:
        """Get the current regime report for a specific timeframe."""
        return self.regime_reports.get(timeframe)

    def get_all_regimes(self) -> Dict[Timeframe, 'RegimeReport']:
        """Get current regime reports for all timeframes."""
        return self.regime_reports.copy()

    def get_dominant_regime(self) -> str:
        """
        Get the dominant regime across all timeframes using voting logic.

        Returns:
            Dominant regime string, or highest timeframe's regime if no consensus
        """
        if not self.regime_reports:
            return "UNKNOWN"

        # Count regime votes
        regime_votes = {}
        for report in self.regime_reports.values():
            regime = report.regime
            regime_votes[regime] = regime_votes.get(regime, 0) + 1

        # Find majority regime
        max_votes = max(regime_votes.values())
        majority_regimes = [r for r, v in regime_votes.items() if v == max_votes]

        if len(majority_regimes) == 1:
            return majority_regimes[0]

        # No majority, return highest timeframe's regime
        # Sort timeframes by duration (longer timeframes have higher precedence)
        sorted_timeframes = sorted(self.timeframes, key=lambda tf: tf.seconds, reverse=True)

        for tf in sorted_timeframes:
            if tf in self.regime_reports:
                return self.regime_reports[tf].regime

        return "UNKNOWN"
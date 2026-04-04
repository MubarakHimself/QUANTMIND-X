"""
HMM Lag Buffer Service
=======================

Manages the 3-day calendar lag enforcement for HMM training data.
Ensures live trade outcomes enter the training pool only after a 3-day lag
to prevent look-ahead bias.

Data-level lag enforcement: the model cannot access data until the lag expires.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional, Dict, Any, Callable, Awaitable
import asyncio

logger = logging.getLogger(__name__)


class TradeOutcomeData:
    """
    Trade outcome data container with lag_days property.

    This class provides the lag_days property as specified in AC1:
    "TradeOutcomeData.lag_days property returning 3 (calendar days)"
    """
    lag_days: int = 3  # Calendar days for the 3-day feedback lag


class TradeOutcome(Enum):
    """Trade outcome types."""
    WIN = "WIN"
    LOSS = "LOSS"
    HOLDING = "HOLDING"


@dataclass
class HmmTrainingTradeRecord:
    """
    Record for a trade in the HMM training pool.

    Attributes:
        trade_id: Unique trade identifier
        bot_id: Bot that executed the trade
        close_date: When trade closed
        eligible_date: close_date + 3 calendar days (lag enforced)
        outcome: WIN/LOSS/HOLDING
        pnl: Profit/loss amount
        holding_time_minutes: How long the trade was held
        regime_at_entry: HMM regime when trade was entered
        in_lag_buffer: True if eligible_date > now (still in lag period)
    """
    trade_id: str
    bot_id: str
    close_date: datetime
    eligible_date: datetime
    outcome: TradeOutcome
    pnl: float
    holding_time_minutes: int
    regime_at_entry: str
    in_lag_buffer: bool = True

    @property
    def lag_days_remaining(self) -> int:
        """Returns days remaining in lag buffer (0 if eligible)."""
        delta = self.eligible_date - datetime.now()
        return max(0, delta.days)

    @property
    def is_eligible(self) -> bool:
        """Returns True if lag period has expired."""
        return datetime.now() >= self.eligible_date

    @property
    def time_remaining(self) -> str:
        """Returns human-readable time remaining in lag buffer."""
        if not self.in_lag_buffer:
            return "Eligible"
        days = self.lag_days_remaining
        if days == 0:
            return "< 1 day"
        elif days == 1:
            return "1 day"
        else:
            return f"{days} days"

    @property
    def status(self) -> str:
        """Returns status string for UI display."""
        return "eligible" if not self.in_lag_buffer else "in_buffer"


@dataclass
class HmmBufferStatus:
    """Status of the HMM training buffer."""
    total_in_buffer: int
    total_eligible: int
    trades: List[HmmTrainingTradeRecord]
    avg_lag_remaining: float

    @property
    def total_trades(self) -> int:
        return self.total_in_buffer + self.total_eligible


@dataclass
class BatchResult:
    """Result of a Monday batch inclusion operation."""
    included: int
    trades: List[HmmTrainingTradeRecord]


class HmmLagBuffer:
    """
    Manages the 3-day lag buffer for HMM training data.

    This service enforces the data-level lag - trades are only included
    in the training pool after their 3 calendar-day lag expires.

    The lag is always calculated in calendar days, not trading days,
    to prevent look-ahead bias from weekend effects.
    """

    LAG_CALENDAR_DAYS: int = 3
    # Default pool size threshold to trigger HMM retrain
    POOL_SIZE_THRESHOLD: int = 100

    def __init__(self, pool_size_threshold: int = POOL_SIZE_THRESHOLD):
        """Initialize the lag buffer with in-memory storage.

        Args:
            pool_size_threshold: Minimum number of eligible trades to trigger
                               HMM retrain workflow (default: 100)
        """
        self._trades: Dict[str, HmmTrainingTradeRecord] = {}
        self._pool_size_threshold = pool_size_threshold
        self._retrain_callback: Optional[Callable[[int], Awaitable[None]]] = None
        logger.info(
            f"HmmLagBuffer initialized with 3-day calendar lag enforcement, "
            f"pool_size_threshold={pool_size_threshold}"
        )

    def set_retrain_callback(self, callback: Callable[[int], Awaitable[None]]) -> None:
        """
        Set the callback function to trigger HMM retrain workflow.

        The callback receives the number of newly eligible trades and should
        trigger the HMM retrain workflow when pool size threshold is reached.

        Args:
            callback: Async function that takes trade_count and triggers retrain
        """
        self._retrain_callback = callback
        logger.info("HMM retrain callback registered")

    async def trigger_retrain_if_needed(self, newly_eligible_count: int) -> bool:
        """
        Check if pool size threshold is reached and trigger retrain.

        Called after batch inclusion to check if the eligible pool has grown
        large enough to warrant an HMM retrain.

        Args:
            newly_eligible_count: Number of trades just moved to eligible

        Returns:
            True if retrain was triggered, False otherwise
        """
        if self._retrain_callback is None:
            logger.debug("No retrain callback registered, skipping retrain check")
            return False

        eligible_count = len(self.get_eligible_trades())
        total_count = eligible_count + len(self.get_buffer_trades())

        logger.info(
            f"Pool size check: eligible={eligible_count}, "
            f"total={total_count}, threshold={self._pool_size_threshold}"
        )

        if eligible_count >= self._pool_size_threshold:
            logger.info(
                f"Pool size threshold reached ({eligible_count} >= {self._pool_size_threshold}), "
                f"triggering HMM retrain workflow"
            )
            try:
                await self._retrain_callback(eligible_count)
                return True
            except Exception as e:
                logger.error(f"Retrain callback failed: {e}")
                return False
        return False

    def submit_trade_outcome(
        self,
        trade_id: str,
        bot_id: str,
        close_date: datetime,
        outcome: TradeOutcome,
        pnl: float,
        holding_time_minutes: int,
        regime_at_entry: str
    ) -> HmmTrainingTradeRecord:
        """
        Submit a closed trade to the lag buffer.

        The trade's eligible_date is computed as close_date + 3 calendar days.
        The trade starts in the lag buffer and is not eligible for training
        until the lag expires.

        Args:
            trade_id: Unique trade identifier
            bot_id: Bot that executed the trade
            close_date: When trade closed
            outcome: WIN/LOSS/HOLDING
            pnl: Profit/loss amount
            holding_time_minutes: How long trade was held
            regime_at_entry: HMM regime when trade was entered

        Returns:
            The created HmmTrainingTradeRecord
        """
        eligible_date = close_date + timedelta(days=self.LAG_CALENDAR_DAYS)

        trade_record = HmmTrainingTradeRecord(
            trade_id=trade_id,
            bot_id=bot_id,
            close_date=close_date,
            eligible_date=eligible_date,
            outcome=outcome,
            pnl=pnl,
            holding_time_minutes=holding_time_minutes,
            regime_at_entry=regime_at_entry,
            in_lag_buffer=True  # Trade starts in buffer at submission
        )

        self._trades[trade_id] = trade_record
        logger.info(
            f"Trade {trade_id} submitted to lag buffer: "
            f"close_date={close_date.date()}, eligible_date={eligible_date.date()}"
        )

        return trade_record

    def get_trade(self, trade_id: str) -> Optional[HmmTrainingTradeRecord]:
        """Get a trade record by ID."""
        return self._trades.get(trade_id)

    def get_all_trades(self) -> List[HmmTrainingTradeRecord]:
        """Returns all trades in the buffer."""
        return list(self._trades.values())

    def get_eligible_trades(self) -> List[HmmTrainingTradeRecord]:
        """
        Returns only trades whose lag period has expired.

        This is the primary method for model training - it only returns
        trades that are eligible for inclusion in the training set.
        """
        now = datetime.now()
        return [
            t for t in self._trades.values()
            if t.eligible_date <= now
        ]

    def get_buffer_trades(self) -> List[HmmTrainingTradeRecord]:
        """Returns trades still in the lag buffer (not yet eligible)."""
        return [
            t for t in self._trades.values()
            if t.in_lag_buffer
        ]

    def get_buffer_status(self) -> HmmBufferStatus:
        """
        Returns current buffer state for UX panel.

        Provides aggregate statistics about the buffer including
        counts of eligible vs in-buffer trades and average lag remaining.

        Note: in_buffer vs eligible is computed based on eligible_date,
        not the in_lag_buffer flag (which is only updated by batch processing).
        """
        now = datetime.now()
        all_trades = list(self._trades.values())
        in_buffer = [t for t in all_trades if t.eligible_date > now]
        eligible = [t for t in all_trades if t.eligible_date <= now]

        avg_lag_remaining = 0.0
        if in_buffer:
            lag_values = [t.lag_days_remaining for t in in_buffer]
            avg_lag_remaining = sum(lag_values) / len(lag_values)

        return HmmBufferStatus(
            total_in_buffer=len(in_buffer),
            total_eligible=len(eligible),
            trades=all_trades,
            avg_lag_remaining=avg_lag_remaining
        )

    async def process_monday_batch(self) -> BatchResult:
        """
        Evaluate weekend-expired trades and batch-include them.

        Called on Monday morning to process any trades whose 3-day
        lag expired over the weekend. Saturday and Sunday are counted
        in the lag calculation (calendar days, not trading days).

        After batch inclusion, checks if pool size threshold is reached
        and triggers HMM retrain workflow if needed (AC3 requirement).

        Returns:
            BatchResult with count of included trades
        """
        now = datetime.now()
        expired = [
            t for t in self._trades.values()
            if t.in_lag_buffer and t.eligible_date <= now
        ]

        for trade in expired:
            trade.in_lag_buffer = False
            logger.info(
                f"Trade {trade.trade_id} batch-included: "
                f"eligible_date={trade.eligible_date.date()} reached"
            )

        logger.info(
            f"Monday batch processing: {len(expired)} trades included"
        )

        # Check if we need to trigger HMM retrain (AC3 requirement)
        await self.trigger_retrain_if_needed(len(expired))

        return BatchResult(included=len(expired), trades=expired)

    def _persist(self, trade_record: HmmTrainingTradeRecord) -> None:
        """
        Persist a trade record (in-memory for now, can be extended to DB).

        This is a stub for future database integration.
        """
        self._trades[trade_record.trade_id] = trade_record

    def _update(self, trade_record: HmmTrainingTradeRecord) -> None:
        """Update a trade record in storage."""
        self._trades[trade_record.trade_id] = trade_record

    def _get_all(self) -> List[HmmTrainingTradeRecord]:
        """Returns all trades from storage."""
        return list(self._trades.values())

    async def check_and_process_expired(self) -> BatchResult:
        """
        Check for any expired trades and process them.

        Can be called periodically or on-demand to ensure trades
        are moved from buffer to eligible as soon as their lag expires.
        """
        return await self.process_monday_batch()


# Global singleton instance
_lag_buffer_instance: Optional[HmmLagBuffer] = None


def get_lag_buffer() -> HmmLagBuffer:
    """Get the global HmmLagBuffer singleton instance."""
    global _lag_buffer_instance
    if _lag_buffer_instance is None:
        _lag_buffer_instance = HmmLagBuffer()
    return _lag_buffer_instance

"""
Spread Quality Score (SQS) Engine

Core SQS calculation engine that evaluates spread quality before trade execution.
SQS = historical_avg_spread_at_time_bucket / current_live_spread

Story: 4-7-spread-quality-score-sqs-system
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Literal
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SQSResult(BaseModel):
    """Result of SQS gate evaluation."""
    allowed: bool = Field(..., description="Whether the trade is allowed")
    sqs: float = Field(..., description="Computed SQS value")
    threshold: float = Field(..., description="Applied threshold")
    reason: str = Field(..., description="Reason for allow/block")
    is_hard_block: bool = Field(False, description="True if SQS < 0.50 (hard block)")


class SpreadBucket(BaseModel):
    """Historical spread bucket data."""
    avg_spread: float = Field(..., description="Average spread in this bucket")
    sample_count: int = Field(..., description="Number of samples in bucket")
    updated_at_utc: datetime = Field(..., description="Last update timestamp")


class SQSEngine:
    """
    Spread Quality Score computation engine.

    Computes SQS = historical_avg_spread_at_time_bucket / current_live_spread
    and evaluates against thresholds for scalping and ORB strategies.
    """

    # Threshold constants
    SCALPING_THRESHOLD = 0.75
    ORB_THRESHOLD = 0.80
    HARD_BLOCK_THRESHOLD = 0.50

    # Monday warm-up threshold (first 15 min of market open) — S5-5
    MONDAY_WARMUP_THRESHOLD = 0.60
    MONDAY_WARMUP_MINUTES = 15

    # Minimum samples required for valid bucket
    MIN_SAMPLES_FOR_VALID_BUCKET = 20

    def __init__(self, cache=None, calendar_integration=None):
        """
        Initialize SQS Engine.

        Args:
            cache: SQS cache instance for Redis operations
            calendar_integration: CalendarGovernor integration for news threshold override
        """
        self._cache = cache
        self._calendar = calendar_integration
        logger.info("SQS Engine initialized")

    def is_active_session(self, dt: Optional[datetime] = None) -> bool:
        """
        Check if the given UTC datetime falls within an active trading session.

        S5-5: Weekend guard — no data ingestion on Saturday or Sunday.
        Monday market open uses a relaxed warm-up threshold, not a hard block.

        Args:
            dt: UTC datetime to check. Defaults to now.

        Returns:
            True if within active trading session, False if weekend (block ingestion).
        """
        if dt is None:
            dt = datetime.now(timezone.utc)
        dow = dt.weekday()  # 0=Monday, 6=Sunday
        # Saturday=5, Sunday=6 → block
        if dow >= 5:
            return False
        return True

    def evaluate(
        self,
        symbol: str,
        strategy_type: Literal["scalping", "ORB"],
        current_spread: float,
        historical_buckets: Dict[str, SpreadBucket],
        news_override: Optional[float] = None
    ) -> SQSResult:
        """
        Evaluate SQS for a symbol and strategy type.

        Args:
            symbol: Trading symbol (e.g., "EURUSD")
            strategy_type: "scalping" or "ORB"
            current_spread: Current live spread
            historical_buckets: Dict of bucket_key -> SpreadBucket with historical data
            news_override: Optional threshold modifier from CalendarGovernor

        Returns:
            SQSResult with allow/block decision
        """
        now_utc = datetime.now(timezone.utc)

        # S5-5: Weekend session guard — block weekend data ingestion
        if not self.is_active_session(now_utc):
            logger.info(f"SQS: Weekend — blocking data ingestion for {symbol}")
            return SQSResult(
                allowed=True,  # Allow but flag that no SQS data was used
                sqs=1.0,
                threshold=self.SCALPING_THRESHOLD if strategy_type == "scalping" else self.ORB_THRESHOLD,
                reason="Weekend — no SQS data ingestion (S5-5)",
                is_hard_block=False
            )

        # Get current time bucket
        bucket_key = self._get_bucket_key(now_utc)
        bucket = historical_buckets.get(bucket_key)

        # Fallback if bucket not found - try to aggregate from nearby buckets
        if bucket is None:
            bucket = self._find_valid_bucket(historical_buckets, now_utc)

        # If no valid bucket data, default to allowed per NFR-R1
        if bucket is None or bucket.sample_count < self.MIN_SAMPLES_FOR_VALID_BUCKET:
            logger.warning(
                f"SQS: No valid historical bucket for {symbol} bucket={bucket_key}, "
                f"defaulting to allowed=True"
            )
            return SQSResult(
                allowed=True,
                sqs=1.0,
                threshold=self.SCALPING_THRESHOLD if strategy_type == "scalping" else self.ORB_THRESHOLD,
                reason="No historical spread data available - graceful degradation",
                is_hard_block=False
            )

        # Compute SQS
        if current_spread <= 0:
            logger.warning(f"SQS: Invalid current spread {current_spread} for {symbol}, defaulting to allowed")
            return SQSResult(
                allowed=True,
                sqs=1.0,
                threshold=self.SCALPING_THRESHOLD if strategy_type == "scalping" else self.ORB_THRESHOLD,
                reason="Invalid current spread - graceful degradation",
                is_hard_block=False
            )

        sqs = bucket.avg_spread / current_spread

        # Determine base threshold
        base_threshold = (
            self.SCALPING_THRESHOLD
            if strategy_type == "scalping"
            else self.ORB_THRESHOLD
        )

        # S5-5: Monday warm-up — relaxed threshold (0.60) for first 15 min of Monday open
        # Market open is ~07:00 UTC (London session start)
        dow = now_utc.weekday()
        is_monday_warmup = (
            dow == 0  # Monday
            and now_utc.hour == 7
            and now_utc.minute < self.MONDAY_WARMUP_MINUTES
        )
        if is_monday_warmup:
            effective_threshold = min(self.MONDAY_WARMUP_THRESHOLD, base_threshold)
            logger.info(
                f"SQS: Monday warm-up active for {symbol}: "
                f"threshold {base_threshold} -> {effective_threshold}"
            )
        else:
            effective_threshold = base_threshold

        # Apply news override if present (on top of Monday warm-up if active)
        if news_override is not None:
            effective_threshold = effective_threshold + news_override
            logger.info(
                f"SQS: News override applied for {symbol}: threshold {base_threshold} -> {effective_threshold}"
            )

        # Evaluate against thresholds
        if sqs < self.HARD_BLOCK_THRESHOLD:
            # Hard block - SQS < 0.50
            reason = f"Hard block: SQS {sqs:.4f} < {self.HARD_BLOCK_THRESHOLD} (hard block threshold)"
            logger.warning(f"SQS: {symbol} HARD BLOCK - {reason}")
            return SQSResult(
                allowed=False,
                sqs=sqs,
                threshold=effective_threshold,
                reason=reason,
                is_hard_block=True
            )

        if sqs < effective_threshold:
            # Soft block - below effective threshold
            reason = (
                f"Entry blocked: SQS {sqs:.4f} < {effective_threshold:.2f} "
                f"(threshold for {strategy_type})"
            )
            logger.info(f"SQS: {symbol} blocked - {reason}")
            return SQSResult(
                allowed=False,
                sqs=sqs,
                threshold=effective_threshold,
                reason=reason,
                is_hard_block=False
            )

        # Allowed
        reason = f"Entry allowed: SQS {sqs:.4f} >= {effective_threshold:.2f}"
        logger.debug(f"SQS: {symbol} allowed - {reason}")
        return SQSResult(
            allowed=True,
            sqs=sqs,
            threshold=effective_threshold,
            reason=reason,
            is_hard_block=False
        )

    def _get_bucket_key(self, dt: datetime) -> str:
        """
        Generate bucket key for a given UTC datetime.

        Bucket is 5-minute resolution keyed by day_of_week:hour:minute_bucket
        Format: "{dow}:{hour}:{bucket}" where bucket = minute // 5
        """
        dow = dt.weekday()  # 0=Monday, 6=Sunday
        hour = dt.hour
        minute_bucket = dt.minute // 5
        return f"{dow}:{hour}:{minute_bucket}"

    def _find_valid_bucket(
        self,
        buckets: Dict[str, SpreadBucket],
        reference_dt: datetime
    ) -> Optional[SpreadBucket]:
        """
        Find a valid bucket near the reference time.

        If exact bucket not available, try adjacent buckets within same hour.
        This provides some resilience when market conditions change slightly.
        """
        ref_dow = reference_dt.weekday()
        ref_hour = reference_dt.hour
        ref_min_bucket = reference_dt.minute // 5

        # Try nearby buckets in same hour
        for offset in range(1, 12):  # Check up to 1 hour worth of 5-min buckets
            for direction in [-1, 1]:
                check_bucket = ref_min_bucket + (direction * offset)
                if 0 <= check_bucket < 12:  # 12 5-minute buckets per hour
                    key = f"{ref_dow}:{ref_hour}:{check_bucket}"
                    bucket = buckets.get(key)
                    if bucket and bucket.sample_count >= self.MIN_SAMPLES_FOR_VALID_BUCKET:
                        logger.debug(f"SQS: Using nearby bucket {key} for fallback")
                        return bucket

        return None

    def compute_historical_bucket_key(
        self,
        symbol: str,
        dt: datetime
    ) -> str:
        """
        Compute Redis key for historical spread bucket.

        Format: sqs:history:{symbol}:{dow}:{hour}:{minute_bucket}
        """
        bucket = self._get_bucket_key(dt)
        return f"sqs:history:{symbol}:{bucket}"

    def log_evaluation(
        self,
        symbol: str,
        strategy_type: str,
        result: SQSResult,
        current_spread: float,
        historical_avg: float
    ) -> None:
        """Log SQS evaluation to audit trail."""
        logger.info(
            f"SQS Audit: symbol={symbol} strategy={strategy_type} "
            f"sqs={result.sqs:.4f} threshold={result.threshold:.2f} "
            f"allowed={result.allowed} hard_block={result.is_hard_block} "
            f"current_spread={current_spread} historical_avg={historical_avg:.4f}"
        )


def create_sqs_engine() -> SQSEngine:
    """Factory function to create SQS engine with dependencies."""
    cache = None
    calendar_integration = None

    try:
        from src.risk.sqs_cache import create_sqs_cache
        cache = create_sqs_cache()
    except Exception as e:
        logger.warning(f"Could not initialize SQS cache: {e}")

    try:
        from src.risk.sqs_calendar import create_calendar_integration
        calendar_integration = create_calendar_integration()
    except Exception as e:
        logger.warning(f"Could not initialize calendar integration: {e}")

    return SQSEngine(cache=cache, calendar_integration=calendar_integration)

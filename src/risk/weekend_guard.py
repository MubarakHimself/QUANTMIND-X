"""
Weekend Guard & Monday Warm-up Ramp

Manages:
- Weekend guard: Block Asian-session spread data from entering historical baseline (Fri 21:00 - Sun 21:00 GMT)
- Monday warm-up: SQS threshold ramps from 0.75 to 0.60 over 15 minutes post-market open

Story: 4-7-spread-quality-score-sqs-system
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class WeekendGuardState(BaseModel):
    """State of weekend guard for a symbol."""
    guard_active: bool = False
    warmup_active: bool = False
    warmup_started_at_utc: Optional[datetime] = None
    current_threshold: float = 0.75  # Start at scalping threshold


class WeekendGuardConfig:
    """Configuration for weekend guard behavior."""

    # Guard activation: Friday 21:00 GMT
    GUARD_ACTIVATION_HOUR = 21
    GUARD_ACTIVATION_MINUTE = 0
    GUARD_DAY_OF_WEEK = 4  # Friday (0=Monday, 6=Sunday)

    # Guard deactivation: Sunday 21:00 GMT
    GUARD_DEACTIVATION_HOUR = 21
    GUARD_DEACTIVATION_MINUTE = 0
    DEACTIVATION_DAY_OF_WEEK = 6  # Sunday

    # Monday warm-up: 15 minutes post-open (market opens ~22:00 GMT Sunday)
    WARMUP_DURATION_MINUTES = 15
    WARMUP_START_THRESHOLD = 0.75
    WARMUP_END_THRESHOLD = 0.60
    WARMUP_RATE_PER_MINUTE = (WARMUP_START_THRESHOLD - WARMUP_END_THRESHOLD) / WARMUP_DURATION_MINUTES


class WeekendGuard:
    """
    Weekend guard and Monday warm-up manager.

    Prevents Asian session spread data from affecting historical baseline
    during weekend when markets are closed. Implements Monday warm-up ramp.
    """

    def __init__(self, cache=None):
        """
        Initialize Weekend Guard.

        Args:
            cache: SQS cache instance for persisting guard state
        """
        self._cache = cache
        self._config = WeekendGuardConfig()
        logger.info("Weekend Guard initialized")

    def is_weekend_guard_active(self, symbol: str) -> bool:
        """
        Check if weekend guard is currently active for symbol.

        Guard is active from Friday 21:00 GMT to Sunday 21:00 GMT.
        """
        now_utc = datetime.now(timezone.utc)

        # Check if Friday 21:00 to Sunday 21:00
        if now_utc.weekday() == self._config.GUARD_DAY_OF_WEEK:
            # Friday - check if after 21:00
            if now_utc.hour >= self._config.GUARD_ACTIVATION_HOUR:
                return True
        elif now_utc.weekday() == self._config.DEACTIVATION_DAY_OF_WEEK:
            # Sunday - check if before 21:00
            if now_utc.hour < self._config.GUARD_DEACTIVATION_HOUR:
                return True
        elif now_utc.weekday() in (5,):  # Saturday
            # All of Saturday is weekend
            return True

        return False

    def should_exclude_spread_observation(self, symbol: str) -> bool:
        """
        Determine if a spread observation should be excluded from historical baseline.

        Returns True if weekend guard is active and observation should be excluded.
        """
        return self.is_weekend_guard_active(symbol)

    def is_monday_warmup_active(self, symbol: str) -> bool:
        """
        Check if Monday warm-up ramp is active.

        Warm-up is active for 15 minutes after market open (~22:00 GMT Sunday,
        which means Monday 00:00-00:15 roughly).
        """
        now_utc = datetime.now(timezone.utc)

        if now_utc.weekday() != 0:  # Not Monday
            return False

        # Monday market open is ~22:00 GMT Sunday = 00:00 Monday GMT
        # Warmup runs for 15 minutes
        if now_utc.hour == 0 and now_utc.minute < self._config.WARMUP_DURATION_MINUTES:
            return True

        return False

    def get_effective_threshold(
        self,
        symbol: str,
        base_threshold: float = 0.75
    ) -> float:
        """
        Get effective SQS threshold considering weekend/warmup state.

        Args:
            symbol: Trading symbol
            base_threshold: Base threshold (e.g., 0.75 for scalping)

        Returns:
            Effective threshold (may be higher during warm-up)
        """
        if self.is_monday_warmup_active(symbol):
            return self._calculate_warmup_threshold()

        return base_threshold

    def _calculate_warmup_threshold(self) -> float:
        """
        Calculate current warm-up threshold.

        Ramps linearly from 0.75 to 0.60 over 15 minutes.
        """
        now_utc = datetime.now(timezone.utc)

        # Minutes since midnight (warmup started at 00:00)
        minutes_since_warmup = now_utc.hour * 60 + now_utc.minute

        if minutes_since_warmup >= self._config.WARMUP_DURATION_MINUTES:
            # Warmup complete
            return self._config.WARMUP_END_THRESHOLD

        # Linear ramp: 0.75 -> 0.60 over 15 minutes
        reduction = minutes_since_warmup * self._config.WARMUP_RATE_PER_MINUTE
        threshold = self._config.WARMUP_START_THRESHOLD - reduction

        return max(threshold, self._config.WARMUP_END_THRESHOLD)

    def get_guard_state(self, symbol: str) -> WeekendGuardState:
        """
        Get complete guard state for symbol.

        Returns:
            WeekendGuardState with guard and warmup status
        """
        guard_active = self.is_weekend_guard_active(symbol)
        warmup_active = self.is_monday_warmup_active(symbol)

        warmup_started = None
        current_threshold = self._config.WARMUP_START_THRESHOLD

        if warmup_active:
            warmup_started = datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            current_threshold = self._calculate_warmup_threshold()

        return WeekendGuardState(
            guard_active=guard_active,
            warmup_active=warmup_active,
            warmup_started_at_utc=warmup_started,
            current_threshold=current_threshold
        )

    async def activate_weekend_guard(self, symbol: str) -> None:
        """
        Activate weekend guard for symbol.

        Clears existing historical data and sets guard state.
        Should be called when guard activates.
        """
        if self._cache is None:
            logger.warning("Weekend Guard: No cache available, cannot activate guard")
            return

        logger.info(f"Weekend Guard: Activating for {symbol}")

        # Clear existing historical spread data
        cleared = await self._cache.clear_stale_entries(symbol)

        # Set guard state
        await self._cache.set_weekend_guard_state(symbol, active=True)

        logger.info(f"Weekend Guard: Activated for {symbol}, cleared {cleared} history entries")

    async def deactivate_weekend_guard(self, symbol: str) -> None:
        """Deactivate weekend guard for symbol."""
        if self._cache is None:
            return

        logger.info(f"Weekend Guard: Deactivating for {symbol}")
        await self._cache.set_weekend_guard_state(symbol, active=False)

    async def start_monday_warmup(self, symbol: str) -> None:
        """
        Start Monday warm-up ramp for symbol.

        Warmup begins automatically when system detects Monday conditions.
        """
        if self._cache is None:
            return

        now = datetime.now(timezone.utc)
        started_at = now.timestamp()

        await self._cache.set_monday_warmup_state(
            symbol,
            enabled=True,
            started_at=started_at,
            current_threshold=self._config.WARMUP_START_THRESHOLD
        )

        logger.info(f"Weekend Guard: Monday warmup started for {symbol}")

    async def update_warmup_state(self, symbol: str) -> None:
        """Update warmup state in cache with current threshold."""
        if self._cache is None:
            return

        warmup_state = self.get_guard_state(symbol)

        if not warmup_state.warmup_active:
            return

        # Get stored state to check started_at
        stored = await self._cache.get_monday_warmup_state(symbol)

        started_at = stored.get("started_at") if stored else datetime.now(timezone.utc).timestamp()

        await self._cache.set_monday_warmup_state(
            symbol,
            enabled=True,
            started_at=started_at,
            current_threshold=warmup_state.current_threshold
        )


def create_weekend_guard() -> WeekendGuard:
    """Factory function to create Weekend Guard with cache dependency."""
    cache = None

    try:
        from src.risk.sqs_cache import create_sqs_cache
        cache = create_sqs_cache()
    except Exception as e:
        logger.warning(f"Could not initialize SQS cache for weekend guard: {e}")

    return WeekendGuard(cache=cache)

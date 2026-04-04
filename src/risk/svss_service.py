"""
SVSS Service Module

Authoritative calculator for VWAP, Volume Profile, MFI, and RVOL.
Runs once per tick on Cloudzy; 200 bots consume results from Redis.

This is the Shared Service pattern: SVSSService computes indicators
once and publishes to Redis for all bots to consume.

Key features:
- VWAP: Running sum reset at each session open
- Volume Profile: 0.1-pip bucket histogram with POC, VAH, VAL
- RVOL: Current bar volume / 20-session rolling average at same time-of-day
- MFI: 14-period Money Flow Index with overbought/oversold signals

Redis key: svss:{symbol}:readings (TTL 60s)
"""

import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Dict, List, Tuple

import redis

logger = logging.getLogger(__name__)


def _resolve_redis_url(redis_url: Optional[str] = None) -> str:
    """Resolve Redis URL from explicit argument or deployment environment."""
    return redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")

# Redis key patterns
READINGS_KEY_PATTERN = "svss:{symbol}:readings"
READINGS_TTL = 60  # seconds

# Volume Profile bucket size: 0.1 pips for 5-digit forex
# EURUSD: 0.00001 = 0.1 pip
# USDJPY: 0.001 = 0.1 pip (3-digit)
VOLUME_PROFILE_BUCKET_SIZE = 0.00001

# MFI configuration
MFI_PERIOD = 14

# RVOL configuration
RVOL_ROLLING_SESSIONS = 20

# MFI thresholds
MFI_OVERBOUGHT = 80.0
MFI_OVERSOLD = 20.0


@dataclass
class VolumeProfileResult:
    """Volume Profile calculation result."""

    poc: float  # Point of Control - highest volume price level
    vah: float  # Value Area High - 70% of volume above this
    val: float  # Value Area Low - 70% of volume below this
    profile: Dict[float, float]  # Full histogram: price_level -> volume
    total_volume: float


@dataclass
class MFIResult:
    """MFI calculation result."""

    value: float  # MFI value 0-100
    is_overbought: bool  # > 80
    is_oversold: bool  # < 20
    typical_price: float
    money_flow: float


@dataclass
class RVOLResult:
    """RVOL calculation result."""

    value: float  # Relative volume ratio
    current_volume: float
    rolling_avg: float
    minute_of_day: int  # 0-1439


@dataclass
class VWAPResult:
    """VWAP calculation result."""

    value: float
    cumulative_price_volume: float
    cumulative_volume: float
    session_id: str


@dataclass
class SVSSReadings:
    """
    All SVSS indicator readings for a symbol at a point in time.

    Published to Redis at key svss:{symbol}:readings with TTL 60s.
    """

    symbol: str
    timestamp: datetime
    session_id: str
    vwap: VWAPResult
    volume_profile: VolumeProfileResult
    mfi: MFIResult
    rvol: RVOLResult

    def to_dict(self) -> dict:
        """Serialize to dictionary for Redis."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "session_id": self.session_id,
            "vwap": {
                "value": self.vwap.value,
                "cumulative_price_volume": self.vwap.cumulative_price_volume,
                "cumulative_volume": self.vwap.cumulative_volume,
                "session_id": self.vwap.session_id,
            },
            "volume_profile": {
                "poc": self.volume_profile.poc,
                "vah": self.volume_profile.vah,
                "val": self.volume_profile.val,
                "profile": {str(k): v for k, v in self.volume_profile.profile.items()},
                "total_volume": self.volume_profile.total_volume,
            },
            "mfi": {
                "value": self.mfi.value,
                "is_overbought": self.mfi.is_overbought,
                "is_oversold": self.mfi.is_oversold,
                "typical_price": self.mfi.typical_price,
                "money_flow": self.mfi.money_flow,
            },
            "rvol": {
                "value": self.rvol.value,
                "current_volume": self.rvol.current_volume,
                "rolling_avg": self.rvol.rolling_avg,
                "minute_of_day": self.rvol.minute_of_day,
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SVSSReadings":
        """Deserialize from dictionary."""
        timestamp = datetime.fromisoformat(data["timestamp"])
        return cls(
            symbol=data["symbol"],
            timestamp=timestamp,
            session_id=data["session_id"],
            vwap=VWAPResult(
                value=data["vwap"]["value"],
                cumulative_price_volume=data["vwap"]["cumulative_price_volume"],
                cumulative_volume=data["vwap"]["cumulative_volume"],
                session_id=data["vwap"]["session_id"],
            ),
            volume_profile=VolumeProfileResult(
                poc=data["volume_profile"]["poc"],
                vah=data["volume_profile"]["vah"],
                val=data["volume_profile"]["val"],
                profile={float(k): v for k, v in data["volume_profile"]["profile"].items()},
                total_volume=data["volume_profile"]["total_volume"],
            ),
            mfi=MFIResult(
                value=data["mfi"]["value"],
                is_overbought=data["mfi"]["is_overbought"],
                is_oversold=data["mfi"]["is_oversold"],
                typical_price=data["mfi"]["typical_price"],
                money_flow=data["mfi"]["money_flow"],
            ),
            rvol=RVOLResult(
                value=data["rvol"]["value"],
                current_volume=data["rvol"]["current_volume"],
                rolling_avg=data["rvol"]["rolling_avg"],
                minute_of_day=data["rvol"]["minute_of_day"],
            ),
        )


class SVSSService:
    """
    Authoritative SVSS calculator for VWAP, Volume Profile, MFI, and RVOL.

    This service runs once per tick on Cloudzy and publishes results to Redis.
    All 200 bots consume from Redis instead of calculating independently.

    Usage:
        service = SVSSService(symbol="EURUSD", redis_url="redis://localhost:6379")
        service.connect()
        service.reset_session("london_open_20260328_08")

        # On each tick:
        service.update_vwap(price=1.08542, volume=1000, timestamp=now)
        service.update_volume_profile(price=1.08542, volume=1000)
        service.update_mfi(bid=1.08540, ask=1.08544, last=1.08542, volume=1000, timestamp=now)
        service.update_rvol(volume=1000, timestamp=now)  # Only accurate on bar close

        readings = service.get_current_readings()
        service.publish_to_redis()
    """

    def __init__(
        self,
        symbol: str,
        redis_url: Optional[str] = None,
        bucket_size: float = VOLUME_PROFILE_BUCKET_SIZE,
        mfi_period: int = MFI_PERIOD,
        rvol_sessions: int = RVOL_ROLLING_SESSIONS,
    ):
        """
        Initialize SVSS Service.

        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            redis_url: Redis connection URL
            bucket_size: Volume Profile bucket size in price units
            mfi_period: MFI lookback period
            rvol_sessions: Number of sessions for RVOL rolling average
        """
        self._symbol = symbol.upper()
        self._redis_url = _resolve_redis_url(redis_url)
        self._redis_client: Optional[redis.Redis] = None
        self._connected = False

        # Bucket size for volume profile (0.1 pip)
        self._bucket_size = bucket_size

        # MFI period
        self._mfi_period = mfi_period

        # RVOL rolling sessions
        self._rvol_sessions = rvol_sessions

        # Session tracking
        self._current_session_id: Optional[str] = None
        self._session_reset_time: Optional[datetime] = None

        # VWAP accumulators
        self._vwap_cumulative_pv: float = 0.0
        self._vwap_cumulative_vol: float = 0.0
        self._vwap_last_value: Optional[float] = None

        # Volume Profile accumulators
        self._vp_profile: Dict[float, float] = defaultdict(float)
        self._vp_total_volume: float = 0.0
        self._vp_poc: Optional[float] = None
        self._vp_vah: Optional[float] = None
        self._vp_val: Optional[float] = None

        # MFI accumulators
        self._mfi_typical_prices: List[float] = []
        self._mfi_money_flows: List[float] = []
        self._mfi_positive_flow: float = 0.0
        self._mfi_negative_flow: float = 0.0
        self._mfi_prev_typical: Optional[float] = None
        self._mfi_last_value: Optional[float] = None

        # RVOL accumulators
        self._rvol_current_bar_volume: float = 0.0
        self._rvol_last_minute: Optional[int] = None
        self._rvol_rolling_avg_profile: Dict[int, float] = {}  # minute_of_day -> avg_volume
        self._rvol_last_value: Optional[float] = None

    def connect(self) -> bool:
        """
        Connect to Redis.

        Returns:
            True if connected successfully, False otherwise.
        """
        try:
            self._redis_client = redis.from_url(self._redis_url)
            self._redis_client.ping()
            self._connected = True
            logger.info(f"SVSS Service connected to Redis: {self._redis_url}")
            return True
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self._connected = False
            return False

    def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self._redis_client:
            self._redis_client.close()
            self._redis_client = None
        self._connected = False
        logger.info("SVSS Service disconnected")

    @property
    def is_connected(self) -> bool:
        """Check if connected to Redis."""
        return self._connected

    def reset_session(self, session_id: str, timestamp: Optional[datetime] = None) -> None:
        """
        Reset all accumulators for a new session.

        Called on session boundary (e.g., London Open, NY Open).

        Args:
            session_id: Unique session identifier
            timestamp: Session start time (defaults to now)
        """
        self._current_session_id = session_id
        self._session_reset_time = timestamp or datetime.now(timezone.utc)

        # Reset VWAP
        self._vwap_cumulative_pv = 0.0
        self._vwap_cumulative_vol = 0.0
        self._vwap_last_value = None

        # Reset Volume Profile
        self._vp_profile.clear()
        self._vp_total_volume = 0.0
        self._vp_poc = None
        self._vp_vah = None
        self._vp_val = None

        # Reset MFI
        self._mfi_typical_prices.clear()
        self._mfi_money_flows.clear()
        self._mfi_positive_flow = 0.0
        self._mfi_negative_flow = 0.0
        self._mfi_prev_typical = None
        self._mfi_last_value = None

        # Reset RVOL
        self._rvol_current_bar_volume = 0.0
        self._rvol_last_minute = None
        self._rvol_last_value = None

        logger.info(f"SVSS Service session reset: {session_id}")

    def set_rvol_rolling_avg_profile(self, profile: Dict[int, float]) -> None:
        """
        Set the rolling average volume profile for RVOL calculation.

        This should be loaded from warm storage (DuckDB) on service startup
        or when transitioning between sessions.

        Args:
            profile: Dict mapping minute-of-day (0-1439) to average volume
        """
        self._rvol_rolling_avg_profile = profile
        logger.debug(
            f"RVOL rolling avg profile set: {len(profile)} buckets for {self._symbol}"
        )

    # -------------------------------------------------------------------------
    # VWAP Calculation
    # -------------------------------------------------------------------------

    def calculate_vwap(self, price: float, volume: float) -> float:
        """
        Calculate VWAP from a single tick.

        VWAP = Σ(price × volume) / Σ(volume)

        Args:
            price: Tick price
            volume: Tick volume

        Returns:
            Current VWAP value
        """
        if volume > 0:
            self._vwap_cumulative_pv += price * volume
            self._vwap_cumulative_vol += volume

            if self._vwap_cumulative_vol > 0:
                self._vwap_last_value = self._vwap_cumulative_pv / self._vwap_cumulative_vol
            else:
                self._vwap_last_value = price

        return self._vwap_last_value or 0.0

    def update_vwap(self, price: float, volume: float, timestamp: datetime) -> VWAPResult:
        """
        Update VWAP with a new tick.

        Args:
            price: Tick price
            volume: Tick volume
            timestamp: Tick timestamp

        Returns:
            VWAPResult with current VWAP state
        """
        value = self.calculate_vwap(price, volume)
        return VWAPResult(
            value=value,
            cumulative_price_volume=self._vwap_cumulative_pv,
            cumulative_volume=self._vwap_cumulative_vol,
            session_id=self._current_session_id or "",
        )

    # -------------------------------------------------------------------------
    # Volume Profile Calculation
    # -------------------------------------------------------------------------

    def _bucket_price(self, price: float) -> float:
        """Bucket price to nearest 0.1-pip level."""
        return round(price / self._bucket_size) * self._bucket_size

    def _update_poc_vah_val(self) -> None:
        """Update POC, VAH, and VAL from current profile."""
        if not self._vp_profile:
            self._vp_poc = None
            self._vp_vah = None
            self._vp_val = None
            return

        # POC: price level with highest volume
        self._vp_poc = max(self._vp_profile.keys(), key=lambda p: self._vp_profile[p])

        # Calculate total volume for VAH/VAL (70% of volume area)
        total_vol = sum(self._vp_profile.values())
        target_vol = total_vol * 0.70

        # Sort price levels
        sorted_levels = sorted(self._vp_profile.keys())

        # Find VAH (70% of volume below this price)
        cumsum = 0.0
        vah_price = sorted_levels[-1]
        for price in sorted_levels:
            cumsum += self._vp_profile[price]
            if cumsum >= target_vol:
                vah_price = price
                break
        self._vp_vah = vah_price

        # Find VAL (70% of volume above this price)
        cumsum = 0.0
        val_price = sorted_levels[0]
        for price in reversed(sorted_levels):
            cumsum += self._vp_profile[price]
            if cumsum >= target_vol:
                val_price = price
                break
        self._vp_val = val_price

    def calculate_volume_profile(self, price: float, volume: float) -> VolumeProfileResult:
        """
        Update Volume Profile with a new tick.

        Tracks price distribution by volume in 0.1-pip buckets.
        Identifies POC (Point of Control), VAH (Value Area High), VAL (Value Area Low).

        Args:
            price: Tick price
            volume: Tick volume

        Returns:
            VolumeProfileResult with current profile state
        """
        if volume > 0:
            price_level = self._bucket_price(price)
            self._vp_profile[price_level] += volume
            self._vp_total_volume += volume
            self._update_poc_vah_val()

        return VolumeProfileResult(
            poc=self._vp_poc or 0.0,
            vah=self._vp_vah or 0.0,
            val=self._vp_val or 0.0,
            profile=dict(self._vp_profile),
            total_volume=self._vp_total_volume,
        )

    def update_volume_profile(self, price: float, volume: float) -> VolumeProfileResult:
        """
        Update Volume Profile with a new tick.

        Args:
            price: Tick price
            volume: Tick volume

        Returns:
            VolumeProfileResult with current profile state
        """
        return self.calculate_volume_profile(price, volume)

    # -------------------------------------------------------------------------
    # MFI Calculation
    # -------------------------------------------------------------------------

    def _compute_typical_price(self, bid: float, ask: float, last: float) -> float:
        """
        Compute typical price from tick data.

        Typical Price = (High + Low + Close) / 3
        For tick data: High ≈ ask, Low ≈ bid, Close ≈ last
        """
        return (ask + bid + last) / 3.0

    def _compute_mfi_from_accumulators(self) -> float:
        """
        Compute MFI from accumulated positive/negative money flows.

        Returns:
            MFI value (0-100)
        """
        if self._mfi_negative_flow == 0:
            return 100.0  # No negative flow = overbought

        money_flow_ratio = self._mfi_positive_flow / self._mfi_negative_flow
        mfi = 100.0 - (100.0 / (1.0 + money_flow_ratio))
        return max(0.0, min(100.0, mfi))

    def calculate_mfi(
        self,
        bid: float,
        ask: float,
        last: float,
        volume: float,
        timestamp: datetime,
    ) -> MFIResult:
        """
        Calculate MFI (Money Flow Index) from tick data.

        MFI = 100 - (100 / (1 + money_flow_ratio))
        Where money_flow_ratio = positive_money_flow / negative_money_flow

        Uses 14-period lookback as specified.

        Args:
            bid: Bid price
            ask: Ask price
            last: Last/deal price
            volume: Tick volume
            timestamp: Tick timestamp

        Returns:
            MFIResult with MFI value and overbought/oversold flags
        """
        typical_price = self._compute_typical_price(bid, ask, last)

        if volume > 0:
            money_flow = typical_price * volume
            self._mfi_money_flows.append(money_flow)

            # Determine positive/negative flow based on typical price change
            if self._mfi_prev_typical is not None:
                if typical_price > self._mfi_prev_typical:
                    self._mfi_positive_flow += money_flow
                elif typical_price < self._mfi_prev_typical:
                    self._mfi_negative_flow += money_flow

            self._mfi_prev_typical = typical_price
            self._mfi_typical_prices.append(typical_price)

            # Maintain rolling window of period values
            if len(self._mfi_typical_prices) > self._mfi_period:
                # Remove oldest values
                removed_tp = self._mfi_typical_prices.pop(0)
                removed_mf = self._mfi_money_flows.pop(0)

                # Adjust flows for removed values
                if len(self._mfi_typical_prices) > 0:
                    prev_tp = self._mfi_typical_prices[0]
                    if removed_tp > prev_tp:
                        self._mfi_positive_flow -= removed_mf
                    elif removed_tp < prev_tp:
                        self._mfi_negative_flow -= removed_mf

            # Compute MFI if we have enough data
            if len(self._mfi_typical_prices) >= self._mfi_period:
                self._mfi_last_value = self._compute_mfi_from_accumulators()

        mfi_value = self._mfi_last_value if self._mfi_last_value is not None else 50.0

        return MFIResult(
            value=mfi_value,
            is_overbought=mfi_value > MFI_OVERBOUGHT,
            is_oversold=mfi_value < MFI_OVERSOLD,
            typical_price=typical_price,
            money_flow=typical_price * volume if volume > 0 else 0.0,
        )

    def update_mfi(
        self,
        bid: float,
        ask: float,
        last: float,
        volume: float,
        timestamp: datetime,
    ) -> MFIResult:
        """
        Update MFI with a new tick.

        Args:
            bid: Bid price
            ask: Ask price
            last: Last/deal price
            volume: Tick volume
            timestamp: Tick timestamp

        Returns:
            MFIResult with current MFI state
        """
        return self.calculate_mfi(bid, ask, last, volume, timestamp)

    # -------------------------------------------------------------------------
    # RVOL Calculation
    # -------------------------------------------------------------------------

    def _get_minute_of_day(self, timestamp: datetime) -> int:
        """Get minute of day (0-1439) from timestamp."""
        return timestamp.hour * 60 + timestamp.minute

    def _detect_bar_boundary(self, timestamp: datetime) -> bool:
        """
        Detect if we've crossed a bar boundary.

        Returns:
            True if new bar detected, False otherwise.
        """
        minute_of_day = self._get_minute_of_day(timestamp)

        if self._rvol_last_minute is not None and self._rvol_last_minute != minute_of_day:
            self._rvol_current_bar_volume = 0.0
            return True

        self._rvol_last_minute = minute_of_day
        return False

    def calculate_rvol(self, volume: float, timestamp: datetime) -> RVOLResult:
        """
        Calculate RVOL (Relative Volume).

        RVOL = current_bar_volume / rolling_avg_volume_at_time_of_day

        Rolling average is loaded from warm storage (20-session average per minute-of-day).

        Note: RVOL is most accurate on bar close (when volume for the bar is complete).
        During the bar, volume accumulates and RVOL will increase.

        Args:
            volume: Tick volume
            timestamp: Tick timestamp

        Returns:
            RVOLResult with current RVOL state
        """
        # Detect bar boundary
        is_new_bar = self._detect_bar_boundary(timestamp)

        # Accumulate volume for current bar
        if volume > 0:
            self._rvol_current_bar_volume += volume

        minute_of_day = self._get_minute_of_day(timestamp)

        # Get rolling average for this minute of day
        rolling_avg = self._rvol_rolling_avg_profile.get(minute_of_day, 0.0)

        if rolling_avg > 0 and self._rvol_current_bar_volume > 0:
            self._rvol_last_value = self._rvol_current_bar_volume / rolling_avg
        else:
            self._rvol_last_value = 1.0  # Default when no historical data

        return RVOLResult(
            value=self._rvol_last_value or 1.0,
            current_volume=self._rvol_current_bar_volume,
            rolling_avg=rolling_avg,
            minute_of_day=minute_of_day,
        )

    def update_rvol(self, volume: float, timestamp: datetime) -> RVOLResult:
        """
        Update RVOL with a new tick.

        Note: For accurate RVOL, call this on bar close with the total bar volume.
        During the bar, the RVOL value will grow as volume accumulates.

        Args:
            volume: Tick volume
            timestamp: Tick timestamp

        Returns:
            RVOLResult with current RVOL state
        """
        return self.calculate_rvol(volume, timestamp)

    # -------------------------------------------------------------------------
    # Combined Reading
    # -------------------------------------------------------------------------

    def get_current_readings(self) -> Optional[SVSSReadings]:
        """
        Get current SVSS readings for all indicators.

        Returns:
            SVSSReadings with all indicator values, or None if no session started.
        """
        if self._current_session_id is None:
            return None

        timestamp = datetime.now(timezone.utc)

        return SVSSReadings(
            symbol=self._symbol,
            timestamp=timestamp,
            session_id=self._current_session_id,
            vwap=VWAPResult(
                value=self._vwap_last_value or 0.0,
                cumulative_price_volume=self._vwap_cumulative_pv,
                cumulative_volume=self._vwap_cumulative_vol,
                session_id=self._current_session_id,
            ),
            volume_profile=VolumeProfileResult(
                poc=self._vp_poc or 0.0,
                vah=self._vp_vah or 0.0,
                val=self._vp_val or 0.0,
                profile=dict(self._vp_profile),
                total_volume=self._vp_total_volume,
            ),
            mfi=MFIResult(
                value=self._mfi_last_value if self._mfi_last_value is not None else 50.0,
                is_overbought=(self._mfi_last_value or 50.0) > MFI_OVERBOUGHT,
                is_oversold=(self._mfi_last_value or 50.0) < MFI_OVERSOLD,
                typical_price=self._mfi_typical_prices[-1] if self._mfi_typical_prices else 0.0,
                money_flow=self._mfi_money_flows[-1] if self._mfi_money_flows else 0.0,
            ),
            rvol=RVOLResult(
                value=self._rvol_last_value or 1.0,
                current_volume=self._rvol_current_bar_volume,
                rolling_avg=self._rvol_rolling_avg_profile.get(
                    self._rvol_last_minute or 0, 0.0
                ),
                minute_of_day=self._rvol_last_minute or 0,
            ),
        )

    # -------------------------------------------------------------------------
    # Redis Publishing
    # -------------------------------------------------------------------------

    def publish_to_redis(self) -> bool:
        """
        Publish current readings to Redis.

        Key: svss:{symbol}:readings
        TTL: 60 seconds

        Returns:
            True if published successfully, False otherwise.
        """
        if not self._connected or not self._redis_client:
            logger.warning("Cannot publish - not connected to Redis")
            return False

        readings = self.get_current_readings()
        if readings is None:
            logger.warning("No readings to publish - session not started")
            return False

        try:
            key = READINGS_KEY_PATTERN.format(symbol=self._symbol.lower())
            data = json.dumps(readings.to_dict())
            self._redis_client.setex(key, READINGS_TTL, data)
            logger.debug(f"Published SVSS readings to {key}")
            return True
        except Exception as e:
            logger.error(f"Failed to publish SVSS readings: {e}")
            return False

    @classmethod
    def get_readings_from_redis(
        cls,
        symbol: str,
        redis_client: redis.Redis,
    ) -> Optional[SVSSReadings]:
        """
        Read current SVSS readings from Redis.

        This is a class method for consumers (bots) to read shared readings.

        Args:
            symbol: Trading symbol
            redis_client: Redis client

        Returns:
            SVSSReadings if available, None otherwise.
        """
        try:
            key = READINGS_KEY_PATTERN.format(symbol=symbol.lower())
            data = redis_client.get(key)
            if data is None:
                return None
            return cls.from_dict(json.loads(data))
        except Exception as e:
            logger.error(f"Failed to read SVSS readings from Redis: {e}")
            return None


# -----------------------------------------------------------------------------
# Convenience Functions
# -----------------------------------------------------------------------------

def get_svss_readings(symbol: str, redis_url: Optional[str] = None) -> Optional[SVSSReadings]:
    """
    Convenience function to get current SVSS readings for a symbol.

    Args:
        symbol: Trading symbol (e.g., 'EURUSD')
        redis_url: Redis connection URL

    Returns:
        SVSSReadings if available in Redis, None otherwise.
    """
    try:
        client = redis.from_url(_resolve_redis_url(redis_url))
        readings = SVSSService.get_readings_from_redis(symbol, client)
        client.close()
        return readings
    except Exception as e:
        logger.error(f"Failed to get SVSS readings: {e}")
        return None

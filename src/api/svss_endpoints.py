"""
SVSS REST API Endpoints.

Provides REST API endpoints for the Shared Volume Session Service (SVSS):
- GET /api/svss/vwap/{symbol} — VWAP + Volume Profile data
- GET /api/svss/rvol/{symbol} — Relative Volume data
- GET /api/svss/profile/{symbol} — Volume Profile data
- GET /api/svss/mfi/{symbol} — Money Flow Index data
- GET /api/svss/summary — Summary of all tracked symbols

Data is fetched from Redis cache where SVSS publishes indicator values.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

import redis
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/svss", tags=["SVSS"])

SVSS_CACHE_PREFIX = "svss:cache:"


def _get_redis_url() -> str:
    """Resolve the Redis URL from deployment config."""
    return os.getenv("REDIS_URL", "redis://localhost:6379")


def _get_redis_client() -> redis.Redis:
    """Get a Redis client connection."""
    return redis.from_url(_get_redis_url())


def _get_cached_indicator(symbol: str, indicator: str) -> Optional[dict]:
    """
    Get cached indicator value from Redis.

    Args:
        symbol: Trading symbol (e.g., 'EURUSD')
        indicator: Indicator name (e.g., 'vwap', 'rvvol', 'volume_profile', 'mfi')

    Returns:
        Cached data dict or None if not found.
    """
    try:
        client = _get_redis_client()
        cache_key = f"{SVSS_CACHE_PREFIX}{symbol.lower()}:{indicator.lower()}"
        data = client.get(cache_key)
        if data:
            return json.loads(data)
        return None
    except redis.ConnectionError as e:
        logger.warning(f"Redis connection error: {e}")
        return None
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to decode cached data for {symbol}/{indicator}: {e}")
        return None


def _get_tracked_symbols() -> list[str]:
    """
    Get list of symbols currently tracked by SVSS.

    Scans Redis for svss:cache:* keys and extracts unique symbols.

    Returns:
        List of tracked symbol names.
    """
    try:
        client = _get_redis_client()
        pattern = f"{SVSS_CACHE_PREFIX}*"
        keys = client.keys(pattern)

        symbols = set()
        for key in keys:
            # Key format: svss:cache:{symbol}:{indicator}
            key_str = key.decode() if isinstance(key, bytes) else key
            parts = key_str.split(":")
            if len(parts) >= 3:
                symbols.add(parts[2].upper())

        return sorted(list(symbols))
    except redis.ConnectionError as e:
        logger.warning(f"Redis connection error getting tracked symbols: {e}")
        return []


# Pydantic response models

class VWAPResponse(BaseModel):
    """Response model for VWAP endpoint."""
    symbol: str = Field(..., description="Trading symbol")
    vwap: float = Field(..., description="Volume-Weighted Average Price")
    poc: Optional[float] = Field(None, description="Point of Control price")
    vah: Optional[float] = Field(None, description="Value Area High price")
    val: Optional[float] = Field(None, description="Value Area Low price")
    timestamp: str = Field(..., description="Timestamp of the VWAP value")


class RVOLResponse(BaseModel):
    """Response model for RVOL endpoint."""
    symbol: str = Field(..., description="Trading symbol")
    rvol: float = Field(..., description="Relative Volume ratio")
    quality_score: float = Field(..., description="Data quality score (0-1)")
    timestamp: str = Field(..., description="Timestamp of the RVOL value")


class VolumeProfileResponse(BaseModel):
    """Response model for Volume Profile endpoint."""
    symbol: str = Field(..., description="Trading symbol")
    poc: Optional[float] = Field(None, description="Point of Control price")
    vah: Optional[float] = Field(None, description="Value Area High price")
    val: Optional[float] = Field(None, description="Value Area Low price")
    volume_bid: float = Field(default=0.0, description="Volume on bid side")
    volume_ask: float = Field(default=0.0, description="Volume on ask side")
    timestamp: str = Field(..., description="Timestamp of the profile")


class MFIResponse(BaseModel):
    """Response model for MFI endpoint."""
    symbol: str = Field(..., description="Trading symbol")
    mfi: float = Field(..., description="Money Flow Index value (0-100)")
    zone: str = Field(..., description="MFI zone: overbought (>80), oversold (<20), neutral")
    timestamp: str = Field(..., description="Timestamp of the MFI value")


class SVSSSummaryResponse(BaseModel):
    """Response model for SVSS summary endpoint."""
    symbols: list[str] = Field(..., description="List of tracked symbols")
    indicators: list[str] = Field(
        default=["vwap", "rvvol", "volume_profile", "mfi"],
        description="Available indicator types"
    )
    timestamp: str = Field(..., description="Current server timestamp")


# Endpoints

@router.get("/vwap/{symbol}", response_model=VWAPResponse)
async def get_vwap(symbol: str):
    """
    Get VWAP (Volume-Weighted Average Price) and Volume Profile for a symbol.

    Returns VWAP value along with Point of Control (POC) and Value Area (VAH/VAL)
    derived from the volume profile.

    The VWAP is computed from the current session's tick data.
    POC, VAH, and VAL are derived from the Volume Profile indicator.
    """
    symbol = symbol.upper()

    # Get VWAP value
    vwap_data = _get_cached_indicator(symbol, "vwap")
    if not vwap_data:
        raise HTTPException(
            status_code=404,
            detail=f"Vwap data not available for {symbol}. SVSS may not be running."
        )

    # Get Volume Profile for POC/VAH/VAL
    profile_data = _get_cached_indicator(symbol, "volume_profile")

    # Extract POC from volume profile metadata
    poc = None
    vah = None
    val = None

    if profile_data and "metadata" in profile_data:
        profile_meta = profile_data.get("metadata", {})
        profile_info = profile_meta.get("profile", {})

        # POC is the price level with highest volume
        poc = profile_info.get("poc")

        # Value Area is typically POC ± 1 standard deviation of volume distribution
        # For now, estimate VAH/VAL from profile levels if available
        levels = profile_info.get("levels", {})
        if levels and poc is not None:
            try:
                # Convert levels to numeric and sort by price
                price_levels = [(float(p), float(v)) for p, v in levels.items()]
                price_levels.sort(key=lambda x: x[0])

                # Find total volume
                total_volume = sum(v for _, v in price_levels)
                if total_volume > 0:
                    # Value Area High/Low: price range containing 70% of volume centered on POC
                    target_volume = total_volume * 0.70

                    # Find VAH (price at which 85% of volume is below)
                    cumvol = 0
                    for price, vol in price_levels:
                        cumvol += vol
                        if cumvol >= total_volume * 0.85:
                            vah = price
                            break

                    # Find VAL (price at which 15% of volume is below)
                    cumvol = 0
                    for price, vol in price_levels:
                        cumvol += vol
                        if cumvol >= total_volume * 0.15:
                            val = price
                            break
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to compute VAH/VAL for {symbol}: {e}")

    return VWAPResponse(
        symbol=symbol,
        vwap=vwap_data.get("value", 0.0),
        poc=poc,
        vah=vah,
        val=val,
        timestamp=vwap_data.get("timestamp", datetime.now(timezone.utc).isoformat())
    )


@router.get("/rvol/{symbol}", response_model=RVOLResponse)
async def get_rvol(symbol: str):
    """
    Get RVOL (Relative Volume) for a symbol.

    RVOL = current_bar_volume / rolling_avg_volume_at_time_of_day

    The quality_score indicates confidence in the RVOL value based on
    how much historical data is available (1.0 = full history, 0.0 = no history).
    """
    symbol = symbol.upper()

    rvol_data = _get_cached_indicator(symbol, "rvvol")
    if not rvol_data:
        raise HTTPException(
            status_code=404,
            detail=f"RVOL data not available for {symbol}. SVSS may not be running."
        )

    # Extract metadata for quality score calculation
    metadata = rvol_data.get("metadata", {})
    current_volume = metadata.get("current_volume", 0.0)
    rolling_avg = metadata.get("rolling_avg", 0.0)

    # Quality score: ratio of rolling_avg to expected (1.0 if rolling_avg > 0)
    # Also consider if we have enough data points
    if rolling_avg > 0:
        # Higher quality if current volume is within expected range
        ratio = min(current_volume / rolling_avg, 2.0) / 2.0  # Normalize to 0-1
        quality_score = max(0.0, min(1.0, ratio))
    else:
        quality_score = 0.5  # Neutral when no historical data

    return RVOLResponse(
        symbol=symbol,
        rvol=rvol_data.get("value", 1.0),
        quality_score=quality_score,
        timestamp=rvol_data.get("timestamp", datetime.now(timezone.utc).isoformat())
    )


@router.get("/profile/{symbol}", response_model=VolumeProfileResponse)
async def get_profile(symbol: str):
    """
    Get Volume Profile for a symbol.

    Returns Point of Control (POC), Value Area High (VAH), Value Area Low (VAL),
    and bid/ask volume breakdown.
    """
    symbol = symbol.upper()

    profile_data = _get_cached_indicator(symbol, "volume_profile")
    if not profile_data:
        raise HTTPException(
            status_code=404,
            detail=f"Volume profile not available for {symbol}. SVSS may not be running."
        )

    metadata = profile_data.get("metadata", {})
    profile_info = metadata.get("profile", {})

    # Extract POC
    poc = profile_info.get("poc")

    # Extract levels for VAH/VAL calculation and volume breakdown
    levels = profile_info.get("levels", {})
    vah = None
    val = None
    volume_bid = 0.0
    volume_ask = 0.0

    if levels:
        try:
            price_levels = [(float(p), float(v)) for p, v in levels.items()]
            price_levels.sort(key=lambda x: x[0])

            total_volume = sum(v for _, v in price_levels)
            if total_volume > 0:
                # Value Area High/Low: price range containing 70% of volume centered on POC
                target_volume = total_volume * 0.70

                # Find VAH (price at which 85% of volume is below)
                cumvol = 0
                for price, vol in price_levels:
                    cumvol += vol
                    if cumvol >= total_volume * 0.85:
                        vah = price
                        break

                # Find VAL (price at which 15% of volume is below)
                cumvol = 0
                for price, vol in price_levels:
                    cumvol += vol
                    if cumvol >= total_volume * 0.15:
                        val = price
                        break

                # Volume bid/ask breakdown (approximate using price relative to mid)
                if poc:
                    mid_price = poc
                    for price, vol in price_levels:
                        if price < mid_price:
                            volume_bid += vol
                        else:
                            volume_ask += vol
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to compute VAH/VAL/volume breakdown for {symbol}: {e}")

    return VolumeProfileResponse(
        symbol=symbol,
        poc=poc,
        vah=vah,
        val=val,
        volume_bid=volume_bid,
        volume_ask=volume_ask,
        timestamp=profile_data.get("timestamp", datetime.now(timezone.utc).isoformat())
    )


@router.get("/mfi/{symbol}", response_model=MFIResponse)
async def get_mfi(symbol: str):
    """
    Get MFI (Money Flow Index) for a symbol.

    MFI is a 14-period typical price * volume indicator ranging from 0-100.
    Zones:
    - Overbought: MFI > 80
    - Oversold: MFI < 20
    - Neutral: 20 <= MFI <= 80
    """
    symbol = symbol.upper()

    mfi_data = _get_cached_indicator(symbol, "mfi")
    if not mfi_data:
        raise HTTPException(
            status_code=404,
            detail=f"MFI data not available for {symbol}. SVSS may not be running."
        )

    mfi_value = mfi_data.get("value", 50.0)

    # Determine zone
    if mfi_value > 80:
        zone = "overbought"
    elif mfi_value < 20:
        zone = "oversold"
    else:
        zone = "neutral"

    return MFIResponse(
        symbol=symbol,
        mfi=mfi_value,
        zone=zone,
        timestamp=mfi_data.get("timestamp", datetime.now(timezone.utc).isoformat())
    )


@router.get("/summary", response_model=SVSSSummaryResponse)
async def get_summary():
    """
    Get summary of all tracked symbols and available indicators.

    Returns a list of symbols currently being tracked by SVSS and
    the available indicator types.
    """
    symbols = _get_tracked_symbols()

    return SVSSSummaryResponse(
        symbols=symbols,
        indicators=["vwap", "rvvol", "volume_profile", "mfi"],
        timestamp=datetime.now(timezone.utc).isoformat()
    )

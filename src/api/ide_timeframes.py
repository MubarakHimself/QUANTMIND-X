"""
QuantMind IDE Timeframes Endpoints

API endpoints for multi-timeframe regime analysis.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Any
from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/trading/timeframes", tags=["timeframes"])

# In-memory sentinel instances
_mtf_sentinel_instance: Dict[str, Any] = {}


def _get_multi_timeframe_sentinel():
    """Get or create the multi-timeframe sentinel instance."""
    sentinel_key = "default"

    if sentinel_key not in _mtf_sentinel_instance:
        try:
            from src.routing.signals.confluence_signal import MultiTimeframeSentinel, Timeframe
            _mtf_sentinel_instance[sentinel_key] = MultiTimeframeSentinel(timeframes=[Timeframe.M5, Timeframe.H1, Timeframe.H4])
        except Exception as e:
            logger.warning(f"Could not create MultiTimeframeSentinel: {e}")
            _mtf_sentinel_instance[sentinel_key] = None

    return _mtf_sentinel_instance.get(sentinel_key)


@router.get("/regimes")
async def get_all_timeframe_regimes():
    """
    Get all timeframe regimes from the MultiTimeframeSentinel.

    Returns dictionary of timeframe -> regime report for all tracked timeframes.
    """
    sentinel = _get_multi_timeframe_sentinel()
    if sentinel is None:
        raise HTTPException(503, "Multi-timeframe sentinel not available")

    all_regimes = sentinel.get_all_regimes()
    return {
        "regimes": {
            tf.name: {
                "regime": report.regime,
                "chaos_score": report.chaos_score,
                "regime_quality": report.regime_quality,
                "susceptibility": report.susceptibility,
                "news_state": report.news_state,
            }
            for tf, report in all_regimes.items()
        },
        "dominant_regime": sentinel.get_dominant_regime()
    }


@router.get("/dominant")
async def get_dominant_regime():
    """
    Get the dominant regime across all timeframes.

    Returns the dominant regime string using voting logic.
    """
    sentinel = _get_multi_timeframe_sentinel()
    if sentinel is None:
        raise HTTPException(503, "Multi-timeframe sentinel not available")

    return {
        "dominant_regime": sentinel.get_dominant_regime()
    }


@router.post("/tick")
async def process_timeframe_tick(request: dict):
    """
    Process a tick through the MultiTimeframeSentinel.

    Accepts symbol, price, and optional timestamp.
    """
    sentinel = _get_multi_timeframe_sentinel()
    if sentinel is None:
        raise HTTPException(503, "Multi-timeframe sentinel not available")

    symbol = request.get("symbol", "EURUSD")
    price = request.get("price")
    if price is None:
        raise HTTPException(400, "price is required")

    timestamp = request.get("timestamp")
    if timestamp:
        try:
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        except ValueError:
            timestamp = datetime.now(timezone.utc)
    else:
        timestamp = datetime.now(timezone.utc)

    # Process tick
    updated_regimes = sentinel.on_tick(symbol, price, timestamp)

    return {
        "symbol": symbol,
        "price": price,
        "timestamp": timestamp.isoformat(),
        "updated_timeframes": [tf.name for tf in updated_regimes.keys()],
        "dominant_regime": sentinel.get_dominant_regime()
    }

"""
SQS API Endpoints

REST API endpoints for Spread Quality Score system:
- GET /api/risk/sqs/{symbol} - Get SQS for specific symbol
- GET /api/risk/sqs/all - Get SQS for all active symbols
- GET /api/risk/sqs/history/{symbol} - Get historical spread data for symbol

Story: 4-7-spread-quality-score-sqs-system
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/risk/sqs", tags=["sqs"])


# =============================================================================
# Request/Response Models
# =============================================================================

class SQSResponse(BaseModel):
    """SQS evaluation response for a symbol."""
    symbol: str
    sqs: float
    threshold: float
    allowed: bool
    is_hard_block: bool
    reason: str
    strategy_type: str
    current_spread: float
    historical_avg_spread: float
    bucket_sample_count: int
    news_override_active: bool
    weekend_guard_active: bool
    warmup_active: bool
    evaluated_at_utc: datetime


class SQSSummaryResponse(BaseModel):
    """Summary of SQS for all symbols."""
    symbols: List[str]
    evaluations: Dict[str, SQSResponse]
    evaluated_at_utc: datetime


class SQSEvaluationRequest(BaseModel):
    """Request to evaluate SQS for a symbol."""
    symbol: str = Field(..., description="Trading symbol (e.g., EURUSD)")
    strategy_type: Literal["scalping", "ORB"] = Field(
        default="scalping",
        description="Strategy type for threshold selection"
    )
    current_spread: Optional[float] = Field(
        None,
        description="Current spread (fetched from MT5 if not provided)"
    )


class HistoricalSpreadBucket(BaseModel):
    """Historical spread bucket data."""
    bucket_key: str
    avg_spread: float
    sample_count: int
    updated_at_utc: datetime


class HistoricalSpreadResponse(BaseModel):
    """Historical spread data for a symbol."""
    symbol: str
    buckets: List[HistoricalSpreadBucket]
    retrieved_at_utc: datetime


# =============================================================================
# Global SQS Engine Instance
# =============================================================================

_sqs_engine = None
_sqs_cache = None
_weekend_guard = None
_calendar_integration = None


def _get_sqs_engine():
    """Get or create SQS engine instance."""
    global _sqs_engine, _sqs_cache, _weekend_guard, _calendar_integration

    if _sqs_engine is None:
        try:
            from src.risk.sqs_engine import SQSEngine
            from src.risk.sqs_cache import create_sqs_cache
            from src.risk.sqs_calendar import create_calendar_integration
            from src.risk.weekend_guard import create_weekend_guard

            _sqs_cache = create_sqs_cache()
            _calendar_integration = create_calendar_integration()
            _weekend_guard = create_weekend_guard()
            _sqs_engine = SQSEngine(
                cache=_sqs_cache,
                calendar_integration=_calendar_integration
            )
            logger.info("SQS engine initialized")
        except Exception as e:
            logger.error(f"Failed to initialize SQS engine: {e}")
            return None

    return _sqs_engine


def _get_current_spread_from_mt5(symbol: str) -> float:
    """
    Get current spread from MT5 ZMQ tick feed.

    Returns the current spread or raises RuntimeError if MT5 is not connected.
    """
    try:
        from src.risk.integrations.mt5_client import get_mt5_client
        client = get_mt5_client()
        if client and client.is_connected():
            tick = client.get_tick(symbol)
            if tick:
                return tick.spread
        raise RuntimeError(f"MT5 not connected for symbol {symbol}")
    except ImportError:
        raise RuntimeError("MT5 client not available")


def _get_historical_buckets(symbol: str, cache) -> Dict[str, Any]:
    """
    Get historical spread buckets for symbol.

    Returns dict of bucket_key -> SpreadBucket data.
    Raises RuntimeError if SQS engine is not available.
    """
    if cache is None:
        raise RuntimeError("SQS cache not initialized")

    try:
        return cache.get_buckets(symbol)
    except Exception as e:
        raise RuntimeError(f"Failed to get historical buckets: {e}")


# =============================================================================
# API Endpoints
# =============================================================================

@router.get("/{symbol}", response_model=SQSResponse)
async def get_symbol_sqs(
    symbol: str,
    strategy_type: Literal["scalping", "ORB"] = "scalping"
):
    """
    Get SQS evaluation for a specific symbol.

    Computes SQS = historical_avg_spread / current_spread
    and evaluates against thresholds for the strategy type.
    """
    engine = _get_sqs_engine()

    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="SQS engine not available"
        )

    # Get current spread (from MT5 or demo)
    current_spread = _get_current_spread_from_mt5(symbol)

    # Get historical buckets
    buckets = _get_historical_buckets(symbol, _sqs_cache)

    # Get news override
    news_override = None
    if _calendar_integration:
        news_override = _calendar_integration.get_threshold_override(symbol)

    # Get weekend/warmup state
    weekend_active = False
    warmup_active = False
    if _weekend_guard:
        state = _weekend_guard.get_guard_state(symbol)
        weekend_active = state.guard_active
        warmup_active = state.warmup_active

    # Find current bucket
    bucket_key = engine._get_bucket_key(datetime.now(timezone.utc))
    current_bucket = buckets.get(bucket_key)
    historical_avg = current_bucket.avg_spread if current_bucket else current_spread
    sample_count = current_bucket.sample_count if current_bucket else 0

    # Evaluate SQS
    result = engine.evaluate(
        symbol=symbol,
        strategy_type=strategy_type,
        current_spread=current_spread,
        historical_buckets=buckets,
        news_override=news_override
    )

    return SQSResponse(
        symbol=symbol,
        sqs=result.sqs,
        threshold=result.threshold,
        allowed=result.allowed,
        is_hard_block=result.is_hard_block,
        reason=result.reason,
        strategy_type=strategy_type,
        current_spread=current_spread,
        historical_avg_spread=historical_avg,
        bucket_sample_count=sample_count,
        news_override_active=news_override is not None,
        weekend_guard_active=weekend_active,
        warmup_active=warmup_active,
        evaluated_at_utc=datetime.now(timezone.utc)
    )


@router.get("/all", response_model=SQSSummaryResponse)
async def get_all_symbols_sqs():
    """
    Get SQS evaluation for all active symbols.

    Returns summary with evaluations for each tracked symbol.
    """
    engine = _get_sqs_engine()

    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="SQS engine not available"
        )

    # Active symbols to track
    symbols = ["EURUSD", "GBPUSD", "USDJPY", "GBPJPY", "AUDUSD", "USDCAD", "NZDUSD"]
    evaluations = {}

    now = datetime.now(timezone.utc)

    for symbol in symbols:
        try:
            current_spread = _get_current_spread_from_mt5(symbol)
            buckets = _get_historical_buckets(symbol, _sqs_cache)
            news_override = None

            if _calendar_integration:
                news_override = _calendar_integration.get_threshold_override(symbol)

            weekend_active = False
            warmup_active = False
            if _weekend_guard:
                state = _weekend_guard.get_guard_state(symbol)
                weekend_active = state.guard_active
                warmup_active = state.warmup_active

            bucket_key = engine._get_bucket_key(now)
            current_bucket = buckets.get(bucket_key)
            historical_avg = current_bucket.avg_spread if current_bucket else current_spread
            sample_count = current_bucket.sample_count if current_bucket else 0

            result = engine.evaluate(
                symbol=symbol,
                strategy_type="scalping",  # Default to scalping for overview
                current_spread=current_spread,
                historical_buckets=buckets,
                news_override=news_override
            )

            evaluations[symbol] = SQSResponse(
                symbol=symbol,
                sqs=result.sqs,
                threshold=result.threshold,
                allowed=result.allowed,
                is_hard_block=result.is_hard_block,
                reason=result.reason,
                strategy_type="scalping",
                current_spread=current_spread,
                historical_avg_spread=historical_avg,
                bucket_sample_count=sample_count,
                news_override_active=news_override is not None,
                weekend_guard_active=weekend_active,
                warmup_active=warmup_active,
                evaluated_at_utc=now
            )

        except Exception as e:
            logger.warning(f"Error evaluating SQS for {symbol}: {e}")

    return SQSSummaryResponse(
        symbols=list(evaluations.keys()),
        evaluations=evaluations,
        evaluated_at_utc=now
    )


@router.post("/evaluate", response_model=SQSResponse)
async def evaluate_sqs(request: SQSEvaluationRequest):
    """
    Explicitly evaluate SQS for a symbol with provided data.

    Use this for testing or when you want to override the current spread.
    """
    engine = _get_sqs_engine()

    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="SQS engine not available"
        )

    symbol = request.symbol.upper()
    strategy_type = request.strategy_type

    # Use provided spread or fetch current
    current_spread = request.current_spread
    if current_spread is None:
        current_spread = _get_current_spread_from_mt5(symbol)

    # Get historical buckets
    buckets = _get_historical_buckets(symbol, _sqs_cache)

    # Get news override
    news_override = None
    if _calendar_integration:
        news_override = _calendar_integration.get_threshold_override(symbol)

    # Evaluate
    result = engine.evaluate(
        symbol=symbol,
        strategy_type=strategy_type,
        current_spread=current_spread,
        historical_buckets=buckets,
        news_override=news_override
    )

    # Get bucket info
    bucket_key = engine._get_bucket_key(datetime.now(timezone.utc))
    current_bucket = buckets.get(bucket_key)
    historical_avg = current_bucket.avg_spread if current_bucket else current_spread
    sample_count = current_bucket.sample_count if current_bucket else 0

    # Get weekend/warmup state
    weekend_active = False
    warmup_active = False
    if _weekend_guard:
        state = _weekend_guard.get_guard_state(symbol)
        weekend_active = state.guard_active
        warmup_active = state.warmup_active

    return SQSResponse(
        symbol=symbol,
        sqs=result.sqs,
        threshold=result.threshold,
        allowed=result.allowed,
        is_hard_block=result.is_hard_block,
        reason=result.reason,
        strategy_type=strategy_type,
        current_spread=current_spread,
        historical_avg_spread=historical_avg,
        bucket_sample_count=sample_count,
        news_override_active=news_override is not None,
        weekend_guard_active=weekend_active,
        warmup_active=warmup_active,
        evaluated_at_utc=datetime.now(timezone.utc)
    )


@router.get("/history/{symbol}", response_model=HistoricalSpreadResponse)
async def get_historical_spread(symbol: str):
    """
    Get historical spread data for a symbol.

    Returns all 5-minute buckets with their average spreads.
    """
    symbol = symbol.upper()
    buckets = _get_historical_buckets(symbol, _sqs_cache)

    historical_buckets = []
    now = datetime.now(timezone.utc)

    for key, bucket in buckets.items():
        historical_buckets.append(HistoricalSpreadBucket(
            bucket_key=key,
            avg_spread=bucket.avg_spread,
            sample_count=bucket.sample_count,
            updated_at_utc=bucket.updated_at_utc
        ))

    return HistoricalSpreadResponse(
        symbol=symbol,
        buckets=historical_buckets,
        retrieved_at_utc=now
    )

"""
Variant Browser API Endpoints

REST API for EA variant browser functionality on the Development canvas.
Provides strategy variant data with backtest summaries.
"""
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from src.mql5.versions.schema import VariantType

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/variant-browser", tags=["variant-browser"])


# ============================================================================
# Request/Response Models
# ============================================================================


class BacktestSummary(BaseModel):
    """Backtest summary for a variant."""
    total_pnl: float = Field(default=0.0, description="Total profit/loss")
    sharpe_ratio: float = Field(default=0.0, description="Sharpe ratio")
    max_drawdown: float = Field(default=0.0, description="Maximum drawdown percentage")
    trade_count: int = Field(default=0, description="Number of trades")
    win_rate: float = Field(default=0.0, description="Win rate percentage")
    profit_factor: float = Field(default=0.0, description="Profit factor")
    period: str = Field(default="", description="Backtest period")
    last_updated: Optional[str] = None


class VariantInfo(BaseModel):
    """Information about a single variant."""
    variant_type: str = Field(..., description="Variant type: vanilla, spiced, mode_b, mode_c")
    version_tag: str = Field(..., description="Version tag (e.g., 1.0.0)")
    improvement_cycle: int = Field(default=0, description="Improvement cycle number")
    author: str = Field(default="", description="Author of this version")
    created_at: str = Field(default="", description="Creation timestamp")
    is_active: bool = Field(default=False, description="Is this the active variant")
    backtest: Optional[BacktestSummary] = None
    promotion_status: str = Field(default="development", description="Pipeline stage")


class StrategyVariants(BaseModel):
    """All variants for a single strategy."""
    strategy_id: str = Field(..., description="Strategy identifier")
    strategy_name: str = Field(default="", description="Strategy display name")
    variants: List[VariantInfo] = Field(default_factory=list, description="All variants")
    variant_counts: Dict[str, int] = Field(default_factory=dict, description="Count per variant type")


class VariantBrowserResponse(BaseModel):
    """Response for variant browser data."""
    strategies: List[StrategyVariants] = Field(default_factory=list, description="All strategies with variants")
    total_strategies: int = Field(default=0, description="Total number of strategies")
    total_variants: int = Field(default=0, description="Total number of variants")


class VariantDetailResponse(BaseModel):
    """Detailed response for a specific variant."""
    strategy_id: str
    strategy_name: str
    variant: VariantInfo
    version_timeline: List[Dict[str, Any]] = Field(default_factory=list, description="Version history")
    code_content: Optional[str] = None


# ============================================================================
# Helper Functions
# ============================================================================


def _get_mock_backtest_summary(variant_type: str, strategy_id: str) -> BacktestSummary:
    """Generate mock backtest summary for demo purposes."""
    # In production, this would query backtest results from database
    import random
    random.seed(hash(strategy_id + variant_type) % 1000)

    return BacktestSummary(
        total_pnl=round(random.uniform(-500, 5000), 2),
        sharpe_ratio=round(random.uniform(0.5, 3.5), 2),
        max_drawdown=round(random.uniform(2, 25), 2),
        trade_count=random.randint(50, 500),
        win_rate=round(random.uniform(35, 75), 1),
        profit_factor=round(random.uniform(1.0, 3.0), 2),
        period="2024-Q4",
        last_updated=datetime.now().isoformat(),
    )


def _get_promotion_status(improvement_cycle: int) -> str:
    """Determine promotion status based on improvement cycle."""
    if improvement_cycle >= 3:
        return "live"
    elif improvement_cycle >= 2:
        return "sit"
    elif improvement_cycle >= 1:
        return "paper_trading"
    else:
        return "development"


def _load_variants_from_storage() -> List[StrategyVariants]:
    """Load variants from version storage."""
    from src.mql5.versions.storage import get_ea_version_storage

    storage = get_ea_version_storage()

    # Get all strategies (from demo data or storage)
    # In production, would scan the storage directory
    strategies = []

    # Demo strategies for the UI
    demo_strategies = [
        "news-event-breakout",
        "trend-follower-pro",
        "mean-reversion-ai",
        "grid-hedger",
        "scalper-micro",
    ]

    strategy_names = {
        "news-event-breakout": "News Event Breakout",
        "trend-follower-pro": "Trend Follower Pro",
        "mean-reversion-ai": "Mean Reversion AI",
        "grid-hedger": "Grid Hedger",
        "scalper-micro": "Scalper Micro",
    }

    variant_types = [
        VariantType.VANILLA,
        VariantType.SPICED,
        VariantType.MODE_B,
        VariantType.MODE_C,
    ]

    for strategy_id in demo_strategies:
        variants = []
        variant_counts = {"vanilla": 0, "spiced": 0, "mode_b": 0, "mode_c": 0}

        for variant_type in variant_types:
            # Get versions for this strategy and variant type
            try:
                versions = storage.list_versions(strategy_id)
            except Exception:
                versions = []

            # Filter by variant type or create demo versions
            filtered = [v for v in versions if v.variant_type == variant_type]

            if not filtered:
                # Create demo version for UI
                cycle = list(variant_types).index(variant_type)
                variant_info = VariantInfo(
                    variant_type=variant_type.value,
                    version_tag=f"1.{cycle}.0",
                    improvement_cycle=cycle,
                    author="system",
                    created_at=datetime.now().isoformat(),
                    is_active=(variant_type == VariantType.VANILLA),
                    backtest=_get_mock_backtest_summary(variant_type.value, strategy_id),
                    promotion_status=_get_promotion_status(cycle),
                )
            else:
                # Use actual version
                v = filtered[0]
                # Get backtest from artifacts
                bt_ids = v.artifacts.backtest_result_ids
                variant_info = VariantInfo(
                    variant_type=v.variant_type.value,
                    version_tag=v.version_tag,
                    improvement_cycle=v.improvement_cycle,
                    author=v.author,
                    created_at=v.created_at.isoformat() if v.created_at else "",
                    is_active=True,  # Simplified
                    backtest=_get_mock_backtest_summary(v.variant_type.value, strategy_id),
                    promotion_status=_get_promotion_status(v.improvement_cycle),
                )

            variants.append(variant_info)
            variant_counts[variant_type.value] = 1

        strategies.append(StrategyVariants(
            strategy_id=strategy_id,
            strategy_name=strategy_names.get(strategy_id, strategy_id),
            variants=variants,
            variant_counts=variant_counts,
        ))

    return strategies


# ============================================================================
# Endpoints
# ============================================================================


@router.get("", response_model=VariantBrowserResponse)
async def get_all_variants(
    strategy_id: Optional[str] = Query(None, description="Filter by strategy ID"),
) -> VariantBrowserResponse:
    """
    Get all strategy variants for the variant browser.

    GET /api/variant-browser

    Returns all strategies with their variants (vanilla, spiced, mode_b, mode_c)
    including backtest summaries.
    """
    all_strategies = _load_variants_from_storage()

    if strategy_id:
        all_strategies = [s for s in all_strategies if s.strategy_id == strategy_id]

    total_variants = sum(len(s.variants) for s in all_strategies)

    return VariantBrowserResponse(
        strategies=all_strategies,
        total_strategies=len(all_strategies),
        total_variants=total_variants,
    )


@router.get("/{strategy_id}", response_model=StrategyVariants)
async def get_strategy_variants(
    strategy_id: str,
) -> StrategyVariants:
    """
    Get all variants for a specific strategy.

    GET /api/variant-browser/{strategy_id}

    Returns all variant types with backtest summaries.
    """
    strategies = _load_variants_from_storage()

    for strategy in strategies:
        if strategy.strategy_id == strategy_id:
            return strategy

    raise HTTPException(
        status_code=404,
        detail=f"Strategy {strategy_id} not found"
    )


@router.get("/{strategy_id}/{variant_type}", response_model=VariantDetailResponse)
async def get_variant_detail(
    strategy_id: str,
    variant_type: str,
) -> VariantDetailResponse:
    """
    Get detailed information about a specific variant.

    GET /api/variant-browser/{strategy_id}/{variant_type}

    Returns variant info, version timeline, and code content.
    """
    # Validate variant type
    try:
        vt = VariantType(variant_type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid variant type: {variant_type}. Must be one of: vanilla, spiced, mode_b, mode_c"
        )

    # Get strategy variants
    strategies = _load_variants_from_storage()
    strategy_variants = None
    for s in strategies:
        if s.strategy_id == strategy_id:
            strategy_variants = s
            break

    if not strategy_variants:
        raise HTTPException(
            status_code=404,
            detail=f"Strategy {strategy_id} not found"
        )

    # Find the specific variant
    variant_info = None
    for v in strategy_variants.variants:
        if v.variant_type == variant_type:
            variant_info = v
            break

    if not variant_info:
        raise HTTPException(
            status_code=404,
            detail=f"Variant {variant_type} not found for strategy {strategy_id}"
        )

    # Get version timeline from strategy versions API
    timeline = []
    try:
        from src.mql5.versions.storage import get_ea_version_storage
        storage = get_ea_version_storage()
        versions = storage.list_versions(strategy_id)

        # Filter for this variant type and get timeline
        vt_versions = [v for v in versions if v.variant_type == vt]
        for v in sorted(vt_versions, key=lambda x: x.version_tag):
            timeline.append({
                "version_tag": v.version_tag,
                "created_at": v.created_at.isoformat() if v.created_at else "",
                "author": v.author,
                "improvement_cycle": v.improvement_cycle,
                "is_active": True,  # Simplified
            })

        # If no versions, create demo timeline
        if not timeline:
            for i in range(3):
                timeline.append({
                    "version_tag": f"1.{i}.0",
                    "created_at": datetime.now().isoformat(),
                    "author": "system",
                    "improvement_cycle": i,
                    "is_active": i == variant_info.improvement_cycle,
                })
    except Exception as e:
        logger.warning(f"Failed to load version timeline: {e}")
        # Create demo timeline
        for i in range(3):
            timeline.append({
                "version_tag": f"1.{i}.0",
                "created_at": datetime.now().isoformat(),
                "author": "system",
                "improvement_cycle": i,
                "is_active": i == variant_info.improvement_cycle,
            })

    # Get strategy name
    strategy_name = strategy_variants.strategy_name

    return VariantDetailResponse(
        strategy_id=strategy_id,
        strategy_name=strategy_name,
        variant=variant_info,
        version_timeline=timeline,
        code_content=None,  # Would load from storage in production
    )


@router.get("/{strategy_id}/{variant_type}/code")
async def get_variant_code(
    strategy_id: str,
    variant_type: str,
    version: Optional[str] = Query(None, description="Specific version to get code for"),
) -> Dict[str, Any]:
    """
    Get the source code for a specific variant.

    GET /api/variant-browser/{strategy_id}/{variant_type}/code

    Returns the MQL5 source code for the variant.
    """
    # Validate variant type
    try:
        vt = VariantType(variant_type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid variant type: {variant_type}"
        )

    # In production, would load from file storage
    # For now, return demo code
    demo_code = f"""// QUANTMINDX EA - {variant_type.upper()} variant
// Strategy: {strategy_id}
// Generated by Alpha Forge

#property copyright "QuantMindX"
#property version   "1.0.0"
#property strict

#include <Trade/Trade.mqh>

input group "=== Risk Management ==="
input double LotSize = 0.1;           // Lot size
input double RiskPercent = 1.0;       // Risk per trade %
input int MaxSpread = 30;            // Max spread in points

input group "=== Strategy Parameters ==="
input int FastMA = 9;                 // Fast MA period
input int SlowMA = 21;               // Slow MA period
input int MagicNumber = 12345;        // Magic number

CTrade trade;

int OnInit() {{
    // Initialize expert advisor
    Print("EA Initialized: {strategy_id} ({variant_type})");
    return(INIT_SUCCEEDED);
}}

void OnTick() {{
    // Main trading logic here
    // Strategy: {strategy_id}
    // Variant: {variant_type}

    double maFast = iMA(_Symbol, PERIOD_H1, FastMA, 0, MODE_SMA, PRICE_CLOSE);
    double maSlow = iMA(_Symbol, PERIOD_H1, SlowMA, 0, MODE_SMA, PRICE_CLOSE);

    // Signal processing
    if (maFast > maSlow) {{
        // Buy signal
    }} else if (maFast < maSlow) {{
        // Sell signal
    }}
}}

void OnDeinit(const int reason) {{
    Print("EA Deinitialized");
}}
"""

    return {
        "strategy_id": strategy_id,
        "variant_type": variant_type,
        "version": version or "1.0.0",
        "language": "mql5",
        "code": demo_code,
    }


@router.get("/{strategy_id}/{variant_type}/compare")
async def compare_variant_versions(
    strategy_id: str,
    variant_type: str,
    version_a: str = Query(..., description="First version to compare"),
    version_b: str = Query(..., description="Second version to compare"),
) -> Dict[str, Any]:
    """
    Compare two versions of a variant.

    GET /api/variant-browser/{strategy_id}/{variant_type}/compare?version_a=1.0.0&version_b=1.1.0

    Returns the diff between two versions.
    """
    # Validate variant type
    try:
        VariantType(variant_type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid variant type: {variant_type}"
        )

    # In production, would load actual versions and compute diff
    # For demo, return a comparison structure
    return {
        "strategy_id": strategy_id,
        "variant_type": variant_type,
        "version_a": version_a,
        "version_b": version_b,
        "changes": [
            {"type": "modified", "line": 15, "old": "input int FastMA = 9;", "new": "input int FastMA = 12;"},
            {"type": "added", "line": 25, "new": "// Added trailing stop logic"},
            {"type": "removed", "line": 30, "old": "// Removed old parameter"},
        ],
        "summary": {
            "additions": 5,
            "deletions": 2,
            "modifications": 3,
        }
    }
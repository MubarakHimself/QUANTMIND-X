"""
A/B Race Board API Endpoints

REST API for comparing strategy variants and real-time metrics.
"""
import logging
import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from scipy import stats
import numpy as np

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/strategies", tags=["ab-race-board"])


# ============================================================================
# Request/Response Models
# ============================================================================


class VariantMetrics(BaseModel):
    """Metrics for a single variant."""
    pnl: float = Field(..., description="Total profit/loss")
    trade_count: int = Field(..., description="Number of trades")
    drawdown: float = Field(..., description="Maximum drawdown percentage")
    sharpe: float = Field(..., description="Sharpe ratio")
    win_rate: float = Field(0, description="Win rate percentage")
    avg_profit: float = Field(0, description="Average profit per trade")
    avg_loss: float = Field(0, description="Average loss per trade")
    profit_factor: float = Field(0, description="Profit factor")
    max_consecutive_wins: int = Field(0, description="Max consecutive wins")
    max_consecutive_losses: int = Field(0, description="Max consecutive losses")


class StatisticalSignificance(BaseModel):
    """Statistical significance of comparison."""
    p_value: float = Field(..., description="P-value from t-test")
    is_significant: bool = Field(..., description="Whether result is statistically significant (p < 0.05)")
    winner: Optional[str] = Field(None, description="Which variant is winning (A/B/none)")
    confidence_level: float = Field(..., description="Confidence level percentage")
    sample_size_a: int = Field(..., description="Trade count for variant A")
    sample_size_b: int = Field(..., description="Trade count for variant B")


class ABComparisonResponse(BaseModel):
    """Response for A/B variant comparison."""
    strategy_id: str
    variant_a: str
    variant_b: str
    metrics_a: VariantMetrics
    metrics_b: VariantMetrics
    statistical_significance: Optional[StatisticalSignificance]
    timestamp: str


class ABComparisonListResponse(BaseModel):
    """List of available comparisons."""
    comparisons: List[ABComparisonResponse]
    total: int


# ============================================================================
# Mock Data Service (Replace with actual data sources)
# ============================================================================


class VariantMetricsService:
    """Service to fetch variant metrics from trading data."""

    async def get_variant_metrics(
        self,
        strategy_id: str,
        variant_id: str,
    ) -> VariantMetrics:
        """
        Fetch metrics for a variant from trading data.

        In production, this would query:
        - Paper trading positions for variant
        - Backtest results
        - Live trading stats
        """
        # Mock implementation - returns simulated data
        # In production: query paper_trading_positions, backtest_results tables
        import random

        trade_count = random.randint(20, 200)
        pnl = random.uniform(-500, 2000)
        drawdown = random.uniform(0, 25)
        sharpe = random.uniform(-0.5, 3.5)
        win_rate = random.uniform(30, 70)

        wins = int(trade_count * win_rate / 100)
        losses = trade_count - wins

        avg_profit = abs(pnl * 0.7 / max(wins, 1)) if wins > 0 else 0
        avg_loss = abs(pnl * 0.3 / max(losses, 1)) if losses > 0 else 0
        profit_factor = (avg_profit * wins) / max(avg_loss * losses, 0.001)

        return VariantMetrics(
            pnl=round(pnl, 2),
            trade_count=trade_count,
            drawdown=round(drawdown, 2),
            sharpe=round(sharpe, 2),
            win_rate=round(win_rate, 2),
            avg_profit=round(avg_profit, 2),
            avg_loss=round(avg_loss, 2),
            profit_factor=round(profit_factor, 2),
            max_consecutive_wins=random.randint(2, 10),
            max_consecutive_losses=random.randint(1, 8),
        )

    async def get_trade_history(
        self,
        strategy_id: str,
        variant_id: str,
    ) -> List[float]:
        """
        Get individual trade P&L values for statistical testing.

        Returns list of trade P&L values.
        """
        # Mock: return simulated trade history
        import random
        trades = []
        for _ in range(random.randint(50, 200)):
            # Random trade outcome with bias towards profit
            if random.random() < 0.55:
                trades.append(random.uniform(10, 200))
            else:
                trades.append(random.uniform(-200, -10))
        return trades


# Singleton instance
_metrics_service = VariantMetricsService()


# ============================================================================
# Statistical Significance Engine
# ============================================================================


def calculate_statistical_significance(
    trades_a: List[float],
    trades_b: List[float],
) -> Optional[StatisticalSignificance]:
    """
    Calculate statistical significance using two-sample t-test.

    Returns StatisticalSignificance if enough data, None otherwise.
    """
    if len(trades_a) < 50 or len(trades_b) < 50:
        return None

    # One-sided t-test (A vs B)
    t_stat, p_value = stats.ttest_ind(trades_a, trades_b, equal_var=False)

    # Determine winner
    mean_a = np.mean(trades_a)
    mean_b = np.mean(trades_b)

    if p_value < 0.05:
        if mean_a > mean_b:
            winner = "A"
        elif mean_b > mean_a:
            winner = "B"
        else:
            winner = None
    else:
        winner = None

    return StatisticalSignificance(
        p_value=round(p_value, 4),
        is_significant=p_value < 0.05,
        winner=winner,
        confidence_level=round((1 - p_value) * 100, 2),
        sample_size_a=len(trades_a),
        sample_size_b=len(trades_b),
    )


# ============================================================================
# Endpoints
# ============================================================================


@router.get("/variants/{strategy_id}/compare")
async def compare_variants(
    strategy_id: str,
    variant_a: str = Query(..., description="First variant ID"),
    variant_b: str = Query(..., description="Second variant ID"),
) -> ABComparisonResponse:
    """
    Compare two strategy variants.

    GET /api/strategies/variants/{strategy_id}/compare?variant_a=X&variant_b=Y

    Returns side-by-side metrics and statistical significance.
    """
    logger.info(f"Comparing variants {variant_a} vs {variant_b} for strategy {strategy_id}")

    # Get metrics for both variants
    metrics_a = await _metrics_service.get_variant_metrics(strategy_id, variant_a)
    metrics_b = await _metrics_service.get_variant_metrics(strategy_id, variant_b)

    # Get trade history for statistical significance
    trades_a = await _metrics_service.get_trade_history(strategy_id, variant_a)
    trades_b = await _metrics_service.get_trade_history(strategy_id, variant_b)

    # Calculate statistical significance
    stat_sig = calculate_statistical_significance(trades_a, trades_b)

    return ABComparisonResponse(
        strategy_id=strategy_id,
        variant_a=variant_a,
        variant_b=variant_b,
        metrics_a=metrics_a,
        metrics_b=metrics_b,
        statistical_significance=stat_sig,
        timestamp=datetime.utcnow().isoformat(),
    )


@router.get("/variants/{strategy_id}/metrics")
async def get_variant_metrics(
    strategy_id: str,
    variant_id: str = Query(..., description="Variant ID"),
) -> VariantMetrics:
    """
    Get metrics for a single variant.

    GET /api/strategies/variants/{strategy_id}/metrics?variant_id=X

    Returns: P&L, trade count, drawdown, Sharpe ratio.
    """
    return await _metrics_service.get_variant_metrics(strategy_id, variant_id)


# ============================================================================
# WebSocket for Real-Time Updates
# ============================================================================


class ConnectionManager:
    """WebSocket connection manager for A/B race updates."""

    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, strategy_id: str):
        await websocket.accept()
        if strategy_id not in self.active_connections:
            self.active_connections[strategy_id] = []
        self.active_connections[strategy_id].append(websocket)

    def disconnect(self, websocket: WebSocket, strategy_id: str):
        if strategy_id in self.active_connections:
            self.active_connections[strategy_id].remove(websocket)

    async def broadcast(self, strategy_id: str, message: Dict[str, Any]):
        if strategy_id in self.active_connections:
            for connection in self.active_connections[strategy_id]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error(f"Error broadcasting to {strategy_id}: {e}")


_manager = ConnectionManager()


@router.websocket("/ws/ab-race/{strategy_id}")
async def ab_race_websocket(
    websocket: WebSocket,
    strategy_id: str,
):
    """
    WebSocket endpoint for real-time A/B race board updates.

    Sends updated metrics every 5 seconds.
    """
    await _manager.connect(websocket, strategy_id)
    logger.info(f"WebSocket connected for A/B race: {strategy_id}")

    try:
        while True:
            # Send updated comparison data every 5 seconds
            # In production: query live data sources
            await asyncio.sleep(5)

            # Get current metrics (would be real-time data in production)
            # For demo, generate mock updates
            import random

            update = {
                "type": "metrics_update",
                "strategy_id": strategy_id,
                "timestamp": datetime.utcnow().isoformat(),
                "metrics": {
                    "variant_a": {
                        "pnl": round(random.uniform(-500, 2000), 2),
                        "trade_count": random.randint(20, 200),
                    },
                    "variant_b": {
                        "pnl": round(random.uniform(-500, 2000), 2),
                        "trade_count": random.randint(20, 200),
                    },
                },
            }

            await _manager.broadcast(strategy_id, update)

    except WebSocketDisconnect:
        _manager.disconnect(websocket, strategy_id)
        logger.info(f"WebSocket disconnected: {strategy_id}")
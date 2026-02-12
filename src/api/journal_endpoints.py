"""
Trade Journal API Endpoints

Provides access to complete trade history with full context including
regime, chaos, Kelly sizing, house money adjustments, and trade reasoning.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import random

router = APIRouter(prefix="/api/journal", tags=["journal"])

# Data models
class RegimeContext(BaseModel):
    quality: float
    trend: str  # 'bullish', 'bearish', 'ranging'
    chaos: float

class KellyContext(BaseModel):
    fraction: float
    edge: float
    odds: float

class HouseMoneyContext(BaseModel):
    enabled: bool
    amount: float

class TradeContext(BaseModel):
    regime: RegimeContext
    kelly: KellyContext
    houseMoney: HouseMoneyContext
    sentiment: Optional[str] = None

class Trade(BaseModel):
    id: str
    timestamp: str
    symbol: str
    type: str  # 'BUY', 'SELL'
    lots: float
    openPrice: float
    closePrice: float
    profit: float
    strategy: str
    status: str  # 'open', 'closed', 'pending'
    reason: str
    context: TradeContext

class TradeFilters(BaseModel):
    symbol: Optional[str] = None
    status: Optional[str] = None
    dateFrom: Optional[str] = None
    dateTo: Optional[str] = None
    minProfit: Optional[float] = None
    maxProfit: Optional[float] = None

class TradeStatistics(BaseModel):
    total: int
    wins: int
    losses: int
    winRate: float
    totalProfit: float
    avgProfit: float
    largestWin: float
    largestLoss: float

# Mock data for development
def get_mock_trades() -> List[Dict[str, Any]]:
    """Get mock trade journal data."""
    base_time = datetime.utcnow()

    return [
        {
            "id": "trade-1",
            "timestamp": (base_time - timedelta(hours=1)).isoformat(),
            "symbol": "EURUSD",
            "type": "BUY",
            "lots": 0.05,
            "openPrice": 1.0876,
            "closePrice": 1.0912,
            "profit": 18.00,
            "strategy": "ICT Scalper",
            "status": "closed",
            "reason": "FVG setup confirmed with ATR filter",
            "context": {
                "regime": {"quality": 0.82, "trend": "bullish", "chaos": 18.5},
                "kelly": {"fraction": 0.025, "edge": 0.52, "odds": 2.0},
                "houseMoney": {"enabled": True, "amount": 87.15},
                "sentiment": "London session, low spread, FVG at key level"
            }
        },
        {
            "id": "trade-2",
            "timestamp": (base_time - timedelta(hours=2)).isoformat(),
            "symbol": "GBPUSD",
            "type": "SELL",
            "lots": 0.04,
            "openPrice": 1.2654,
            "closePrice": 1.2678,
            "profit": -9.60,
            "strategy": "SMC Reversal",
            "status": "closed",
            "reason": "Order block fail, volatility spike",
            "context": {
                "regime": {"quality": 0.65, "trend": "ranging", "chaos": 35.2},
                "kelly": {"fraction": 0.02, "edge": 0.45, "odds": 1.8},
                "houseMoney": {"enabled": True, "amount": 77.55},
                "sentiment": "Late session, widening spreads"
            }
        },
        {
            "id": "trade-3",
            "timestamp": (base_time - timedelta(hours=3)).isoformat(),
            "symbol": "USDJPY",
            "type": "BUY",
            "lots": 0.03,
            "openPrice": 149.85,
            "closePrice": 150.12,
            "profit": 8.10,
            "strategy": "ICT Scalper",
            "status": "closed",
            "reason": "Premium discount zone with RSI divergence",
            "context": {
                "regime": {"quality": 0.78, "trend": "bullish", "chaos": 22.1},
                "kelly": {"fraction": 0.022, "edge": 0.50, "odds": 1.9},
                "houseMoney": {"enabled": True, "amount": 87.15},
                "sentiment": "Asian session, clean move"
            }
        },
        {
            "id": "trade-4",
            "timestamp": (base_time - timedelta(hours=5)).isoformat(),
            "symbol": "EURUSD",
            "type": "BUY",
            "lots": 0.06,
            "openPrice": 1.0845,
            "closePrice": 1.0898,
            "profit": 31.80,
            "strategy": "ICT Scalper",
            "status": "closed",
            "reason": "London session breakout with volume confirmation",
            "context": {
                "regime": {"quality": 0.88, "trend": "bullish", "chaos": 15.2},
                "kelly": {"fraction": 0.028, "edge": 0.55, "odds": 2.1},
                "houseMoney": {"enabled": True, "amount": 117.95},
                "sentiment": "Strong momentum, low news risk"
            }
        },
        {
            "id": "trade-5",
            "timestamp": (base_time - timedelta(hours=6)).isoformat(),
            "symbol": "GBPUSD",
            "type": "SELL",
            "lots": 0.05,
            "openPrice": 1.2680,
            "closePrice": 1.2625,
            "profit": 27.50,
            "strategy": "SMC Reversal",
            "status": "closed",
            "reason": "CHoCH confirmation at key resistance",
            "context": {
                "regime": {"quality": 0.75, "trend": "bearish", "chaos": 28.5},
                "kelly": {"fraction": 0.024, "edge": 0.48, "odds": 1.9},
                "houseMoney": {"enabled": True, "amount": 145.45},
                "sentiment": "Clean rejection, good risk/reward"
            }
        }
    ]

def calculate_stats(trades: List[Dict[str, Any]]) -> TradeStatistics:
    """Calculate trade statistics."""
    closed_trades = [t for t in trades if t["status"] == "closed"]

    if not closed_trades:
        return TradeStatistics(
            total=0, wins=0, losses=0, winRate=0.0,
            totalProfit=0.0, avgProfit=0.0, largestWin=0.0, largestLoss=0.0
        )

    wins = [t for t in closed_trades if t["profit"] > 0]
    losses = [t for t in closed_trades if t["profit"] < 0]

    profits = [t["profit"] for t in closed_trades]

    return TradeStatistics(
        total=len(closed_trades),
        wins=len(wins),
        losses=len(losses),
        winRate=(len(wins) / len(closed_trades)) * 100 if closed_trades else 0,
        totalProfit=sum(profits),
        avgProfit=sum(profits) / len(profits),
        largestWin=max(profits) if profits else 0,
        largestLoss=min(profits) if profits else 0
    )

def filter_trades(trades: List[Dict[str, Any]], filters: TradeFilters) -> List[Dict[str, Any]]:
    """Apply filters to trade list."""
    filtered = trades

    if filters.symbol:
        filtered = [t for t in filtered if t["symbol"] == filters.symbol]

    if filters.status:
        filtered = [t for t in filtered if t["status"] == filters.status]

    if filters.dateFrom:
        date_from = datetime.fromisoformat(filters.dateFrom)
        filtered = [t for t in filtered if datetime.fromisoformat(t["timestamp"]) >= date_from]

    if filters.dateTo:
        date_to = datetime.fromisoformat(filters.dateTo)
        filtered = [t for t in filtered if datetime.fromisoformat(t["timestamp"]) <= date_to]

    if filters.minProfit is not None:
        filtered = [t for t in filtered if t["profit"] >= filters.minProfit]

    if filters.maxProfit is not None:
        filtered = [t for t in filtered if t["profit"] <= filters.maxProfit]

    return filtered

# Endpoints

@router.get("/trades")
async def get_trades(
    symbol: Optional[str] = None,
    status: Optional[str] = None,
    dateFrom: Optional[str] = None,
    dateTo: Optional[str] = None,
    minProfit: Optional[float] = None,
    maxProfit: Optional[float] = None,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """Get trade journal with optional filters."""
    trades = get_mock_trades()

    filters = TradeFilters(
        symbol=symbol,
        status=status,
        dateFrom=dateFrom,
        dateTo=dateTo,
        minProfit=minProfit,
        maxProfit=maxProfit
    )

    filtered = filter_trades(trades, filters)
    return filtered[:limit]

@router.get("/trades/{trade_id}")
async def get_trade(trade_id: str) -> Dict[str, Any]:
    """Get a specific trade by ID."""
    trades = get_mock_trades()
    trade = next((t for t in trades if t["id"] == trade_id), None)

    if not trade:
        raise HTTPException(status_code=404, detail="Trade not found")

    return trade

@router.get("/statistics")
async def get_trade_statistics(
    symbol: Optional[str] = None,
    dateFrom: Optional[str] = None,
    dateTo: Optional[str] = None
) -> TradeStatistics:
    """Get trade statistics with optional filters."""
    trades = get_mock_trades()

    filters = TradeFilters(
        symbol=symbol,
        dateFrom=dateFrom,
        dateTo=dateTo
    )

    filtered = filter_trades(trades, filters)
    return calculate_stats(filtered)

@router.get("/export/{trade_id}")
async def export_trade(trade_id: str) -> Dict[str, Any]:
    """Export a trade with full context as JSON."""
    trades = get_mock_trades()
    trade = next((t for t in trades if t["id"] == trade_id), None)

    if not trade:
        raise HTTPException(status_code=404, detail="Trade not found")

    return {
        "trade": trade,
        "exported_at": datetime.utcnow().isoformat(),
        "format": "json"
    }

@router.post("/export")
async def export_journal(
    symbol: Optional[str] = None,
    dateFrom: Optional[str] = None,
    dateTo: Optional[str] = None
) -> Dict[str, Any]:
    """Export entire trade journal as JSON."""
    trades = get_mock_trades()

    filters = TradeFilters(
        symbol=symbol,
        dateFrom=dateFrom,
        dateTo=dateTo
    )

    filtered = filter_trades(trades, filters)
    stats = calculate_stats(filtered)

    return {
        "trades": filtered,
        "statistics": stats.dict(),
        "exported_at": datetime.utcnow().isoformat(),
        "format": "json"
    }

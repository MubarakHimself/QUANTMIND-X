"""
Router State API Endpoints

Provides real-time information about the Strategy Router including
auction queue, bot rankings, market regime, and correlations.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import random

router = APIRouter(prefix="/api/router", tags=["router"])

# Data models
class MarketRegime(BaseModel):
    quality: float
    trend: str  # 'bullish', 'bearish', 'ranging'
    chaos: float
    volatility: str  # 'low', 'medium', 'high'

class SymbolData(BaseModel):
    symbol: str
    price: float
    change: float
    spread: float

class BotSignal(BaseModel):
    id: str
    name: str
    symbol: str
    status: str  # 'idle', 'ready', 'paused', 'quarantined'
    signalStrength: float
    conditions: List[str]
    score: float
    lastSignal: Optional[str]

class Auction(BaseModel):
    id: str
    timestamp: str
    participants: List[str]
    winner: str
    winningScore: float
    status: str  # 'pending', 'completed', 'failed'

class RouterState(BaseModel):
    active: bool
    mode: str  # 'auction', 'priority', 'round-robin'
    auctionInterval: int
    lastAuction: Optional[str]
    queuedSignals: int
    activeAuctions: int

class HouseMoneyState(BaseModel):
    dailyProfit: float
    threshold: float
    houseMoneyAmount: float
    mode: str  # 'conservative', 'normal', 'aggressive'

# Mock data for development
def get_mock_router_state() -> Dict[str, Any]:
    """Get mock router state for development."""
    return {
        "active": True,
        "mode": "auction",
        "auctionInterval": 5000,
        "lastAuction": datetime.utcnow().isoformat(),
        "queuedSignals": 0,
        "activeAuctions": 1
    }

def get_mock_market_state() -> Dict[str, Any]:
    """Get mock market state."""
    return {
        "regime": {
            "quality": 0.82,
            "trend": "bullish",
            "chaos": 18.5,
            "volatility": "medium"
        },
        "symbols": [
            {"symbol": "EURUSD", "price": 1.0876, "change": 0.12, "spread": 1.2},
            {"symbol": "GBPUSD", "price": 1.2654, "change": -0.08, "spread": 1.5},
            {"symbol": "USDJPY", "price": 149.85, "change": 0.25, "spread": 0.8}
        ]
    }

def get_mock_bots() -> List[Dict[str, Any]]:
    """Get mock bot signals."""
    return [
        {
            "id": "ict-eur",
            "name": "ICT Scalper",
            "symbol": "EURUSD",
            "status": "ready",
            "signalStrength": 0.85,
            "conditions": ["fvg", "order_block", "london_session"],
            "score": 8.5,
            "lastSignal": datetime.utcnow().isoformat()
        },
        {
            "id": "smc-gbp",
            "name": "SMC Reversal",
            "symbol": "GBPUSD",
            "status": "ready",
            "signalStrength": 0.72,
            "conditions": ["choch", "bos", "session_filter"],
            "score": 7.2,
            "lastSignal": datetime.utcnow().isoformat()
        }
    ]

def get_mock_auctions() -> List[Dict[str, Any]]:
    """Get mock auction queue."""
    return [
        {
            "id": "auction-1",
            "timestamp": datetime.utcnow().isoformat(),
            "participants": ["ict-eur", "smc-gbp"],
            "winner": "ict-eur",
            "winningScore": 8.5,
            "status": "completed"
        }
    ]

def get_mock_rankings() -> Dict[str, List[Dict[str, Any]]]:
    """Get mock bot rankings."""
    return {
        "daily": [
            {"botId": "ict-eur", "name": "ICT Scalper", "profit": 245.80, "trades": 12, "winRate": 75.0},
            {"botId": "smc-gbp", "name": "SMC Reversal", "profit": 128.50, "trades": 8, "winRate": 62.5},
            {"botId": "breakthrough-eur", "name": "Breakthrough", "profit": 0.0, "trades": 0, "winRate": 0.0}
        ],
        "weekly": [
            {"botId": "ict-eur", "name": "ICT Scalper", "profit": 1245.60, "trades": 58, "winRate": 70.7},
            {"botId": "smc-gbp", "name": "SMC Reversal", "profit": 678.30, "trades": 42, "winRate": 64.3}
        ]
    }

def get_mock_correlations() -> List[Dict[str, Any]]:
    """Get mock correlation data."""
    return [
        {"pair": "EURUSD/GBPUSD", "value": 0.72, "status": "warning"},
        {"pair": "EURUSD/USDJPY", "value": -0.45, "status": "ok"},
        {"pair": "GBPUSD/USDJPY", "value": -0.38, "status": "ok"}
    ]

def get_mock_house_money() -> Dict[str, Any]:
    """Get mock house money state."""
    return {
        "dailyProfit": 374.30,
        "threshold": 0.5,
        "houseMoneyAmount": 187.15,
        "mode": "aggressive"
    }

# Endpoints

@router.get("/state")
async def get_router_state() -> RouterState:
    """Get current router state."""
    return RouterState(**get_mock_router_state())

@router.post("/toggle")
async def toggle_router(active: bool = None) -> Dict[str, Any]:
    """Toggle router on/off."""
    state = get_mock_router_state()
    if active is not None:
        state["active"] = active
    return {"success": True, "active": state["active"]}

@router.get("/market")
async def get_market_state() -> Dict[str, Any]:
    """Get current market state including regime and symbols."""
    return get_mock_market_state()

@router.get("/bots")
async def get_bot_signals() -> List[BotSignal]:
    """Get current bot signals."""
    bots = get_mock_bots()
    return [BotSignal(**bot) for bot in bots]

@router.get("/auctions")
async def get_auction_queue() -> List[Auction]:
    """Get current auction queue."""
    auctions = get_mock_auctions()
    return [Auction(**auction) for auction in auctions]

@router.post("/auction")
async def run_auction() -> Dict[str, Any]:
    """Run a new auction round."""
    # Mock auction result
    bots = get_mock_bots()
    ready_bots = [b for b in bots if b["status"] == "ready"]

    if not ready_bots:
        raise HTTPException(status_code=400, detail="No bots ready for auction")

    # Select winner based on score
    winner = max(ready_bots, key=lambda x: x["score"])

    auction = {
        "id": f"auction-{datetime.utcnow().timestamp()}",
        "timestamp": datetime.utcnow().isoformat(),
        "participants": [b["id"] for b in ready_bots],
        "winner": winner["id"],
        "winningScore": winner["score"],
        "status": "completed"
    }

    return {"success": True, "auction": auction}

@router.get("/rankings")
async def get_rankings(period: str = "daily") -> List[Dict[str, Any]]:
    """Get bot rankings for specified period."""
    rankings = get_mock_rankings()
    if period not in rankings:
        raise HTTPException(status_code=400, detail="Invalid period. Use 'daily' or 'weekly'")
    return rankings[period]

@router.get("/correlations")
async def get_correlations() -> List[Dict[str, Any]]:
    """Get current symbol correlations."""
    return get_mock_correlations()

@router.get("/house-money")
async def get_house_money_state() -> HouseMoneyState:
    """Get current house money state."""
    return HouseMoneyState(**get_mock_house_money())

@router.post("/settings")
async def update_router_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Update router settings."""
    # In production, this would save to database
    return {"success": True, "settings": settings}

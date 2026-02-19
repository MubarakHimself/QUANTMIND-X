"""
Trade Journal API Endpoints

Provides access to complete trade history with full context including
regime, chaos, Kelly sizing, house money adjustments, and trade reasoning.
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

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
    mode: str  # 'demo' or 'live'

class TradeFilters(BaseModel):
    symbol: Optional[str] = None
    status: Optional[str] = None
    dateFrom: Optional[str] = None
    dateTo: Optional[str] = None
    minProfit: Optional[float] = None
    maxProfit: Optional[float] = None
    mode: Optional[str] = None

class TradeStatistics(BaseModel):
    total: int
    wins: int
    losses: int
    winRate: float
    totalProfit: float
    avgProfit: float
    largestWin: float
    largestLoss: float


def get_db_session():
    """Get or create database session for DB access."""
    try:
        from src.database.engine import get_session
        return get_session()
    except Exception as e:
        print(f"Error getting database session: {e}")
        return None


def query_trade_journal(
    symbol: Optional[str] = None,
    status: Optional[str] = None,
    dateFrom: Optional[datetime] = None,
    dateTo: Optional[datetime] = None,
    minProfit: Optional[float] = None,
    maxProfit: Optional[float] = None,
    mode: Optional[str] = None,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """Query trade journal from database with filters."""
    session = get_db_session()
    if session is None:
        return []
    
    try:
        from src.database.models import TradeJournal, TradingMode
        from sqlalchemy import desc
        
        query = session.query(TradeJournal)
        
        # Apply filters
        if symbol:
            query = query.filter(TradeJournal.symbol == symbol)
        
        if status:
            if status == 'closed':
                query = query.filter(TradeJournal.pnl.isnot(None))
            elif status == 'open':
                query = query.filter(TradeJournal.pnl.is_(None))
        
        if dateFrom:
            query = query.filter(TradeJournal.timestamp >= dateFrom)
        
        if dateTo:
            query = query.filter(TradeJournal.timestamp <= dateTo)
        
        if minProfit is not None:
            query = query.filter(TradeJournal.pnl >= minProfit)
        
        if maxProfit is not None:
            query = query.filter(TradeJournal.pnl <= maxProfit)
        
        if mode and mode != 'all':
            trading_mode = TradingMode.DEMO if mode == 'demo' else TradingMode.LIVE
            query = query.filter(TradeJournal.mode == trading_mode)
        
        # Order by timestamp descending and limit
        query = query.order_by(desc(TradeJournal.timestamp)).limit(limit)
        
        trades = query.all()
        
        # Convert to list of dicts
        result = []
        for trade in trades:
            # Determine status based on pnl
            trade_status = 'closed' if trade.pnl is not None else 'open'
            
            # Determine direction from trade.direction
            direction = trade.direction if trade.direction else 'BUY'
            
            # Convert mode to string
            trade_mode = 'demo' if trade.mode == TradingMode.DEMO else 'live'
            
            result.append({
                "id": str(trade.id),
                "timestamp": trade.timestamp.isoformat() if trade.timestamp else datetime.now().isoformat(),
                "symbol": trade.symbol,
                "type": direction,
                "lots": trade.lot_size,
                "openPrice": trade.entry_price,
                "closePrice": trade.exit_price,
                "profit": trade.pnl if trade.pnl is not None else 0.0,
                "strategy": trade.bot_id,
                "status": trade_status,
                "reason": trade.exit_reason or "",
                "context": {
                    "regime": {
                        "quality": 0.5,  # Default values if not stored
                        "trend": trade.regime or "unknown",
                        "chaos": trade.chaos_score or 0.0
                    },
                    "kelly": {
                        "fraction": trade.kelly_recommendation or 0.0,
                        "edge": 0.5,
                        "odds": 2.0
                    },
                    "houseMoney": {
                        "enabled": trade.house_money_adjustment,
                        "amount": 0.0
                    },
                    "sentiment": trade.regime or "No sentiment data"
                },
                "mode": trade_mode
            })
        
        session.close()
        return result
        
    except Exception as e:
        print(f"Error querying trade journal: {e}")
        if session:
            session.close()
        return []


def calculate_stats(trades: List[Dict[str, Any]], mode_filter: Optional[str] = None) -> TradeStatistics:
    """Calculate trade statistics."""
    # Filter by mode if specified
    if mode_filter and mode_filter != 'all':
        trades = [t for t in trades if t.get('mode') == mode_filter]
    
    closed_trades = [t for t in trades if t.get("status") == "closed"]

    if not closed_trades:
        return TradeStatistics(
            total=0, wins=0, losses=0, winRate=0.0,
            totalProfit=0.0, avgProfit=0.0, largestWin=0.0, largestLoss=0.0
        )

    wins = [t for t in closed_trades if t.get("profit", 0) > 0]
    losses = [t for t in closed_trades if t.get("profit", 0) < 0]

    profits = [t.get("profit", 0) for t in closed_trades]

    return TradeStatistics(
        total=len(closed_trades),
        wins=len(wins),
        losses=len(losses),
        winRate=(len(wins) / len(closed_trades)) * 100 if closed_trades else 0,
        totalProfit=sum(profits),
        avgProfit=sum(profits) / len(profits) if profits else 0,
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
        filtered = [t for t in filtered if t.get("profit", 0) >= filters.minProfit]

    if filters.maxProfit is not None:
        filtered = [t for t in filtered if t.get("profit", 0) <= filters.maxProfit]
    
    if filters.mode and filters.mode != 'all':
        filtered = [t for t in filtered if t.get("mode") == filters.mode]

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
    mode: Optional[str] = Query(None, description="Filter by mode: 'demo', 'live', or 'all'"),
    limit: int = 100
) -> List[Dict[str, Any]]:
    """Get trade journal with optional filters."""
    # Parse dates if provided
    parsed_dateFrom = None
    parsed_dateTo = None
    
    if dateFrom:
        try:
            parsed_dateFrom = datetime.fromisoformat(dateFrom)
        except ValueError:
            pass
    
    if dateTo:
        try:
            parsed_dateTo = datetime.fromisoformat(dateTo)
        except ValueError:
            pass
    
    # Query database
    trades = query_trade_journal(
        symbol=symbol,
        status=status,
        dateFrom=parsed_dateFrom,
        dateTo=parsed_dateTo,
        minProfit=minProfit,
        maxProfit=maxProfit,
        mode=mode,
        limit=limit
    )
    
    # If no trades found in database, return empty list
    # (removed fallback to mock data)
    return trades


@router.get("/trades/{trade_id}")
async def get_trade(trade_id: str) -> Dict[str, Any]:
    """Get a specific trade by ID."""
    session = get_db_session()
    if session is None:
        raise HTTPException(status_code=500, detail="Database unavailable")
    
    try:
        from src.database.models import TradeJournal, TradingMode
        
        trade = session.query(TradeJournal).filter(TradeJournal.id == int(trade_id)).first()
        
        if not trade:
            raise HTTPException(status_code=404, detail="Trade not found")
        
        direction = trade.direction if trade.direction else 'BUY'
        trade_status = 'closed' if trade.pnl is not None else 'open'
        trade_mode = 'demo' if trade.mode == TradingMode.DEMO else 'live'
        
        result = {
            "id": str(trade.id),
            "timestamp": trade.timestamp.isoformat() if trade.timestamp else datetime.now().isoformat(),
            "symbol": trade.symbol,
            "type": direction,
            "lots": trade.lot_size,
            "openPrice": trade.entry_price,
            "closePrice": trade.exit_price,
            "profit": trade.pnl if trade.pnl is not None else 0.0,
            "strategy": trade.bot_id,
            "status": trade_status,
            "reason": trade.exit_reason or "",
            "context": {
                "regime": {
                    "quality": 0.5,
                    "trend": trade.regime or "unknown",
                    "chaos": trade.chaos_score or 0.0
                },
                "kelly": {
                    "fraction": trade.kelly_recommendation or 0.0,
                    "edge": 0.5,
                    "odds": 2.0
                },
                "houseMoney": {
                    "enabled": trade.house_money_adjustment,
                    "amount": 0.0
                },
                "sentiment": trade.regime or "No sentiment data"
            },
            "mode": trade_mode
        }
        
        session.close()
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting trade: {e}")
        if session:
            session.close()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics")
async def get_trade_statistics(
    symbol: Optional[str] = None,
    dateFrom: Optional[str] = None,
    dateTo: Optional[str] = None,
    mode: Optional[str] = Query(None, description="Filter by mode: 'demo', 'live', or 'all'")
) -> TradeStatistics:
    """Get trade statistics with optional filters."""
    # Parse dates if provided
    parsed_dateFrom = None
    parsed_dateTo = None
    
    if dateFrom:
        try:
            parsed_dateFrom = datetime.fromisoformat(dateFrom)
        except ValueError:
            pass
    
    if dateTo:
        try:
            parsed_dateTo = datetime.fromisoformat(dateTo)
        except ValueError:
            pass
    
    # Query database
    trades = query_trade_journal(
        symbol=symbol,
        dateFrom=parsed_dateFrom,
        dateTo=parsed_dateTo,
        mode=mode,
        limit=1000  # Get more trades for stats
    )
    
    return calculate_stats(trades, mode_filter=mode)


@router.get("/export/{trade_id}")
async def export_trade(trade_id: str) -> Dict[str, Any]:
    """Export a trade with full context as JSON."""
    trade = await get_trade(trade_id)
    
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
    dateTo: Optional[str] = None,
    mode: Optional[str] = None
) -> Dict[str, Any]:
    """Export entire trade journal as JSON."""
    # Parse dates if provided
    parsed_dateFrom = None
    parsed_dateTo = None
    
    if dateFrom:
        try:
            parsed_dateFrom = datetime.fromisoformat(dateFrom)
        except ValueError:
            pass
    
    if dateTo:
        try:
            parsed_dateTo = datetime.fromisoformat(dateTo)
        except ValueError:
            pass
    
    # Query database
    trades = query_trade_journal(
        symbol=symbol,
        dateFrom=parsed_dateFrom,
        dateTo=parsed_dateTo,
        mode=mode,
        limit=10000  # Get all trades for export
    )
    
    stats = calculate_stats(trades, mode_filter=mode)

    return {
        "trades": trades,
        "statistics": stats.dict(),
        "exported_at": datetime.utcnow().isoformat(),
        "format": "json"
    }

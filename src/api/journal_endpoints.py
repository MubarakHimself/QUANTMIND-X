"""
Trade Journal API Endpoints

Provides access to complete trade history with full context including
regime, chaos, Kelly sizing, house money adjustments, and trade reasoning.
"""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta, timezone
import io
import csv

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


class ReviewCandidate(BaseModel):
    bot_id: str
    closed_trades: int
    losses: int
    win_rate: float
    net_pnl: float
    consecutive_losses: int
    is_quarantined: bool
    quarantine_reason: Optional[str] = None
    last_trade_time: Optional[str] = None


class JournalReviewSummaryResponse(BaseModel):
    generated_at: str
    total_reviewed: int
    quarantined_bots: int
    pressure_accounts: int
    review_watchlist: List[ReviewCandidate]


class TradeAnnotationRequest(BaseModel):
    """Request model for trade annotation."""
    note: str = Field(..., description="Annotation note for the trade")


class TradeAnnotationResponse(BaseModel):
    """Response model for trade annotation."""
    trade_id: int
    note: str
    annotated_at: str


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

            # Format entry/exit times for Trading Journal (Story 9-5)
            entry_time = trade.timestamp.isoformat() if trade.timestamp else ""
            exit_time = ""
            if trade_status == 'closed' and trade.timestamp and trade.duration_minutes:
                exit_dt = trade.timestamp + timedelta(minutes=trade.duration_minutes)
                exit_time = exit_dt.isoformat()

            result.append({
                "id": str(trade.id),
                "entryTime": entry_time,
                "exitTime": exit_time,
                "symbol": trade.symbol,
                "direction": direction,
                "pnl": trade.pnl if trade.pnl is not None else 0.0,
                "session": trade_mode,
                "holdDuration": trade.duration_minutes,
                "eaName": trade.bot_id,
                # Detail view fields
                "entryPrice": trade.entry_price,
                "exitPrice": trade.exit_price,
                "spreadAtEntry": trade.spread_at_entry,
                "slippage": 0.0,  # Not stored, default to 0
                "strategyVersion": str(trade.strategy_folder_id) if trade.strategy_folder_id else "N/A",
                # Annotation fields
                "note": trade.note or "",
                "annotatedAt": trade.annotated_at.isoformat() if trade.annotated_at else "",
                # Additional context
                "lots": trade.lot_size,
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
            })

        return result

    except Exception as e:
        print(f"Error querying trade journal: {e}")
        return []
    finally:
        session.close()


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

    wins = [t for t in closed_trades if t.get("pnl", 0) > 0]
    losses = [t for t in closed_trades if t.get("pnl", 0) < 0]

    profits = [t.get("pnl", 0) for t in closed_trades]

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


def build_review_summary(limit: int = 200) -> JournalReviewSummaryResponse:
    """Build a lightweight bot review/watchlist summary from journal and breaker state."""
    trades = query_trade_journal(limit=limit)
    by_bot: Dict[str, Dict[str, Any]] = {}

    for trade in trades:
        bot_id = trade.get("eaName") or trade.get("strategy") or "unknown"
        aggregate = by_bot.setdefault(
            bot_id,
            {
                "bot_id": bot_id,
                "closed_trades": 0,
                "losses": 0,
                "wins": 0,
                "net_pnl": 0.0,
                "last_trade_time": trade.get("entryTime") or trade.get("timestamp"),
            },
        )

        if trade.get("status") == "closed":
            aggregate["closed_trades"] += 1
            pnl = float(trade.get("pnl", 0.0) or 0.0)
            aggregate["net_pnl"] += pnl
            if pnl < 0:
                aggregate["losses"] += 1
            elif pnl > 0:
                aggregate["wins"] += 1

    breaker_states: Dict[str, Dict[str, Any]] = {}
    try:
        from src.router.bot_circuit_breaker import BotCircuitBreakerManager

        breaker_states = {
            state["bot_id"]: state
            for state in BotCircuitBreakerManager().get_all_states()
        }
    except Exception:
        breaker_states = {}

    pressure_accounts = 0
    try:
        from src.router.account_monitor import get_account_monitor

        pressure_accounts = len(get_account_monitor().get_accounts_at_risk())
    except Exception:
        pressure_accounts = 0

    candidates: List[ReviewCandidate] = []
    quarantined_count = 0
    for bot_id, aggregate in by_bot.items():
        state = breaker_states.get(bot_id, {})
        is_quarantined = bool(state.get("is_quarantined", False))
        if is_quarantined:
            quarantined_count += 1

        closed_trades = aggregate["closed_trades"]
        wins = aggregate["wins"]
        candidates.append(
            ReviewCandidate(
                bot_id=bot_id,
                closed_trades=closed_trades,
                losses=aggregate["losses"],
                win_rate=(wins / closed_trades) * 100 if closed_trades else 0.0,
                net_pnl=aggregate["net_pnl"],
                consecutive_losses=int(state.get("consecutive_losses", 0) or 0),
                is_quarantined=is_quarantined,
                quarantine_reason=state.get("quarantine_reason"),
                last_trade_time=aggregate["last_trade_time"],
            )
        )

    candidates.sort(
        key=lambda c: (
            not c.is_quarantined,
            c.net_pnl,
            -c.consecutive_losses,
            -c.losses,
        )
    )

    return JournalReviewSummaryResponse(
        generated_at=datetime.now(timezone.utc).isoformat(),
        total_reviewed=len(candidates),
        quarantined_bots=quarantined_count,
        pressure_accounts=pressure_accounts,
        review_watchlist=candidates[:10],
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

        return result

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting trade: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()


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


@router.get("/review-summary", response_model=JournalReviewSummaryResponse)
async def get_review_summary(limit: int = Query(200, ge=1, le=1000)) -> JournalReviewSummaryResponse:
    """Get bot review/watchlist context for weekly journal and agentic analysis."""
    return build_review_summary(limit=limit)


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
        "statistics": stats.model_dump(),
        "exported_at": datetime.utcnow().isoformat(),
        "format": "json"
    }


# ============================================================================
# Trading Journal Annotation Endpoints (Story 9-5)
# ============================================================================

@router.post("/trades/{trade_id}/annotation", response_model=TradeAnnotationResponse)
async def create_or_update_annotation(
    trade_id: str,
    annotation: TradeAnnotationRequest
) -> TradeAnnotationResponse:
    """
    Add or update annotation for a trade.

    AC #3: Store annotation with { trade_id, note, annotated_at_utc }
    """
    session = get_db_session()
    if session is None:
        raise HTTPException(status_code=500, detail="Database unavailable")

    try:
        from src.database.models import TradeJournal

        trade = session.query(TradeJournal).filter(TradeJournal.id == int(trade_id)).first()

        if not trade:
            raise HTTPException(status_code=404, detail="Trade not found")

        # Update or create annotation
        trade.note = annotation.note
        trade.annotated_at = datetime.now(timezone.utc)

        session.commit()
        session.close()

        return TradeAnnotationResponse(
            trade_id=int(trade_id),
            note=annotation.note,
            annotated_at=trade.annotated_at.isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        if session:
            session.rollback()
            session.close()
        print(f"Error creating annotation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trades/{trade_id}/annotation", response_model=TradeAnnotationResponse)
async def get_annotation(trade_id: str) -> TradeAnnotationResponse:
    """Get annotation for a trade."""
    session = get_db_session()
    if session is None:
        raise HTTPException(status_code=500, detail="Database unavailable")

    try:
        from src.database.models import TradeJournal

        trade = session.query(TradeJournal).filter(TradeJournal.id == int(trade_id)).first()

        if not trade:
            raise HTTPException(status_code=404, detail="Trade not found")

        session.close()

        return TradeAnnotationResponse(
            trade_id=int(trade_id),
            note=trade.note or "",
            annotated_at=trade.annotated_at.isoformat() if trade.annotated_at else ""
        )

    except HTTPException:
        raise
    except Exception as e:
        if session:
            session.close()
        print(f"Error getting annotation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Trading Journal CSV Export (Story 9-5)
# ============================================================================

@router.get("/trades/export/csv")
async def export_trades_csv(
    symbol: Optional[str] = None,
    direction: Optional[str] = None,
    session: Optional[str] = None,
    ea_name: Optional[str] = None,
    dateFrom: Optional[str] = None,
    dateTo: Optional[str] = None
):
    """
    Export trades as CSV file.

    AC #4: CSV export with all filtered trades including annotations
    """
    # Parse dates
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

    # Query database with filters
    trades = query_trade_journal(
        symbol=symbol,
        dateFrom=parsed_dateFrom,
        dateTo=parsed_dateTo,
        mode=session,
        limit=10000
    )

    # Apply additional filters not in query_trade_journal
    if direction:
        trades = [t for t in trades if t.get("direction", "").upper() == direction.upper()]

    if ea_name:
        trades = [t for t in trades if ea_name.lower() in t.get("eaName", "").lower()]

    # Generate CSV
    output = io.StringIO()
    writer = csv.writer(output)

    # Header row
    writer.writerow([
        "ID", "Entry Time (UTC)", "Exit Time (UTC)", "Symbol", "Direction",
        "P&L", "Session", "Hold Duration (min)", "EA Name",
        "Entry Price", "Exit Price", "Spread at Entry", "Slippage", "Strategy Version",
        "Note", "Annotated At (UTC)"
    ])

    # Data rows
    for trade in trades:
        writer.writerow([
            trade.get("id", ""),
            trade.get("entryTime", ""),
            trade.get("exitTime", ""),
            trade.get("symbol", ""),
            trade.get("direction", ""),
            trade.get("pnl", 0),
            trade.get("session", ""),
            trade.get("holdDuration", ""),
            trade.get("eaName", ""),
            trade.get("entryPrice", ""),
            trade.get("exitPrice", ""),
            trade.get("spreadAtEntry", ""),
            trade.get("slippage", ""),
            trade.get("strategyVersion", ""),
            trade.get("note", ""),
            trade.get("annotatedAt", "")
        ])

    output.seek(0)

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=trading-journal.csv"}
    )

"""
Trading Results Endpoints — T1 Node

Provides trade result summaries for DPR scoring on T2.
Exposed at /api/trading/trade-results/* — call only from internal T2 node.
"""

from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

router = APIRouter(prefix="/api/trading", tags=["trading-results"])


class TradeResultSummary(BaseModel):
    """Summary of trade results for a single bot."""
    bot_id: str
    symbol: str
    timeframe: str
    total_trades: int
    win_rate: float
    total_pnl: float
    avg_pnl_per_trade: float
    max_drawdown: float
    ev_per_trade: float


class SessionSummaryResponse(BaseModel):
    session_date: Optional[str]
    summaries: List[TradeResultSummary]


def verify_node_token(token: str = None) -> str:
    """Verify internal node token for cross-node auth."""
    import os
    expected = os.getenv("NODE_INTERNAL_TOKEN", "")
    if not expected:
        return token  # Skip verification if not configured
    if token != expected:
        raise HTTPException(status_code=401, detail="Invalid node token")
    return token


@router.get("/trade-results/summary", response_model=SessionSummaryResponse)
async def get_session_summary(
    session_date: Optional[str] = Query(None, description="ISO date string (YYYY-MM-DD)"),
    token: Optional[str] = Query(None, alias="token"),
    _: str = Depends(verify_node_token),
) -> SessionSummaryResponse:
    """
    DPR scoring on T2 calls this once per session at Dead Zone.

    Returns aggregated trade summaries per bot for the specified session date.
    """
    try:
        from src.database.models.trade_record import TradeRecord
        from sqlalchemy import select, func, Integer
        from src.config import get_database_url
        from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession

        db_url = get_database_url()
        async_db_url = db_url.replace("sqlite:///", "sqlite+aiosqlite:///")

        engine = create_async_engine(async_db_url, echo=False)

        async with AsyncSession(engine) as session:
            # Build query
            if session_date:
                from datetime import datetime
                date_obj = datetime.strptime(session_date, "%Y-%m-%d")
                next_day = datetime(date_obj.year, date_obj.month, date_obj.day + 1)

                stmt = (
                    select(
                        TradeRecord.bot_id,
                        TradeRecord.symbol,
                        TradeRecord.timeframe,
                        func.count(TradeRecord.id).label("total_trades"),
                        func.sum(func.cast(TradeRecord.pnl > 0, Integer)).label("wins"),
                        func.sum(TradeRecord.pnl).label("total_pnl"),
                        func.avg(TradeRecord.pnl).label("avg_pnl"),
                        func.max(TradeRecord.closed_at).label("last_trade"),
                    )
                    .where(TradeRecord.closed_at >= date_obj)
                    .where(TradeRecord.closed_at < next_day)
                    .group_by(TradeRecord.bot_id)
                )
            else:
                # Last 24 hours
                from datetime import datetime, timedelta, timezone
                cutoff = datetime.now(timezone.utc) - timedelta(days=1)

                stmt = (
                    select(
                        TradeRecord.bot_id,
                        TradeRecord.symbol,
                        TradeRecord.timeframe,
                        func.count(TradeRecord.id).label("total_trades"),
                        func.sum(func.cast(TradeRecord.pnl > 0, Integer)).label("wins"),
                        func.sum(TradeRecord.pnl).label("total_pnl"),
                        func.avg(TradeRecord.pnl).label("avg_pnl"),
                        func.max(TradeRecord.closed_at).label("last_trade"),
                    )
                    .where(TradeRecord.closed_at >= cutoff)
                    .group_by(TradeRecord.bot_id)
                )

            result = await session.execute(stmt)
            rows = result.all()

            summaries = []
            for row in rows:
                total = row.total_trades or 0
                wins = row.wins or 0
                win_rate = wins / total if total > 0 else 0.0

                summaries.append(TradeResultSummary(
                    bot_id=row.bot_id,
                    symbol=row.symbol,
                    timeframe=row.timeframe,
                    total_trades=total,
                    win_rate=round(win_rate, 4),
                    total_pnl=round(float(row.total_pnl or 0), 2),
                    avg_pnl_per_trade=round(float(row.avg_pnl or 0), 2),
                    max_drawdown=0.0,  # Would need separate calculation
                    ev_per_trade=round(float(row.avg_pnl or 0), 2),
                ))

        return SessionSummaryResponse(
            session_date=session_date,
            summaries=summaries,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch trade results: {e}")

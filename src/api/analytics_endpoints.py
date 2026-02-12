"""
Analytics API Endpoints

Exposes DuckDB analytics for the frontend.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from src.api.analytics_db import query_backtests, query_trades, run_custom_query, init_db

router = APIRouter(prefix="/api/analytics", tags=["analytics"])

@router.on_event("startup")
async def startup_event():
    init_db()

@router.get("/backtests")
async def get_backtests_route(limit: int = 50):
    """Get list of recent backtest runs."""
    return query_backtests(limit)

@router.get("/trades")
async def get_trades_route(run_id: str):
    """Get trades for a specific backtest run."""
    return query_trades(run_id)

class QueryRequest(BaseModel):
    sql: str

@router.post("/query")
async def execute_query(request: QueryRequest):
    """Execute custom SQL query."""
    result = run_custom_query(request.sql)
    if isinstance(result, dict) and "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

"""
QuantMind IDE Strategy Endpoints

API endpoints for strategy folder management.
"""

from fastapi import APIRouter, HTTPException
from typing import Optional

from src.api.ide_handlers import StrategyAPIHandler

router = APIRouter(prefix="/api/strategies", tags=["strategies"])

# Initialize handler
strategy_handler = StrategyAPIHandler()


@router.get("")
async def list_strategies():
    """List all strategy folders."""
    return strategy_handler.list_strategies()


@router.get("/{strategy_id}")
async def get_strategy(strategy_id: str):
    """Get strategy folder details."""
    result = strategy_handler.get_strategy(strategy_id)
    if not result:
        raise HTTPException(404, "Strategy not found")
    return result


@router.post("")
async def create_strategy(name: str):
    """Create a new strategy folder."""
    folder_name = strategy_handler.create_strategy_folder(name)
    return {"id": folder_name, "name": name}

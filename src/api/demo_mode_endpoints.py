"""
Demo Mode API Endpoints

Provides REST API endpoints for EA management, virtual accounts,
and mode-specific operations.
"""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import List, Optional
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["demo-mode"])


# Pydantic models for request/response
class EAConfigRequest(BaseModel):
    """Request model for EA registration."""
    ea_id: str
    name: str
    symbol: str
    timeframe: str
    magic_number: int
    mode: str = "demo"
    virtual_balance: float = 1000.0
    preferred_regime: Optional[str] = None
    preferred_volatility: Optional[str] = None
    max_lot_size: float = 1.0
    max_daily_loss_pct: float = 5.0


class EAConfigResponse(BaseModel):
    """Response model for EA configuration."""
    ea_id: str
    name: str
    symbol: str
    timeframe: str
    magic_number: int
    mode: str
    virtual_balance: float
    preferred_regime: Optional[str] = None
    preferred_volatility: Optional[str] = None
    max_lot_size: float
    max_daily_loss_pct: float
    tags: List[str]


class VirtualAccountResponse(BaseModel):
    """Response model for virtual account."""
    ea_id: str
    initial_balance: float
    current_balance: float
    equity: float
    margin_used: float
    free_margin: float
    pnl: float
    pnl_pct: float
    last_updated: Optional[str]


class EAPromotionRequest(BaseModel):
    """Request model for EA promotion."""
    confirm: bool = False


class TradeStatsResponse(BaseModel):
    """Response model for trade statistics."""
    mode: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    average_pnl: float


# EA Management Endpoints
@router.get("/eas", response_model=List[EAConfigResponse])
async def list_eas(mode: Optional[str] = Query(None, description="Filter by mode (demo/live)")):
    """
    List all registered EAs.
    
    Args:
        mode: Optional filter by trading mode
        
    Returns:
        List of EA configurations
    """
    from src.router.ea_registry import get_ea_registry
    
    registry = get_ea_registry()
    
    if mode:
        from src.router.ea_registry import EAMode
        ea_mode = EAMode.DEMO if mode == "demo" else EAMode.LIVE
        eas = registry.get_by_mode(ea_mode)
    else:
        eas = registry.list_all()
    
    return [EAConfigResponse(**ea.to_dict()) for ea in eas]


@router.get("/eas/modes")
async def get_eas_by_mode():
    """
    Get all EAs grouped by mode.
    
    Returns:
        Dictionary with demo and live EA lists
    """
    from src.router.ea_registry import get_ea_registry
    
    registry = get_ea_registry()
    
    return {
        "demo": [ea.to_dict() for ea in registry.get_demo_eas()],
        "live": [ea.to_dict() for ea in registry.get_live_eas()],
        "counts": registry.count()
    }


@router.post("/eas", response_model=EAConfigResponse)
async def register_ea(config: EAConfigRequest):
    """
    Register a new EA.
    
    Args:
        config: EA configuration
        
    Returns:
        Registered EA configuration
    """
    from src.router.ea_registry import get_ea_registry, EAConfig, EAMode
    from src.router.virtual_balance import get_virtual_balance_manager
    
    registry = get_ea_registry()
    
    # Create EA config
    mode = EAMode.DEMO if config.mode == "demo" else EAMode.LIVE
    ea_config = EAConfig(
        ea_id=config.ea_id,
        name=config.name,
        symbol=config.symbol,
        timeframe=config.timeframe,
        magic_number=config.magic_number,
        mode=mode,
        virtual_balance=config.virtual_balance,
        preferred_regime=config.preferred_regime,
        preferred_volatility=config.preferred_volatility,
        max_lot_size=config.max_lot_size,
        max_daily_loss_pct=config.max_daily_loss_pct
    )
    
    # Register EA
    registry.register(ea_config)
    
    # Create virtual account for demo EAs
    if mode == EAMode.DEMO:
        vb_manager = get_virtual_balance_manager()
        vb_manager.create_account(config.ea_id, config.virtual_balance)
        logger.info(f"Created virtual account for demo EA: {config.ea_id}")
    
    logger.info(f"Registered EA: {config.ea_id} (mode={config.mode})")
    
    return EAConfigResponse(**ea_config.to_dict())


@router.get("/eas/{ea_id}", response_model=EAConfigResponse)
async def get_ea(ea_id: str):
    """
    Get EA configuration by ID.
    
    Args:
        ea_id: EA identifier
        
    Returns:
        EA configuration
    """
    from src.router.ea_registry import get_ea_registry
    
    registry = get_ea_registry()
    ea = registry.get(ea_id)
    
    if ea is None:
        raise HTTPException(status_code=404, detail=f"EA not found: {ea_id}")
    
    return EAConfigResponse(**ea.to_dict())


@router.post("/eas/{ea_id}/promote")
async def promote_ea(ea_id: str, request: EAPromotionRequest):
    """
    Promote an EA from demo to live mode.
    
    Args:
        ea_id: EA identifier
        request: Promotion confirmation
        
    Returns:
        Updated EA configuration
    """
    from src.router.ea_registry import get_ea_registry
    
    if not request.confirm:
        raise HTTPException(status_code=400, detail="Promotion must be confirmed")
    
    registry = get_ea_registry()
    result = registry.promote_to_live(ea_id)
    
    if not result:
        raise HTTPException(status_code=404, detail=f"EA not found: {ea_id}")
    
    ea = registry.get(ea_id)
    logger.info(f"Promoted EA {ea_id} to live mode")
    
    return {"status": "promoted", "ea": ea.to_dict()}


@router.post("/eas/{ea_id}/demote")
async def demote_ea(ea_id: str, virtual_balance: float = Query(1000.0, description="Initial virtual balance")):
    """
    Demote an EA from live to demo mode.
    
    Args:
        ea_id: EA identifier
        virtual_balance: Initial virtual balance for demo mode
        
    Returns:
        Updated EA configuration
    """
    from src.router.ea_registry import get_ea_registry
    from src.router.virtual_balance import get_virtual_balance_manager
    
    registry = get_ea_registry()
    result = registry.demote_to_demo(ea_id, virtual_balance)
    
    if not result:
        raise HTTPException(status_code=404, detail=f"EA not found: {ea_id}")
    
    # Create virtual account
    vb_manager = get_virtual_balance_manager()
    vb_manager.create_account(ea_id, virtual_balance)
    
    ea = registry.get(ea_id)
    logger.info(f"Demoted EA {ea_id} to demo mode with balance {virtual_balance}")
    
    return {"status": "demoted", "ea": ea.to_dict()}


@router.delete("/eas/{ea_id}")
async def unregister_ea(ea_id: str):
    """
    Unregister an EA.
    
    Args:
        ea_id: EA identifier
        
    Returns:
        Confirmation message
    """
    from src.router.ea_registry import get_ea_registry
    from src.router.virtual_balance import get_virtual_balance_manager
    
    registry = get_ea_registry()
    result = registry.unregister(ea_id)
    
    if not result:
        raise HTTPException(status_code=404, detail=f"EA not found: {ea_id}")
    
    # Also delete virtual account if exists
    vb_manager = get_virtual_balance_manager()
    vb_manager.delete_account(ea_id)
    
    logger.info(f"Unregistered EA: {ea_id}")
    
    return {"status": "unregistered", "ea_id": ea_id}


# Virtual Account Endpoints
@router.get("/virtual-accounts", response_model=List[VirtualAccountResponse])
async def list_virtual_accounts():
    """
    List all virtual accounts.
    
    Returns:
        List of virtual account states
    """
    from src.router.virtual_balance import get_virtual_balance_manager
    
    manager = get_virtual_balance_manager()
    summary = manager.get_summary()
    
    return [VirtualAccountResponse(**acc) for acc in summary["accounts"]]


@router.get("/virtual-accounts/{ea_id}", response_model=VirtualAccountResponse)
async def get_virtual_account(ea_id: str):
    """
    Get virtual account for an EA.
    
    Args:
        ea_id: EA identifier
        
    Returns:
        Virtual account state
    """
    from src.router.virtual_balance import get_virtual_balance_manager
    
    manager = get_virtual_balance_manager()
    account = manager.get_account(ea_id)
    
    if account is None:
        raise HTTPException(status_code=404, detail=f"Virtual account not found: {ea_id}")
    
    return VirtualAccountResponse(**account.to_dict())


@router.post("/virtual-accounts/{ea_id}/reset")
async def reset_virtual_account(ea_id: str, new_balance: Optional[float] = Query(None, description="New balance (optional)")):
    """
    Reset virtual account balance.
    
    Args:
        ea_id: EA identifier
        new_balance: Optional new balance (defaults to initial balance)
        
    Returns:
        Updated virtual account state
    """
    from src.router.virtual_balance import get_virtual_balance_manager
    
    manager = get_virtual_balance_manager()
    result = manager.reset_account(ea_id, new_balance)
    
    if not result:
        raise HTTPException(status_code=404, detail=f"Virtual account not found: {ea_id}")
    
    account = manager.get_account(ea_id)
    logger.info(f"Reset virtual account for {ea_id}")
    
    return VirtualAccountResponse(**account.to_dict())


# Trade Statistics Endpoints
@router.get("/trades/stats", response_model=TradeStatsResponse)
async def get_trade_stats(mode: str = Query("all", description="Filter by mode (demo/live/all)")):
    """
    Get trade statistics by mode.
    
    Args:
        mode: Filter by trading mode
        
    Returns:
        Trade statistics
    """
    from src.database.engine import get_session
    from src.database.models import TradeJournal, TradingMode
    from sqlalchemy import func
    
    try:
        session = get_session()
        
        query = session.query(
            func.count(TradeJournal.id).label('total_trades'),
            func.sum(func.case((TradeJournal.pnl > 0, 1), else_=0)).label('winning_trades'),
            func.sum(func.case((TradeJournal.pnl < 0, 1), else_=0)).label('losing_trades'),
            func.avg(TradeJournal.pnl).label('average_pnl'),
            func.sum(TradeJournal.pnl).label('total_pnl')
        )
        
        if mode != "all":
            trading_mode = TradingMode.DEMO if mode == "demo" else TradingMode.LIVE
            query = query.filter(TradeJournal.mode == trading_mode)
        
        result = query.first()
        
        total_trades = result.total_trades or 0
        winning_trades = result.winning_trades or 0
        losing_trades = result.losing_trades or 0
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        session.close()
        
        return TradeStatsResponse(
            mode=mode,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=float(result.total_pnl or 0),
            average_pnl=float(result.average_pnl or 0)
        )
        
    except Exception as e:
        logger.error(f"Error getting trade stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Health check endpoint
@router.get("/demo-mode/health")
async def demo_mode_health():
    """
    Health check for demo mode system.
    
    Returns:
        System status
    """
    from src.router.ea_registry import get_ea_registry
    from src.router.virtual_balance import get_virtual_balance_manager
    
    registry = get_ea_registry()
    vb_manager = get_virtual_balance_manager()
    
    return {
        "status": "healthy",
        "ea_registry": {
            "total": len(registry.list_all()),
            "demo": len(registry.get_demo_eas()),
            "live": len(registry.get_live_eas())
        },
        "virtual_accounts": {
            "total": vb_manager.get_summary()["total_accounts"],
            "total_equity": vb_manager.get_total_equity()
        }
    }


def include_router(app):
    """Include demo mode router in FastAPI app."""
    app.include_router(router)
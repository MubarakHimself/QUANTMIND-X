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

from src.api.pagination import PaginatedResponse, DEFAULT_LIMIT, DEFAULT_OFFSET, MAX_LIMIT

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
@router.get("/eas", response_model=PaginatedResponse[EAConfigResponse])
async def list_eas(
    mode: Optional[str] = Query(None, description="Filter by mode (demo/live)"),
    limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT, description="Maximum items to return"),
    offset: int = Query(DEFAULT_OFFSET, ge=0, description="Number of items to skip")
):
    """
    List all registered EAs with pagination.

    Args:
        mode: Optional filter by trading mode
        limit: Maximum items to return
        offset: Number of items to skip

    Returns:
        Paginated list of EA configurations
    """
    from src.router.ea_registry import get_ea_registry

    registry = get_ea_registry()

    if mode:
        from src.router.ea_registry import EAMode
        ea_mode = EAMode.DEMO if mode == "demo" else EAMode.LIVE
        eas = registry.get_by_mode(ea_mode)
    else:
        eas = registry.list_all()

    items = [EAConfigResponse(**ea.to_dict()) for ea in eas]
    total = len(items)
    paginated_items = items[offset:offset + limit]
    return PaginatedResponse.create(
        items=paginated_items,
        total=total,
        limit=limit,
        offset=offset
    )


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
@router.get("/virtual-accounts", response_model=PaginatedResponse[VirtualAccountResponse])
async def list_virtual_accounts(
    limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT, description="Maximum items to return"),
    offset: int = Query(DEFAULT_OFFSET, ge=0, description="Number of items to skip")
):
    """
    List all virtual accounts with pagination.

    Returns:
        Paginated list of virtual account states
    """
    from src.router.virtual_balance import get_virtual_balance_manager

    manager = get_virtual_balance_manager()
    summary = manager.get_summary()

    items = [VirtualAccountResponse(**acc) for acc in summary["accounts"]]
    total = len(items)
    paginated_items = items[offset:offset + limit]
    return PaginatedResponse.create(
        items=paginated_items,
        total=total,
        limit=limit,
        offset=offset
    )


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


# Pydantic models for video strategies response
class VideoStrategyResponse(BaseModel):
    """Response model for video strategy."""
    strategy_id: str
    name: str
    video_id: str
    trd_file: str
    config_file: str
    created_at: str
    tags: List[str]
    symbols: List[str]
    timeframes: List[str]
    strategy_type: str
    has_ea_code: bool
    has_backtest_reports: bool


# Video Strategies Endpoints
@router.get("/video-strategies", response_model=List[VideoStrategyResponse])
async def list_video_strategies():
    """
    List all processed video strategies from YouTube.

    Returns:
        List of video strategies with their TRD and metadata
    """
    from pathlib import Path
    import json
    import os

    strategies_dir = Path("/home/mubarkahimself/Desktop/QUANTMINDX/src/strategies-yt")
    strategies = []

    if not strategies_dir.exists():
        return strategies

    for strategy_folder in strategies_dir.iterdir():
        if not strategy_folder.is_dir():
            continue

        # Skip non-strategy directories
        if strategy_folder.name.startswith('.') or strategy_folder.name == 'prompts':
            continue

        # Find TRD file
        trd_file = None
        config_file = None

        for file in strategy_folder.iterdir():
            if file.name.endswith('_TRD.md'):
                trd_file = str(file)
            elif file.name.endswith('_config.json'):
                config_file = str(file)

        if not trd_file:
            continue

        # Read config if available
        config_data = {}
        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read config for {strategy_folder.name}: {e}")

        # Check for EA code and backtest reports
        ea_output_path = Path("/home/mubarkahimself/Desktop/QUANTMINDX/output/expert_advisors")
        ea_code_name = config_data.get('ea_id', strategy_folder.name)
        has_ea_code = False

        if ea_output_path.exists():
            for ea_file in ea_output_path.glob("*.mq5"):
                if ea_code_name.lower() in ea_file.stem.lower():
                    has_ea_code = True
                    break

        # Check for backtest reports
        backtest_path = Path("/home/mubarkahimself/Desktop/QUANTMINDX/output/backtests")
        has_backtest_reports = False
        if backtest_path.exists():
            for bt_file in backtest_path.rglob(f"*{ea_code_name}*"):
                if bt_file.suffix in ['.html', '.xml', '.json']:
                    has_backtest_reports = True
                    break

        # Extract video ID from folder name
        video_id = strategy_folder.name.replace('strategy_from_', '')

        strategies.append(VideoStrategyResponse(
            strategy_id=strategy_folder.name,
            name=config_data.get('name', strategy_folder.name.replace('_', ' ').title()),
            video_id=video_id,
            trd_file=trd_file,
            config_file=config_file or "",
            created_at=config_data.get('created_at', ''),
            tags=config_data.get('tags', []),
            symbols=config_data.get('symbols', {}).get('primary', []),
            timeframes=config_data.get('symbols', {}).get('timeframes', []),
            strategy_type=config_data.get('strategy', {}).get('type', 'UNKNOWN'),
            has_ea_code=has_ea_code,
            has_backtest_reports=has_backtest_reports
        ))

    # Sort by creation date (newest first)
    strategies.sort(key=lambda x: x.created_at, reverse=True)

    return strategies


@router.get("/video-strategies/{strategy_id}/trd")
async def get_strategy_trd(strategy_id: str):
    """
    Get TRD content for a video strategy.

    Args:
        strategy_id: Strategy identifier

    Returns:
        TRD content as markdown
    """
    from pathlib import Path

    strategies_dir = Path("/home/mubarkahimself/Desktop/QUANTMINDX/src/strategies-yt")
    strategy_folder = strategies_dir / strategy_id

    if not strategy_folder.exists():
        raise HTTPException(status_code=404, detail=f"Strategy not found: {strategy_id}")

    # Find TRD file
    trd_file = None
    for file in strategy_folder.iterdir():
        if file.name.endswith('_TRD.md'):
            trd_file = file
            break

    if not trd_file or not trd_file.exists():
        raise HTTPException(status_code=404, detail=f"TRD file not found for: {strategy_id}")

    content = trd_file.read_text()

    return {
        "strategy_id": strategy_id,
        "content": content,
        "file_path": str(trd_file)
    }


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
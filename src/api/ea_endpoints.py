"""
EA Lifecycle API Endpoints

FastAPI router for Expert Advisor lifecycle management.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, Dict, Any, List
from pydantic import BaseModel
import logging

from src.agents.tools.ea_lifecycle import EALifecycleTools, EALifecycleManager
from src.api.pagination import PaginatedResponse, DEFAULT_LIMIT, DEFAULT_OFFSET, MAX_LIMIT


class EAListItem(BaseModel):
    """EA list item model."""
    name: str
    file_path: str
    size: int
    modified: float


# EA Request/Response Models (stubs for API compatibility)
class EACreationRequest(BaseModel):
    """EA creation request."""
    name: str
    strategy_code: Optional[str] = None


class EAValidationRequest(BaseModel):
    """EA validation request."""
    ea_path: str


class EABacktestRequest(BaseModel):
    """EA backtest request."""
    ea_path: str
    symbol: str
    timeframe: int = 1


class EAStressTestRequest(BaseModel):
    """EA stress test request."""
    ea_path: str


class EAMonteCarloRequest(BaseModel):
    """EA Monte Carlo request."""
    ea_path: str
    simulations: int = 1000


class EADeployPaperRequest(BaseModel):
    """EA deploy to paper request."""
    ea_path: str
    symbol: str


class EAOptimizationRequest(BaseModel):
    """EA optimization request."""
    ea_path: str


class EACreationResponse(BaseModel):
    """EA creation response."""
    success: bool
    message: str
    ea_path: Optional[str] = None


class EAValidationResponse(BaseModel):
    """EA validation response."""
    success: bool
    message: str


class EABacktestResponse(BaseModel):
    """EA backtest response."""
    success: bool
    message: str
    result: Optional[Dict[str, Any]] = None


class EAStressTestResponse(BaseModel):
    """EA stress test response."""
    success: bool
    message: str


class EAMonteCarloResponse(BaseModel):
    """EA Monte Carlo response."""
    success: bool
    message: str


class EADeployPaperResponse(BaseModel):
    """EA deploy paper response."""
    success: bool
    message: str


class EAMonitorResponse(BaseModel):
    """EA monitor response."""
    status: str


class EAOptimizationResponse(BaseModel):
    """EA optimization response."""
    success: bool
    message: str


class EAStopResponse(BaseModel):
    """EA stop response."""
    success: bool
    message: str


class EAListResponse(BaseModel):
    """EA list response."""
    items: List[EAListItem]

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ea", tags=["ea"])

# Initialize EA tools
ea_tools = EALifecycleTools()
ea_manager = EALifecycleManager()


@router.post("/create")
async def create_ea(
    strategy_name: str,
    ea_name: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create an Expert Advisor from a strategy.

    Args:
        strategy_name: Name of the strategy to convert
        ea_name: Name for the EA (optional)
        parameters: Trading parameters (optional)

    Returns:
        EA creation result with file path
    """
    try:
        result = ea_tools.create_ea(
            strategy_name=strategy_name,
            ea_name=ea_name,
            parameters=parameters
        )

        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))

        return result

    except Exception as e:
        logger.error(f"Error creating EA: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/create-variants")
async def create_ea_variants(
    trd_variants: Dict[str, Any],
    create_variants: str = "both"
) -> Dict[str, Any]:
    """
    Create EA variants (vanilla and/or spiced) from TRD.

    Args:
        trd_variants: Dict containing 'vanilla' and/or 'spiced' strategy configs
        create_variants: Which variants to create - "vanilla", "spiced", or "both"

    Returns:
        Dict with created EA variant(s)
    """
    try:
        result = ea_manager.create_ea_from_trd(
            trd_variants=trd_variants,
            create_variants=create_variants
        )

        if not result:
            raise HTTPException(status_code=400, detail="No variants could be created")

        return {
            "success": True,
            "variants": result,
            "created": list(result.keys())
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating EA variants: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list", response_model=PaginatedResponse[EAListItem])
async def list_eas(
    limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT, description="Maximum items to return"),
    offset: int = Query(DEFAULT_OFFSET, ge=0, description="Number of items to skip")
) -> PaginatedResponse[EAListItem]:
    """
    List all available Expert Advisors with pagination.

    Returns:
        Paginated list of EAs with metadata
    """
    try:
        from pathlib import Path

        ea_output_path = Path("/home/mubarkahimself/Desktop/QUANTMINDX/output/expert_advisors")
        ea_files: List[EAListItem] = []

        if ea_output_path.exists():
            for ea_file in ea_output_path.glob("*.mq5"):
                ea_files.append(EAListItem(
                    name=ea_file.stem,
                    file_path=str(ea_file),
                    size=ea_file.stat().st_size,
                    modified=ea_file.stat().st_mtime
                ))

        total = len(ea_files)
        paginated_items = ea_files[offset:offset + limit]

        return PaginatedResponse.create(
            items=paginated_items,
            total=total,
            limit=limit,
            offset=offset
        )

    except Exception as e:
        logger.error(f"Error listing EAs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/bots")
async def list_ea_bots() -> List[Dict[str, Any]]:
    """
    Compatibility route for the current EA Manager UI.

    Returns real EA file and manifest inventory normalized into the existing UI
    shape without fabricating performance values.
    """
    from src.router.bot_manifest import BotRegistry

    registry = BotRegistry()
    manifest_by_name = {manifest.name: manifest for manifest in registry.list_all()}
    result = await list_eas(limit=MAX_LIMIT, offset=0)
    bots: List[Dict[str, Any]] = []
    for item in result.items:
        manifest = manifest_by_name.get(item.name)
        stats = manifest.get_current_stats() if manifest else None
        tags = list(manifest.tags) if manifest else []

        if manifest:
            if "@quarantine" in tags or "@paper_only" in tags:
                status = "quarantine"
            elif "@under_review" in tags:
                status = "review"
            elif manifest.trading_mode.value == "live":
                status = "live"
            elif manifest.promotion_eligible:
                status = "primal"
            elif "@pending" in tags:
                status = "pending"
            else:
                status = "development"
        else:
            status = "development"

        bots.append({
            "id": item.name,
            "name": item.name,
            "symbol": manifest.symbols[0] if manifest and manifest.symbols else "",
            "status": status,
            "tags": tags,
            "version": str(int(item.modified)),
            "lastModified": item.modified,
            "author": getattr(manifest, "created_by", None) if manifest else None,
            "performance": {
                "winRate": stats.win_rate if stats else None,
                "profitFactor": stats.profit_factor if stats else None,
                "maxDrawdown": stats.max_drawdown if stats else None,
                "sharpeRatio": stats.sharpe_ratio if stats else None,
            },
            "backtestStatus": "passed" if stats and stats.total_trades > 0 else "pending",
        })
    return bots


@router.get("/reviews")
async def get_ea_reviews() -> List[Dict[str, Any]]:
    """Compatibility route for EA review history."""
    return []


@router.post("/validate")
async def validate_ea(ea_name: str) -> Dict[str, Any]:
    """
    Validate an Expert Advisor.

    Args:
        ea_name: Name of the EA to validate

    Returns:
        Validation results
    """
    try:
        result = ea_tools.validate_ea(ea_name=ea_name)

        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))

        return result

    except Exception as e:
        logger.error(f"Error validating EA: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/backtest")
async def backtest_ea(
    ea_name: str,
    symbol: str = "EURUSD",
    timeframe: str = "H1",
    date_from: str = "2023-01-01",
    date_to: str = "2023-12-31",
    deposit: int = 10000
) -> Dict[str, Any]:
    """
    Run backtest for an Expert Advisor.

    Args:
        ea_name: Name of the EA to backtest
        symbol: Trading symbol
        timeframe: Timeframe
        date_from: Start date
        date_to: End date
        deposit: Initial deposit

    Returns:
        Backtest results
    """
    try:
        result = ea_tools.backtest_ea(
            ea_name=ea_name,
            symbol=symbol,
            timeframe=timeframe,
            date_from=date_from,
            date_to=date_to,
            deposit=deposit
        )

        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))

        return result

    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/deploy-paper")
async def deploy_paper(
    ea_name: str,
    account_id: Optional[str] = None,
    strategy_id: Optional[str] = None,
    strategy_code: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    virtual_balance: float = 10000.0,
    magic_number: int = 0
) -> Dict[str, Any]:
    """
    Deploy EA to paper trading via MT5 demo mode.

    Args:
        ea_name: Name of the EA to deploy
        account_id: Paper trading account ID
        strategy_id: Strategy identifier for bot manifest lookup
        strategy_code: Strategy Python code or template reference
        config: Strategy configuration parameters
        virtual_balance: Starting virtual balance (default: $10,000)
        magic_number: Unique magic number for trade identification

    Returns:
        Deployment status with container_id and agent_id
    """
    try:
        result = ea_tools.deploy_paper(
            ea_name=ea_name,
            account_id=account_id,
            strategy_id=strategy_id,
            strategy_code=strategy_code,
            config=config,
            virtual_balance=virtual_balance,
            magic_number=magic_number
        )

        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))

        return result

    except Exception as e:
        logger.error(f"Error deploying to paper: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def collect_paper_results(
    agent_id: str,
    container_id: Optional[str] = None,
    magic_number: Optional[int] = None,
    from_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Collect paper trading results from MT5 and log to TradeLogger.

    This function:
    1. Connects to MT5 demo via SocketAdapter
    2. Retrieves deal history
    3. Logs each trade to TradeLogger
    4. Returns aggregated summary metrics

    Args:
        agent_id: Paper trading agent ID
        container_id: Docker container ID (optional, for logging)
        magic_number: MT5 magic number to filter trades (optional)
        from_date: Start date for deal history (ISO format, optional)

    Returns:
        Summary dict with win rate, P&L, total trades, max drawdown
    """
    import asyncio
    import os

    from src.router.trade_logger import get_trade_logger
    from src.data.brokers.mt5_socket_adapter import MT5SocketAdapter

    # Get MT5 connection config from environment
    mt5_config = {
        "vps_host": os.getenv("MT5_VPS_HOST", "localhost"),
        "vps_port": int(os.getenv("MT5_VPS_PORT", "5555")),
        "account_id": os.getenv("MT5_ACCOUNT_ID", "demo"),
        "timeout": 10.0,
    }

    adapter = MT5SocketAdapter(mt5_config)
    trade_logger = get_trade_logger()

    try:
        # Get deal history from MT5
        deals = await adapter.get_deal_history(
            from_date=from_date,
            magic_number=magic_number,
        )

        if not deals:
            return {
                "agent_id": agent_id,
                "total_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "max_drawdown": 0.0,
                "deals_collected": 0,
                "deals_logged": 0,
            }

        # Log each deal to TradeLogger
        logged_count = 0
        for deal in deals:
            # Determine close reason from deal properties
            close_reason = _determine_close_reason(deal)

            trade_logger.log_paper_trade(
                symbol=deal.get("symbol", "UNKNOWN"),
                direction=deal.get("direction", "BUY").upper(),
                requested_volume=deal.get("volume", 0.0),
                approved_volume=deal.get("volume", 0.0),
                risk_multiplier=1.0,
                entry_price=deal.get("entry_price", 0.0),
                exit_price=deal.get("exit_price", 0.0),
                pnl=deal.get("pnl", 0.0),
                close_reason=close_reason,
                notes=[
                    f"agent_id: {agent_id}",
                    f"deal_id: {deal.get('deal_id', 'N/A')}",
                    f"container_id: {container_id or 'N/A'}",
                ],
            )
            logged_count += 1

        # Get summary from TradeLogger
        summary = trade_logger.get_paper_trade_summary(agent_id=agent_id)

        result = {
            "agent_id": agent_id,
            "total_trades": summary.get("total_trades", 0),
            "winning_trades": summary.get("winning_trades", 0),
            "losing_trades": summary.get("losing_trades", 0),
            "win_rate": summary.get("win_rate", 0.0),
            "total_pnl": summary.get("total_pnl", 0.0),
            "average_pnl": summary.get("average_pnl", 0.0),
            "max_drawdown": summary.get("max_drawdown", 0.0),
            "deals_collected": len(deals),
            "deals_logged": logged_count,
            "close_reasons": summary.get("close_reasons", {}),
        }

        logger.info(
            f"Collected paper results for {agent_id}: "
            f"{result['total_trades']} trades, WR={result['win_rate']:.2%}, "
            f"PnL=${result['total_pnl']:.2f}"
        )

        return result

    except Exception as e:
        logger.error(f"Failed to collect paper results for {agent_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to collect paper results: {str(e)}"
        )
    finally:
        adapter.close()


def _determine_close_reason(deal: Dict[str, Any]) -> str:
    """
    Determine the close reason from a deal dictionary.

    MT5 deal reasons typically include:
    - DEAL_REASON_TP = Take Profit
    - DEAL_REASON_SL = Stop Loss
    - DEAL_REASON手动 = Manual close

    Args:
        deal: Deal dictionary from MT5

    Returns:
        Close reason string
    """
    reason_code = deal.get("reason", deal.get("close_reason", ""))

    if isinstance(reason_code, int):
        # MT5 reason codes
        if reason_code == 1:  # DEAL_REASON_TP
            return "TP_HIT"
        elif reason_code == 2:  # DEAL_REASON_SL
            return "SL_HIT"
        elif reason_code == 3:  # DEAL_REASON手动 (manual)
            return "MANUAL_CLOSE"

    reason_str = str(reason_code).upper()
    if "TP" in reason_str or "TAKE" in reason_str or "PROFIT" in reason_str:
        return "TP_HIT"
    elif "SL" in reason_str or "STOP" in reason_str or "LOSS" in reason_str:
        return "SL_HIT"
    elif "MANUAL" in reason_str or "USER" in reason_str:
        return "MANUAL_CLOSE"
    else:
        return "PAPER_SESSION_END"


@router.get("/paper-results")
async def get_paper_results(
    agent_id: str = Query(..., description="Paper trading agent ID"),
    container_id: Optional[str] = Query(None, description="Docker container ID"),
    magic_number: Optional[int] = Query(None, description="MT5 magic number filter"),
    from_date: Optional[str] = Query(None, description="Start date (ISO format)"),
) -> Dict[str, Any]:
    """
    Get paper trading results and log to TradeLogger.

    Collects closed trades from MT5 demo, logs them to TradeLogger,
    and returns aggregated summary metrics.

    This endpoint is called by the ImprovementLoopFlow during paper gate
    monitoring to collect and evaluate paper trading performance.

    Args:
        agent_id: Paper trading agent ID
        container_id: Docker container ID (optional)
        magic_number: MT5 magic number to filter trades (optional)
        from_date: Start date for deal history in ISO format (optional)

    Returns:
        Summary dict with win rate, P&L, total trades, max drawdown
    """
    return await collect_paper_results(
        agent_id=agent_id,
        container_id=container_id,
        magic_number=magic_number,
        from_date=from_date,
    )


@router.post("/stop")
async def stop_ea(
    ea_name: str,
    environment: str = "paper"
) -> Dict[str, Any]:
    """
    Stop a running EA.

    Args:
        ea_name: Name of the EA to stop
        environment: Environment where EA is running

    Returns:
        Stop status
    """
    try:
        result = ea_tools.stop_ea(
            ea_name=ea_name,
            environment=environment
        )

        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))

        return result

    except Exception as e:
        logger.error(f"Error stopping EA: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ EA Tagging Endpoints ============

class EATagRequest(BaseModel):
    """Request model for adding/removing EA tags."""
    tags: List[str]


class PropFirmSettings(BaseModel):
    """Prop firm settings model."""
    name: str
    account_size: float
    max_drawdown_pct: float
    profit_target_pct: float
    monthly_loss_limit_pct: float
    daily_loss_limit_pct: float


@router.get("/tags")
async def get_ea_tags(
    bot_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get tags for a specific EA or list all available tags.

    Args:
        bot_name: Optional bot name to get tags for

    Returns:
        Dict with tags and available tag options
    """
    try:
        from src.database.manager import DatabaseManager
        from src.database.models.bots import BotManifest

        db = DatabaseManager()

        # Default available tags
        available_tags = [
            {"id": "scalper", "label": "Scalper", "color": "#3b82f6", "description": "Short-term trades"},
            {"id": "swing", "label": "Swing", "color": "#8b5cf6", "description": "Medium-term trades"},
            {"id": "trend", "label": "Trend", "color": "#06b6d4", "description": "Trend following"},
            {"id": "mean-reversion", "label": "Mean Reversion", "color": "#ec4899", "description": "Reversal strategy"},
            {"id": "breakout", "label": "Breakout", "color": "#f59e0b", "description": "Breakout strategy"},
            {"id": "grid", "label": "Grid", "color": "#10b981", "description": "Grid trading"},
            {"id": "martingale", "label": "Martingale", "color": "#ef4444", "description": "Martingale strategy"},
            {"id": "news", "label": "News", "color": "#6366f1", "description": "News trading"},
        ]

        # Prop firm tags from database
        prop_firm_tags = [
            {"id": "@fundednext", "label": "FundedNext", "color": "#22c55e", "description": "FundedNext account"},
            {"id": "@ftmo", "label": "FTMO", "color": "#3b82f6", "description": "FTMO account"},
            {"id": "@apex", "label": "ApexTrader", "color": "#8b5cf6", "description": "ApexTrader account"},
            {"id": "@topstep", "label": "Topstep", "color": "#f59e0b", "description": "Topstep account"},
            {"id": "@challenge_active", "label": "Challenge Active", "color": "#06b6d4", "description": "Active challenge"},
            {"id": "@funded", "label": "Funded", "color": "#10b981", "description": "Funded account"},
        ]

        all_tags = available_tags + prop_firm_tags

        # If bot_name provided, get its tags
        bot_tags = []
        if bot_name:
            with db.get_session() as session:
                bot = session.query(BotManifest).filter(BotManifest.bot_name == bot_name).first()
                if bot:
                    bot_tags = bot.prop_firm_tags

        return {
            "available_tags": all_tags,
            "bot_tags": bot_tags,
            "prop_firm_tags": [t["id"] for t in prop_firm_tags]
        }

    except Exception as e:
        logger.error(f"Error getting EA tags: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tags/{bot_name}")
async def set_ea_tags(
    bot_name: str,
    request: EATagRequest
) -> Dict[str, Any]:
    """
    Set tags for an EA.

    Args:
        bot_name: Bot name to set tags for
        request: List of tags to set

    Returns:
        Success status and updated tags
    """
    session = None
    try:
        from src.database.manager import DatabaseManager
        from src.database.models.bots import BotManifest

        db = DatabaseManager()

        with db.get_session() as session:
            bot = session.query(BotManifest).filter(BotManifest.bot_name == bot_name).first()
            if not bot:
                # Create new bot manifest if doesn't exist
                bot = BotManifest(
                    bot_name=bot_name,
                    bot_type="general",
                    strategy_type="mixed",
                    broker_type="STANDARD"
                )
                session.add(bot)
                session.flush()

            bot.prop_firm_tags = request.tags

        return {
            "success": True,
            "bot_name": bot_name,
            "tags": request.tags
        }

    except Exception as e:
        if session is not None:
            session.rollback()
        logger.error(f"Error setting EA tags: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tags/{bot_name}/add")
async def add_ea_tag(
    bot_name: str,
    tag: str = Query(..., description="Tag to add")
) -> Dict[str, Any]:
    """
    Add a tag to an EA.

    Args:
        bot_name: Bot name
        tag: Tag to add

    Returns:
        Success status and updated tags
    """
    session = None
    try:
        from src.database.manager import DatabaseManager
        from src.database.models.bots import BotManifest

        db = DatabaseManager()

        with db.get_session() as session:
            bot = session.query(BotManifest).filter(BotManifest.bot_name == bot_name).first()
            if not bot:
                raise HTTPException(status_code=404, detail=f"Bot '{bot_name}' not found")

            bot.add_prop_firm_tag(tag)

        return {
            "success": True,
            "bot_name": bot_name,
            "tag": tag,
            "tags": bot.prop_firm_tags
        }

    except HTTPException:
        raise
    except Exception as e:
        if session is not None:
            session.rollback()
        logger.error(f"Error adding EA tag: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tags/{bot_name}/remove")
async def remove_ea_tag(
    bot_name: str,
    tag: str = Query(..., description="Tag to remove")
) -> Dict[str, Any]:
    """
    Remove a tag from an EA.

    Args:
        bot_name: Bot name
        tag: Tag to remove

    Returns:
        Success status and updated tags
    """
    session = None
    try:
        from src.database.manager import DatabaseManager
        from src.database.models.bots import BotManifest

        db = DatabaseManager()

        with db.get_session() as session:
            bot = session.query(BotManifest).filter(BotManifest.bot_name == bot_name).first()
            if not bot:
                raise HTTPException(status_code=404, detail=f"Bot '{bot_name}' not found")

            bot.remove_prop_firm_tag(tag)

        return {
            "success": True,
            "bot_name": bot_name,
            "tag": tag,
            "tags": bot.prop_firm_tags
        }

    except HTTPException:
        raise
    except Exception as e:
        if session is not None:
            session.rollback()
        logger.error(f"Error removing EA tag: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ Prop Firm Settings Endpoints ============

@router.get("/prop-firm/settings")
async def get_prop_firm_settings() -> Dict[str, Any]:
    """
    Get all prop firm settings.

    Returns:
        List of prop firm configurations
    """
    try:
        from src.database.manager import DatabaseManager
        from src.database.models.bots import BotManifest

        db = DatabaseManager()
        with db.get_session() as session:
            # Get all bots with prop firm tags
            bots = session.query(BotManifest).filter(
                BotManifest.bot_metadata.isnot(None)
            ).all()

        prop_firms = []
        for bot in bots:
            if bot.prop_firm_tags:
                firm_config = bot.firm_config or {}
                prop_firms.append({
                    "bot_name": bot.bot_name,
                    "tags": bot.prop_firm_tags,
                    "config": firm_config
                })

        # Return default prop firms if none configured
        if not prop_firms:
            prop_firms = [
                {
                    "bot_name": "default",
                    "tags": ["@fundednext"],
                    "config": {
                        "name": "FundedNext",
                        "account_size": 100000,
                        "max_drawdown_pct": 4.0,
                        "profit_target_pct": 10.0,
                        "monthly_loss_limit_pct": 5.0,
                        "daily_loss_limit_pct": 3.0
                    }
                },
                {
                    "bot_name": "default",
                    "tags": ["@ftmo"],
                    "config": {
                        "name": "FTMO",
                        "account_size": 50000,
                        "max_drawdown_pct": 10.0,
                        "profit_target_pct": 10.0,
                        "monthly_loss_limit_pct": 5.0,
                        "daily_loss_limit_pct": 4.0
                    }
                }
            ]

        return {"prop_firms": prop_firms}

    except Exception as e:
        logger.error(f"Error getting prop firm settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/prop-firm/settings")
async def save_prop_firm_settings(
    settings: PropFirmSettings,
    bot_name: str = Query(...),
    tag: str = Query(...)
) -> Dict[str, Any]:
    """
    Save prop firm settings for an EA.

    Args:
        settings: Prop firm configuration
        bot_name: Bot name
        tag: Prop firm tag (e.g., @fundednext)

    Returns:
        Success status
    """
    session = None
    try:
        from src.database.manager import DatabaseManager
        from src.database.models.bots import BotManifest

        db = DatabaseManager()

        with db.get_session() as session:
            bot = session.query(BotManifest).filter(BotManifest.bot_name == bot_name).first()
            if not bot:
                bot = BotManifest(
                    bot_name=bot_name,
                    bot_type="general",
                    strategy_type="mixed",
                    broker_type="STANDARD"
                )
                session.add(bot)
                session.flush()

            # Add tag if not exists
            if not bot.has_prop_firm_tag(tag):
                bot.add_prop_firm_tag(tag)

            # Save firm config
            bot.firm_config = settings.dict()

        return {
            "success": True,
            "bot_name": bot_name,
            "tag": tag,
            "config": settings.dict()
        }

    except Exception as e:
        if session is not None:
            session.rollback()
        logger.error(f"Error saving prop firm settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

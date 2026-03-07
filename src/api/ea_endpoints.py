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
    account_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Deploy EA to paper trading.

    Args:
        ea_name: Name of the EA to deploy
        account_id: Paper trading account ID

    Returns:
        Deployment status
    """
    try:
        result = ea_tools.deploy_paper(
            ea_name=ea_name,
            account_id=account_id
        )

        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))

        return result

    except Exception as e:
        logger.error(f"Error deploying to paper: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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

        db = DatabaseManager.get_instance()
        session = db.get_session()

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
    try:
        from src.database.manager import DatabaseManager
        from src.database.models.bots import BotManifest

        db = DatabaseManager.get_instance()
        session = db.get_session()

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
        session.commit()

        return {
            "success": True,
            "bot_name": bot_name,
            "tags": request.tags
        }

    except Exception as e:
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
    try:
        from src.database.manager import DatabaseManager
        from src.database.models.bots import BotManifest

        db = DatabaseManager.get_instance()
        session = db.get_session()

        bot = session.query(BotManifest).filter(BotManifest.bot_name == bot_name).first()
        if not bot:
            raise HTTPException(status_code=404, detail=f"Bot '{bot_name}' not found")

        bot.add_prop_firm_tag(tag)
        session.commit()

        return {
            "success": True,
            "bot_name": bot_name,
            "tag": tag,
            "tags": bot.prop_firm_tags
        }

    except HTTPException:
        raise
    except Exception as e:
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
    try:
        from src.database.manager import DatabaseManager
        from src.database.models.bots import BotManifest

        db = DatabaseManager.get_instance()
        session = db.get_session()

        bot = session.query(BotManifest).filter(BotManifest.bot_name == bot_name).first()
        if not bot:
            raise HTTPException(status_code=404, detail=f"Bot '{bot_name}' not found")

        bot.remove_prop_firm_tag(tag)
        session.commit()

        return {
            "success": True,
            "bot_name": bot_name,
            "tag": tag,
            "tags": bot.prop_firm_tags
        }

    except HTTPException:
        raise
    except Exception as e:
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

        db = DatabaseManager.get_instance()
        session = db.get_session()

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
    try:
        from src.database.manager import DatabaseManager
        from src.database.models.bots import BotManifest

        db = DatabaseManager.get_instance()
        session = db.get_session()

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
        session.commit()

        return {
            "success": True,
            "bot_name": bot_name,
            "tag": tag,
            "config": settings.dict()
        }

    except Exception as e:
        session.rollback()
        logger.error(f"Error saving prop firm settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

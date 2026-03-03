"""
Paper Trading API Routes

FastAPI router combining all paper trading endpoints.
"""

import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/paper-trading", tags=["paper-trading"])


# Add model imports
from .models import (
    PromotionRequest,
    PromotionResult,
    AgentPerformanceResponse,
    TradeRecordRequest,
    AddDemoAccountRequest,
    DemoAccountResponse,
)


# Import and setup route modules
from . import deploy
from . import agents
from . import promotion

# Setup routes from submodules
deploy.setup_deploy_routes(router)
agents.setup_agent_routes(router)
promotion.setup_promotion_routes(router)


# -----------------------------------------------------------------------------
# Demo Account Endpoints
# -----------------------------------------------------------------------------

def get_demo_account_manager():
    """Dependency injection for DemoAccountManager."""
    from src.agents.demo_account_manager import DemoAccountManager
    return DemoAccountManager()


@router.get("/demo-accounts", response_model=List[DemoAccountResponse])
async def list_demo_accounts(
    manager=Depends(get_demo_account_manager),
) -> List[DemoAccountResponse]:
    """
    List all configured demo accounts.
    """
    try:
        accounts = manager.list_demo_accounts()
        return [
            DemoAccountResponse(
                login=acc["login"],
                server=acc["server"],
                broker=acc["broker"],
                nickname=acc.get("nickname", ""),
                account_type=acc.get("account_type", "demo"),
                is_active=acc.get("is_active", True)
            )
            for acc in accounts
        ]
    except Exception as e:
        logger.error(f"Failed to list demo accounts: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list demo accounts: {str(e)}"
        )


@router.post("/demo-accounts", response_model=DemoAccountResponse)
async def add_demo_account_endpoint(
    request: AddDemoAccountRequest,
    manager=Depends(get_demo_account_manager),
) -> DemoAccountResponse:
    """
    Add a new demo account.
    """
    try:
        result = manager.add_demo_account(
            login=request.login,
            password=request.password,
            server=request.server,
            broker=request.broker,
            nickname=request.nickname or f"{request.broker}_demo_{request.login}"
        )

        return DemoAccountResponse(
            login=result["login"],
            server=result["server"],
            broker=result["broker"],
            nickname=result.get("nickname", ""),
            account_type=result.get("account_type", "demo"),
            is_active=result.get("is_active", True)
        )
    except Exception as e:
        logger.error(f"Failed to add demo account: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to add demo account: {str(e)}"
        )


@router.get("/demo-accounts/{login}/verify")
async def verify_demo_account_endpoint(
    login: int,
    manager=Depends(get_demo_account_manager),
) -> dict:
    """
    Verify demo account connection and get account details.
    """
    try:
        result = manager.verify_demo_account(login)
        return result
    except Exception as e:
        logger.error(f"Failed to verify demo account {login}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to verify demo account: {str(e)}"
        )


# -----------------------------------------------------------------------------
# Tick Data Endpoints
# -----------------------------------------------------------------------------

@router.post("/tick-data/subscribe")
async def subscribe_tick_data(
    symbol: str = Query(..., description="Symbol to subscribe to tick data (e.g., EURUSD)"),
):
    """
    Subscribe to live tick data for a symbol.
    """
    try:
        from src.data.brokers.mt5_socket_adapter import MT5SocketAdapter
        from src.api.tick_stream_handler import get_tick_handler

        config = {
            "vps_host": os.getenv("MT5_VPS_HOST", "localhost"),
            "vps_port": int(os.getenv("MT5_VPS_PORT", "5555")),
            "account_id": os.getenv("MT5_ACCOUNT_ID"),
        }
        mt5_adapter = MT5SocketAdapter(config)
        tick_handler = get_tick_handler(mt5_adapter)
        await tick_handler.subscribe(symbol)
        return {"success": True, "message": f"Subscribed to tick data for {symbol}"}
    except Exception as e:
        logger.error(f"Failed to subscribe to tick data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to subscribe: {str(e)}")


@router.post("/tick-data/unsubscribe")
async def unsubscribe_tick_data(
    symbol: str = Query(..., description="Symbol to unsubscribe from tick data"),
):
    """
    Unsubscribe from live tick data for a symbol.
    """
    try:
        from src.data.brokers.mt5_socket_adapter import MT5SocketAdapter
        from src.api.tick_stream_handler import get_tick_handler

        config = {
            "vps_host": os.getenv("MT5_VPS_HOST", "localhost"),
            "vps_port": int(os.getenv("MT5_VPS_PORT", "5555")),
            "account_id": os.getenv("MT5_ACCOUNT_ID"),
        }
        mt5_adapter = MT5SocketAdapter(config)
        tick_handler = get_tick_handler(mt5_adapter)
        await tick_handler.unsubscribe(symbol)
        return {"success": True, "message": f"Unsubscribed from tick data for {symbol}"}
    except Exception as e:
        logger.error(f"Failed to unsubscribe from tick data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to unsubscribe: {str(e)}")


# -----------------------------------------------------------------------------
# Bot Tag Endpoints
# -----------------------------------------------------------------------------

@router.get("/bots/by-tag")
async def get_bots_by_tag(
    deployer=Depends(deploy.get_enhanced_deployer),
) -> dict:
    """
    Get all bots grouped by tag.
    """
    try:
        return deployer.list_bots_by_tag()
    except Exception as e:
        logger.error(f"Failed to get bots by tag: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get bots by tag: {str(e)}"
        )


@router.post("/bots/{bot_id}/promote-tag")
async def promote_bot_tag_endpoint(
    bot_id: str,
    deployer=Depends(deploy.get_enhanced_deployer),
) -> dict:
    """
    Manually promote a bot's tag.
    """
    try:
        result = deployer.promote_bot(bot_id)

        if result.get("success"):
            # Broadcast promotion update
            await broadcast_promotion(
                agent_id=bot_id,
                bot_id=bot_id,
                target_account="",
                performance_summary={}
            )

        return result
    except Exception as e:
        logger.error(f"Failed to promote bot {bot_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to promote bot: {str(e)}"
        )


async def broadcast_promotion(agent_id: str, bot_id: str, target_account: str, performance_summary: dict):
    """Broadcast promotion event."""
    from src.api.websocket_endpoints import broadcast_paper_trading_promotion
    await broadcast_paper_trading_promotion(
        agent_id=agent_id,
        bot_id=bot_id,
        target_account=target_account,
        performance_summary=performance_summary
    )

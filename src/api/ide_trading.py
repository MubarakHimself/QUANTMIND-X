"""
QuantMind IDE Trading Endpoints

API endpoints for trading, broker accounts, and bot control.
"""

import logging
from fastapi import APIRouter, HTTPException
from typing import Optional

from src.api.ide_handlers import BrokerAccountsAPIHandler, LiveTradingAPIHandler
from src.api.ide_models import BotControl, CloneBotRequest

logger = logging.getLogger(__name__)

# Initialize routers
broker_router = APIRouter(prefix="/api/trading/broker-accounts", tags=["broker-accounts"])
bots_router = APIRouter(prefix="/api/trading", tags=["trading"])
bots_control_router = APIRouter(prefix="/api/bots", tags=["bots"])

# Initialize handlers
broker_handler = BrokerAccountsAPIHandler()
trading_handler = LiveTradingAPIHandler()


# =============================================================================
# Broker Account Endpoints
# =============================================================================

@broker_router.get("")
async def get_broker_accounts():
    """List available broker accounts."""
    return broker_handler.list_broker_accounts()


@broker_router.get("/{broker_id}")
async def get_broker_account(broker_id: str):
    """Get broker account details."""
    result = broker_handler.get_broker_account(broker_id)
    if not result:
        raise HTTPException(404, f"Broker account {broker_id} not found")
    return result


# =============================================================================
# Bot Control Endpoints
# =============================================================================

@bots_router.get("/bots")
async def get_bots():
    """Get active bots."""
    return trading_handler.get_active_bots()


@bots_router.post("/bots/control")
async def control_bot(control: BotControl):
    """Control a bot."""
    return trading_handler.control_bot(control.bot_id, control.action)


@bots_router.get("/status")
async def get_trading_status():
    """Get system trading status."""
    return trading_handler.get_system_status()


@bots_router.post("/kill")
async def kill_all():
    """Emergency kill all bots."""
    try:
        from src.router.kill_switch import KillSwitch, KillReason
        ks = KillSwitch()
        await ks.trigger(
            reason=KillReason.API_COMMAND,
            triggered_by="api_user",
            message="Emergency kill triggered via API"
        )
        return {"success": True, "message": "Kill switch triggered"}
    except Exception as e:
        raise HTTPException(500, str(e))


# =============================================================================
# Bot Cloning Endpoints
# =============================================================================

@bots_control_router.post("/clone")
async def clone_bot(request: CloneBotRequest):
    """Clone a bot with new configuration."""
    logger.info(f"Cloning bot {request.source_bot_id} to {request.new_name}")
    return {
        "success": True,
        "source_bot_id": request.source_bot_id,
        "new_bot_id": f"{request.source_bot_id}_clone",
        "new_name": request.new_name,
    }


@bots_control_router.get("/clone/eligibility/{bot_id}")
async def check_clone_eligibility(bot_id: str):
    """Check if a bot can be cloned."""
    # Mock eligibility check
    return {
        "bot_id": bot_id,
        "eligible": True,
        "reasons": [],
    }

"""
Trading API Routes

FastAPI router combining all trading endpoints.
"""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter

from .models import (
    BacktestRunRequest,
    BacktestRunResponse,
    BacktestResultResponse,
    DataUploadRequest,
    DataUploadResponse,
    DataStatusResponse,
    DataRefreshRequest,
    EmergencyStopRequest,
    EmergencyStopResponse,
    TradingStatusResponse,
    BotStatusResponse,
    BotParamsResponse,
    BrokerConnectRequest,
    BrokerConnectResponse,
    BrokerCreateRequest,
    BrokerResponse,
    BrokersListResponse,
    ClosePositionRequest,
    ClosePositionResponse,
    CloseAllRequest,
    CloseAllResponse,
)
from .backtest import BacktestAPIHandler
from .data import DataManagementAPIHandler
from .control import TradingControlAPIHandler
from .broker import BrokerRegistryAPIHandler, BrokerConnectionHandler

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1", tags=["trading"])

# Initialize handlers
backtest_handler = BacktestAPIHandler()
data_handler = DataManagementAPIHandler()
trading_handler = TradingControlAPIHandler()
broker_handler = BrokerRegistryAPIHandler()
broker_connection_handler = BrokerConnectionHandler()


# -----------------------------------------------------------------------------
# Backtest Endpoints
# -----------------------------------------------------------------------------

@router.post("/backtest/run", response_model=BacktestRunResponse)
async def run_backtest(request: BacktestRunRequest):
    """Run a backtest with specified parameters."""
    return backtest_handler.run_backtest(request)


@router.get("/backtest/results/{backtest_id}", response_model=BacktestResultResponse)
async def get_backtest_results(backtest_id: str):
    """Retrieve backtest results by ID."""
    return backtest_handler.get_backtest_results(backtest_id)


@router.get("/backtest/status/{backtest_id}")
async def get_backtest_status(backtest_id: str):
    """Get the status of a running or completed backtest."""
    return backtest_handler.get_backtest_status(backtest_id)


# -----------------------------------------------------------------------------
# Data Management Endpoints
# -----------------------------------------------------------------------------

@router.post("/data/upload", response_model=DataUploadResponse)
async def upload_data(request: DataUploadRequest):
    """Upload and cache historical trading data."""
    return data_handler.upload_data(request)


@router.get("/data/status", response_model=DataStatusResponse)
async def get_data_status():
    """Get current data cache status."""
    return data_handler.get_data_status()


@router.post("/data/refresh")
async def refresh_data(request: DataRefreshRequest):
    """Trigger data refresh for specified symbols."""
    return data_handler.refresh_data(request)


# -----------------------------------------------------------------------------
# Trading Control Endpoints
# -----------------------------------------------------------------------------

@router.post("/trading/emergency_stop", response_model=EmergencyStopResponse)
async def emergency_stop(request: EmergencyStopRequest):
    """Trigger emergency stop (kill switch)."""
    return trading_handler.emergency_stop(request)


@router.get("/trading/status", response_model=TradingStatusResponse)
async def get_trading_status():
    """Get current trading status with regime information."""
    return trading_handler.get_trading_status()


@router.get("/trading/bots", response_model=BotStatusResponse)
async def get_bot_status():
    """Get status of all registered bots."""
    return trading_handler.get_bot_status()


@router.get("/trading/bots/{bot_id}/params", response_model=BotParamsResponse)
async def get_bot_params(bot_id: str):
    """Get trading parameters for a specific bot.

    Returns session mask, Islamic compliance status, daily loss cap,
    and force-close countdown when within the 60-minute window.
    """
    return trading_handler.get_bot_params(bot_id)


# -----------------------------------------------------------------------------
# Loss Cap Breach Endpoints (Story 3-3)
# -----------------------------------------------------------------------------

@router.get("/trading/loss-cap/breaches")
async def get_loss_cap_breaches():
    """Get all loss cap breach audit log entries."""
    from src.router.sessions import get_loss_cap_audit_logs
    return {
        "breaches": get_loss_cap_audit_logs(),
        "total_count": len(get_loss_cap_audit_logs())
    }


@router.get("/trading/loss-cap/breaches/{bot_id}")
async def get_bot_loss_cap_breaches(bot_id: str):
    """Get loss cap breach events for a specific bot."""
    from src.router.sessions import get_loss_cap_breach_by_bot
    return {
        "bot_id": bot_id,
        "breaches": get_loss_cap_breach_by_bot(bot_id),
        "total_count": len(get_loss_cap_breach_by_bot(bot_id))
    }


# -----------------------------------------------------------------------------
# Broker Connection Endpoints
# -----------------------------------------------------------------------------

@router.post("/trading/broker/connect", response_model=BrokerConnectResponse)
async def connect_to_broker(request: BrokerConnectRequest):
    """Connect to MT5 broker with actual MT5 integration."""
    return broker_connection_handler.connect_broker(request)


@router.post("/trading/broker/disconnect")
async def disconnect_from_broker():
    """Disconnect from MT5 broker."""
    return broker_connection_handler.disconnect_broker()


# -----------------------------------------------------------------------------
# Broker Registry Endpoints
# -----------------------------------------------------------------------------

@router.get("/brokers", response_model=BrokersListResponse)
async def list_brokers():
    """List all registered broker profiles."""
    return broker_handler.list_brokers()


@router.post("/brokers", response_model=BrokerResponse)
async def create_broker(request: BrokerCreateRequest):
    """Create a new broker profile."""
    return broker_handler.create_broker(request)


# -----------------------------------------------------------------------------
# Position Close Endpoints (Story 3-6)
# -----------------------------------------------------------------------------

@router.post("/trading/close", response_model=ClosePositionResponse)
async def close_position(request: ClosePositionRequest):
    """Close a single position by ticket.

    Args:
        request: Position ticket and bot ID

    Returns:
        Close result with filled price, slippage, and final P&L
    """
    # Extract user from request state if available (set by auth middleware)
    user_context = getattr(request.state, 'user', 'system') if hasattr(request, 'state') else 'system'
    return trading_handler.close_position(request, user_context=user_context)


@router.post("/trading/close-all", response_model=CloseAllResponse)
async def close_all_positions(request: CloseAllRequest):
    """Close all positions for a bot or all bots.

    Args:
        request: Optional bot ID to filter positions

    Returns:
        Results per position (filled/partial/rejected)
    """
    # Extract user from request state if available (set by auth middleware)
    user_context = getattr(request.state, 'user', 'system') if hasattr(request, 'state') else 'system'
    return trading_handler.close_all_positions(request, user_context=user_context)

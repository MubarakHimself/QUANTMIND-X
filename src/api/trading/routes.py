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
    BrokerConnectRequest,
    BrokerConnectResponse,
    BrokerCreateRequest,
    BrokerResponse,
    BrokersListResponse,
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

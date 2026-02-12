"""
API Module

Contains API endpoints and handlers for the QuantMindX backend.
"""

from .heartbeat import HeartbeatHandler, HeartbeatPayload, HeartbeatResponse, create_heartbeat_endpoint

# Import trading system endpoints
try:
    from .trading_endpoints import (
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
        BacktestAPIHandler,
        DataManagementAPIHandler,
        TradingControlAPIHandler,
        create_fastapi_app
    )

    _trading_endpoints_available = True
except ImportError:
    _trading_endpoints_available = False

__all__ = [
    'HeartbeatHandler',
    'HeartbeatPayload',
    'HeartbeatResponse',
    'create_heartbeat_endpoint'
]

# Add trading endpoints if available
if _trading_endpoints_available:
    __all__.extend([
        'BacktestRunRequest',
        'BacktestRunResponse',
        'BacktestResultResponse',
        'DataUploadRequest',
        'DataUploadResponse',
        'DataStatusResponse',
        'DataRefreshRequest',
        'EmergencyStopRequest',
        'EmergencyStopResponse',
        'TradingStatusResponse',
        'BotStatusResponse',
        'BacktestAPIHandler',
        'DataManagementAPIHandler',
        'TradingControlAPIHandler',
        'create_fastapi_app'
    ])

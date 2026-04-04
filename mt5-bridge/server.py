"""
QuantMind MT5 Bridge Server

Provides REST API for MT5 trading operations.
Includes Prometheus metrics for monitoring.
"""

from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
import MetaTrader5 as mt5
import os
import sys
import time
import logging
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mt5-bridge")

# Set up JSON file logging for Promtail to scrape
try:
    # Add project root to path for imports
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from src.monitoring.json_logging import configure_mt5_logging
    configure_mt5_logging()
    logger.info("JSON file logging configured for MT5 Bridge -> /app/logs/mt5-bridge.log")
except Exception as e:
    logger.warning(f"Could not configure JSON file logging: {e}")

# --- CONFIGURATION ---
API_TOKEN = os.getenv("MT5_BRIDGE_TOKEN", "secret-token-change-me")
MT5_METRICS_PORT = int(os.getenv("MT5_PROMETHEUS_PORT", "9091"))
MT5_BRIDGE_PORT = int(os.getenv("MT5_BRIDGE_PORT", "5005"))
MT5_TERMINAL_PATH = os.getenv("MT5_TERMINAL_PATH")
MT5_LOGIN = os.getenv("MT5_LOGIN")
MT5_PASSWORD = os.getenv("MT5_PASSWORD")
MT5_SERVER = os.getenv("MT5_SERVER")
MT5_TIMEOUT_MS = int(os.getenv("MT5_TIMEOUT_MS", "60000"))
MT5_PORTABLE = os.getenv("MT5_PORTABLE", "false").lower() == "true"

app = FastAPI(title="QuantMind MT5 Bridge")

# --- DATA MODELS ---
class TradeRequest(BaseModel):
    symbol: str
    action_type: str  # 'BUY' or 'SELL'
    volume: float
    stop_loss: float = 0.0
    take_profit: float = 0.0

# --- AUTH ---
async def verify_token(
    authorization: Optional[str] = Header(default=None),
    x_token: Optional[str] = Header(default=None, alias="X-Token"),
):
    token = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    elif x_token:
        token = x_token.strip()

    if token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid Auth Token")
    return token


def _build_initialize_kwargs() -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "timeout": MT5_TIMEOUT_MS,
        "portable": MT5_PORTABLE,
    }
    if MT5_LOGIN:
        kwargs["login"] = int(MT5_LOGIN)
    if MT5_PASSWORD:
        kwargs["password"] = MT5_PASSWORD
    if MT5_SERVER:
        kwargs["server"] = MT5_SERVER
    return kwargs


def _initialize_mt5() -> bool:
    kwargs = _build_initialize_kwargs()
    if MT5_TERMINAL_PATH:
        return mt5.initialize(MT5_TERMINAL_PATH, **kwargs)
    return mt5.initialize(**kwargs)


def _namedtuple_to_dict(value):
    if value is None:
        return None
    if hasattr(value, "_asdict"):
        return value._asdict()
    return value


def _trade_retcode_ok(retcode: Optional[int]) -> bool:
    if retcode is None:
        return False
    accepted = {0}
    if hasattr(mt5, "TRADE_RETCODE_DONE"):
        accepted.add(mt5.TRADE_RETCODE_DONE)
    if hasattr(mt5, "TRADE_RETCODE_PLACED"):
        accepted.add(mt5.TRADE_RETCODE_PLACED)
    return retcode in accepted

# --- ENDPOINTS ---

@app.on_event("startup")
def startup_mt5():
    """Attempt to initialize MT5 on startup."""
    # Start Prometheus metrics server
    try:
        from prometheus_client import start_http_server
        start_http_server(MT5_METRICS_PORT)
        logger.info(f"MT5 Bridge metrics server started on port {MT5_METRICS_PORT}")
    except OSError as e:
        if "Address already in use" in str(e):
            logger.warning(f"Metrics server port {MT5_METRICS_PORT} already in use")
        else:
            logger.error(f"Failed to start metrics server: {e}")
    except Exception as e:
        logger.warning(f"Could not start metrics server: {e}")
    
    # Start Grafana Cloud metrics pusher if configured
    _start_grafana_cloud_pusher()
    
    # Official MetaTrader5 Python contract:
    # initialize(path?, login?, password?, server?, timeout?, portable?)
    connected = _initialize_mt5()
    if not connected:
        error_code = mt5.last_error()
        logger.error(
            "MT5 initialization failed: %s (path=%s, login=%s, server=%s)",
            error_code,
            MT5_TERMINAL_PATH,
            MT5_LOGIN,
            MT5_SERVER,
        )
    else:
        logger.info(
            "MT5 initialized successfully (path=%s, login=%s, server=%s)",
            MT5_TERMINAL_PATH,
            MT5_LOGIN,
            MT5_SERVER,
        )
    
    # Update connection status metric
    _update_mt5_status_metric(connected)


def _start_grafana_cloud_pusher():
    """Start Grafana Cloud metrics pusher if configured.
    
    Note: Disabled to avoid duplicate metric ingestion. Using Prometheus agent
    for remote_write instead. See docker-compose.production.yml.
    """
    logger.debug("Grafana Cloud in-process pusher disabled - using Prometheus agent for remote_write")


def _update_mt5_status_metric(connected: bool):
    """Update MT5 connection status gauge."""
    try:
        from src.monitoring import update_mt5_status
        update_mt5_status(connected)
    except ImportError:
        # Monitoring module not available, use direct prometheus
        try:
            from prometheus_client import Gauge
            status_gauge = Gauge('quantmind_mt5_connection_status', 'MT5 connection status')
            status_gauge.set(1 if connected else 0)
        except Exception as e:
            logger.debug(f"Could not update MT5 status metric: {e}")
    except Exception as e:
        logger.debug(f"Could not update MT5 status metric: {e}")


def _track_mt5_latency(operation: str, duration: float, success: bool):
    """Track MT5 operation latency."""
    try:
        from src.monitoring import track_mt5_operation
        track_mt5_operation(operation, duration, success)
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"Could not track MT5 latency: {e}")


def _track_mt5_trade(symbol: str, action: str, success: bool):
    """Track MT5 trade result."""
    try:
        from src.monitoring import mt5_trades_total
        mt5_trades_total.labels(
            symbol=symbol,
            action=action,
            result='success' if success else 'failed'
        ).inc()
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"Could not track MT5 trade: {e}")


@app.get("/status", dependencies=[Depends(verify_token)])
def get_status():
    """Check connection status."""
    start_time = time.time()
    
    info = mt5.terminal_info()
    
    duration = time.time() - start_time
    connected = info is not None
    
    # Update metrics
    _update_mt5_status_metric(connected)
    _track_mt5_latency("status", duration, connected)
    
    if info is None:
        return {
            "status": "disconnected",
            "error": mt5.last_error(),
            "terminal_path": MT5_TERMINAL_PATH,
            "configured_login": MT5_LOGIN,
            "configured_server": MT5_SERVER,
        }
    version = mt5.version()
    return {
        "status": "connected",
        "trade_allowed": info.trade_allowed,
        "connected": info.connected,
        "terminal_path": MT5_TERMINAL_PATH,
        "configured_login": MT5_LOGIN,
        "configured_server": MT5_SERVER,
        "version": list(version) if version else None,
    }


@app.get("/account", dependencies=[Depends(verify_token)])
def get_account():
    """Get account balance and equity."""
    start_time = time.time()
    
    info = mt5.account_info()
    
    duration = time.time() - start_time
    _track_mt5_latency("account_info", duration, info is not None)
    
    if info is None:
        error = mt5.last_error()
        _update_mt5_status_metric(False)
        raise HTTPException(status_code=500, detail=f"Failed to get account info: {error}")
    
    _update_mt5_status_metric(True)
    return info._asdict()


@app.post("/trade", dependencies=[Depends(verify_token)])
def execute_trade(trade: TradeRequest):
    """Execute a trade (Buy/Sell)."""
    start_time = time.time()
    
    # map action string to mt5 constant
    action = mt5.ORDER_TYPE_BUY if trade.action_type.upper() == 'BUY' else mt5.ORDER_TYPE_SELL
    
    symbol_info = mt5.symbol_info(trade.symbol)
    if not symbol_info:
        _track_mt5_trade(trade.symbol, trade.action_type.upper(), False)
        raise HTTPException(status_code=404, detail="Symbol not found")
    
    if not symbol_info.visible:
        if not mt5.symbol_select(trade.symbol, True):
            _track_mt5_trade(trade.symbol, trade.action_type.upper(), False)
            raise HTTPException(status_code=404, detail="Symbol not visible and cannot be selected")

    tick = mt5.symbol_info_tick(trade.symbol)
    if tick is None:
        _track_mt5_trade(trade.symbol, trade.action_type.upper(), False)
        raise HTTPException(status_code=503, detail=f"No live tick available for {trade.symbol}")

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": trade.symbol,
        "volume": trade.volume,
        "type": action,
        "price": tick.ask if action == mt5.ORDER_TYPE_BUY else tick.bid,
        "sl": trade.stop_loss,
        "tp": trade.take_profit,
        "deviation": 20,
        "magic": 234000,
        "comment": "QuantMind-Bridge",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    # Pre-flight the request before execution.
    check_result = mt5.order_check(request)
    if check_result is None:
        _track_mt5_latency("trade", time.time() - start_time, False)
        _track_mt5_trade(trade.symbol, trade.action_type.upper(), False)
        raise HTTPException(status_code=400, detail=f"Order check failed: {mt5.last_error()}")

    check_retcode = getattr(check_result, "retcode", None)
    if not _trade_retcode_ok(check_retcode):
        _track_mt5_latency("trade", time.time() - start_time, False)
        _track_mt5_trade(trade.symbol, trade.action_type.upper(), False)
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Order check rejected the request",
                "retcode": check_retcode,
                "comment": getattr(check_result, "comment", None),
                "check_result": _namedtuple_to_dict(check_result),
            },
        )

    result = mt5.order_send(request)
    duration = time.time() - start_time
    
    success = result is not None and _trade_retcode_ok(getattr(result, "retcode", None))
    _track_mt5_latency("trade", duration, success)
    _track_mt5_trade(trade.symbol, trade.action_type.upper(), success)
    
    if not success:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Order send failed",
                "retcode": getattr(result, "retcode", None),
                "comment": getattr(result, "comment", None),
                "result": _namedtuple_to_dict(result),
            },
        )
    
    # Track trade execution
    try:
        from src.monitoring import track_trade
        track_trade(
            symbol=trade.symbol,
            action=trade.action_type.upper(),
            mode='live'  # MT5 bridge is always live
        )
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"Could not track trade metric: {e}")
    
    return {
        "result": result._asdict(),
        "check_result": _namedtuple_to_dict(check_result),
    }


@app.get("/health")
def health_check():
    """Health check endpoint (no auth required)."""
    return {"status": "healthy", "service": "mt5-bridge"}


if __name__ == "__main__":
    import uvicorn
    # Listen on all interfaces so the VPS is accessible from outside
    uvicorn.run(app, host="0.0.0.0", port=MT5_BRIDGE_PORT)

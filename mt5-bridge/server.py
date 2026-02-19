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
# In production, use environment variables
API_TOKEN = os.getenv("MT5_BRIDGE_TOKEN", "secret-token-change-me")

# Prometheus metrics port
MT5_METRICS_PORT = int(os.getenv("MT5_PROMETHEUS_PORT", "9091"))

app = FastAPI(title="QuantMind MT5 Bridge")

# --- DATA MODELS ---
class TradeRequest(BaseModel):
    symbol: str
    action_type: str  # 'BUY' or 'SELL'
    volume: float
    stop_loss: float = 0.0
    take_profit: float = 0.0

# --- AUTH ---
async def verify_token(x_token: str = Header(...)):
    if x_token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid Auth Token")
    return x_token

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
    
    # Initialize MT5
    connected = mt5.initialize()
    if not connected:
        error_code = mt5.last_error()
        print(f"MT5 Ensure: Initialize failed, error code = {error_code}", file=sys.stderr)
        logger.error(f"MT5 initialization failed: {error_code}")
    else:
        print("MT5 Ensure: Initialized successfully")
        logger.info("MT5 initialized successfully")
    
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
        return {"status": "disconnected", "error": mt5.last_error()}
    return {
        "status": "connected",
        "trade_allowed": info.trade_allowed,
        "connected": info.connected
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

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": trade.symbol,
        "volume": trade.volume,
        "type": action,
        "price": mt5.symbol_info_tick(trade.symbol).ask if action == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(trade.symbol).bid,
        "sl": trade.stop_loss,
        "tp": trade.take_profit,
        "deviation": 20,
        "magic": 234000,
        "comment": "QuantMind-Bridge",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    duration = time.time() - start_time
    
    success = result.retcode == mt5.TRADE_RETCODE_DONE
    _track_mt5_latency("trade", duration, success)
    _track_mt5_trade(trade.symbol, trade.action_type.upper(), success)
    
    if not success:
        raise HTTPException(status_code=400, detail=f"Order failed: {result.comment} ({result.retcode})")
    
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
    
    return result._asdict()


@app.get("/health")
def health_check():
    """Health check endpoint (no auth required)."""
    return {"status": "healthy", "service": "mt5-bridge"}


if __name__ == "__main__":
    import uvicorn
    # Listen on all interfaces so the VPS is accessible from outside
    uvicorn.run(app, host="0.0.0.0", port=5005)
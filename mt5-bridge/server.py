from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
import MetaTrader5 as mt5
import os
import sys

# --- CONFIGURATION ---
# In production, use environment variables
API_TOKEN = os.getenv("MT5_BRIDGE_TOKEN", "secret-token-change-me")

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
    if not mt5.initialize():
        print(f"MT5 Ensure: Initialize failed, error code = {mt5.last_error()}", file=sys.stderr)
    else:
        print("MT5 Ensure: Initialized successfully")

@app.get("/status", dependencies=[Depends(verify_token)])
def get_status():
    """Check connection status."""
    info = mt5.terminal_info()
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
    info = mt5.account_info()
    if info is None:
        raise HTTPException(status_code=500, detail=f"Failed to get account info: {mt5.last_error()}")
    return info._asdict()

@app.post("/trade", dependencies=[Depends(verify_token)])
def execute_trade(trade: TradeRequest):
    """Execute a trade (Buy/Sell)."""
    # map action string to mt5 constant
    action = mt5.ORDER_TYPE_BUY if trade.action_type.upper() == 'BUY' else mt5.ORDER_TYPE_SELL
    
    symbol_info = mt5.symbol_info(trade.symbol)
    if not symbol_info:
        raise HTTPException(status_code=404, detail="Symbol not found")
    
    if not symbol_info.visible:
        if not mt5.symbol_select(trade.symbol, True):
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
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        raise HTTPException(status_code=400, detail=f"Order failed: {result.comment} ({result.retcode})")
    
    return result._asdict()

if __name__ == "__main__":
    import uvicorn
    # Listen on all interfaces so the VPS is accessible from outside
    uvicorn.run(app, host="0.0.0.0", port=5005)

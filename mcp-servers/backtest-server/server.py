import asyncio
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from mcp.server.fastmcp import FastMCP
import datetime

# Add project root to path to import src
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import Core Engine
try:
    from src.backtesting.core_engine import QuantMindBacktester
except ImportError:
    # Fallback/Mock for standalone testing w/o src
    QuantMindBacktester = None

mcp = FastMCP("backtest-server")

def generate_mock_data(days=100):
    """Generate mock OHLCV data for testing."""
    dates = pd.date_range(end=datetime.datetime.now(), periods=days * 24, freq="H")
    data = pd.DataFrame(index=dates)
    data['open'] = np.random.randn(len(dates)).cumsum() + 100
    data['high'] = data['open'] + np.random.rand(len(dates))
    data['low'] = data['open'] - np.random.rand(len(dates))
    data['close'] = data['open'] + np.random.randn(len(dates)) * 0.5
    data['volume'] = np.random.randint(100, 1000, len(dates))
    return data

@mcp.tool()
async def run_backtest(code_content: str, symbol: str = "EURUSD", timeframe: str = "H1") -> str:
    """
    Run a fast backtest on the provided python strategy code.
    
    Args:
        code_content: The python source code inheriting from backtrader.Strategy
        symbol: Symbol to test (e.g. EURUSD)
        timeframe: Timeframe (e.g. H1)
    """
    if not QuantMindBacktester:
         return json.dumps({"error": "QuantMindBacktester module not found."})

    # For MVP, we use mock data. 
    # Real implementation would fetch from `data/` or `AssetHub`.
    data = generate_mock_data()
    
    tester = QuantMindBacktester()
    result = tester.run(code_content, data)
    
    return json.dumps(result, indent=2)

if __name__ == "__main__":
    mcp.run()

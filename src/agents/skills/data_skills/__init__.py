"""
Data Skills Module

Contains skills for data operations including:
- fetch_historical_data: Fetch OHLCV historical data
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any


def fetch_historical_data(
    symbol: str,
    timeframe: str,
    bars_count: int = 100
) -> Dict[str, Any]:
    """
    Fetch historical OHLCV data for a trading symbol.

    Note: This is a mock implementation. In production, this would connect
    to MT5 terminal or a data provider API.

    Args:
        symbol: Trading symbol (e.g., EURUSD, XAUUSD)
        timeframe: Timeframe (M1, M5, M15, M30, H1, H4, D1)
        bars_count: Number of bars to fetch

    Returns:
        Dict with symbol, timeframe, data (OHLCV array), and bars_count
    """
    # Map timeframe to minutes
    timeframe_minutes = {
        "M1": 1, "M5": 5, "M15": 15, "M30": 30,
        "H1": 60, "H4": 240, "D1": 1440
    }

    if timeframe not in timeframe_minutes:
        raise ValueError(f"Invalid timeframe: {timeframe}")

    if bars_count < 10 or bars_count > 10000:
        raise ValueError("bars_count must be between 10 and 10000")

    minutes = timeframe_minutes[timeframe]

    # Generate mock historical data
    # In production, this would use MetaTrader5 Python API
    base_prices = {
        "EURUSD": 1.1000,
        "GBPUSD": 1.3000,
        "USDJPY": 145.00,
        "XAUUSD": 2000.00,
        "XAGUSD": 25.00,
    }
    start_price = base_prices.get(symbol, 1.0)

    # Generate price data using random walk
    np.random.seed(hash(symbol) % (2**32))

    timestamps = []
    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []

    current_time = datetime.now()
    current_price = start_price

    for i in range(bars_count):
        # Calculate timestamp for this bar
        bar_time = current_time - timedelta(minutes=minutes * (bars_count - i))
        timestamps.append(bar_time)

        # Generate price movement
        change = np.random.normal(0, 0.0002)  # Small random changes
        open_price = current_price
        close_price = current_price * (1 + change)
        high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.0001)))
        low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.0001)))
        volume = np.random.randint(1000, 10000)

        opens.append(round(open_price, 5))
        highs.append(round(high_price, 5))
        lows.append(round(low_price, 5))
        closes.append(round(close_price, 5))
        volumes.append(volume)

        current_price = close_price

    # Format data as array of objects
    data = [
        {
            "time": timestamps[i].isoformat(),
            "open": opens[i],
            "high": highs[i],
            "low": lows[i],
            "close": closes[i],
            "volume": volumes[i]
        }
        for i in range(bars_count)
    ]

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "data": data,
        "bars_count": bars_count
    }


__all__ = ["fetch_historical_data"]

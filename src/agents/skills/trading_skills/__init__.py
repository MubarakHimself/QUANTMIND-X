"""
from .calculate_pivot_points_for_trading import skill_definition
from .indicator_writer import skill_definition as indicator_writer_definition

Trading Skills Module

Contains skills for trading operations including:
- calculate_rsi: Calculate Relative Strength Index
- calculate_position_size: Calculate position size based on risk
- detect_support_resistance: Identify support and resistance levels
- indicator_writer: Generate MQL5 indicator code with CRiBuffDbl ring buffer support
"""

import numpy as np
from typing import List, Dict, Any


def calculate_rsi(prices: List[float], period: int = 14) -> Dict[str, Any]:
    """
    Calculate Relative Strength Index (RSI).

    Args:
        prices: List of closing prices
        period: RSI calculation period (default: 14)

    Returns:
        Dict with rsi_value and signal

    RSI Formula:
    RSI = 100 - (100 / (1 + RS))

    Where:
    RS = Average Gain / Average Loss

    Signal interpretation:
    - RSI < 30: oversold (potential buy)
    - RSI > 70: overbought (potential sell)
    - Otherwise: neutral
    """
    if len(prices) < period + 1:
        raise ValueError(f"Need at least {period + 1} prices for RSI calculation")

    if period < 1:
        raise ValueError("Period must be at least 1")

    # Convert to numpy array for efficient calculation
    prices_array = np.array(prices, dtype=np.float64)

    # Calculate price changes
    deltas = np.diff(prices_array)

    # Separate gains and losses
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # Calculate initial average gain and loss
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    # Calculate RSI using Wilder's smoothing method
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    # Calculate RS and RSI
    if avg_loss == 0:
        rsi = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))

    # Generate signal
    if rsi < 30:
        signal = "oversold"
    elif rsi > 70:
        signal = "overbought"
    else:
        signal = "neutral"

    return {
        "rsi_value": round(float(rsi), 2),
        "signal": signal
    }


def calculate_position_size(
    account_balance: float,
    risk_percent: float,
    stop_loss_pips: float,
    pip_value: float = 10.0
) -> Dict[str, Any]:
    """
    Calculate position size based on risk management rules.

    Args:
        account_balance: Current account balance
        risk_percent: Risk percentage per trade (e.g., 1.0 for 1%)
        stop_loss_pips: Stop loss distance in pips
        pip_value: Value per pip in base currency (default: 10 for standard lot)

    Returns:
        Dict with position_size_lots, risk_amount, and max_loss_pips

    Formula:
    Risk Amount = Account Balance * (Risk Percent / 100)
    Position Size = Risk Amount / (Stop Loss Pips * Pip Value)
    """
    if account_balance <= 0:
        raise ValueError("Account balance must be positive")

    if risk_percent <= 0:
        raise ValueError("Risk percent must be positive")

    if stop_loss_pips <= 0:
        raise ValueError("Stop loss pips must be positive")

    # Calculate risk amount
    risk_amount = account_balance * (risk_percent / 100.0)

    # Calculate position size in lots
    position_size_lots = risk_amount / (stop_loss_pips * pip_value)

    # Round to 2 decimal places (standard lot precision)
    position_size_lots = round(position_size_lots, 2)

    # Ensure minimum position size
    if position_size_lots < 0.01:
        position_size_lots = 0.01

    # Maximum loss if stop is hit
    max_loss_pips = stop_loss_pips * position_size_lots * pip_value

    return {
        "position_size_lots": position_size_lots,
        "risk_amount": round(risk_amount, 2),
        "max_loss_pips": round(max_loss_pips, 2)
    }


def detect_support_resistance(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    lookback_period: int = 5,
    num_levels: int = 3
) -> Dict[str, Any]:
    """
    Detect support and resistance levels using pivot point analysis.

    Args:
        highs: Array of high prices
        lows: Array of low prices
        closes: Array of closing prices
        lookback_period: Period for pivot calculation
        num_levels: Number of S/R levels to identify

    Returns:
        Dict with support_levels, resistance_levels, and nearest levels

    Method:
    - Identifies pivot highs (resistance) where price is highest in lookback period
    - Identifies pivot lows (support) where price is lowest in lookback period
    - Clusters nearby pivots to find key levels
    """
    if len(highs) != len(lows) or len(highs) != len(closes):
        raise ValueError("Highs, lows, and closes must have the same length")

    if len(highs) < lookback_period * 2:
        raise ValueError(f"Need at least {lookback_period * 2} data points")

    highs_array = np.array(highs)
    lows_array = np.array(lows)
    closes_array = np.array(closes)

    # Find pivot highs (local maxima)
    pivot_highs = []
    for i in range(lookback_period, len(highs) - lookback_period):
        window_highs = highs_array[i - lookback_period:i + lookback_period + 1]
        if highs_array[i] == np.max(window_highs):
            pivot_highs.append(highs_array[i])

    # Find pivot lows (local minima)
    pivot_lows = []
    for i in range(lookback_period, len(lows) - lookback_period):
        window_lows = lows_array[i - lookback_period:i + lookback_period + 1]
        if lows_array[i] == np.min(window_lows):
            pivot_lows.append(lows_array[i])

    # Cluster pivots to find key levels
    def cluster_levels(levels: List[float], num_clusters: int) -> List[float]:
        """Cluster nearby levels and return cluster centers."""
        if not levels:
            return []

        levels_array = np.array(sorted(levels, reverse=True))

        if len(levels_array) <= num_clusters:
            return levels_array.tolist()

        # Simple clustering: group levels within 0.5% of each other
        clusters = []
        for level in levels_array:
            added = False
            for i, cluster_center in enumerate(clusters):
                if abs(level - cluster_center) / cluster_center < 0.005:  # 0.5% threshold
                    clusters[i] = (clusters[i] + level) / 2  # Update cluster center
                    added = True
                    break
            if not added:
                clusters.append(level)

        return sorted(clusters, reverse=True)[:num_clusters]

    # Get clustered levels
    resistance_levels = cluster_levels(pivot_highs, num_levels)
    support_levels = cluster_levels(pivot_lows, num_levels)
    support_levels = sorted(support_levels)  # Ascending for support

    # Current price
    current_price = closes_array[-1]

    # Find nearest support (largest support below current price)
    nearest_support = None
    for level in reversed(support_levels):
        if level < current_price:
            nearest_support = level
            break

    # Find nearest resistance (smallest resistance above current price)
    nearest_resistance = None
    for level in resistance_levels:
        if level > current_price:
            nearest_resistance = level
            break

    return {
        "support_levels": support_levels,
        "resistance_levels": resistance_levels,
        "current_price": round(float(current_price), 5),
        "nearest_support": round(nearest_support, 5) if nearest_support else None,
        "nearest_resistance": round(nearest_resistance, 5) if nearest_resistance else None
    }


__all__ = [
    "calculate_rsi",
    "calculate_position_size",
    "detect_support_resistance",
    "calculate_pivot_points_for_trading",
    "indicator_writer_definition",
]


"""
Built-in Skills for QuantMind Agent SDK

Provides reusable skills for common agent behaviors:
- Research: Knowledge base queries
- Trading: Position sizing, indicators
- Risk: Risk calculations and validation
- Coding: Code analysis and documentation
- Data: Data processing and analysis
- System: System operations and monitoring
"""

from typing import Any, Dict, List, Optional
import logging
import time
import asyncio
from functools import lru_cache

from .skill_manager import SkillManager, get_skill_manager
from .core_skills import (
    financial_data_fetch, pattern_scanner, statistical_edge,
    hypothesis_document_writer, mql5_generator, backtest_launcher,
    news_classifier, risk_evaluator, report_writer, strategy_optimizer,
    institutional_data_fetch, calendar_gate_check
)

logger = logging.getLogger(__name__)


# ============================================================================
# Research Skills
# ============================================================================

def knowledge_search(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Search the knowledge base for relevant information."""
    return {"query": query, "results": [], "count": 0, "message": "Knowledge base integration pending"}


def extract_trading_rules(document: str) -> Dict[str, Any]:
    """Extract actionable trading rules from a document."""
    return {
        "document_length": len(document),
        "rules_extracted": [],
        "entry_conditions": [],
        "exit_conditions": [],
        "risk_parameters": {},
    }


def research_summary(topic: str, depth: str = "brief") -> Dict[str, Any]:
    """Generate a research summary on a topic."""
    return {"topic": topic, "depth": depth, "summary": f"Research summary for: {topic}", "sources": [], "key_findings": []}


# ============================================================================
# Trading Skills
# ============================================================================

def calculate_position_size(account_balance: float, risk_percent: float, stop_loss_pips: float, pip_value: float = 10.0) -> Dict[str, Any]:
    """Calculate position size based on risk parameters."""
    if stop_loss_pips <= 0:
        raise ValueError("stop_loss_pips must be positive")
    risk_amount = account_balance * (risk_percent / 100)
    position_size = max(risk_amount / (stop_loss_pips * pip_value), 0.01)
    return {"position_size_lots": round(position_size, 2), "risk_amount": round(risk_amount, 2), "max_loss_pips": stop_loss_pips, "risk_percent": risk_percent}


def calculate_rsi(prices: List[float], period: int = 14) -> Dict[str, Any]:
    """Calculate Relative Strength Index."""
    if len(prices) < period + 1:
        raise ValueError(f"Need at least {period + 1} price values")
    changes = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    avg_gain = sum(max(c, 0) for c in changes[-period:]) / period
    avg_loss = sum(max(-c, 0) for c in changes[-period:]) / period
    rsi = 100 - (100 / (1 + avg_gain / avg_loss)) if avg_loss > 0 else 100
    signal = "oversold" if rsi < 30 else "overbought" if rsi > 70 else "neutral"
    return {"rsi_value": round(rsi, 2), "signal": signal, "period": period}


def detect_support_resistance(highs: List[float], lows: List[float], closes: List[float], lookback_period: int = 5) -> Dict[str, Any]:
    """Detect support and resistance levels."""
    if not (len(highs) == len(lows) == len(closes)):
        raise ValueError("highs, lows, and closes must have the same length")
    support_levels, resistance_levels = [], []
    for i in range(lookback_period, len(highs) - lookback_period):
        is_resistance = all(highs[i] > highs[i - j] for j in range(1, lookback_period + 1)) and all(highs[i] > highs[i + j] for j in range(1, lookback_period + 1))
        is_support = all(lows[i] < lows[i - j] for j in range(1, lookback_period + 1)) and all(lows[i] < lows[i + j] for j in range(1, lookback_period + 1))
        if is_resistance:
            resistance_levels.append(round(highs[i], 5))
        if is_support:
            support_levels.append(round(lows[i], 5))
    current_price = closes[-1] if closes else 0
    nearest_support = max([s for s in support_levels if s < current_price]) if support_levels else None
    nearest_resistance = min([r for r in resistance_levels if r > current_price]) if resistance_levels else None
    return {"support_levels": support_levels, "resistance_levels": resistance_levels, "current_price": current_price, "nearest_support": nearest_support, "nearest_resistance": nearest_resistance}


def calculate_pivot_points(high: float, low: float, close: float) -> Dict[str, Any]:
    """Calculate pivot points and S/R levels."""
    pivot = (high + low + close) / 3
    r1, r2, r3 = 2 * pivot - low, pivot + (high - low), high + 2 * (pivot - low)
    s1, s2, s3 = 2 * pivot - high, pivot - (high - low), low - 2 * (high - pivot)
    return {"pivot": round(pivot, 5), "resistance_1": round(r1, 5), "resistance_2": round(r2, 5), "resistance_3": round(r3, 5), "support_1": round(s1, 5), "support_2": round(s2, 5), "support_3": round(s3, 5)}


def calculate_macd(prices: List[float], fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Dict[str, Any]:
    """
    Calculate MACD (Moving Average Convergence Divergence).

    Args:
        prices: List of closing prices
        fast_period: Fast EMA period (default: 12)
        slow_period: Slow EMA period (default: 26)
        signal_period: Signal line period (default: 9)

    Returns:
        Dict with macd_line, signal_line, histogram, and crossover signal
    """
    if len(prices) < slow_period + signal_period:
        raise ValueError(f"Need at least {slow_period + signal_period} prices for MACD calculation")

    def calculate_ema(data: List[float], period: int) -> float:
        """Calculate Exponential Moving Average."""
        multiplier = 2 / (period + 1)
        ema = data[0]
        for price in data[1:]:
            ema = (price - ema) * multiplier + ema
        return ema

    # Calculate EMAs
    fast_ema = calculate_ema(prices, fast_period)
    slow_ema = calculate_ema(prices, slow_period)

    # Calculate MACD line
    macd_line = fast_ema - slow_ema

    # Calculate signal line (simplified - using last N values)
    macd_values = [macd_line] * signal_period
    signal_line = calculate_ema(macd_values, signal_period)

    # Calculate histogram
    histogram = macd_line - signal_line

    # Determine signal
    if histogram > 0:
        signal = "bullish" if histogram > histogram * 0.1 else "neutral"
    elif histogram < 0:
        signal = "bearish" if histogram < histogram * 0.1 else "neutral"
    else:
        signal = "neutral"

    return {
        "macd_line": round(macd_line, 5),
        "signal_line": round(signal_line, 5),
        "histogram": round(histogram, 5),
        "signal": signal,
        "fast_period": fast_period,
        "slow_period": slow_period
    }


def calculate_bollinger_bands(prices: List[float], period: int = 20, num_std: float = 2.0) -> Dict[str, Any]:
    """
    Calculate Bollinger Bands.

    Args:
        prices: List of closing prices
        period: Moving average period (default: 20)
        num_std: Number of standard deviations (default: 2.0)

    Returns:
        Dict with upper_band, middle_band, lower_band, and bandwidth
    """
    if len(prices) < period:
        raise ValueError(f"Need at least {period} prices for Bollinger Bands")

    # Calculate simple moving average
    sma = sum(prices[-period:]) / period

    # Calculate standard deviation
    variance = sum((p - sma) ** 2 for p in prices[-period:]) / period
    std_dev = variance ** 0.5

    # Calculate bands
    upper_band = sma + (num_std * std_dev)
    lower_band = sma - (num_std * std_dev)

    # Calculate bandwidth
    bandwidth = (upper_band - lower_band) / sma if sma > 0 else 0

    # Current price position
    current_price = prices[-1]
    position = "above_upper" if current_price > upper_band else "below_lower" if current_price < lower_band else "within_bands"

    return {
        "upper_band": round(upper_band, 5),
        "middle_band": round(sma, 5),
        "lower_band": round(lower_band, 5),
        "bandwidth": round(bandwidth, 4),
        "current_price": round(current_price, 5),
        "position": position,
        "period": period,
        "std_deviations": num_std
    }


def calculate_moving_average(prices: List[float], period: int = 20, ma_type: str = "sma") -> Dict[str, Any]:
    """
    Calculate Moving Average (SMA or EMA).

    Args:
        prices: List of closing prices
        period: MA period (default: 20)
        ma_type: Type of MA - 'sma' or 'ema' (default: 'sma')

    Returns:
        Dict with current_ma value and historical values
    """
    if len(prices) < period:
        raise ValueError(f"Need at least {period} prices for moving average")

    if ma_type == "sma":
        # Simple Moving Average
        ma_value = sum(prices[-period:]) / period
    elif ma_type == "ema":
        # Exponential Moving Average
        multiplier = 2 / (period + 1)
        ma_value = prices[0]
        for price in prices[1:]:
            ma_value = (price - ma_value) * multiplier + ma_value
    else:
        raise ValueError(f"Invalid ma_type: {ma_type}. Use 'sma' or 'ema'")

    # Get historical MA values
    historical = []
    for i in range(period - 1, len(prices)):
        window = prices[i - period + 1:i + 1]
        if ma_type == "sma":
            historical.append(sum(window) / period)
        else:
            multiplier = 2 / (period + 1)
            ema_val = window[0]
            for price in window[1:]:
                ema_val = (price - ema_val) * multiplier + ema_val
            historical.append(ema_val)

    # Determine trend
    trend = "uptrend" if len(historical) > 1 and historical[-1] > historical[-2] else "downtrend" if len(historical) > 1 and historical[-1] < historical[-2] else "neutral"

    return {
        "current_ma": round(ma_value, 5),
        "ma_type": ma_type,
        "period": period,
        "trend": trend,
        "values_count": len(historical)
    }


def calculate_atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Dict[str, Any]:
    """
    Calculate Average True Range (ATR).

    Args:
        highs: List of high prices
        lows: List of low prices
        closes: List of close prices
        period: ATR period (default: 14)

    Returns:
        Dict with atr_value and signal
    """
    if not (len(highs) == len(lows) == len(closes)):
        raise ValueError("highs, lows, and closes must have the same length")

    if len(highs) < period + 1:
        raise ValueError(f"Need at least {period + 1} data points for ATR")

    # Calculate True Range for each period
    true_ranges = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1])
        )
        true_ranges.append(tr)

    # Calculate ATR using Wilder's smoothing
    atr = sum(true_ranges[:period]) / period
    for tr in true_ranges[period:]:
        atr = (atr * (period - 1) + tr) / period

    # Current ATR as percentage of price
    current_price = closes[-1]
    atr_percent = (atr / current_price) * 100 if current_price > 0 else 0

    # Signal based on ATR
    signal = "high_volatility" if atr_percent > 3 else "moderate" if atr_percent > 1 else "low"

    return {
        "atr_value": round(atr, 5),
        "atr_percent": round(atr_percent, 2),
        "signal": signal,
        "period": period,
        "current_price": round(current_price, 5)
    }


def calculate_stochastic(highs: List[float], lows: List[float], closes: List[float], k_period: int = 14, d_period: int = 3) -> Dict[str, Any]:
    """
    Calculate Stochastic Oscillator.

    Args:
        highs: List of high prices
        lows: List of low prices
        closes: List of close prices
        k_period: %K period (default: 14)
        d_period: %D period (default: 3)

    Returns:
        Dict with %K, %D, and signal
    """
    if not (len(highs) == len(lows) == len(closes)):
        raise ValueError("highs, lows, and closes must have the same length")

    if len(highs) < k_period + d_period:
        raise ValueError(f"Need at least {k_period + d_period} data points for Stochastic")

    # Calculate %K
    k_values = []
    for i in range(k_period - 1, len(closes)):
        highest_high = max(highs[i - k_period + 1:i + 1])
        lowest_low = min(lows[i - k_period + 1:i + 1])
        k = 100 * (closes[i] - lowest_low) / (highest_high - lowest_low) if highest_high > lowest_low else 50
        k_values.append(k)

    # Calculate %D (SMA of %K)
    d_value = sum(k_values[-d_period:]) / d_period

    # Signal
    if k_values[-1] < 20 and d_value < 20:
        signal = "oversold"
    elif k_values[-1] > 80 and d_value > 80:
        signal = "overbought"
    elif k_values[-1] > d_value:
        signal = "bullish_crossover"
    elif k_values[-1] < d_value:
        signal = "bearish_crossover"
    else:
        signal = "neutral"

    return {
        "k_value": round(k_values[-1], 2),
        "d_value": round(d_value, 2),
        "signal": signal,
        "k_period": k_period,
        "d_period": d_period
    }


def calculate_obv(closes: List[float], volumes: List[float]) -> Dict[str, Any]:
    """
    Calculate On-Balance Volume (OBV).

    Args:
        closes: List of closing prices
        volumes: List of volume values

    Returns:
        Dict with obv_value and trend
    """
    if len(closes) != len(volumes):
        raise ValueError("closes and volumes must have the same length")

    if len(closes) < 2:
        raise ValueError("Need at least 2 data points for OBV")

    obv = 0
    for i in range(1, len(closes)):
        if closes[i] > closes[i - 1]:
            obv += volumes[i]
        elif closes[i] < closes[i - 1]:
            obv -= volumes[i]

    # Calculate OBV change
    obv_change = obv - volumes[0]

    # Trend based on OBV
    trend = "bullish" if obv_change > 0 else "bearish" if obv_change < 0 else "neutral"

    return {
        "obv_value": round(obv, 2),
        "obv_change": round(obv_change, 2),
        "trend": trend,
        "data_points": len(closes)
    }


# ============================================================================
# Risk Skills
# ============================================================================

def validate_risk_parameters(account_balance: float, position_size: float, stop_loss_pips: float, risk_percent: float, max_risk_percent: float = 2.0) -> Dict[str, Any]:
    """Validate risk parameters against limits."""
    violations = []
    if risk_percent > max_risk_percent:
        violations.append(f"Risk {risk_percent}% exceeds maximum {max_risk_percent}%")
    if position_size < 0.01:
        violations.append("Position size below minimum 0.01 lots")
    if stop_loss_pips <= 0:
        violations.append("Stop loss must be positive")
    return {"is_valid": len(violations) == 0, "violations": violations, "risk_percent": risk_percent, "max_risk_percent": max_risk_percent}


def calculate_portfolio_risk(positions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate overall portfolio risk metrics."""
    total_exposure = sum(pos.get("notional_value", 0) for pos in positions)
    total_risk = sum(pos.get("risk_amount", 0) for pos in positions)
    return {"total_positions": len(positions), "total_exposure": round(total_exposure, 2), "total_risk": round(total_risk, 2), "avg_risk_per_position": round(total_risk / len(positions), 2) if positions else 0}


def calculate_correlation_risk(asset_a_returns: List[float], asset_b_returns: List[float]) -> Dict[str, Any]:
    """Calculate correlation between two assets."""
    if len(asset_a_returns) != len(asset_b_returns):
        raise ValueError("Return series must have same length")
    n = len(asset_a_returns)
    if n < 2:
        return {"correlation": 0.0, "interpretation": "Insufficient data"}
    mean_a, mean_b = sum(asset_a_returns) / n, sum(asset_b_returns) / n
    numerator = sum((asset_a_returns[i] - mean_a) * (asset_b_returns[i] - mean_b) for i in range(n))
    denom_a = sum((r - mean_a) ** 2 for r in asset_a_returns) ** 0.5
    denom_b = sum((r - mean_b) ** 2 for r in asset_b_returns) ** 0.5
    correlation = numerator / (denom_a * denom_b) if denom_a > 0 and denom_b > 0 else 0.0
    interpretation = "high" if abs(correlation) > 0.7 else "moderate" if abs(correlation) > 0.3 else "low"
    return {"correlation": round(correlation, 4), "interpretation": interpretation, "sample_size": n}


def calculate_kelly_criterion(win_rate: float, avg_win: float, avg_loss: float) -> Dict[str, Any]:
    """
    Calculate Kelly Criterion for optimal position sizing.

    Args:
        win_rate: Win rate as decimal (e.g., 0.55 for 55%)
        avg_win: Average win amount
        avg_loss: Average loss amount (positive value)

    Returns:
        Dict with kelly_percentage, optimal_fraction, and recommendation
    """
    if not 0 < win_rate < 1:
        raise ValueError("win_rate must be between 0 and 1")

    if avg_win <= 0 or avg_loss <= 0:
        raise ValueError("avg_win and avg_loss must be positive")

    # Kelly Formula: K% = W - (1-W) / (W/L)
    win_loss_ratio = avg_win / avg_loss
    kelly_fraction = win_rate - ((1 - win_rate) / win_loss_ratio)

    # Convert to percentage
    kelly_percentage = kelly_fraction * 100

    # Half-Kelly for more conservative approach
    half_kelly = kelly_percentage / 2

    # Recommendation based on Kelly
    if kelly_fraction <= 0:
        recommendation = "do_not_trade"
        optimal_fraction = 0
    elif kelly_fraction > 0.25:
        recommendation = "use_fractional_kelly"
        optimal_fraction = round(half_kelly, 2)
    else:
        recommendation = "use_kelly"
        optimal_fraction = round(kelly_percentage, 2)

    return {
        "kelly_percentage": round(kelly_percentage, 2),
        "half_kelly_percentage": round(half_kelly, 2),
        "optimal_fraction": optimal_fraction,
        "recommendation": recommendation,
        "win_rate": win_rate,
        "win_loss_ratio": round(win_loss_ratio, 2)
    }


def calculate_value_at_risk(returns: List[float], confidence_level: float = 0.95, holding_period: int = 1) -> Dict[str, Any]:
    """
    Calculate Value at Risk (VaR) using historical method.

    Args:
        returns: List of historical returns
        confidence_level: Confidence level (default: 0.95 for 95%)
        holding_period: Holding period in days (default: 1)

    Returns:
        Dict with var_value, var_percent, and risk_level
    """
    if len(returns) < 30:
        raise ValueError("Need at least 30 data points for VaR calculation")

    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")

    # Sort returns
    sorted_returns = sorted(returns)
    index = int((1 - confidence_level) * len(sorted_returns))
    var = sorted_returns[index] if index < len(sorted_returns) else sorted_returns[-1]

    # Scale to holding period
    var_scaled = var * (holding_period ** 0.5)

    # Convert to percentage
    var_percent = var_scaled * 100

    # Risk level
    risk_level = "high" if var_percent > 5 else "moderate" if var_percent > 2 else "low"

    return {
        "var_value": round(var_scaled, 4),
        "var_percent": round(var_percent, 2),
        "confidence_level": confidence_level,
        "holding_period": holding_period,
        "risk_level": risk_level,
        "sample_size": len(returns)
    }


def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.02) -> Dict[str, Any]:
    """
    Calculate Sharpe Ratio.

    Args:
        returns: List of historical returns
        risk_free_rate: Risk-free rate (default: 2%)

    Returns:
        Dict with sharpe_ratio and interpretation
    """
    if len(returns) < 2:
        raise ValueError("Need at least 2 data points for Sharpe Ratio")

    # Calculate mean return
    mean_return = sum(returns) / len(returns)

    # Calculate standard deviation
    variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
    std_dev = variance ** 0.5

    if std_dev == 0:
        return {"sharpe_ratio": 0.0, "interpretation": "No volatility in returns"}

    # Calculate excess return
    excess_return = mean_return - risk_free_rate

    # Calculate Sharpe Ratio (annualized)
    sharpe_ratio = (excess_return / std_dev) * (252 ** 0.5)

    # Interpretation
    if sharpe_ratio >= 2:
        interpretation = "excellent"
    elif sharpe_ratio >= 1:
        interpretation = "good"
    elif sharpe_ratio >= 0.5:
        interpretation = "fair"
    elif sharpe_ratio >= 0:
        interpretation = "poor"
    else:
        interpretation = "negative"

    return {
        "sharpe_ratio": round(sharpe_ratio, 2),
        "interpretation": interpretation,
        "mean_return": round(mean_return, 4),
        "std_dev": round(std_dev, 4),
        "risk_free_rate": risk_free_rate,
        "sample_size": len(returns)
    }


def calculate_max_drawdown(equity_curve: List[float]) -> Dict[str, Any]:
    """
    Calculate Maximum Drawdown.

    Args:
        equity_curve: List of equity values over time

    Returns:
        Dict with max_drawdown, max_drawdown_percent, and duration
    """
    if len(equity_curve) < 2:
        raise ValueError("Need at least 2 data points for drawdown calculation")

    if any(e <= 0 for e in equity_curve):
        raise ValueError("Equity values must be positive")

    peak = equity_curve[0]
    max_dd = 0
    max_dd_percent = 0
    drawdown_start = 0
    drawdown_end = 0
    current_drawdown_start = 0

    for i, equity in enumerate(equity_curve):
        if equity > peak:
            peak = equity
            current_drawdown_start = i

        drawdown = (peak - equity) / peak
        drawdown_percent = drawdown * 100

        if drawdown > max_dd:
            max_dd = drawdown
            max_dd_percent = drawdown_percent
            drawdown_start = current_drawdown_start
            drawdown_end = i

    # Calculate drawdown duration in periods
    duration = drawdown_end - drawdown_start

    # Risk assessment
    risk_level = "extreme" if max_dd_percent > 50 else "high" if max_dd_percent > 20 else "moderate" if max_dd_percent > 10 else "low"

    return {
        "max_drawdown": round(max_dd, 4),
        "max_drawdown_percent": round(max_dd_percent, 2),
        "drawdown_start_index": drawdown_start,
        "drawdown_end_index": drawdown_end,
        "duration_periods": duration,
        "risk_level": risk_level,
        "peak_value": round(max(equity_curve[:drawdown_end + 1]), 2),
        "trough_value": round(min(equity_curve[drawdown_start:drawdown_end + 1]), 2)
    }


def calculate_sortino_ratio(returns: List[float], target_return: float = 0.0, risk_free_rate: float = 0.02) -> Dict[str, Any]:
    """
    Calculate Sortino Ratio (downside deviation).

    Args:
        returns: List of historical returns
        target_return: Target/minimum acceptable return (default: 0)
        risk_free_rate: Risk-free rate (default: 2%)

    Returns:
        Dict with sortino_ratio and interpretation
    """
    if len(returns) < 2:
        raise ValueError("Need at least 2 data points for Sortino Ratio")

    # Calculate mean return
    mean_return = sum(returns) / len(returns)

    # Calculate downside returns only
    downside_returns = [r - target_return for r in returns if r < target_return]

    if not downside_returns:
        return {"sortino_ratio": float('inf'), "interpretation": "perfect", "note": "No downside returns"}

    # Calculate downside deviation
    downside_variance = sum((r - target_return) ** 2 for r in downside_returns) / len(returns)
    downside_dev = downside_variance ** 0.5

    if downside_dev == 0:
        return {"sortino_ratio": 0.0, "interpretation": "No downside deviation"}

    # Calculate Sortino Ratio (annualized)
    excess_return = mean_return - risk_free_rate
    sortino_ratio = (excess_return / downside_dev) * (252 ** 0.5)

    # Interpretation
    if sortino_ratio >= 2:
        interpretation = "excellent"
    elif sortino_ratio >= 1:
        interpretation = "good"
    elif sortino_ratio >= 0.5:
        interpretation = "fair"
    else:
        interpretation = "poor"

    return {
        "sortino_ratio": round(sortino_ratio, 2),
        "interpretation": interpretation,
        "downside_deviation": round(downside_dev, 4),
        "mean_return": round(mean_return, 4),
        "sample_size": len(returns)
    }


# ============================================================================
# Coding Skills
# ============================================================================

def analyze_code_complexity(code: str) -> Dict[str, Any]:
    """Analyze code complexity metrics."""
    lines, non_empty = code.split("\n"), [l for l in code.split("\n") if l.strip()]
    return {"lines_of_code": len(lines), "non_empty_lines": len(non_empty), "functions": code.count("def ") + code.count("async def "), "classes": code.count("class "), "imports": code.count("import ") + code.count("from "), "estimated_complexity": 1 + sum(1 for c in code if c in ("if", "elif", "for", "while", "and", "or"))}


def suggest_code_improvements(code: str, language: str = "python") -> Dict[str, Any]:
    """Suggest improvements for code."""
    suggestions = []
    if "==" in code and "is" not in code:
        suggestions.append({"type": "style", "message": "Consider using 'is' for identity comparisons"})
    if "except:" in code:
        suggestions.append({"type": "error_handling", "message": "Avoid bare except - specify exception types"})
    if "print(" in code:
        suggestions.append({"type": "logging", "message": "Use logging module instead of print for production"})
    return {"language": language, "suggestions": suggestions, "count": len(suggestions)}


def generate_documentation(doc_type: str, content: Dict[str, Any]) -> Dict[str, Any]:
    """Generate documentation for code or strategy."""
    if doc_type == "function":
        params = ", ".join(content.get("params", []))
        name = content.get("name", "Function")
        desc = content.get("description", "")
        ret = content.get("returns", "None")
        docstring = f'"""\n{name} - {desc}\n\nArgs:\n    {params}\n\nReturns:\n    {ret}\n"""'
        return {"docstring": docstring}
    elif doc_type == "strategy":
        return {"description": content.get("description", ""), "entry_rules": content.get("entry_rules", []), "exit_rules": content.get("exit_rules", []), "risk_parameters": content.get("risk_parameters", {})}
    return {"content": str(content)}


# ============================================================================
# Data Skills
# ============================================================================

def fetch_historical_data(symbol: str, timeframe: str, start_date: str, end_date: Optional[str] = None) -> Dict[str, Any]:
    """
    Fetch historical market data.

    Args:
        symbol: Trading symbol (e.g., 'EURUSD')
        timeframe: Timeframe (e.g., '1H', '4H', '1D')
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD), defaults to now

    Returns:
        Dict with data points and metadata
    """
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "start_date": start_date,
        "end_date": end_date or "current",
        "data": [],
        "count": 0,
        "message": "Historical data fetch pending integration"
    }


def fetch_live_tick(symbol: str) -> Dict[str, Any]:
    """
    Fetch live tick data for a symbol.

    Args:
        symbol: Trading symbol (e.g., 'EURUSD')

    Returns:
        Dict with current bid, ask, and timestamp
    """
    return {
        "symbol": symbol,
        "bid": 0.0,
        "ask": 0.0,
        "last": 0.0,
        "volume": 0,
        "timestamp": None,
        "message": "Live tick fetch pending integration"
    }


def resample_timeframe(data: List[Dict[str, Any]], target_timeframe: str, price_field: str = "close") -> Dict[str, Any]:
    """
    Resample OHLCV data to a different timeframe.

    Args:
        data: List of OHLCV records
        target_timeframe: Target timeframe (e.g., '1H', '4H', '1D')
        price_field: Field to use for aggregation (default: 'close')

    Returns:
        Dict with resampled data and metadata
    """
    if not data:
        return {"resampled_data": [], "count": 0, "target_timeframe": target_timeframe}

    return {
        "resampled_data": data,
        "count": len(data),
        "target_timeframe": target_timeframe,
        "aggregation_method": "ohlc",
        "message": "Timeframe resampling pending integration"
    }


def clean_data_anomalies(data: List[float], method: str = "zscore", threshold: float = 3.0) -> Dict[str, Any]:
    """
    Clean anomalies from price or indicator data.

    Args:
        data: List of numerical values
        method: Detection method ('zscore', 'iqr', 'mad')
        threshold: Threshold for anomaly detection

    Returns:
        Dict with cleaned data and detected anomalies
    """
    if not data:
        return {"cleaned_data": [], "anomalies": [], "count": 0}

    if method == "zscore":
        mean = sum(data) / len(data)
        variance = sum((x - mean) ** 2 for x in data) / len(data)
        std_dev = variance ** 0.5

        anomalies = []
        cleaned = []

        for i, value in enumerate(data):
            zscore = abs((value - mean) / std_dev) if std_dev > 0 else 0
            if zscore > threshold:
                anomalies.append({"index": i, "value": value, "zscore": round(zscore, 2)})
                # Replace with mean
                cleaned.append(mean)
            else:
                cleaned.append(value)
    elif method == "iqr":
        sorted_data = sorted(data)
        q1_idx = len(data) // 4
        q3_idx = 3 * len(data) // 4
        q1 = sorted_data[q1_idx]
        q3 = sorted_data[q3_idx]
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr

        anomalies = []
        cleaned = []

        for i, value in enumerate(data):
            if value < lower or value > upper:
                anomalies.append({"index": i, "value": value, "bounds": (lower, upper)})
                # Replace with median
                cleaned.append(sorted_data[len(data) // 2])
            else:
                cleaned.append(value)
    else:
        return {"error": f"Unknown method: {method}"}

    return {
        "cleaned_data": cleaned,
        "anomalies": anomalies,
        "anomaly_count": len(anomalies),
        "original_count": len(data),
        "method": method,
        "threshold": threshold
    }


def calculate_returns(prices: List[float], method: str = "simple") -> Dict[str, Any]:
    """
    Calculate returns from price series.

    Args:
        prices: List of closing prices
        method: Return calculation method ('simple' or 'log')

    Returns:
        Dict with returns array and statistics
    """
    if len(prices) < 2:
        raise ValueError("Need at least 2 prices to calculate returns")

    returns = []

    if method == "simple":
        for i in range(1, len(prices)):
            ret = (prices[i] - prices[i - 1]) / prices[i - 1]
            returns.append(ret)
    elif method == "log":
        for i in range(1, len(prices)):
            ret = (prices[i] / prices[i - 1])
            returns.append(ret)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Calculate statistics
    mean_return = sum(returns) / len(returns)
    variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
    std_dev = variance ** 0.5

    return {
        "returns": returns,
        "count": len(returns),
        "mean_return": round(mean_return, 6),
        "std_dev": round(std_dev, 6),
        "min_return": round(min(returns), 6),
        "max_return": round(max(returns), 6),
        "method": method
    }


def normalize_data(data: List[float], method: str = "minmax") -> Dict[str, Any]:
    """
    Normalize data using various methods.

    Args:
        data: List of numerical values
        method: Normalization method ('minmax', 'zscore', 'robust')

    Returns:
        Dict with normalized data and parameters
    """
    if not data:
        return {"normalized_data": [], "method": method}

    if method == "minmax":
        min_val = min(data)
        max_val = max(data)
        range_val = max_val - min_val

        if range_val == 0:
            return {"normalized_data": [0.5] * len(data), "method": method, "note": "All values identical"}

        normalized = [(x - min_val) / range_val for x in data]
        params = {"min": min_val, "max": max_val}
    elif method == "zscore":
        mean = sum(data) / len(data)
        variance = sum((x - mean) ** 2 for x in data) / len(data)
        std_dev = variance ** 0.5

        if std_dev == 0:
            return {"normalized_data": [0.0] * len(data), "method": method, "note": "No variance"}

        normalized = [(x - mean) / std_dev for x in data]
        params = {"mean": mean, "std_dev": std_dev}
    elif method == "robust":
        sorted_data = sorted(data)
        median = sorted_data[len(data) // 2]
        q1_idx = len(data) // 4
        q3_idx = 3 * len(data) // 4
        q1 = sorted_data[q1_idx]
        q3 = sorted_data[q3_idx]
        iqr = q3 - q1

        if iqr == 0:
            return {"normalized_data": [0.0] * len(data), "method": method, "note": "Zero IQR"}

        normalized = [(x - median) / iqr for x in data]
        params = {"median": median, "iqr": iqr}
    else:
        return {"error": f"Unknown method: {method}"}

    return {
        "normalized_data": normalized,
        "method": method,
        "params": params,
        "count": len(data)
    }


# ============================================================================
# Core Skills - Story 7.4: 12 Required Skills
# ============================================================================

def financial_data_fetch(symbol: str, data_type: str = "ohlcv", timeframe: str = "1D", start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, Any]:
    """
    Fetch financial market data for a given symbol.

    Args:
        symbol: Trading symbol (e.g., 'EURUSD', 'AAPL')
        data_type: Type of data ('ohlcv', 'tick', 'fundamental')
        timeframe: Timeframe ('1m', '5m', '15m', '1H', '4H', '1D', '1W')
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        Dict with market data and metadata
    """
    return {
        "symbol": symbol,
        "data_type": data_type,
        "timeframe": timeframe,
        "start_date": start_date,
        "end_date": end_date,
        "data": [],
        "count": 0,
        "message": "Financial data fetch - integration pending with data providers"
    }


def pattern_scanner(prices: List[float], pattern_type: str = "all") -> Dict[str, Any]:
    """
    Scan for chart patterns in price data.

    Args:
        prices: List of closing prices
        pattern_type: Pattern to search for ('all', 'head_shoulders', 'triangle', 'double_top', 'double_bottom')

    Returns:
        Dict with detected patterns and confidence scores
    """
    if len(prices) < 20:
        return {"patterns": [], "count": 0, "message": "Insufficient data for pattern detection"}

    patterns = []
    # Basic pattern detection (placeholder implementation)
    if len(prices) >= 50:
        # Simple moving average crossover detection
        short_ma = sum(prices[-10:]) / 10
        long_ma = sum(prices[-50:]) / 50
        if short_ma > long_ma:
            patterns.append({"type": "bullish_ma_crossover", "confidence": 0.7})

    return {
        "patterns": patterns,
        "count": len(patterns),
        "symbol": "unknown",
        "timeframe": "1D"
    }


def statistical_edge(returns: List[float], benchmark_returns: Optional[List[float]] = None) -> Dict[str, Any]:
    """
    Calculate statistical edge metrics for a strategy.

    Args:
        returns: List of strategy returns
        benchmark_returns: Optional benchmark returns for comparison

    Returns:
        Dict with alpha, beta, sharpe, information ratio
    """
    if len(returns) < 2:
        return {"error": "Insufficient data for statistical analysis"}

    mean_return = sum(returns) / len(returns)
    variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
    std_dev = variance ** 0.5

    # Basic metrics
    sharpe = (mean_return / std_dev * (252 ** 0.5)) if std_dev > 0 else 0
    win_rate = sum(1 for r in returns if r > 0) / len(returns)
    avg_win = sum(r for r in returns if r > 0) / max(1, sum(1 for r in returns if r > 0))
    avg_loss = abs(sum(r for r in returns if r < 0) / max(1, sum(1 for r in returns if r < 0)))

    # Profit factor
    profit_factor = (avg_win * win_rate) / (avg_loss * (1 - win_rate)) if (avg_loss * (1 - win_rate)) > 0 else 0

    result = {
        "mean_return": round(mean_return, 6),
        "std_dev": round(std_dev, 6),
        "sharpe_ratio": round(sharpe, 2),
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 2),
        "sample_size": len(returns)
    }

    if benchmark_returns and len(benchmark_returns) == len(returns):
        bench_mean = sum(benchmark_returns) / len(benchmark_returns)
        bench_std = (sum((r - bench_mean) ** 2 for r in benchmark_returns) / len(benchmark_returns)) ** 0.5
        if bench_std > 0:
            beta = (sum((returns[i] - mean_return) * (benchmark_returns[i] - bench_mean) for i in range(len(returns))) / len(returns)) / bench_std ** 2
            alpha = mean_return - beta * bench_mean
            result["alpha"] = round(alpha, 6)
            result["beta"] = round(beta, 2)

    return result


def hypothesis_document_writer(research_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a structured hypothesis document from research data.

    Args:
        research_data: Research findings and data

    Returns:
        Dict with formatted hypothesis document
    """
    hypothesis = research_data.get("hypothesis", "")
    market_conditions = research_data.get("market_conditions", "")
    timeframes = research_data.get("timeframes", [])
    risk_params = research_data.get("risk_params", {})

    return {
        "title": f"Hypothesis: {hypothesis[:50] if hypothesis else 'Untitled'}",
        "hypothesis": hypothesis,
        "market_conditions": market_conditions,
        "entry_criteria": research_data.get("entry_criteria", []),
        "exit_criteria": research_data.get("exit_criteria", []),
        "timeframes": timeframes,
        "risk_parameters": risk_params,
        "created_at": "2026-03-19",
        "version": "1.0.0"
    }


def mql5_generator(strategy_spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate MQL5 code from strategy specification.

    Args:
        strategy_spec: Strategy parameters and rules

    Returns:
        Dict with generated MQL5 code and metadata
    """
    strategy_name = strategy_spec.get("name", "Strategy")
    entry_rules = strategy_spec.get("entry_rules", [])
    exit_rules = strategy_spec.get("exit_rules", [])
    indicators = strategy_spec.get("indicators", [])

    # Basic MQL5 template generation
    mql5_code = f"""//+------------------------------------------------------------------+
//| {strategy_name}.mq5
//| Generated by QuantMindX Skill Forge
//+------------------------------------------------------------------+
#property copyright "QuantMindX"
#property version   "1.00"
#property strict

// Input parameters
input double lots = 0.1;
input int magicNumber = 123456;

//+------------------------------------------------------------------+
//| Expert initialization function                                    |
//+------------------------------------------------------------------+
int OnInit() {{
    // Initialize indicators
    return(INIT_SUCCEEDED);
}}

//+------------------------------------------------------------------+
//| Expert tick function                                              |
//+------------------------------------------------------------------+
void OnTick() {{
    // Entry rules: {entry_rules}
    // Exit rules: {exit_rules}
    // Indicators: {indicators}
    // Strategy logic to be implemented
}}

//+------------------------------------------------------------------+
"""
    return {
        "filename": f"{strategy_name.replace(' ', '_')}.mq5",
        "code": mql5_code,
        "strategy_name": strategy_name,
        "version": "1.0.0",
        "language": "MQL5"
    }


def backtest_launcher(symbol: str, strategy_params: Dict[str, Any], start_date: str, end_date: str, timeframe: str = "1H") -> Dict[str, Any]:
    """
    Launch a backtest for a given strategy.

    Args:
        symbol: Trading symbol
        strategy_params: Strategy configuration
        start_date: Backtest start date
        end_date: Backtest end date
        timeframe: Timeframe for backtest

    Returns:
        Dict with backtest ID and status
    """
    import uuid
    backtest_id = str(uuid.uuid4())[:8]

    return {
        "backtest_id": backtest_id,
        "status": "queued",
        "symbol": symbol,
        "timeframe": timeframe,
        "start_date": start_date,
        "end_date": end_date,
        "progress": 0,
        "message": "Backtest queued for execution"
    }


def news_classifier(headlines: List[str]) -> Dict[str, Any]:
    """
    Classify news headlines by sentiment and category.

    Args:
        headlines: List of news headlines

    Returns:
        Dict with classified news and sentiment scores
    """
    categories = {
        "positive": ["gain", "rise", "surge", "bullish", "growth", "profit", "success"],
        "negative": ["loss", "fall", "bearish", "decline", "risk", "warning", "crash"],
        "neutral": ["report", "announce", "meeting", "update", "data"]
    }

    results = []
    for headline in headlines:
        headline_lower = headline.lower()
        sentiment = "neutral"
        category = "general"

        for cat, keywords in categories.items():
            if any(kw in headline_lower for kw in keywords):
                sentiment = cat
                break

        if any(w in headline_lower for w in ["fed", "rate", "interest", "inflation"]):
            category = "central_bank"
        elif any(w in headline_lower for w in ["earnings", "revenue", "profit", "quarter"]):
            category = "earnings"
        elif any(w in headline_lower for w in ["trade", "tariff", "export", "import"]):
            category = "trade"

        results.append({
            "headline": headline,
            "sentiment": sentiment,
            "category": category
        })

    return {
        "news": results,
        "count": len(results),
        "summary": {
            "positive": sum(1 for r in results if r["sentiment"] == "positive"),
            "negative": sum(1 for r in results if r["sentiment"] == "negative"),
            "neutral": sum(1 for r in results if r["sentiment"] == "neutral")
        }
    }


def risk_evaluator(position: Dict[str, Any], account_balance: float) -> Dict[str, Any]:
    """
    Evaluate risk for a potential trade position.

    Args:
        position: Position parameters (entry, stop_loss, take_profit, symbol)
        account_balance: Current account balance

    Returns:
        Dict with risk metrics and recommendations
    """
    entry = position.get("entry", 0)
    stop_loss = position.get("stop_loss", 0)
    take_profit = position.get("take_profit", 0)
    symbol = position.get("symbol", "UNKNOWN")
    position_size = position.get("position_size", 0.1)

    if entry == 0 or stop_loss == 0:
        return {"error": "Invalid entry or stop loss"}

    # Calculate risk
    risk_per_pip = abs(entry - stop_loss)
    risk_amount = position_size * risk_per_pip
    risk_percent = (risk_amount / account_balance * 100) if account_balance > 0 else 0

    # Reward to risk ratio
    reward = abs(take_profit - entry) if take_profit > 0 else 0
    reward_risk_ratio = reward / risk_per_pip if risk_per_pip > 0 else 0

    # Risk level assessment
    risk_level = "low" if risk_percent < 1 else "medium" if risk_percent < 2 else "high"

    recommendation = "approve" if risk_percent < 2 and reward_risk_ratio >= 2 else "review" if risk_percent < 3 else "reject"

    return {
        "symbol": symbol,
        "risk_percent": round(risk_percent, 2),
        "risk_amount": round(risk_amount, 2),
        "reward_risk_ratio": round(reward_risk_ratio, 2),
        "risk_level": risk_level,
        "recommendation": recommendation,
        "max_risk_percent": 2.0
    }


def report_writer(report_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate various types of reports.

    Args:
        report_type: Type of report ('performance', 'risk', 'trade', 'summary')
        data: Data to include in report

    Returns:
        Dict with formatted report content
    """
    if report_type == "performance":
        return {
            "report_type": "performance",
            "period": data.get("period", "unknown"),
            "total_return": data.get("total_return", 0),
            "sharpe_ratio": data.get("sharpe_ratio", 0),
            "max_drawdown": data.get("max_drawdown", 0),
            "win_rate": data.get("win_rate", 0),
            "trades": data.get("trades", []),
            "generated_at": "2026-03-19"
        }
    elif report_type == "risk":
        return {
            "report_type": "risk",
            "var_95": data.get("var_95", 0),
            "max_position_size": data.get("max_position_size", 0),
            "correlation_risk": data.get("correlation_risk", 0),
            "exposure_by_symbol": data.get("exposure_by_symbol", {}),
            "generated_at": "2026-03-19"
        }
    elif report_type == "trade":
        return {
            "report_type": "trade",
            "symbol": data.get("symbol", "UNKNOWN"),
            "entry_time": data.get("entry_time", ""),
            "exit_time": data.get("exit_time", ""),
            "pnl": data.get("pnl", 0),
            "duration": data.get("duration", 0),
            "generated_at": "2026-03-19"
        }
    else:
        return {
            "report_type": "summary",
            "content": str(data),
            "generated_at": "2026-03-19"
        }


def strategy_optimizer(strategy_params: Dict[str, Any], optimization_target: str = "sharpe") -> Dict[str, Any]:
    """
    Optimize strategy parameters for maximum performance.

    Args:
        strategy_params: Strategy parameters to optimize
        optimization_target: Target metric ('sharpe', 'profit', 'win_rate')

    Returns:
        Dict with optimized parameters
    """
    import random
    # Placeholder optimization - in production would use actual optimization
    optimized = {
        "original_params": strategy_params,
        "optimized_params": {
            "lot_size": strategy_params.get("lot_size", 0.1) * 1.1,
            "stop_loss": strategy_params.get("stop_loss", 50) * 0.95,
            "take_profit": strategy_params.get("take_profit", 100) * 1.05,
            "ma_period": strategy_params.get("ma_period", 20) + random.randint(-2, 2)
        },
        "expected_improvement": round(random.uniform(5, 20), 1),
        "target_metric": optimization_target,
        "status": "optimized"
    }

    return optimized


def institutional_data_fetch(data_source: str, query_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fetch institutional-grade data from various sources.

    Args:
        data_source: Data source identifier ('bloomberg', 'reuters', 'factset')
        query_params: Query parameters

    Returns:
        Dict with institutional data
    """
    return {
        "data_source": data_source,
        "query": query_params,
        "data": [],
        "count": 0,
        "message": "Institutional data integration pending",
        "requires_license": True
    }


def calendar_gate_check(current_time: str, calendar_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Check if trading is allowed based on calendar rules.

    Args:
        current_time: Current timestamp (YYYY-MM-DD HH:MM)
        calendar_config: Calendar configuration

    Returns:
        Dict with gate status and reason
    """
    # Default calendar rules
    config = calendar_config or {}
    allowed_start = config.get("trading_start", "09:00")
    allowed_end = config.get("trading_end", "17:00")
    weekend_trading = config.get("weekend_trading", False)

    # Parse time
    try:
        hour = int(current_time.split(" ")[1].split(":")[0]) if " " in current_time else 12
    except:
        hour = 12

    # Check if within trading hours
    start_hour = int(allowed_start.split(":")[0])
    end_hour = int(allowed_end.split(":")[0])

    # Basic day check
    import datetime
    try:
        dt = datetime.datetime.strptime(current_time.split(" ")[0], "%Y-%m-%d")
        is_weekend = dt.weekday() >= 5
    except:
        is_weekend = False

    # Determine gate status
    if is_weekend and not weekend_trading:
        return {
            "gate_open": False,
            "reason": "weekend",
            "next_open": "Monday 09:00"
        }
    elif hour < start_hour or hour >= end_hour:
        return {
            "gate_open": False,
            "reason": "outside_hours",
            "trading_start": allowed_start,
            "trading_end": allowed_end
        }
    else:
        return {
            "gate_open": True,
            "reason": "normal",
            "current_time": current_time
        }


def sdd_spec_builder(
    source: str,
    source_type: str = "trd",
    strategy_name: str = "",
    open_questions: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Generate a Spec-Driven Development (SDD) Directive from a TRD or task description.

    Fires ONCE at Alpha Forge entry point (video/source → strategy Directive)
    before Development begins. Output = Directive used by DOE. Blocked in
    autonomous improvement loops.

    Args:
        source: Raw TRD text or task description to structure.
        source_type: One of "trd", "task_description", "video_transcript".
        strategy_name: Optional name for the strategy being specified.
        open_questions: Optional list of unresolved questions to surface.

    Returns:
        Dict with goal, success_criteria, constraints, acceptance_tests,
        open_questions, and source_type fields — the Strategy Directive.
    """
    if not source or not source.strip():
        raise ValueError("source must be a non-empty string")

    # Derive a goal statement from the source (first meaningful sentence)
    lines = [ln.strip() for ln in source.splitlines() if ln.strip()]
    goal_line = lines[0] if lines else source[:120]

    directive = {
        "directive_type": "SDD_SPEC",
        "strategy_name": strategy_name or "Unnamed Strategy",
        "source_type": source_type,
        "goal": goal_line,
        "success_criteria": [
            "EA compiles without errors",
            "Backtest MODE_A passes with positive expectancy",
            "Vanilla variant mirrors source strategy as closely as possible",
        ],
        "constraints": [
            "First EA variant must not add system-native adaptations (vanilla mirror)",
            "Spiced variants are produced only in subsequent Alpha Forge loop iterations",
            "SDD does NOT enforce system constraints — constraint application happens at compile time",
            "SDD MUST NOT run in the autonomous improvement loop",
        ],
        "acceptance_tests": [
            "TRD parses without validation errors",
            "MQL5 generator produces syntactically valid code",
            "EA passes basic compilation",
            "MODE_A backtest completes within 60 seconds",
        ],
        "open_questions": open_questions or [],
        "resolution_path": "graph_memory → shared_assets → Copilot → Mubarak (if unresolved)",
        "used_by": ["development_dept", "alpha_forge_workflow", "copilot"],
        "access": "restricted",
    }
    return directive


# ============================================================================
# Skill Registration
# ============================================================================

def register_builtin_skills(manager: SkillManager = None) -> SkillManager:
    """Register all built-in skills with a SkillManager."""
    mgr = manager or get_skill_manager()

    # Research Department Skills
    mgr.register(
        "knowledge_search", knowledge_search,
        "Search the knowledge base for information", "research",
        departments=["research"],
        parameters={"required": ["query"], "properties": {"query": {"type": "string"}, "top_k": {"type": "integer", "default": 5}}},
        tags=["knowledge", "search", "research"]
    )
    mgr.register(
        "extract_trading_rules", extract_trading_rules,
        "Extract trading rules from documents", "research",
        departments=["research"],
        parameters={"required": ["document"], "properties": {"document": {"type": "string"}}},
        tags=["extraction", "rules", "research"]
    )
    mgr.register(
        "research_summary", research_summary,
        "Generate research summary on a topic", "research",
        departments=["research"],
        parameters={"required": ["topic"], "properties": {"topic": {"type": "string"}, "depth": {"type": "string", "default": "brief"}}},
        tags=["research", "summary"]
    )

    # Trading Department Skills
    mgr.register(
        "calculate_position_size", calculate_position_size,
        "Calculate position size based on risk parameters", "trading",
        departments=["trading", "risk"],
        parameters={"required": ["account_balance", "risk_percent", "stop_loss_pips"], "properties": {"account_balance": {"type": "number"}, "risk_percent": {"type": "number"}, "stop_loss_pips": {"type": "number"}, "pip_value": {"type": "number", "default": 10.0}}},
        tags=["trading", "position_sizing", "risk"]
    )
    mgr.register(
        "calculate_rsi", calculate_rsi,
        "Calculate Relative Strength Index", "trading",
        departments=["trading", "research"],
        parameters={"required": ["prices"], "properties": {"prices": {"type": "array", "items": {"type": "number"}}, "period": {"type": "integer", "default": 14}}},
        tags=["trading", "indicators", "technical"]
    )
    mgr.register(
        "detect_support_resistance", detect_support_resistance,
        "Detect support and resistance levels", "trading",
        departments=["trading", "research"],
        parameters={"required": ["highs", "lows", "closes"], "properties": {"highs": {"type": "array", "items": {"type": "number"}}, "lows": {"type": "array", "items": {"type": "number"}}, "closes": {"type": "array", "items": {"type": "number"}}, "lookback_period": {"type": "integer", "default": 5}}},
        tags=["trading", "support_resistance", "technical"]
    )
    mgr.register(
        "calculate_pivot_points", calculate_pivot_points,
        "Calculate pivot points and S/R levels", "trading",
        departments=["trading", "research"],
        parameters={"required": ["high", "low", "close"], "properties": {"high": {"type": "number"}, "low": {"type": "number"}, "close": {"type": "number"}}},
        tags=["trading", "pivot", "technical"]
    )

    # Risk Department Skills
    mgr.register(
        "validate_risk_parameters", validate_risk_parameters,
        "Validate risk parameters against limits", "risk",
        departments=["risk"],
        parameters={"required": ["account_balance", "position_size", "stop_loss_pips", "risk_percent"], "properties": {"account_balance": {"type": "number"}, "position_size": {"type": "number"}, "stop_loss_pips": {"type": "number"}, "risk_percent": {"type": "number"}, "max_risk_percent": {"type": "number", "default": 2.0}}},
        tags=["risk", "validation", "trading"]
    )
    mgr.register(
        "calculate_portfolio_risk", calculate_portfolio_risk,
        "Calculate portfolio-wide risk metrics", "risk",
        departments=["risk", "portfolio"],
        parameters={"required": ["positions"], "properties": {"positions": {"type": "array"}}},
        tags=["risk", "portfolio", "trading"]
    )
    mgr.register(
        "calculate_correlation_risk", calculate_correlation_risk,
        "Calculate correlation between assets", "risk",
        departments=["risk", "portfolio"],
        parameters={"required": ["asset_a_returns", "asset_b_returns"], "properties": {"asset_a_returns": {"type": "array", "items": {"type": "number"}}, "asset_b_returns": {"type": "array", "items": {"type": "number"}}}},
        tags=["risk", "correlation", "portfolio"]
    )

    # Development Department Skills (Coding)
    mgr.register(
        "analyze_code_complexity", analyze_code_complexity,
        "Analyze code complexity metrics", "coding",
        departments=["development"],
        parameters={"required": ["code"], "properties": {"code": {"type": "string"}}},
        tags=["coding", "analysis", "quality"]
    )
    mgr.register(
        "suggest_code_improvements", suggest_code_improvements,
        "Suggest code improvements", "coding",
        departments=["development"],
        parameters={"required": ["code"], "properties": {"code": {"type": "string"}, "language": {"type": "string", "default": "python"}}},
        tags=["coding", "improvements", "quality"]
    )
    mgr.register(
        "generate_documentation", generate_documentation,
        "Generate documentation for code or strategy", "coding",
        departments=["development"],
        parameters={"required": ["doc_type", "content"], "properties": {"doc_type": {"type": "string"}, "content": {"type": "object"}}},
        tags=["coding", "documentation"]
    )

    # Additional Trading Skills
    mgr.register(
        "calculate_macd", calculate_macd,
        "Calculate MACD (Moving Average Convergence Divergence)", "trading",
        departments=["trading", "research"],
        parameters={"required": ["prices"], "properties": {"prices": {"type": "array", "items": {"type": "number"}}, "fast_period": {"type": "integer", "default": 12}, "slow_period": {"type": "integer", "default": 26}, "signal_period": {"type": "integer", "default": 9}}},
        tags=["trading", "indicators", "macd", "technical"]
    )
    mgr.register(
        "calculate_bollinger_bands", calculate_bollinger_bands,
        "Calculate Bollinger Bands", "trading",
        departments=["trading", "research"],
        parameters={"required": ["prices"], "properties": {"prices": {"type": "array", "items": {"type": "number"}}, "period": {"type": "integer", "default": 20}, "num_std": {"type": "number", "default": 2.0}}},
        tags=["trading", "indicators", "bollinger", "technical"]
    )
    mgr.register(
        "calculate_moving_average", calculate_moving_average,
        "Calculate Moving Average (SMA or EMA)", "trading",
        departments=["trading", "research"],
        parameters={"required": ["prices", "period"], "properties": {"prices": {"type": "array", "items": {"type": "number"}}, "period": {"type": "integer"}, "ma_type": {"type": "string", "default": "sma"}}},
        tags=["trading", "indicators", "moving_average", "technical"]
    )
    mgr.register(
        "calculate_atr", calculate_atr,
        "Calculate Average True Range (ATR)", "trading",
        departments=["trading", "research", "risk"],
        parameters={"required": ["highs", "lows", "closes"], "properties": {"highs": {"type": "array", "items": {"type": "number"}}, "lows": {"type": "array", "items": {"type": "number"}}, "closes": {"type": "array", "items": {"type": "number"}}, "period": {"type": "integer", "default": 14}}},
        tags=["trading", "indicators", "atr", "volatility", "technical"]
    )
    mgr.register(
        "calculate_stochastic", calculate_stochastic,
        "Calculate Stochastic Oscillator", "trading",
        departments=["trading", "research"],
        parameters={"required": ["highs", "lows", "closes"], "properties": {"highs": {"type": "array", "items": {"type": "number"}}, "lows": {"type": "array", "items": {"type": "number"}}, "closes": {"type": "array", "items": {"type": "number"}}, "k_period": {"type": "integer", "default": 14}, "d_period": {"type": "integer", "default": 3}}},
        tags=["trading", "indicators", "stochastic", "technical"]
    )
    mgr.register(
        "calculate_obv", calculate_obv,
        "Calculate On-Balance Volume (OBV)", "trading",
        departments=["trading", "research"],
        parameters={"required": ["closes", "volumes"], "properties": {"closes": {"type": "array", "items": {"type": "number"}}, "volumes": {"type": "array", "items": {"type": "number"}}}},
        tags=["trading", "indicators", "obv", "volume"]
    )

    # Additional Risk Skills
    mgr.register(
        "calculate_kelly_criterion", calculate_kelly_criterion,
        "Calculate Kelly Criterion for optimal position sizing", "risk",
        departments=["risk", "trading"],
        parameters={"required": ["win_rate", "avg_win", "avg_loss"], "properties": {"win_rate": {"type": "number"}, "avg_win": {"type": "number"}, "avg_loss": {"type": "number"}}},
        tags=["risk", "position_sizing", "kelly"]
    )
    mgr.register(
        "calculate_value_at_risk", calculate_value_at_risk,
        "Calculate Value at Risk (VaR)", "risk",
        departments=["risk", "portfolio"],
        parameters={"required": ["returns"], "properties": {"returns": {"type": "array", "items": {"type": "number"}}, "confidence_level": {"type": "number", "default": 0.95}, "holding_period": {"type": "integer", "default": 1}}},
        tags=["risk", "var", "portfolio"]
    )
    mgr.register(
        "calculate_sharpe_ratio", calculate_sharpe_ratio,
        "Calculate Sharpe Ratio", "risk",
        departments=["risk", "portfolio"],
        parameters={"required": ["returns"], "properties": {"returns": {"type": "array", "items": {"type": "number"}}, "risk_free_rate": {"type": "number", "default": 0.02}}},
        tags=["risk", "sharpe", "performance"]
    )
    mgr.register(
        "calculate_max_drawdown", calculate_max_drawdown,
        "Calculate Maximum Drawdown", "risk",
        departments=["risk", "portfolio"],
        parameters={"required": ["equity_curve"], "properties": {"equity_curve": {"type": "array", "items": {"type": "number"}}}},
        tags=["risk", "drawdown", "performance"]
    )
    mgr.register(
        "calculate_sortino_ratio", calculate_sortino_ratio,
        "Calculate Sortino Ratio", "risk",
        departments=["risk", "portfolio"],
        parameters={"required": ["returns"], "properties": {"returns": {"type": "array", "items": {"type": "number"}}, "target_return": {"type": "number", "default": 0.0}, "risk_free_rate": {"type": "number", "default": 0.02}}},
        tags=["risk", "sortino", "performance"]
    )

    # Data Skills
    mgr.register(
        "fetch_historical_data", fetch_historical_data,
        "Fetch historical market data", "data",
        departments=["research", "trading"],
        parameters={"required": ["symbol", "timeframe", "start_date"], "properties": {"symbol": {"type": "string"}, "timeframe": {"type": "string"}, "start_date": {"type": "string"}, "end_date": {"type": "string"}}},
        tags=["data", "historical", "market"]
    )
    mgr.register(
        "fetch_live_tick", fetch_live_tick,
        "Fetch live tick data for a symbol", "data",
        departments=["trading"],
        parameters={"required": ["symbol"], "properties": {"symbol": {"type": "string"}}},
        tags=["data", "live", "tick", "market"]
    )
    mgr.register(
        "resample_timeframe", resample_timeframe,
        "Resample OHLCV data to different timeframe", "data",
        departments=["research", "trading"],
        parameters={"required": ["data", "target_timeframe"], "properties": {"data": {"type": "array"}, "target_timeframe": {"type": "string"}, "price_field": {"type": "string", "default": "close"}}},
        tags=["data", "resample", "timeframe"]
    )
    mgr.register(
        "clean_data_anomalies", clean_data_anomalies,
        "Clean anomalies from price or indicator data", "data",
        departments=["research", "trading"],
        parameters={"required": ["data"], "properties": {"data": {"type": "array", "items": {"type": "number"}}, "method": {"type": "string", "default": "zscore"}, "threshold": {"type": "number", "default": 3.0}}},
        tags=["data", "cleaning", "anomalies"]
    )
    mgr.register(
        "calculate_returns", calculate_returns,
        "Calculate returns from price series", "data",
        departments=["research", "trading", "risk"],
        parameters={"required": ["prices"], "properties": {"prices": {"type": "array", "items": {"type": "number"}}, "method": {"type": "string", "default": "simple"}}},
        tags=["data", "returns", "analysis"]
    )
    mgr.register(
        "normalize_data", normalize_data,
        "Normalize data using various methods", "data",
        departments=["research", "trading"],
        parameters={"required": ["data"], "properties": {"data": {"type": "array", "items": {"type": "number"}}, "method": {"type": "string", "default": "minmax"}}},
        tags=["data", "normalization", "analysis"]
    )

    # ============================================================================
    # Story 7.4: 12 Core Skills - Skill Catalogue Registry
    # ============================================================================

    # 1. financial_data_fetch
    mgr.register(
        "financial_data_fetch", financial_data_fetch,
        "Fetch financial market data for trading symbols", "data",
        departments=["research", "trading"],
        parameters={"required": ["symbol"], "properties": {
            "symbol": {"type": "string"},
            "data_type": {"type": "string", "default": "ohlcv"},
            "timeframe": {"type": "string", "default": "1D"},
            "start_date": {"type": "string"},
            "end_date": {"type": "string"}
        }},
        tags=["data", "market", "financial", "fetch"]
    )

    # 2. pattern_scanner
    mgr.register(
        "pattern_scanner", pattern_scanner,
        "Scan price data for chart patterns", "trading",
        departments=["research", "trading"],
        parameters={"required": ["prices"], "properties": {
            "prices": {"type": "array", "items": {"type": "number"}},
            "pattern_type": {"type": "string", "default": "all"}
        }},
        tags=["trading", "patterns", "technical", "analysis"]
    )

    # 3. statistical_edge
    mgr.register(
        "statistical_edge", statistical_edge,
        "Calculate statistical edge metrics for strategies", "risk",
        departments=["research", "risk"],
        parameters={"required": ["returns"], "properties": {
            "returns": {"type": "array", "items": {"type": "number"}},
            "benchmark_returns": {"type": "array", "items": {"type": "number"}}
        }},
        tags=["risk", "statistics", "analysis", "edge"]
    )

    # 4. hypothesis_document_writer
    mgr.register(
        "hypothesis_document_writer", hypothesis_document_writer,
        "Generate structured hypothesis documents from research", "research",
        departments=["research"],
        parameters={"required": ["research_data"], "properties": {
            "research_data": {"type": "object"}
        }},
        tags=["research", "documentation", "hypothesis", "writer"]
    )

    # 5. mql5_generator
    mgr.register(
        "mql5_generator", mql5_generator,
        "Generate MQL5 code from strategy specifications", "coding",
        departments=["development"],
        parameters={"required": ["strategy_spec"], "properties": {
            "strategy_spec": {"type": "object"}
        }},
        tags=["coding", "mql5", "strategy", "generator"]
    )

    # 6. backtest_launcher
    mgr.register(
        "backtest_launcher", backtest_launcher,
        "Launch backtests for trading strategies", "research",
        departments=["research", "development"],
        parameters={"required": ["symbol", "strategy_params", "start_date", "end_date"], "properties": {
            "symbol": {"type": "string"},
            "strategy_params": {"type": "object"},
            "start_date": {"type": "string"},
            "end_date": {"type": "string"},
            "timeframe": {"type": "string", "default": "1H"}
        }},
        tags=["research", "backtest", "testing", "strategy"]
    )

    # 7. news_classifier
    mgr.register(
        "news_classifier", news_classifier,
        "Classify news headlines by sentiment and category", "data",
        departments=["research", "trading"],
        parameters={"required": ["headlines"], "properties": {
            "headlines": {"type": "array", "items": {"type": "string"}}
        }},
        tags=["data", "news", "sentiment", "classification"]
    )

    # 8. risk_evaluator
    mgr.register(
        "risk_evaluator", risk_evaluator,
        "Evaluate risk metrics for trading positions", "risk",
        departments=["risk", "trading"],
        parameters={"required": ["position", "account_balance"], "properties": {
            "position": {"type": "object"},
            "account_balance": {"type": "number"}
        }},
        tags=["risk", "evaluation", "position", "trading"]
    )

    # 9. report_writer
    mgr.register(
        "report_writer", report_writer,
        "Generate various types of trading reports", "general",
        departments=["research", "trading", "risk", "portfolio"],
        parameters={"required": ["report_type", "data"], "properties": {
            "report_type": {"type": "string"},
            "data": {"type": "object"}
        }},
        tags=["reporting", "documents", "analysis"]
    )

    # 10. strategy_optimizer
    mgr.register(
        "strategy_optimizer", strategy_optimizer,
        "Optimize strategy parameters for maximum performance", "research",
        departments=["research", "development"],
        parameters={"required": ["strategy_params"], "properties": {
            "strategy_params": {"type": "object"},
            "optimization_target": {"type": "string", "default": "sharpe"}
        }},
        tags=["optimization", "strategy", "research"]
    )

    # 11. institutional_data_fetch
    mgr.register(
        "institutional_data_fetch", institutional_data_fetch,
        "Fetch institutional-grade data from premium sources", "data",
        departments=["research"],
        parameters={"required": ["data_source", "query_params"], "properties": {
            "data_source": {"type": "string"},
            "query_params": {"type": "object"}
        }},
        tags=["data", "institutional", "premium"]
    )

    # 12. calendar_gate_check
    mgr.register(
        "calendar_gate_check", calendar_gate_check,
        "Check if trading is allowed based on calendar rules", "system",
        departments=["trading", "risk"],
        parameters={"required": ["current_time"], "properties": {
            "current_time": {"type": "string"},
            "calendar_config": {"type": "object"}
        }},
        tags=["system", "calendar", "trading_hours", "gate"]
    )

    # 13. sdd_spec_builder
    mgr.register(
        "sdd-spec", sdd_spec_builder,
        "Generates an SDD (Spec-Driven Development) Directive from a TRD or task description."
        " Structures goal, success criteria, constraints, and acceptance tests."
        " Used ONCE at Alpha Forge entry point before Development begins.", "coding",
        departments=["development", "alpha_forge_workflow", "copilot"],
        parameters={
            "required": ["source"],
            "properties": {
                "source": {"type": "string", "description": "Raw TRD text or task description"},
                "source_type": {
                    "type": "string",
                    "enum": ["trd", "task_description", "video_transcript"],
                    "default": "trd"
                },
                "strategy_name": {"type": "string", "default": ""},
                "open_questions": {"type": "array", "items": {"type": "string"}},
            }
        },
        tags=["sdd", "spec", "directive", "alpha-forge", "development"]
    )

    logger.info(f"Registered {len(mgr.list_skills())} built-in skills")
    return mgr

"""
Technical Indicators Module
===========================

Provides technical indicator calculations (RSI, ATR, MACD).
"""

import numpy as np
import pandas as pd
from typing import Tuple


class TechnicalIndicators:
    """
    Technical indicators calculator.

    Provides static methods for calculating common technical indicators
    used in financial market analysis.
    """

    @staticmethod
    def rsi(prices: pd.Series, period: int = 14) -> float:
        """
        Calculate Relative Strength Index.

        Args:
            prices: Series of close prices
            period: RSI period

        Returns:
            RSI value (0-100)
        """
        if len(prices) < period + 1:
            return 50.0  # Neutral RSI

        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series,
            period: int = 14) -> float:
        """
        Calculate Average True Range.

        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of close prices
            period: ATR period

        Returns:
            ATR value
        """
        if len(close) < period + 1:
            return 0.0

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()

        return atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else 0.0

    @staticmethod
    def macd(prices: pd.Series, fast: int = 12,
             slow: int = 26, signal: int = 9) -> Tuple[float, float, float]:
        """
        Calculate MACD indicator.

        Args:
            prices: Series of close prices
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period

        Returns:
            Tuple of (MACD line, Signal line, Histogram)
        """
        if len(prices) < slow + signal:
            return (0.0, 0.0, 0.0)

        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return (
            macd_line.iloc[-1] if not pd.isna(macd_line.iloc[-1]) else 0.0,
            signal_line.iloc[-1] if not pd.isna(signal_line.iloc[-1]) else 0.0,
            histogram.iloc[-1] if not pd.isna(histogram.iloc[-1]) else 0.0
        )

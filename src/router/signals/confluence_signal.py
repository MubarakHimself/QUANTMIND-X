"""
Confluence Signal Generator

Evaluates multiple technical indicators to generate trading signals based on
indicator consensus. BUY/SELL signals require 2+ indicators to agree.
"""

import numpy as np
import pandas as pd
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple


class SignalType(str, Enum):
    """Trading signal types."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class IndicatorSignal:
    """Individual indicator signal result."""
    name: str
    signal: SignalType
    value: float
    details: Dict[str, float] = field(default_factory=dict)


@dataclass
class ConfluenceResult:
    """Result of confluence signal analysis."""
    primary_signal: SignalType
    indicators: List[IndicatorSignal]
    buy_count: int
    sell_count: int
    hold_count: int
    confidence: float  # 0-1 based on consensus strength
    details: Dict[str, Any] = field(default_factory=dict)


class ConfluenceSignal:
    """
    Confluence-based trading signal generator.

    Evaluates multiple technical indicators (RSI, MACD, Bollinger Bands, SMA, EMA)
    and generates signals based on consensus. BUY/SELL requires 2+ indicators to agree.
    """

    def __init__(
        self,
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        bb_period: int = 20,
        bb_std: float = 2.0,
        sma_period: int = 20,
        ema_period: int = 20,
        min_consensus: int = 2
    ):
        """
        Initialize ConfluenceSignal with indicator parameters.

        Args:
            rsi_period: RSI lookback period
            macd_fast: MACD fast EMA period
            macd_slow: MACD slow EMA period
            macd_signal: MACD signal line period
            bb_period: Bollinger Bands period
            bb_std: Bollinger Bands standard deviation multiplier
            sma_period: Simple Moving Average period
            ema_period: Exponential Moving Average period
            min_consensus: Minimum indicators that must agree for BUY/SELL
        """
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.sma_period = sma_period
        self.ema_period = ema_period
        self.min_consensus = min_consensus

    def calculate_rsi(self, prices: pd.Series) -> float:
        """Calculate RSI indicator."""
        if len(prices) < self.rsi_period + 1:
            return 50.0

        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        result = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
        return float(result)

    def calculate_macd(
        self, prices: pd.Series
    ) -> Tuple[float, float, float]:
        """Calculate MACD indicator. Returns (macd_line, signal_line, histogram)."""
        if len(prices) < self.macd_slow + self.macd_signal:
            return (0.0, 0.0, 0.0)

        ema_fast = prices.ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = prices.ewm(span=self.macd_slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.macd_signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return (
            float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else 0.0,
            float(signal_line.iloc[-1]) if not pd.isna(signal_line.iloc[-1]) else 0.0,
            float(histogram.iloc[-1]) if not pd.isna(histogram.iloc[-1]) else 0.0
        )

    def calculate_bollinger_bands(
        self, prices: pd.Series
    ) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands. Returns (upper, middle, lower)."""
        if len(prices) < self.bb_period:
            middle = prices.mean() if len(prices) > 0 else 0.0
            return (middle, middle, middle)

        middle = prices.rolling(window=self.bb_period).mean()
        std = prices.rolling(window=self.bb_period).std()

        upper = middle + (std * self.bb_std)
        lower = middle - (std * self.bb_std)

        middle_val = middle.iloc[-1] if not pd.isna(middle.iloc[-1]) else 0.0
        upper_val = upper.iloc[-1] if not pd.isna(upper.iloc[-1]) else middle_val
        lower_val = lower.iloc[-1] if not pd.isna(lower.iloc[-1]) else middle_val

        return (float(upper_val), float(middle_val), float(lower_val))

    def calculate_sma(self, prices: pd.Series) -> float:
        """Calculate Simple Moving Average."""
        if len(prices) < self.sma_period:
            return float(prices.mean()) if len(prices) > 0 else 0.0

        sma = prices.rolling(window=self.sma_period).mean()
        result = sma.iloc[-1] if not pd.isna(sma.iloc[-1]) else prices.mean()
        return float(result)

    def calculate_ema(self, prices: pd.Series) -> float:
        """Calculate Exponential Moving Average."""
        if len(prices) < self.ema_period:
            return float(prices.mean()) if len(prices) > 0 else 0.0

        ema = prices.ewm(span=self.ema_period, adjust=False).mean()
        result = ema.iloc[-1] if not pd.isna(ema.iloc[-1]) else prices.mean()
        return float(result)

    def get_rsi_signal(self, prices: pd.Series) -> IndicatorSignal:
        """Get RSI-based signal."""
        rsi = self.calculate_rsi(prices)

        if rsi < 30:
            signal = SignalType.BUY  # Oversold
        elif rsi > 70:
            signal = SignalType.SELL  # Overbought
        else:
            signal = SignalType.HOLD

        return IndicatorSignal(
            name="RSI",
            signal=signal,
            value=rsi,
            details={"rsi": rsi}
        )

    def get_macd_signal(self, prices: pd.Series) -> IndicatorSignal:
        """Get MACD-based signal."""
        macd_line, signal_line, histogram = self.calculate_macd(prices)

        if histogram > 0 and macd_line > signal_line:
            signal = SignalType.BUY
        elif histogram < 0 and macd_line < signal_line:
            signal = SignalType.SELL
        else:
            signal = SignalType.HOLD

        return IndicatorSignal(
            name="MACD",
            signal=signal,
            value=histogram,
            details={"macd": macd_line, "signal_line": signal_line, "histogram": histogram}
        )

    def get_bollinger_signal(self, prices: pd.Series) -> IndicatorSignal:
        """Get Bollinger Bands-based signal."""
        upper, middle, lower = self.calculate_bollinger_bands(prices)
        current_price = float(prices.iloc[-1])

        if current_price <= lower:
            signal = SignalType.BUY  # Near lower band - oversold
        elif current_price >= upper:
            signal = SignalType.SELL  # Near upper band - overbought
        else:
            # Check if price is in upper or lower half of bands
            range_size = upper - lower
            position = (current_price - lower) / range_size if range_size > 0 else 0.5

            if position < 0.3:
                signal = SignalType.BUY
            elif position > 0.7:
                signal = SignalType.SELL
            else:
                signal = SignalType.HOLD

        return IndicatorSignal(
            name="Bollinger",
            signal=signal,
            value=current_price,
            details={"upper": upper, "middle": middle, "lower": lower, "price": current_price}
        )

    def get_sma_signal(self, prices: pd.Series) -> IndicatorSignal:
        """Get SMA-based signal."""
        if len(prices) < self.sma_period + 1:
            return IndicatorSignal(
                name="SMA",
                signal=SignalType.HOLD,
                value=0.0,
                details={"sma": 0.0, "price": 0.0}
            )

        sma = self.calculate_sma(prices)
        current_price = float(prices.iloc[-1])

        if current_price > sma:
            signal = SignalType.BUY
        elif current_price < sma:
            signal = SignalType.SELL
        else:
            signal = SignalType.HOLD

        return IndicatorSignal(
            name="SMA",
            signal=signal,
            value=current_price - sma,
            details={"sma": sma, "price": current_price}
        )

    def get_ema_signal(self, prices: pd.Series) -> IndicatorSignal:
        """Get EMA-based signal."""
        if len(prices) < self.ema_period + 1:
            return IndicatorSignal(
                name="EMA",
                signal=SignalType.HOLD,
                value=0.0,
                details={"ema": 0.0, "price": 0.0}
            )

        ema = self.calculate_ema(prices)
        current_price = float(prices.iloc[-1])

        if current_price > ema:
            signal = SignalType.BUY
        elif current_price < ema:
            signal = SignalType.SELL
        else:
            signal = SignalType.HOLD

        return IndicatorSignal(
            name="EMA",
            signal=signal,
            value=current_price - ema,
            details={"ema": ema, "price": current_price}
        )

    def evaluate(self, prices: pd.Series) -> ConfluenceResult:
        """
        Evaluate all indicators and generate confluence signal.

        Args:
            prices: Series of closing prices

        Returns:
            ConfluenceResult with combined signal and indicator details
        """
        indicators = [
            self.get_rsi_signal(prices),
            self.get_macd_signal(prices),
            self.get_bollinger_signal(prices),
            self.get_sma_signal(prices),
            self.get_ema_signal(prices)
        ]

        buy_count = sum(1 for i in indicators if i.signal == SignalType.BUY)
        sell_count = sum(1 for i in indicators if i.signal == SignalType.SELL)
        hold_count = sum(1 for i in indicators if i.signal == SignalType.HOLD)

        # Determine primary signal based on consensus (prioritize stronger consensus)
        if buy_count >= self.min_consensus and buy_count >= sell_count:
            primary_signal = SignalType.BUY
        elif sell_count >= self.min_consensus and sell_count >= buy_count:
            primary_signal = SignalType.SELL
        elif buy_count >= self.min_consensus:
            # BUY meets threshold but SELL has more - favor SELL in conflict
            primary_signal = SignalType.SELL
        elif sell_count >= self.min_consensus:
            # SELL meets threshold but BUY has more - favor BUY in conflict
            primary_signal = SignalType.BUY
        else:
            primary_signal = SignalType.HOLD

        # Calculate confidence based on consensus strength
        total = len(indicators)
        if primary_signal == SignalType.BUY:
            confidence = buy_count / total
        elif primary_signal == SignalType.SELL:
            confidence = sell_count / total
        else:
            confidence = 0.0

        return ConfluenceResult(
            primary_signal=primary_signal,
            indicators=indicators,
            buy_count=buy_count,
            sell_count=sell_count,
            hold_count=hold_count,
            confidence=round(confidence, 2),
            details={
                "price": float(prices.iloc[-1]) if len(prices) > 0 else 0.0,
                "rsi": indicators[0].details.get("rsi", 50.0),
                "macd_histogram": indicators[1].details.get("histogram", 0.0)
            }
        )

    def analyze(self, prices: pd.Series) -> Dict[str, Any]:
        """
        Convenience method that returns analysis as dictionary.

        Args:
            prices: Series of closing prices

        Returns:
            Dictionary with signal and indicator details
        """
        result = self.evaluate(prices)

        return {
            "signal": result.primary_signal.value,
            "confidence": result.confidence,
            "buy_indicators": result.buy_count,
            "sell_indicators": result.sell_count,
            "hold_indicators": result.hold_count,
            "indicator_signals": [
                {
                    "name": i.name,
                    "signal": i.signal.value,
                    "value": round(i.value, 4),
                    "details": {k: round(v, 4) for k, v in i.details.items()}
                }
                for i in result.indicators
            ],
            "details": {k: round(v, 4) if isinstance(v, float) else v
                       for k, v in result.details.items()}
        }

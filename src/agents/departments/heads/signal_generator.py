"""
Signal Generation Module

Provides RSI and MACD based signal generation with confluence detection.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum

from src.risk.physics.hmm.indicators import TechnicalIndicators


class SignalAction(str, Enum):
    """Trading signal actions."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class SignalConfig:
    """Configuration for signal generation."""
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0

    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    min_confluence: int = 2  # Minimum indicators agreeing for strong signal


class RSISignalGenerator:
    """RSI-based signal generator."""

    def __init__(self, config: SignalConfig):
        self.config = config

    def generate(self, prices: pd.Series) -> Dict[str, Any]:
        """
        Generate RSI-based signal.

        Args:
            prices: Series of close prices

        Returns:
            Dict with signal details
        """
        rsi = TechnicalIndicators.rsi(prices, self.config.rsi_period)

        if rsi < self.config.rsi_oversold:
            action = SignalAction.BUY
            reason = f"RSI oversold ({rsi:.1f} < {self.config.rsi_oversold})"
            confidence = min(1.0, (self.config.rsi_oversold - rsi) / 20)
        elif rsi > self.config.rsi_overbought:
            action = SignalAction.SELL
            reason = f"RSI overbought ({rsi:.1f} > {self.config.rsi_overbought})"
            confidence = min(1.0, (rsi - self.config.rsi_overbought) / 20)
        else:
            action = SignalAction.HOLD
            reason = f"RSI neutral ({rsi:.1f})"
            confidence = 0.3

        return {
            "indicator": "RSI",
            "value": round(rsi, 2),
            "action": action.value,
            "confidence": round(confidence, 2),
            "reason": reason,
        }


class MACDSignalGenerator:
    """MACD-based signal generator."""

    def __init__(self, config: SignalConfig):
        self.config = config

    def generate(self, prices: pd.Series) -> Dict[str, Any]:
        """
        Generate MACD-based signal.

        Args:
            prices: Series of close prices

        Returns:
            Dict with signal details
        """
        macd_line, signal_line, histogram = TechnicalIndicators.macd(
            prices,
            fast=self.config.macd_fast,
            slow=self.config.macd_slow,
            signal=self.config.macd_signal
        )

        # Determine signal based on crossover
        if histogram > 0 and macd_line > signal_line:
            action = SignalAction.BUY
            reason = f"MACD bullish crossover (histogram: {histogram:.5f})"
            confidence = min(1.0, abs(histogram) * 10)
        elif histogram < 0 and macd_line < signal_line:
            action = SignalAction.SELL
            reason = f"MACD bearish crossover (histogram: {histogram:.5f})"
            confidence = min(1.0, abs(histogram) * 10)
        else:
            action = SignalAction.HOLD
            reason = f"MACD neutral (histogram: {histogram:.5f})"
            confidence = 0.3

        return {
            "indicator": "MACD",
            "macd_line": round(macd_line, 5),
            "signal_line": round(signal_line, 5),
            "histogram": round(histogram, 5),
            "action": action.value,
            "confidence": round(confidence, 2),
            "reason": reason,
        }


class ConfluenceSignalGenerator:
    """Combines multiple indicators for confluence signals."""

    def __init__(self, config: SignalConfig):
        self.config = config
        self.rsi_gen = RSISignalGenerator(config)
        self.macd_gen = MACDSignalGenerator(config)

    def generate(self, prices: pd.Series) -> Dict[str, Any]:
        """
        Generate confluence signal from RSI and MACD.

        Args:
            prices: Series of close prices

        Returns:
            Dict with confluence signal details
        """
        rsi_signal = self.rsi_gen.generate(prices)
        macd_signal = self.macd_gen.generate(prices)

        signals = [rsi_signal, macd_signal]

        # Count BUY/SELL signals
        buy_count = sum(1 for s in signals if s["action"] == SignalAction.BUY.value)
        sell_count = sum(1 for s in signals if s["action"] == SignalAction.SELL.value)

        # Determine confluence action
        if buy_count >= self.config.min_confluence:
            action = SignalAction.BUY
            reason = f"Confluence: {buy_count}/{len(signals)} indicators suggest BUY"
        elif sell_count >= self.config.min_confluence:
            action = SignalAction.SELL
            reason = f"Confluence: {sell_count}/{len(signals)} indicators suggest SELL"
        else:
            action = SignalAction.HOLD
            reason = "No clear confluence"

        # Calculate confidence as average of individual confidences
        confidences = [s["confidence"] for s in signals if s["action"] != SignalAction.HOLD.value]
        confidence = np.mean(confidences) if confidences else 0.3

        return {
            "action": action.value,
            "confidence": round(confidence, 2),
            "reason": reason,
            "signals": signals,
            "confluence_count": max(buy_count, sell_count),
            "total_indicators": len(signals),
        }


class SignalGenerator:
    """Main signal generator that combines all strategies."""

    def __init__(self, config: Optional[SignalConfig] = None):
        self.config = config or SignalConfig()
        self.rsi_gen = RSISignalGenerator(self.config)
        self.macd_gen = MACDSignalGenerator(self.config)
        self.confluence_gen = ConfluenceSignalGenerator(self.config)

    def generate_all(self, prices: pd.Series) -> Dict[str, Any]:
        """
        Generate all signal types.

        Args:
            prices: Series of close prices

        Returns:
            Dict with all signal types
        """
        return {
            "rsi": self.rsi_gen.generate(prices),
            "macd": self.macd_gen.generate(prices),
            "confluence": self.confluence_gen.generate(prices),
        }

    def generate(self, prices: pd.Series, strategy: str = "confluence") -> Dict[str, Any]:
        """
        Generate signal using specified strategy.

        Args:
            prices: Series of close prices
            strategy: Signal strategy (rsi, macd, confluence)

        Returns:
            Signal dict
        """
        strategies = {
            "rsi": self.rsi_gen.generate,
            "macd": self.macd_gen.generate,
            "confluence": self.confluence_gen.generate,
        }

        if strategy not in strategies:
            raise ValueError(f"Unknown strategy: {strategy}. Choose from {list(strategies.keys())}")

        return strategies[strategy](prices)

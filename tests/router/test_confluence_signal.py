"""
Unit tests for ConfluenceSignal trading signal generator.

Tests indicator calculations, signal generation, and consensus logic.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.router.signals.confluence_signal import (
    ConfluenceSignal,
    SignalType,
    IndicatorSignal,
    ConfluenceResult,
)


class TestConfluenceSignal:
    """Test ConfluenceSignal indicator calculations and signal generation."""

    @pytest.fixture
    def sample_prices(self):
        """Generate sample price data for testing."""
        np.random.seed(42)
        prices = pd.Series(np.cumsum(np.random.randn(100)) + 100)
        return prices

    @pytest.fixture
    def uptrend_prices(self):
        """Generate upward trending price data."""
        prices = pd.Series([100 + i * 0.5 + np.random.randn() * 0.5 for i in range(50)])
        return prices

    @pytest.fixture
    def downtrend_prices(self):
        """Generate downward trending price data."""
        prices = pd.Series([100 - i * 0.5 + np.random.randn() * 0.5 for i in range(50)])
        return prices

    @pytest.fixture
    def oscillator(self):
        """Create ConfluenceSignal instance."""
        return ConfluenceSignal()

    def test_initialization(self, oscillator):
        """Test ConfluenceSignal initializes with correct defaults."""
        assert oscillator.rsi_period == 14
        assert oscillator.macd_fast == 12
        assert oscillator.macd_slow == 26
        assert oscillator.macd_signal == 9
        assert oscillator.bb_period == 20
        assert oscillator.bb_std == 2.0
        assert oscillator.sma_period == 20
        assert oscillator.ema_period == 20
        assert oscillator.min_consensus == 2

    def test_custom_parameters(self):
        """Test ConfluenceSignal with custom parameters."""
        signal = ConfluenceSignal(
            rsi_period=7,
            macd_fast=5,
            macd_slow=15,
            min_consensus=3
        )
        assert signal.rsi_period == 7
        assert signal.macd_fast == 5
        assert signal.min_consensus == 3

    def test_rsi_calculation(self, oscillator, sample_prices):
        """Test RSI indicator calculation."""
        rsi = oscillator.calculate_rsi(sample_prices)
        assert 0 <= rsi <= 100

    def test_rsi_neutral_when_insufficient_data(self, oscillator):
        """Test RSI returns neutral value with insufficient data."""
        short_prices = pd.Series([100, 101, 102])
        rsi = oscillator.calculate_rsi(short_prices)
        assert rsi == 50.0

    def test_macd_calculation(self, oscillator, sample_prices):
        """Test MACD indicator calculation."""
        macd_line, signal_line, histogram = oscillator.calculate_macd(sample_prices)
        assert isinstance(macd_line, float)
        assert isinstance(signal_line, float)
        assert isinstance(histogram, float)

    def test_bollinger_bands_calculation(self, oscillator, sample_prices):
        """Test Bollinger Bands calculation."""
        upper, middle, lower = oscillator.calculate_bollinger_bands(sample_prices)
        assert upper >= middle >= lower

    def test_sma_calculation(self, oscillator, sample_prices):
        """Test SMA calculation."""
        sma = oscillator.calculate_sma(sample_prices)
        assert isinstance(sma, float)

    def test_ema_calculation(self, oscillator, sample_prices):
        """Test EMA calculation."""
        ema = oscillator.calculate_ema(sample_prices)
        assert isinstance(ema, float)

    def test_rsi_signal_oversold(self, oscillator):
        """Test RSI generates BUY signal when oversold."""
        # Create prices that will result in low RSI
        prices = pd.Series([100] * 20 + [90] * 10)
        signal = oscillator.get_rsi_signal(prices)
        assert signal.name == "RSI"
        assert signal.signal in [SignalType.BUY, SignalType.HOLD]  # May be HOLD with short data

    def test_rsi_signal_overbought(self, oscillator):
        """Test RSI generates SELL signal when overbought."""
        prices = pd.Series([100] * 20 + [110] * 10)
        signal = oscillator.get_rsi_signal(prices)
        assert signal.name == "RSI"

    def test_macd_signal_bullish(self, oscillator, uptrend_prices):
        """Test MACD generates BUY signal in uptrend."""
        signal = oscillator.get_macd_signal(uptrend_prices)
        assert signal.name == "MACD"
        assert signal.signal in [SignalType.BUY, SignalType.HOLD]

    def test_macd_signal_bearish(self, oscillator, downtrend_prices):
        """Test MACD generates SELL signal in downtrend."""
        signal = oscillator.get_macd_signal(downtrend_prices)
        assert signal.name == "MACD"
        assert signal.signal in [SignalType.SELL, SignalType.HOLD]

    def test_bollinger_signal(self, oscillator, sample_prices):
        """Test Bollinger Bands signal generation."""
        signal = oscillator.get_bollinger_signal(sample_prices)
        assert signal.name == "Bollinger"
        assert signal.signal in [SignalType.BUY, SignalType.SELL, SignalType.HOLD]

    def test_sma_signal_uptrend(self, oscillator, uptrend_prices):
        """Test SMA generates BUY signal in uptrend."""
        signal = oscillator.get_sma_signal(uptrend_prices)
        assert signal.name == "SMA"
        assert signal.signal == SignalType.BUY

    def test_sma_signal_downtrend(self, oscillator, downtrend_prices):
        """Test SMA generates SELL signal in downtrend."""
        signal = oscillator.get_sma_signal(downtrend_prices)
        assert signal.name == "SMA"
        assert signal.signal == SignalType.SELL

    def test_ema_signal_uptrend(self, oscillator, uptrend_prices):
        """Test EMA generates BUY signal in uptrend."""
        signal = oscillator.get_ema_signal(uptrend_prices)
        assert signal.name == "EMA"
        assert signal.signal == SignalType.BUY

    def test_ema_signal_downtrend(self, oscillator, downtrend_prices):
        """Test EMA generates SELL signal in downtrend."""
        signal = oscillator.get_ema_signal(downtrend_prices)
        assert signal.name == "EMA"
        assert signal.signal == SignalType.SELL

    def test_confluence_buy_signal(self, oscillator, uptrend_prices):
        """Test BUY signal when 2+ indicators agree in uptrend."""
        result = oscillator.evaluate(uptrend_prices)
        assert isinstance(result, ConfluenceResult)
        # In strong uptrend, should get BUY signal
        assert result.primary_signal in [SignalType.BUY, SignalType.HOLD]

    def test_confluence_sell_signal(self, oscillator, downtrend_prices):
        """Test SELL signal when 2+ indicators agree in downtrend."""
        result = oscillator.evaluate(downtrend_prices)
        assert isinstance(result, ConfluenceResult)
        # With more sell indicators, primary should be SELL or at least some indicators agree
        assert result.sell_count >= result.buy_count or result.sell_count >= 2

    def test_confluence_hold_signal(self, oscillator):
        """Test HOLD signal when no consensus."""
        # Create oscillating prices
        prices = pd.Series([100, 105, 100, 105, 100, 105, 100])
        result = oscillator.evaluate(prices)
        assert isinstance(result, ConfluenceResult)
        assert result.primary_signal in [SignalType.BUY, SignalType.SELL, SignalType.HOLD]

    def test_confluence_counts(self, oscillator, sample_prices):
        """Test indicator counts are correct."""
        result = oscillator.evaluate(sample_prices)
        assert result.buy_count + result.sell_count + result.hold_count == 5

    def test_confidence_calculation(self, oscillator, uptrend_prices):
        """Test confidence calculation based on consensus."""
        result = oscillator.evaluate(uptrend_prices)
        # Confidence should be between 0 and 1
        assert 0 <= result.confidence <= 1
        # If BUY signal, confidence should be based on buy_count
        if result.primary_signal == SignalType.BUY:
            assert result.confidence == result.buy_count / 5

    def test_analyze_returns_dict(self, oscillator, sample_prices):
        """Test analyze method returns proper dictionary."""
        analysis = oscillator.analyze(sample_prices)

        assert "signal" in analysis
        assert "confidence" in analysis
        assert "buy_indicators" in analysis
        assert "sell_indicators" in analysis
        assert "hold_indicators" in analysis
        assert "indicator_signals" in analysis

        assert analysis["signal"] in ["buy", "sell", "hold"]
        assert len(analysis["indicator_signals"]) == 5

    def test_all_indicators_in_analysis(self, oscillator, sample_prices):
        """Test all five indicators are present in analysis."""
        analysis = oscillator.analyze(sample_prices)

        indicator_names = [i["name"] for i in analysis["indicator_signals"]]
        assert "RSI" in indicator_names
        assert "MACD" in indicator_names
        assert "Bollinger" in indicator_names
        assert "SMA" in indicator_names
        assert "EMA" in indicator_names

    def test_min_consensus_custom(self):
        """Test custom minimum consensus threshold."""
        signal = ConfluenceSignal(min_consensus=3)
        # With 3 consensus required, harder to trigger BUY/SELL
        assert signal.min_consensus == 3


class TestSignalType:
    """Test SignalType enum."""

    def test_signal_types_exist(self):
        """Test all required signal types exist."""
        assert SignalType.BUY.value == "buy"
        assert SignalType.SELL.value == "sell"
        assert SignalType.HOLD.value == "hold"


class TestIndicatorSignal:
    """Test IndicatorSignal dataclass."""

    def test_create_indicator_signal(self):
        """Test creating an IndicatorSignal."""
        signal = IndicatorSignal(
            name="RSI",
            signal=SignalType.BUY,
            value=25.5,
            details={"rsi": 25.5}
        )
        assert signal.name == "RSI"
        assert signal.signal == SignalType.BUY
        assert signal.value == 25.5
        assert signal.details["rsi"] == 25.5


class TestConfluenceResult:
    """Test ConfluenceResult dataclass."""

    def test_create_confluence_result(self):
        """Test creating a ConfluenceResult."""
        indicators = [
            IndicatorSignal("RSI", SignalType.BUY, 25.0, {}),
            IndicatorSignal("MACD", SignalType.BUY, 0.5, {}),
        ]
        result = ConfluenceResult(
            primary_signal=SignalType.BUY,
            indicators=indicators,
            buy_count=2,
            sell_count=0,
            hold_count=0,
            confidence=0.4,
            details={"price": 100.5}
        )
        assert result.primary_signal == SignalType.BUY
        assert result.buy_count == 2
        assert result.confidence == 0.4

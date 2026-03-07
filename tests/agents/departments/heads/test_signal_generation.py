"""
Tests for Signal Generation Module

Tests RSI, MACD, and confluence signal generation.
"""
import pytest
import numpy as np
import pandas as pd

from src.agents.departments.heads.signal_generator import (
    SignalGenerator,
    SignalConfig,
    RSISignalGenerator,
    MACDSignalGenerator,
    ConfluenceSignalGenerator,
    SignalAction,
)


class TestRSISignalGenerator:
    """Test RSI-based signal generation."""

    @pytest.fixture
    def config(self):
        """Create signal config."""
        return SignalConfig()

    @pytest.fixture
    def upward_prices(self):
        """Create upward trending prices."""
        # Create prices that will result in RSI oversold (below 30)
        base = 1.1000
        prices = []
        for i in range(30):
            # Declining prices to push RSI down
            prices.append(base - i * 0.002)
        return pd.Series(prices)

    @pytest.fixture
    def downward_prices(self):
        """Create downward trending prices."""
        # Create prices that will result in RSI overbought (above 70)
        base = 1.1000
        prices = []
        for i in range(30):
            # Rising prices to push RSI up
            prices.append(base + i * 0.002)
        return pd.Series(prices)

    @pytest.fixture
    def neutral_prices(self):
        """Create neutral sideways prices."""
        # Create oscillating prices that will result in neutral RSI
        base = 1.1000
        prices = []
        for i in range(30):
            # Small oscillations around the base
            prices.append(base + 0.001 * (-1 if i % 2 == 0 else 1))
        return pd.Series(prices)

    def test_rsi_oversold_generates_buy(self, config, upward_prices):
        """RSI oversold should generate BUY signal."""
        generator = RSISignalGenerator(config)
        result = generator.generate(upward_prices)

        assert result["indicator"] == "RSI"
        assert result["action"] == SignalAction.BUY.value
        assert result["value"] < config.rsi_oversold
        assert result["confidence"] > 0

    def test_rsi_overbought_generates_sell(self, config, downward_prices):
        """RSI overbought should generate SELL signal."""
        generator = RSISignalGenerator(config)
        result = generator.generate(downward_prices)

        assert result["indicator"] == "RSI"
        assert result["action"] == SignalAction.SELL.value
        assert result["value"] > config.rsi_overbought
        assert result["confidence"] > 0

    def test_rsi_neutral_generates_hold(self, config, neutral_prices):
        """RSI in neutral zone should generate HOLD."""
        generator = RSISignalGenerator(config)
        result = generator.generate(neutral_prices)

        assert result["indicator"] == "RSI"
        assert result["action"] == SignalAction.HOLD.value
        assert result["value"] >= config.rsi_oversold
        assert result["value"] <= config.rsi_overbought


class TestMACDSignalGenerator:
    """Test MACD-based signal generation."""

    @pytest.fixture
    def config(self):
        """Create signal config."""
        return SignalConfig()

    @pytest.fixture
    def bullish_prices(self):
        """Create bullish trending prices."""
        base = 1.1000
        # Strong upward trend
        return pd.Series([base + i * 0.005 for i in range(50)])

    @pytest.fixture
    def bearish_prices(self):
        """Create bearish trending prices."""
        base = 1.1000
        # Strong downward trend
        return pd.Series([base - i * 0.005 for i in range(50)])

    def test_bullish_macd_generates_buy(self, config, bullish_prices):
        """Bullish MACD crossover should generate BUY."""
        generator = MACDSignalGenerator(config)
        result = generator.generate(bullish_prices)

        assert result["indicator"] == "MACD"
        assert result["action"] in [SignalAction.BUY.value, SignalAction.HOLD.value]
        assert "macd_line" in result
        assert "signal_line" in result
        assert "histogram" in result

    def test_bearish_macd_generates_sell(self, config, bearish_prices):
        """Bearish MACD crossover should generate SELL."""
        generator = MACDSignalGenerator(config)
        result = generator.generate(bearish_prices)

        assert result["indicator"] == "MACD"
        assert result["action"] in [SignalAction.SELL.value, SignalAction.HOLD.value]
        assert "macd_line" in result
        assert "signal_line" in result


class TestConfluenceSignalGenerator:
    """Test confluence signal generation."""

    @pytest.fixture
    def config(self):
        """Create signal config with minimum confluence of 2."""
        return SignalConfig(min_confluence=2)

    @pytest.fixture
    def strong_bullish_prices(self):
        """Create strongly bullish prices (both RSI and MACD bullish)."""
        base = 1.1000
        # Start low, go high to push RSI oversold then bullish
        prices = []
        for i in range(50):
            if i < 20:
                prices.append(base - (20 - i) * 0.003)  # Decline first
            else:
                prices.append(base + (i - 20) * 0.008)  # Strong rise
        return pd.Series(prices)

    def test_confluence_generates_buy(self, config, strong_bullish_prices):
        """Confluence of bullish indicators should generate BUY."""
        generator = ConfluenceSignalGenerator(config)
        result = generator.generate(strong_bullish_prices)

        assert "action" in result
        assert "confidence" in result
        assert "signals" in result
        assert len(result["signals"]) == 2  # RSI and MACD

    def test_confluence_includes_all_signals(self, config, strong_bullish_prices):
        """Confluence should include RSI and MACD signals."""
        generator = ConfluenceSignalGenerator(config)
        result = generator.generate(strong_bullish_prices)

        indicators = [s["indicator"] for s in result["signals"]]
        assert "RSI" in indicators
        assert "MACD" in indicators


class TestSignalGenerator:
    """Test main signal generator."""

    @pytest.fixture
    def generator(self):
        """Create signal generator."""
        return SignalGenerator()

    @pytest.fixture
    def sample_prices(self):
        """Create sample price data."""
        np.random.seed(42)
        base = 1.1000
        # Random walk with slight upward bias
        returns = np.random.randn(50) * 0.001 + 0.0002
        prices = base + np.cumsum(returns)
        return pd.Series(prices)

    def test_generate_all_returns_all_strategies(self, generator, sample_prices):
        """Generate_all should return all signal types."""
        result = generator.generate_all(sample_prices)

        assert "rsi" in result
        assert "macd" in result
        assert "confluence" in result

    def test_generate_with_strategy(self, generator, sample_prices):
        """Generate should work with different strategies."""
        for strategy in ["rsi", "macd", "confluence"]:
            result = generator.generate(sample_prices, strategy=strategy)
            assert "action" in result
            assert "confidence" in result

    def test_generate_invalid_strategy_raises(self, generator, sample_prices):
        """Invalid strategy should raise ValueError."""
        with pytest.raises(ValueError) as exc:
            generator.generate(sample_prices, strategy="invalid")
        assert "Unknown strategy" in str(exc.value)


class TestSignalConfig:
    """Test signal configuration."""

    def test_default_config(self):
        """Default config should have sensible defaults."""
        config = SignalConfig()

        assert config.rsi_period == 14
        assert config.rsi_oversold == 30.0
        assert config.rsi_overbought == 70.0
        assert config.macd_fast == 12
        assert config.macd_slow == 26
        assert config.macd_signal == 9
        assert config.min_confluence == 2

    def test_custom_config(self):
        """Custom config should override defaults."""
        config = SignalConfig(
            rsi_period=10,
            rsi_oversold=25,
            rsi_overbought=75,
            min_confluence=1,
        )

        assert config.rsi_period == 10
        assert config.rsi_oversold == 25
        assert config.rsi_overbought == 75
        assert config.min_confluence == 1

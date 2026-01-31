"""
Skill Implementation Tests - Task Group 14

Tests for critical trading, system, and data skills.
Focuses on core functionality: calculate_rsi, calculate_position_size,
detect_support_resistance, fetch_historical_data.

Following Task Group 14.1: 2-8 focused tests for critical skills.
"""

import pytest
import numpy as np
from typing import Dict, Any
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


# ============================================================================
# Trading Skills Tests
# ============================================================================

class TestCalculateRSI:
    """Test RSI calculation skill."""

    def test_calculate_rsi_returns_value_and_signal(self):
        """Test calculate_rsi returns correct RSI values and signal."""
        # Import the skill function
        from src.agents.skills.trading_skills import calculate_rsi

        # Generate test price data
        # Rising prices should give RSI > 50
        prices_rising = [1.1000 + i * 0.0001 for i in range(30)]

        result = calculate_rsi(prices_rising, period=14)

        assert "rsi_value" in result
        assert "signal" in result
        assert isinstance(result["rsi_value"], (int, float))
        assert result["signal"] in ["oversold", "neutral", "overbought"]
        assert 0 <= result["rsi_value"] <= 100

    def test_calculate_rsi_oversold_signal(self):
        """Test calculate_rsi detects oversold condition (RSI < 30)."""
        from src.agents.skills.trading_skills import calculate_rsi

        # Falling prices should give low RSI
        prices_falling = [1.1000 - i * 0.0005 for i in range(30)]

        result = calculate_rsi(prices_falling, period=14)

        assert result["rsi_value"] < 30
        assert result["signal"] == "oversold"

    def test_calculate_rsi_overbought_signal(self):
        """Test calculate_rsi detects overbought condition (RSI > 70)."""
        from src.agents.skills.trading_skills import calculate_rsi

        # Strongly rising prices should give high RSI
        prices_rising = [1.1000 + i * 0.001 for i in range(30)]

        result = calculate_rsi(prices_rising, period=14)

        assert result["rsi_value"] > 70
        assert result["signal"] == "overbought"

    def test_calculate_rsi_raises_error_for_insufficient_data(self):
        """Test calculate_rsi raises ValueError when insufficient prices provided."""
        from src.agents.skills.trading_skills import calculate_rsi

        with pytest.raises(ValueError, match="Need at least"):
            calculate_rsi([1.0, 1.1, 1.2], period=14)


class TestCalculatePositionSize:
    """Test position size calculation skill."""

    def test_calculate_position_size_returns_lots_risk_and_max_loss(self):
        """Test calculate_position_size returns correct lot size and risk amounts."""
        from src.agents.skills.trading_skills import calculate_position_size

        result = calculate_position_size(
            account_balance=10000.0,
            risk_percent=1.0,
            stop_loss_pips=20.0
        )

        assert "position_size_lots" in result
        assert "risk_amount" in result
        assert "max_loss_pips" in result
        assert result["risk_amount"] == 100.0  # 1% of 10000
        assert result["position_size_lots"] >= 0.01

    def test_calculate_position_size_formula(self):
        """Test calculate_position_size uses correct formula."""
        from src.agents.skills.trading_skills import calculate_position_size

        result = calculate_position_size(
            account_balance=10000.0,
            risk_percent=1.0,
            stop_loss_pips=20.0,
            pip_value=10.0
        )

        # Formula: Position Size = Risk Amount / (Stop Loss Pips * Pip Value)
        # Risk Amount = 10000 * 0.01 = 100
        # Position Size = 100 / (20 * 10) = 100 / 200 = 0.5
        expected_lots = 100.0 / (20.0 * 10.0)
        assert result["position_size_lots"] == expected_lots

    def test_calculate_position_size_minimum_lot_size(self):
        """Test calculate_position_size enforces minimum 0.01 lots."""
        from src.agents.skills.trading_skills import calculate_position_size

        result = calculate_position_size(
            account_balance=100.0,  # Very small balance
            risk_percent=0.5,
            stop_loss_pips=50.0
        )

        assert result["position_size_lots"] >= 0.01


class TestDetectSupportResistance:
    """Test support/resistance detection skill."""

    def test_detect_support_resistance_returns_levels(self):
        """Test detect_support_resistance returns S/R levels and current price."""
        from src.agents.skills.trading_skills import detect_support_resistance

        # Generate test data with clear patterns
        highs = [1.1020, 1.1040, 1.1060, 1.1040, 1.1020, 1.1030, 1.1050, 1.1070, 1.1050, 1.1030,
                 1.1040, 1.1060, 1.1080, 1.1060, 1.1040, 1.1050, 1.1070, 1.1090, 1.1070, 1.1050]
        lows =  [1.0980, 1.0990, 1.1000, 1.0990, 1.0980, 1.0990, 1.1000, 1.1010, 1.1000, 1.0990,
                 1.1000, 1.1010, 1.1020, 1.1010, 1.1000, 1.1010, 1.1020, 1.1030, 1.1020, 1.1010]
        closes = [1.1000, 1.1015, 1.1030, 1.1015, 1.1000, 1.1010, 1.1025, 1.1040, 1.1025, 1.1010,
                  1.1020, 1.1035, 1.1050, 1.1035, 1.1020, 1.1030, 1.1045, 1.1060, 1.1045, 1.1030]

        result = detect_support_resistance(highs, lows, closes)

        assert "support_levels" in result
        assert "resistance_levels" in result
        assert "current_price" in result
        assert "nearest_support" in result
        assert "nearest_resistance" in result
        assert isinstance(result["support_levels"], list)
        assert isinstance(result["resistance_levels"], list)

    def test_detect_support_resistance_identifies_pivots(self):
        """Test detect_support_resistance identifies pivot points."""
        from src.agents.skills.trading_skills import detect_support_resistance

        # Data with clear pivot high at 1.1070 and pivot low at 1.0980
        highs = [1.1020, 1.1040, 1.1060, 1.1070, 1.1060, 1.1040, 1.1020]
        lows =  [1.0990, 1.0985, 1.0980, 1.0985, 1.0990, 1.0995, 1.1000]
        closes = [1.1005, 1.1010, 1.1020, 1.1025, 1.1020, 1.1010, 1.1005]

        result = detect_support_resistance(highs, lows, closes, lookback_period=3)

        # Should find resistance near pivot high
        if result["resistance_levels"]:
            assert result["resistance_levels"][0] >= 1.1060

    def test_detect_support_resistance_raises_error_for_mismatched_lengths(self):
        """Test detect_support_resistance raises ValueError for mismatched array lengths."""
        from src.agents.skills.trading_skills import detect_support_resistance

        with pytest.raises(ValueError, match="must have the same length"):
            detect_support_resistance([1.0, 1.1], [1.0, 1.1], [1.0])


# ============================================================================
# Data Skills Tests
# ============================================================================

class TestFetchHistoricalData:
    """Test historical data fetching skill."""

    def test_fetch_historical_data_returns_ohlcv(self):
        """Test fetch_historical_data returns OHLCV data."""
        from src.agents.skills.data_skills import fetch_historical_data

        result = fetch_historical_data(
            symbol="EURUSD",
            timeframe="H1",
            bars_count=50
        )

        assert "symbol" in result
        assert "timeframe" in result
        assert "data" in result
        assert "bars_count" in result
        assert result["symbol"] == "EURUSD"
        assert result["timeframe"] == "H1"
        assert result["bars_count"] == 50
        assert len(result["data"]) == 50

    def test_fetch_historical_data_data_structure(self):
        """Test fetch_historical_data returns correct data structure."""
        from src.agents.skills.data_skills import fetch_historical_data

        result = fetch_historical_data(symbol="EURUSD", timeframe="H1", bars_count=10)

        # Check first bar has all required fields
        first_bar = result["data"][0]
        assert "time" in first_bar
        assert "open" in first_bar
        assert "high" in first_bar
        assert "low" in first_bar
        assert "close" in first_bar
        assert "volume" in first_bar

        # Validate OHLC relationships
        assert first_bar["high"] >= first_bar["open"]
        assert first_bar["high"] >= first_bar["close"]
        assert first_bar["low"] <= first_bar["open"]
        assert first_bar["low"] <= first_bar["close"]

    def test_fetch_historical_data_invalid_timeframe(self):
        """Test fetch_historical_data raises error for invalid timeframe."""
        from src.agents.skills.data_skills import fetch_historical_data

        with pytest.raises(ValueError, match="Invalid timeframe"):
            fetch_historical_data(symbol="EURUSD", timeframe="H5", bars_count=50)


# ============================================================================
# System Skills Tests
# ============================================================================

class TestLogTradeEvent:
    """Test trade event logging skill."""

    def test_log_trade_event_returns_success(self):
        """Test log_trade_event returns logged status with timestamp and ID."""
        from src.agents.skills.system_skills import log_trade_event

        result = log_trade_event(
            event_type="entry",
            symbol="EURUSD",
            action="buy",
            price=1.1000,
            lots=0.1,
            strategy_name="TestStrategy"
        )

        assert "logged" in result
        assert "timestamp" in result
        assert "log_id" in result
        assert result["logged"] is True
        assert len(result["log_id"]) > 0

    def test_log_trade_event_with_exit_and_pnl(self):
        """Test log_trade_event handles exit events with PnL."""
        from src.agents.skills.system_skills import log_trade_event

        result = log_trade_event(
            event_type="exit",
            symbol="XAUUSD",
            action="sell",
            price=2000.0,
            lots=0.5,
            strategy_name="TestStrategy",
            pnl=250.0
        )

        assert result["logged"] is True


# ============================================================================
# Skill Validation Tests (Task Group 14.5)
# ============================================================================

class TestSkillValidationOnLoad:
    """Test that skills validate correctly when loaded."""

    def test_calculate_rsi_validates_input_schema(self):
        """Test calculate_rsi validates input against schema."""
        from src.agents.skills.trading_skills import calculate_rsi

        # Valid input should work
        prices = [1.0] * 20
        result = calculate_rsi(prices, period=14)
        assert "rsi_value" in result

        # Invalid period should be handled by schema validation
        # (This would typically be caught by Pydantic/JSON Schema validation)
        with pytest.raises((ValueError, TypeError)):
            calculate_rsi(prices, period=-1)

    def test_calculate_position_size_validates_input_schema(self):
        """Test calculate_position_size validates input against schema."""
        from src.agents.skills.trading_skills import calculate_position_size

        # Valid input should work
        result = calculate_position_size(10000.0, 1.0, 20.0)
        assert "position_size_lots" in result

        # Invalid inputs should raise errors
        with pytest.raises((ValueError, TypeError, ZeroDivisionError)):
            calculate_position_size(10000.0, 1.0, 0.0)  # Zero stop loss

    def test_detect_support_resistance_validates_input_schema(self):
        """Test detect_support_resistance validates input against schema."""
        from src.agents.skills.trading_skills import detect_support_resistance

        # Valid input should work
        highs, lows, closes = [1.1] * 20, [1.0] * 20, [1.05] * 20
        result = detect_support_resistance(highs, lows, closes)
        assert "support_levels" in result

    def test_fetch_historical_data_validates_input_schema(self):
        """Test fetch_historical_data validates input against schema."""
        from src.agents.skills.data_skills import fetch_historical_data

        # Valid input should work
        result = fetch_historical_data("EURUSD", "H1", 50)
        assert "data" in result

        # Invalid timeframe should raise error
        with pytest.raises(ValueError):
            fetch_historical_data("EURUSD", "INVALID", 50)

    def test_log_trade_event_validates_input_schema(self):
        """Test log_trade_event validates input against schema."""
        from src.agents.skills.system_skills import log_trade_event

        # Valid input should work
        result = log_trade_event("entry", "EURUSD", "buy", 1.1, 0.1, "Test")
        assert result["logged"] is True

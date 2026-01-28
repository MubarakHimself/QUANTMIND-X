"""
Tests for Kelly Statistics Analyzer.

Tests cover:
- Kelly parameter extraction from trade history
- Rolling window analysis
- Edge decay detection
- Edge case handling
- Insufficient data handling
"""

import pytest
import numpy as np
from datetime import datetime

from src.position_sizing.kelly_analyzer import (
    KellyStatisticsAnalyzer,
    KellyParameters
)


@pytest.mark.analyzer
class TestKellyParameterExtraction:
    """Test suite for Kelly parameter extraction."""

    def test_profitable_strategy_extraction(self, kelly_analyzer, winning_trade_history):
        """Test parameter extraction from profitable strategy."""
        params = kelly_analyzer.calculate_kelly_parameters(winning_trade_history)

        # Win rate should be around 60%
        assert 0.55 <= params.win_rate <= 0.65
        assert params.avg_win > 0
        assert params.avg_loss > 0
        assert params.risk_reward_ratio > 1.0
        assert params.base_kelly_f > 0
        assert params.sample_size == 50
        assert params.is_reliable  # 50 trades >= min_trades (30)

    def test_losing_strategy_extraction(self, kelly_analyzer, losing_trade_history):
        """Test parameter extraction from losing strategy."""
        params = kelly_analyzer.calculate_kelly_parameters(losing_trade_history)

        # Win rate should be around 40%
        assert 0.35 <= params.win_rate <= 0.45
        assert params.base_kelly_f < 0  # Negative expectancy
        assert params.expectancy < 0
        assert params.sample_size == 50

    def test_risk_reward_ratio_calculation(self, kelly_analyzer):
        """Test risk-reward ratio calculation."""
        trades = [
            {"profit": 300.0},   # Win
            {"profit": -100.0},  # Loss (3:1 R:R)
            {"profit": 300.0},
            {"profit": -100.0},
            {"profit": 300.0},
            {"profit": -100.0},
        ]

        params = kelly_analyzer.calculate_kelly_parameters(trades)

        # R:R = 300/100 = 3
        assert params.risk_reward_ratio == 3.0

    def test_expectancy_calculation(self, kelly_analyzer):
        """Test expectancy (average profit per trade)."""
        trades = [
            {"profit": 500.0},   # Win
            {"profit": -200.0},  # Loss
            {"profit": 600.0},
            {"profit": -150.0},
        ]

        params = kelly_analyzer.calculate_kelly_parameters(trades)

        # Expectancy = (500 + 600 - 200 - 150) / 4 = 187.5
        assert abs(params.expectancy - 187.5) < 1.0

    def test_profit_factor_calculation(self, kelly_analyzer):
        """Test profit factor (total wins / total losses)."""
        trades = [
            {"profit": 400.0},   # Total wins: 400 + 600 = 1000
            {"profit": -200.0},  # Total losses: 200 + 150 = 350
            {"profit": 600.0},
            {"profit": -150.0},
        ]

        params = kelly_analyzer.calculate_kelly_parameters(trades)

        # Profit factor = 1000 / 350 = 2.86
        assert abs(params.profit_factor - 2.86) < 0.1

    def test_base_kelly_formula(self, kelly_analyzer):
        """Test base Kelly formula: f = ((B+1)*P - 1) / B"""
        trades = []
        # Create strategy with 55% win rate, 2:1 R:R
        for _ in range(55):
            trades.append({"profit": 200.0})  # Win
        for _ in range(45):
            trades.append({"profit": -100.0})  # Loss

        params = kelly_analyzer.calculate_kelly_parameters(trades)

        # B = 2, P = 0.55
        # f = ((2+1)*0.55 - 1) / 2 = 0.325
        expected_kelly = ((2 + 1) * 0.55 - 1) / 2
        assert abs(params.base_kelly_f - expected_kelly) < 0.05


@pytest.mark.analyzer
class TestDataReliability:
    """Test suite for data reliability assessment."""

    def test_sufficient_sample_size(self, kelly_analyzer, sample_trade_history):
        """Test with sufficient sample size (> 30 trades)."""
        params = kelly_analyzer.calculate_kelly_parameters(sample_trade_history)
    
        assert params.sample_size == len(sample_trade_history)
        assert params.is_reliable is True
        assert "HIGH" in params.confidence_note or "MODERATE" in params.confidence_note

    def test_insufficient_sample_size_low(self, kelly_analyzer, insufficient_trade_history):
        """Test with insufficient sample size (< 10 trades)."""
        params = kelly_analyzer.calculate_kelly_parameters(insufficient_trade_history)

        assert params.sample_size == 8
        assert params.is_reliable is False
        assert "VERY LOW" in params.confidence_note

    def test_insufficient_sample_size_moderate(self, kelly_analyzer):
        """Test with moderate sample size (10-29 trades)."""
        trades = [{"profit": 100.0 if i % 2 == 0 else -50.0} for i in range(20)]

        params = kelly_analyzer.calculate_kelly_parameters(trades)

        assert params.sample_size == 20
        assert params.is_reliable is False
        assert "LOW" in params.confidence_note

    def test_empty_trade_history(self, kelly_analyzer, empty_trade_history):
        """Test with empty trade history."""
        params = kelly_analyzer.calculate_kelly_parameters(empty_trade_history)

        assert params.sample_size == 0
        assert params.is_reliable is False
        assert params.win_rate == 0.0
        assert params.avg_win == 0.0
        assert params.avg_loss == 0.0
        assert "NO DATA" in params.confidence_note


@pytest.mark.analyzer
class TestAlternativeTradeFormats:
    """Test suite for alternative trade data formats."""

    def test_profit_format(self, kelly_analyzer):
        """Test trades with 'profit' field."""
        trades = [
            {"profit": 500.0},
            {"profit": -200.0},
            {"profit": 600.0},
        ]

        params = kelly_analyzer.calculate_kelly_parameters(trades)

        assert params.sample_size == 3
        assert params.win_rate == 2/3

    def test_pnl_format(self, kelly_analyzer):
        """Test trades with 'pnl' field."""
        trades = [
            {"pnl": 500.0},
            {"pnl": -200.0},
            {"pnl": 600.0},
        ]

        params = kelly_analyzer.calculate_kelly_parameters(trades)

        assert params.sample_size == 3

    def test_result_amount_format(self, kelly_analyzer):
        """Test trades with 'result' and 'amount' fields."""
        trades = [
            {"result": "win", "amount": 500.0},
            {"result": "won", "amount": 200.0},
            {"result": "won", "amount": 600.0},
            {"result": "profit", "amount": 400.0},
        ]

        params = kelly_analyzer.calculate_kelly_parameters(trades)

        assert params.sample_size == 4
        assert params.win_rate == 1.0  # All marked as wins

    def test_mixed_formats(self, kelly_analyzer):
        """Test mixed trade formats."""
        trades = [
            {"profit": 500.0},
            {"pnl": -200.0},
            {"result": "win", "amount": 600.0},
        ]

        params = kelly_analyzer.calculate_kelly_parameters(trades)

        assert params.sample_size == 3


@pytest.mark.analyzer
class TestRollingWindowAnalysis:
    """Test suite for rolling window Kelly analysis."""

    def test_rolling_kelly_calculation(self, kelly_analyzer, sample_trade_history):
        """Test rolling window Kelly calculation."""
        window_size = 20
        rolling = kelly_analyzer.calculate_rolling_kelly(
            sample_trade_history,
            window_size=window_size
        )

        # Should have n - window_size + 1 windows
        expected_len = len(sample_trade_history) - window_size + 1
        assert len(rolling) == expected_len

        # Each window should have window_size trades
        for params in rolling:
            assert params.sample_size == window_size

    def test_rolling_kelly_with_small_history(self, kelly_analyzer):
        """Test rolling Kelly with history smaller than window size."""
        trades = [{"profit": 100.0 if i % 2 == 0 else -50.0} for i in range(15)]

        rolling = kelly_analyzer.calculate_rolling_kelly(trades, window_size=20)

        # Should return empty list if history < window_size
        assert len(rolling) == 0

    def test_rolling_kelly_edge_decay_detection(self, kelly_analyzer):
        """Test edge decay detection using rolling windows."""
        # Create decaying performance
        trades = []
        # First 50 trades: 60% win rate
        for i in range(50):
            trades.append({"profit": 100.0 if i % 10 < 6 else -50.0})
        # Next 30 trades: 40% win rate (decay)
        for i in range(30):
            trades.append({"profit": 100.0 if i % 10 < 4 else -50.0})

        rolling = kelly_analyzer.calculate_rolling_kelly(trades, window_size=40)
        decay_analysis = kelly_analyzer.detect_edge_decay(rolling, decay_threshold=0.10)

        # Should detect decay
        assert decay_analysis['status'] == 'analyzed'
        # Recent win rate should be lower than historical
        assert decay_analysis['recent_win_rate'] < decay_analysis['historical_win_rate']
        assert decay_analysis['win_rate_change_pct'] < -5.0  # At least 5% drop


@pytest.mark.edge_case
class TestAnalyzerEdgeCases:
    """Test suite for analyzer edge cases."""

    def test_all_wins_no_losses(self, kelly_analyzer):
        """Test with all winning trades (no losses)."""
        trades = [{"profit": 100.0} for _ in range(30)]

        params = kelly_analyzer.calculate_kelly_parameters(trades)

        # Should handle gracefully
        assert params.win_rate == 1.0
        assert params.avg_loss == 0.0
        assert params.risk_reward_ratio == 0.0  # Division by zero
        assert params.profit_factor == float('inf')

    def test_all_losses_no_wins(self, kelly_analyzer):
        """Test with all losing trades (no wins)."""
        trades = [{"profit": -100.0} for _ in range(30)]

        params = kelly_analyzer.calculate_kelly_parameters(trades)

        # Should handle gracefully
        assert params.win_rate == 0.0
        assert params.avg_win == 0.0
        assert params.base_kelly_f < 0

    def test_zero_profit_trades(self, kelly_analyzer):
        """Test with zero-profit trades."""
        trades = [
            {"profit": 100.0},
            {"profit": 0.0},  # Break even
            {"profit": -50.0},
            {"profit": 0.0},  # Break even
        ]

        params = kelly_analyzer.calculate_kelly_parameters(trades)

        # Zero-profit trades should be counted as wins (profit >= 0)
        assert params.sample_size == 4
        assert params.win_rate == 0.75  # 3 out of 4

    def test_very_large_win(self, kelly_analyzer):
        """Test with outlier large win."""
        trades = [
            {"profit": 100.0} for _ in range(29)
        ] + [
            {"profit": 10000.0}  # Outlier
        ] + [
            {"profit": -50.0} for _ in range(20)
        ]

        params = kelly_analyzer.calculate_kelly_parameters(trades)

        # Should calculate Kelly correctly despite outlier
        assert params.sample_size == 50
        assert params.avg_win > 100  # Should be influenced by outlier
        assert params.base_kelly_f > 0

    def test_very_large_loss(self, kelly_analyzer):
        """Test with outlier large loss."""
        trades = [
            {"profit": 100.0} for _ in range(30)
        ] + [
            {"profit": -5000.0}  # Outlier loss
        ] + [
            {"profit": -50.0} for _ in range(19)
        ]

        params = kelly_analyzer.calculate_kelly_parameters(trades)

        # Should calculate Kelly correctly despite outlier
        assert params.sample_size == 50
        assert params.avg_loss > 50  # Should be influenced by outlier


@pytest.mark.integration
class TestAnalyzerIntegration:
    """Integration tests for Kelly analyzer."""

    def test_realistic_strategy_analysis(self, kelly_analyzer, sample_trade_history):
        """Test analysis of realistic trading strategy."""
        params = kelly_analyzer.calculate_kelly_parameters(sample_trade_history)

        # All parameters should be calculated
        assert params.win_rate > 0
        assert params.avg_win > 0
        assert params.avg_loss > 0
        assert params.risk_reward_ratio > 0
        assert params.expectancy != 0
        assert params.profit_factor > 0
        assert params.is_reliable is True

    def test_confidence_levels(self, kelly_analyzer):
        """Test confidence level assessment."""
        # Very low confidence (< 10 trades)
        very_low = kelly_analyzer.calculate_kelly_parameters(
            [{"profit": 100.0 if i % 2 == 0 else -50.0} for i in range(8)]
        )
        assert "VERY LOW" in very_low.confidence_note

        # Low confidence (10-29 trades)
        low = kelly_analyzer.calculate_kelly_parameters(
            [{"profit": 100.0 if i % 2 == 0 else -50.0} for i in range(20)]
        )
        assert "LOW" in low.confidence_note

        # Moderate confidence (30-99 trades)
        moderate = kelly_analyzer.calculate_kelly_parameters(
            [{"profit": 100.0 if i % 2 == 0 else -50.0} for i in range(50)]
        )
        assert "MODERATE" in moderate.confidence_note

        # High confidence (100+ trades)
        high = kelly_analyzer.calculate_kelly_parameters(
            [{"profit": 100.0 if i % 2 == 0 else -50.0} for i in range(150)]
        )
        assert "HIGH" in high.confidence_note

"""
Tests for Portfolio Kelly Scaler.

Tests cover:
- Multi-bot position scaling
- Portfolio risk management
- Correlation adjustment
- Equal and performance-based allocation
- Edge case handling
"""

import pytest

from src.position_sizing.portfolio_kelly import (
    PortfolioKellyScaler,
    PortfolioStatus
)


@pytest.mark.portfolio
class TestPortfolioScaling:
    """Test suite for portfolio position scaling."""

    def test_no_scaling_needed(self, portfolio_scaler):
        """Test when total risk is within limits."""
        bot_kelly = {
            "bot1": 0.005,  # 0.5%
            "bot2": 0.010,  # 1.0%
            "bot3": 0.008,  # 0.8%
        }
        # Total = 2.3% < 3% (max_portfolio_risk)

        scaled = portfolio_scaler.scale_bot_positions(bot_kelly)

        # Should not scale
        assert scaled == bot_kelly

    def test_scaling_required(self, portfolio_scaler):
        """Test when total risk exceeds limits."""
        bot_kelly = {
            "bot1": 0.015,  # 1.5%
            "bot2": 0.020,  # 2.0%
            "bot3": 0.015,  # 1.5%
        }
        # Total = 5.0% > 3% (max_portfolio_risk)

        scaled = portfolio_scaler.scale_bot_positions(bot_kelly)

        # Should scale proportionally
        total_raw = sum(bot_kelly.values())
        total_scaled = sum(scaled.values())

        assert total_scaled <= portfolio_scaler.max_portfolio_risk

        # Scale factor should be consistent
        scale_factor = portfolio_scaler.max_portfolio_risk / total_raw
        for bot_id, kelly_f in scaled.items():
            expected = bot_kelly[bot_id] * scale_factor
            assert abs(kelly_f - expected) < 0.001

    def test_single_bot_no_scaling(self, portfolio_scaler):
        """Test with single bot (no scaling needed)."""
        bot_kelly = {"bot1": 0.02}

        scaled = portfolio_scaler.scale_bot_positions(bot_kelly)

        # Should not scale single bot
        assert scaled == bot_kelly

    def test_empty_bot_dict(self, portfolio_scaler):
        """Test with empty bot dictionary."""
        scaled = portfolio_scaler.scale_bot_positions({})

        assert scaled == {}

    def test_correlation_adjustment(self, portfolio_scaler):
        """Test correlation penalty for correlated bots."""
        bot_kelly = {
            "bot1": 0.015,
            "bot2": 0.020,
            "bot3": 0.015,
        }
        # High correlation between bot1 and bot2
        correlations = {
            ("bot1", "bot2"): 0.9,
            ("bot1", "bot3"): 0.3,
            ("bot2", "bot3"): 0.2,
        }

        scaled = portfolio_scaler.scale_bot_positions(bot_kelly, correlations)

        # Should scale more due to high correlation
        total_raw = sum(bot_kelly.values())
        total_scaled = sum(scaled.values())

        # With correlation, scale factor should be smaller
        # Scale factor without correlation: 3% / 5% = 0.6
        # With correlation penalty: should be < 0.6
        scale_factor = total_scaled / total_raw
        assert scale_factor < 0.6


@pytest.mark.portfolio
class TestPortfolioStatus:
    """Test suite for portfolio status reporting."""

    def test_safe_status(self, portfolio_scaler):
        """Test safe portfolio status (< 70% utilization)."""
        bot_kelly = {
            "bot1": 0.005,
            "bot2": 0.010,
        }
        # Total = 1.5% < 2.1% (70% of 3%)

        status = portfolio_scaler.get_portfolio_status(bot_kelly)

        assert status.total_raw_risk == 0.015
        assert status.total_scaled_risk == 0.015  # No scaling
        assert status.risk_utilization < 0.7
        assert status.status == 'safe'
        assert status.scale_factor == 1.0
        assert 'healthy' in status.recommendation.lower()

    def test_caution_status(self, portfolio_scaler):
        """Test caution portfolio status (70-90% utilization)."""
        bot_kelly = {
            "bot1": 0.010,
            "bot2": 0.012,
        }
        # Total = 2.2% (~73% of 3%)

        status = portfolio_scaler.get_portfolio_status(bot_kelly)

        assert status.risk_utilization >= 0.7
        assert status.risk_utilization < 0.9
        assert status.status == 'caution'
        assert 'monitor' in status.recommendation.lower()

    def test_danger_status(self, portfolio_scaler):
        """Test danger portfolio status (> 90% utilization)."""
        bot_kelly = {
            "bot1": 0.015,
            "bot2": 0.020,
        }
        # Total = 3.5% > 3% (will be scaled)

        status = portfolio_scaler.get_portfolio_status(bot_kelly)

        assert status.status == 'danger'
        assert 'risk limit' in status.recommendation.lower() or 'reduce' in status.recommendation.lower()
        assert status.scale_factor < 1.0

    def test_empty_portfolio_status(self, portfolio_scaler):
        """Test status with no active bots."""
        status = portfolio_scaler.get_portfolio_status({})

        assert status.total_raw_risk == 0.0
        assert status.total_scaled_risk == 0.0
        assert status.risk_utilization == 0.0
        assert status.bot_count == 0
        assert status.status == 'safe'


@pytest.mark.portfolio
class TestRiskAllocation:
    """Test suite for risk allocation strategies."""

    def test_equal_allocation(self, portfolio_scaler):
        """Test equal risk allocation across bots."""
        bot_ids = ["bot1", "bot2", "bot3", "bot4"]

        allocation = portfolio_scaler.allocate_risk_equally(bot_ids)

        # Each bot should get 3% / 4 = 0.75%
        expected_risk = portfolio_scaler.max_portfolio_risk / len(bot_ids)

        for bot_id in bot_ids:
            assert abs(allocation[bot_id] - expected_risk) < 0.001

    def test_equal_allocation_custom_budget(self, portfolio_scaler):
        """Test equal allocation with custom budget."""
        bot_ids = ["bot1", "bot2", "bot3"]
        custom_budget = 0.02  # 2%

        allocation = portfolio_scaler.allocate_risk_equally(bot_ids, custom_budget)

        # Each bot should get 2% / 3
        expected_risk = custom_budget / len(bot_ids)

        for bot_id in bot_ids:
            assert abs(allocation[bot_id] - expected_risk) < 0.001

    def test_performance_based_allocation(self, portfolio_scaler):
        """Test allocation based on bot performance."""
        bot_performance = {
            "bot1": 2.0,  # Sharpe ratio
            "bot2": 1.5,
            "bot3": 1.0,
            "bot4": 0.5,
        }

        allocation = portfolio_scaler.allocate_risk_by_performance(bot_performance)

        # Higher performers should get more allocation
        assert allocation["bot1"] > allocation["bot2"]
        assert allocation["bot2"] > allocation["bot3"]
        assert allocation["bot3"] > allocation["bot4"]

        # Total should not exceed max_portfolio_risk
        total = sum(allocation.values())
        assert total <= portfolio_scaler.max_portfolio_risk

    def test_performance_allocation_with_negative(self, portfolio_scaler):
        """Test performance allocation with negative performers."""
        bot_performance = {
            "bot1": 2.0,
            "bot2": -0.5,  # Negative performer
            "bot3": 1.5,
            "bot4": -1.0,  # Negative performer
        }

        allocation = portfolio_scaler.allocate_risk_by_performance(bot_performance)

        # Negative performers should get minimum allocation
        min_allocation = 0.005
        assert allocation["bot2"] == min_allocation
        assert allocation["bot4"] == min_allocation

        # Positive performers should get more
        assert allocation["bot1"] > min_allocation
        assert allocation["bot3"] > min_allocation

    def test_performance_allocation_all_negative(self, portfolio_scaler):
        """Test performance allocation when all bots are negative."""
        bot_performance = {
            "bot1": -0.5,
            "bot2": -1.0,
            "bot3": -0.3,
        }

        allocation = portfolio_scaler.allocate_risk_by_performance(bot_performance)

        # All should get minimum allocation
        min_allocation = 0.005
        for bot_id in bot_performance:
            assert allocation[bot_id] == min_allocation


@pytest.mark.portfolio
class TestBotLimitRecommendation:
    """Test suite for bot limit recommendations."""

    def test_recommend_bot_limit_1_risk(self, portfolio_scaler):
        """Test bot limit recommendation for 1% risk per bot."""
        max_bots = portfolio_scaler.recommend_bot_limit(target_risk_per_bot=0.01)

        # 3% max / 1% per bot = 3 bots
        assert max_bots == 3

    def test_recommend_bot_limit_2_risk(self, portfolio_scaler):
        """Test bot limit recommendation for 2% risk per bot."""
        max_bots = portfolio_scaler.recommend_bot_limit(target_risk_per_bot=0.02)

        # 3% max / 2% per bot = 1 bot
        assert max_bots == 1

    def test_recommend_bot_limit_small_risk(self, portfolio_scaler):
        """Test bot limit recommendation for small risk per bot."""
        max_bots = portfolio_scaler.recommend_bot_limit(target_risk_per_bot=0.005)

        # 3% max / 0.5% per bot = 6 bots
        assert max_bots == 6

    def test_recommend_bot_limit_zero_risk(self, portfolio_scaler):
        """Test bot limit recommendation with zero risk."""
        max_bots = portfolio_scaler.recommend_bot_limit(target_risk_per_bot=0.0)

        assert max_bots == 0


@pytest.mark.edge_case
class TestPortfolioEdgeCases:
    """Test suite for portfolio edge cases."""

    def test_perfect_correlation(self, portfolio_scaler):
        """Test with perfectly correlated bots."""
        bot_kelly = {
            "bot1": 0.015,
            "bot2": 0.015,
            "bot3": 0.015,
        }
        correlations = {
            ("bot1", "bot2"): 1.0,
            ("bot1", "bot3"): 1.0,
            ("bot2", "bot3"): 1.0,
        }

        scaled = portfolio_scaler.scale_bot_positions(bot_kelly, correlations)

        # Should apply maximum correlation penalty
        total_raw = sum(bot_kelly.values())
        total_scaled = sum(scaled.values())

        # Correlation penalty with correlation_adjustment=1.5:
        # penalty = 1.0 + (1.0 * 0.5) = 1.5
        # Scale factor = 3% / (4.5% * 1.5) = 0.444
        scale_factor = total_scaled / total_raw
        assert scale_factor < (portfolio_scaler.max_portfolio_risk / total_raw)

    def test_zero_correlation(self, portfolio_scaler):
        """Test with uncorrelated bots."""
        bot_kelly = {
            "bot1": 0.015,
            "bot2": 0.020,
        }
        correlations = {
            ("bot1", "bot2"): 0.0,
        }

        scaled = portfolio_scaler.scale_bot_positions(bot_kelly, correlations)

        # Should not apply correlation penalty
        total_raw = sum(bot_kelly.values())
        total_scaled = sum(scaled.values())

        scale_factor = total_scaled / total_raw
        expected_factor = portfolio_scaler.max_portfolio_risk / total_raw
        assert abs(scale_factor - expected_factor) < 0.01

    def test_negative_correlation(self, portfolio_scaler):
        """Test with negatively correlated bots (good for diversification)."""
        bot_kelly = {
            "bot1": 0.015,
            "bot2": 0.020,
        }
        correlations = {
            ("bot1", "bot2"): -0.8,  # Strong negative correlation
        }

        scaled = portfolio_scaler.scale_bot_positions(bot_kelly, correlations)

        # Should use absolute value of correlation
        # Negative correlation is good, but we still penalize any high |correlation|
        total_scaled = sum(scaled.values())

        # Should still scale but penalty based on |correlation|
        assert total_scaled <= portfolio_scaler.max_portfolio_risk

    def test_very_large_bot_count(self, portfolio_scaler):
        """Test with very large number of bots."""
        bot_kelly = {f"bot{i}": 0.005 for i in range(20)}
        # Total = 10% (way over 3% limit)

        scaled = portfolio_scaler.scale_bot_positions(bot_kelly)

        # Should scale down significantly
        total_scaled = sum(scaled.values())
        assert total_scaled <= portfolio_scaler.max_portfolio_risk

        # Scale factor should be consistent across all bots
        scale_factors = [
            scaled[f"bot{i}"] / bot_kelly[f"bot{i}"]
            for i in range(20)
        ]
        # All scale factors should be approximately equal
        assert max(scale_factors) - min(scale_factors) < 0.001


@pytest.mark.integration
class TestPortfolioIntegration:
    """Integration tests for portfolio scaling."""

    def test_multi_bot_portfolio_management(self, portfolio_scaler):
        """Test realistic multi-bot portfolio scenario."""
        # 5 bots with different risk levels
        bot_kelly = {
            "trend_bot": 0.012,
            "mean_revert_bot": 0.008,
            "breakout_bot": 0.015,
            "scalping_bot": 0.005,
            "swing_bot": 0.010,
        }

        # Get status before scaling
        status = portfolio_scaler.get_portfolio_status(bot_kelly)

        # Total raw risk = 5%
        assert status.total_raw_risk == 0.05

        # Should be in danger zone
        assert status.status == 'danger'

        # Scale positions
        scaled = portfolio_scaler.scale_bot_positions(bot_kelly)

        # Total should be within limit
        total_scaled = sum(scaled.values())
        assert total_scaled <= portfolio_scaler.max_portfolio_risk

        # All bots should be scaled proportionally
        for bot_id in bot_kelly:
            assert scaled[bot_id] < bot_kelly[bot_id]

    def test_dynamic_bot_addition_removal(self, portfolio_scaler):
        """Test adding/removing bots dynamically."""
        # Start with 2 bots
        bot_kelly_v1 = {
            "bot1": 0.010,
            "bot2": 0.010,
        }

        scaled_v1 = portfolio_scaler.scale_bot_positions(bot_kelly_v1)
        assert sum(scaled_v1.values()) == 0.02  # No scaling needed

        # Add 2 more bots
        bot_kelly_v2 = {
            "bot1": 0.010,
            "bot2": 0.010,
            "bot3": 0.010,
            "bot4": 0.010,
        }

        scaled_v2 = portfolio_scaler.scale_bot_positions(bot_kelly_v2)
        assert sum(scaled_v2.values()) <= portfolio_scaler.max_portfolio_risk

        # Each bot should receive less after adding more bots
        assert scaled_v2["bot1"] < scaled_v1["bot1"]

    def test_portfolio_with_correlation_matrix(self, portfolio_scaler):
        """Test portfolio with full correlation matrix."""
        bot_kelly = {
            "bot1": 0.015,
            "bot2": 0.015,
            "bot3": 0.015,
        }

        # Full correlation matrix
        correlations = {
            ("bot1", "bot2"): 0.8,
            ("bot1", "bot3"): 0.6,
            ("bot2", "bot3"): 0.7,
        }

        scaled = portfolio_scaler.scale_bot_positions(bot_kelly, correlations)
        status = portfolio_scaler.get_portfolio_status(bot_kelly, correlations)

        # Should account for correlations in scaling
        assert status.total_scaled_risk <= portfolio_scaler.max_portfolio_risk
        # Correlation-adjusted scale should be more conservative
        # than simple proportional scaling

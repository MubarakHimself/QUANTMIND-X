"""
Integration tests for RiskGovernor orchestrator.

These tests verify the end-to-end functionality of the RiskGovernor class,
including position sizing calculations, physics adjustments, prop firm constraints,
and portfolio scaling.
"""

import unittest
from unittest.mock import Mock, patch
import time
from datetime import datetime, timedelta

from src.risk.governor import RiskGovernor
from src.risk.models.position_sizing_result import PositionSizingResult
from src.risk.models.sizing_recommendation import SizingRecommendation
from src.risk.config import FTPreset, The5ersPreset, FundingPipsPreset


class TestRiskGovernor(unittest.TestCase):
    """Test suite for RiskGovernor functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.governor = RiskGovernor()
        self.account_balance = 10000.0
        self.win_rate = 0.55
        self.avg_win = 400.0
        self.avg_loss = 200.0
        self.current_atr = 0.0012
        self.average_atr = 0.0010
        self.stop_loss_pips = 20.0
        self.pip_value = 10.0

    def test_basic_position_sizing(self):
        """Test basic position sizing calculation."""
        result = self.governor.calculate_position_size(
            self.account_balance,
            self.win_rate,
            self.avg_win,
            self.avg_loss,
            self.current_atr,
            self.average_atr,
            self.stop_loss_pips,
            self.pip_value
        )

        self.assertIsInstance(result, PositionSizingResult)
        self.assertGreater(result.lot_size, 0)
        self.assertLessEqual(result.risk_amount, self.account_balance * 0.02)  # Max 2% risk
        self.assertEqual(len(result.calculation_steps), 9)  # 8 steps + final

    def test_negative_expectancy_handling(self):
        """Test handling of negative expectancy scenarios."""
        # Use win rate that results in negative expectancy
        negative_win_rate = 0.35
        result = self.governor.calculate_position_size(
            self.account_balance,
            negative_win_rate,
            self.avg_win,
            self.avg_loss,
            self.current_atr,
            self.average_atr,
            self.stop_loss_pips,
            self.pip_value
        )

        self.assertGreater(result.lot_size, 0)
        # Should use max risk as fallback
        self.assertAlmostEqual(result.risk_amount, self.account_balance * 0.02, places=2)

    def test_prop_firm_preset_ftmo(self):
        """Test FTMO prop firm preset constraints."""
        self.governor.set_prop_firm_preset("ftmo")
        self.assertEqual(self.governor.get_max_risk_pct(), 0.02)  # 10% drawdown * 20% = 2%

    def test_prop_firm_preset_the5ers(self):
        """Test The5ers prop firm preset constraints."""
        self.governor.set_prop_firm_preset("the5ers")
        self.assertEqual(self.governor.get_max_risk_pct(), 0.016)  # 8% drawdown * 20% = 1.6%

    def test_prop_firm_preset_fundingpips(self):
        """Test FundingPips prop firm preset constraints."""
        self.governor.set_prop_firm_preset("fundingpips")
        self.assertEqual(self.governor.get_max_risk_pct(), 0.024)  # 12% drawdown * 20% = 2.4%

    def test_portfolio_scaling(self):
        """Test portfolio scaling functionality."""
        # Create portfolio risk data
        portfolio_risk = {
            ('bot1', 'bot2'): 0.7,  # High correlation
            ('bot1', 'bot3'): 0.3   # Low correlation
        }

        # Test with portfolio scaling
        result_with_scaling = self.governor.calculate_position_size(
            self.account_balance,
            self.win_rate,
            self.avg_win,
            self.avg_loss,
            self.current_atr,
            self.average_atr,
            self.stop_loss_pips,
            self.pip_value,
            bot_id="bot1",
            portfolio_risk=portfolio_risk
        )

        # Test without portfolio scaling (should be same as basic)
        result_without_scaling = self.governor.calculate_position_size(
            self.account_balance,
            self.win_rate,
            self.avg_win,
            self.avg_loss,
            self.current_atr,
            self.average_atr,
            self.stop_loss_pips,
            self.pip_value
        )

        # With high correlation, position should be scaled down
        self.assertLess(result_with_scaling.lot_size, result_without_scaling.lot_size)

    def test_sizing_recommendation(self):
        """Test sizing recommendation generation."""
        recommendation = self.governor.get_sizing_recommendation(
            self.account_balance,
            self.win_rate,
            self.avg_win,
            self.avg_loss,
            self.current_atr,
            self.average_atr,
            self.stop_loss_pips,
            self.pip_value
        )

        self.assertIsInstance(recommendation, SizingRecommendation)
        self.assertGreater(recommendation.raw_kelly, 0)
        self.assertGreater(recommendation.physics_multiplier, 0)
        self.assertGreater(recommendation.final_risk_pct, 0)
        self.assertGreater(recommendation.position_size_lots, 0)

    def test_portfolio_status(self):
        """Test portfolio status calculation."""
        portfolio_risk = {
            'bot1': 0.015,
            'bot2': 0.012,
            'bot3': 0.008
        }

        status = self.governor.get_portfolio_status(portfolio_risk)

        self.assertIn('total_raw_risk', status)
        self.assertIn('total_scaled_risk', status)
        self.assertIn('risk_utilization', status)
        self.assertIn('status', status)
        self.assertIn('recommendation', status)

        # Total raw risk should be sum of individual risks
        self.assertAlmostEqual(status['total_raw_risk'], sum(portfolio_risk.values()), places=6)

    def test_cache_functionality(self):
        """Test caching functionality."""
        # First calculation
        result1 = self.governor.calculate_position_size(
            self.account_balance,
            self.win_rate,
            self.avg_win,
            self.avg_loss,
            self.current_atr,
            self.average_atr,
            self.stop_loss_pips,
            self.pip_value
        )

        # Second calculation (should use cache)
        result2 = self.governor.calculate_position_size(
            self.account_balance,
            self.win_rate,
            self.avg_win,
            self.avg_loss,
            self.current_atr,
            self.average_atr,
            self.stop_loss_pips,
            self.pip_value
        )

        # Results should be identical
        self.assertEqual(result1.lot_size, result2.lot_size)
        self.assertEqual(result1.risk_amount, result2.risk_amount)

    def test_json_serialization(self):
        """Test JSON serialization of governor state."""
        json_str = self.governor.to_json()
        self.assertIsInstance(json_str, str)
        self.assertIn('prop_firm_preset', json_str)
        self.assertIn('max_risk_pct', json_str)

    def test_string_representation(self):
        """Test string representation of governor."""
        str_repr = str(self.governor)
        self.assertIsInstance(str_repr, str)
        self.assertIn('RiskGovernor', str_repr)


class TestRiskGovernorEdgeCases(unittest.TestCase):
    """Test edge cases for RiskGovernor."""

    def setUp(self):
        """Set up test fixtures."""
        self.governor = RiskGovernor()

    def test_zero_account_balance(self):
        """Test with zero account balance."""
        with self.assertRaises(ValueError):
            self.governor.calculate_position_size(
                account_balance=0.0,
                win_rate=self.governor.win_rate,
                avg_win=self.governor.avg_win,
                avg_loss=self.governor.avg_loss,
                current_atr=self.governor.current_atr,
                average_atr=self.governor.average_atr,
                stop_loss_pips=self.governor.stop_loss_pips,
                pip_value=self.governor.pip_value
            )

    def test_negative_win_rate(self):
        """Test with negative win rate."""
        with self.assertRaises(ValueError):
            self.governor.calculate_position_size(
                account_balance=self.governor.account_balance,
                win_rate=-0.1,
                avg_win=self.governor.avg_win,
                avg_loss=self.governor.avg_loss,
                current_atr=self.governor.current_atr,
                average_atr=self.governor.average_atr,
                stop_loss_pips=self.governor.stop_loss_pips,
                pip_value=self.governor.pip_value
            )

    def test_zero_stop_loss(self):
        """Test with zero stop loss."""
        with self.assertRaises(ValueError):
            self.governor.calculate_position_size(
                account_balance=self.governor.account_balance,
                win_rate=self.governor.win_rate,
                avg_win=self.governor.avg_win,
                avg_loss=self.governor.avg_loss,
                current_atr=self.governor.current_atr,
                average_atr=self.governor.average_atr,
                stop_loss_pips=0.0,
                pip_value=self.governor.pip_value
            )

    def test_invalid_preset(self):
        """Test with invalid prop firm preset."""
        with self.assertRaises(ValueError):
            self.governor.set_prop_firm_preset("invalid_preset")


if __name__ == '__main__':
    unittest.main()
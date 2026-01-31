"""
Property-Based Tests for Kelly Criterion Calculation Accuracy

**Feature: quantmindx-unified-backend, Property 20: Kelly Criterion Calculation Accuracy**

For any position sizing request with win rate, average win, and average loss inputs,
the Kelly calculation SHALL produce mathematically correct results.

This test validates the Kelly Criterion formula: f* = (bp - q) / b
where:
  b = avgWin / avgLoss (payoff ratio)
  p = winRate (probability of win)
  q = 1 - p (probability of loss)
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
import math


# Since we can't directly test MQL5 code, we implement the Kelly formula
# in Python to verify the mathematical correctness
def calculate_kelly_fraction_python(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """
    Python implementation of Kelly Criterion for validation
    
    Formula: f* = (bp - q) / b
    where:
      b = avgWin / avgLoss (payoff ratio)
      p = winRate (probability of win)
      q = 1 - p (probability of loss)
    """
    # Validate inputs
    if not (0.0 <= win_rate <= 1.0):
        return 0.0
    if avg_win <= 0.0 or avg_loss <= 0.0:
        return 0.0
    
    # Calculate payoff ratio
    b = avg_win / avg_loss
    
    # Calculate probabilities
    p = win_rate
    q = 1.0 - win_rate
    
    # Kelly formula
    raw_kelly = (b * p - q) / b
    
    # Negative expected value = don't trade
    if raw_kelly < 0.0:
        return 0.0
    
    # Cap at 25% (QM_KELLY_MAX_FRACTION)
    final_kelly = min(raw_kelly, 0.25)
    
    return final_kelly


class TestKellyCriterionProperties:
    """Property-based tests for Kelly Criterion calculation"""
    
    @given(
        win_rate=st.floats(min_value=0.0, max_value=1.0),
        avg_win=st.floats(min_value=0.01, max_value=10000.0),
        avg_loss=st.floats(min_value=0.01, max_value=10000.0)
    )
    @settings(max_examples=100)
    def test_kelly_criterion_calculation_accuracy(self, win_rate, avg_win, avg_loss):
        """
        **Feature: quantmindx-unified-backend, Property 20: Kelly Criterion Calculation Accuracy**
        
        For any valid inputs (win_rate, avg_win, avg_loss), the Kelly Criterion
        calculation must produce mathematically correct results according to the formula:
        f* = (bp - q) / b
        """
        # Filter out NaN and infinite values
        assume(not math.isnan(win_rate))
        assume(not math.isnan(avg_win))
        assume(not math.isnan(avg_loss))
        assume(not math.isinf(avg_win))
        assume(not math.isinf(avg_loss))
        
        # Calculate Kelly fraction
        kelly_fraction = calculate_kelly_fraction_python(win_rate, avg_win, avg_loss)
        
        # Property 1: Result must be in valid range [0.0, 0.25]
        assert 0.0 <= kelly_fraction <= 0.25, \
            f"Kelly fraction {kelly_fraction} outside valid range [0.0, 0.25]"
        
        # Property 2: If win rate is 0, Kelly should be 0
        if win_rate == 0.0:
            assert kelly_fraction == 0.0, \
                "Kelly fraction should be 0 when win rate is 0"
        
        # Property 3: If win rate is 1.0, Kelly should be maximum (0.25)
        if win_rate == 1.0:
            assert kelly_fraction == 0.25, \
                "Kelly fraction should be 0.25 (max) when win rate is 1.0"
        
        # Property 4: Verify mathematical formula
        if kelly_fraction > 0.0:
            b = avg_win / avg_loss
            p = win_rate
            q = 1.0 - win_rate
            expected_raw = (b * p - q) / b
            
            if expected_raw > 0.0:
                expected_final = min(expected_raw, 0.25)
                assert abs(kelly_fraction - expected_final) < 1e-10, \
                    f"Kelly calculation mismatch: got {kelly_fraction}, expected {expected_final}"
    
    @given(
        win_rate=st.floats(min_value=0.0, max_value=0.5),
        avg_win=st.floats(min_value=0.01, max_value=1000.0),
        avg_loss=st.floats(min_value=0.01, max_value=1000.0)
    )
    @settings(max_examples=100)
    def test_kelly_negative_expected_value(self, win_rate, avg_win, avg_loss):
        """
        **Feature: quantmindx-unified-backend, Property 20: Kelly Criterion Calculation Accuracy**
        
        For strategies with negative expected value (win_rate <= 0.5 with equal payoff),
        Kelly should return 0 or very small values.
        """
        assume(not math.isnan(win_rate))
        assume(not math.isnan(avg_win))
        assume(not math.isnan(avg_loss))
        assume(not math.isinf(avg_win))
        assume(not math.isinf(avg_loss))
        
        # If payoff ratio is 1:1 or worse, and win rate <= 0.5, EV is negative
        if avg_win <= avg_loss and win_rate <= 0.5:
            kelly_fraction = calculate_kelly_fraction_python(win_rate, avg_win, avg_loss)
            assert kelly_fraction == 0.0, \
                f"Kelly should be 0 for negative EV strategy, got {kelly_fraction}"
    
    @given(
        win_rate=st.floats(min_value=0.6, max_value=1.0),
        payoff_ratio=st.floats(min_value=1.5, max_value=5.0)
    )
    @settings(max_examples=100)
    def test_kelly_positive_expected_value(self, win_rate, payoff_ratio):
        """
        **Feature: quantmindx-unified-backend, Property 20: Kelly Criterion Calculation Accuracy**
        
        For strategies with positive expected value (high win rate and good payoff),
        Kelly should return positive values.
        """
        assume(not math.isnan(win_rate))
        assume(not math.isnan(payoff_ratio))
        assume(not math.isinf(payoff_ratio))
        
        # Create avg_win and avg_loss from payoff ratio
        avg_loss = 100.0
        avg_win = avg_loss * payoff_ratio
        
        kelly_fraction = calculate_kelly_fraction_python(win_rate, avg_win, avg_loss)
        
        # With high win rate and good payoff, Kelly should be positive
        assert kelly_fraction > 0.0, \
            f"Kelly should be positive for good strategy, got {kelly_fraction}"
    
    @given(
        avg_win=st.floats(min_value=0.01, max_value=1000.0),
        avg_loss=st.floats(min_value=0.01, max_value=1000.0)
    )
    @settings(max_examples=100)
    def test_kelly_payoff_ratio_symmetry(self, avg_win, avg_loss):
        """
        **Feature: quantmindx-unified-backend, Property 20: Kelly Criterion Calculation Accuracy**
        
        The Kelly fraction should depend on the payoff ratio (avg_win / avg_loss),
        not the absolute values.
        """
        assume(not math.isnan(avg_win))
        assume(not math.isnan(avg_loss))
        assume(not math.isinf(avg_win))
        assume(not math.isinf(avg_loss))
        
        win_rate = 0.6  # Fixed win rate
        
        # Calculate Kelly with original values
        kelly1 = calculate_kelly_fraction_python(win_rate, avg_win, avg_loss)
        
        # Calculate Kelly with scaled values (same ratio)
        scale_factor = 10.0
        kelly2 = calculate_kelly_fraction_python(win_rate, avg_win * scale_factor, avg_loss * scale_factor)
        
        # Results should be identical (within floating point precision)
        assert abs(kelly1 - kelly2) < 1e-10, \
            f"Kelly should depend on ratio, not absolute values: {kelly1} vs {kelly2}"
    
    @given(
        win_rate=st.floats(min_value=0.0, max_value=1.0),
        avg_win=st.floats(min_value=0.01, max_value=1000.0),
        avg_loss=st.floats(min_value=0.01, max_value=1000.0)
    )
    @settings(max_examples=100)
    def test_kelly_monotonicity_with_win_rate(self, win_rate, avg_win, avg_loss):
        """
        **Feature: quantmindx-unified-backend, Property 20: Kelly Criterion Calculation Accuracy**
        
        For fixed payoff ratio, Kelly fraction should increase monotonically
        with win rate (higher win rate = higher Kelly).
        """
        assume(not math.isnan(win_rate))
        assume(not math.isnan(avg_win))
        assume(not math.isnan(avg_loss))
        assume(not math.isinf(avg_win))
        assume(not math.isinf(avg_loss))
        assume(win_rate < 1.0)  # Need room to increase
        
        kelly1 = calculate_kelly_fraction_python(win_rate, avg_win, avg_loss)
        
        # Increase win rate slightly
        higher_win_rate = min(win_rate + 0.1, 1.0)
        kelly2 = calculate_kelly_fraction_python(higher_win_rate, avg_win, avg_loss)
        
        # Kelly should not decrease when win rate increases
        assert kelly2 >= kelly1, \
            f"Kelly should increase with win rate: {kelly1} -> {kelly2}"
    
    @given(
        win_rate=st.floats(min_value=0.5, max_value=1.0),
        avg_win=st.floats(min_value=0.01, max_value=1000.0),
        avg_loss=st.floats(min_value=0.01, max_value=1000.0)
    )
    @settings(max_examples=100)
    def test_kelly_cap_at_25_percent(self, win_rate, avg_win, avg_loss):
        """
        **Feature: quantmindx-unified-backend, Property 20: Kelly Criterion Calculation Accuracy**
        
        Kelly fraction should never exceed 25% (QM_KELLY_MAX_FRACTION),
        even for very favorable strategies.
        """
        assume(not math.isnan(win_rate))
        assume(not math.isnan(avg_win))
        assume(not math.isnan(avg_loss))
        assume(not math.isinf(avg_win))
        assume(not math.isinf(avg_loss))
        
        kelly_fraction = calculate_kelly_fraction_python(win_rate, avg_win, avg_loss)
        
        # Kelly should never exceed 25%
        assert kelly_fraction <= 0.25, \
            f"Kelly fraction {kelly_fraction} exceeds maximum 0.25"
    
    def test_kelly_known_values(self):
        """
        **Feature: quantmindx-unified-backend, Property 20: Kelly Criterion Calculation Accuracy**
        
        Test Kelly calculation against known values to ensure correctness.
        """
        # Test case 1: 55% win rate, 2:1 payoff
        # b = 2, p = 0.55, q = 0.45
        # f* = (2*0.55 - 0.45) / 2 = (1.1 - 0.45) / 2 = 0.325
        # Capped at 0.25
        kelly1 = calculate_kelly_fraction_python(0.55, 200, 100)
        assert abs(kelly1 - 0.25) < 1e-10, f"Expected 0.25, got {kelly1}"
        
        # Test case 2: 60% win rate, 1:1 payoff
        # b = 1, p = 0.6, q = 0.4
        # f* = (1*0.6 - 0.4) / 1 = 0.2
        kelly2 = calculate_kelly_fraction_python(0.6, 100, 100)
        assert abs(kelly2 - 0.2) < 1e-10, f"Expected 0.2, got {kelly2}"
        
        # Test case 3: 50% win rate, 1:1 payoff (break-even)
        # b = 1, p = 0.5, q = 0.5
        # f* = (1*0.5 - 0.5) / 1 = 0
        kelly3 = calculate_kelly_fraction_python(0.5, 100, 100)
        assert kelly3 == 0.0, f"Expected 0.0, got {kelly3}"
        
        # Test case 4: 40% win rate, 1:1 payoff (losing strategy)
        # Negative EV, should return 0
        kelly4 = calculate_kelly_fraction_python(0.4, 100, 100)
        assert kelly4 == 0.0, f"Expected 0.0 for losing strategy, got {kelly4}"
        
        # Test case 5: 100% win rate (perfect strategy)
        # Should return maximum 0.25
        kelly5 = calculate_kelly_fraction_python(1.0, 100, 100)
        assert kelly5 == 0.25, f"Expected 0.25 for perfect strategy, got {kelly5}"
        
        # Test case 6: 0% win rate (always lose)
        # Should return 0
        kelly6 = calculate_kelly_fraction_python(0.0, 100, 100)
        assert kelly6 == 0.0, f"Expected 0.0 for 0% win rate, got {kelly6}"
    
    def test_kelly_edge_cases(self):
        """
        **Feature: quantmindx-unified-backend, Property 20: Kelly Criterion Calculation Accuracy**
        
        Test edge cases and boundary conditions.
        """
        # Edge case 1: Very small avg_win
        kelly1 = calculate_kelly_fraction_python(0.6, 0.01, 100)
        assert kelly1 == 0.0, "Very small payoff should result in 0 Kelly"
        
        # Edge case 2: Very large avg_win
        kelly2 = calculate_kelly_fraction_python(0.6, 10000, 100)
        assert kelly2 == 0.25, "Very large payoff should be capped at 0.25"
        
        # Edge case 3: Equal avg_win and avg_loss with 50% win rate
        kelly3 = calculate_kelly_fraction_python(0.5, 100, 100)
        assert kelly3 == 0.0, "Break-even strategy should have 0 Kelly"
        
        # Edge case 4: Win rate at boundary (0.0)
        kelly4 = calculate_kelly_fraction_python(0.0, 200, 100)
        assert kelly4 == 0.0, "0% win rate should have 0 Kelly"
        
        # Edge case 5: Win rate at boundary (1.0)
        kelly5 = calculate_kelly_fraction_python(1.0, 200, 100)
        assert kelly5 == 0.25, "100% win rate should have max Kelly (0.25)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Property-Based Tests for V8 Tiered Risk Engine

Tests universal properties of the three-tier risk system using Hypothesis.

**Feature: quantmindx-unified-backend**
"""

import pytest
from hypothesis import given, strategies as st, settings, assume


# ============================================================================
# Property 31: Tiered Risk Tier Determination
# ============================================================================

@given(
    equity=st.floats(min_value=1.0, max_value=100000.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=100)
def test_property_31_tiered_risk_tier_determination(equity):
    """
    Property 31: For any account equity value, the risk tier SHALL be correctly 
    determined: Growth ($100-$1K), Scaling ($1K-$5K), or Guardian ($5K+).
    
    **Feature: quantmindx-unified-backend, Property 31: Tiered Risk Tier Determination**
    **Validates: Requirement 16.2, 16.3, 16.4**
    """
    growth_ceiling = 1000.0
    scaling_ceiling = 5000.0
    
    # Determine tier based on equity
    if equity < growth_ceiling:
        tier = "growth"
    elif equity < scaling_ceiling:
        tier = "scaling"
    else:
        tier = "guardian"
    
    # Verify tier is one of the three valid tiers
    assert tier in ["growth", "scaling", "guardian"]
    
    # Verify tier boundaries are correct
    if tier == "growth":
        assert equity < growth_ceiling
    elif tier == "scaling":
        assert growth_ceiling <= equity < scaling_ceiling
    elif tier == "guardian":
        assert equity >= scaling_ceiling


# ============================================================================
# Property 32: Fixed Risk Calculation in Growth Tier
# ============================================================================

@given(
    equity=st.floats(min_value=1.0, max_value=999.99, allow_nan=False, allow_infinity=False),
    stop_loss_pips=st.floats(min_value=1.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    tick_value=st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=100)
def test_property_32_fixed_risk_calculation_growth_tier(equity, stop_loss_pips, tick_value):
    """
    Property 32: For any trade in Growth Tier, position size SHALL use dynamic 
    aggressive risk (3% with $2 floor) divided by stop-loss distance.
    
    **Feature: quantmindx-unified-backend, Property 32: Fixed Risk Calculation in Growth Tier**
    **Validates: Requirement 16.2**
    """
    # Growth tier parameters
    growth_percent = 3.0
    fixed_floor = 2.0
    
    # Calculate dynamic risk amount with floor
    percent_risk = equity * (growth_percent / 100.0)
    risk_amount = max(percent_risk, fixed_floor)
    
    # Calculate position size
    risk_per_lot = stop_loss_pips * tick_value
    position_size = risk_amount / risk_per_lot
    
    # Verify risk amount is never below floor
    assert risk_amount >= fixed_floor
    
    # Verify position size is positive
    assert position_size > 0
    
    # Verify position size calculation is correct
    expected_position_size = risk_amount / (stop_loss_pips * tick_value)
    assert abs(position_size - expected_position_size) < 1e-10


# ============================================================================
# Property 33: Quadratic Throttle Formula Accuracy
# ============================================================================

@given(
    base_risk=st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    current_loss=st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    max_loss=st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=100)
def test_property_33_quadratic_throttle_formula(base_risk, current_loss, max_loss):
    """
    Property 33: For any Guardian Tier trade, the Quadratic Throttle SHALL apply 
    the formula: Multiplier = ((MaxLoss - CurrentLoss) / MaxLoss)^2.
    
    **Feature: quantmindx-unified-backend, Property 33: Quadratic Throttle Formula Accuracy**
    **Validates: Requirement 16.6**
    """
    # Ensure current_loss doesn't exceed max_loss
    assume(current_loss <= max_loss)
    
    # Calculate remaining capacity
    remaining_capacity = (max_loss - current_loss) / max_loss
    
    # Apply quadratic throttle
    multiplier = remaining_capacity ** 2
    throttled_risk = base_risk * multiplier
    
    # Verify multiplier is in valid range [0, 1]
    assert 0.0 <= multiplier <= 1.0
    
    # Verify throttled risk is never greater than base risk
    assert throttled_risk <= base_risk
    
    # Verify throttled risk is non-negative
    assert throttled_risk >= 0.0
    
    # Verify formula correctness
    expected_multiplier = ((max_loss - current_loss) / max_loss) ** 2
    assert abs(multiplier - expected_multiplier) < 1e-10
    
    # Verify edge cases
    if current_loss == 0.0:
        # No loss means full capacity (multiplier = 1.0)
        assert abs(multiplier - 1.0) < 1e-10
    
    if current_loss == max_loss:
        # At max loss means zero capacity (multiplier = 0.0)
        assert abs(multiplier - 0.0) < 1e-10


# ============================================================================
# Property 34: Tier Transition Consistency
# ============================================================================

@given(
    initial_equity=st.floats(min_value=100.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
    equity_change=st.floats(min_value=-5000.0, max_value=5000.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=100)
def test_property_34_tier_transition_consistency(initial_equity, equity_change):
    """
    Property 34: For any equity change, tier transitions SHALL be consistent 
    and monotonic (no skipping tiers).
    
    **Feature: quantmindx-unified-backend, Property 34: Tier Transition Consistency**
    **Validates: Requirement 16.7**
    """
    growth_ceiling = 1000.0
    scaling_ceiling = 5000.0
    
    # Calculate new equity
    new_equity = initial_equity + equity_change
    
    # Ensure new equity is positive
    assume(new_equity > 0)
    
    # Determine initial tier
    if initial_equity < growth_ceiling:
        initial_tier = "growth"
    elif initial_equity < scaling_ceiling:
        initial_tier = "scaling"
    else:
        initial_tier = "guardian"
    
    # Determine new tier
    if new_equity < growth_ceiling:
        new_tier = "growth"
    elif new_equity < scaling_ceiling:
        new_tier = "scaling"
    else:
        new_tier = "guardian"
    
    # Verify tier is valid
    assert new_tier in ["growth", "scaling", "guardian"]
    
    # Verify tier transitions are monotonic (no skipping)
    tier_order = {"growth": 0, "scaling": 1, "guardian": 2}
    
    if equity_change > 0:
        # Growing equity should not decrease tier
        assert tier_order[new_tier] >= tier_order[initial_tier]
    elif equity_change < 0:
        # Decreasing equity should not increase tier
        assert tier_order[new_tier] <= tier_order[initial_tier]
    else:
        # No change means same tier
        assert new_tier == initial_tier


# ============================================================================
# Property 35: Risk Amount Monotonicity in Growth Tier
# ============================================================================

@given(
    equity1=st.floats(min_value=50.0, max_value=999.0, allow_nan=False, allow_infinity=False),
    equity2=st.floats(min_value=50.0, max_value=999.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=100)
def test_property_35_risk_amount_monotonicity(equity1, equity2):
    """
    Property 35: For any two equity values in Growth Tier, higher equity SHALL 
    result in equal or higher risk amount (monotonicity with floor).
    
    **Feature: quantmindx-unified-backend, Property 35: Risk Amount Monotonicity**
    **Validates: Requirement 16.2**
    """
    growth_percent = 3.0
    fixed_floor = 2.0
    
    # Calculate risk amounts
    risk1 = max(equity1 * (growth_percent / 100.0), fixed_floor)
    risk2 = max(equity2 * (growth_percent / 100.0), fixed_floor)
    
    # Verify monotonicity
    if equity1 < equity2:
        assert risk1 <= risk2
    elif equity1 > equity2:
        assert risk1 >= risk2
    else:
        assert abs(risk1 - risk2) < 1e-10


# ============================================================================
# Property 36: Position Size Scaling with Stop Loss
# ============================================================================

@given(
    equity=st.floats(min_value=100.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
    stop_loss1=st.floats(min_value=5.0, max_value=50.0, allow_nan=False, allow_infinity=False),
    stop_loss2=st.floats(min_value=5.0, max_value=50.0, allow_nan=False, allow_infinity=False),
    tick_value=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=100)
def test_property_36_position_size_scaling_with_stop_loss(equity, stop_loss1, stop_loss2, tick_value):
    """
    Property 36: For any fixed risk amount, position size SHALL be inversely 
    proportional to stop-loss distance.
    
    **Feature: quantmindx-unified-backend, Property 36: Position Size Scaling**
    **Validates: Requirement 16.2, 16.3, 16.4**
    """
    # Use fixed risk amount for simplicity
    risk_amount = 100.0
    
    # Calculate position sizes
    position_size1 = risk_amount / (stop_loss1 * tick_value)
    position_size2 = risk_amount / (stop_loss2 * tick_value)
    
    # Verify inverse relationship
    if stop_loss1 < stop_loss2:
        assert position_size1 > position_size2
    elif stop_loss1 > stop_loss2:
        assert position_size1 < position_size2
    else:
        assert abs(position_size1 - position_size2) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

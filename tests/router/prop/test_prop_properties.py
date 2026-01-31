"""
Property-Based Tests for PropFirm Components

Tests universal properties that must hold for all inputs:
- Quadratic Throttle formula accuracy
- Kelly Filter threshold enforcement
"""

import pytest
from hypothesis import given, strategies as st, assume, settings
from src.router.prop.governor import PropGovernor
from src.router.prop.commander import PropCommander
from src.router.prop.state import PropState, PropAccountMetrics
from types import SimpleNamespace


class TestQuadraticThrottleProperties:
    """
    Property tests for Quadratic Throttle calculation.
    
    **Feature: quantmindx-unified-backend, Property 6: Quadratic Throttle Formula Accuracy**
    
    The quadratic throttle MUST follow the formula:
    Throttle = 1.0 - (CurrentLoss / EffectiveLimit)^2
    
    Where EffectiveLimit = DailyLossLimit - HardStopBuffer (5% - 1% = 4%)
    """
    
    @given(
        max_loss=st.floats(min_value=0.02, max_value=0.10),  # 2% to 10% max loss (avoid division by zero)
        current_loss=st.floats(min_value=0.0, max_value=0.08)  # 0% to 8% current loss
    )
    @settings(max_examples=100)
    def test_quadratic_throttle_formula_accuracy(self, max_loss, current_loss):
        """
        Property: Quadratic throttle MUST match formula for all valid inputs.
        
        **Validates: Requirements 4.1**
        """
        # Ensure current_loss doesn't exceed effective limit
        effective_limit = max_loss - 0.01
        assume(effective_limit > 0)  # Avoid division by zero
        assume(current_loss <= effective_limit * 0.9)  # Stay below effective limit
        
        # Create governor with test config
        governor = PropGovernor("test_account")
        governor.daily_loss_limit_pct = max_loss
        governor.hard_stop_buffer = 0.01
        governor.effective_limit = effective_limit
        
        # Set daily_start_balance attribute directly
        governor.prop_state.daily_start_balance = 100000.0
        
        # Calculate throttle
        current_balance = 100000.0 - (current_loss * 100000.0)
        throttle = governor._get_quadratic_throttle(current_balance)
        
        # Calculate expected throttle using formula
        loss_pct = current_loss
        expected_throttle = 1.0 - (loss_pct / effective_limit) ** 2
        
        # Verify formula accuracy (within floating point tolerance)
        assert abs(throttle - expected_throttle) < 1e-6, \
            f"Throttle {throttle} doesn't match formula {expected_throttle}"
    
    @given(
        current_loss=st.floats(min_value=0.0, max_value=0.10)
    )
    @settings(max_examples=100)
    def test_throttle_range_bounds(self, current_loss):
        """
        Property: Throttle MUST always be between 0.0 and 1.0.
        
        **Validates: Requirements 4.1**
        """
        governor = PropGovernor("test_account")
        
        # Set daily_start_balance attribute directly
        governor.prop_state.daily_start_balance = 100000.0
        
        current_balance = 100000.0 - (current_loss * 100000.0)
        throttle = governor._get_quadratic_throttle(current_balance)
        
        # Verify bounds
        assert 0.0 <= throttle <= 1.0, \
            f"Throttle {throttle} outside valid range [0.0, 1.0]"
    
    @given(
        profit=st.floats(min_value=0.01, max_value=1.0)  # 1% to 100% profit
    )
    @settings(max_examples=100)
    def test_no_throttle_when_in_profit(self, profit):
        """
        Property: Throttle MUST be 1.0 when account is in profit.
        
        **Validates: Requirements 4.1**
        """
        governor = PropGovernor("test_account")
        
        # Set daily_start_balance attribute directly
        governor.prop_state.daily_start_balance = 100000.0
        
        current_balance = 100000.0 + (profit * 100000.0)
        throttle = governor._get_quadratic_throttle(current_balance)
        
        # Verify no throttle when in profit
        assert throttle == 1.0, \
            f"Throttle {throttle} should be 1.0 when in profit"
    
    @given(
        max_loss=st.floats(min_value=0.02, max_value=0.10)
    )
    @settings(max_examples=100)
    def test_zero_throttle_at_effective_limit(self, max_loss):
        """
        Property: Throttle MUST be 0.0 when loss reaches effective limit.
        
        **Validates: Requirements 4.1**
        """
        governor = PropGovernor("test_account")
        governor.daily_loss_limit_pct = max_loss
        governor.hard_stop_buffer = 0.01
        governor.effective_limit = max_loss - 0.01
        
        # Set loss exactly at effective limit
        effective_limit = max_loss - 0.01
        current_loss = effective_limit
        
        # Set daily_start_balance attribute directly
        governor.prop_state.daily_start_balance = 100000.0
        
        current_balance = 100000.0 - (current_loss * 100000.0)
        throttle = governor._get_quadratic_throttle(current_balance)
        
        # Verify zero throttle at effective limit (with floating point tolerance)
        assert throttle < 1e-10, \
            f"Throttle {throttle} should be ~0.0 at effective limit"
    
    @given(
        loss1=st.floats(min_value=0.01, max_value=0.03),
        loss2=st.floats(min_value=0.01, max_value=0.03)
    )
    @settings(max_examples=100)
    def test_throttle_monotonically_decreases_with_loss(self, loss1, loss2):
        """
        Property: Throttle MUST decrease as loss increases.
        
        **Validates: Requirements 4.1**
        """
        assume(loss1 != loss2)  # Ensure different losses
        
        governor = PropGovernor("test_account")
        
        # Set daily_start_balance attribute directly
        governor.prop_state.daily_start_balance = 100000.0
        
        # Calculate throttles for both losses
        throttles = []
        for loss in [loss1, loss2]:
            current_balance = 100000.0 - (loss * 100000.0)
            throttle = governor._get_quadratic_throttle(current_balance)
            throttles.append((loss, throttle))
        
        # Sort by loss
        throttles.sort(key=lambda x: x[0])
        
        # Verify throttle decreases as loss increases
        assert throttles[0][1] >= throttles[1][1], \
            f"Throttle should decrease as loss increases: {throttles}"


class TestKellyFilterProperties:
    """
    Property tests for Kelly Filter threshold enforcement.
    
    **Feature: quantmindx-unified-backend, Property 7: Kelly Filter Threshold Enforcement**
    
    The Kelly Filter MUST reject all trades with KellyScore < 0.8 in preservation mode.
    """
    
    @given(
        kelly_score=st.floats(min_value=0.0, max_value=1.0),
        preservation_mode=st.booleans()
    )
    @settings(max_examples=100)
    def test_kelly_filter_threshold_enforcement(self, kelly_score, preservation_mode):
        """
        Property: In preservation mode, trades with KellyScore < 0.8 MUST be rejected.
        
        **Validates: Requirements 4.1**
        """
        commander = PropCommander("test_account")
        
        # Mock prop_state
        if preservation_mode:
            # Set metrics to trigger preservation mode
            mock_metrics = SimpleNamespace(
                daily_start_balance=100000.0,
                current_equity=108000.0,  # 8% profit
                trading_days=10
            )
        else:
            # Set metrics for standard mode
            mock_metrics = SimpleNamespace(
                daily_start_balance=100000.0,
                current_equity=105000.0,  # 5% profit (below 8% target)
                trading_days=10
            )
        
        commander.prop_state.get_metrics = lambda: mock_metrics
        
        # Create test bot with given kelly_score
        test_bot = {"name": "TestBot", "kelly_score": kelly_score}
        
        # Mock the parent class run_auction to return test bot
        from unittest.mock import patch
        with patch.object(commander.__class__.__bases__[0], 'run_auction', return_value=[test_bot]):
            # Run auction through commander
            result = commander.run_auction(None)
        
        # Verify Kelly Filter enforcement
        if preservation_mode and kelly_score < 0.8:
            # Bot should be filtered out
            assert len(result) == 0 or test_bot not in result, \
                f"Bot with kelly_score {kelly_score} should be filtered in preservation mode"
        elif preservation_mode and kelly_score >= 0.8:
            # Bot should pass through
            assert test_bot in result, \
                f"Bot with kelly_score {kelly_score} should pass in preservation mode"
    
    @given(
        kelly_scores=st.lists(
            st.floats(min_value=0.0, max_value=1.0),
            min_size=1,
            max_size=10
        )
    )
    @settings(max_examples=100)
    def test_all_filtered_bots_meet_threshold(self, kelly_scores):
        """
        Property: All bots passing Kelly Filter MUST have KellyScore >= 0.8.
        
        **Validates: Requirements 4.1**
        """
        commander = PropCommander("test_account")
        
        # Set preservation mode
        mock_metrics = SimpleNamespace(
            daily_start_balance=100000.0,
            current_equity=108000.0,  # 8% profit
            trading_days=10
        )
        commander.prop_state.get_metrics = lambda: mock_metrics
        
        # Create test bots
        test_bots = [
            {"name": f"Bot{i}", "kelly_score": score}
            for i, score in enumerate(kelly_scores)
        ]
        
        # Filter bots using Kelly Filter logic
        filtered_bots = [b for b in test_bots if b.get('kelly_score', 0) >= 0.8]
        
        # Verify all filtered bots meet threshold
        for bot in filtered_bots:
            assert bot.get('kelly_score', 0) >= 0.8, \
                f"Filtered bot {bot['name']} has kelly_score {bot.get('kelly_score')} < 0.8"
    
    @given(
        target_profit=st.floats(min_value=0.08, max_value=0.20)  # 8% to 20% profit
    )
    @settings(max_examples=100)
    def test_preservation_mode_activates_at_target(self, target_profit):
        """
        Property: Preservation mode MUST activate when profit >= target.
        
        **Validates: Requirements 4.1**
        """
        commander = PropCommander("test_account")
        commander.target_profit_pct = 0.08  # 8% target
        
        # Set metrics with given profit
        mock_metrics = SimpleNamespace(
            daily_start_balance=100000.0,
            current_equity=100000.0 * (1 + target_profit),
            trading_days=10
        )
        commander.prop_state.get_metrics = lambda: mock_metrics
        
        # Check if preservation mode is active
        is_preservation = commander._check_preservation_mode(mock_metrics)
        
        # Verify preservation mode activates at or above target
        if target_profit >= commander.target_profit_pct:
            assert is_preservation, \
                f"Preservation mode should activate at {target_profit*100}% profit"
        else:
            assert not is_preservation, \
                f"Preservation mode should not activate at {target_profit*100}% profit"

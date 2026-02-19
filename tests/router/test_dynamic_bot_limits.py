"""
Unit tests for DynamicBotLimiter component.

Tests the 7-tier bot limiting system including small accounts under $200.
"""

import pytest
from src.router.dynamic_bot_limits import DynamicBotLimiter


class TestGetMaxBots:
    """Test getting max bots for various account balances."""
    
    def test_tier_0_no_trading(self):
        """Test Tier 0: $0-50 should allow 0 bots (trading disabled)."""
        assert DynamicBotLimiter.get_max_bots(0) == 0
        assert DynamicBotLimiter.get_max_bots(25) == 0
        assert DynamicBotLimiter.get_max_bots(49) == 0
    
    def test_tier_1_small_account(self):
        """Test Tier 1: $50-100 should allow 1 bot."""
        assert DynamicBotLimiter.get_max_bots(50) == 1
        assert DynamicBotLimiter.get_max_bots(75) == 1
        assert DynamicBotLimiter.get_max_bots(99) == 1
    
    def test_tier_2_small_account(self):
        """Test Tier 2: $100-200 should allow 2 bots."""
        assert DynamicBotLimiter.get_max_bots(100) == 2
        assert DynamicBotLimiter.get_max_bots(150) == 2
        assert DynamicBotLimiter.get_max_bots(199) == 2
    
    def test_tier_3_medium_account(self):
        """Test Tier 3: $200-500 should allow 3 bots."""
        assert DynamicBotLimiter.get_max_bots(200) == 3
        assert DynamicBotLimiter.get_max_bots(300) == 3
        assert DynamicBotLimiter.get_max_bots(499) == 3
    
    def test_tier_4_large_account(self):
        """Test Tier 4: $500-1k should allow 5 bots."""
        assert DynamicBotLimiter.get_max_bots(500) == 5
        assert DynamicBotLimiter.get_max_bots(750) == 5
        assert DynamicBotLimiter.get_max_bots(999) == 5
    
    def test_tier_5_very_large_account(self):
        """Test Tier 5: $1k-5k should allow 10 bots."""
        assert DynamicBotLimiter.get_max_bots(1000) == 10
        assert DynamicBotLimiter.get_max_bots(2000) == 10
        assert DynamicBotLimiter.get_max_bots(4999) == 10
    
    def test_tier_6_institutional(self):
        """Test Tier 6: $5k+ should allow 20 bots."""
        assert DynamicBotLimiter.get_max_bots(5000) == 20
        assert DynamicBotLimiter.get_max_bots(10000) == 20
        assert DynamicBotLimiter.get_max_bots(100000) == 20


class TestCanAddBot:
    """Test bot addition checks with safety buffer."""
    
    def test_can_add_bot_under_limit(self):
        """Test that adding bot under limit succeeds."""
        can_add, reason = DynamicBotLimiter.can_add_bot(
            account_balance=500,
            current_bots=2
        )
        assert can_add is True
        assert "allowed" in reason.lower() or "ok" in reason.lower()
    
    def test_cannot_add_bot_at_limit(self):
        """Test that adding bot at limit fails."""
        can_add, reason = DynamicBotLimiter.can_add_bot(
            account_balance=100,
            current_bots=2
        )
        assert can_add is False
        assert "limit" in reason.lower()
    
    def test_cannot_add_bot_tier_0(self):
        """Test that tier 0 accounts cannot add bots."""
        can_add, reason = DynamicBotLimiter.can_add_bot(
            account_balance=40,
            current_bots=0
        )
        assert can_add is False
        assert "insufficient" in reason.lower() or "disabled" in reason.lower()
    
    def test_can_add_with_safety_buffer(self):
        """Test safety buffer calculation."""
        # $200 with 2 bots should require safety buffer
        can_add, reason = DynamicBotLimiter.can_add_bot(
            account_balance=200,
            current_bots=2,
            safety_buffer_multiplier=2.0
        )
        # With 2 bots at $50 each * 2 = $200 required
        # At limit, can't add more
        assert can_add is False


class TestGetTierInfo:
    """Test tier information retrieval."""
    
    def test_get_tier_info_tier_0(self):
        """Test getting tier info for tier 0."""
        info = DynamicBotLimiter.get_tier_info(25)
        assert info["tier"] == 0
        assert info["max_bots"] == 0
        assert info["range"] == (0, 50)
        assert info["trading_enabled"] is False
    
    def test_get_tier_info_tier_1(self):
        """Test getting tier info for tier 1."""
        info = DynamicBotLimiter.get_tier_info(75)
        assert info["tier"] == 1
        assert info["max_bots"] == 1
        assert info["range"] == (50, 100)
        assert info["trading_enabled"] is True
    
    def test_get_tier_info_tier_6(self):
        """Test getting tier info for tier 6."""
        info = DynamicBotLimiter.get_tier_info(10000)
        assert info["tier"] == 6
        assert info["max_bots"] == 20
        assert info["range"] == (5000, float('inf'))


class TestCalculateSafetyBuffer:
    """Test safety buffer calculations."""
    
    def test_safety_buffer_single_bot(self):
        """Test safety buffer for single bot."""
        # 1 bot * $50 * 2 = $100 required
        required = DynamicBotLimiter.calculate_safety_buffer(
            num_bots=1,
            min_capital_per_bot=50,
            safety_buffer_multiplier=2.0
        )
        assert required == 100
    
    def test_safety_buffer_multiple_bots(self):
        """Test safety buffer for multiple bots."""
        # 5 bots * $50 * 2 = $500 required
        required = DynamicBotLimiter.calculate_safety_buffer(
            num_bots=5,
            min_capital_per_bot=50,
            safety_buffer_multiplier=2.0
        )
        assert required == 500
    
    def test_safety_buffer_custom_params(self):
        """Test safety buffer with custom parameters."""
        # 3 bots * $100 * 1.5 = $450 required
        required = DynamicBotLimiter.calculate_safety_buffer(
            num_bots=3,
            min_capital_per_bot=100,
            safety_buffer_multiplier=1.5
        )
        assert required == 450


class TestGetRecommendedRiskPerBot:
    """Test recommended risk per bot calculations."""
    
    def test_risk_per_bot_medium_account(self):
        """Test risk per bot for medium account."""
        risk = DynamicBotLimiter.get_recommended_risk_per_bot(1000)
        # With 10 bots max at $1k, should distribute risk
        assert risk > 0
        assert risk <= 1.0  # Should be reasonable percentage
    
    def test_risk_per_bot_small_account(self):
        """Test risk per bot for small account."""
        risk = DynamicBotLimiter.get_recommended_risk_per_bot(100)
        # With only 2 bots, risk per bot should be conservative
        assert risk > 0
        assert risk <= 1.5
    
    def test_risk_per_bot_tier_0(self):
        """Test risk per bot for tier 0 (no trading)."""
        risk = DynamicBotLimiter.get_recommended_risk_per_bot(25)
        # No trading allowed
        assert risk == 0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_negative_balance(self):
        """Test handling of negative balance."""
        max_bots = DynamicBotLimiter.get_max_bots(-100)
        assert max_bots == 0
    
    def test_zero_balance(self):
        """Test handling of zero balance."""
        max_bots = DynamicBotLimiter.get_max_bots(0)
        assert max_bots == 0
    
    def test_exact_tier_boundary(self):
        """Test exact tier boundary values."""
        # Test exact boundaries
        assert DynamicBotLimiter.get_max_bots(50) == 1   # Tier 1 start
        assert DynamicBotLimiter.get_max_bots(100) == 2  # Tier 2 start
        assert DynamicBotLimiter.get_max_bots(200) == 3  # Tier 3 start
        assert DynamicBotLimiter.get_max_bots(500) == 5  # Tier 4 start
        assert DynamicBotLimiter.get_max_bots(1000) == 10  # Tier 5 start
        assert DynamicBotLimiter.get_max_bots(5000) == 20  # Tier 6 start


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
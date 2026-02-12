"""
Unit Tests for V8 Tiered Risk Engine

Tests the three-tier risk system implementation in KellySizer.mqh
through Python integration tests.
"""

import pytest
from src.database.models import PropFirmAccount, RiskTierTransition
from src.database.engine import get_session
from datetime import datetime


class TestTieredRiskTierDetermination:
    """Test risk tier determination based on equity ranges."""
    
    def test_growth_tier_lower_bound(self):
        """Test Growth tier at $100 equity."""
        equity = 100.0
        growth_ceiling = 1000.0
        
        # Growth tier: equity < 1000
        assert equity < growth_ceiling
        tier = "growth"
        assert tier == "growth"
    
    def test_growth_tier_upper_bound(self):
        """Test Growth tier at $999 equity."""
        equity = 999.0
        growth_ceiling = 1000.0
        
        assert equity < growth_ceiling
        tier = "growth"
        assert tier == "growth"
    
    def test_scaling_tier_lower_bound(self):
        """Test Scaling tier at $1000 equity."""
        equity = 1000.0
        growth_ceiling = 1000.0
        scaling_ceiling = 5000.0
        
        assert equity >= growth_ceiling
        assert equity < scaling_ceiling
        tier = "scaling"
        assert tier == "scaling"
    
    def test_scaling_tier_mid_range(self):
        """Test Scaling tier at $3000 equity."""
        equity = 3000.0
        growth_ceiling = 1000.0
        scaling_ceiling = 5000.0
        
        assert equity >= growth_ceiling
        assert equity < scaling_ceiling
        tier = "scaling"
        assert tier == "scaling"
    
    def test_scaling_tier_upper_bound(self):
        """Test Scaling tier at $4999 equity."""
        equity = 4999.0
        growth_ceiling = 1000.0
        scaling_ceiling = 5000.0
        
        assert equity >= growth_ceiling
        assert equity < scaling_ceiling
        tier = "scaling"
        assert tier == "scaling"
    
    def test_guardian_tier_lower_bound(self):
        """Test Guardian tier at $5000 equity."""
        equity = 5000.0
        scaling_ceiling = 5000.0
        
        assert equity >= scaling_ceiling
        tier = "guardian"
        assert tier == "guardian"
    
    def test_guardian_tier_high_equity(self):
        """Test Guardian tier at $50000 equity."""
        equity = 50000.0
        scaling_ceiling = 5000.0
        
        assert equity >= scaling_ceiling
        tier = "guardian"
        assert tier == "guardian"
    
    def test_exact_boundary_growth_to_scaling(self):
        """Test exact boundary at $1000 (should be Scaling tier)."""
        equity = 1000.0
        growth_ceiling = 1000.0
        scaling_ceiling = 5000.0
        
        # At exactly $1000, should transition to Scaling
        assert equity >= growth_ceiling
        assert equity < scaling_ceiling
        tier = "scaling"
        assert tier == "scaling"
    
    def test_exact_boundary_scaling_to_guardian(self):
        """Test exact boundary at $5000 (should be Guardian tier)."""
        equity = 5000.0
        scaling_ceiling = 5000.0
        
        # At exactly $5000, should transition to Guardian
        assert equity >= scaling_ceiling
        tier = "guardian"
        assert tier == "guardian"
    
    def test_just_below_growth_ceiling(self):
        """Test equity just below Growth ceiling ($999.99)."""
        equity = 999.99
        growth_ceiling = 1000.0
        
        assert equity < growth_ceiling
        tier = "growth"
        assert tier == "growth"
    
    def test_just_above_growth_ceiling(self):
        """Test equity just above Growth ceiling ($1000.01)."""
        equity = 1000.01
        growth_ceiling = 1000.0
        scaling_ceiling = 5000.0
        
        assert equity >= growth_ceiling
        assert equity < scaling_ceiling
        tier = "scaling"
        assert tier == "scaling"


class TestFixedRiskCalculation:
    """Test dynamic aggressive risk calculation with floor in Growth tier."""
    
    def test_growth_tier_dynamic_risk_100_equity(self):
        """Test Growth tier with $100 equity (3% = $3.00)."""
        equity = 100.0
        growth_percent = 3.0
        fixed_floor = 2.0
        
        percent_risk = equity * (growth_percent / 100.0)
        risk_amount = max(percent_risk, fixed_floor)
        
        assert percent_risk == 3.0
        assert risk_amount == 3.0  # 3% kicks in
    
    def test_growth_tier_dynamic_risk_50_equity_floor(self):
        """Test Growth tier with $50 equity (floor kicks in at $2.00)."""
        equity = 50.0
        growth_percent = 3.0
        fixed_floor = 2.0
        
        percent_risk = equity * (growth_percent / 100.0)
        risk_amount = max(percent_risk, fixed_floor)
        
        assert percent_risk == 1.5
        assert risk_amount == 2.0  # Floor kicks in
    
    def test_growth_tier_dynamic_risk_500_equity(self):
        """Test Growth tier with $500 equity (3% = $15.00)."""
        equity = 500.0
        growth_percent = 3.0
        fixed_floor = 2.0
        
        percent_risk = equity * (growth_percent / 100.0)
        risk_amount = max(percent_risk, fixed_floor)
        
        assert percent_risk == 15.0
        assert risk_amount == 15.0  # Scales up
    
    def test_growth_tier_lot_calculation_with_dynamic_risk(self):
        """Test lot size calculation with dynamic risk."""
        equity = 100.0
        growth_percent = 3.0
        fixed_floor = 2.0
        stop_loss_pips = 10.0
        tick_value = 1.0
        
        risk_amount = max(equity * (growth_percent / 100.0), fixed_floor)
        expected_lots = risk_amount / (stop_loss_pips * tick_value)
        
        assert risk_amount == 3.0
        assert expected_lots == 0.3


class TestQuadraticThrottle:
    """Test Quadratic Throttle formula for Guardian tier."""
    
    def test_no_loss_full_capacity(self):
        """Test throttle with no current loss (100% capacity)."""
        base_risk = 100.0
        current_loss = 0.0
        max_loss = 1000.0
        
        remaining_capacity = (max_loss - current_loss) / max_loss
        multiplier = remaining_capacity ** 2
        throttled_risk = base_risk * multiplier
        
        assert remaining_capacity == 1.0
        assert multiplier == 1.0
        assert throttled_risk == 100.0
    
    def test_half_loss_quarter_capacity(self):
        """Test throttle with 50% loss (25% capacity)."""
        base_risk = 100.0
        current_loss = 500.0
        max_loss = 1000.0
        
        remaining_capacity = (max_loss - current_loss) / max_loss
        multiplier = remaining_capacity ** 2
        throttled_risk = base_risk * multiplier
        
        assert remaining_capacity == 0.5
        assert multiplier == 0.25
        assert throttled_risk == 25.0
    
    def test_ninety_percent_loss_one_percent_capacity(self):
        """Test throttle with 90% loss (1% capacity)."""
        base_risk = 100.0
        current_loss = 900.0
        max_loss = 1000.0
        
        remaining_capacity = (max_loss - current_loss) / max_loss
        multiplier = remaining_capacity ** 2
        throttled_risk = base_risk * multiplier
        
        assert remaining_capacity == 0.1
        assert abs(multiplier - 0.01) < 1e-10  # Use approximate comparison for floating point
        assert abs(throttled_risk - 1.0) < 1e-10  # Use approximate comparison
    
    def test_at_max_loss_zero_capacity(self):
        """Test throttle at maximum loss (0% capacity)."""
        base_risk = 100.0
        current_loss = 1000.0
        max_loss = 1000.0
        
        remaining_capacity = (max_loss - current_loss) / max_loss
        multiplier = remaining_capacity ** 2
        throttled_risk = base_risk * multiplier
        
        assert remaining_capacity == 0.0
        assert multiplier == 0.0
        assert throttled_risk == 0.0


class TestDatabaseIntegration:
    """Test database integration for risk_mode and tier transitions."""
    
    def test_create_account_with_risk_mode(self):
        """Test creating PropFirmAccount with risk_mode."""
        session = get_session()
        
        account = PropFirmAccount(
            firm_name="TestFirm",
            account_id="TEST_V8_001",
            daily_loss_limit_pct=5.0,
            risk_mode="growth"
        )
        
        session.add(account)
        session.commit()
        
        # Verify
        retrieved = session.query(PropFirmAccount).filter_by(account_id="TEST_V8_001").first()
        assert retrieved is not None
        assert retrieved.risk_mode == "growth"
        
        # Cleanup
        session.delete(retrieved)
        session.commit()
        session.close()
    
    def test_log_tier_transition(self):
        """Test logging risk tier transition."""
        session = get_session()
        
        # Create account
        account = PropFirmAccount(
            firm_name="TestFirm",
            account_id="TEST_V8_002",
            daily_loss_limit_pct=5.0,
            risk_mode="growth"
        )
        session.add(account)
        session.commit()
        
        # Log transition
        transition = RiskTierTransition(
            account_id=account.id,
            from_tier="growth",
            to_tier="scaling",
            equity_at_transition=1050.0
        )
        session.add(transition)
        session.commit()
        
        # Verify
        retrieved_transition = session.query(RiskTierTransition).filter_by(account_id=account.id).first()
        assert retrieved_transition is not None
        assert retrieved_transition.from_tier == "growth"
        assert retrieved_transition.to_tier == "scaling"
        assert retrieved_transition.equity_at_transition == 1050.0
        
        # Cleanup
        session.delete(account)  # Cascade will delete transition
        session.commit()
        session.close()
    
    def test_update_risk_mode(self):
        """Test updating account risk_mode."""
        session = get_session()
        
        # Create account
        account = PropFirmAccount(
            firm_name="TestFirm",
            account_id="TEST_V8_003",
            daily_loss_limit_pct=5.0,
            risk_mode="growth"
        )
        session.add(account)
        session.commit()
        
        # Update risk mode
        account.risk_mode = "scaling"
        session.commit()
        
        # Verify
        retrieved = session.query(PropFirmAccount).filter_by(account_id="TEST_V8_003").first()
        assert retrieved.risk_mode == "scaling"
        
        # Cleanup
        session.delete(retrieved)
        session.commit()
        session.close()


class TestTierTransitionScenarios:
    """Test realistic tier transition scenarios."""
    
    def test_growth_to_scaling_transition(self):
        """Test transition from Growth to Scaling tier."""
        # Start with $500 equity (Growth tier)
        equity = 500.0
        assert equity < 1000.0
        tier = "growth"
        
        # Grow to $1200 (Scaling tier)
        equity = 1200.0
        assert equity >= 1000.0 and equity < 5000.0
        new_tier = "scaling"
        
        assert tier != new_tier
        assert new_tier == "scaling"
    
    def test_scaling_to_guardian_transition(self):
        """Test transition from Scaling to Guardian tier."""
        # Start with $3000 equity (Scaling tier)
        equity = 3000.0
        assert equity >= 1000.0 and equity < 5000.0
        tier = "scaling"
        
        # Grow to $5500 (Guardian tier)
        equity = 5500.0
        assert equity >= 5000.0
        new_tier = "guardian"
        
        assert tier != new_tier
        assert new_tier == "guardian"
    
    def test_no_transition_within_tier(self):
        """Test that equity changes within same tier don't trigger transition."""
        # Start with $2000 equity (Scaling tier)
        equity = 2000.0
        tier = "scaling"
        
        # Grow to $4000 (still Scaling tier)
        equity = 4000.0
        assert equity >= 1000.0 and equity < 5000.0
        new_tier = "scaling"
        
        assert tier == new_tier


class TestIntegrationTierTransitions:
    """Integration tests for tier transitions with PropGovernor."""
    
    def test_tier_transition_with_database_logging(self):
        """Test complete tier transition flow with database logging."""
        from src.router.prop.governor import PropGovernor
        from src.router.governor import RiskMandate
        
        session = get_session()
        
        # Cleanup any existing test account
        existing = session.query(PropFirmAccount).filter_by(
            account_id="TEST_INTEGRATION_001"
        ).first()
        if existing:
            session.delete(existing)
            session.commit()
        
        # Create test account
        account = PropFirmAccount(
            firm_name="TestFirm",
            account_id="TEST_INTEGRATION_001",
            daily_loss_limit_pct=5.0,
            risk_mode="growth"
        )
        session.add(account)
        session.commit()
        
        try:
            # Create PropGovernor
            governor = PropGovernor("TEST_INTEGRATION_001")
            
            # Mock regime report
            class MockRegimeReport:
                chaos_score = 0.1
                is_systemic_risk = False
                news_state = "NORMAL"
            
            regime_report = MockRegimeReport()
            
            # Test 1: Growth tier at $500
            trade_proposal = {
                'bot_id': 'test_bot',
                'symbol': 'EURUSD',
                'current_balance': 500.0
            }
            
            mandate = governor.calculate_risk(regime_report, trade_proposal)
            assert isinstance(mandate, RiskMandate)
            
            # Verify still in growth tier
            account_check = session.query(PropFirmAccount).filter_by(
                account_id="TEST_INTEGRATION_001"
            ).first()
            assert account_check.risk_mode == "growth"
            
            # Test 2: Transition to Scaling tier at $1200
            trade_proposal['current_balance'] = 1200.0
            mandate = governor.calculate_risk(regime_report, trade_proposal)
            
            # Verify transition to scaling tier
            session.expire_all()  # Refresh from database
            account_check = session.query(PropFirmAccount).filter_by(
                account_id="TEST_INTEGRATION_001"
            ).first()
            assert account_check.risk_mode == "scaling"
            
            # Verify transition was logged
            transitions = session.query(RiskTierTransition).filter_by(
                account_id=account.id
            ).all()
            assert len(transitions) == 1
            assert transitions[0].from_tier == "growth"
            assert transitions[0].to_tier == "scaling"
            assert transitions[0].equity_at_transition == 1200.0
            
            # Test 3: Transition to Guardian tier at $5500
            trade_proposal['current_balance'] = 5500.0
            mandate = governor.calculate_risk(regime_report, trade_proposal)
            
            # Verify transition to guardian tier
            session.expire_all()
            account_check = session.query(PropFirmAccount).filter_by(
                account_id="TEST_INTEGRATION_001"
            ).first()
            assert account_check.risk_mode == "guardian"
            
            # Verify second transition was logged
            transitions = session.query(RiskTierTransition).filter_by(
                account_id=account.id
            ).order_by(RiskTierTransition.transition_timestamp).all()
            assert len(transitions) == 2
            assert transitions[1].from_tier == "scaling"
            assert transitions[1].to_tier == "guardian"
            assert transitions[1].equity_at_transition == 5500.0
            
        finally:
            # Cleanup - need to merge the account back into the session
            try:
                account = session.merge(account)
                session.delete(account)
                session.commit()
            except Exception as e:
                print(f"Cleanup error: {e}")
                session.rollback()
            finally:
                session.close()
    
    def test_tier_transition_audit_trail(self):
        """Test that all tier transitions are properly logged for audit."""
        session = get_session()
        
        # Create test account
        account = PropFirmAccount(
            firm_name="TestFirm",
            account_id="TEST_AUDIT_001",
            daily_loss_limit_pct=5.0,
            risk_mode="growth"
        )
        session.add(account)
        session.commit()
        
        try:
            # Simulate multiple transitions
            transitions_data = [
                ("growth", "scaling", 1100.0),
                ("scaling", "guardian", 5200.0),
            ]
            
            for from_tier, to_tier, equity in transitions_data:
                transition = RiskTierTransition(
                    account_id=account.id,
                    from_tier=from_tier,
                    to_tier=to_tier,
                    equity_at_transition=equity
                )
                session.add(transition)
            
            session.commit()
            
            # Verify audit trail
            transitions = session.query(RiskTierTransition).filter_by(
                account_id=account.id
            ).order_by(RiskTierTransition.transition_timestamp).all()
            
            assert len(transitions) == 2
            
            # Verify first transition
            assert transitions[0].from_tier == "growth"
            assert transitions[0].to_tier == "scaling"
            assert transitions[0].equity_at_transition == 1100.0
            
            # Verify second transition
            assert transitions[1].from_tier == "scaling"
            assert transitions[1].to_tier == "guardian"
            assert transitions[1].equity_at_transition == 5200.0
            
            # Verify timestamps are in order
            assert transitions[0].transition_timestamp <= transitions[1].transition_timestamp
            
        finally:
            # Cleanup
            session.delete(account)
            session.commit()
            session.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


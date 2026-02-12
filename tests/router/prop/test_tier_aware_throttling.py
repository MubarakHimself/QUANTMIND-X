"""
Test V8 Tier-Aware Throttling in PropGovernor

This test verifies that PropGovernor correctly:
1. Queries risk_mode from the database
2. Applies tier-aware throttling based on the current tier
3. Detects and logs tier transitions
4. Updates the database when tiers change

**Validates: Requirements 16.1, 16.7**
"""

import pytest
from types import SimpleNamespace
from src.router.prop.governor import PropGovernor
from src.router.governor import RiskMandate
from src.database.models import PropFirmAccount, RiskTierTransition
from src.database.engine import get_session


def make_regime_report(**kwargs):
    """Create a mock regime report for testing."""
    base = {
        "regime": "TREND_STABLE",
        "chaos_score": 0.1,
        "regime_quality": 0.9,
        "news_state": "SAFE",
        "is_systemic_risk": False
    }
    base.update(kwargs)
    return SimpleNamespace(**base)


@pytest.fixture
def test_account():
    """Create a test account in the database."""
    session = get_session()
    
    # Cleanup any existing test account
    existing = session.query(PropFirmAccount).filter_by(
        account_id="TEST_TIER_AWARE_001"
    ).first()
    if existing:
        session.delete(existing)
        session.commit()
    
    # Create test account
    account = PropFirmAccount(
        firm_name="TestFirm",
        account_id="TEST_TIER_AWARE_001",
        daily_loss_limit_pct=5.0,
        risk_mode="growth"
    )
    session.add(account)
    session.commit()
    
    yield account
    
    # Cleanup
    try:
        session = get_session()
        account = session.query(PropFirmAccount).filter_by(
            account_id="TEST_TIER_AWARE_001"
        ).first()
        if account:
            session.delete(account)
            session.commit()
        session.close()
    except Exception as e:
        print(f"Cleanup error: {e}")


class TestTierAwareThrottling:
    """Test tier-aware throttling in PropGovernor."""
    
    def test_growth_tier_no_throttle(self, test_account):
        """Test that Growth tier does not apply quadratic throttle."""
        governor = PropGovernor("TEST_TIER_AWARE_001")
        
        # Verify account is in growth tier
        assert governor._current_tier == "growth"
        
        # Create trade proposal with $500 equity (Growth tier)
        trade_proposal = {
            'bot_id': 'test_bot',
            'symbol': 'EURUSD',
            'current_balance': 500.0
        }
        
        regime_report = make_regime_report()
        mandate = governor.calculate_risk(regime_report, trade_proposal)
        
        # Growth tier should not apply throttle (returns 1.0)
        # Even if there's a loss, growth tier doesn't throttle
        assert isinstance(mandate, RiskMandate)
        # The allocation_scalar should be 1.0 (no throttle in growth tier)
        assert mandate.allocation_scalar == pytest.approx(1.0)
    
    def test_scaling_tier_no_throttle(self, test_account):
        """Test that Scaling tier does not apply quadratic throttle."""
        session = get_session()
        
        # Update account to scaling tier
        account = session.query(PropFirmAccount).filter_by(
            account_id="TEST_TIER_AWARE_001"
        ).first()
        account.risk_mode = "scaling"
        session.commit()
        session.close()
        
        governor = PropGovernor("TEST_TIER_AWARE_001")
        
        # Verify account is in scaling tier
        assert governor._current_tier == "scaling"
        
        # Create trade proposal with $3000 equity (Scaling tier)
        trade_proposal = {
            'bot_id': 'test_bot',
            'symbol': 'EURUSD',
            'current_balance': 3000.0
        }
        
        regime_report = make_regime_report()
        mandate = governor.calculate_risk(regime_report, trade_proposal)
        
        # Scaling tier should not apply throttle (returns 1.0)
        assert isinstance(mandate, RiskMandate)
        assert mandate.allocation_scalar == pytest.approx(1.0)
    
    def test_guardian_tier_applies_throttle(self, test_account):
        """Test that Guardian tier applies quadratic throttle."""
        session = get_session()
        
        # Update account to guardian tier
        account = session.query(PropFirmAccount).filter_by(
            account_id="TEST_TIER_AWARE_001"
        ).first()
        account.risk_mode = "guardian"
        session.commit()
        session.close()
        
        governor = PropGovernor("TEST_TIER_AWARE_001")
        governor.prop_state.daily_start_balance = 10000.0
        
        # Verify account is in guardian tier
        assert governor._current_tier == "guardian"
        
        # Create trade proposal with $9800 equity (2% loss)
        trade_proposal = {
            'bot_id': 'test_bot',
            'symbol': 'EURUSD',
            'current_balance': 9800.0
        }
        
        regime_report = make_regime_report()
        mandate = governor.calculate_risk(regime_report, trade_proposal)
        
        # Guardian tier should apply quadratic throttle
        assert isinstance(mandate, RiskMandate)
        # With 2% loss and 4% effective limit, throttle should be < 1.0
        assert 0.0 < mandate.allocation_scalar < 1.0
        assert mandate.risk_mode == "THROTTLED"
        assert "Throttle" in (mandate.notes or "")
    
    def test_tier_transition_detection_growth_to_scaling(self, test_account):
        """Test that tier transitions are detected and logged."""
        governor = PropGovernor("TEST_TIER_AWARE_001")
        
        # Verify starting in growth tier
        assert governor._current_tier == "growth"
        
        # Create trade proposal that triggers transition to scaling
        trade_proposal = {
            'bot_id': 'test_bot',
            'symbol': 'EURUSD',
            'current_balance': 1200.0  # Above $1000 threshold
        }
        
        regime_report = make_regime_report()
        mandate = governor.calculate_risk(regime_report, trade_proposal)
        
        # Verify tier transition occurred
        assert governor._current_tier == "scaling"
        
        # Verify transition was logged in database
        session = get_session()
        account = session.query(PropFirmAccount).filter_by(
            account_id="TEST_TIER_AWARE_001"
        ).first()
        
        # Check account risk_mode was updated
        assert account.risk_mode == "scaling"
        
        # Check transition was logged
        transitions = session.query(RiskTierTransition).filter_by(
            account_id=account.id
        ).all()
        
        assert len(transitions) == 1
        assert transitions[0].from_tier == "growth"
        assert transitions[0].to_tier == "scaling"
        assert transitions[0].equity_at_transition == 1200.0
        
        session.close()
    
    def test_tier_transition_detection_scaling_to_guardian(self, test_account):
        """Test transition from Scaling to Guardian tier."""
        session = get_session()
        
        # Set account to scaling tier
        account = session.query(PropFirmAccount).filter_by(
            account_id="TEST_TIER_AWARE_001"
        ).first()
        account.risk_mode = "scaling"
        session.commit()
        session.close()
        
        governor = PropGovernor("TEST_TIER_AWARE_001")
        
        # Verify starting in scaling tier
        assert governor._current_tier == "scaling"
        
        # Create trade proposal that triggers transition to guardian
        trade_proposal = {
            'bot_id': 'test_bot',
            'symbol': 'EURUSD',
            'current_balance': 5500.0  # Above $5000 threshold
        }
        
        regime_report = make_regime_report()
        mandate = governor.calculate_risk(regime_report, trade_proposal)
        
        # Verify tier transition occurred
        assert governor._current_tier == "guardian"
        
        # Verify transition was logged in database
        session = get_session()
        account = session.query(PropFirmAccount).filter_by(
            account_id="TEST_TIER_AWARE_001"
        ).first()
        
        # Check account risk_mode was updated
        assert account.risk_mode == "guardian"
        
        # Check transition was logged
        transitions = session.query(RiskTierTransition).filter_by(
            account_id=account.id
        ).order_by(RiskTierTransition.transition_timestamp).all()
        
        assert len(transitions) == 1
        assert transitions[0].from_tier == "scaling"
        assert transitions[0].to_tier == "guardian"
        assert transitions[0].equity_at_transition == 5500.0
        
        session.close()
    
    def test_no_transition_within_same_tier(self, test_account):
        """Test that equity changes within same tier don't trigger transitions."""
        governor = PropGovernor("TEST_TIER_AWARE_001")
        
        # Verify starting in growth tier
        assert governor._current_tier == "growth"
        
        # Create multiple trade proposals within growth tier
        for equity in [200.0, 500.0, 800.0]:
            trade_proposal = {
                'bot_id': 'test_bot',
                'symbol': 'EURUSD',
                'current_balance': equity
            }
            
            regime_report = make_regime_report()
            mandate = governor.calculate_risk(regime_report, trade_proposal)
            
            # Should still be in growth tier
            assert governor._current_tier == "growth"
        
        # Verify no transitions were logged
        session = get_session()
        account = session.query(PropFirmAccount).filter_by(
            account_id="TEST_TIER_AWARE_001"
        ).first()
        
        transitions = session.query(RiskTierTransition).filter_by(
            account_id=account.id
        ).all()
        
        assert len(transitions) == 0
        
        session.close()
    
    def test_database_query_on_initialization(self, test_account):
        """Test that PropGovernor queries risk_mode from database on initialization."""
        session = get_session()
        
        # Set account to guardian tier in database
        account = session.query(PropFirmAccount).filter_by(
            account_id="TEST_TIER_AWARE_001"
        ).first()
        account.risk_mode = "guardian"
        session.commit()
        session.close()
        
        # Create new governor instance
        governor = PropGovernor("TEST_TIER_AWARE_001")
        
        # Verify it loaded the correct tier from database
        assert governor._current_tier == "guardian"
    
    def test_tier_aware_throttle_with_loss(self, test_account):
        """Test tier-aware throttling with different loss scenarios."""
        session = get_session()
        
        # Set account to guardian tier
        account = session.query(PropFirmAccount).filter_by(
            account_id="TEST_TIER_AWARE_001"
        ).first()
        account.risk_mode = "guardian"
        session.commit()
        session.close()
        
        governor = PropGovernor("TEST_TIER_AWARE_001")
        governor.prop_state.daily_start_balance = 10000.0
        
        # Test different loss scenarios
        test_cases = [
            (10000.0, 1.0),   # No loss: full capacity
            (9800.0, 0.75),   # 2% loss: 75% capacity (quadratic)
            (9600.0, 0.0),    # 4% loss: 0% capacity (at effective limit)
        ]
        
        for current_balance, expected_min_throttle in test_cases:
            trade_proposal = {
                'bot_id': 'test_bot',
                'symbol': 'EURUSD',
                'current_balance': current_balance
            }
            
            regime_report = make_regime_report()
            mandate = governor.calculate_risk(regime_report, trade_proposal)
            
            if expected_min_throttle == 0.0:
                assert mandate.allocation_scalar == 0.0
            else:
                # Allow some tolerance for floating point comparison
                assert mandate.allocation_scalar >= expected_min_throttle - 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

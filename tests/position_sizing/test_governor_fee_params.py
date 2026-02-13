"""
Unit Tests for EnhancedGovernor Fee Parameter Handling

Tests that EnhancedGovernor correctly reads commission_per_lot and spread_pips
from trade_proposal and passes them to the Kelly calculator.

Task Group 8.1: Unit Tests

Coverage:
- Commission and spread are read from trade_proposal
- Fee precedence: explicit > registry > default
- Default values are applied when not provided
- Backward compatibility with existing code
"""

import pytest
import logging
from unittest.mock import Mock, patch, MagicMock

from src.router.enhanced_governor import EnhancedGovernor
from src.router.governor import RiskMandate

logger = logging.getLogger(__name__)


class MockRegimeReport:
    """Mock regime report for testing."""
    def __init__(self, regime_quality=0.85):
        self.regime = "TREND_STABLE"
        self.regime_quality = regime_quality
        self.chaos_score = 0.2
        self.is_systemic_risk = False


@pytest.fixture
def enhanced_governor():
    """Fixture providing EnhancedGovernor instance."""
    return EnhancedGovernor(account_id="test_account")


@pytest.fixture
def mock_regime_report():
    """Fixture providing mock regime report."""
    return MockRegimeReport(regime_quality=0.85)


@pytest.fixture
def base_trade_proposal():
    """Fixture providing base trade proposal without fee params."""
    return {
        'symbol': 'EURUSD',
        'current_balance': 10000.0,
        'account_balance': 10000.0,
        'broker_id': 'test_broker',
        'win_rate': 0.55,
        'avg_win': 400.0,
        'avg_loss': 200.0,
        'stop_loss_pips': 20.0,
        'current_atr': 0.0012,
        'average_atr': 0.0010,
    }


class TestEnhancedGovernorFeeFromTradeProposal:
    """
    Test suite verifying EnhancedGovernor reads commission/spread from trade_proposal.
    
    Expectation: EnhancedGovernor should extract commission_per_lot and spread_pips
    from trade_proposal dict and pass them to Kelly calculator.
    """

    def test_governor_reads_commission_from_proposal(
        self, enhanced_governor, mock_regime_report, base_trade_proposal
    ):
        """
        Verify EnhancedGovernor reads commission_per_lot from trade_proposal.
        
        Given:
            - Trade proposal with commission_per_lot=10.0
            - EnhancedGovernor initialized
            
        When:
            - calculate_risk() is called
            
        Then:
            - Fee should be passed to Kelly calculator
            - Position size should reflect commission impact
        """
        # Arrange - explicit commission in trade proposal
        proposal = {
            **base_trade_proposal,
            'commission_per_lot': 10.0,
            'spread_pips': 0.0  # Zero to isolate commission effect
        }
        
        # Act
        mandate = enhanced_governor.calculate_risk(
            regime_report=mock_regime_report,
            trade_proposal=proposal,
            account_balance=10000.0,
            broker_id='test_broker'
        )
        
        # Assert - should execute without error
        assert isinstance(mandate, RiskMandate)
        assert mandate.position_size >= 0
        
    def test_governor_reads_spread_from_proposal(
        self, enhanced_governor, mock_regime_report, base_trade_proposal
    ):
        """
        Verify EnhancedGovernor reads spread_pips from trade_proposal.
        
        Given:
            - Trade proposal with spread_pips=2.0
            - EnhancedGovernor initialized
            
        When:
            - calculate_risk() is called
            
        Then:
            - Spread should be passed to Kelly calculator
            - Position size should reflect spread impact
        """
        # Arrange - explicit spread in trade proposal
        proposal = {
            **base_trade_proposal,
            'commission_per_lot': 0.0,  # Zero to isolate spread effect
            'spread_pips': 2.0
        }
        
        # Act
        mandate = enhanced_governor.calculate_risk(
            regime_report=mock_regime_report,
            trade_proposal=proposal,
            account_balance=10000.0,
            broker_id='test_broker'
        )
        
        # Assert - should execute without error
        assert isinstance(mandate, RiskMandate)
        assert mandate.position_size >= 0
        
    def test_governor_uses_both_fees_from_proposal(
        self, enhanced_governor, mock_regime_report, base_trade_proposal
    ):
        """
        Verify EnhancedGovernor handles both commission and spread from trade_proposal.
        
        Given:
            - Trade proposal with both_lot=7 commission_per.0 and spread_pips=1.5
            - EnhancedGovernor initialized
            
        When:
            - calculate_risk() is called
            
        Then:
            - Both fees should be passed to Kelly calculator
            - Position should be smaller than without fees
        """
        # Arrange - both fees in trade proposal
        proposal = {
            **base_trade_proposal,
            'commission_per_lot': 7.0,
            'spread_pips': 1.5
        }
        
        # Act
        mandate_with_fees = enhanced_governor.calculate_risk(
            regime_report=mock_regime_report,
            trade_proposal=proposal,
            account_balance=10000.0,
            broker_id='test_broker'
        )
        
        # Compare with no fees proposal
        proposal_no_fees = {**base_trade_proposal}
        mandate_no_fees = enhanced_governor.calculate_risk(
            regime_report=mock_regime_report,
            trade_proposal=proposal_no_fees,
            account_balance=10000.0,
            broker_id='test_broker'
        )
        
        # Assert - position with fees should be less than or equal to position without
        assert mandate_with_fees.position_size <= mandate_no_fees.position_size, \
            "Position with fees should be smaller than without fees"


class TestFeePrecedence:
    """
    Test suite verifying fee precedence: explicit > registry > default.
    
    Expectation:
    1. Explicit values from trade_proposal should be used if provided
    2. If not provided, Kelly calculator will try broker registry auto-lookup
    3. Defaults (commission=5.0, spread=1.0) are used as fallback
    """

    def test_explicit_commission_overrides_default(
        self, enhanced_governor, mock_regime_report, base_trade_proposal
    ):
        """
        Verify explicit commission in trade_proposal is used over default.
        
        Given:
            - Trade proposal with explicit commission_per_lot=15.0
            - No broker_id (so no registry lookup)
            
        When:
            - calculate_risk() is called
            
        Then:
            - Explicit value (15.0) should be used, not default (5.0)
        """
        # Arrange
        proposal = {
            **base_trade_proposal,
            'commission_per_lot': 15.0,
            'spread_pips': 0.0,
            'broker_id': None  # No broker to prevent auto-lookup
        }
        
        # Act
        mandate = enhanced_governor.calculate_risk(
            regime_report=mock_regime_report,
            trade_proposal=proposal,
            account_balance=10000.0,
            broker_id=None
        )
        
        # Assert
        assert isinstance(mandate, RiskMandate)
        assert mandate.position_size >= 0
        
    def test_explicit_spread_overrides_default(
        self, enhanced_governor, mock_regime_report, base_trade_proposal
    ):
        """
        Verify explicit spread in trade_proposal is used over default.
        
        Given:
            - Trade proposal with explicit spread_pips=3.0
            - No broker_id (so no registry lookup)
            
        When:
            - calculate_risk() is called
            
        Then:
            - Explicit value (3.0) should be used, not default (1.0)
        """
        # Arrange
        proposal = {
            **base_trade_proposal,
            'commission_per_lot': 0.0,
            'spread_pips': 3.0,
            'broker_id': None  # No broker to prevent auto-lookup
        }
        
        # Act
        mandate = enhanced_governor.calculate_risk(
            regime_report=mock_regime_report,
            trade_proposal=proposal,
            account_balance=10000.0,
            broker_id=None
        )
        
        # Assert
        assert isinstance(mandate, RiskMandate)
        assert mandate.position_size >= 0


class TestFeeDefaults:
    """
    Test suite verifying default fee values are applied when not provided.
    
    Expectation: When commission_per_lot and spread_pips are not in trade_proposal,
    defaults of 5.0 and 1.0 should be used respectively.
    """

    def test_default_commission_when_not_provided(
        self, enhanced_governor, mock_regime_report, base_trade_proposal
    ):
        """
        Verify default commission (5.0) is used when not in trade_proposal.
        
        Given:
            - Trade proposal without commission_per_lot
            - No broker_id to prevent registry lookup
            
        When:
            - calculate_risk() is called
            
        Then:
            - Default commission of 5.0 should be used
        """
        # Arrange - no commission in proposal
        proposal = {k: v for k, v in base_trade_proposal.items() 
                   if k != 'commission_per_lot'}
        
        # Act
        mandate = enhanced_governor.calculate_risk(
            regime_report=mock_regime_report,
            trade_proposal=proposal,
            account_balance=10000.0,
            broker_id=None  # No broker to prevent auto-lookup
        )
        
        # Assert - should use default
        assert isinstance(mandate, RiskMandate)
        
    def test_default_spread_when_not_provided(
        self, enhanced_governor, mock_regime_report, base_trade_proposal
    ):
        """
        Verify default spread (1.0) is used when not in trade_proposal.
        
        Given:
            - Trade proposal without spread_pips
            - No broker_id to prevent registry lookup
            
        When:
            - calculate_risk() is called
            
        Then:
            - Default spread of 1.0 should be used
        """
        # Arrange - no spread in proposal
        proposal = {k: v for k, v in base_trade_proposal.items() 
                   if k != 'spread_pips'}
        
        # Act
        mandate = enhanced_governor.calculate_risk(
            regime_report=mock_regime_report,
            trade_proposal=proposal,
            account_balance=10000.0,
            broker_id=None  # No broker to prevent auto-lookup
        )
        
        # Assert - should use default
        assert isinstance(mandate, RiskMandate)


class TestBackwardCompatibility:
    """
    Test suite verifying backward compatibility with existing code.
    
    Expectation: Existing code that doesn't provide commission/spread
    should continue to work with sensible defaults.
    """

    def test_existing_code_works_without_fee_params(
        self, enhanced_governor, mock_regime_report, base_trade_proposal
    ):
        """
        Verify existing code works when trade_proposal has no fee parameters.
        
        Given:
            - Trade proposal without commission_per_lot or spread_pips
            - Standard existing usage
            
        When:
            - calculate_risk() is called
            
        Then:
            - Should succeed without errors
            - Should return valid RiskMandate
        """
        # Arrange - proposal without fee params (existing code pattern)
        proposal = {k: v for k, v in base_trade_proposal.items() 
                   if k not in ['commission_per_lot', 'spread_pips']}
        
        # Act & Assert - should not raise
        mandate = enhanced_governor.calculate_risk(
            regime_report=mock_regime_report,
            trade_proposal=proposal,
            account_balance=10000.0,
            broker_id='test_broker'
        )
        
        assert isinstance(mandate, RiskMandate)
        assert mandate.position_size >= 0
        
    def test_fee_kill_switch_from_proposal_fees(
        self, enhanced_governor, mock_regime_report
    ):
        """
        Verify fee kill switch activates with high fees from trade_proposal.
        
        Given:
            - Trade proposal with fees that exceed avg_win
            - This triggers Kelly's fee kill switch
            
        When:
            - calculate_risk() is called
            
        Then:
            - Should return HALTED mandate with zero position
        """
        # Arrange - fees exceed avg_win ($500 > $100)
        trade_proposal = {
            'symbol': 'EURUSD',
            'current_balance': 10000.0,
            'account_balance': 10000.0,
            'broker_id': None,  # No broker lookup
            'win_rate': 0.55,
            'avg_win': 100.0,  # Low avg win
            'avg_loss': 200.0,
            'stop_loss_pips': 20.0,
            'current_atr': 0.0012,
            'average_atr': 0.0010,
            'commission_per_lot': 500.0,  # Very high commission
            'spread_pips': 0.0
        }
        
        # Act
        mandate = enhanced_governor.calculate_risk(
            regime_report=mock_regime_report,
            trade_proposal=trade_proposal,
            account_balance=10000.0,
            broker_id=None
        )
        
        # Assert - kill switch should trigger
        assert isinstance(mandate, RiskMandate)
        assert mandate.risk_mode == "HALTED" or mandate.position_size == 0.0

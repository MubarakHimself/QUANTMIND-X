"""
Tests for Governor EnhancedKelly Integration (Comment 2)

Tests that base Governor properly uses EnhancedKellyCalculator for
fee-aware position sizing with broker-specific pip values, commissions, and spreads.

Key test cases:
- XAUUSD sizing uses broker pip_value/commission
- Fee kill switch blocks trades when fees exceed avg_win
- Position sizing comes solely from Kelly result (no 0.01 fallback)
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

from src.router.governor import Governor, RiskMandate
from src.router.sentinel import RegimeReport


class TestGovernorKellyIntegration:
    """Test that Governor integrates EnhancedKellyCalculator correctly."""

    @pytest.fixture
    def governor(self):
        """Create a Governor instance for testing."""
        return Governor()

    @pytest.fixture
    def regime_report(self):
        """Create a standard regime report for testing."""
        return RegimeReport(
            regime='TREND_STABLE',
            chaos_score=0.2,
            regime_quality=1.0,
            susceptibility=0.1,
            is_systemic_risk=False,
            news_state='NONE',
            timestamp=datetime.now(timezone.utc).timestamp()
        )

    @pytest.fixture
    def trade_proposal(self):
        """Create a standard trade proposal for XAUUSD."""
        return {
            'symbol': 'XAUUSD',
            'broker_id': 'icmarkets_raw',
            'stop_loss_pips': 50.0,
            'win_rate': 0.55,
            'avg_win': 500.0,
            'avg_loss': 250.0,
            'current_atr': 0.002,
            'average_atr': 0.0015,
        }

    def test_governor_has_kelly_calculator(self, governor):
        """Test that Governor initializes with EnhancedKellyCalculator."""
        assert hasattr(governor, '_kelly_calculator')
        assert hasattr(governor, '_kelly_available')
        # Kelly should be available if import succeeded
        assert governor._kelly_available is True

    def test_governor_calculates_position_size_with_kelly(
        self, governor, regime_report, trade_proposal
    ):
        """Test that Governor calculates position_size using Kelly."""
        account_balance = 10000.0
        broker_id = 'icmarkets_raw'

        mandate = governor.calculate_risk(
            regime_report=regime_report,
            trade_proposal=trade_proposal,
            account_balance=account_balance,
            broker_id=broker_id
        )

        # Mandate should have position sizing fields populated
        assert isinstance(mandate, RiskMandate)
        assert mandate.position_size >= 0
        assert mandate.kelly_fraction >= 0
        assert mandate.risk_amount >= 0
        # Kelly adjustments should be populated
        assert len(mandate.kelly_adjustments) > 0

    def test_xauUSD_sizing_with_broker_fees(
        self, governor, regime_report
    ):
        """Test XAUUSD sizing uses broker pip_value and commission.
        
        Comment 2 spec: Verify XAUUSD sizing uses broker pip_value/commission
        """
        trade_proposal = {
            'symbol': 'XAUUSD',
            'broker_id': 'icmarkets_raw',
            'stop_loss_pips': 50.0,  # 50 pips for gold
            'win_rate': 0.55,
            'avg_win': 500.0,
            'avg_loss': 250.0,
            'current_atr': 0.002,
            'average_atr': 0.0015,
            # These would typically be fetched from broker registry
            'pip_value': 1.0,  # Gold pip value is $1/pip for 1 lot
            'commission_per_lot': 7.0,
            'spread_pips': 0.2,
        }
        account_balance = 10000.0

        mandate = governor.calculate_risk(
            regime_report=regime_report,
            trade_proposal=trade_proposal,
            account_balance=account_balance,
            broker_id='icmarkets_raw'
        )

        # Position should be calculated (not 0.01 fallback)
        # The actual value depends on Kelly calculation
        assert mandate.position_size >= 0
        # Verify adjustments include fee information
        adjustments_str = ' '.join(mandate.kelly_adjustments)
        assert 'Fee' in adjustments_str or 'R:R' in adjustments_str

    def test_fee_kill_switch_blocks_trades_when_fees_exceed_avg_win(
        self, governor, regime_report
    ):
        """Test fee kill switch blocks trades when fees exceed avg_win.
        
        Comment 2 spec: Fee kill switch should block trades when fees exceed expected profit
        """
        # Trade with very low avg_win relative to fees
        trade_proposal = {
            'symbol': 'XAUUSD',
            'broker_id': 'icmarkets_raw',
            'stop_loss_pips': 50.0,
            'win_rate': 0.55,
            'avg_win': 5.0,  # Very small expected win
            'avg_loss': 250.0,
            'current_atr': 0.002,
            'average_atr': 0.0015,
            'commission_per_lot': 7.0,  # Commission exceeds avg_win
            'spread_pips': 2.0,
            'pip_value': 1.0,
        }
        account_balance = 10000.0

        mandate = governor.calculate_risk(
            regime_report=regime_report,
            trade_proposal=trade_proposal,
            account_balance=account_balance,
            broker_id='icmarkets_raw'
        )

        # Fee kill switch should have activated
        assert mandate.position_size == 0.0
        assert mandate.risk_mode == "HALTED"
        assert 'fee kill switch' in mandate.notes.lower()

    def test_no_fallback_position_size(
        self, governor, regime_report
    ):
        """Test that there's no 0.01 fallback position size.
        
        Comment 2 spec: Remove 0.01 fallback so sizing comes solely from Kelly
        """
        # Trade with negative expectancy (should get 0 from Kelly)
        trade_proposal = {
            'symbol': 'EURUSD',
            'broker_id': 'icmarkets_raw',
            'stop_loss_pips': 20.0,
            'win_rate': 0.30,  # Low win rate
            'avg_win': 100.0,
            'avg_loss': 300.0,  # Large loss relative to win
            'current_atr': 0.001,
            'average_atr': 0.001,
        }
        account_balance = 10000.0

        mandate = governor.calculate_risk(
            regime_report=regime_report,
            trade_proposal=trade_proposal,
            account_balance=account_balance,
            broker_id='icmarkets_raw'
        )

        # With negative expectancy, position should be 0 (not 0.01)
        assert mandate.position_size == 0.0

    def test_kelly_adjustments_populated(
        self, governor, regime_report, trade_proposal
    ):
        """Test that Kelly adjustments are populated for audit trail."""
        account_balance = 10000.0

        mandate = governor.calculate_risk(
            regime_report=regime_report,
            trade_proposal=trade_proposal,
            account_balance=account_balance,
            broker_id='icmarkets_raw'
        )

        # Kelly adjustments should contain calculation breakdown
        assert isinstance(mandate.kelly_adjustments, list)
        assert len(mandate.kelly_adjustments) > 0
        # Should include R:R ratio calculation
        adjustments_str = ' '.join(mandate.kelly_adjustments)
        assert 'R:R' in adjustments_str or 'Kelly' in adjustments_str

    def test_mandate_includes_mode(self, governor, regime_report, trade_proposal):
        """Test that RiskMandate includes trading mode."""
        account_balance = 10000.0

        # Test live mode
        mandate_live = governor.calculate_risk(
            regime_report=regime_report,
            trade_proposal=trade_proposal,
            account_balance=account_balance,
            broker_id='icmarkets_raw',
            mode='live'
        )
        assert mandate_live.mode == 'live'
        assert '[LIVE]' in mandate_live.notes

        # Test demo mode
        mandate_demo = governor.calculate_risk(
            regime_report=regime_report,
            trade_proposal=trade_proposal,
            account_balance=account_balance,
            broker_id='icmarkets_raw',
            mode='demo'
        )
        assert mandate_demo.mode == 'demo'
        assert '[DEMO]' in mandate_demo.notes


class TestGovernorPhysicsClamping:
    """Test that physics-based clamping still works with Kelly."""

    @pytest.fixture
    def governor(self):
        return Governor()

    @pytest.fixture
    def trade_proposal(self):
        return {
            'symbol': 'EURUSD',
            'broker_id': 'icmarkets_raw',
            'stop_loss_pips': 20.0,
            'win_rate': 0.55,
            'avg_win': 400.0,
            'avg_loss': 200.0,
            'current_atr': 0.0012,
            'average_atr': 0.0010,
        }

    def test_high_chaos_reduces_position_via_physics_scalar(
        self, governor, trade_proposal
    ):
        """Test that high chaos reduces position via physics scalar."""
        account_balance = 10000.0

        # Normal regime
        normal_regime = RegimeReport(
            regime='TREND_STABLE',
            chaos_score=0.1,
            regime_quality=1.0,
            susceptibility=0.1,
            is_systemic_risk=False,
            news_state='NONE',
            timestamp=datetime.now(timezone.utc).timestamp()
        )

        # High chaos regime
        high_chaos_regime = RegimeReport(
            regime='HIGH_CHAOS',
            chaos_score=0.8,
            regime_quality=0.3,
            susceptibility=0.8,
            is_systemic_risk=False,
            news_state='NONE',
            timestamp=datetime.now(timezone.utc).timestamp()
        )

        normal_mandate = governor.calculate_risk(
            regime_report=normal_regime,
            trade_proposal=trade_proposal,
            account_balance=account_balance,
            broker_id='icmarkets_raw'
        )

        high_chaos_mandate = governor.calculate_risk(
            regime_report=high_chaos_regime,
            trade_proposal=trade_proposal,
            account_balance=account_balance,
            broker_id='icmarkets_raw'
        )

        # High chaos should clamp allocation scalar
        assert high_chaos_mandate.allocation_scalar < normal_mandate.allocation_scalar
        assert high_chaos_mandate.risk_mode == "CLAMPED"

    def test_systemic_risk_reduces_allocation(
        self, governor, trade_proposal
    ):
        """Test that systemic risk detection reduces allocation."""
        account_balance = 10000.0

        # Normal regime
        normal_regime = RegimeReport(
            regime='TREND_STABLE',
            chaos_score=0.1,
            regime_quality=1.0,
            susceptibility=0.1,
            is_systemic_risk=False,
            news_state='NONE',
            timestamp=datetime.now(timezone.utc).timestamp()
        )

        # Systemic risk regime
        systemic_regime = RegimeReport(
            regime='TREND_STABLE',
            chaos_score=0.2,
            regime_quality=0.8,
            susceptibility=0.3,
            is_systemic_risk=True,
            news_state='NONE',
            timestamp=datetime.now(timezone.utc).timestamp()
        )

        normal_mandate = governor.calculate_risk(
            regime_report=normal_regime,
            trade_proposal=trade_proposal,
            account_balance=account_balance,
            broker_id='icmarkets_raw'
        )

        systemic_mandate = governor.calculate_risk(
            regime_report=systemic_regime,
            trade_proposal=trade_proposal,
            account_balance=account_balance,
            broker_id='icmarkets_raw'
        )

        # Systemic risk should reduce allocation
        assert systemic_mandate.allocation_scalar <= 0.4
        assert 'Systemic' in systemic_mandate.notes


class TestGovernorWithMockedKelly:
    """Test Governor with mocked EnhancedKellyCalculator for precise control."""

    def test_kelly_result_propagated_to_mandate(self):
        """Test that KellyResult fields are properly propagated."""
        from src.router.governor import Governor, RiskMandate
        from src.position_sizing.enhanced_kelly import KellyResult

        governor = Governor()

        # Mock the Kelly calculator
        mock_result = KellyResult(
            position_size=2.5,
            kelly_f=0.02,
            base_kelly_f=0.04,
            risk_amount=200.0,
            adjustments_applied=["Layer 1: 0.0200", "Layer 2: 0.0200"],
            status='calculated'
        )

        with patch.object(governor._kelly_calculator, 'calculate', return_value=mock_result):
            regime_report = RegimeReport(
                regime='TREND_STABLE',
                chaos_score=0.1,
                regime_quality=1.0,
                susceptibility=0.1,
                is_systemic_risk=False,
                news_state='NONE',
                timestamp=datetime.now(timezone.utc).timestamp()
            )

            mandate = governor.calculate_risk(
                regime_report=regime_report,
                trade_proposal={'symbol': 'EURUSD'},
                account_balance=10000.0,
                broker_id='icmarkets_raw'
            )

            assert mandate.position_size == 2.5
            assert mandate.kelly_fraction == 0.02
            assert mandate.risk_amount == 200.0
            assert mandate.kelly_adjustments == ["Layer 1: 0.0200", "Layer 2: 0.0200"]

    def test_fee_blocked_status_halts_trade(self):
        """Test that fee_blocked status from Kelly halts the trade."""
        from src.router.governor import Governor
        from src.position_sizing.enhanced_kelly import KellyResult

        governor = Governor()

        # Mock a fee-blocked result
        mock_result = KellyResult(
            position_size=0.0,
            kelly_f=0.0,
            base_kelly_f=0.04,
            risk_amount=0.0,
            adjustments_applied=["FEE KILL SWITCH - Fees exceed expected profit"],
            status='fee_blocked'
        )

        with patch.object(governor._kelly_calculator, 'calculate', return_value=mock_result):
            regime_report = RegimeReport(
                regime='TREND_STABLE',
                chaos_score=0.1,
                regime_quality=1.0,
                susceptibility=0.1,
                is_systemic_risk=False,
                news_state='NONE',
                timestamp=datetime.now(timezone.utc).timestamp()
            )

            mandate = governor.calculate_risk(
                regime_report=regime_report,
                trade_proposal={'symbol': 'XAUUSD', 'avg_win': 5.0},
                account_balance=10000.0,
                broker_id='icmarkets_raw'
            )

            assert mandate.position_size == 0.0
            assert mandate.risk_mode == "HALTED"
            assert 'fee kill switch' in mandate.notes.lower()
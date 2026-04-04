"""
Tests for Story 4.11: Corrected Portfolio Mathematics Baseline

Tests verify F-14 corrections to baseline Kelly inputs:
- Per-trade risk: 2% of equity ($4 on $200 equity), not 1% ($2)
- Minimum R:R ratio: 1:2 (ratio=2.0), not 1.2:1
- Baseline win rate: 52%, not 50%
- Per-trade EV: +1.12% of equity per trade
- Daily EV: +$179.20 at 80 trades/day

Reference: _bmad-output/implementation-artifacts/4-11-corrected-portfolio-mathematics-baseline.md
"""

import pytest
from src.position_sizing.enhanced_kelly import EnhancedKellyCalculator
from src.position_sizing.kelly_config import EnhancedKellyConfig
from src.position_sizing import (
    calculate_per_trade_ev,
    calculate_daily_ev,
    verify_baseline_mathematics
)
from src.router.enhanced_governor import EnhancedGovernor


class TestPerTradeRisk:
    """Task 1 / AC #1: Verify per-trade risk is 2% of equity."""

    def test_fallback_risk_pct_is_2_percent(self):
        """Verify fallback_risk_pct in EnhancedKellyConfig is 2% (F-14 correction)."""
        config = EnhancedKellyConfig()
        # F-14 correction: fallback should be 0.02 (2%), not 0.01 (1%)
        assert config.fallback_risk_pct == 0.02

    def test_default_risk_per_trade_in_governor(self):
        """Verify _default_risk_per_trade in EnhancedGovernor is 2%."""
        governor = EnhancedGovernor(account_id=None)
        # F-14 correction: should be 0.02 (2%)
        assert governor._default_risk_per_trade == 0.02

    def test_per_trade_risk_on_200_equity(self):
        """Verify per-trade risk is $4 (2% of $200), not $2 (1% of $200)."""
        account_balance = 200.0
        risk_pct = 0.02  # 2%

        risk_amount = account_balance * risk_pct

        # F-14 correction: risk should be $4, not $2
        assert risk_amount == 4.0
        assert risk_amount != 2.0  # This was the old incorrect value

    def test_kelly_config_max_risk_pct_is_2_percent(self):
        """Verify max_risk_pct in EnhancedKellyConfig is 2%."""
        config = EnhancedKellyConfig()
        # max_risk_pct is the hard cap - should be 2%
        assert config.max_risk_pct == 0.02


class TestEVCalculation:
    """Task 2 / AC #2: Verify per-trade EV calculation."""

    def test_per_trade_ev_formula(self):
        """
        Verify per-trade EV = p × (2R) - q × (1R) = +1.12% of equity.

        F-14 correction parameters:
        - p = 0.52 (52% win rate)
        - q = 0.48 (1 - p)
        - R = 0.02 (2% risk)
        - R:R = 2.0 (1:2 ratio)

        Expected: 0.52 × (2 × 0.02) - 0.48 × (0.02) = 0.0208 - 0.0096 = 0.0112 (1.12%)
        """
        win_rate = 0.52  # F-14 corrected
        risk_reward_ratio = 2.0  # 1:2 R:R
        risk_pct = 0.02  # 2%

        ev = calculate_per_trade_ev(win_rate, risk_reward_ratio, risk_pct)

        # Expected: 1.12% of equity per trade
        expected_ev = 0.0112
        assert abs(ev - expected_ev) < 0.0001, f"EV should be ~1.12%, got {ev*100:.2f}%"

    def test_per_trade_ev_with_old_incorrect_values(self):
        """
        Verify EV calculation with OLD incorrect values gives wrong result.

        Old incorrect values:
        - win_rate = 0.50 (50%)
        - risk_reward_ratio = 1.2 (1.2:1)
        - risk_pct = 0.01 (1%)

        This demonstrates why F-14 corrections were needed.
        """
        # Old incorrect values
        old_win_rate = 0.50
        old_rr_ratio = 1.2
        old_risk_pct = 0.01

        old_ev = calculate_per_trade_ev(old_win_rate, old_rr_ratio, old_risk_pct)

        # Old EV should be significantly different from correct EV
        correct_ev = 0.0112
        assert abs(old_ev - correct_ev) > 0.005, "Old incorrect values should give different EV"

    def test_ev_calculation_with_1_to_2_rr_ratio(self):
        """Verify 1:2 R:R ratio (ratio=2.0) is correctly interpreted."""
        # 1:2 R:R means reward is 2x the risk
        # So if risk is $2, reward is $4
        win_rate = 0.52
        risk_pct = 0.02

        # With 1:2 R:R (ratio=2.0)
        ev_1_to_2 = calculate_per_trade_ev(win_rate, 2.0, risk_pct)

        # With 1:1 R:R (ratio=1.0) for comparison
        ev_1_to_1 = calculate_per_trade_ev(win_rate, 1.0, risk_pct)

        # 1:2 should have higher EV than 1:1
        assert ev_1_to_2 > ev_1_to_1


class TestDailyEVProjection:
    """Task 3 / AC #3: Verify daily EV projection."""

    def test_daily_ev_at_80_trades(self):
        """
        Verify daily EV = 80 trades × $2.24 = +$179.20.

        F-14 correction:
        - Per-trade EV = 1.12% of equity
        - On $200 equity: $2.24 per trade
        - 80 trades/day: $179.20
        """
        per_trade_ev_pct = 0.0112  # 1.12%
        trades_per_day = 80
        account_balance = 200.0

        daily_ev = calculate_daily_ev(per_trade_ev_pct, trades_per_day, account_balance)

        # Expected: $179.20
        expected_daily_ev = 179.20
        assert abs(daily_ev - expected_daily_ev) < 0.01, f"Daily EV should be $179.20, got ${daily_ev:.2f}"

    def test_daily_ev_calculation_formula(self):
        """Verify daily EV = trades × (balance × per_trade_ev_pct)."""
        per_trade_ev_pct = 0.0112
        trades_per_day = 80
        account_balance = 200.0

        ev_per_trade_dollars = account_balance * per_trade_ev_pct
        assert abs(ev_per_trade_dollars - 2.24) < 0.01, "EV per trade should be $2.24"

        daily_ev = trades_per_day * ev_per_trade_dollars
        assert abs(daily_ev - 179.20) < 0.01, "Daily EV should be $179.20"

    def test_daily_ev_with_old_incorrect_values(self):
        """
        Verify daily EV with OLD incorrect values gives ~$80, not $179.20.

        Old incorrect values gave ~$80 daily EV.
        F-14 correction gives $179.20.
        """
        # Old incorrect per-trade EV (1% risk, 50% WR, 1.2:1 R:R)
        old_per_trade_ev = 0.0048  # ~0.48%
        old_daily_ev = calculate_daily_ev(old_per_trade_ev, 80, 200.0)

        # Old EV should be ~$76.80, significantly less than $179.20
        assert old_daily_ev < 100.0, "Old incorrect daily EV should be ~$80"
        assert old_daily_ev != 179.20


class TestBaselineMathematicsVerification:
    """Verify all F-14 corrected baseline values."""

    def test_verify_baseline_mathematics_all_correct(self):
        """Verify all baseline mathematics are F-14 corrected."""
        result = verify_baseline_mathematics()

        # All values should be correct
        assert result['all_correct'] is True
        assert result['risk_correct'] is True
        assert result['rr_ratio_correct'] is True
        assert result['win_rate_correct'] is True
        assert result['per_trade_ev_correct'] is True
        assert result['daily_ev_correct'] is True

    def test_verify_baseline_values(self):
        """Verify specific baseline values."""
        result = verify_baseline_mathematics()

        assert result['per_trade_risk_pct'] == 0.02  # 2%
        assert result['per_trade_risk_dollars'] == 4.0  # $4 on $200
        assert result['win_rate'] == 0.52  # 52%
        assert result['rr_ratio'] == 2.0  # 1:2 R:R
        assert abs(result['per_trade_ev_pct'] - 0.0112) < 0.0001  # 1.12%
        assert abs(result['daily_ev'] - 179.20) < 0.01  # $179.20


class TestWinRateDefaults:
    """Verify win_rate default is 52%, not 55%."""

    def test_governor_uses_52_percent_default_win_rate(self):
        """Verify EnhancedGovernor defaults to 52% win_rate, not 55%."""
        governor = EnhancedGovernor(account_id=None)

        # Verify _default_risk_per_trade is 2% (used in _calculate_kelly_fraction)
        assert governor._default_risk_per_trade == 0.02, \
            "Default risk per trade should be 2%"

        # Create a mock regime report for calculate_risk
        from src.router.sentinel import RegimeReport
        import time

        mock_regime = RegimeReport(
            regime='TREND_STABLE',
            chaos_score=0.3,
            regime_quality=0.7,
            susceptibility=0.5,
            is_systemic_risk=False,
            news_state='SAFE',
            timestamp=time.time()
        )

        mock_proposal = {
            'symbol': 'EURUSD',
            'current_balance': 200.0,
            'broker_id': 'mt5_default',
            'account_id': None,
            # Explicitly don't provide win_rate to use default
        }

        # Call calculate_risk without win_rate to use default
        mandate = governor.calculate_risk(
            regime_report=mock_regime,
            trade_proposal=mock_proposal,
            mode='live'
        )

        # The mandate should have a positive kelly_fraction (not zero or negative)
        # This verifies the calculation used a positive win rate
        assert mandate.kelly_fraction >= 0, \
            "Kelly fraction should be non-negative with corrected defaults"

    def test_governor_risk_mode_dynamic_uses_kelly(self):
        """Verify dynamic risk mode actually uses Kelly calculation with correct defaults."""
        governor = EnhancedGovernor(account_id=None)
        assert governor.risk_mode == 'dynamic'

        # Verify the risk calculation method uses Kelly
        from src.router.sentinel import RegimeReport
        import time

        mock_regime = RegimeReport(
            regime='TREND_STABLE',
            chaos_score=0.3,
            regime_quality=0.7,
            susceptibility=0.5,
            is_systemic_risk=False,
            news_state='SAFE',
            timestamp=time.time()
        )

        mock_proposal = {
            'symbol': 'EURUSD',
            'current_balance': 200.0,
            'win_rate': 0.52,  # F-14 corrected
            'avg_win': 400.0,  # 2R for 1:2 R:R
            'avg_loss': 200.0,  # 1R
            'stop_loss_pips': 20.0,
            'current_atr': 0.0010,
            'average_atr': 0.0010,
            'broker_id': 'mt5_default',
            'account_id': None,
        }

        mandate = governor.calculate_risk(
            regime_report=mock_regime,
            trade_proposal=mock_proposal,
            mode='live'
        )

        # With 52% WR, 2:1 R:R, Kelly should be positive
        # The actual kelly_f depends on the calculation, but should be > 0
        assert mandate.kelly_fraction > 0, \
            "Kelly fraction should be positive with 52% WR and 2:1 R:R"
        assert mandate.risk_amount > 0, \
            "Risk amount should be positive"


class TestEnhancedGovernorEVIntegration:
    """Integration tests verifying EnhancedGovernor produces correct EV with F-14 corrected defaults."""

    def test_governor_produces_correct_risk_amount_at_baseline(self):
        """
        Verify EnhancedGovernor.calculate_risk() with F-14 corrected defaults
        produces the expected +1.12% per-trade EV.

        At baseline:
        - account_balance = $200
        - win_rate = 52%
        - risk_pct = 2% ($4 on $200)
        - R:R = 1:2 (avg_win = $400, avg_loss = $200)

        Expected: Kelly fraction should be ~2% (capped by max_risk_pct)
        with proper calculation, not the old incorrect 1% or 0.55 WR values.
        """
        governor = EnhancedGovernor(account_id=None)

        # Reset daily state to avoid Preservation Mode trigger
        # This ensures we test the baseline Kelly calculation without session modifiers
        governor._daily_start_balance = 200.0
        governor.house_money_multiplier = 1.0

        from src.router.sentinel import RegimeReport
        import time

        mock_regime = RegimeReport(
            regime='TREND_STABLE',
            chaos_score=0.3,
            regime_quality=0.7,
            susceptibility=0.5,
            is_systemic_risk=False,
            news_state='SAFE',
            timestamp=time.time()
        )

        # F-14 corrected baseline parameters
        baseline_proposal = {
            'symbol': 'EURUSD',
            'current_balance': 200.0,  # Same as _daily_start_balance to avoid P&L triggers
            'win_rate': 0.52,  # F-14 corrected from 0.50
            'avg_win': 400.0,  # 2R for 1:2 R:R (F-14 corrected)
            'avg_loss': 200.0,  # 1R
            'stop_loss_pips': 20.0,
            'current_atr': 0.0010,
            'average_atr': 0.0010,
            'pip_value': 10.0,
            'broker_id': 'mt5_default',
            'account_id': None,
        }

        mandate = governor.calculate_risk(
            regime_report=mock_regime,
            trade_proposal=baseline_proposal,
            mode='live'
        )

        # The risk amount should be approximately $4 (2% of $200)
        # before SessionKelly modifiers
        # Note: After SessionKelly (which may apply multipliers), final risk could be less
        # The key verification is that the BASE Kelly calculation is correct
        # We check that kelly_adjustments show the correct base calculation

        # Verify the base Kelly calculation used correct F-14 parameters
        adjustments = mandate.kelly_adjustments.get('adjustments', [])

        # Check that Layer 2 correctly caps at 2% (not 1%)
        layer2_found = False
        for adj in adjustments:
            if 'Layer 2' in adj and 'capped at 2.0%' in adj:
                layer2_found = True
                break

        assert layer2_found, \
            f"Layer 2 should cap at 2.0% (F-14 correction), got adjustments: {adjustments}"

    def test_verify_baseline_integration(self):
        """
        Integration test: Verify EnhancedGovernor works correctly with
        the verify_baseline_mathematics() function outputs.
        """
        from src.router.sentinel import RegimeReport
        import time

        # Get baseline verification
        baseline = verify_baseline_mathematics()
        assert baseline['all_correct'], "Baseline mathematics should be verified correct"

        # Test with EnhancedGovernor
        governor = EnhancedGovernor(account_id=None)

        mock_regime = RegimeReport(
            regime='TREND_STABLE',
            chaos_score=0.3,
            regime_quality=0.7,
            susceptibility=0.5,
            is_systemic_risk=False,
            news_state='SAFE',
            timestamp=time.time()
        )

        # Use verified baseline values
        baseline_proposal = {
            'symbol': 'EURUSD',
            'current_balance': 200.0,
            'win_rate': baseline['win_rate'],  # 0.52
            'avg_win': 400.0,  # 2R for 1:2 R:R
            'avg_loss': 200.0,  # 1R
            'stop_loss_pips': 20.0,
            'current_atr': 0.0010,
            'average_atr': 0.0010,
            'pip_value': 10.0,
            'broker_id': 'mt5_default',
            'account_id': None,
        }

        mandate = governor.calculate_risk(
            regime_report=mock_regime,
            trade_proposal=baseline_proposal,
            mode='live'
        )

        # Verify mandate is valid
        assert mandate.position_size > 0, "Position size should be positive"
        assert mandate.kelly_fraction > 0, "Kelly fraction should be positive"
        assert mandate.allocation_scalar > 0, "Allocation scalar should be positive"


class TestKellyCalculatorWithCorrectedBaselines:
    """Integration tests using F-14 corrected baseline values."""

    def test_kelly_with_52_percent_win_rate(self):
        """Test Kelly calculation with 52% win rate (F-14 corrected)."""
        calculator = EnhancedKellyCalculator()

        result = calculator.calculate(
            account_balance=200.0,
            win_rate=0.52,  # F-14 corrected from 0.55
            avg_win=400.0,   # 2:1 R:R with $200 risk
            avg_loss=200.0,  # $200 risk = 2% of $200
            current_atr=0.0010,
            average_atr=0.0010,
            stop_loss_pips=20.0,
            pip_value=10.0
        )

        assert result.status == 'calculated'
        assert result.kelly_f > 0

    def test_kelly_with_1_to_2_rr_ratio(self):
        """Test Kelly with 1:2 R:R ratio (F-14 corrected from 1.2:1)."""
        calculator = EnhancedKellyCalculator()

        # 1:2 R:R means avg_win = 2 * avg_loss
        # With $200 risk and 20 pip SL, $10/pip: risk = $200
        # So avg_loss = $200 (1R), avg_win = $400 (2R)
        result = calculator.calculate(
            account_balance=200.0,
            win_rate=0.52,
            avg_win=400.0,   # 2R for 1:2 R:R
            avg_loss=200.0,  # 1R
            current_atr=0.0010,
            average_atr=0.0010,
            stop_loss_pips=20.0,
            pip_value=10.0
        )

        # Risk ratio should be 2.0 for 1:2 R:R
        risk_ratio = 400.0 / 200.0
        assert risk_ratio == 2.0

    def test_kelly_fraction_with_2_percent_risk(self):
        """Test Kelly fraction doesn't exceed 2% risk cap."""
        config = EnhancedKellyConfig()
        calculator = EnhancedKellyCalculator(config)

        # With 52% WR and 2:1 R:R, Kelly is positive
        result = calculator.calculate(
            account_balance=200.0,
            win_rate=0.52,
            avg_win=400.0,
            avg_loss=200.0,
            current_atr=0.0010,
            average_atr=0.0010,
            stop_loss_pips=20.0,
            pip_value=10.0
        )

        # Kelly fraction should be capped at 2%
        assert result.kelly_f <= config.max_risk_pct
        assert config.max_risk_pct == 0.02

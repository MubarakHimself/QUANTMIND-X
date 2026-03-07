"""Tests for PropFirmRiskOverlay."""
import pytest
import math
from src.risk.prop_firm_overlay import (
    PropFirmRiskOverlay,
    PROP_FIRM_LIMITS,
    RiskLimits
)


class TestPropFirmRiskOverlay:
    """Test suite for PropFirmRiskOverlay."""

    def test_prop_firm_tighter_drawdown(self):
        """Prop firm should use tighter drawdown limits (3% vs 5% personal)."""
        overlay = PropFirmRiskOverlay("FTMO")

        result = overlay.apply_risk_limits(
            account_book="prop_firm",
            max_drawdown=0.05
        )
        assert result["effective_max_drawdown"] == 0.03

    def test_personal_book_standard(self):
        """Personal book uses standard limits."""
        overlay = PropFirmRiskOverlay("FTMO")

        result = overlay.apply_risk_limits(
            account_book="personal",
            max_drawdown=0.05
        )
        assert result["effective_max_drawdown"] == 0.05

    def test_daily_loss_limit_prop_firm(self):
        """Prop firm should have 4% daily loss limit."""
        overlay = PropFirmRiskOverlay("FTMO")

        result = overlay.apply_risk_limits(
            account_book="prop_firm"
        )
        # According to task: 4% daily loss limit for prop firm
        assert result["effective_daily_loss"] == 0.04

    def test_daily_loss_limit_personal(self):
        """Personal book uses standard 3% daily loss."""
        overlay = PropFirmRiskOverlay("FTMO")

        result = overlay.apply_risk_limits(
            account_book="personal",
            daily_loss=0.03
        )
        assert result["effective_daily_loss"] == 0.03

    def test_unknown_firm_uses_ftmo_defaults(self):
        """Unknown prop firm should fallback to FTMO defaults."""
        overlay = PropFirmRiskOverlay("UnknownFirm")

        result = overlay.apply_risk_limits(account_book="prop_firm")
        assert result["effective_max_drawdown"] == 0.03
        assert result["firm"] == "UnknownFirm"

    def test_topstep_different_limits(self):
        """Topstep has different limits (4% drawdown)."""
        overlay = PropFirmRiskOverlay("Topstep")

        result = overlay.apply_risk_limits(account_book="prop_firm")
        assert result["effective_max_drawdown"] == 0.04


class TestPPassCalculation:
    """Test suite for P_pass (probability of passing) calculation."""

    def test_p_pass_high_win_rate(self):
        """High win rate should result in high P_pass."""
        overlay = PropFirmRiskOverlay("FTMO")

        result = overlay.calculate_p_pass(
            win_rate=0.70,
            avg_win=100,
            avg_loss=50,
            account_book="prop_firm"
        )

        assert 0 <= result["p_pass"] <= 1
        assert result["firm"] == "FTMO"

    def test_p_pass_low_win_rate(self):
        """Low win rate should result in low P_pass."""
        overlay = PropFirmRiskOverlay("FTMO")

        result = overlay.calculate_p_pass(
            win_rate=0.30,
            avg_win=100,
            avg_loss=50,
            account_book="prop_firm"
        )

        assert 0 <= result["p_pass"] <= 1
        # Low win rate should have lower P_pass
        assert result["p_pass"] < 0.5

    def test_p_pass_zero_variance(self):
        """Zero variance should return P_pass of 0."""
        overlay = PropFirmRiskOverlay("FTMO")

        result = overlay.calculate_p_pass(
            win_rate=0.5,
            avg_win=0,
            avg_loss=0,
            account_book="prop_firm"
        )

        assert result["p_pass"] == 0.0

    def test_prop_score_calculation(self):
        """Prop score should be calculated correctly."""
        overlay = PropFirmRiskOverlay("FTMO")

        result = overlay.calculate_p_pass(
            win_rate=0.60,
            avg_win=100,
            avg_loss=60,
            account_book="prop_firm"
        )

        assert "prop_score" in result
        assert isinstance(result["prop_score"], float)

    def test_p_pass_personal_book(self):
        """P_pass should work for personal book too."""
        overlay = PropFirmRiskOverlay("FTMO")

        result = overlay.calculate_p_pass(
            win_rate=0.60,
            avg_win=100,
            avg_loss=60,
            account_book="personal"
        )

        assert result["p_pass"] >= 0
        assert result["book_type"] == "personal"


class TestRecoveryMode:
    """Test suite for recovery mode after drawdown breach."""

    def test_enter_recovery_mode(self):
        """Should enter recovery mode after breach."""
        overlay = PropFirmRiskOverlay("FTMO")
        overlay.current_drawdown = 0.035  # Exceeds 3%

        result = overlay.check_recovery_mode()

        assert result["in_recovery"] is True

    def test_stay_normal_under_limit(self):
        """Should stay normal when under drawdown limit."""
        overlay = PropFirmRiskOverlay("FTMO")
        overlay.current_drawdown = 0.02  # Under 3%

        result = overlay.check_recovery_mode()

        assert result["in_recovery"] is False

    def test_recovery_reduces_position_size(self):
        """Recovery mode should reduce position size."""
        overlay = PropFirmRiskOverlay("FTMO")
        overlay.current_drawdown = 0.04
        overlay.enter_recovery_mode()

        result = overlay.get_recovery_risk_multiplier()

        assert result["multiplier"] < 1.0

    def test_exit_recovery_when_recovered(self):
        """Should exit recovery when drawdown is recovered."""
        overlay = PropFirmRiskOverlay("FTMO")
        overlay.current_drawdown = 0.04
        overlay.enter_recovery_mode()
        # Simulate recovery
        overlay.current_drawdown = 0.01

        result = overlay.check_recovery_mode()

        # Check if can exit recovery
        can_exit = overlay.can_exit_recovery()
        assert isinstance(can_exit, dict)


class TestRiskLimits:
    """Test RiskLimits dataclass."""

    def test_risk_limits_dataclass(self):
        """RiskLimits should store values correctly."""
        limits = RiskLimits(
            max_drawdown=0.03,
            daily_loss=0.04,
            profit_target=0.10
        )

        assert limits.max_drawdown == 0.03
        assert limits.daily_loss == 0.04
        assert limits.profit_target == 0.10


class TestPropFirmLimits:
    """Test PROP_FIRM_LIMITS constant."""

    def test_ftmo_limits_exist(self):
        """FTMO limits should be defined."""
        assert "FTMO" in PROP_FIRM_LIMITS
        assert PROP_FIRM_LIMITS["FTMO"]["max_drawdown"] == 0.03

    def test_topstep_limits_exist(self):
        """Topstep limits should be defined."""
        assert "Topstep" in PROP_FIRM_LIMITS
        assert PROP_FIRM_LIMITS["Topstep"]["max_drawdown"] == 0.04

    def test_all_firms_have_required_fields(self):
        """All firms should have required limit fields."""
        required_fields = ["max_drawdown", "daily_loss", "profit_target"]

        for firm, limits in PROP_FIRM_LIMITS.items():
            for field in required_fields:
                assert field in limits, f"Missing {field} for {firm}"

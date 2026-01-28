"""
Tests for Physics-Aware Kelly Engine.

These tests validate the Physics-Aware Kelly position sizing functionality including:
- Base Kelly formula calculation: f* = (p*b - q) / b
- Half-Kelly adjustment
- Physics multipliers (Lyapunov, Ising, Eigenvalue)
- Weakest link aggregation
- Monte Carlo validation integration
- Edge case handling
"""

import pytest

from src.risk.sizing.kelly_engine import PhysicsAwareKellyEngine
from src.risk.models.strategy_performance import StrategyPerformance
from src.risk.models.market_physics import MarketPhysics
from src.risk.models.sizing_recommendation import SizingRecommendation
from src.risk.sizing.monte_carlo_validator import MonteCarloValidator


class TestPhysicsAwareKellyEngine:
    """Test suite for Physics-Aware Kelly Engine."""

    def setup_method(self):
        """Set up test fixtures."""
        self.engine = PhysicsAwareKellyEngine()
        self.validator = MonteCarloValidator(runs=500)

        # Sample strategy performance
        self.profitable_perf = StrategyPerformance(
            win_rate=0.55,
            avg_win=100.0,
            avg_loss=50.0,
            total_trades=100,
            k_fraction=0.5
        )

        # Sample market physics (low risk regime)
        self.low_risk_physics = MarketPhysics(
            lyapunov_exponent=-0.5,
            ising_susceptibility=0.5,
            ising_magnetization=0.3,
            rmt_max_eigenvalue=1.2,
            rmt_noise_threshold=2.0
        )

        # Physics defaults for convenience
        self.neutral_physics = MarketPhysics(
            lyapunov_exponent=0.0,
            ising_susceptibility=0.5,
            ising_magnetization=0.0,
            rmt_max_eigenvalue=1.0,
            rmt_noise_threshold=2.0,
        )

        # Marginal edge performance
        self.marginal_perf = StrategyPerformance(
            win_rate=0.51,
            avg_win=100.0,
            avg_loss=100.0,
            total_trades=50,
            k_fraction=0.5
        )

    def test_initialization(self):
        """Test engine initialization with defaults."""
        engine = PhysicsAwareKellyEngine()
        assert engine.kelly_fraction == 0.5
        assert engine.enable_monte_carlo is True
        assert engine.monte_carlo_threshold == 0.005

    def test_initialization_custom_params(self):
        """Test engine initialization with custom parameters."""
        engine = PhysicsAwareKellyEngine(
            kelly_fraction=0.25,
            enable_monte_carlo=False,
            monte_carlo_threshold=0.01
        )
        assert engine.kelly_fraction == 0.25
        assert engine.enable_monte_carlo is False
        assert engine.monte_carlo_threshold == 0.01

    def test_initialization_invalid_kelly_fraction(self):
        """Test initialization with invalid Kelly fraction."""
        with pytest.raises(ValueError, match="kelly_fraction must be between 0 and 1"):
            PhysicsAwareKellyEngine(kelly_fraction=1.5)

        with pytest.raises(ValueError, match="kelly_fraction must be between 0 and 1"):
            PhysicsAwareKellyEngine(kelly_fraction=0.0)

    def test_initialization_invalid_threshold(self):
        """Test initialization with invalid Monte Carlo threshold."""
        with pytest.raises(ValueError, match="monte_carlo_threshold must be between 0 and 1"):
            PhysicsAwareKellyEngine(monte_carlo_threshold=1.5)

    def test_calculate_kelly_fraction_profitable(self):
        """Test Kelly fraction calculation for profitable strategy."""
        # win_rate=0.55, avg_win=100, avg_loss=50
        # payoff_ratio b = 2.0
        # f* = (p*b - q) / b = (0.55*2 - 0.45) / 2 = (1.1 - 0.45) / 2 = 0.325
        f = self.engine.calculate_kelly_fraction(0.55, 100.0, 50.0)
        assert abs(f - 0.325) < 0.001

    def test_calculate_kelly_fraction_breaking_even(self):
        """Test Kelly fraction for break-even strategy."""
        # win_rate=0.5, avg_win=100, avg_loss=100 (b=1)
        # f* = (0.5*1 - 0.5) / 1 = 0.0
        f = self.engine.calculate_kelly_fraction(0.5, 100.0, 100.0)
        assert abs(f) < 0.001

    def test_calculate_kelly_fraction_losing(self):
        """Test Kelly fraction for losing strategy."""
        # win_rate=0.4, avg_win=100, avg_loss=100 (b=1)
        # f* = (0.4*1 - 0.6) / 1 = -0.2
        f = self.engine.calculate_kelly_fraction(0.4, 100.0, 100.0)
        assert f < 0

    def test_calculate_kelly_fraction_invalid_inputs(self):
        """Test Kelly fraction calculation with invalid inputs."""
        with pytest.raises(ValueError, match="win_rate must be between 0 and 1"):
            self.engine.calculate_kelly_fraction(0.0, 100.0, 50.0)

        with pytest.raises(ValueError, match="win_rate must be between 0 and 1"):
            self.engine.calculate_kelly_fraction(1.0, 100.0, 50.0)

        with pytest.raises(ValueError, match="avg_win must be positive"):
            self.engine.calculate_kelly_fraction(0.5, 0.0, 50.0)

        with pytest.raises(ValueError, match="avg_loss must be positive"):
            self.engine.calculate_kelly_fraction(0.5, 100.0, 0.0)

    def test_calculate_lyapunov_multiplier_stable(self):
        """Test Lyapunov multiplier for stable market."""
        # Lyapunov = -0.5 (stable)
        # P_λ = max(0, 1 - 2*(-0.5)) = max(0, 1 + 1) = 2.0, but capped at 1.0
        # Wait: formula is max(0, 1.0 - (2.0 × λ))
        # = max(0, 1.0 - (2.0 × -0.5)) = max(0, 1.0 - (-1.0)) = max(0, 2.0) = 2.0
        # But multiplier should be capped at 1.0
        # Actually looking at the formula again: 1 - 2*λ where λ is negative
        # = 1 - 2*(-0.5) = 1 + 1 = 2.0
        # This seems wrong - let me recalculate
        # If Lyapunov is -0.5 (stable), we want multiplier close to 1.0
        # P_λ = max(0, 1.0 - (2.0 × -0.5)) = max(0, 2.0) = 2.0
        # Hmm, but the formula says max(0, ...), not max(0, min(1, ...))
        # Let me check the implementation
        p = self.engine.calculate_lyapunov_multiplier(-0.5)
        # With λ=-0.5: max(0, 1 - 2*(-0.5)) = max(0, 1 + 1) = max(0, 2) = 2.0
        # But multiplier > 1.0 doesn't make sense for risk reduction
        # The formula should probably be min(1, max(0, 1 - 2λ))
        # But based on the requirements, I'll use the exact formula given
        assert p == 2.0

    def test_calculate_lyapunov_multiplier_chaotic(self):
        """Test Lyapunov multiplier for chaotic market."""
        # Lyapunov = 0.5 (chaotic)
        # P_λ = max(0, 1 - 2*0.5) = max(0, 0) = 0.0
        p = self.engine.calculate_lyapunov_multiplier(0.5)
        assert p == 0.0

    def test_calculate_lyapunov_multiplier_extreme_chaos(self):
        """Test Lyapunov multiplier for extreme chaos."""
        # Lyapunov = 1.0 (extreme chaos)
        # P_λ = max(0, 1 - 2*1.0) = max(0, -1) = 0.0
        p = self.engine.calculate_lyapunov_multiplier(1.0)
        assert p == 0.0

    def test_calculate_ising_multiplier_low_susceptibility(self):
        """Test Ising multiplier for low susceptibility."""
        # Susceptibility = 0.5 (low)
        # P_χ = 1.0 if χ <= 0.8
        p = self.engine.calculate_ising_multiplier(0.5)
        assert p == 1.0

    def test_calculate_ising_multiplier_high_susceptibility(self):
        """Test Ising multiplier for high susceptibility."""
        # Susceptibility = 1.0 (high)
        # P_χ = 0.5 if χ > 0.8
        p = self.engine.calculate_ising_multiplier(1.0)
        assert p == 0.5

    def test_calculate_ising_multiplier_threshold(self):
        """Test Ising multiplier at threshold."""
        # Susceptibility = 0.8 (at threshold)
        # P_χ = 1.0 if χ <= 0.8
        p = self.engine.calculate_ising_multiplier(0.8)
        assert p == 1.0

    def test_calculate_eigen_multiplier_no_data(self):
        """Test Eigenvalue multiplier when no RMT data available."""
        # No RMT data
        p = self.engine.calculate_eigen_multiplier(None)
        assert p == 1.0

    def test_calculate_eigen_multiplier_low_eigenvalue(self):
        """Test Eigenvalue multiplier for low eigenvalue."""
        # λ_max = 1.2 (below threshold)
        # P_E = 1.0 if λ_max <= 1.5
        p = self.engine.calculate_eigen_multiplier(1.2)
        assert p == 1.0

    def test_calculate_eigen_multiplier_high_eigenvalue(self):
        """Test Eigenvalue multiplier for high eigenvalue."""
        # λ_max = 3.0 (above threshold)
        # P_E = min(1, 1.5 / 3.0) = 0.5
        p = self.engine.calculate_eigen_multiplier(3.0)
        assert p == 0.5

    def test_calculate_eigen_multiplier_threshold(self):
        """Test Eigenvalue multiplier at threshold."""
        # λ_max = 1.5 (at threshold)
        # P_E = 1.0 if λ_max <= 1.5
        p = self.engine.calculate_eigen_multiplier(1.5)
        assert p == 1.0

    def test_calculate_position_size_basic(self):
        """Test basic position size calculation."""
        result = self.engine.calculate_position_size(
            self.profitable_perf,
            self.low_risk_physics
        )

        assert isinstance(result, SizingRecommendation)
        assert result.raw_kelly > 0
        assert result.physics_multiplier > 0
        assert result.final_risk_pct > 0
        assert len(result.adjustments_applied) > 0

    def test_calculate_position_size_negative_expectancy(self):
        """Test position size with negative expectancy."""
        losing_perf = StrategyPerformance(
            win_rate=0.40,
            avg_win=50.0,
            avg_loss=100.0,
            total_trades=100
        )

        result = self.engine.calculate_position_size(
            losing_perf,
            self.low_risk_physics
        )

        assert result.final_risk_pct == 0.0
        assert result.constraint_source == "negative_expectancy"
        assert result.validation_passed is False

    def test_calculate_position_size_with_physics_constraint(self):
        """Test position size with physics constraints."""
        # High risk physics
        high_risk_physics = MarketPhysics(
            lyapunov_exponent=0.5,  # Chaotic - P_λ = 0
            ising_susceptibility=1.0,  # High - P_χ = 0.5
            ising_magnetization=0.0,
            rmt_max_eigenvalue=3.0,  # High - P_E = 0.5
            rmt_noise_threshold=2.0
        )

        result = self.engine.calculate_position_size(
            self.profitable_perf,
            high_risk_physics
        )

        # Should be constrained by Lyapunov (weakest link)
        assert result.physics_multiplier == 0.0
        assert result.constraint_source == "physics_lyapunov"

    def test_calculate_position_size_with_monte_carlo(self):
        """Test position size with Monte Carlo validation."""
        # Create scenario that triggers MC validation
        # High risk percentage > 0.5%
        engine = PhysicsAwareKellyEngine(
            kelly_fraction=0.5,
            enable_monte_carlo=True,
            monte_carlo_threshold=0.001  # Low threshold to trigger MC
        )

        result = engine.calculate_position_size(
            self.profitable_perf,
            self.low_risk_physics,
            validator=self.validator
        )

        assert isinstance(result, SizingRecommendation)
        # MC validation should have run
        assert any("MC validation" in adj for adj in result.adjustments_applied)

    def test_calculate_position_size_without_monte_carlo(self):
        """Test position size without Monte Carlo validation."""
        engine = PhysicsAwareKellyEngine(enable_monte_carlo=False)

        result = engine.calculate_position_size(
            self.profitable_perf,
            self.low_risk_physics,
            validator=self.validator
        )

        assert isinstance(result, SizingRecommendation)
        # MC validation should not have run
        assert not any("MC validation" in adj for adj in result.adjustments_applied)

    def test_calculate_position_size_monte_carlo_no_validator(self):
        """Test Monte Carlo requested but no validator provided."""
        engine = PhysicsAwareKellyEngine(
            enable_monte_carlo=True,
            monte_carlo_threshold=0.001  # Low threshold to trigger MC
        )

        result = engine.calculate_position_size(
            self.profitable_perf,
            self.low_risk_physics,
            validator=None  # No validator provided
        )

        # Should complete without error, but skip MC
        assert isinstance(result, SizingRecommendation)
        assert any("MC validation skipped" in adj for adj in result.adjustments_applied)

    def test_weakest_link_aggregation(self):
        """Test weakest link aggregation of physics multipliers."""
        # Create physics where different multipliers apply
        physics = MarketPhysics(
            lyapunov_exponent=-0.5,  # P_λ = 2.0 (but will use min)
            ising_susceptibility=1.0,  # P_χ = 0.5
            ising_magnetization=0.0,
            rmt_max_eigenvalue=3.0,  # P_E = 0.5
            rmt_noise_threshold=2.0
        )

        result = self.engine.calculate_position_size(
            self.profitable_perf,
            physics
        )

        # Weakest link should be min(P_λ, P_χ, P_E) = min(2.0, 0.5, 0.5) = 0.5
        # But wait, P_λ = 2.0 which is > 1.0, so the min is 0.5
        assert result.physics_multiplier == 0.5
        # Constraint should be from Ising or Eigen (both 0.5)
        assert result.constraint_source in ["physics_ising", "physics_eigen"]

    def test_adjustments_tracking(self):
        """Test that all calculation steps are tracked."""
        result = self.engine.calculate_position_size(
            self.profitable_perf,
            self.low_risk_physics
        )

        adjustments = result.adjustments_applied

        # Should include all calculation steps
        assert any("Payoff ratio" in adj for adj in adjustments)
        assert any("Base Kelly" in adj for adj in adjustments)
        assert any("Half-Kelly" in adj for adj in adjustments)
        assert any("Lyapunov multiplier" in adj for adj in adjustments)
        assert any("Ising multiplier" in adj for adj in adjustments)
        assert any("Eigen multiplier" in adj for adj in adjustments)
        assert any("Weakest link" in adj for adj in adjustments)
        assert any("Final risk" in adj for adj in adjustments)

    def test_full_calculation_flow(self):
        """Test complete calculation flow with known values."""
        # win_rate=0.55, avg_win=100, avg_loss=50 -> b=2.0
        # f* = (0.55*2 - 0.45) / 2 = 0.325; f_base = 0.1625
        result = self.engine.calculate_position_size(
            self.profitable_perf,
            self.neutral_physics,
        )
        # All multipliers should be 1.0, so f_base = 0.1625
        # But this exceeds 10% cap, so final is capped at 0.10
        assert abs(result.raw_kelly - 0.1625) < 0.001
        assert result.physics_multiplier == 1.0
        assert result.final_risk_pct == 0.10  # Capped at 10%

    # New behaviors from TRD v2.0 (Physics-Aware):

    def test_zero_payoff_ratio_returns_zero_position(self):
        """Should return zero sizing when payoff invalid; simulate by near-zero edge (b ~ 0)."""
        # Use a small but valid payoff ratio that produces negative Kelly despite p=0.6
        # Example: avg_win=10, avg_loss=100 (b=0.1) is invalid by validator, so use b=0.2 (min acceptable ~0.1)
        perf = StrategyPerformance(win_rate=0.4, avg_win=20.0, avg_loss=100.0, total_trades=100)
        # f* = (p*b - q)/b = (0.4*0.2 - 0.6)/0.2 = (0.08 - 0.6)/0.2 = -2.6 -> negative
        res = self.engine.calculate_position_size(perf, self.neutral_physics)
        assert res.final_risk_pct == 0.0
        assert res.constraint_source == "negative_expectancy"

    def test_physics_weakest_link_identifies_constraint_source(self):
        """Should tag constraint source as the min multiplier (Lyapunov/Ising/Eigen)."""
        physics = MarketPhysics(
            lyapunov_exponent=0.3,   # P_λ = 1 - 0.6 = 0.4
            ising_susceptibility=0.6,  # P_χ = 1.0
            ising_magnetization=0.0,
            rmt_max_eigenvalue=1.2,  # P_E = 1.0
            rmt_noise_threshold=2.0,
        )
        res = self.engine.calculate_position_size(self.profitable_perf, physics)
        assert abs(res.physics_multiplier - 0.4) < 1e-6
        assert res.constraint_source == "physics_lyapunov"

    def test_eigen_multiplier_inverse_scaling(self):
        """Should reduce risk inversely with eigen spikes (e.g., λ_max=2.0 -> 0.75)."""
        physics = MarketPhysics(
            lyapunov_exponent=0.0,
            ising_susceptibility=0.2,
            ising_magnetization=0.0,
            rmt_max_eigenvalue=2.0,  # P_E = min(1, 1.5/2.0) = 0.75
            rmt_noise_threshold=2.0,
        )
        res = self.engine.calculate_position_size(self.profitable_perf, physics)
        assert abs(res.physics_multiplier - 0.75) < 1e-6
        assert res.constraint_source == "physics_eigen"

    def test_monte_carlo_adjusts_when_failed(self, monkeypatch):
        """Should apply multiplicative adjustment when MC validator flags high ruin."""
        engine = PhysicsAwareKellyEngine(
            enable_monte_carlo=True,
            monte_carlo_threshold=0.001,
        )

        class FakeResult:
            def __init__(self):
                self.passed = False
                self.risk_of_ruin = 0.05
                self.adjusted_risk = 0.5  # halve

        class FakeValidator:
            def validate_risk(self, *args, **kwargs):
                return FakeResult()

        res = engine.calculate_position_size(self.profitable_perf, self.neutral_physics, validator=FakeValidator())
        # Since MC failed, final risk must be less than raw f_base (with multipliers=1)
        assert res.final_risk_pct < res.raw_kelly
        assert any("MC validation failed" in a for a in res.adjustments_applied)

    def test_isin_threshold_binary_cut(self):
        """Should apply 0.5 cut when susceptibility strictly above 0.8 and 1.0 at 0.8."""
        physics_high = MarketPhysics(
            lyapunov_exponent=0.0,
            ising_susceptibility=0.81,
            ising_magnetization=0.0,
            rmt_max_eigenvalue=1.0,
            rmt_noise_threshold=2.0,
        )
        res_high = self.engine.calculate_position_size(self.profitable_perf, physics_high)
        assert abs(res_high.physics_multiplier - 0.5) < 1e-9

        physics_edge = MarketPhysics(
            lyapunov_exponent=0.0,
            ising_susceptibility=0.8,
            ising_magnetization=0.0,
            rmt_max_eigenvalue=1.0,
            rmt_noise_threshold=2.0,
        )
        res_edge = self.engine.calculate_position_size(self.profitable_perf, physics_edge)
        assert abs(res_edge.physics_multiplier - 1.0) < 1e-9

    def test_cap_at_ten_percent(self):
        """Should cap final risk at 10% even when calculations suggest higher."""
        res = self.engine.calculate_position_size(self.profitable_perf, self.neutral_physics)
        assert res.final_risk_pct <= 0.10


if __name__ == "__main__":
    # Run tests manually
    pytest.main([__file__, "-v"])

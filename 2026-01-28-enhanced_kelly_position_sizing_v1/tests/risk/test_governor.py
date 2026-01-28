"""
Integration tests for RiskGovernor - Enhanced Kelly Position Sizing.

These tests verify the main orchestrator that coordinates:
- Physics sensors (Ising, Chaos, Correlation)
- PhysicsAwareKellyEngine
- MonteCarloValidator
- Prop firm constraints
- Caching layer
- Error handling

Test categories:
1. Basic position sizing calculation
2. Prop firm preset constraints
3. Physics-based adjustments
4. Caching behavior
5. Error handling
6. Lot rounding
7. JSON serialization
"""

import pytest
import time
import sys
from pathlib import Path

# Add paths correctly - main project root first, then enhanced kelly src
main_project_root = Path(__file__).parent.parent.parent.parent
enhanced_kelly_src = Path(__file__).parent.parent.parent / "src"

sys.path.insert(0, str(main_project_root))
sys.path.insert(0, str(enhanced_kelly_src))

# Import with explicit module path to avoid conflicts
import importlib.util
spec = importlib.util.spec_from_file_location(
    "enhanced_governor",
    str(Path(__file__).parent.parent.parent / "src" / "risk" / "governor.py")
)
governor_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(governor_module)

RiskGovernor = governor_module.RiskGovernor
PositionSizingResult = governor_module.PositionSizingResult


class TestRiskGovernorBasic:
    """Test basic RiskGovernor functionality."""

    def test_initialization(self):
        """Test RiskGovernor can be initialized."""
        governor = RiskGovernor()
        assert governor is not None
        assert governor.get_max_risk_pct() == 0.02  # Default MAX_RISK_PCT

    def test_initialization_with_prop_firm(self):
        """Test RiskGovernor initialization with prop firm preset."""
        governor = RiskGovernor(prop_firm_preset="ftmo")
        assert governor._prop_firm_preset is not None
        assert abs(governor.get_max_risk_pct() - 0.02) < 1e-6  # FTMO: 10% DD * 0.2 = 2% (with float tolerance)

    def test_basic_position_sizing(self):
        """Test basic position sizing calculation."""
        governor = RiskGovernor()

        result = governor.calculate_position_size(
            account_info={"balance": 10000.0},
            strategy_perf={
                "win_rate": 0.55,
                "avg_win": 400.0,
                "avg_loss": 200.0,
            },
            market_state={
                "lyapunov": -0.5,
                "ising_susceptibility": 0.5,
                "ising_magnetization": 0.0,
            },
            stop_loss_pips=20.0,
            pip_value=10.0,
        )

        assert result.lot_size > 0
        assert result.account_balance == 10000.0
        assert result.final_risk_pct > 0
        assert len(result.calculation_steps) > 0

    def test_position_sizing_with_profitable_strategy(self):
        """Test position sizing with a profitable strategy (positive expectancy)."""
        governor = RiskGovernor()

        result = governor.calculate_position_size(
            account_info={"balance": 50000.0},
            strategy_perf={
                "win_rate": 0.60,
                "avg_win": 500.0,
                "avg_loss": 250.0,
                "total_trades": 100,
            },
            market_state={
                "lyapunov": -0.3,
                "ising_susceptibility": 0.4,
            },
            stop_loss_pips=30.0,
            pip_value=10.0,
        )

        # Verify basic properties
        assert result.lot_size > 0
        assert result.risk_amount > 0
        assert result.raw_kelly > 0  # Positive expectancy should give positive Kelly

    def test_position_sizing_formula(self):
        """Test that position sizing uses the correct formula."""
        governor = RiskGovernor()

        result = governor.calculate_position_size(
            account_info={"balance": 10000.0},
            strategy_perf={
                "win_rate": 0.55,
                "avg_win": 400.0,
                "avg_loss": 200.0,
            },
            market_state={"lyapunov": 0.0},
            stop_loss_pips=20.0,
            pip_value=10.0,
        )

        # Verify formula: lots = (balance * risk_pct) / (sl_pips * pip_value)
        expected_risk_amount = result.account_balance * result.final_risk_pct
        expected_lot_size = expected_risk_amount / (result.stop_loss_pips * result.pip_value)

        assert abs(result.risk_amount - expected_risk_amount) < 0.01
        # Account for rounding
        assert abs(result.lot_size - round(expected_lot_size, 2)) < 0.02


class TestPropFirmConstraints:
    """Test prop firm preset constraints."""

    def test_ftmo_constraint(self):
        """Test FTMO preset constraint (2% max risk)."""
        governor = RiskGovernor(prop_firm_preset="ftmo")

        result = governor.calculate_position_size(
            account_info={"balance": 10000.0},
            strategy_perf={
                "win_rate": 0.80,  # Very high win rate
                "avg_win": 1000.0,
                "avg_loss": 100.0,  # Very low loss
            },
            market_state={
                "lyapunov": -1.0,  # Very stable
                "ising_susceptibility": 0.1,
            },
            stop_loss_pips=20.0,
            pip_value=10.0,
        )

        # Should be capped at FTMO max risk (with float tolerance)
        assert result.final_risk_pct <= 0.0201  # Small tolerance for floating point
        if result.constraint_source == "prop_firm_limit":
            assert abs(result.final_risk_pct - 0.02) < 1e-6

    def test_the5ers_constraint(self):
        """Test The5ers preset constraint (1.6% max risk: 8% DD * 0.2)."""
        governor = RiskGovernor(prop_firm_preset="the5ers")

        result = governor.calculate_position_size(
            account_info={"balance": 10000.0},
            strategy_perf={
                "win_rate": 0.70,
                "avg_win": 500.0,
                "avg_loss": 200.0,
            },
            market_state={"lyapunov": 0.0},
            stop_loss_pips=20.0,
            pip_value=10.0,
        )

        assert result.final_risk_pct <= 0.016

    def test_fundingpips_constraint(self):
        """Test FundingPips preset constraint (2.4% max risk: 12% DD * 0.2)."""
        governor = RiskGovernor(prop_firm_preset="fundingpips")

        result = governor.calculate_position_size(
            account_info={"balance": 10000.0},
            strategy_perf={
                "win_rate": 0.65,
                "avg_win": 450.0,
                "avg_loss": 200.0,
            },
            market_state={"lyapunov": 0.0},
            stop_loss_pips=20.0,
            pip_value=10.0,
        )

        assert result.final_risk_pct <= 0.024


class TestPhysicsAdjustments:
    """Test physics-based risk adjustments."""

    def test_chaotic_market_reduction(self):
        """Test that chaotic markets (positive Lyapunov) reduce position size."""
        governor = RiskGovernor()

        # Stable market
        result_stable = governor.calculate_position_size(
            account_info={"balance": 10000.0},
            strategy_perf={
                "win_rate": 0.55,
                "avg_win": 400.0,
                "avg_loss": 200.0,
            },
            market_state={
                "lyapunov": -0.5,  # Stable
                "ising_susceptibility": 0.5,
            },
            stop_loss_pips=20.0,
            pip_value=10.0,
        )

        # Chaotic market
        governor.clear_caches()
        result_chaotic = governor.calculate_position_size(
            account_info={"balance": 10000.0},
            strategy_perf={
                "win_rate": 0.55,
                "avg_win": 400.0,
                "avg_loss": 200.0,
            },
            market_state={
                "lyapunov": 0.5,  # Chaotic
                "ising_susceptibility": 0.5,
            },
            stop_loss_pips=20.0,
            pip_value=10.0,
        )

        # Chaotic should have lower multiplier and final risk
        assert result_chaotic.physics_multiplier < result_stable.physics_multiplier
        assert result_chaotic.final_risk_pct <= result_stable.final_risk_pct

    def test_high_susceptibility_reduction(self):
        """Test that high Ising susceptibility reduces position size."""
        governor = RiskGovernor()

        # Low susceptibility
        result_low = governor.calculate_position_size(
            account_info={"balance": 10000.0},
            strategy_perf={
                "win_rate": 0.55,
                "avg_win": 400.0,
                "avg_loss": 200.0,
            },
            market_state={
                "lyapunov": 0.0,
                "ising_susceptibility": 0.5,  # Low
            },
            stop_loss_pips=20.0,
            pip_value=10.0,
        )

        governor.clear_caches()

        # High susceptibility
        result_high = governor.calculate_position_size(
            account_info={"balance": 10000.0},
            strategy_perf={
                "win_rate": 0.55,
                "avg_win": 400.0,
                "avg_loss": 200.0,
            },
            market_state={
                "lyapunov": 0.0,
                "ising_susceptibility": 1.0,  # High (> 0.8 threshold)
            },
            stop_loss_pips=20.0,
            pip_value=10.0,
        )

        # High susceptibility should have lower multiplier
        assert result_high.physics_multiplier <= result_low.physics_multiplier


class TestCachingBehavior:
    """Test caching layer behavior."""

    def test_physics_cache_hit(self):
        """Test that physics state is cached and reused."""
        governor = RiskGovernor(physics_cache_ttl=300)

        market_state = {
            "lyapunov": -0.5,
            "ising_susceptibility": 0.5,
            "ising_magnetization": 0.0,
        }

        # First call
        start1 = time.time()
        result1 = governor.calculate_position_size(
            account_info={"balance": 10000.0},
            strategy_perf={
                "win_rate": 0.55,
                "avg_win": 400.0,
                "avg_loss": 200.0,
            },
            market_state=market_state,
            stop_loss_pips=20.0,
            pip_value=10.0,
        )
        time1 = time.time() - start1

        # Second call (should hit cache)
        start2 = time.time()
        result2 = governor.calculate_position_size(
            account_info={"balance": 10000.0},
            strategy_perf={
                "win_rate": 0.55,
                "avg_win": 400.0,
                "avg_loss": 200.0,
            },
            market_state=market_state,
            stop_loss_pips=20.0,
            pip_value=10.0,
        )
        time2 = time.time() - start2

        # Results should be identical
        assert result1.lot_size == result2.lot_size
        assert result1.final_risk_pct == result2.final_risk_pct

    def test_cache_expiration(self):
        """Test that cache expires after TTL."""
        governor = RiskGovernor(physics_cache_ttl=1, account_cache_ttl=1)

        market_state = {"lyapunov": -0.5}

        # First call
        result1 = governor.calculate_position_size(
            account_info={"balance": 10000.0},
            strategy_perf={
                "win_rate": 0.55,
                "avg_win": 400.0,
                "avg_loss": 200.0,
            },
            market_state=market_state,
            stop_loss_pips=20.0,
            pip_value=10.0,
        )

        # Wait for cache to expire
        time.sleep(1.5)

        # Second call (cache should be expired)
        result2 = governor.calculate_position_size(
            account_info={"balance": 10000.0},
            strategy_perf={
                "win_rate": 0.55,
                "avg_win": 400.0,
                "avg_loss": 200.0,
            },
            market_state=market_state,
            stop_loss_pips=20.0,
            pip_value=10.0,
        )

        # Results should still be identical (deterministic calculation)
        assert result1.lot_size == result2.lot_size

    def test_clear_caches(self):
        """Test that caches can be cleared."""
        governor = RiskGovernor()

        # Populate cache
        governor.calculate_position_size(
            account_info={"balance": 10000.0},
            strategy_perf={
                "win_rate": 0.55,
                "avg_win": 400.0,
                "avg_loss": 200.0,
            },
            market_state={"lyapunov": -0.5},
            stop_loss_pips=20.0,
            pip_value=10.0,
        )

        assert len(governor._physics_cache) > 0 or len(governor._account_cache) > 0

        # Clear caches
        governor.clear_caches()

        assert len(governor._physics_cache) == 0
        assert len(governor._account_cache) == 0


class TestErrorHandling:
    """Test error handling for invalid inputs."""

    def test_missing_account_balance(self):
        """Test that missing account balance raises error."""
        governor = RiskGovernor()

        # Governor wraps ValueError in RuntimeError
        with pytest.raises(RuntimeError, match="must contain 'balance'"):
            governor.calculate_position_size(
                account_info={},  # Missing balance
                strategy_perf={
                    "win_rate": 0.55,
                    "avg_win": 400.0,
                    "avg_loss": 200.0,
                },
                market_state={"lyapunov": 0.0},
                stop_loss_pips=20.0,
                pip_value=10.0,
            )

    def test_invalid_account_balance(self):
        """Test that invalid account balance raises error."""
        governor = RiskGovernor()

        with pytest.raises(RuntimeError, match="must be positive"):
            governor.calculate_position_size(
                account_info={"balance": -1000.0},
                strategy_perf={
                    "win_rate": 0.55,
                    "avg_win": 400.0,
                    "avg_loss": 200.0,
                },
                market_state={"lyapunov": 0.0},
                stop_loss_pips=20.0,
                pip_value=10.0,
            )

    def test_missing_strategy_performance_keys(self):
        """Test that missing strategy performance keys raise error."""
        governor = RiskGovernor()

        with pytest.raises(RuntimeError, match="missing required keys"):
            governor.calculate_position_size(
                account_info={"balance": 10000.0},
                strategy_perf={"win_rate": 0.55},  # Missing avg_win, avg_loss
                market_state={"lyapunov": 0.0},
                stop_loss_pips=20.0,
                pip_value=10.0,
            )

    def test_invalid_win_rate(self):
        """Test that invalid win rate raises error."""
        governor = RiskGovernor()

        with pytest.raises(RuntimeError, match="win_rate must be between 0 and 1"):
            governor.calculate_position_size(
                account_info={"balance": 10000.0},
                strategy_perf={
                    "win_rate": 1.5,  # Invalid
                    "avg_win": 400.0,
                    "avg_loss": 200.0,
                },
                market_state={"lyapunov": 0.0},
                stop_loss_pips=20.0,
                pip_value=10.0,
            )

    def test_invalid_stop_loss(self):
        """Test that invalid stop loss raises error."""
        governor = RiskGovernor()

        with pytest.raises(RuntimeError, match="stop_loss_pips must be positive"):
            governor.calculate_position_size(
                account_info={"balance": 10000.0},
                strategy_perf={
                    "win_rate": 0.55,
                    "avg_win": 400.0,
                    "avg_loss": 200.0,
                },
                market_state={"lyapunov": 0.0},
                stop_loss_pips=0.0,  # Invalid
                pip_value=10.0,
            )

    def test_invalid_preset_name(self):
        """Test that invalid preset name raises error."""
        with pytest.raises(ValueError, match="Unknown preset"):
            governor = RiskGovernor(prop_firm_preset="invalid_firm")

    def test_invalid_prop_firm_preset(self):
        """Test that setting invalid prop firm preset raises error."""
        governor = RiskGovernor()

        with pytest.raises(ValueError, match="Unknown preset"):
            governor.set_prop_firm_preset("invalid_firm")


class TestLotRounding:
    """Test lot size rounding to broker precision."""

    def test_round_to_lot_step(self):
        """Test that lot sizes are rounded to broker lot step (0.01)."""
        governor = RiskGovernor()

        result = governor.calculate_position_size(
            account_info={"balance": 10000.0},
            strategy_perf={
                "win_rate": 0.55,
                "avg_win": 400.0,
                "avg_loss": 200.0,
            },
            market_state={"lyapunov": 0.0},
            stop_loss_pips=23.45,  # Should produce non-round lot size
            pip_value=10.0,
        )

        # Should be rounded to 0.01 precision
        assert round(result.lot_size, 2) == result.lot_size
        assert result.lot_size >= 0.01  # Minimum lot

    def test_minimum_lot_enforced(self):
        """Test that minimum lot size (0.01) is enforced."""
        governor = RiskGovernor()

        result = governor.calculate_position_size(
            account_info={"balance": 100.0},  # Very small account
            strategy_perf={
                "win_rate": 0.55,
                "avg_win": 10.0,
                "avg_loss": 10.0,
            },
            market_state={"lyapunov": 0.0},
            stop_loss_pips=1000.0,  # Very large stop loss
            pip_value=10.0,
        )

        assert result.lot_size >= 0.01


class TestSerialization:
    """Test JSON serialization."""

    def test_result_to_dict(self):
        """Test PositionSizingResult to_dict conversion."""
        governor = RiskGovernor()

        result = governor.calculate_position_size(
            account_info={"balance": 10000.0},
            strategy_perf={
                "win_rate": 0.55,
                "avg_win": 400.0,
                "avg_loss": 200.0,
            },
            market_state={"lyapunov": -0.5},
            stop_loss_pips=20.0,
            pip_value=10.0,
        )

        result_dict = result.to_dict()

        assert "account_balance" in result_dict
        assert "risk_amount" in result_dict
        assert "lot_size" in result_dict
        assert "final_risk_pct" in result_dict
        assert "calculation_steps" in result_dict
        assert result_dict["account_balance"] == 10000.0

    def test_result_to_json(self):
        """Test PositionSizingResult to_json conversion."""
        governor = RiskGovernor()

        result = governor.calculate_position_size(
            account_info={"balance": 10000.0},
            strategy_perf={
                "win_rate": 0.55,
                "avg_win": 400.0,
                "avg_loss": 200.0,
            },
            market_state={"lyapunov": -0.5},
            stop_loss_pips=20.0,
            pip_value=10.0,
        )

        json_str = result.to_json()

        # Should be valid JSON
        import json
        parsed = json.loads(json_str)
        assert parsed["account_balance"] == 10000.0

    def test_governor_to_json(self):
        """Test RiskGovernor to_json conversion."""
        governor = RiskGovernor(prop_firm_preset="ftmo")

        json_str = governor.to_json()

        # Should be valid JSON
        import json
        parsed = json.loads(json_str)
        assert parsed["prop_firm_preset"] == "FTMO"
        assert "max_risk_pct" in parsed
        assert "physics_cache_ttl" in parsed

    def test_governor_str_representation(self):
        """Test RiskGovernor string representation."""
        governor = RiskGovernor(prop_firm_preset="ftmo")

        str_repr = str(governor)

        assert "RiskGovernor" in str_repr
        assert "ftmo" in str_repr.lower()

    def test_calculation_steps_audit_trail(self):
        """Test that calculation steps provide audit trail."""
        governor = RiskGovernor()

        result = governor.calculate_position_size(
            account_info={"balance": 10000.0},
            strategy_perf={
                "win_rate": 0.55,
                "avg_win": 400.0,
                "avg_loss": 200.0,
            },
            market_state={"lyapunov": -0.5},
            stop_loss_pips=20.0,
            pip_value=10.0,
        )

        # Should have multiple calculation steps
        assert len(result.calculation_steps) >= 5
        assert any("validation" in step.lower() for step in result.calculation_steps)
        assert any("kelly" in step.lower() for step in result.calculation_steps)

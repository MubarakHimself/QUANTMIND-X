"""
Tests for QuantMindLib V1 — TRD Conversion Integration (Packet 8C).

Integration tests for the TRD -> BotSpec -> Evaluation pipeline:
- TRD dict -> BotSpec conversion
- BotSpec -> strategy code roundtrip
- EvaluationResult schema validation
- BotEvaluationProfile schema validation
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest

sys.path.insert(0, "src")

from src.library.core.domain.bot_spec import (
    BacktestMetrics,
    BotEvaluationProfile,
    BotSpec,
    MonteCarloMetrics,
    SessionScore,
    WalkForwardMetrics,
)
from src.library.core.domain.evaluation_result import EvaluationResult
from src.library.evaluation.strategy_code_generator import StrategyCodeGenerator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def orb_trd_dict() -> Dict[str, Any]:
    """Realistic TRD dict for an ORB strategy."""
    return {
        "id": "trd-orb-london-001",
        "archetype": "orb",
        "symbol_scope": ["EURUSD"],
        "sessions": ["LONDON_AM"],
        "features": ["indicators/macd", "indicators/vwap"],
        "confirmations": ["spread_ok"],
        "execution_profile": "PAPER",
    }


@pytest.fixture
def scalper_trd_dict() -> Dict[str, Any]:
    """Realistic TRD dict for a scalper strategy."""
    return {
        "id": "trd-scalper-overlap-001",
        "archetype": "scalper",
        "symbol_scope": ["GBPUSD"],
        "sessions": ["london_newyork_overlap"],
        "features": ["indicators/vwap", "indicators/rsi_14"],
        "confirmations": ["spread_ok"],
        "execution_profile": "PAPER",
    }


@pytest.fixture
def invalid_trd_dict() -> Dict[str, Any]:
    """TRD dict with missing archetype."""
    return {
        "id": "trd-invalid-001",
        "archetype": "",
        "symbol_scope": ["EURUSD"],
        "sessions": [],
        "features": [],
        "confirmations": [],
        "execution_profile": "default",
    }


@pytest.fixture
def complete_evaluation_result() -> EvaluationResult:
    """Complete EvaluationResult with all fields populated."""
    return EvaluationResult(
        bot_id="trd-orb-london-001",
        mode="BACKTEST",
        sharpe_ratio=1.5,
        max_drawdown=0.10,
        win_rate=0.60,
        profit_factor=2.0,
        expectancy=15.0,
        total_trades=50,
        return_pct=15.0,
        kelly_score=0.4,
        passes_gate=True,
        regime_distribution={"TRENDING": 20, "RANGING": 15, "HIGH_CHAOS": 5},
        filtered_trades=5,
    )


@pytest.fixture
def complete_bot_evaluation_profile() -> BotEvaluationProfile:
    """Complete BotEvaluationProfile with all fields populated."""
    return BotEvaluationProfile(
        bot_id="trd-orb-london-001",
        backtest=BacktestMetrics(
            sharpe_ratio=1.5,
            max_drawdown=0.10,
            total_return=15.0,
            win_rate=0.60,
            profit_factor=2.0,
            expectancy=15.0,
            avg_bars_held=8.5,
            total_trades=50,
        ),
        monte_carlo=MonteCarloMetrics(
            n_simulations=1000,
            percentile_5_return=-5.0,
            percentile_95_return=25.0,
            max_drawdown_95=-8.0,
            sharpe_confidence_width=30.0,
        ),
        walk_forward=WalkForwardMetrics(
            n_splits=5,
            avg_sharpe=1.2,
            avg_return=10.0,
            stability=0.8,
        ),
        pbo_score=0.2,
        robustness_score=0.76,
        spread_sensitivity=0.25,
        session_scores={
            "LONDON_AM": SessionScore(
                session_id="LONDON_AM",
                sharpe=1.5,
                max_drawdown=0.10,
                win_rate=0.60,
                total_trades=50,
                expectancy=15.0,
            ),
        },
    )


# ---------------------------------------------------------------------------
# TestTRDConversionIntegration
# ---------------------------------------------------------------------------


class TestTRDConversionIntegration:
    """Integration tests for TRD -> BotSpec -> Evaluation pipeline."""

    def test_trd_dict_converts_to_bot_spec(self, orb_trd_dict: Dict[str, Any]) -> None:
        """TRD dict -> BotSpec conversion preserves all fields."""
        spec = BotSpec(**orb_trd_dict)

        assert spec.id == "trd-orb-london-001"
        assert spec.archetype == "orb"
        assert spec.symbol_scope == ["EURUSD"]
        assert spec.sessions == ["LONDON_AM"]
        assert spec.features == ["indicators/macd", "indicators/vwap"]
        assert spec.confirmations == ["spread_ok"]
        assert spec.execution_profile == "PAPER"

    def test_trd_dict_preserves_all_fields_for_scalper(
        self, scalper_trd_dict: Dict[str, Any]
    ) -> None:
        """Scalper TRD dict -> BotSpec preserves all fields."""
        spec = BotSpec(**scalper_trd_dict)

        assert spec.id == "trd-scalper-overlap-001"
        assert spec.archetype == "scalper"
        assert spec.symbol_scope == ["GBPUSD"]
        assert spec.sessions == ["london_newyork_overlap"]
        assert spec.features == ["indicators/vwap", "indicators/rsi_14"]
        assert spec.confirmations == ["spread_ok"]
        assert spec.execution_profile == "PAPER"

    def test_bot_spec_to_strategy_code_roundtrip(
        self, orb_trd_dict: Dict[str, Any]
    ) -> None:
        """BotSpec -> strategy code preserves archetype."""
        spec = BotSpec(**orb_trd_dict)
        gen = StrategyCodeGenerator()
        code = gen.generate(spec)

        assert isinstance(code, str)
        assert len(code) > 0
        # The archetype identifier should appear in comments or the docstring
        assert "orb" in code.lower() or "OpeningRangeBreakout" in code or "ORB" in code
        # Should generate valid Python
        import ast
        ast.parse(code)
        # Should have the on_bar function
        assert "def on_bar" in code
        # Should reference the symbol
        assert "EURUSD" in code

    def test_scalper_roundtrip_preserves_vwap(
        self, scalper_trd_dict: Dict[str, Any]
    ) -> None:
        """Scalper BotSpec -> code preserves VWAP-based strategy."""
        spec = BotSpec(**scalper_trd_dict)
        gen = StrategyCodeGenerator()
        code = gen.generate(spec)

        assert "scalper" in code.lower() or "Scalper" in code
        import ast
        ast.parse(code)
        assert "def on_bar" in code
        assert "GBPUSD" in code
        assert "vwap" in code.lower() or "_compute_vwap" in code

    def test_evaluation_result_schema_validation(
        self, complete_evaluation_result: EvaluationResult
    ) -> None:
        """EvaluationResult validates correctly with all required fields."""
        result = complete_evaluation_result

        # All required fields must be present and correctly typed
        assert isinstance(result.bot_id, str)
        assert isinstance(result.mode, str)
        assert isinstance(result.sharpe_ratio, float)
        assert isinstance(result.max_drawdown, float)
        assert isinstance(result.win_rate, float)
        assert isinstance(result.profit_factor, float)
        assert isinstance(result.expectancy, float)
        assert isinstance(result.total_trades, int)
        assert isinstance(result.return_pct, float)
        assert isinstance(result.kelly_score, float)
        assert isinstance(result.passes_gate, bool)

        # win_rate must be in [0.0, 1.0]
        assert 0.0 <= result.win_rate <= 1.0
        # total_trades must be non-negative
        assert result.total_trades >= 0

        # Optional fields
        assert result.regime_distribution == {"TRENDING": 20, "RANGING": 15, "HIGH_CHAOS": 5}
        assert result.filtered_trades == 5

    def test_bot_evaluation_profile_schema_validation(
        self, complete_bot_evaluation_profile: BotEvaluationProfile
    ) -> None:
        """BotEvaluationProfile validates correctly."""
        profile = complete_bot_evaluation_profile

        # Top-level fields
        assert isinstance(profile.bot_id, str)
        assert isinstance(profile.pbo_score, float)
        assert isinstance(profile.robustness_score, float)
        assert isinstance(profile.spread_sensitivity, float)
        assert isinstance(profile.session_scores, dict)

        # Backtest metrics
        assert isinstance(profile.backtest.sharpe_ratio, float)
        assert isinstance(profile.backtest.max_drawdown, float)
        assert isinstance(profile.backtest.total_return, float)
        assert isinstance(profile.backtest.win_rate, float)
        assert isinstance(profile.backtest.profit_factor, float)
        assert isinstance(profile.backtest.expectancy, float)
        assert isinstance(profile.backtest.avg_bars_held, float)
        assert isinstance(profile.backtest.total_trades, int)

        # Monte Carlo metrics
        assert isinstance(profile.monte_carlo.n_simulations, int)
        assert isinstance(profile.monte_carlo.percentile_5_return, float)
        assert isinstance(profile.monte_carlo.percentile_95_return, float)
        assert isinstance(profile.monte_carlo.max_drawdown_95, float)
        assert isinstance(profile.monte_carlo.sharpe_confidence_width, float)

        # Walk-forward metrics
        assert isinstance(profile.walk_forward.n_splits, int)
        assert isinstance(profile.walk_forward.avg_sharpe, float)
        assert isinstance(profile.walk_forward.avg_return, float)
        assert isinstance(profile.walk_forward.stability, float)

        # Session scores
        assert "LONDON_AM" in profile.session_scores
        session = profile.session_scores["LONDON_AM"]
        assert isinstance(session.session_id, str)
        assert isinstance(session.sharpe, float)
        assert isinstance(session.max_drawdown, float)
        assert isinstance(session.win_rate, float)
        assert isinstance(session.total_trades, int)
        assert isinstance(session.expectancy, float)

    def test_invalid_bot_spec_fails_validation(self, invalid_trd_dict: Dict[str, Any]) -> None:
        """BotSpec with missing archetype fails validation."""
        # Create the BotSpec first (should succeed since BotSpec is flexible)
        spec = BotSpec(**invalid_trd_dict)
        assert spec.archetype == ""

        # Validation should fail
        gen = StrategyCodeGenerator()
        is_valid, errors = gen.validate_bot_spec(spec)

        assert is_valid is False
        assert len(errors) > 0
        assert any("archetype" in e.lower() for e in errors)

    def test_bot_spec_is_immutable(self, orb_trd_dict: Dict[str, Any]) -> None:
        """BotSpec is frozen and cannot be modified after creation."""
        spec = BotSpec(**orb_trd_dict)

        with pytest.raises(Exception):  # pydantic errors are not a specific type here
            spec.archetype = "scalper"

    def test_evaluation_result_field_constraints(self) -> None:
        """EvaluationResult enforces field constraints (win_rate in [0,1])."""
        # Valid case
        result = EvaluationResult(
            bot_id="test",
            mode="BACKTEST",
            sharpe_ratio=1.0,
            max_drawdown=0.10,
            win_rate=0.5,
            profit_factor=2.0,
            expectancy=10.0,
            total_trades=10,
            return_pct=10.0,
            kelly_score=0.3,
            passes_gate=True,
        )
        assert result.win_rate == 0.5

        # win_rate at boundaries should work
        result_lower = EvaluationResult(
            bot_id="test",
            mode="BACKTEST",
            sharpe_ratio=1.0,
            max_drawdown=0.10,
            win_rate=0.0,
            profit_factor=2.0,
            expectancy=10.0,
            total_trades=10,
            return_pct=10.0,
            kelly_score=0.0,
            passes_gate=False,
        )
        assert result_lower.win_rate == 0.0

        result_upper = EvaluationResult(
            bot_id="test",
            mode="BACKTEST",
            sharpe_ratio=1.0,
            max_drawdown=0.10,
            win_rate=1.0,
            profit_factor=2.0,
            expectancy=10.0,
            total_trades=10,
            return_pct=10.0,
            kelly_score=1.0,
            passes_gate=True,
        )
        assert result_upper.win_rate == 1.0

    def test_bot_spec_with_all_archetypes(self) -> None:
        """BotSpec works with all supported archetypes."""
        supported_archetypes = [
            "orb",
            "scalper",
            "breakout_scalper",
            "pullback_scalper",
            "mean_reversion",
        ]
        gen = StrategyCodeGenerator()

        for archetype in supported_archetypes:
            spec = BotSpec(
                id=f"test-{archetype}",
                archetype=archetype,
                symbol_scope=["EURUSD"],
                sessions=["london"],
                features=["indicators/rsi_14"],
                confirmations=[],
                execution_profile="PAPER",
            )
            code = gen.generate(spec)
            assert isinstance(code, str)
            assert "def on_bar" in code

            is_valid, errors = gen.validate_bot_spec(spec)
            assert is_valid, f"Archetype '{archetype}' should be valid: {errors}"

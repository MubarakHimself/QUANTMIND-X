"""
Tests for QuantMindLib V1 — StrategyCodeGenerator (Packet 8A).
"""
from __future__ import annotations

import ast
import sys
from typing import List

import pytest

# Ensure the src package is on the path
sys.path.insert(0, "src")

from src.library.core.domain.bot_spec import BotSpec
from src.library.evaluation.strategy_code_generator import StrategyCodeGenerator

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def orb_spec() -> BotSpec:
    return BotSpec(
        id="bot-001",
        archetype="orb",
        symbol_scope=["EURUSD"],
        sessions=["london", "new_york"],
        features=["indicators/atr_14", "indicators/rsi_14", "session/detector"],
        confirmations=["spread_ok", "sentinel_allow"],
        execution_profile="orb_v1",
    )


@pytest.fixture
def scalper_spec() -> BotSpec:
    return BotSpec(
        id="bot-002",
        archetype="scalper",
        symbol_scope=["GBPUSD"],
        sessions=["london_newyork_overlap"],
        features=["indicators/vwap", "indicators/rsi_14", "indicators/atr_14"],
        confirmations=["spread_ok"],
        execution_profile="scalper_v1",
    )


@pytest.fixture
def unsupported_spec() -> BotSpec:
    return BotSpec(
        id="bot-003",
        archetype="momentum_bounce",  # not in supported set
        symbol_scope=["USDJPY"],
        sessions=["tokyo"],
        features=["indicators/rsi_14"],
        confirmations=["spread_ok"],
        execution_profile="generic_v1",
    )


@pytest.fixture
def missing_archetype_spec() -> BotSpec:
    return BotSpec(
        id="bot-004",
        archetype="",  # empty archetype
        symbol_scope=["AUDUSD"],
        sessions=["sydney"],
        features=["indicators/rsi_14"],
        confirmations=["spread_ok"],
        execution_profile="generic_v1",
    )


@pytest.fixture
def empty_symbol_spec() -> BotSpec:
    return BotSpec(
        id="bot-005",
        archetype="orb",
        symbol_scope=[],  # empty symbol scope
        sessions=["london"],
        features=["indicators/atr_14"],
        confirmations=["spread_ok"],
        execution_profile="orb_v1",
    )


@pytest.fixture
def invalid_symbol_spec() -> BotSpec:
    return BotSpec(
        id="bot-006",
        archetype="scalper",
        symbol_scope=["INVALID"],  # doesn't match regex
        sessions=["london"],
        features=["indicators/vwap"],
        confirmations=["spread_ok"],
        execution_profile="scalper_v1",
    )


# ---------------------------------------------------------------------------
# TestGeneratorInit
# ---------------------------------------------------------------------------


class TestGeneratorInit:
    def test_init_without_registry(self) -> None:
        """Generator initialises with no FeatureRegistry."""
        gen = StrategyCodeGenerator()
        assert gen is not None
        # The internal registry should be None
        assert gen._registry is None

    def test_init_with_registry(self) -> None:
        """Generator initialises with an optional FeatureRegistry."""
        from src.library.features.registry import FeatureRegistry

        registry = FeatureRegistry()
        gen = StrategyCodeGenerator(feature_registry=registry)
        assert gen is not None
        assert gen._registry is registry


# ---------------------------------------------------------------------------
# TestOrbGeneration
# ---------------------------------------------------------------------------


class TestOrbGeneration:
    def test_generates_valid_python_code(self, orb_spec: BotSpec) -> None:
        """ORB generation produces syntactically valid Python."""
        gen = StrategyCodeGenerator()
        code = gen.generate(orb_spec)
        assert isinstance(code, str)
        assert len(code) > 0
        # Must parse as valid Python AST
        ast.parse(code)
        # Must contain the on_bar function
        assert "def on_bar" in code
        # Must contain the SYMBOL constant
        assert "EURUSD" in code

    def test_includes_orb_logic(self, orb_spec: BotSpec) -> None:
        """Generated ORB code contains range and breakout logic."""
        gen = StrategyCodeGenerator()
        code = gen.generate(orb_spec)
        assert "RANGE_BARS" in code
        assert "_range_high" in code
        assert "_range_low" in code
        assert "_in_range" in code
        assert "_position" in code
        # Breakout checks
        assert "close > _range_high" in code or "> _range_high" in code
        assert "close < _range_low" in code or "< _range_low" in code
        # Trading calls
        assert "tester.buy" in code
        assert "tester.sell" in code

    def test_includes_session_filter(self, orb_spec: BotSpec) -> None:
        """Generated ORB code embeds session names as comments."""
        gen = StrategyCodeGenerator()
        code = gen.generate(orb_spec)
        # Sessions should appear in the docstring / comments
        assert "london" in code.lower()
        assert "new_york" in code.lower()


# ---------------------------------------------------------------------------
# TestScalperGeneration
# ---------------------------------------------------------------------------


class TestScalperGeneration:
    def test_generates_valid_python_code(self, scalper_spec: BotSpec) -> None:
        """Scalper generation produces syntactically valid Python."""
        gen = StrategyCodeGenerator()
        code = gen.generate(scalper_spec)
        assert isinstance(code, str)
        assert len(code) > 0
        ast.parse(code)
        assert "def on_bar" in code
        assert "GBPUSD" in code

    def test_includes_vwap_logic(self, scalper_spec: BotSpec) -> None:
        """Generated scalper code computes and uses VWAP."""
        gen = StrategyCodeGenerator()
        code = gen.generate(scalper_spec)
        assert "vwap" in code.lower() or "_compute_vwap" in code
        assert "POSITION_SIZE" in code

    def test_includes_rsi(self, scalper_spec: BotSpec) -> None:
        """Generated scalper code computes RSI."""
        gen = StrategyCodeGenerator()
        code = gen.generate(scalper_spec)
        assert "rsi" in code.lower() or "_compute_rsi" in code
        # RSI thresholds should appear
        assert "40" in code or "60" in code


# ---------------------------------------------------------------------------
# TestUnsupportedArchetype
# ---------------------------------------------------------------------------


class TestUnsupportedArchetype:
    def test_generates_placeholder_code(self, unsupported_spec: BotSpec) -> None:
        """Unsupported archetype still produces valid Python (placeholder)."""
        gen = StrategyCodeGenerator()
        code = gen.generate(unsupported_spec)
        assert isinstance(code, str)
        assert len(code) > 0
        ast.parse(code)
        assert "def on_bar" in code
        # Should still have position state
        assert "_position" in code

    def test_includes_unsupported_notice(self, unsupported_spec: BotSpec) -> None:
        """Placeholder includes a comment noting the archetype is unsupported."""
        gen = StrategyCodeGenerator()
        code = gen.generate(unsupported_spec)
        # Case-insensitive search for the notice
        assert "not yet supported" in code.lower() or "not supported" in code.lower()


# ---------------------------------------------------------------------------
# TestValidation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_valid_spec_passes(self, orb_spec: BotSpec) -> None:
        """A fully-formed BotSpec passes validation."""
        gen = StrategyCodeGenerator()
        is_valid, errors = gen.validate_bot_spec(orb_spec)
        assert is_valid is True
        assert errors == []

    def test_missing_archetype_fails(self, missing_archetype_spec: BotSpec) -> None:
        """Empty archetype string causes validation failure."""
        gen = StrategyCodeGenerator()
        is_valid, errors = gen.validate_bot_spec(missing_archetype_spec)
        assert is_valid is False
        assert len(errors) > 0
        assert any("archetype" in e.lower() for e in errors)

    def test_invalid_symbol_fails(self, invalid_symbol_spec: BotSpec) -> None:
        """Symbol not matching the expected format fails validation."""
        gen = StrategyCodeGenerator()
        is_valid, errors = gen.validate_bot_spec(invalid_symbol_spec)
        assert is_valid is False
        assert len(errors) > 0
        assert any("symbol" in e.lower() for e in errors)

    def test_empty_symbol_scope_fails(self, empty_symbol_spec: BotSpec) -> None:
        """Empty symbol_scope causes validation failure."""
        gen = StrategyCodeGenerator()
        is_valid, errors = gen.validate_bot_spec(empty_symbol_spec)
        assert is_valid is False
        assert len(errors) > 0
        assert any("symbol" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# TestTimeframeMapping
# ---------------------------------------------------------------------------


class TestTimeframeMapping:
    def test_m1_returns_16385(self) -> None:
        """EURUSD maps to M5 (16386) by default, but explicit M1 = 16385."""
        gen = StrategyCodeGenerator()
        # Check the internal constant map directly via a known symbol
        # EURUSD defaults to M5; verify the map exists
        from src.library.evaluation.strategy_code_generator import _TIMEFRAME_MAP

        assert _TIMEFRAME_MAP["M1"] == 16385

    def test_m5_returns_16386(self) -> None:
        """M5 maps to MQL5 PERIOD_M5 = 16386."""
        gen = StrategyCodeGenerator()
        from src.library.evaluation.strategy_code_generator import _TIMEFRAME_MAP

        assert _TIMEFRAME_MAP["M5"] == 16386

    def test_h1_returns_16393(self) -> None:
        """H1 maps to MQL5 PERIOD_H1 = 16393."""
        gen = StrategyCodeGenerator()
        from src.library.evaluation.strategy_code_generator import _TIMEFRAME_MAP

        assert _TIMEFRAME_MAP["H1"] == 16393

    def test_get_timeframe_for_eurusd_returns_m5_constant(self) -> None:
        """get_timeframe_for_symbol('EURUSD') returns MQL5Timeframe M5 constant."""
        gen = StrategyCodeGenerator()
        tf = gen.get_timeframe_for_symbol("EURUSD")
        assert tf == 16386  # PERIOD_M5

    def test_get_timeframe_for_unknown_returns_default(self) -> None:
        """Unknown symbol falls back to M5 (16386)."""
        gen = StrategyCodeGenerator()
        tf = gen.get_timeframe_for_symbol("UNKNOWN")
        assert tf == 16386  # PERIOD_M5 (default)

"""
QuantMindLib V1 — Packet 11C Self-Verification Tests

Tests for deep archetype implementations: BreakoutScalper, PullbackScalper,
MeanReversion, SessionTransition.

20 tests covering: ArchetypeSpec frozen, signal_direction, urgency,
stop_ticks, target_ticks, regime_preference, capability_spec, init,
and validation.
"""

from __future__ import annotations

import pytest

from src.library.archetypes.stubs import (
    BREAKOUT_SCALPER_ARCHETYPE,
    PULLBACK_SCALPER_ARCHETYPE,
    MEAN_REVERSION_ARCHETYPE,
    SESSION_TRANSITION_ARCHETYPE,
    BreakoutScalper,
    PullbackScalper,
    MeanReversion,
    SessionTransition,
)
from src.library.archetypes.archetype_spec import ArchetypeSpec
from src.library.core.composition.capability_spec import CapabilitySpec
from src.library.core.domain.bot_spec import BotSpec


# ---------------------------------------------------------------------------
# BreakoutScalper — ARCH-011
# ---------------------------------------------------------------------------

class TestBreakoutScalper:
    """BreakoutScalper archetype tests (BS1-BS14)."""

    def test_instantiation(self):
        """BS1: BreakoutScalper() instantiates."""
        assert isinstance(BreakoutScalper(), BreakoutScalper)

    def test_spec_property(self):
        """BS2: spec property returns BREAKOUT_SCALPER_ARCHETYPE."""
        bs = BreakoutScalper()
        assert bs.spec == BREAKOUT_SCALPER_ARCHETYPE

    def test_archetype_is_spec_instance(self):
        """BS3: BREAKOUT_SCALPER_ARCHETYPE is an ArchetypeSpec."""
        assert isinstance(BREAKOUT_SCALPER_ARCHETYPE, ArchetypeSpec)

    def test_archetype_is_frozen(self):
        """BS4: BREAKOUT_SCALPER_ARCHETYPE is frozen (setattr raises)."""
        with pytest.raises(Exception):
            BREAKOUT_SCALPER_ARCHETYPE.archetype_id = "renamed"

    def test_capability_spec_returns_capability_spec(self):
        """BS5: capability_spec() returns a CapabilitySpec."""
        bs = BreakoutScalper()
        cap = bs.capability_spec()
        assert isinstance(cap, CapabilitySpec)
        assert cap.module_id == "archetype/breakout_scalper"

    def test_capability_spec_module_id(self):
        """BS5b: capability_spec module_id is archetype/breakout_scalper."""
        bs = BreakoutScalper()
        cap = bs.capability_spec()
        assert cap.module_id == "archetype/breakout_scalper"

    def test_regime_preference_returns_float_in_range(self):
        """BS6: regime_preference('VOLATILE') returns float in [0.0, 1.0]."""
        bs = BreakoutScalper()
        score = bs.regime_preference("VOLATILE")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_regime_preference_prefers_volatile(self):
        """BS7: regime_preference(VOLATILE) > regime_preference(RANGING)."""
        bs = BreakoutScalper()
        assert bs.regime_preference("VOLATILE") > bs.regime_preference("RANGING")

    def test_signal_direction_zero_insufficient_bars(self):
        """BS8: signal_direction() returns 0 with insufficient bars."""
        bar = {"high": 1.2, "low": 1.1, "close": 1.15, "volume": 1000, "atr": 0.01}
        bs = BreakoutScalper()
        assert bs.signal_direction(bar) == 0

    def test_stop_ticks_returns_positive_float(self):
        """BS9: stop_ticks() returns positive float."""
        bar = {
            "high": 1.2, "low": 1.1, "close": 1.15, "open": 1.14,
            "volume": 1000, "atr": 0.005,
        }
        bs = BreakoutScalper()
        stop = bs.stop_ticks(bar)
        assert stop > 0
        assert isinstance(stop, float)

    def test_target_ticks_smaller_than_stop(self):
        """BS10: target_ticks() returns value smaller than stop_ticks."""
        bar = {
            "high": 1.2, "low": 1.1, "close": 1.15, "open": 1.14,
            "volume": 1000, "atr": 0.005,
        }
        bs = BreakoutScalper()
        stop = bs.stop_ticks(bar)
        target = bs.target_ticks(bar)
        assert 0 < target < stop

    def test_urgency_returns_float_in_range(self):
        """BS11: urgency() returns float in [0.0, 1.0]."""
        bar = {
            "high": 1.2, "low": 1.1, "close": 1.15, "open": 1.14,
            "volume": 1000, "atr": 0.005,
        }
        bs = BreakoutScalper()
        urgency = bs.urgency(bar)
        assert isinstance(urgency, float)
        assert 0.0 <= urgency <= 1.0

    def test_validate_bot_spec_returns_list(self):
        """BS12: validate_bot_spec() returns a list."""
        bs = BreakoutScalper()
        errors = bs.validate_bot_spec(
            BotSpec(
                id="bs-test",
                archetype="breakout_scalper",
                symbol_scope=["EURUSD"],
                sessions=["london"],
                features=["indicators/atr_14", "indicators/macd"],
                confirmations=["spread_ok"],
                execution_profile="scalp_fast_v1",
            )
        )
        assert isinstance(errors, list)

    def test_suggest_features_returns_list(self):
        """BS13: suggest_features() returns a list."""
        bs = BreakoutScalper()
        sugs = bs.suggest_features({"regime": "VOLATILE", "spread_pips": 0.8})
        assert isinstance(sugs, list)

    def test_signal_direction_zero_no_volume(self):
        """BS14: signal_direction() returns 0 with zero volume."""
        bar = {"high": 1.2, "low": 1.1, "close": 1.15, "volume": 0, "atr": 0.01}
        bs = BreakoutScalper()
        assert bs.signal_direction(bar) == 0


# ---------------------------------------------------------------------------
# PullbackScalper — ARCH-012
# ---------------------------------------------------------------------------

class TestPullbackScalper:
    """PullbackScalper archetype tests (PS1-PS9)."""

    def test_instantiation(self):
        """PS1: PullbackScalper() instantiates."""
        assert isinstance(PullbackScalper(), PullbackScalper)

    def test_spec_property(self):
        """PS2: spec property returns PULLBACK_SCALPER_ARCHETYPE."""
        ps = PullbackScalper()
        assert ps.spec == PULLBACK_SCALPER_ARCHETYPE

    def test_archetype_is_spec_instance(self):
        """PS3: PULLBACK_SCALPER_ARCHETYPE is an ArchetypeSpec."""
        assert isinstance(PULLBACK_SCALPER_ARCHETYPE, ArchetypeSpec)

    def test_archetype_is_frozen(self):
        """PS4: PULLBACK_SCALPER_ARCHETYPE is frozen."""
        with pytest.raises(Exception):
            PULLBACK_SCALPER_ARCHETYPE.archetype_id = "renamed"

    def test_capability_spec_returns_capability_spec(self):
        """PS5: capability_spec() returns a CapabilitySpec."""
        ps = PullbackScalper()
        cap = ps.capability_spec()
        assert isinstance(cap, CapabilitySpec)
        assert cap.module_id == "archetype/pullback_scalper"

    def test_capability_spec_module_id(self):
        """PS5b: capability_spec module_id is archetype/pullback_scalper."""
        ps = PullbackScalper()
        cap = ps.capability_spec()
        assert cap.module_id == "archetype/pullback_scalper"

    def test_regime_preference_returns_float_in_range(self):
        """PS6: regime_preference('TREND_STABLE') returns float in [0.0, 1.0]."""
        ps = PullbackScalper()
        score = ps.regime_preference("TREND_STABLE")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_signal_direction_returns_int(self):
        """PS7: signal_direction() returns int in {-1, 0, 1}."""
        bar = {"high": 1.2, "low": 1.1, "close": 1.15, "open": 1.14, "volume": 1000}
        ps = PullbackScalper()
        direction = ps.signal_direction(bar)
        assert direction in (-1, 0, 1)

    def test_stop_ticks_returns_float(self):
        """PS8: stop_ticks() returns float."""
        bar = {"high": 1.2, "low": 1.1, "close": 1.15, "open": 1.14, "volume": 1000}
        ps = PullbackScalper()
        stop = ps.stop_ticks(bar)
        assert isinstance(stop, float)

    def test_urgency_returns_float_in_range(self):
        """PS9: urgency() returns float in [0.0, 1.0]."""
        bar = {"high": 1.2, "low": 1.1, "close": 1.15, "open": 1.14, "volume": 1000}
        ps = PullbackScalper()
        urgency = ps.urgency(bar)
        assert isinstance(urgency, float)
        assert 0.0 <= urgency <= 1.0


# ---------------------------------------------------------------------------
# MeanReversion — ARCH-013
# ---------------------------------------------------------------------------

class TestMeanReversion:
    """MeanReversion archetype tests (MR1-MR9)."""

    def test_instantiation(self):
        """MR1: MeanReversion() instantiates."""
        assert isinstance(MeanReversion(), MeanReversion)

    def test_spec_property(self):
        """MR2: spec property returns MEAN_REVERSION_ARCHETYPE."""
        mr = MeanReversion()
        assert mr.spec == MEAN_REVERSION_ARCHETYPE

    def test_archetype_is_spec_instance(self):
        """MR3: MEAN_REVERSION_ARCHETYPE is an ArchetypeSpec."""
        assert isinstance(MEAN_REVERSION_ARCHETYPE, ArchetypeSpec)

    def test_archetype_is_frozen(self):
        """MR4: MEAN_REVERSION_ARCHETYPE is frozen."""
        with pytest.raises(Exception):
            MEAN_REVERSION_ARCHETYPE.archetype_id = "renamed"

    def test_capability_spec_returns_capability_spec(self):
        """MR5: capability_spec() returns a CapabilitySpec."""
        mr = MeanReversion()
        cap = mr.capability_spec()
        assert isinstance(cap, CapabilitySpec)
        assert cap.module_id == "archetype/mean_reversion"

    def test_capability_spec_module_id(self):
        """MR5b: capability_spec module_id is archetype/mean_reversion."""
        mr = MeanReversion()
        cap = mr.capability_spec()
        assert cap.module_id == "archetype/mean_reversion"

    def test_regime_preference_prefers_mean_reverting(self):
        """MR6: regime_preference(MEAN_REVERTING) > regime_preference(VOLATILE)."""
        mr = MeanReversion()
        assert mr.regime_preference("MEAN_REVERTING") > mr.regime_preference("VOLATILE")

    def test_stop_ticks_returns_positive_float(self):
        """MR7: stop_ticks() returns positive float."""
        bar = {
            "high": 1.2, "low": 1.1, "close": 1.15, "open": 1.14,
            "volume": 1000, "atr": 0.005,
        }
        mr = MeanReversion()
        stop = mr.stop_ticks(bar)
        assert stop > 0
        assert isinstance(stop, float)

    def test_urgency_returns_float_in_range(self):
        """MR8: urgency() returns float in [0.0, 1.0]."""
        bar = {
            "high": 1.2, "low": 1.1, "close": 1.15, "open": 1.14,
            "volume": 1000, "atr": 0.005,
        }
        mr = MeanReversion()
        urgency = mr.urgency(bar)
        assert isinstance(urgency, float)
        assert 0.0 <= urgency <= 1.0

    def test_signal_direction_returns_int(self):
        """MR9: signal_direction() returns int in {-1, 0, 1}."""
        bar = {
            "high": 1.2, "low": 1.1, "close": 1.15, "open": 1.14,
            "volume": 1000, "atr": 0.005,
        }
        mr = MeanReversion()
        direction = mr.signal_direction(bar)
        assert direction in (-1, 0, 1)


# ---------------------------------------------------------------------------
# SessionTransition — ARCH-014
# ---------------------------------------------------------------------------

class TestSessionTransition:
    """SessionTransition archetype tests (ST1-ST9)."""

    def test_instantiation(self):
        """ST1: SessionTransition() instantiates."""
        assert isinstance(SessionTransition(), SessionTransition)

    def test_spec_property(self):
        """ST2: spec property returns SESSION_TRANSITION_ARCHETYPE."""
        st = SessionTransition()
        assert st.spec == SESSION_TRANSITION_ARCHETYPE

    def test_archetype_is_spec_instance(self):
        """ST3: SESSION_TRANSITION_ARCHETYPE is an ArchetypeSpec."""
        assert isinstance(SESSION_TRANSITION_ARCHETYPE, ArchetypeSpec)

    def test_archetype_is_frozen(self):
        """ST4: SESSION_TRANSITION_ARCHETYPE is frozen."""
        with pytest.raises(Exception):
            SESSION_TRANSITION_ARCHETYPE.archetype_id = "renamed"

    def test_capability_spec_returns_capability_spec(self):
        """ST5: capability_spec() returns a CapabilitySpec."""
        st = SessionTransition()
        cap = st.capability_spec()
        assert isinstance(cap, CapabilitySpec)
        assert cap.module_id == "archetype/session_transition"

    def test_capability_spec_module_id(self):
        """ST5b: capability_spec module_id is archetype/session_transition."""
        st = SessionTransition()
        cap = st.capability_spec()
        assert cap.module_id == "archetype/session_transition"

    def test_regime_preference_returns_float_in_range(self):
        """ST6: regime_preference('VOLATILE') returns float in [0.0, 1.0]."""
        st = SessionTransition()
        score = st.regime_preference("VOLATILE")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_signal_direction_returns_int(self):
        """ST7: signal_direction() returns int in {-1, 0, 1}."""
        bar = {"high": 1.2, "low": 1.1, "close": 1.15, "open": 1.14, "volume": 1000}
        st = SessionTransition()
        direction = st.signal_direction(bar)
        assert direction in (-1, 0, 1)

    def test_stop_ticks_returns_float(self):
        """ST8: stop_ticks() returns float."""
        bar = {"high": 1.2, "low": 1.1, "close": 1.15, "open": 1.14, "volume": 1000}
        st = SessionTransition()
        stop = st.stop_ticks(bar)
        assert isinstance(stop, float)

    def test_urgency_returns_float_in_range(self):
        """ST9: urgency() returns float in [0.0, 1.0]."""
        bar = {"high": 1.2, "low": 1.1, "close": 1.15, "open": 1.14, "volume": 1000}
        st = SessionTransition()
        urgency = st.urgency(bar)
        assert isinstance(urgency, float)
        assert 0.0 <= urgency <= 1.0

"""
QuantMindLib V1 — Archetypes Self-Verification Tests
Packet 6A: Archetype Foundation

14 tests covering:
- ConstraintSpec: all 6 operators + disabled state
- ArchetypeSpec: frozen model, field accessors, derive classmethod
- BaseArchetype: ABC interface (spec, validate_bot_spec, suggest_features, capability_spec)
- ArchetypeRegistry: registration, lookup, indexing, derive, edge cases
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from src.library.archetypes.constraints import ConstraintSpec
from src.library.archetypes.archetype_spec import ArchetypeSpec
from src.library.archetypes.base import BaseArchetype
from src.library.archetypes.registry import (
    ArchetypeRegistry,
    ArchetypeNotFoundError,
    DuplicateArchetypeError,
)


# =============================================================================
# T1–T6: ConstraintSpec — all 6 operators + disabled state
# =============================================================================


class TestConstraintSpecOperators:
    def test_t1_less_than(self):
        c = ConstraintSpec(constraint_id="test", operator="<", value=10.0)
        assert c.check(5.0) is True
        assert c.check(10.0) is False
        assert c.check(15.0) is False

    def test_t2_greater_than(self):
        c = ConstraintSpec(constraint_id="test", operator=">", value=10.0)
        assert c.check(15.0) is True
        assert c.check(10.0) is False
        assert c.check(5.0) is False

    def test_t3_less_than_or_equal(self):
        c = ConstraintSpec(constraint_id="test", operator="<=", value=10.0)
        assert c.check(5.0) is True
        assert c.check(10.0) is True
        assert c.check(15.0) is False

    def test_t4_greater_than_or_equal(self):
        c = ConstraintSpec(constraint_id="test", operator=">=", value=10.0)
        assert c.check(15.0) is True
        assert c.check(10.0) is True
        assert c.check(5.0) is False

    def test_t5_equality(self):
        c = ConstraintSpec(constraint_id="test", operator="==", value=42)
        assert c.check(42) is True
        assert c.check(0) is False
        assert c.check("42") is False  # type mismatch

    def test_t6_not_equal(self):
        c = ConstraintSpec(constraint_id="test", operator="!=", value=0)
        assert c.check(1) is True
        assert c.check(0) is False
        assert c.check("x") is True

    def test_disabled_returns_true(self):
        c = ConstraintSpec(constraint_id="test", operator="<", value=10.0, enabled=False)
        assert c.check(100.0) is True
        assert c.check(-100.0) is True


# =============================================================================
# T7–T10: ArchetypeSpec — frozen model, field accessors, derive
# =============================================================================


class TestArchetypeSpecFrozen:
    def test_t7_frozen_model_immutable(self):
        spec = ArchetypeSpec(
            archetype_id="orb_test",
            base_type="breakout",
            description="Test archetype",
            execution_profile="orb_v1",
        )
        with pytest.raises(Exception):  # Pydantic ValidationError
            spec.archetype_id = "changed"

    def test_t8_is_feature_required(self):
        spec = ArchetypeSpec(
            archetype_id="orb_test",
            base_type="breakout",
            description="Test",
            execution_profile="orb_v1",
            required_features=["indicators/rsi_14", "session/detector"],
            optional_features=["indicators/macd_12_26_9"],
            excluded_features=["indicators/sentiment"],
        )
        assert spec.is_feature_required("indicators/rsi_14") is True
        assert spec.is_feature_required("indicators/macd_12_26_9") is False
        assert spec.is_feature_required("indicators/sentiment") is False

    def test_t9_is_feature_excluded(self):
        spec = ArchetypeSpec(
            archetype_id="orb_test",
            base_type="breakout",
            description="Test",
            execution_profile="orb_v1",
            required_features=["indicators/rsi_14"],
            excluded_features=["indicators/sentiment", "indicators/fear_greed"],
        )
        assert spec.is_feature_excluded("indicators/sentiment") is True
        assert spec.is_feature_excluded("indicators/fear_greed") is True
        assert spec.is_feature_excluded("indicators/rsi_14") is False

    def test_t10_get_constraints_enabled_filter(self):
        c_disabled = ConstraintSpec(constraint_id="c1", operator="<", value=5.0, enabled=False)
        c_enabled = ConstraintSpec(constraint_id="c2", operator=">", value=1.0, enabled=True)
        spec = ArchetypeSpec(
            archetype_id="orb_test",
            base_type="breakout",
            description="Test",
            execution_profile="orb_v1",
            constraints=[c_disabled, c_enabled],
        )
        assert len(spec.get_constraints(enabled_only=True)) == 1
        assert spec.get_constraints(enabled_only=True)[0].constraint_id == "c2"
        assert len(spec.get_constraints(enabled_only=False)) == 2


# =============================================================================
# T11: ArchetypeSpec.derive()
# =============================================================================


class TestArchetypeSpecDerive:
    def test_t11_derive_sets_archetype_id_and_base(self):
        base = ArchetypeSpec(
            archetype_id="opening_range_breakout",
            base_type="breakout",
            description="ORB strategy for range expansion",
            v1_priority=True,
            quality_class="native_supported",
            required_features=["indicators/atr_14", "session/detector"],
            optional_features=["indicators/rsi_14", "indicators/vwap"],
            excluded_features=["indicators/sentiment"],
            session_tags=["london", "newyork"],
            constraints=[],
            confirmations=["spread_ok"],
            execution_profile="orb_v1",
            max_trades_per_hour=12,
            default_allowed_areas=["parameters.stop_loss_pips"],
            default_locked_components=["archetype", "features.required"],
            base_archetype_id=None,
        )

        derived = ArchetypeSpec.derive(base, {
            "archetype_id": "orb_tight_spread",
            "description": "ORB variant for tight spread conditions",
            "max_trades_per_hour": 24,
        })

        assert derived.archetype_id == "orb_tight_spread"
        assert derived.base_archetype_id == "opening_range_breakout"
        assert derived.description == "ORB variant for tight spread conditions"
        assert derived.max_trades_per_hour == 24
        # Inherited from base
        assert derived.required_features == ["indicators/atr_14", "session/detector"]
        assert derived.base_type == "breakout"
        assert derived.execution_profile == "orb_v1"
        assert derived.v1_priority is True


# =============================================================================
# T12: BaseArchetype ABC methods
# =============================================================================


class _ConcreteArchetype(BaseArchetype):
    """Minimal concrete implementation for testing."""

    @property
    def spec(self) -> ArchetypeSpec:
        return ArchetypeSpec(
            archetype_id="test_archetype",
            base_type="breakout",
            description="Test archetype",
            execution_profile="orb_v1",
            required_features=["indicators/rsi_14"],
            optional_features=["indicators/macd_12_26_9"],
            excluded_features=["indicators/fear_greed"],
            session_tags=["london"],
        )

    def validate_bot_spec(self, bot_spec) -> List[str]:
        errors = []
        spec = self.spec
        for req in spec.required_features:
            if req not in bot_spec.features:
                errors.append(f"Missing required feature: {req}")
        for excl in spec.excluded_features:
            if excl in bot_spec.features:
                errors.append(f"Excluded feature present: {excl}")
        return errors

    def suggest_features(
        self, context: Dict[str, Any]
    ) -> List[str]:
        suggestions = []
        if context.get("regime") == "HIGH_CHAOS":
            suggestions.append("indicators/macd_12_26_9")
        return suggestions


class TestBaseArchetype:
    def test_t12_spec_property_returns_archetype_spec(self):
        impl = _ConcreteArchetype()
        assert isinstance(impl.spec, ArchetypeSpec)
        assert impl.spec.archetype_id == "test_archetype"

    def test_t12_validate_bot_spec_missing_required(self):
        impl = _ConcreteArchetype()
        mock_bot = MagicMock()
        mock_bot.features = []  # Missing all required features
        errors = impl.validate_bot_spec(mock_bot)
        assert len(errors) > 0
        assert any("indicators/rsi_14" in e for e in errors)

    def test_t12_validate_bot_spec_excluded_present(self):
        impl = _ConcreteArchetype()
        mock_bot = MagicMock()
        mock_bot.features = ["indicators/rsi_14", "indicators/fear_greed"]
        errors = impl.validate_bot_spec(mock_bot)
        assert len(errors) > 0
        assert any("indicators/fear_greed" in e for e in errors)

    def test_t12_validate_bot_spec_valid(self):
        impl = _ConcreteArchetype()
        mock_bot = MagicMock()
        mock_bot.features = ["indicators/rsi_14"]  # Only required, no excluded
        errors = impl.validate_bot_spec(mock_bot)
        assert errors == []

    def test_t12_suggest_features(self):
        impl = _ConcreteArchetype()
        suggestions = impl.suggest_features({"regime": "HIGH_CHAOS", "spread": 0.5})
        assert "indicators/macd_12_26_9" in suggestions

    def test_t12_capability_spec_derivation(self):
        impl = _ConcreteArchetype()
        cap = impl.capability_spec()
        assert cap.module_id == "archetype/test_archetype"
        assert "archetype/test_archetype" in cap.provides
        assert cap.requires == ["indicators/rsi_14"]
        assert cap.optional == ["indicators/macd_12_26_9"]
        assert cap.outputs == ["BotSpec"]
        assert cap.compatibility == ["test_archetype"]


# =============================================================================
# T13: ArchetypeRegistry — registration / lookup / indexing
# =============================================================================


def _make_orb_spec(v1_priority: bool = True) -> ArchetypeSpec:
    return ArchetypeSpec(
        archetype_id="opening_range_breakout",
        base_type="breakout",
        description="Opening Range Breakout strategy",
        v1_priority=v1_priority,
        quality_class="native_supported",
        required_features=["indicators/atr_14", "session/detector"],
        optional_features=["indicators/rsi_14"],
        excluded_features=["indicators/sentiment"],
        session_tags=["london", "newyork"],
        confirmations=["spread_ok"],
        execution_profile="orb_v1",
        max_trades_per_hour=12,
        base_archetype_id=None,
    )


def _make_scalper_spec() -> ArchetypeSpec:
    return ArchetypeSpec(
        archetype_id="breakout_scalper",
        base_type="scalping",
        description="Breakout scalping strategy",
        v1_priority=False,
        quality_class="native_supported",
        required_features=["indicators/rsi_14"],
        session_tags=["asian"],
        execution_profile="scalp_fast_v1",
        max_trades_per_hour=60,
        base_archetype_id=None,
    )


class TestArchetypeRegistry:
    def test_t13_register_and_get(self):
        reg = ArchetypeRegistry()
        spec = _make_orb_spec()
        reg.register(spec)
        retrieved = reg.get("opening_range_breakout")
        assert retrieved is not None
        assert retrieved.archetype_id == "opening_range_breakout"
        assert retrieved.base_type == "breakout"

    def test_t13_duplicate_raises(self):
        reg = ArchetypeRegistry()
        reg.register(_make_orb_spec())
        with pytest.raises(DuplicateArchetypeError):
            reg.register(_make_orb_spec())

    def test_t13_list_all(self):
        reg = ArchetypeRegistry()
        reg.register(_make_orb_spec())
        reg.register(_make_scalper_spec())
        all_ids = reg.list_all()
        assert "opening_range_breakout" in all_ids
        assert "breakout_scalper" in all_ids
        assert len(all_ids) == 2

    def test_t13_list_by_base_type(self):
        reg = ArchetypeRegistry()
        reg.register(_make_orb_spec())
        reg.register(_make_scalper_spec())
        breakout_ids = reg.list_by_base_type("breakout")
        scalping_ids = reg.list_by_base_type("scalping")
        assert "opening_range_breakout" in breakout_ids
        assert "breakout_scalper" in scalping_ids
        assert len(breakout_ids) == 1
        assert len(scalping_ids) == 1

    def test_t13_list_by_session(self):
        reg = ArchetypeRegistry()
        reg.register(_make_orb_spec())  # london, newyork
        reg.register(_make_scalper_spec())  # asian
        london_ids = reg.list_by_session("london")
        asian_ids = reg.list_by_session("asian")
        assert "opening_range_breakout" in london_ids
        assert "breakout_scalper" in asian_ids
        # ORB also covers newyork
        newyork_ids = reg.list_by_session("newyork")
        assert "opening_range_breakout" in newyork_ids

    def test_t13_list_v1_priority(self):
        reg = ArchetypeRegistry()
        reg.register(_make_orb_spec(v1_priority=True))
        reg.register(_make_scalper_spec())  # v1_priority=False
        v1_ids = reg.list_v1_priority()
        assert "opening_range_breakout" in v1_ids
        assert "breakout_scalper" not in v1_ids

    def test_t13_compatible_features(self):
        reg = ArchetypeRegistry()
        reg.register(_make_orb_spec())
        features = reg.compatible_features("opening_range_breakout")
        assert "indicators/atr_14" in features
        assert "indicators/rsi_14" in features
        assert "indicators/sentiment" not in features

    def test_t13_compatible_features_unknown_archetype(self):
        reg = ArchetypeRegistry()
        features = reg.compatible_features("unknown_archetype")
        assert features == []

    def test_t13_get_unknown_returns_none(self):
        reg = ArchetypeRegistry()
        assert reg.get("does_not_exist") is None

    def test_t13_derive(self):
        reg = ArchetypeRegistry()
        reg.register(_make_orb_spec())

        derived = reg.derive(
            base_archetype_id="opening_range_breakout",
            derived_id="orb_low_spread",
            overrides={
                "description": "ORB variant for low spread conditions",
                "max_trades_per_hour": 30,
            },
        )

        assert derived.archetype_id == "orb_low_spread"
        assert derived.base_archetype_id == "opening_range_breakout"
        assert derived.description == "ORB variant for low spread conditions"
        assert derived.max_trades_per_hour == 30
        # Inherited
        assert derived.required_features == ["indicators/atr_14", "session/detector"]
        assert derived.base_type == "breakout"
        assert derived.execution_profile == "orb_v1"

    def test_t13_derive_unknown_base_raises(self):
        reg = ArchetypeRegistry()
        with pytest.raises(ArchetypeNotFoundError):
            reg.derive("nonexistent_base", "derived_id", {})

    def test_t13_derive_then_register(self):
        reg = ArchetypeRegistry()
        reg.register(_make_orb_spec())
        derived = reg.derive(
            "opening_range_breakout", "orb_variant_v2", {"max_trades_per_hour": 50}
        )
        reg.register(derived)
        assert len(reg.list_all()) == 2
        assert reg.get("orb_variant_v2") is not None
        assert reg.get("orb_variant_v2").max_trades_per_hour == 50

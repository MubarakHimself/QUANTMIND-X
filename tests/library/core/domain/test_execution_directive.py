"""
Self-Verification Tests for ERR-004: ExecutionDirective approved + rejection_reason fields.
"""
from __future__ import annotations

from src.library.core.domain.execution_directive import ExecutionDirective
from src.library.core.types.enums import RiskMode, TradeDirection


def _make_directive(**overrides):
    """Factory with all required fields plus optional overrides."""
    defaults = dict(
        bot_id="bot-scalper-01",
        direction=TradeDirection.LONG,
        symbol="EURUSD",
        quantity=10000.0,
        risk_mode=RiskMode.STANDARD,
        max_slippage_ticks=2,
        stop_ticks=20,
        timestamp_ms=1744354800000,
        authorization="governor::sig-abc123",
    )
    defaults.update(overrides)
    return ExecutionDirective(**defaults)


class TestExecutionDirectiveApprovedDefault:
    """Test that approved=True by default (backward compatibility)."""

    def test_approved_defaults_to_true(self):
        """approved is True when not specified."""
        directive = _make_directive()
        assert directive.approved is True

    def test_rejection_reason_defaults_to_none(self):
        """rejection_reason is None when not specified."""
        directive = _make_directive()
        assert directive.rejection_reason is None

    def test_explicit_approved_true(self):
        """approved=True is preserved when set explicitly."""
        directive = _make_directive(approved=True)
        assert directive.approved is True
        assert directive.rejection_reason is None


class TestExecutionDirectiveRejection:
    """Test approved=False with rejection_reason."""

    def test_approved_false_with_reason(self):
        """approved=False carries a human-readable rejection_reason."""
        directive = _make_directive(
            approved=False,
            rejection_reason="Position size exceeds per-trade limit",
        )
        assert directive.approved is False
        assert directive.rejection_reason == "Position size exceeds per-trade limit"

    def test_approved_false_reason_is_optional_string(self):
        """rejection_reason accepts any non-None string value."""
        directive = _make_directive(approved=False, rejection_reason="DPR tier below THRESHOLD")
        assert isinstance(directive.rejection_reason, str)
        assert len(directive.rejection_reason) > 0

    def test_approved_true_with_explicit_none_reason(self):
        """approved=True with rejection_reason=None is valid."""
        directive = _make_directive(approved=True, rejection_reason=None)
        assert directive.approved is True
        assert directive.rejection_reason is None


class TestExecutionDirectiveRiskMode:
    """Test that risk_mode still works independently of approved field."""

    def test_halted_risk_mode_approved_false(self):
        """HALTED risk_mode can coexist with approved=False."""
        directive = _make_directive(
            risk_mode=RiskMode.HALTED,
            approved=False,
            rejection_reason="Risk envelope HALTED — market circuit open",
        )
        assert directive.risk_mode == RiskMode.HALTED
        assert directive.approved is False

    def test_clamped_risk_mode_approved_true(self):
        """CLAMPED risk_mode can coexist with approved=True (partial authorization)."""
        directive = _make_directive(
            risk_mode=RiskMode.CLAMPED,
            approved=True,
            rejection_reason=None,
        )
        assert directive.risk_mode == RiskMode.CLAMPED
        assert directive.approved is True

    def test_standard_risk_mode_approved_true(self):
        """STANDARD risk_mode with default approved=True."""
        directive = _make_directive(risk_mode=RiskMode.STANDARD)
        assert directive.risk_mode == RiskMode.STANDARD
        assert directive.approved is True


class TestExecutionDirectiveRoundTrip:
    """Test serialize / deserialize round-trip (Pydantic model validation)."""

    def test_to_dict_round_trip(self):
        """Full directive serializes and deserializes without data loss."""
        original = _make_directive(
            approved=False,
            rejection_reason="DPR threshold breached",
        )
        json_str = original.model_dump_json()
        restored = ExecutionDirective.model_validate_json(json_str)

        assert restored.bot_id == original.bot_id
        assert restored.direction == original.direction
        assert restored.symbol == original.symbol
        assert restored.quantity == original.quantity
        assert restored.risk_mode == original.risk_mode
        assert restored.max_slippage_ticks == original.max_slippage_ticks
        assert restored.stop_ticks == original.stop_ticks
        assert restored.limit_ticks == original.limit_ticks
        assert restored.timestamp_ms == original.timestamp_ms
        assert restored.authorization == original.authorization
        assert restored.approved == original.approved
        assert restored.rejection_reason == original.rejection_reason

    def test_default_round_trip(self):
        """Directive with default approved/rejection_reason round-trips correctly."""
        original = _make_directive()
        json_str = original.model_dump_json()
        restored = ExecutionDirective.model_validate_json(json_str)

        assert restored.approved is True
        assert restored.rejection_reason is None

    def test_to_dict_includes_new_fields(self):
        """model_dump_json output includes approved and rejection_reason."""
        directive = _make_directive(
            approved=False,
            rejection_reason="Position size limit",
        )
        data = directive.model_dump()
        assert "approved" in data
        assert "rejection_reason" in data
        assert data["approved"] is False
        assert data["rejection_reason"] == "Position size limit"

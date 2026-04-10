"""
Self-Verification Tests for Packet 11B: PatternSignal Schema Refinement.
12 tests covering PatternSignal basic fields, PatternConfidence, PatternMetadata,
is_defined refinement, and workflow integration.
"""
from __future__ import annotations

import pytest

from datetime import datetime

from src.library.core.domain.pattern_signal import (
    PatternSignal,
    PatternConfidence,
    PatternMetadata,
)
from src.library.core.types.enums import SignalDirection


class TestPatternSignalBasic:
    """Test existing PatternSignal fields (backward compatibility)."""

    def test_bot_id_field(self):
        sig = PatternSignal(bot_id="bot-001")
        assert sig.bot_id == "bot-001"

    def test_pattern_type_field(self):
        sig = PatternSignal(pattern_type="DOJI")
        assert sig.pattern_type == "DOJI"

    def test_direction_field(self):
        sig = PatternSignal(direction=SignalDirection.BULLISH)
        assert sig.direction == SignalDirection.BULLISH

    def test_confidence_field(self):
        sig = PatternSignal(confidence=0.75)
        assert sig.confidence == 0.75

    def test_confidence_bounds(self):
        """Confidence must be in [0.0, 1.0]."""
        sig = PatternSignal(confidence=0.5)
        assert sig.confidence == 0.5
        # Pydantic validation: values outside range raise errors
        with pytest.raises(Exception):
            PatternSignal(confidence=1.5)

    def test_is_defined_defaults_to_false(self):
        """is_defined is False by default (V1 placeholder)."""
        sig = PatternSignal(bot_id="bot-001", pattern_type="DOJI")
        assert sig.is_defined is False


class TestPatternConfidenceFields:
    """Test PatternConfidence breakdown model."""

    def test_defaults(self):
        conf = PatternConfidence()
        assert conf.structural == 0.0
        assert conf.temporal == 0.0
        assert conf.volume_confirm == 0.0
        assert conf.overall == 0.0

    def test_full_values(self):
        conf = PatternConfidence(
            structural=0.8,
            temporal=0.7,
            volume_confirm=0.6,
            overall=0.72,
        )
        assert conf.structural == 0.8
        assert conf.temporal == 0.7
        assert conf.volume_confirm == 0.6
        assert conf.overall == 0.72

    def test_bounds(self):
        """All fields must be in [0.0, 1.0]."""
        conf = PatternConfidence(structural=0.5, temporal=0.5, volume_confirm=0.5, overall=0.5)
        assert conf.structural == 0.5
        # Pydantic validation: values outside range raise errors
        with pytest.raises(Exception):
            PatternConfidence(structural=-0.1)
        with pytest.raises(Exception):
            PatternConfidence(overall=1.1)


class TestPatternMetadataFields:
    """Test PatternMetadata model."""

    def test_defaults(self):
        meta = PatternMetadata()
        assert meta.pattern_family == ""
        assert meta.time_frame == ""
        assert meta.detected_at_idx == 0
        assert meta.bars_in_pattern == 0
        assert meta.age_bars == 0

    def test_full_values(self):
        meta = PatternMetadata(
            pattern_family="REVERSAL",
            time_frame="SWING",
            detected_at_idx=150,
            bars_in_pattern=5,
            age_bars=2,
        )
        assert meta.pattern_family == "REVERSAL"
        assert meta.time_frame == "SWING"
        assert meta.detected_at_idx == 150
        assert meta.bars_in_pattern == 5
        assert meta.age_bars == 2


class TestPatternSignalRefinement:
    """Test is_defined logic with new fields."""

    def test_is_defined_false_without_breakdown(self):
        """is_defined is False when confidence_breakdown is None."""
        sig = PatternSignal(
            bot_id="bot-001",
            pattern_type="DOJI",
            confidence_breakdown=None,
            metadata=PatternMetadata(),
        )
        assert sig.is_defined is False

    def test_is_defined_false_without_metadata(self):
        """is_defined is False when metadata is None."""
        sig = PatternSignal(
            bot_id="bot-001",
            pattern_type="DOJI",
            confidence_breakdown=PatternConfidence(),
            metadata=None,
        )
        assert sig.is_defined is False

    def test_is_defined_true_with_both_fields(self):
        """is_defined is True when both confidence_breakdown and metadata are set."""
        sig = PatternSignal(
            bot_id="bot-001",
            pattern_type="DOJI",
            confidence_breakdown=PatternConfidence(
                structural=0.8,
                temporal=0.7,
                volume_confirm=0.6,
                overall=0.72,
            ),
            metadata=PatternMetadata(
                pattern_family="REVERSAL",
                time_frame="SWING",
                detected_at_idx=150,
                bars_in_pattern=5,
                age_bars=2,
            ),
        )
        assert sig.is_defined is True

    def test_is_defined_false_with_one_field(self):
        """is_defined is False when only one of the two fields is set."""
        sig = PatternSignal(
            bot_id="bot-001",
            pattern_type="DOJI",
            confidence_breakdown=PatternConfidence(),
            # metadata is None
        )
        assert sig.is_defined is False


class TestPatternSignalIntegration:
    """Test PatternSignal in a workflow context."""

    def test_pattern_signal_with_full_refinement(self):
        """Full PatternSignal with all fields for a complete pattern signal."""
        sig = PatternSignal(
            bot_id="bot-scalper-01",
            pattern_type="ENGULFING",
            direction=SignalDirection.BULLISH,
            confidence=0.78,
            timestamp=datetime(2026, 4, 10, 9, 30, 0),
            confidence_breakdown=PatternConfidence(
                structural=0.9,
                temporal=0.8,
                volume_confirm=0.65,
                overall=0.8,
            ),
            metadata=PatternMetadata(
                pattern_family="REVERSAL",
                time_frame="INTRA",
                detected_at_idx=200,
                bars_in_pattern=2,
                age_bars=1,
            ),
        )
        assert sig.is_defined is True
        assert sig.confidence_breakdown.overall == 0.8
        assert sig.metadata.pattern_family == "REVERSAL"
        assert sig.direction == SignalDirection.BULLISH

    def test_pattern_signal_workflow_composition(self):
        """Simulate composing a pattern signal from multiple sources."""
        # Simulate a pattern detector output
        breakdown = PatternConfidence(
            structural=0.85,
            temporal=0.75,
            volume_confirm=0.70,
            overall=(0.85 + 0.75 + 0.70) / 3.0,
        )
        metadata = PatternMetadata(
            pattern_family="CONTINUATION",
            time_frame="SWING",
            detected_at_idx=500,
            bars_in_pattern=4,
            age_bars=0,
        )
        sig = PatternSignal(
            bot_id="bot-orb-01",
            pattern_type="BULL_FLAG",
            direction=SignalDirection.BULLISH,
            confidence=breakdown.overall,
            timestamp=datetime.now(),
            confidence_breakdown=breakdown,
            metadata=metadata,
        )
        assert sig.is_defined is True
        assert sig.confidence == breakdown.overall
        assert sig.metadata.time_frame == "SWING"

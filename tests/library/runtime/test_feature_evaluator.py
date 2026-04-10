"""Tests for QuantMindLib V1 -- FeatureEvaluator."""

from datetime import datetime
from typing import Any, Dict, Set
from unittest.mock import MagicMock, patch

import pytest

from src.library.core.domain.feature_vector import FeatureVector, FeatureConfidence
from src.library.core.domain.market_context import MarketContext
from src.library.core.types.enums import NewsState, RegimeType
from src.library.features.registry import FeatureRegistry
from src.library.runtime.feature_evaluator import FeatureEvaluator


def _make_fv(
    bot_id: str = "bot-test",
    features: dict = None,
    confidence: dict = None,
) -> FeatureVector:
    """Helper: create a FeatureVector with predictable defaults."""
    return FeatureVector(
        bot_id=bot_id,
        timestamp=datetime.now(),
        features=features or {},
        feature_confidence=confidence or {},
    )


def _mock_feature(
    feature_id: str,
    required_inputs: Set[str],
    output_features: Dict[str, float],
    quality: float = 0.9,
) -> MagicMock:
    """
    Build a mock FeatureModule that returns predictable FeatureVectors.
    """
    mock = MagicMock()
    mock.feature_id = feature_id
    mock.required_inputs = required_inputs
    mock.output_keys = set(output_features.keys())
    mock.quality_class = "native_supported"
    mock.source = "test"
    mock.enabled = True

    confidence = {
        k: FeatureConfidence(source="test", quality=quality, latency_ms=0.1, feed_quality_tag="HIGH")
        for k in output_features
    }
    mock.compute.return_value = _make_fv(features=output_features, confidence=confidence)
    return mock


class TestFeatureEvaluatorInit:
    """Tests for FeatureEvaluator initialization."""

    def test_initializes_with_default_registry(self):
        evaluator = FeatureEvaluator()
        assert evaluator._registry is not None

    def test_initializes_with_provided_registry(self):
        registry = FeatureRegistry()
        evaluator = FeatureEvaluator(registry=registry)
        # Same type and equivalent (Pydantic models create new instances per call)
        assert isinstance(evaluator._registry, FeatureRegistry)
        assert evaluator._registry is not None

    def test_is_not_pydantic_model(self):
        evaluator = FeatureEvaluator()
        # FeatureEvaluator is not a Pydantic model, just a plain Python class
        assert not hasattr(evaluator, "model_config")


class TestEvaluate:
    """Tests for FeatureEvaluator.evaluate()."""

    def test_evaluate_with_empty_inputs_returns_empty_vector(self):
        evaluator = FeatureEvaluator(registry=FeatureRegistry())
        result = evaluator.evaluate("bot-1", [], {})
        assert isinstance(result, FeatureVector)
        assert result.bot_id == "bot-1"
        assert result.features == {}
        assert result.feature_confidence == {}

    def test_evaluate_with_unknown_feature_ids_skips_them(self):
        mock_registry = MagicMock(spec=FeatureRegistry)
        mock_registry.get.return_value = None  # No features registered
        evaluator = FeatureEvaluator(registry=mock_registry)

        result = evaluator.evaluate("bot-2", ["indicators/rsi", "volume/mfi"], {"close_prices": [1.0]})

        assert result.bot_id == "bot-2"
        assert result.features == {}
        assert mock_registry.get.call_count == 2

    def test_evaluate_with_partial_inputs_computes_available(self):
        """Only features whose required inputs are satisfied are computed."""
        # RSI requires close_prices; MFI requires volume. Only close_prices is provided.
        rsi_mock = _mock_feature(
            "indicators/rsi",
            required_inputs={"close_prices"},
            output_features={"rsi_14": 55.0, "rsi_signal": 0.5},
        )
        mfi_mock = _mock_feature(
            "volume/mfi",
            required_inputs={"volume"},
            output_features={"mfi_14": 45.0},
        )

        mock_registry = MagicMock(spec=FeatureRegistry)
        mock_registry.get.side_effect = lambda fid: {
            "indicators/rsi": rsi_mock,
            "volume/mfi": mfi_mock,
        }.get(fid)

        evaluator = FeatureEvaluator(registry=mock_registry)

        result = evaluator.evaluate(
            "bot-partial",
            ["indicators/rsi", "volume/mfi"],
            {"close_prices": [1.0850, 1.0851]},
        )

        # RSI should be computed; MFI skipped (volume not provided)
        assert "rsi_14" in result.features
        assert result.features["rsi_14"] == 55.0
        assert "mfi_14" not in result.features
        # rsi_mock.compute was called
        rsi_mock.compute.assert_called_once()

    def test_evaluate_with_complete_inputs_computes_all(self):
        rsi_mock = _mock_feature(
            "indicators/rsi",
            required_inputs={"close_prices"},
            output_features={"rsi_14": 60.0},
        )
        mfi_mock = _mock_feature(
            "volume/mfi",
            required_inputs={"close_prices", "volume"},
            output_features={"mfi_14": 50.0},
        )

        mock_registry = MagicMock(spec=FeatureRegistry)
        mock_registry.get.side_effect = lambda fid: {
            "indicators/rsi": rsi_mock,
            "volume/mfi": mfi_mock,
        }.get(fid)

        evaluator = FeatureEvaluator(registry=mock_registry)

        result = evaluator.evaluate(
            "bot-full",
            ["indicators/rsi", "volume/mfi"],
            {"close_prices": [1.0], "volume": [1000.0]},
        )

        assert "rsi_14" in result.features
        assert "mfi_14" in result.features
        assert result.features["rsi_14"] == 60.0
        assert result.features["mfi_14"] == 50.0

    def test_evaluate_aggregates_features_from_multiple_modules(self):
        macd_mock = _mock_feature(
            "indicators/macd",
            required_inputs={"close_prices"},
            output_features={"macd_line": 0.002, "macd_signal": 0.001},
        )
        atr_mock = _mock_feature(
            "indicators/atr",
            required_inputs={"high_prices", "low_prices", "close_prices"},
            output_features={"atr_14": 0.0015},
        )

        mock_registry = MagicMock(spec=FeatureRegistry)
        mock_registry.get.side_effect = lambda fid: {
            "indicators/macd": macd_mock,
            "indicators/atr": atr_mock,
        }.get(fid)

        evaluator = FeatureEvaluator(registry=mock_registry)

        result = evaluator.evaluate(
            "bot-agg",
            ["indicators/macd", "indicators/atr"],
            {
                "close_prices": [1.0],
                "high_prices": [1.01],
                "low_prices": [0.99],
            },
        )

        assert result.features["macd_line"] == 0.002
        assert result.features["macd_signal"] == 0.001
        assert result.features["atr_14"] == 0.0015

    def test_evaluate_preserves_timestamp(self):
        evaluator = FeatureEvaluator(registry=FeatureRegistry())
        result = evaluator.evaluate("bot-ts", [], {})
        assert isinstance(result.timestamp, datetime)

    def test_evaluate_includes_confidence_metadata(self):
        rsi_mock = _mock_feature(
            "indicators/rsi",
            required_inputs={"close_prices"},
            output_features={"rsi_14": 65.0},
            quality=0.95,
        )
        mock_registry = MagicMock(spec=FeatureRegistry)
        mock_registry.get.return_value = rsi_mock

        evaluator = FeatureEvaluator(registry=mock_registry)
        result = evaluator.evaluate(
            "bot-conf",
            ["indicators/rsi"],
            {"close_prices": [1.0]},
        )

        assert "rsi_14" in result.feature_confidence
        assert result.feature_confidence["rsi_14"].quality == 0.95
        assert result.feature_confidence["rsi_14"].source == "test"

    def test_evaluate_with_market_context_snapshot(self):
        evaluator = FeatureEvaluator(registry=FeatureRegistry())
        mc = MarketContext(
            regime=RegimeType.TREND_STABLE,
            news_state=NewsState.CLEAR,
            regime_confidence=0.85,
        )

        result = evaluator.evaluate(
            "bot-mc",
            [],
            {"close_prices": [1.0], "market_context": mc},
        )

        assert result.market_context_snapshot is mc

    def test_evaluate_resolves_chained_dependencies(self):
        """
        Feature A requires {x}; Feature B requires {A_output}.
        If A_output is in the output keys of A, B should be computable
        after A is computed (x is provided, then A_output becomes available).
        """
        # Step 1: first feature produces derived input
        derived_mock = _mock_feature(
            "transforms/normalize",
            required_inputs={"close_prices"},
            output_features={"normalized_close": 0.5},
        )
        # Step 2: second feature uses derived output
        follow_mock = _mock_feature(
            "derived/follow",
            required_inputs={"normalized_close"},
            output_features={"follow_signal": 0.8},
        )

        mock_registry = MagicMock(spec=FeatureRegistry)
        mock_registry.get.side_effect = lambda fid: {
            "transforms/normalize": derived_mock,
            "derived/follow": follow_mock,
        }.get(fid)

        evaluator = FeatureEvaluator(registry=mock_registry)

        result = evaluator.evaluate(
            "bot-chain",
            ["transforms/normalize", "derived/follow"],
            {"close_prices": [1.0]},
        )

        # Both should be computed: normalize first (needs close), then follow (needs normalized_close)
        assert "normalized_close" in result.features
        assert "follow_signal" in result.features
        assert result.features["follow_signal"] == 0.8

    def test_evaluate_max_iterations_prevents_infinite_loop(self):
        """
        When circular dependencies exist, max_iterations cap prevents infinite loop.
        """
        cyclic_mock = _mock_feature(
            "cyclic/feature",
            required_inputs={"other_output"},  # Will never be satisfied
            output_features={"cyclic_value": 1.0},
        )
        mock_registry = MagicMock(spec=FeatureRegistry)
        mock_registry.get.return_value = cyclic_mock

        evaluator = FeatureEvaluator(registry=mock_registry)

        # Should not raise, just stop after max_iterations
        result = evaluator.evaluate(
            "bot-cycle",
            ["cyclic/feature"],
            {"some_input": [1.0]},
        )

        # Feature not computed due to unmet dependency
        assert "cyclic_value" not in result.features


class TestCanEvaluate:
    """Tests for FeatureEvaluator.can_evaluate()."""

    def test_can_evaluate_empty_feature_list(self):
        evaluator = FeatureEvaluator(registry=FeatureRegistry())
        can, missing = evaluator.can_evaluate([], {"close_prices"})
        assert can is True
        assert missing == []

    def test_can_evaluate_with_satisfied_dependencies(self):
        rsi_mock = _mock_feature(
            "indicators/rsi",
            required_inputs={"close_prices"},
            output_features={"rsi_14": 55.0},
        )
        mock_registry = MagicMock(spec=FeatureRegistry)
        mock_registry.get.return_value = rsi_mock

        evaluator = FeatureEvaluator(registry=mock_registry)
        can, missing = evaluator.can_evaluate(
            ["indicators/rsi"],
            {"close_prices", "volume"},
        )

        assert can is True
        assert missing == []

    def test_can_evaluate_with_missing_required_inputs(self):
        rsi_mock = _mock_feature(
            "indicators/rsi",
            required_inputs={"close_prices", "volume"},
            output_features={"rsi_14": 55.0},
        )
        mock_registry = MagicMock(spec=FeatureRegistry)
        mock_registry.get.return_value = rsi_mock

        evaluator = FeatureEvaluator(registry=mock_registry)
        can, missing = evaluator.can_evaluate(
            ["indicators/rsi"],
            {"close_prices"},  # volume is missing
        )

        assert can is False
        assert len(missing) == 1
        assert "volume" in missing[0]

    def test_can_evaluate_with_unknown_feature(self):
        mock_registry = MagicMock(spec=FeatureRegistry)
        mock_registry.get.return_value = None

        evaluator = FeatureEvaluator(registry=mock_registry)
        can, missing = evaluator.can_evaluate(["indicators/nonexistent"], set())

        assert can is False
        assert "indicators/nonexistent" in missing[0]

    def test_can_evaluate_with_mixed_satisfied_and_unsatisfied(self):
        rsi_mock = _mock_feature(
            "indicators/rsi",
            required_inputs={"close_prices"},
            output_features={"rsi_14": 55.0},
        )
        mfi_mock = _mock_feature(
            "volume/mfi",
            required_inputs={"volume"},
            output_features={"mfi_14": 45.0},
        )
        unknown_mock = None

        mock_registry = MagicMock(spec=FeatureRegistry)
        mock_registry.get.side_effect = lambda fid: {
            "indicators/rsi": rsi_mock,
            "volume/mfi": mfi_mock,
            "indicators/unknown": unknown_mock,
        }.get(fid)

        evaluator = FeatureEvaluator(registry=mock_registry)
        can, missing = evaluator.can_evaluate(
            ["indicators/rsi", "volume/mfi", "indicators/unknown"],
            {"close_prices"},  # volume missing, unknown unknown
        )

        assert can is False
        assert len(missing) == 2
        # Unknown feature listed by id
        assert any("indicators/unknown" in m for m in missing)
        # MFI missing volume
        assert any("volume" in m for m in missing)

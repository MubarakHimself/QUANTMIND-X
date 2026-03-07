"""
Tests for Advanced Evaluation Metrics.

Tests the metrics module including precision/recall, token tracking,
resource metrics, and advanced criteria.
"""

import pytest
from typing import Any, Dict

from src.agents.evaluation.metrics import (
    PrecisionRecallMetrics,
    TokenMetrics,
    ResourceMetrics,
    EvaluationMetrics,
    MetricsCollector,
    FuzzyMatchCriteria,
    SemanticSimilarityCriteria,
    JSONStructureCriteria,
    TypeCheckCriteria,
)


class TestPrecisionRecallMetrics:
    """Tests for PrecisionRecallMetrics."""

    def test_perfect_precision(self):
        metrics = PrecisionRecallMetrics(true_positives=10, false_positives=0)
        assert metrics.precision == 1.0

    def test_zero_precision(self):
        metrics = PrecisionRecallMetrics(true_positives=0, false_positives=10)
        assert metrics.precision == 0.0

    def test_perfect_recall(self):
        metrics = PrecisionRecallMetrics(true_positives=10, false_negatives=0)
        assert metrics.recall == 1.0

    def test_zero_recall(self):
        metrics = PrecisionRecallMetrics(true_positives=0, false_negatives=10)
        assert metrics.recall == 0.0

    def test_f1_score_perfect(self):
        metrics = PrecisionRecallMetrics(true_positives=10, false_positives=0, false_negatives=0)
        assert metrics.f1_score == 1.0

    def test_f1_score_zero(self):
        metrics = PrecisionRecallMetrics(true_positives=0, false_positives=10, false_negatives=10)
        assert metrics.f1_score == 0.0

    def test_f1_score_balanced(self):
        metrics = PrecisionRecallMetrics(true_positives=8, false_positives=2, false_negatives=2)
        assert 0.79 < metrics.f1_score < 0.81

    def test_to_dict(self):
        metrics = PrecisionRecallMetrics(true_positives=5, false_positives=1, false_negatives=2, true_negatives=10)
        result = metrics.to_dict()
        assert "precision" in result
        assert "recall" in result
        assert "f1_score" in result
        assert result["true_positives"] == 5


class TestTokenMetrics:
    """Tests for TokenMetrics."""

    def test_calculate_from_text(self):
        text = "This is a test string with multiple words"
        metrics = TokenMetrics.calculate_from_text(text, cost_per_token=0.001)
        assert metrics.total_tokens > 0
        assert metrics.total_cost > 0
        assert metrics.input_tokens + metrics.output_tokens == metrics.total_tokens

    def test_empty_text(self):
        metrics = TokenMetrics.calculate_from_text("", cost_per_token=0.001)
        assert metrics.total_tokens == 0
        assert metrics.total_cost == 0

    def test_to_dict(self):
        metrics = TokenMetrics(
            input_tokens=100,
            output_tokens=200,
            total_tokens=300,
            input_cost=0.1,
            output_cost=0.2,
            total_cost=0.3,
            cost_per_token=0.001
        )
        result = metrics.to_dict()
        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 200
        assert result["total_tokens"] == 300


class TestResourceMetrics:
    """Tests for ResourceMetrics."""

    def test_capture(self):
        metrics = ResourceMetrics.capture()
        assert metrics.memory_usage_mb >= 0
        assert metrics.cpu_percent >= 0

    def test_to_dict(self):
        metrics = ResourceMetrics(memory_usage_mb=100.0, cpu_percent=25.5, peak_memory_mb=150.0)
        result = metrics.to_dict()
        assert result["memory_usage_mb"] == 100.0
        assert result["cpu_percent"] == 25.5


class TestEvaluationMetrics:
    """Tests for EvaluationMetrics."""

    def test_to_dict(self):
        from datetime import datetime
        metrics = EvaluationMetrics(
            accuracy=0.95,
            precision=0.92,
            recall=0.93,
            f1_score=0.925,
            latency_ms=50.0,
            throughput=20.0,
            total_cost=0.5,
            total_tokens=500,
            memory_usage_mb=100.0,
            cpu_percent=25.0,
            error_rate=0.05,
            timeout_rate=0.0,
            metadata={"test": "value"}
        )
        result = metrics.to_dict()
        assert result["accuracy"] == 0.95
        assert result["f1_score"] == 0.925
        assert result["test"] == "value"


class TestMetricsCollector:
    """Tests for MetricsCollector."""

    def test_record_and_aggregate(self):
        collector = MetricsCollector()

        metrics1 = EvaluationMetrics(accuracy=0.9, latency_ms=50.0)
        metrics2 = EvaluationMetrics(accuracy=0.8, latency_ms=60.0)

        collector.record(metrics1)
        collector.record(metrics2)

        aggregated = collector.aggregate()
        assert abs(aggregated.accuracy - 0.85) < 0.01
        assert abs(aggregated.latency_ms - 55.0) < 0.01

    def test_empty_aggregate(self):
        collector = MetricsCollector()
        aggregated = collector.aggregate()
        assert aggregated.accuracy == 0.0


class TestFuzzyMatchCriteria:
    """Tests for FuzzyMatchCriteria."""

    def test_exact_match(self):
        criteria = FuzzyMatchCriteria()
        result = criteria.evaluate("hello", "hello", {})
        assert result == 1.0

    def test_similar_strings(self):
        criteria = FuzzyMatchCriteria(threshold=0.5)
        result = criteria.evaluate("hello", "hallo", {})
        assert 0.5 < result < 1.0

    def test_different_strings(self):
        criteria = FuzzyMatchCriteria(threshold=0.8)
        result = criteria.evaluate("hello", "world", {})
        assert result < 0.5

    def test_empty_strings(self):
        criteria = FuzzyMatchCriteria()
        result = criteria.evaluate("", "hello", {})
        assert result == 0.0

    def test_list_similarity(self):
        criteria = FuzzyMatchCriteria()
        result = criteria.evaluate([1, 2, 3, 4], [1, 2, 5], {})
        assert result == 0.5  # 1 out of 2 matched

    def test_dict_similarity(self):
        criteria = FuzzyMatchCriteria()
        result = criteria.evaluate(
            {"a": 1, "b": 2, "c": 3},
            {"a": 1, "b": 2},
            {}
        )
        assert result == 1.0  # a=1 and b=2 match exactly


class TestSemanticSimilarityCriteria:
    """Tests for SemanticSimilarityCriteria."""

    def test_identical_strings(self):
        criteria = SemanticSimilarityCriteria()
        result = criteria.evaluate("the quick brown fox", "the quick brown fox", {})
        assert result == 1.0

    def test_common_keywords(self):
        criteria = SemanticSimilarityCriteria()
        result = criteria.evaluate("machine learning is great", "learning artificial intelligence", {})
        # Should have "learning" in common
        assert 0 < result <= 1.0


class TestJSONStructureCriteria:
    """Tests for JSONStructureCriteria."""

    def test_exact_match(self):
        criteria = JSONStructureCriteria()
        result = criteria.evaluate(
            {"a": 1, "b": 2},
            {"a": 1, "b": 2},
            {}
        )
        assert result == 1.0

    def test_partial_keys(self):
        criteria = JSONStructureCriteria()
        result = criteria.evaluate(
            {"a": 1, "b": 2, "c": 3},
            {"a": 1, "b": 2},
            {}
        )
        assert result == 1.0

    def test_nested_structure(self):
        criteria = JSONStructureCriteria()
        result = criteria.evaluate(
            {"user": {"name": "John", "age": 30}},
            {"user": {"name": "John"}},
            {}
        )
        assert result == 1.0


class TestTypeCheckCriteria:
    """Tests for TypeCheckCriteria."""

    def test_matching_types(self):
        criteria = TypeCheckCriteria()
        result = criteria.evaluate(42, "string", {})
        assert result == 0.0  # Different types

    def test_schema_validation(self):
        criteria = TypeCheckCriteria(type_schema={"name": str, "age": int})
        result = criteria.evaluate(
            {"name": "John", "age": 30, "extra": "data"},
            {},
            {}
        )
        assert result == 1.0  # Both schema types match

    def test_schema_mismatch(self):
        criteria = TypeCheckCriteria(type_schema={"name": str, "age": int})
        result = criteria.evaluate(
            {"name": "John", "age": "30"},
            {},
            {}
        )
        assert result == 0.5  # age is str, not int

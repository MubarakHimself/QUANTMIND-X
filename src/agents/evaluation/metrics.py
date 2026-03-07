"""
Advanced Evaluation Metrics

Provides additional metrics for comprehensive agent evaluation including
precision, recall, F1 score, token tracking, and resource utilization.
"""

import time
import psutil
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
from enum import Enum


class MetricCategory(Enum):
    """Categories of evaluation metrics."""
    ACCURACY = "accuracy"
    PERFORMANCE = "performance"
    RESOURCE = "resource"
    QUALITY = "quality"
    COST = "cost"


@dataclass
class PrecisionRecallMetrics:
    """Precision, Recall, and F1 Score metrics for classification tasks."""
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_negatives: int = 0

    @property
    def precision(self) -> float:
        """Calculate precision."""
        if self.true_positives + self.false_positives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_positives)

    @property
    def recall(self) -> float:
        """Calculate recall."""
        if self.true_positives + self.false_negatives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_negatives)

    @property
    def f1_score(self) -> float:
        """Calculate F1 score."""
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)

    @property
    def accuracy(self) -> float:
        """Calculate accuracy."""
        total = self.true_positives + self.true_negatives + self.false_positives + self.false_negatives
        if total == 0:
            return 0.0
        return (self.true_positives + self.true_negatives) / total

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "accuracy": self.accuracy,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "true_negatives": self.true_negatives,
        }


@dataclass
class TokenMetrics:
    """Token usage and cost metrics."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    input_cost: float = 0.0
    output_cost: float = 0.0
    total_cost: float = 0.0
    cost_per_token: float = 0.0

    @staticmethod
    def calculate_from_text(
        text: str,
        cost_per_token: float = 0.0,
        input_ratio: float = 0.3
    ) -> "TokenMetrics":
        """Estimate token metrics from text."""
        # Rough estimate: ~4 characters per token
        estimated_tokens = len(text) // 4
        input_tokens = int(estimated_tokens * input_ratio)
        output_tokens = estimated_tokens - input_tokens

        total_cost = estimated_tokens * cost_per_token

        return TokenMetrics(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=estimated_tokens,
            input_cost=input_tokens * cost_per_token,
            output_cost=output_tokens * cost_per_token,
            total_cost=total_cost,
            cost_per_token=cost_per_token
        )

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "input_cost": self.input_cost,
            "output_cost": self.output_cost,
            "total_cost": self.total_cost,
        }


@dataclass
class ResourceMetrics:
    """Resource utilization metrics."""
    memory_usage_mb: float = 0.0
    cpu_percent: float = 0.0
    peak_memory_mb: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @staticmethod
    def capture() -> "ResourceMetrics":
        """Capture current resource usage."""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()

        return ResourceMetrics(
            memory_usage_mb=memory_info.rss / (1024 * 1024),
            cpu_percent=process.cpu_percent(interval=0.1),
            peak_memory_mb=memory_info.rss / (1024 * 1024),
        )

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_percent": self.cpu_percent,
            "peak_memory_mb": self.peak_memory_mb,
        }


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics container."""
    # Core metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0

    # Performance metrics
    latency_ms: float = 0.0
    throughput: float = 0.0

    # Cost metrics
    total_cost: float = 0.0
    total_tokens: int = 0

    # Resource metrics
    memory_usage_mb: float = 0.0
    cpu_percent: float = 0.0

    # Quality metrics
    error_rate: float = 0.0
    timeout_rate: float = 0.0

    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert all metrics to dictionary."""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "latency_ms": self.latency_ms,
            "throughput": self.throughput,
            "total_cost": self.total_cost,
            "total_tokens": self.total_tokens,
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_percent": self.cpu_percent,
            "error_rate": self.error_rate,
            "timeout_rate": self.timeout_rate,
            "timestamp": self.timestamp.isoformat(),
            **self.metadata
        }


class MetricsCollector:
    """Collects and aggregates evaluation metrics."""

    def __init__(self):
        self._metrics: List[EvaluationMetrics] = []
        self._start_time: Optional[float] = None
        self._resource_start: Optional[ResourceMetrics] = None

    def start(self) -> None:
        """Start metrics collection."""
        self._start_time = time.perf_counter()
        self._resource_start = ResourceMetrics.capture()

    def stop(self) -> None:
        """Stop metrics collection."""
        if self._resource_start:
            end_resource = ResourceMetrics.capture()
            self._resource_start.peak_memory_mb = max(
                self._resource_start.peak_memory_mb,
                end_resource.memory_usage_mb
            )

    def record(self, metrics: EvaluationMetrics) -> None:
        """Record metrics."""
        self._metrics.append(metrics)

    def aggregate(self) -> EvaluationMetrics:
        """Aggregate all recorded metrics."""
        if not self._metrics:
            return EvaluationMetrics()

        return EvaluationMetrics(
            accuracy=sum(m.accuracy for m in self._metrics) / len(self._metrics),
            precision=sum(m.precision for m in self._metrics) / len(self._metrics),
            recall=sum(m.recall for m in self._metrics) / len(self._metrics),
            f1_score=sum(m.f1_score for m in self._metrics) / len(self._metrics),
            latency_ms=sum(m.latency_ms for m in self._metrics) / len(self._metrics),
            total_cost=sum(m.total_cost for m in self._metrics),
            total_tokens=sum(m.total_tokens for m in self._metrics),
            memory_usage_mb=sum(m.memory_usage_mb for m in self._metrics) / len(self._metrics),
            cpu_percent=sum(m.cpu_percent for m in self._metrics) / len(self._metrics),
            error_rate=sum(m.error_rate for m in self._metrics) / len(self._metrics),
            timeout_rate=sum(m.timeout_rate for m in self._metrics) / len(self._metrics),
        )

    def get_all(self) -> List[EvaluationMetrics]:
        """Get all recorded metrics."""
        return self._metrics.copy()


# Advanced evaluation criteria

class FuzzyMatchCriteria:
    """Fuzzy string matching criteria using Levenshtein distance."""

    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold

    def evaluate(self, actual: Any, expected: Any, metadata: Dict[str, Any]) -> float:
        """Evaluate using fuzzy string matching."""
        if isinstance(actual, str) and isinstance(expected, str):
            return self._string_similarity(actual, expected)
        elif isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
            return self._list_similarity(list(actual), list(expected))
        elif isinstance(actual, dict) and isinstance(expected, dict):
            return self._dict_similarity(actual, expected)
        return 1.0 if actual == expected else 0.0

    def _string_similarity(self, s1: str, s2: str) -> float:
        """Calculate string similarity using Levenshtein distance."""
        if s1 == s2:
            return 1.0
        if not s1 or not s2:
            return 0.0

        len1, len2 = len(s1), len(s2)
        # Create distance matrix
        dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

        for i in range(len1 + 1):
            dp[i][0] = i
        for j in range(len2 + 1):
            dp[0][j] = j

        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                cost = 0 if s1[i-1] == s2[j-1] else 1
                dp[i][j] = min(
                    dp[i-1][j] + 1,      # deletion
                    dp[i][j-1] + 1,      # insertion
                    dp[i-1][j-1] + cost  # substitution
                )

        distance = dp[len1][len2]
        max_len = max(len1, len2)
        similarity = 1 - (distance / max_len)
        return similarity if similarity >= self.threshold else similarity * 0.5

    def _list_similarity(self, actual: List, expected: List) -> float:
        """Calculate list similarity."""
        if not expected:
            return 1.0 if not actual else 0.0
        if not actual:
            return 0.0

        matches = sum(1 for a in actual if a in expected)
        return matches / max(len(expected), len(actual))

    def _dict_similarity(self, actual: Dict, expected: Dict) -> float:
        """Calculate dictionary similarity."""
        if not expected:
            return 1.0 if not actual else 0.0

        matches = sum(1 for k, v in expected.items() if actual.get(k) == v)
        partial_matches = sum(
            0.5 for k, v in expected.items()
            if k in actual and actual.get(k) != v
        )
        return (matches + partial_matches) / len(expected)


class SemanticSimilarityCriteria:
    """Semantic similarity criteria using embeddings."""

    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
        self._embedding_cache: Dict[str, List[float]] = {}

    def evaluate(self, actual: Any, expected: Any, metadata: Dict[str, Any]) -> float:
        """Evaluate using semantic similarity."""
        # Simple keyword-based similarity as fallback
        # In production, would use actual embeddings
        if isinstance(actual, str) and isinstance(expected, str):
            return self._keyword_similarity(actual, expected)
        return 1.0 if actual == expected else 0.0

    def _keyword_similarity(self, s1: str, s2: str) -> float:
        """Calculate keyword-based similarity."""
        words1 = set(s1.lower().split())
        words2 = set(s2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union) if union else 0.0


class JSONStructureCriteria:
    """JSON structure validation criteria."""

    def __init__(self, strict: bool = False):
        self.strict = strict

    def evaluate(self, actual: Any, expected: Any, metadata: Dict[str, Any]) -> float:
        """Evaluate JSON structure matching."""
        try:
            if isinstance(actual, str):
                import json
                actual = json.loads(actual)
            if isinstance(expected, str):
                import json
                expected = json.loads(expected)
        except (json.JSONDecodeError, ValueError):
            pass

        if not isinstance(actual, dict) or not isinstance(expected, dict):
            return 1.0 if actual == expected else 0.0

        # Check required keys
        expected_keys = set(expected.keys())
        actual_keys = set(actual.keys())

        if self.strict:
            # All expected keys must be present
            key_match = 1.0 if expected_keys == actual_keys else 0.0
        else:
            # Partial key match
            key_match = len(expected_keys & actual_keys) / len(expected_keys) if expected_keys else 1.0

        # Check nested structures
        nested_scores = []
        for key in expected_keys & actual_keys:
            if isinstance(expected[key], dict) and isinstance(actual.get(key), dict):
                nested_scores.append(self.evaluate(actual[key], expected[key], metadata))
            elif expected[key] == actual.get(key):
                nested_scores.append(1.0)
            else:
                nested_scores.append(0.0)

        nested_avg = sum(nested_scores) / len(nested_scores) if nested_scores else 1.0

        return (key_match + nested_avg) / 2


class TypeCheckCriteria:
    """Type validation criteria."""

    def __init__(self, type_schema: Optional[Dict[str, type]] = None):
        self.type_schema = type_schema or {}

    def evaluate(self, actual: Any, expected: Any, metadata: Dict[str, Any]) -> float:
        """Evaluate type correctness."""
        if self.type_schema:
            # Check against schema
            matches = 0
            total = len(self.type_schema)

            for key, expected_type in self.type_schema.items():
                if key in actual:
                    if isinstance(actual[key], expected_type):
                        matches += 1

            return matches / total if total > 0 else 1.0

        # Fallback: check if types match
        return 1.0 if type(actual) == type(expected) else 0.0

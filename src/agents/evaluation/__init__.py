"""
QuantMind Agent Evaluation Framework

This module provides evaluation capabilities for agent test cases,
benchmarks, and success metrics tracking.
"""

from src.agents.evaluation.evaluator import (
    Evaluator,
    EvaluationResult,
    EvaluationReport,
    TestCase,
    ExactMatchCriteria,
    PartialMatchCriteria,
    ThresholdCriteria,
    EvaluationCriteria,
)
from src.agents.evaluation.benchmarks import (
    BenchmarkSuite,
    BenchmarkResult,
    BenchmarkConfig,
    BenchmarkSuiteReport,
)
from src.agents.evaluation.metrics import (
    MetricsCollector,
    EvaluationMetrics,
    TokenMetrics,
    ResourceMetrics,
    PrecisionRecallMetrics,
    FuzzyMatchCriteria,
    SemanticSimilarityCriteria,
    JSONStructureCriteria,
    TypeCheckCriteria,
    MetricCategory,
)
from src.agents.evaluation.realtime import (
    RealTimeEvaluator,
    StreamingEvaluationConfig,
    EvaluationEvent,
    EvaluationProgress,
    EvaluationStatus,
    EventType,
    EventEmitter,
    AgentComparator,
    LiveMetricsPublisher,
)

__all__ = [
    # Core evaluator
    "Evaluator",
    "EvaluationResult",
    "EvaluationReport",
    "TestCase",
    "ExactMatchCriteria",
    "PartialMatchCriteria",
    "ThresholdCriteria",
    "EvaluationCriteria",
    # Benchmarks
    "BenchmarkSuite",
    "BenchmarkResult",
    "BenchmarkConfig",
    "BenchmarkSuiteReport",
    # Metrics
    "MetricsCollector",
    "EvaluationMetrics",
    "TokenMetrics",
    "ResourceMetrics",
    "PrecisionRecallMetrics",
    "FuzzyMatchCriteria",
    "SemanticSimilarityCriteria",
    "JSONStructureCriteria",
    "TypeCheckCriteria",
    "MetricCategory",
    # Real-time evaluation
    "RealTimeEvaluator",
    "StreamingEvaluationConfig",
    "EvaluationEvent",
    "EvaluationProgress",
    "EvaluationStatus",
    "EventType",
    "EventEmitter",
    "AgentComparator",
    "LiveMetricsPublisher",
]

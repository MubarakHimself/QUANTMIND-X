"""
Agent Evaluator Module

Provides the Evaluator class for running test cases against agents
and computing success metrics.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Protocol, Union
from enum import Enum

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of evaluation metrics."""
    ACCURACY = "accuracy"
    LATENCY = "latency"
    COST = "cost"
    CUSTOM = "custom"


@dataclass
class TestCase:
    """Represents a single test case for agent evaluation."""
    id: str
    name: str
    input_data: Dict[str, Any]
    expected_output: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: float = 30.0
    tags: List[str] = field(default_factory=list)


@dataclass
class EvaluationResult:
    """Result of a single test case evaluation."""
    test_case_id: str
    passed: bool
    actual_output: Any
    expected_output: Any
    metrics: Dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationReport:
    """Aggregated evaluation report for multiple test cases."""
    total_tests: int
    passed_tests: int
    failed_tests: int
    pass_rate: float
    avg_latency_ms: float
    total_cost: float
    results: List[EvaluationResult]
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


class EvaluationCriteria(Protocol):
    """Protocol for custom evaluation criteria."""
    def evaluate(self, actual: Any, expected: Any, metadata: Dict[str, Any]) -> float:
        """Return a score between 0.0 and 1.0."""
        ...


class ExactMatchCriteria:
    """Exact match evaluation criteria."""
    def evaluate(self, actual: Any, expected: Any, metadata: Dict[str, Any]) -> float:
        return 1.0 if actual == expected else 0.0


class PartialMatchCriteria:
    """Partial match evaluation criteria for dicts/lists."""
    def evaluate(self, actual: Any, expected: Any, metadata: Dict[str, Any]) -> float:
        if isinstance(actual, dict) and isinstance(expected, dict):
            matches = sum(1 for k, v in expected.items() if actual.get(k) == v)
            return matches / len(expected) if expected else 1.0
        if isinstance(actual, list) and isinstance(expected, list):
            matches = sum(1 for a, e in zip(actual, expected) if a == e)
            return matches / max(len(expected), 1)
        return 1.0 if actual == expected else 0.0


class ThresholdCriteria:
    """Threshold-based evaluation for numeric values."""
    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold

    def evaluate(self, actual: Any, expected: Any, metadata: Dict[str, Any]) -> float:
        if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
            if expected == 0:
                return 1.0 if actual == 0 else 0.0
            ratio = min(actual / expected, expected / actual)
            return 1.0 if ratio >= self.threshold else ratio
        return 1.0 if actual == expected else 0.0


class Evaluator:
    """
    Evaluator class for running test cases against agents.

    Supports custom evaluation criteria, multiple metric types,
    and detailed reporting.
    """

    def __init__(
        self,
        agent: Any = None,
        evaluation_criteria: Optional[EvaluationCriteria] = None,
        cost_per_token: float = 0.0,
        default_threshold: float = 0.8
    ):
        """
        Initialize the evaluator.

        Args:
            agent: The agent to evaluate (must have an invoke or run method)
            evaluation_criteria: Custom evaluation criteria implementation
            cost_per_token: Cost per token for cost calculations
            default_threshold: Default threshold for passing tests
        """
        self.agent = agent
        self.cost_per_token = cost_per_token
        self.default_threshold = default_threshold
        self.evaluation_criteria = evaluation_criteria or ExactMatchCriteria()
        self._custom_criteria: Dict[str, EvaluationCriteria] = {}

    def register_criteria(self, name: str, criteria: EvaluationCriteria) -> None:
        """Register a named custom evaluation criteria."""
        self._custom_criteria[name] = criteria

    async def run_test_case(self, test_case: TestCase) -> EvaluationResult:
        """
        Run a single test case against the agent.

        Args:
            test_case: The test case to run

        Returns:
            EvaluationResult with test outcome and metrics
        """
        start_time = time.perf_counter()

        try:
            actual_output = await self._execute_agent(test_case)
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Calculate accuracy using criteria
            accuracy = self.evaluation_criteria.evaluate(
                actual_output,
                test_case.expected_output,
                test_case.metadata
            )

            # Calculate cost if token tracking available
            cost = self._calculate_cost(actual_output, test_case)

            passed = accuracy >= self.default_threshold

            return EvaluationResult(
                test_case_id=test_case.id,
                passed=passed,
                actual_output=actual_output,
                expected_output=test_case.expected_output,
                metrics={
                    "accuracy": accuracy,
                    "cost": cost,
                    "threshold": self.default_threshold
                },
                latency_ms=latency_ms,
                error=None,
                metadata=test_case.metadata
            )

        except asyncio.TimeoutError:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return EvaluationResult(
                test_case_id=test_case.id,
                passed=False,
                actual_output=None,
                expected_output=test_case.expected_output,
                metrics={"accuracy": 0.0, "cost": 0.0},
                latency_ms=latency_ms,
                error=f"Test timed out after {test_case.timeout_seconds}s"
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.exception(f"Error running test case {test_case.id}")
            return EvaluationResult(
                test_case_id=test_case.id,
                passed=False,
                actual_output=None,
                expected_output=test_case.expected_output,
                metrics={"accuracy": 0.0, "cost": 0.0},
                latency_ms=latency_ms,
                error=str(e)
            )

    async def _execute_agent(self, test_case: TestCase) -> Any:
        """Execute the agent with the test case input."""
        if self.agent is None:
            raise ValueError("No agent configured for evaluation")

        # Support different agent interfaces
        if hasattr(self.agent, "invoke"):
            if asyncio.iscoroutinefunction(self.agent.invoke):
                return await asyncio.wait_for(
                    self.agent.invoke(test_case.input_data),
                    timeout=test_case.timeout_seconds
                )
            return self.agent.invoke(test_case.input_data)

        if hasattr(self.agent, "run"):
            if asyncio.iscoroutinefunction(self.agent.run):
                return await asyncio.wait_for(
                    self.agent.run(test_case.input_data),
                    timeout=test_case.timeout_seconds
                )
            return self.agent.run(test_case.input_data)

        if callable(self.agent):
            return await asyncio.wait_for(
                self.agent(test_case.input_data),
                timeout=test_case.timeout_seconds
            )

        raise ValueError("Agent must have invoke, run method or be callable")

    def _calculate_cost(self, output: Any, test_case: TestCase) -> float:
        """Calculate the cost of the agent execution."""
        if self.cost_per_token == 0 or output is None:
            return 0.0

        # Estimate tokens from output
        output_str = str(output)
        estimated_tokens = len(output_str) // 4
        return estimated_tokens * self.cost_per_token

    async def evaluate(
        self,
        test_cases: List[TestCase],
        parallel: bool = True,
        stop_on_first_failure: bool = False
    ) -> EvaluationReport:
        """
        Run multiple test cases and generate an evaluation report.

        Args:
            test_cases: List of test cases to run
            parallel: Whether to run tests in parallel
            stop_on_first_failure: Stop evaluation after first failure

        Returns:
            EvaluationReport with aggregated results
        """
        results: List[EvaluationResult] = []

        if parallel:
            results = await asyncio.gather(
                *[self.run_test_case(tc) for tc in test_cases],
                return_exceptions=False
            )
        else:
            for tc in test_cases:
                result = await self.run_test_case(tc)
                results.append(result)

                if stop_on_first_failure and not result.passed:
                    logger.info(f"Stopping evaluation after failure: {tc.id}")
                    break

        return self._generate_report(results)

    def _generate_report(self, results: List[EvaluationResult]) -> EvaluationReport:
        """Generate an aggregated report from results."""
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed

        latencies = [r.latency_ms for r in results]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

        total_cost = sum(r.metrics.get("cost", 0.0) for r in results)

        # Aggregate custom metrics
        custom_metrics: Dict[str, float] = {}
        for r in results:
            for key, value in r.metrics.items():
                if key not in ("accuracy", "cost", "threshold"):
                    if key not in custom_metrics:
                        custom_metrics[key] = []
                    custom_metrics[key].append(value)

        # Average custom metrics
        custom_metrics = {
            k: sum(v) / len(v) for k, v in custom_metrics.items()
        }

        return EvaluationReport(
            total_tests=total,
            passed_tests=passed,
            failed_tests=failed,
            pass_rate=passed / total if total > 0 else 0.0,
            avg_latency_ms=avg_latency,
            total_cost=total_cost,
            results=results,
            custom_metrics=custom_metrics
        )

    def evaluate_sync(
        self,
        test_cases: List[TestCase],
        parallel: bool = True
    ) -> EvaluationReport:
        """Synchronous wrapper for evaluate."""
        return asyncio.run(self.evaluate(test_cases, parallel))

    def add_custom_metric(
        self,
        name: str,
        func: Callable[[Any, Any, Dict[str, Any]], float]
    ) -> None:
        """Add a custom metric function."""
        class CustomCriteria:
            def __init__(self, fn):
                self.fn = fn

            def evaluate(self, actual, expected, metadata):
                return self.fn(actual, expected, metadata)

        self.register_criteria(name, CustomCriteria(func))

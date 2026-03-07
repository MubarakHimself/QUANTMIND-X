"""
Tests for the Agent Evaluation Framework.

Tests the Evaluator class, BenchmarkSuite, and related components.
"""

import asyncio
import pytest
from typing import Any, Dict

from src.agents.evaluation.evaluator import (
    Evaluator,
    TestCase,
    EvaluationResult,
    EvaluationReport,
    ExactMatchCriteria,
    PartialMatchCriteria,
    ThresholdCriteria,
)
from src.agents.evaluation.benchmarks import (
    BenchmarkSuite,
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkSuiteReport,
)


# Mock agent for testing
class MockAgent:
    """Mock agent for testing the evaluator."""

    def __init__(self, should_fail: bool = False, latency: float = 0.01):
        self.should_fail = should_fail
        self.latency = latency
        self.call_count = 0

    async def invoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate agent invocation with configurable latency."""
        await asyncio.sleep(self.latency)
        self.call_count += 1

        if self.should_fail:
            raise ValueError("Simulated agent failure")

        # Return input with some transformation
        return {
            "result": input_data.get("value", 0) * 2,
            "status": "success",
            "id": self.call_count
        }


class TestExactMatchCriteria:
    """Tests for ExactMatchCriteria."""

    def test_exact_match_returns_one(self):
        criteria = ExactMatchCriteria()
        result = criteria.evaluate("hello", "hello", {})
        assert result == 1.0

    def test_no_match_returns_zero(self):
        criteria = ExactMatchCriteria()
        result = criteria.evaluate("hello", "world", {})
        assert result == 0.0

    def test_numeric_exact_match(self):
        criteria = ExactMatchCriteria()
        result = criteria.evaluate(42, 42, {})
        assert result == 1.0


class TestPartialMatchCriteria:
    """Tests for PartialMatchCriteria."""

    def test_dict_partial_match(self):
        criteria = PartialMatchCriteria()
        actual = {"a": 1, "b": 2, "c": 3}
        expected = {"a": 1, "b": 2}
        result = criteria.evaluate(actual, expected, {})
        assert result == 1.0

    def test_dict_partial_match_partial(self):
        criteria = PartialMatchCriteria()
        actual = {"a": 1, "b": 2, "c": 3}
        expected = {"a": 1, "b": 99}
        result = criteria.evaluate(actual, expected, {})
        assert result == 0.5  # 1 out of 2 matches

    def test_list_partial_match(self):
        criteria = PartialMatchCriteria()
        actual = [1, 2, 3, 4, 5]
        expected = [1, 2, 3]
        result = criteria.evaluate(actual, expected, {})
        assert result == 1.0


class TestThresholdCriteria:
    """Tests for ThresholdCriteria."""

    def test_within_threshold(self):
        criteria = ThresholdCriteria(threshold=0.9)
        result = criteria.evaluate(100, 100, {})
        assert result == 1.0

    def test_close_to_expected(self):
        criteria = ThresholdCriteria(threshold=0.9)
        result = criteria.evaluate(95, 100, {})
        assert result == 1.0  # 95/100 = 0.95 which is >= 0.9 threshold

    def test_below_threshold(self):
        criteria = ThresholdCriteria(threshold=0.9)
        result = criteria.evaluate(50, 100, {})
        assert result == 0.5


class TestEvaluator:
    """Tests for the Evaluator class."""

    @pytest.mark.asyncio
    async def test_evaluator_passes_matching_output(self):
        """Test that evaluator passes when output matches."""
        agent = MockAgent()
        evaluator = Evaluator(agent=agent, default_threshold=1.0, evaluation_criteria=PartialMatchCriteria())

        test_case = TestCase(
            id="test_1",
            name="Basic test",
            input_data={"value": 5},
            expected_output={"result": 10, "status": "success"}
        )

        result = await evaluator.run_test_case(test_case)

        assert result.passed is True
        assert result.test_case_id == "test_1"
        assert result.actual_output["result"] == 10

    @pytest.mark.asyncio
    async def test_evaluator_fails_on_mismatch(self):
        """Test that evaluator fails when output doesn't match."""
        agent = MockAgent()
        evaluator = Evaluator(agent=agent, default_threshold=1.0)

        test_case = TestCase(
            id="test_2",
            name="Failure test",
            input_data={"value": 5},
            expected_output={"result": 999, "status": "success"}  # Wrong expected
        )

        result = await evaluator.run_test_case(test_case)

        assert result.passed is False
        assert result.metrics["accuracy"] == 0.0

    @pytest.mark.asyncio
    async def test_evaluator_handles_timeout(self):
        """Test that evaluator handles test timeouts."""
        slow_agent = MockAgent(latency=10.0)
        evaluator = Evaluator(agent=slow_agent)

        test_case = TestCase(
            id="test_timeout",
            name="Timeout test",
            input_data={"value": 1},
            expected_output={"result": 2},
            timeout_seconds=0.1  # Very short timeout
        )

        result = await evaluator.run_test_case(test_case)

        assert result.passed is False
        assert "timed out" in result.error.lower()

    @pytest.mark.asyncio
    async def test_evaluator_handles_errors(self):
        """Test that evaluator handles agent errors gracefully."""
        failing_agent = MockAgent(should_fail=True)
        evaluator = Evaluator(agent=failing_agent)

        test_case = TestCase(
            id="test_error",
            name="Error test",
            input_data={"value": 1},
            expected_output={"result": 2}
        )

        result = await evaluator.run_test_case(test_case)

        assert result.passed is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_evaluate_multiple_test_cases(self):
        """Test running multiple test cases."""
        agent = MockAgent()
        evaluator = Evaluator(agent=agent, evaluation_criteria=PartialMatchCriteria())

        test_cases = [
            TestCase(id=f"test_{i}", name=f"Test {i}", input_data={"value": i}, expected_output={"result": i * 2})
            for i in range(5)
        ]

        report = await evaluator.evaluate(test_cases)

        assert report.total_tests == 5
        assert report.passed_tests == 5
        assert report.failed_tests == 0
        assert report.pass_rate == 1.0
        assert len(report.results) == 5

    @pytest.mark.asyncio
    async def test_evaluate_calculates_latency(self):
        """Test that latency is calculated correctly."""
        agent = MockAgent(latency=0.05)
        evaluator = Evaluator(agent=agent)

        test_case = TestCase(
            id="test_latency",
            name="Latency test",
            input_data={"value": 1},
            expected_output={"result": 2}
        )

        result = await evaluator.run_test_case(test_case)

        assert result.latency_ms > 0

    @pytest.mark.asyncio
    async def test_parallel_evaluation(self):
        """Test parallel evaluation of test cases."""
        agent = MockAgent()
        evaluator = Evaluator(agent=agent)

        test_cases = [
            TestCase(id=f"test_{i}", name=f"Test {i}", input_data={"value": i}, expected_output={"result": i * 2})
            for i in range(4)
        ]

        report = await evaluator.evaluate(test_cases, parallel=True)

        assert report.total_tests == 4
        assert agent.call_count == 4


class TestBenchmarkSuite:
    """Tests for the BenchmarkSuite class."""

    @pytest.mark.asyncio
    async def test_benchmark_runs_successfully(self):
        """Test that benchmark runs without errors."""
        async def workload():
            await asyncio.sleep(0.01)

        suite = BenchmarkSuite(suite_name="test_suite")
        config = BenchmarkConfig(
            name="test_benchmark",
            iterations=5,
            warmup_iterations=0
        )

        result = await suite.run_benchmark(config, workload)

        assert result.config_name == "test_benchmark"
        assert result.iterations == 5
        assert result.errors == 0

    @pytest.mark.asyncio
    async def test_benchmark_calculates_metrics(self):
        """Test that benchmark calculates timing metrics."""
        async def workload():
            await asyncio.sleep(0.01)

        suite = BenchmarkSuite()
        config = BenchmarkConfig(name="metrics_test", iterations=5, warmup_iterations=0)

        result = await suite.run_benchmark(config, workload)

        assert result.avg_latency_ms > 0
        assert result.min_latency_ms > 0
        assert result.max_latency_ms > result.min_latency_ms

    @pytest.mark.asyncio
    async def test_benchmark_handles_errors(self):
        """Test that benchmark handles workload errors."""

        async def failing_workload():
            await asyncio.sleep(0.01)
            raise ValueError("Workload error")

        suite = BenchmarkSuite()
        config = BenchmarkConfig(name="error_test", iterations=5, warmup_iterations=0)

        result = await suite.run_benchmark(config, failing_workload)

        assert result.errors > 0
        assert result.error_rate > 0

    @pytest.mark.asyncio
    async def test_suite_report_generation(self):
        """Test benchmark suite report generation."""
        async def workload():
            await asyncio.sleep(0.001)

        suite = BenchmarkSuite(suite_name="full_test")
        suite.create_benchmark("bench_1", iterations=3, warmup=0)
        suite.create_benchmark("bench_2", iterations=3, warmup=0)

        report = await suite.run_benchmarks(workload)

        assert isinstance(report, BenchmarkSuiteReport)
        assert report.suite_name == "full_test"
        assert len(report.benchmarks) == 2

    def test_benchmark_sync_wrapper(self):
        """Test synchronous wrapper for benchmarks."""

        async def workload():
            await asyncio.sleep(0.001)

        suite = BenchmarkSuite()
        config = BenchmarkConfig(name="sync_test", iterations=3, warmup_iterations=0)

        result = suite.run_benchmark_sync(config, workload)

        assert result.config_name == "sync_test"


class TestEvaluatorCustomCriteria:
    """Tests for custom evaluation criteria."""

    @pytest.mark.asyncio
    async def test_custom_metric_function(self):
        """Test adding a custom metric function."""
        agent = MockAgent()
        evaluator = Evaluator(agent=agent, default_threshold=0.9)

        def custom_accuracy(actual: Any, expected: Any, metadata: Dict[str, Any]) -> float:
            """Custom accuracy based on metadata threshold."""
            threshold = metadata.get("threshold", 0.5)
            if isinstance(actual, dict) and isinstance(expected, dict):
                actual_val = actual.get("result", 0)
                expected_val = expected.get("result", 0)
                if expected_val == 0:
                    return 1.0 if actual_val == 0 else 0.0
                ratio = min(actual_val / expected_val, expected_val / actual_val)
                return 1.0 if ratio >= threshold else ratio
            return 0.0

        # Create a proper criteria class that uses our function
        class CustomCriteria:
            def __init__(self, fn):
                self.fn = fn

            def evaluate(self, actual, expected, metadata):
                return self.fn(actual, expected, metadata)

        evaluator.evaluation_criteria = CustomCriteria(custom_accuracy)

        test_case = TestCase(
            id="custom_test",
            name="Custom metric test",
            input_data={"value": 5},  # 5 * 2 = 10 expected
            expected_output={"result": 10},
            metadata={"threshold": 0.9}
        )

        result = await evaluator.run_test_case(test_case)

        assert result.passed is True


# Integration test with real async workload
class TestIntegration:
    """Integration tests for the evaluation framework."""

    @pytest.mark.asyncio
    async def test_end_to_end_evaluation_flow(self):
        """Test complete evaluation flow from test cases to report."""
        agent = MockAgent()
        evaluator = Evaluator(agent=agent, default_threshold=0.8, evaluation_criteria=PartialMatchCriteria())

        test_cases = [
            TestCase(
                id=f"case_{i}",
                name=f"Case {i}",
                input_data={"value": i},
                expected_output={"result": i * 2},
                tags=["basic"]
            )
            for i in range(3)
        ]

        report = await evaluator.evaluate(test_cases)

        # Verify report structure
        assert hasattr(report, "total_tests")
        assert hasattr(report, "passed_tests")
        assert hasattr(report, "pass_rate")
        assert hasattr(report, "avg_latency_ms")
        assert hasattr(report, "results")

        # Verify all passed
        assert report.passed_tests == report.total_tests

    @pytest.mark.asyncio
    async def test_benchmark_with_agent(self):
        """Test benchmarking with a mock agent."""
        agent = MockAgent(latency=0.01)

        async def agent_workload():
            await agent.invoke({"value": 1})

        suite = BenchmarkSuite(agent=agent)
        config = BenchmarkConfig(name="agent_bench", iterations=5, warmup_iterations=1)

        result = await suite.run_benchmark(config, agent_workload)

        assert result.avg_latency_ms > 0
        assert result.throughput > 0

"""
Tests for Real-Time Evaluation Framework.

Tests streaming evaluation, progress tracking, event handling,
and agent comparison.
"""

import pytest
import asyncio
from typing import Any, Dict

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
from src.agents.evaluation.evaluator import (
    Evaluator,
    TestCase,
    PartialMatchCriteria,
)


# Mock agent for testing
class MockAgent:
    """Mock agent for testing."""

    def __init__(self, should_fail: bool = False, latency: float = 0.001):
        self.should_fail = should_fail
        self.latency = latency
        self.call_count = 0

    async def invoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        await asyncio.sleep(self.latency)
        self.call_count += 1

        if self.should_fail:
            raise ValueError("Simulated agent failure")

        return {
            "result": input_data.get("value", 0) * 2,
            "status": "success"
        }


class TestEvaluationEvent:
    """Tests for EvaluationEvent."""

    def test_event_creation(self):
        from datetime import datetime
        event = EvaluationEvent(
            event_type=EventType.STARTED,
            data={"total": 10}
        )
        assert event.event_type == EventType.STARTED
        assert event.data["total"] == 10
        assert event.timestamp is not None


class TestEvaluationProgress:
    """Tests for EvaluationProgress."""

    def test_percent_complete(self):
        progress = EvaluationProgress(total=10, completed=5, failed=1, passed=4)
        assert progress.percent_complete == 50.0

    def test_zero_total(self):
        progress = EvaluationProgress(total=0, completed=0, failed=0, passed=0)
        assert progress.percent_complete == 0.0

    def test_to_dict(self):
        progress = EvaluationProgress(
            total=10,
            completed=5,
            failed=1,
            passed=4,
            status=EvaluationStatus.RUNNING,
            elapsed_seconds=10.5
        )
        result = progress.to_dict()
        assert result["total"] == 10
        assert result["completed"] == 5
        assert result["status"] == "running"
        assert result["percent_complete"] == 50.0


class TestEventEmitter:
    """Tests for EventEmitter."""

    def test_on_and_emit(self):
        emitter = EventEmitter()
        events_received = []

        def callback(event: EvaluationEvent):
            events_received.append(event)

        emitter.on(EventType.STARTED, callback)
        emitter.emit(EvaluationEvent(event_type=EventType.STARTED, data={}))

        assert len(events_received) == 1
        assert events_received[0].event_type == EventType.STARTED

    def test_off(self):
        emitter = EventEmitter()
        events_received = []

        def callback(event: EvaluationEvent):
            events_received.append(event)

        emitter.on(EventType.STARTED, callback)
        emitter.off(EventType.STARTED, callback)
        emitter.emit(EvaluationEvent(event_type=EventType.STARTED, data={}))

        assert len(events_received) == 0


class TestRealTimeEvaluator:
    """Tests for RealTimeEvaluator."""

    @pytest.mark.asyncio
    async def test_evaluate_streaming_basic(self):
        """Test basic streaming evaluation."""
        agent = MockAgent()
        evaluator = Evaluator(agent=agent, evaluation_criteria=PartialMatchCriteria())

        test_cases = [
            TestCase(
                id=f"test_{i}",
                name=f"Test {i}",
                input_data={"value": i},
                expected_output={"result": i * 2}
            )
            for i in range(3)
        ]

        config = StreamingEvaluationConfig(test_cases=test_cases)
        rt_evaluator = RealTimeEvaluator(evaluator, config)

        events = []
        async for event in rt_evaluator.evaluate_streaming():
            events.append(event)

        # Should have: started, progress (x3), test_completed (x3), metrics (x3), completed
        event_types = [e.event_type for e in events]
        assert EventType.STARTED in event_types
        assert EventType.COMPLETED in event_types
        assert EventType.TEST_COMPLETED in event_types
        assert EventType.PROGRESS in event_types

    @pytest.mark.asyncio
    async def test_evaluate_streaming_with_failure(self):
        """Test streaming with agent failures."""
        failing_agent = MockAgent(should_fail=True)
        evaluator = Evaluator(agent=failing_agent, evaluation_criteria=PartialMatchCriteria())

        test_cases = [
            TestCase(
                id="test_fail",
                name="Failure test",
                input_data={"value": 1},
                expected_output={"result": 2}
            )
        ]

        config = StreamingEvaluationConfig(test_cases=test_cases)
        rt_evaluator = RealTimeEvaluator(evaluator, config)

        events = []
        async for event in rt_evaluator.evaluate_streaming():
            events.append(event)

        # Check that we got failure event
        failure_events = [e for e in events if e.event_type == EventType.TEST_FAILED]
        assert len(failure_events) > 0

    @pytest.mark.asyncio
    async def test_progress_tracking(self):
        """Test progress tracking."""
        agent = MockAgent()
        evaluator = Evaluator(agent=agent, evaluation_criteria=PartialMatchCriteria())

        test_cases = [
            TestCase(
                id=f"test_{i}",
                name=f"Test {i}",
                input_data={"value": i},
                expected_output={"result": i * 2}
            )
            for i in range(5)
        ]

        config = StreamingEvaluationConfig(test_cases=test_cases)
        rt_evaluator = RealTimeEvaluator(evaluator, config)

        async for _ in rt_evaluator.evaluate_streaming():
            pass

        progress = rt_evaluator.get_progress()
        assert progress.total == 5
        assert progress.completed == 5
        assert progress.status == EvaluationStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_cancel_evaluation(self):
        """Test cancelling evaluation."""
        agent = MockAgent(latency=0.1)
        evaluator = Evaluator(agent=agent)

        test_cases = [
            TestCase(id=f"test_{i}", name=f"Test {i}", input_data={"value": i}, expected_output={"result": i * 2})
            for i in range(100)  # Many test cases
        ]

        config = StreamingEvaluationConfig(test_cases=test_cases)
        rt_evaluator = RealTimeEvaluator(evaluator, config)

        # Start evaluation and cancel after a few
        count = 0
        async for event in rt_evaluator.evaluate_streaming():
            count += 1
            if count == 5:
                rt_evaluator.cancel()
                break

        progress = rt_evaluator.get_progress()
        assert progress.status == EvaluationStatus.CANCELLED


class TestAgentComparator:
    """Tests for AgentComparator."""

    def test_add_result(self):
        from src.agents.evaluation.evaluator import EvaluationReport

        comparator = AgentComparator()

        report1 = EvaluationReport(
            total_tests=10,
            passed_tests=8,
            failed_tests=2,
            pass_rate=0.8,
            avg_latency_ms=50.0,
            total_cost=1.0,
            results=[]
        )

        comparator.add_result("agent_1", report1)

        assert "agent_1" in comparator._results

    def test_compare(self):
        from src.agents.evaluation.evaluator import EvaluationReport

        comparator = AgentComparator()

        comparator.add_result("agent_1", EvaluationReport(
            total_tests=10, passed_tests=8, failed_tests=2,
            pass_rate=0.8, avg_latency_ms=50.0, total_cost=1.0, results=[]
        ))
        comparator.add_result("agent_2", EvaluationReport(
            total_tests=10, passed_tests=9, failed_tests=1,
            pass_rate=0.9, avg_latency_ms=60.0, total_cost=1.5, results=[]
        ))

        result = comparator.compare()
        assert "agents" in result
        assert len(result["agents"]) == 2
        assert result["best_pass_rate"]["agent"] == "agent_2"

    def test_rank_agents(self):
        from src.agents.evaluation.evaluator import EvaluationReport

        comparator = AgentComparator()

        comparator.add_result("slow_agent", EvaluationReport(
            total_tests=10, passed_tests=10, failed_tests=0,
            pass_rate=1.0, avg_latency_ms=100.0, total_cost=2.0, results=[]
        ))
        comparator.add_result("fast_agent", EvaluationReport(
            total_tests=10, passed_tests=8, failed_tests=2,
            pass_rate=0.8, avg_latency_ms=20.0, total_cost=1.0, results=[]
        ))

        ranked = comparator.rank_agents("avg_latency_ms")
        assert ranked[0][0] == "fast_agent"  # Fastest first


class TestLiveMetricsPublisher:
    """Tests for LiveMetricsPublisher."""

    @pytest.mark.asyncio
    async def test_publish_metrics(self):
        """Test publishing live metrics."""
        published_data = []

        def publish_fn(data):
            published_data.append(data)

        publisher = LiveMetricsPublisher(publish_fn)
        agent = MockAgent()
        evaluator = Evaluator(agent=agent)

        test_cases = [TestCase(id="test_1", name="Test 1", input_data={"value": 1}, expected_output={"result": 2})]
        config = StreamingEvaluationConfig(test_cases=test_cases)
        rt_evaluator = RealTimeEvaluator(evaluator, config)

        await publisher.start(rt_evaluator)
        await asyncio.sleep(0.1)  # Let some metrics publish

        await publisher.stop()

        assert len(published_data) > 0
        assert "progress" in published_data[0]
        assert "metrics" in published_data[0]


class TestStreamingEvaluationConfig:
    """Tests for StreamingEvaluationConfig."""

    def test_default_config(self):
        config = StreamingEvaluationConfig(test_cases=[])
        assert config.emit_interval_seconds == 1.0
        assert config.include_metrics is True
        assert config.include_results is True
        assert config.max_results_in_memory == 100

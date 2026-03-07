"""
Real-Time Evaluation Framework

Provides real-time streaming evaluation with progress tracking,
WebSocket support, and live metrics updates.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, AsyncIterator
from enum import Enum
from collections import defaultdict
import json

from src.agents.evaluation.evaluator import (
    Evaluator,
    TestCase,
    EvaluationResult,
    EvaluationReport,
)
from src.agents.evaluation.metrics import (
    EvaluationMetrics,
    MetricsCollector,
    ResourceMetrics,
    TokenMetrics,
)

logger = logging.getLogger(__name__)


class EvaluationStatus(Enum):
    """Status of evaluation run."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EventType(Enum):
    """Types of evaluation events."""
    STARTED = "started"
    PROGRESS = "progress"
    TEST_COMPLETED = "test_completed"
    TEST_FAILED = "test_failed"
    METRICS_UPDATE = "metrics_update"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class EvaluationEvent:
    """Event emitted during evaluation."""
    event_type: EventType
    timestamp: datetime = field(default_factory=datetime.utcnow)
    data: Dict[str, Any] = field(default_factory=dict)
    test_case_id: Optional[str] = None
    progress: Optional[float] = None


@dataclass
class EvaluationProgress:
    """Progress tracking for evaluation."""
    total: int
    completed: int
    failed: int
    passed: int
    current_test_id: Optional[str] = None
    status: EvaluationStatus = EvaluationStatus.PENDING
    elapsed_seconds: float = 0.0
    estimated_remaining_seconds: float = 0.0

    @property
    def percent_complete(self) -> float:
        """Get percentage complete."""
        return (self.completed / self.total * 100) if self.total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total": self.total,
            "completed": self.completed,
            "failed": self.failed,
            "passed": self.passed,
            "current_test_id": self.current_test_id,
            "status": self.status.value,
            "elapsed_seconds": self.elapsed_seconds,
            "estimated_remaining_seconds": self.estimated_remaining_seconds,
            "percent_complete": self.percent_complete,
        }


@dataclass
class StreamingEvaluationConfig:
    """Configuration for streaming evaluation."""
    test_cases: List[TestCase]
    emit_interval_seconds: float = 1.0
    include_metrics: bool = True
    include_results: bool = True
    max_results_in_memory: int = 100


class EventEmitter:
    """Event emitter for evaluation events."""

    def __init__(self):
        self._listeners: Dict[EventType, List[Callable]] = defaultdict(list)
        self._subscribers: Set[asyncio.Queue] = set()

    def on(self, event_type: EventType, callback: Callable) -> None:
        """Register event listener."""
        self._listeners[event_type].append(callback)

    def off(self, event_type: EventType, callback: Callable) -> None:
        """Remove event listener."""
        if callback in self._listeners[event_type]:
            self._listeners[event_type].remove(callback)

    def emit(self, event: EvaluationEvent) -> None:
        """Emit event to all listeners."""
        for callback in self._listeners[event.event_type]:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in event callback: {e}")

        # Also emit to subscribers
        for queue in self._subscribers:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning("Event queue full, dropping event")

    def subscribe(self, queue: asyncio.Queue) -> None:
        """Subscribe to all events."""
        self._subscribers.add(queue)

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        """Unsubscribe from events."""
        self._subscribers.discard(queue)


class RealTimeEvaluator:
    """
    Real-time evaluation with streaming and progress tracking.

    Provides live updates during evaluation including progress,
    metrics, and individual test results.
    """

    def __init__(
        self,
        evaluator: Evaluator,
        config: Optional[StreamingEvaluationConfig] = None
    ):
        """
        Initialize real-time evaluator.

        Args:
            evaluator: Base evaluator to use
            config: Streaming configuration
        """
        self.evaluator = evaluator
        self.config = config or StreamingEvaluationConfig(test_cases=[])
        self.event_emitter = EventEmitter()
        self._metrics_collector = MetricsCollector()
        self._is_cancelled = False
        self._is_paused = False
        self._progress = EvaluationProgress(
            total=len(self.config.test_cases),
            completed=0,
            failed=0,
            passed=0,
            status=EvaluationStatus.PENDING
        )
        self._start_time: Optional[float] = None

    async def evaluate_streaming(
        self,
        test_cases: Optional[List[TestCase]] = None
    ) -> AsyncIterator[EvaluationEvent]:
        """
        Evaluate test cases with streaming events.

        Yields:
            EvaluationEvent for each significant occurrence
        """
        test_cases = test_cases or self.config.test_cases

        # Initialize progress
        self._progress = EvaluationProgress(
            total=len(test_cases),
            completed=0,
            failed=0,
            passed=0,
            status=EvaluationStatus.RUNNING
        )
        self._start_time = time.perf_counter()
        self._is_cancelled = False
        self._is_paused = False

        # Emit start event
        start_event = EvaluationEvent(
            event_type=EventType.STARTED,
            data={"total": len(test_cases)},
            progress=0.0
        )
        self.event_emitter.emit(start_event)
        yield start_event

        self._metrics_collector.start()

        for i, test_case in enumerate(test_cases):
            if self._is_cancelled:
                cancel_event = EvaluationEvent(
                    event_type=EventType.CANCELLED,
                    data={"completed": self._progress.completed},
                    progress=self._progress.percent_complete
                )
                self.event_emitter.emit(cancel_event)
                yield cancel_event
                break

            # Handle pause
            while self._is_paused and not self._is_cancelled:
                await asyncio.sleep(0.1)

            self._progress.current_test_id = test_case.id

            # Emit progress update
            progress_event = EvaluationEvent(
                event_type=EventType.PROGRESS,
                data=self._progress.to_dict(),
                test_case_id=test_case.id,
                progress=self._progress.percent_complete
            )
            self.event_emitter.emit(progress_event)
            yield progress_event

            # Run test case
            try:
                result = await self.evaluator.run_test_case(test_case)


                self._progress.completed += 1
                if result.passed:
                    self._progress.passed += 1
                else:
                    self._progress.failed += 1

                # Calculate elapsed and estimate
                elapsed = time.perf_counter() - self._start_time
                self._progress.elapsed_seconds = elapsed
                if self._progress.completed > 0:
                    avg_time = elapsed / self._progress.completed
                    remaining = avg_time * (self._progress.total - self._progress.completed)
                    self._progress.estimated_remaining_seconds = remaining

                # Emit result event
                event_type = EventType.TEST_COMPLETED if result.passed else EventType.TEST_FAILED
                result_event = EvaluationEvent(
                    event_type=event_type,
                    data={
                        "test_case_id": test_case.id,
                        "passed": result.passed,
                        "latency_ms": result.latency_ms,
                        "accuracy": result.metrics.get("accuracy", 0.0),
                    },
                    test_case_id=test_case.id,
                    progress=self._progress.percent_complete
                )
                self.event_emitter.emit(result_event)
                yield result_event

                # Emit metrics update if configured
                if self.config.include_metrics:
                    metrics = self._build_metrics(result)
                    self._metrics_collector.record(metrics)

                    metrics_event = EvaluationEvent(
                        event_type=EventType.METRICS_UPDATE,
                        data={
                            "instant": metrics.to_dict(),
                            "aggregated": self._metrics_collector.aggregate().to_dict()
                        },
                        progress=self._progress.percent_complete
                    )
                    self.event_emitter.emit(metrics_event)
                    yield metrics_event

            except Exception as e:
                logger.error(f"Error running test {test_case.id}: {e}")
                self._progress.completed += 1
                self._progress.failed += 1

                error_event = EvaluationEvent(
                    event_type=EventType.TEST_FAILED,
                    data={
                        "test_case_id": test_case.id,
                        "error": str(e)
                    },
                    test_case_id=test_case.id,
                    progress=self._progress.percent_complete
                )
                self.event_emitter.emit(error_event)
                yield error_event

        # Mark as completed
        self._metrics_collector.stop()
        self._progress.status = EvaluationStatus.COMPLETED

        # Emit completion event
        final_metrics = self._metrics_collector.aggregate()
        complete_event = EvaluationEvent(
            event_type=EventType.COMPLETED,
            data={
                "progress": self._progress.to_dict(),
                "metrics": final_metrics.to_dict()
            },
            progress=100.0
        )
        self.event_emitter.emit(complete_event)
        yield complete_event

    def _build_metrics(self, result: EvaluationResult) -> EvaluationMetrics:
        """Build comprehensive metrics from result."""
        return EvaluationMetrics(
            accuracy=result.metrics.get("accuracy", 0.0),
            latency_ms=result.latency_ms,
            total_cost=result.metrics.get("cost", 0.0),
            error_rate=0.0 if result.passed else 1.0,
            timeout_rate=1.0 if result.error and "timeout" in result.error.lower() else 0.0,
            metadata=result.metadata
        )

    def pause(self) -> None:
        """Pause evaluation."""
        self._is_paused = True
        self._progress.status = EvaluationStatus.PAUSED

    def resume(self) -> None:
        """Resume evaluation."""
        self._is_paused = False
        self._progress.status = EvaluationStatus.RUNNING

    def cancel(self) -> None:
        """Cancel evaluation."""
        self._is_cancelled = True
        self._progress.status = EvaluationStatus.CANCELLED

    def get_progress(self) -> EvaluationProgress:
        """Get current progress."""
        return self._progress

    def get_metrics(self) -> EvaluationMetrics:
        """Get aggregated metrics."""
        return self._metrics_collector.aggregate()


class AgentComparator:
    """
    Compare multiple agents on the same test cases.

    Provides side-by-side comparison of agent performance
    with statistical analysis.
    """

    def __init__(self):
        self._results: Dict[str, EvaluationReport] = {}
        self._benchmarks: Dict[str, Dict[str, float]] = {}

    def add_result(self, agent_name: str, report: EvaluationReport) -> None:
        """Add evaluation result for an agent."""
        self._results[agent_name] = report

        # Calculate benchmark metrics
        self._benchmarks[agent_name] = {
            "pass_rate": report.pass_rate,
            "avg_latency_ms": report.avg_latency_ms,
            "total_cost": report.total_cost,
            "total_tests": report.total_tests,
            "passed_tests": report.passed_tests,
        }

    def compare(self) -> Dict[str, Any]:
        """Generate comparison report."""
        if not self._results:
            return {}

        # Find best performers
        agents = list(self._results.keys())

        best_pr = max(self._benchmarks.items(), key=lambda x: x[1]["pass_rate"])
        fastest = min(self._benchmarks.items(), key=lambda x: x[1]["avg_latency_ms"])

        return {
            "agents": agents,
            "metrics": self._benchmarks,
            "best_pass_rate": {"agent": best_pr[0], "pass_rate": best_pr[1]["pass_rate"]},
            "fastest": {"agent": fastest[0], "latency_ms": fastest[1]["avg_latency_ms"]},
            "most_efficient": min(
                self._benchmarks.items(),
                key=lambda x: x[1]["total_cost"] / x[1]["total_tests"] if x[1]["total_tests"] > 0 else float("inf")
            )[0] if self._benchmarks else None,
        }

    def rank_agents(self, metric: str = "pass_rate") -> List[tuple]:
        """Rank agents by a specific metric."""
        # For latency, lower is better
        reverse = metric != "avg_latency_ms"
        ranked = sorted(
            self._benchmarks.items(),
            key=lambda x: x[1].get(metric, 0),
            reverse=reverse
        )
        return [(name, metrics[metric]) for name, metrics in ranked]

    def get_winner(self, metric: str = "pass_rate") -> Optional[str]:
        """Get the winning agent for a metric."""
        ranked = self.rank_agents(metric)
        return ranked[0][0] if ranked else None


class LiveMetricsPublisher:
    """Publishes live metrics for real-time monitoring."""

    def __init__(self, publish_fn: Callable[[Dict[str, Any]], None]):
        """
        Initialize publisher.

        Args:
            publish_fn: Function to publish metrics (e.g., WebSocket send)
        """
        self._publish_fn = publish_fn
        self._publish_interval = 1.0
        self._is_running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self, evaluator: RealTimeEvaluator) -> None:
        """Start publishing live metrics."""
        self._is_running = True

        async def publish_loop():
            while self._is_running:
                if evaluator:
                    progress = evaluator.get_progress()
                    metrics = evaluator.get_metrics()

                    self._publish_fn({
                        "progress": progress.to_dict(),
                        "metrics": metrics.to_dict(),
                        "timestamp": datetime.utcnow().isoformat()
                    })

                await asyncio.sleep(self._publish_interval)

        self._task = asyncio.create_task(publish_loop())

    async def stop(self) -> None:
        """Stop publishing."""
        self._is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

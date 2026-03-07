"""
Agent Benchmark Suite

Provides benchmark functionality for performance testing of agents,
including latency, throughput, and resource utilization metrics.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
from statistics import mean, stdev, median

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""
    name: str
    description: str = ""
    iterations: int = 10
    warmup_iterations: int = 2
    timeout_seconds: float = 60.0
    parallel_workers: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    config_name: str
    iterations: int
    total_time_ms: float
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    median_latency_ms: float
    std_dev_ms: float
    throughput: float
    errors: int = 0
    error_rate: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkSuiteReport:
    """Aggregated report for multiple benchmarks."""
    suite_name: str
    benchmarks: List[BenchmarkResult]
    total_iterations: int
    total_time_ms: float
    avg_throughput: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    summary: Dict[str, Any] = field(default_factory=dict)


class BenchmarkSuite:
    """
    Benchmark suite for performance testing of agents.

    Supports various workload patterns including sequential, parallel,
    and load testing scenarios.
    """

    def __init__(self, agent: Any = None, suite_name: str = "default"):
        """
        Initialize the benchmark suite.

        Args:
            agent: The agent to benchmark
            suite_name: Name identifier for this benchmark suite
        """
        self.agent = agent
        self.suite_name = suite_name
        self._benchmarks: Dict[str, BenchmarkConfig] = {}

    def register_benchmark(self, config: BenchmarkConfig) -> None:
        """Register a benchmark configuration."""
        self._benchmarks[config.name] = config

    def create_benchmark(
        self,
        name: str,
        description: str = "",
        iterations: int = 10,
        warmup: int = 2
    ) -> BenchmarkConfig:
        """Create and register a new benchmark configuration."""
        config = BenchmarkConfig(
            name=name,
            description=description,
            iterations=iterations,
            warmup_iterations=warmup
        )
        self.register_benchmark(config)
        return config

    async def run_benchmark(
        self,
        config: BenchmarkConfig,
        workload_fn: Callable[[], Any]
    ) -> BenchmarkResult:
        """
        Run a single benchmark with the given configuration.

        Args:
            config: Benchmark configuration
            workload_fn: Function that executes the workload

        Returns:
            BenchmarkResult with timing metrics
        """
        # Warmup phase
        for _ in range(config.warmup_iterations):
            try:
                if asyncio.iscoroutinefunction(workload_fn):
                    await workload_fn()
                else:
                    workload_fn()
            except Exception as e:
                logger.warning(f"Warmup error: {e}")

        # Actual benchmark runs
        latencies: List[float] = []
        errors = 0

        async def run_single():
            start = time.perf_counter()
            try:
                if asyncio.iscoroutinefunction(workload_fn):
                    await workload_fn()
                else:
                    workload_fn()
                elapsed = (time.perf_counter() - start) * 1000
                latencies.append(elapsed)
            except Exception as e:
                nonlocal errors
                errors += 1
                logger.warning(f"Benchmark error: {e}")

        if config.parallel_workers > 1:
            # Parallel execution
            for _ in range(0, config.iterations, config.parallel_workers):
                batch_size = min(config.parallel_workers, config.iterations - len(latencies) - errors)
                await asyncio.gather(*[run_single() for _ in range(batch_size)])
        else:
            # Sequential execution
            for _ in range(config.iterations):
                await run_single()

        if not latencies:
            return BenchmarkResult(
                config_name=config.name,
                iterations=config.iterations,
                total_time_ms=0.0,
                avg_latency_ms=0.0,
                min_latency_ms=0.0,
                max_latency_ms=0.0,
                median_latency_ms=0.0,
                std_dev_ms=0.0,
                throughput=0.0,
                errors=config.iterations,
                error_rate=1.0,
                metadata=config.metadata
            )

        total_time = sum(latencies)
        sorted_latencies = sorted(latencies)
        n = len(latencies)

        return BenchmarkResult(
            config_name=config.name,
            iterations=config.iterations,
            total_time_ms=total_time,
            avg_latency_ms=mean(latencies),
            min_latency_ms=min(latencies),
            max_latency_ms=max(latencies),
            median_latency_ms=median(latencies),
            std_dev_ms=stdev(latencies) if n > 1 else 0.0,
            throughput=1000.0 / mean(latencies) if latencies else 0.0,
            errors=errors,
            error_rate=errors / config.iterations,
            metadata=config.metadata
        )

    async def run_benchmarks(
        self,
        workload_fn: Callable[[], Any],
        benchmark_names: Optional[List[str]] = None
    ) -> BenchmarkSuiteReport:
        """
        Run multiple benchmarks.

        Args:
            workload_fn: Function that executes the workload
            benchmark_names: Specific benchmarks to run (None = all)

        Returns:
            BenchmarkSuiteReport with all results
        """
        to_run = benchmark_names or list(self._benchmarks.keys())
        results: List[BenchmarkResult] = []

        for name in to_run:
            if name not in self._benchmarks:
                logger.warning(f"Benchmark {name} not found, skipping")
                continue

            config = self._benchmarks[name]
            result = await self.run_benchmark(config, workload_fn)
            results.append(result)

        total_iterations = sum(r.iterations for r in results)
        total_time = sum(r.total_time_ms for r in results)

        return BenchmarkSuiteReport(
            suite_name=self.suite_name,
            benchmarks=results,
            total_iterations=total_iterations,
            total_time_ms=total_time,
            avg_throughput=total_iterations / (total_time / 1000) if total_time > 0 else 0.0,
            summary=self._generate_summary(results)
        )

    def _generate_summary(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Generate summary statistics from benchmark results."""
        if not results:
            return {}

        return {
            "total_benchmarks": len(results),
            "total_errors": sum(r.errors for r in results),
            "overall_error_rate": sum(r.errors for r in results) / sum(r.iterations for r in results),
            "fastest": min(results, key=lambda r: r.avg_latency_ms).config_name,
            "slowest": max(results, key=lambda r: r.avg_latency_ms).config_name,
            "highest_throughput": max(results, key=lambda r: r.throughput).config_name,
        }

    def run_benchmarks_sync(
        self,
        workload_fn: Callable[[], Any],
        benchmark_names: Optional[List[str]] = None
    ) -> BenchmarkSuiteReport:
        """Synchronous wrapper for run_benchmarks."""
        return asyncio.run(self.run_benchmarks(workload_fn, benchmark_names))

    # Convenience methods for common benchmark patterns

    async def benchmark_latency(
        self,
        workload_fn: Callable[[], Any],
        iterations: int = 100
    ) -> BenchmarkResult:
        """Run a simple latency benchmark."""
        config = BenchmarkConfig(
            name="latency_benchmark",
            description="Measure single operation latency",
            iterations=iterations,
            warmup_iterations=10
        )
        return await self.run_benchmark(config, workload_fn)

    async def benchmark_throughput(
        self,
        workload_fn: Callable[[], Any],
        duration_seconds: float = 10.0,
        workers: int = 4
    ) -> BenchmarkResult:
        """Run a throughput benchmark for a fixed duration."""
        start_time = time.perf_counter()
        iterations = 0

        async def timed_workload():
            nonlocal iterations
            await workload_fn()
            iterations += 1

        config = BenchmarkConfig(
            name="throughput_benchmark",
            description=f"Measure throughput over {duration_seconds}s",
            iterations=10000,  # Will be limited by duration
            warmup_iterations=5,
            timeout_seconds=duration_seconds,
            parallel_workers=workers
        )

        # Run for duration
        async def duration_limited():
            nonlocal iterations
            while (time.perf_counter() - start_time) < duration_seconds:
                await workload_fn()
                iterations += 1

        return await self.run_benchmark(config, duration_limited)

    def benchmark_latency_sync(
        self,
        workload_fn: Callable[[], Any],
        iterations: int = 100
    ) -> BenchmarkResult:
        """Synchronous wrapper for benchmark_latency."""
        return asyncio.run(self.benchmark_latency(workload_fn, iterations))

    def run_benchmark_sync(
        self,
        config: BenchmarkConfig,
        workload_fn: Callable[[], Any]
    ) -> BenchmarkResult:
        """Synchronous wrapper for run_benchmark."""
        return asyncio.run(self.run_benchmark(config, workload_fn))

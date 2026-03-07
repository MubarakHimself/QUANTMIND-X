"""
Evaluation API Endpoints for QuantMindX

Provides HTTP endpoints for running agent evaluations and benchmarks.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

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

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])


# ============== Request Models ==============

class TestCaseInput(BaseModel):
    """Input model for a test case."""
    id: str = Field(..., description="Unique identifier for the test case")
    name: str = Field(..., description="Human-readable name")
    input_data: Dict[str, Any] = Field(..., description="Input data for the agent")
    expected_output: Any = Field(..., description="Expected output for comparison")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    timeout_seconds: float = Field(default=30.0, description="Timeout in seconds")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")


class EvaluationRequest(BaseModel):
    """Request model for running evaluations."""
    test_cases: List[TestCaseInput] = Field(..., description="List of test cases to run")
    criteria_type: str = Field(default="exact", description="Criteria type: exact, partial, threshold")
    threshold: float = Field(default=0.8, description="Threshold for passing tests")
    parallel: bool = Field(default=True, description="Run tests in parallel")
    stop_on_first_failure: bool = Field(default=False, description="Stop after first failure")
    cost_per_token: float = Field(default=0.0, description="Cost per token for calculations")


class BenchmarkConfigInput(BaseModel):
    """Input model for benchmark configuration."""
    name: str = Field(..., description="Benchmark name")
    description: str = Field(default="", description="Benchmark description")
    iterations: int = Field(default=10, description="Number of iterations")
    warmup_iterations: int = Field(default=2, description="Warmup iterations")
    timeout_seconds: float = Field(default=60.0, description="Timeout in seconds")
    parallel_workers: int = Field(default=1, description="Number of parallel workers")


class BenchmarkRequest(BaseModel):
    """Request model for running benchmarks."""
    config: BenchmarkConfigInput = Field(..., description="Benchmark configuration")
    workload_type: str = Field(default="latency", description="Type of workload: latency, throughput")
    custom_workload: Optional[Dict[str, Any]] = Field(default=None, description="Custom workload definition")


# ============== Response Models ==============

class EvaluationResultResponse(BaseModel):
    """Response model for a single test result."""
    test_case_id: str
    passed: bool
    actual_output: Any
    expected_output: Any
    metrics: Dict[str, float]
    latency_ms: float
    error: Optional[str] = None
    timestamp: datetime
    metadata: Dict[str, Any]


class EvaluationReportResponse(BaseModel):
    """Response model for evaluation report."""
    total_tests: int
    passed_tests: int
    failed_tests: int
    pass_rate: float
    avg_latency_ms: float
    total_cost: float
    results: List[EvaluationResultResponse]
    custom_metrics: Dict[str, float]
    timestamp: datetime


class BenchmarkResultResponse(BaseModel):
    """Response model for a single benchmark result."""
    config_name: str
    iterations: int
    total_time_ms: float
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    median_latency_ms: float
    std_dev_ms: float
    throughput: float
    errors: int
    error_rate: float
    timestamp: datetime
    metadata: Dict[str, Any]


class BenchmarkSuiteReportResponse(BaseModel):
    """Response model for benchmark suite report."""
    suite_name: str
    benchmarks: List[BenchmarkResultResponse]
    total_iterations: int
    total_time_ms: float
    avg_throughput: float
    timestamp: datetime
    summary: Dict[str, Any]


# ============== Helper Functions ==============

def get_criteria(criteria_type: str, threshold: float = 0.8):
    """Get the appropriate criteria based on type."""
    if criteria_type == "exact":
        return ExactMatchCriteria()
    elif criteria_type == "partial":
        return PartialMatchCriteria()
    elif criteria_type == "threshold":
        return ThresholdCriteria(threshold=threshold)
    else:
        return ExactMatchCriteria()


def convert_test_case(tc: TestCaseInput) -> TestCase:
    """Convert input model to TestCase."""
    return TestCase(
        id=tc.id,
        name=tc.name,
        input_data=tc.input_data,
        expected_output=tc.expected_output,
        metadata=tc.metadata,
        timeout_seconds=tc.timeout_seconds,
        tags=tc.tags
    )


def convert_result(result: EvaluationResult) -> EvaluationResultResponse:
    """Convert EvaluationResult to response model."""
    return EvaluationResultResponse(
        test_case_id=result.test_case_id,
        passed=result.passed,
        actual_output=result.actual_output,
        expected_output=result.expected_output,
        metrics=result.metrics,
        latency_ms=result.latency_ms,
        error=result.error,
        timestamp=result.timestamp,
        metadata=result.metadata
    )


def convert_benchmark_result(result: BenchmarkResult) -> BenchmarkResultResponse:
    """Convert BenchmarkResult to response model."""
    return BenchmarkResultResponse(
        config_name=result.config_name,
        iterations=result.iterations,
        total_time_ms=result.total_time_ms,
        avg_latency_ms=result.avg_latency_ms,
        min_latency_ms=result.min_latency_ms,
        max_latency_ms=result.max_latency_ms,
        median_latency_ms=result.median_latency_ms,
        std_dev_ms=result.std_dev_ms,
        throughput=result.throughput,
        errors=result.errors,
        error_rate=result.error_rate,
        timestamp=result.timestamp,
        metadata=result.metadata
    )


# ============== Mock Agent for Testing ==============

class EvalMockAgent:
    """Mock agent for evaluation endpoints (for testing purposes)."""

    async def invoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate agent invocation."""
        await asyncio.sleep(0.01)  # Simulate some processing

        # Simple transformation for testing
        value = input_data.get("value", 0)
        return {
            "result": value * 2,
            "status": "success",
            "processed": True
        }


# ============== API Endpoints ==============

@router.post("/evaluate", response_model=EvaluationReportResponse)
async def run_evaluation(request: EvaluationRequest):
    """
    Run an evaluation with the provided test cases.

    Uses a mock agent for demonstration. In production, this would
    connect to actual agents via the agent registry.
    """
    try:
        # Create mock agent (in production, fetch from registry)
        agent = EvalMockAgent()

        # Get criteria
        criteria = get_criteria(request.criteria_type, request.threshold)

        # Create evaluator
        evaluator = Evaluator(
            agent=agent,
            evaluation_criteria=criteria,
            cost_per_token=request.cost_per_token,
            default_threshold=request.threshold
        )

        # Convert test cases
        test_cases = [convert_test_case(tc) for tc in request.test_cases]

        # Run evaluation
        report = await evaluator.evaluate(
            test_cases=test_cases,
            parallel=request.parallel,
            stop_on_first_failure=request.stop_on_first_failure
        )

        # Convert to response
        return EvaluationReportResponse(
            total_tests=report.total_tests,
            passed_tests=report.passed_tests,
            failed_tests=report.failed_tests,
            pass_rate=report.pass_rate,
            avg_latency_ms=report.avg_latency_ms,
            total_cost=report.total_cost,
            results=[convert_result(r) for r in report.results],
            custom_metrics=report.custom_metrics,
            timestamp=report.timestamp
        )

    except Exception as e:
        logger.exception("Error running evaluation")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/benchmark", response_model=BenchmarkResultResponse)
async def run_benchmark(request: BenchmarkRequest):
    """
    Run a single benchmark with the provided configuration.

    Executes a workload and measures performance metrics.
    """
    try:
        # Create mock agent (in production, fetch from registry)
        agent = EvalMockAgent()

        # Convert config
        config = BenchmarkConfig(
            name=request.config.name,
            description=request.config.description,
            iterations=request.config.iterations,
            warmup_iterations=request.config.warmup_iterations,
            timeout_seconds=request.config.timeout_seconds,
            parallel_workers=request.config.parallel_workers
        )

        # Define workload based on type
        if request.workload_type == "latency":
            async def workload():
                await agent.invoke({"value": 1})
        elif request.workload_type == "throughput":
            async def workload():
                await agent.invoke({"value": 1})
        else:
            raise HTTPException(status_code=400, detail=f"Unknown workload type: {request.workload_type}")

        # Create suite and run
        suite = BenchmarkSuite(agent=agent, suite_name="api_benchmark")
        result = await suite.run_benchmark(config, workload)

        return convert_benchmark_result(result)

    except Exception as e:
        logger.exception("Error running benchmark")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/benchmark/suite", response_model=BenchmarkSuiteReportResponse)
async def run_benchmark_suite(configs: List[BenchmarkConfigInput]):
    """
    Run multiple benchmarks as a suite.

    Each config in the list represents a separate benchmark to run.
    """
    try:
        # Create mock agent
        agent = EvalMockAgent()

        # Create suite
        suite = BenchmarkSuite(agent=agent, suite_name="api_suite")

        # Register all benchmarks
        for cfg in configs:
            suite.register_benchmark(BenchmarkConfig(
                name=cfg.name,
                description=cfg.description,
                iterations=cfg.iterations,
                warmup_iterations=cfg.warmup_iterations,
                timeout_seconds=cfg.timeout_seconds,
                parallel_workers=cfg.parallel_workers
            ))

        # Define workload
        async def workload():
            await agent.invoke({"value": 1})

        # Run all benchmarks
        report = await suite.run_benchmarks(workload)

        # Convert to response
        return BenchmarkSuiteReportResponse(
            suite_name=report.suite_name,
            benchmarks=[convert_benchmark_result(b) for b in report.benchmarks],
            total_iterations=report.total_iterations,
            total_time_ms=report.total_time_ms,
            avg_throughput=report.avg_throughput,
            timestamp=report.timestamp,
            summary=report.summary
        )

    except Exception as e:
        logger.exception("Error running benchmark suite")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/criteria")
async def get_criteria_types():
    """
    Get available evaluation criteria types.
    """
    return {
        "criteria": [
            {
                "type": "exact",
                "description": "Exact match - returns 1.0 if outputs match exactly, 0.0 otherwise"
            },
            {
                "type": "partial",
                "description": "Partial match - for dicts/lists, returns ratio of matching fields"
            },
            {
                "type": "threshold",
                "description": "Threshold-based - for numeric values, checks if ratio meets threshold"
            }
        ]
    }


@router.get("/health")
async def evaluation_health():
    """Health check for evaluation service."""
    return {"status": "healthy", "service": "evaluation"}

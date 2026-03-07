"""
Tests for Agent Metrics API Endpoints.

Validates Task 14 - Agent Metrics Dashboard
"""

import pytest
from datetime import datetime
from src.api.agent_metrics import (
    router,
    AgentTokenUsage,
    AgentTaskStats,
    AgentLatencyMetric,
    AgentCostBreakdown,
    AgentMetricsSummary,
    AgentMetricsResponse,
    _get_agent_metrics,
    _calculate_percentile,
    TOKEN_PRICES
)


class TestAgentMetricsModels:
    """Tests for agent metrics Pydantic models."""

    def test_agent_token_usage_creation(self):
        """Test creating an AgentTokenUsage object."""
        token = AgentTokenUsage(
            agent_id="agent_001",
            agent_name="Test Agent",
            input_tokens=1000000,
            output_tokens=500000,
            total_tokens=1500000,
            cost=1.25
        )
        assert token.agent_id == "agent_001"
        assert token.total_tokens == 1500000

    def test_agent_task_stats_creation(self):
        """Test creating an AgentTaskStats object."""
        stats = AgentTaskStats(
            agent_id="agent_001",
            agent_name="Test Agent",
            total_tasks=10,
            successful_tasks=9,
            failed_tasks=1,
            success_rate=90.0,
            avg_latency_ms=250.0
        )
        assert stats.total_tasks == 10
        assert stats.success_rate == 90.0

    def test_agent_latency_metric_creation(self):
        """Test creating an AgentLatencyMetric object."""
        latency = AgentLatencyMetric(
            agent_id="agent_001",
            agent_name="Test Agent",
            avg_latency_ms=200.0,
            min_latency_ms=100.0,
            max_latency_ms=500.0,
            p95_latency_ms=450.0,
            p99_latency_ms=490.0
        )
        assert latency.min_latency_ms == 100.0
        assert latency.p95_latency_ms == 450.0

    def test_agent_cost_breakdown_creation(self):
        """Test creating an AgentCostBreakdown object."""
        cost = AgentCostBreakdown(
            agent_id="agent_001",
            agent_name="Test Agent",
            input_cost=0.5,
            output_cost=0.75,
            total_cost=1.25,
            cost_percentage=50.0
        )
        assert cost.total_cost == 1.25
        assert cost.cost_percentage == 50.0

    def test_agent_metrics_summary_creation(self):
        """Test creating an AgentMetricsSummary object."""
        summary = AgentMetricsSummary(
            total_agents=5,
            total_tokens=10000000,
            total_cost=10.0,
            overall_success_rate=85.5,
            avg_latency_ms=300.0,
            period_start=datetime.now(),
            period_end=datetime.now()
        )
        assert summary.total_agents == 5
        assert summary.overall_success_rate == 85.5


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_calculate_percentile_odd_count(self):
        """Test percentile calculation with odd number of values."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert _calculate_percentile(values, 50) == 3.0

    def test_calculate_percentile_even_count(self):
        """Test percentile calculation with even number of values."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        result = _calculate_percentile(values, 50)
        # With 6 values, index 3 = 4.0 (0-indexed)
        assert 3.0 <= result <= 4.0

    def test_calculate_percentile_empty(self):
        """Test percentile calculation with empty list."""
        assert _calculate_percentile([], 50) == 0.0

    def test_calculate_percentile_single_value(self):
        """Test percentile calculation with single value."""
        assert _calculate_percentile([5.0], 95) == 5.0

    def test_token_prices_configured(self):
        """Test that token prices are properly configured."""
        assert TOKEN_PRICES["input"] > 0
        assert TOKEN_PRICES["output"] > 0
        assert TOKEN_PRICES["input"] < TOKEN_PRICES["output"]


class TestGetAgentMetrics:
    """Tests for _get_agent_metrics function."""

    def test_get_agent_metrics_returns_response(self):
        """Test that get_agent_metrics returns proper response."""
        response = _get_agent_metrics(period_hours=24)
        assert isinstance(response, AgentMetricsResponse)
        assert response.summary is not None
        assert len(response.token_usage) > 0
        assert len(response.task_stats) > 0
        assert len(response.cost_breakdown) > 0

    def test_metrics_summary_values(self):
        """Test that summary values are calculated correctly."""
        response = _get_agent_metrics(period_hours=24)
        summary = response.summary
        assert summary.total_agents > 0
        assert summary.total_tokens > 0
        assert summary.total_cost > 0
        assert 0 <= summary.overall_success_rate <= 100
        assert summary.avg_latency_ms >= 0

    def test_token_usage_totals_match(self):
        """Test that token usage totals match summary."""
        response = _get_agent_metrics(period_hours=24)
        total_input = sum(t.input_tokens for t in response.token_usage)
        total_output = sum(t.output_tokens for t in response.token_usage)
        assert response.summary.total_tokens == total_input + total_output

    def test_cost_breakdown_percentages_sum_to_100(self):
        """Test that cost percentages sum to approximately 100."""
        response = _get_agent_metrics(period_hours=24)
        total_percentage = sum(c.cost_percentage for c in response.cost_breakdown)
        assert 99.0 <= total_percentage <= 101.0

    def test_task_success_rates_valid(self):
        """Test that task success rates are valid percentages."""
        response = _get_agent_metrics(period_hours=24)
        for task in response.task_stats:
            assert 0 <= task.success_rate <= 100
            assert task.total_tasks == task.successful_tasks + task.failed_tasks

    def test_latency_metrics_valid(self):
        """Test that latency metrics are valid."""
        response = _get_agent_metrics(period_hours=24)
        for latency in response.latency_metrics:
            assert latency.min_latency_ms >= 0
            assert latency.max_latency_ms >= latency.min_latency_ms
            assert latency.avg_latency_ms >= latency.min_latency_ms
            assert latency.p95_latency_ms >= latency.avg_latency_ms
            assert latency.p99_latency_ms >= latency.p95_latency_ms


class TestRouterConfiguration:
    """Tests for router configuration."""

    def test_router_prefix(self):
        """Test that router has correct prefix."""
        assert router.prefix == "/api/agent-metrics"

    def test_router_tags(self):
        """Test that router has correct tags."""
        assert "agent-metrics" in router.tags

    def test_router_has_all_endpoints(self):
        """Test that router has all expected endpoints."""
        routes = [route.path for route in router.routes]
        assert "/api/agent-metrics" in routes  # Main endpoint
        assert "/api/agent-metrics/summary" in routes
        assert "/api/agent-metrics/tokens" in routes
        assert "/api/agent-metrics/tasks" in routes
        assert "/api/agent-metrics/latency" in routes
        assert "/api/agent-metrics/costs" in routes

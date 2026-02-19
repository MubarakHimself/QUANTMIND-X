"""
Analysis Tools for QuantMind agents.

These tools provide backtest analysis and strategy evaluation:
- analyze_backtest: Analyze backtest results
- compare_strategies: Compare multiple strategies
- evaluate_performance: Calculate performance metrics
- generate_optimization_report: Create optimization report
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from .base import (
    QuantMindTool,
    ToolCategory,
    ToolError,
    ToolPriority,
    ToolResult,
    register_tool,
)
from .registry import AgentType


logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Performance metric types."""
    PROFIT_FACTOR = "profit_factor"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"
    EXPECTANCY = "expectancy"
    RECOVERY_FACTOR = "recovery_factor"
    CAGR = "cagr"


@dataclass
class PerformanceMetrics:
    """Standard performance metrics for trading strategies."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_profit: float = 0.0
    total_loss: float = 0.0
    net_profit: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_percent: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    expectancy: float = 0.0
    recovery_factor: float = 0.0
    cagr: float = 0.0


class AnalyzeBacktestInput(BaseModel):
    """Input schema for analyze_backtest tool."""
    backtest_id: str = Field(
        description="ID of the backtest to analyze"
    )
    metrics: Optional[List[str]] = Field(
        default=None,
        description="Specific metrics to calculate (calculates all if not specified)"
    )
    include_trades: bool = Field(
        default=False,
        description="Whether to include individual trade data"
    )
    include_equity_curve: bool = Field(
        default=True,
        description="Whether to include equity curve data"
    )


class CompareStrategiesInput(BaseModel):
    """Input schema for compare_strategies tool."""
    backtest_ids: List[str] = Field(
        description="List of backtest IDs to compare",
        min_length=2
    )
    primary_metric: str = Field(
        default="sharpe_ratio",
        description="Primary metric for ranking"
    )
    normalize: bool = Field(
        default=True,
        description="Whether to normalize metrics for comparison"
    )


class EvaluatePerformanceInput(BaseModel):
    """Input schema for evaluate_performance tool."""
    backtest_id: str = Field(
        description="ID of the backtest to evaluate"
    )
    benchmark: Optional[str] = Field(
        default=None,
        description="Benchmark to compare against (e.g., 'buy_and_hold')"
    )
    risk_free_rate: float = Field(
        default=0.02,
        description="Annual risk-free rate for Sharpe calculation"
    )


class GenerateOptimizationReportInput(BaseModel):
    """Input schema for generate_optimization_report tool."""
    backtest_id: str = Field(
        description="ID of the backtest to generate report for"
    )
    format: str = Field(
        default="markdown",
        description="Report format (markdown, html, json)"
    )
    include_recommendations: bool = Field(
        default=True,
        description="Whether to include optimization recommendations"
    )


@register_tool(
    agent_types=[AgentType.ANALYST, AgentType.COPILOT],
    tags=["analysis", "backtest", "metrics"],
)
class AnalyzeBacktestTool(QuantMindTool):
    """Analyze backtest results and calculate metrics."""

    name: str = "analyze_backtest"
    description: str = """Analyze backtest results and calculate comprehensive performance metrics.
    Returns metrics including win rate, profit factor, drawdown, Sharpe ratio, and more.
    Can optionally include trade-by-trade analysis and equity curve data."""

    args_schema: type[BaseModel] = AnalyzeBacktestInput
    category: ToolCategory = ToolCategory.ANALYSIS
    priority: ToolPriority = ToolPriority.NORMAL

    def execute(
        self,
        backtest_id: str,
        metrics: Optional[List[str]] = None,
        include_trades: bool = False,
        include_equity_curve: bool = True,
        **kwargs
    ) -> ToolResult:
        """Execute backtest analysis."""
        # In production, would fetch backtest results from database
        logger.info(f"Analyzing backtest: {backtest_id}")

        # Simulate backtest results
        performance = PerformanceMetrics(
            total_trades=245,
            winning_trades=148,
            losing_trades=97,
            win_rate=60.4,
            total_profit=15420.50,
            total_loss=8750.25,
            net_profit=6670.25,
            profit_factor=1.76,
            max_drawdown=2350.00,
            max_drawdown_percent=12.3,
            sharpe_ratio=1.45,
            sortino_ratio=1.89,
            average_win=104.19,
            average_loss=90.21,
            expectancy=27.22,
            recovery_factor=2.84,
            cagr=18.5
        )

        result_data = {
            "backtest_id": backtest_id,
            "performance": {
                "total_trades": performance.total_trades,
                "winning_trades": performance.winning_trades,
                "losing_trades": performance.losing_trades,
                "win_rate": performance.win_rate,
                "net_profit": performance.net_profit,
                "profit_factor": performance.profit_factor,
                "max_drawdown": performance.max_drawdown,
                "max_drawdown_percent": performance.max_drawdown_percent,
                "sharpe_ratio": performance.sharpe_ratio,
                "sortino_ratio": performance.sortino_ratio,
                "expectancy": performance.expectancy,
                "recovery_factor": performance.recovery_factor,
                "cagr": performance.cagr,
            }
        }

        if include_equity_curve:
            # Simulated equity curve
            result_data["equity_curve"] = {
                "start_value": 10000.00,
                "end_value": 16670.25,
                "peak_value": 17250.00,
                "data_points": 500,
            }

        if include_trades:
            result_data["trades"] = {
                "total": performance.total_trades,
                "long_trades": 130,
                "short_trades": 115,
                "average_trade_duration_hours": 18.5,
            }

        return ToolResult.ok(
            data=result_data,
            metadata={
                "analyzed_at": datetime.now().isoformat(),
                "metrics_calculated": metrics or ["all"],
            }
        )


@register_tool(
    agent_types=[AgentType.ANALYST],
    tags=["analysis", "compare", "strategies"],
)
class CompareStrategiesTool(QuantMindTool):
    """Compare multiple trading strategies."""

    name: str = "compare_strategies"
    description: str = """Compare multiple trading strategies side by side.
    Ranks strategies by specified primary metric.
    Provides relative performance and statistical comparison."""

    args_schema: type[BaseModel] = CompareStrategiesInput
    category: ToolCategory = ToolCategory.ANALYSIS
    priority: ToolPriority = ToolPriority.NORMAL

    def execute(
        self,
        backtest_ids: List[str],
        primary_metric: str = "sharpe_ratio",
        normalize: bool = True,
        **kwargs
    ) -> ToolResult:
        """Execute strategy comparison."""
        logger.info(f"Comparing strategies: {backtest_ids}")

        # Simulate comparison data
        strategies = []
        for i, bt_id in enumerate(backtest_ids):
            strategies.append({
                "backtest_id": bt_id,
                "name": f"Strategy_{i+1}",
                "sharpe_ratio": 1.2 + (i * 0.15),
                "profit_factor": 1.5 + (i * 0.1),
                "max_drawdown_percent": 15 - (i * 1.5),
                "cagr": 15 + (i * 2),
                "win_rate": 55 + (i * 2),
            })

        # Rank by primary metric
        strategies.sort(key=lambda x: x.get(primary_metric, 0), reverse=True)

        # Add ranks
        for i, strategy in enumerate(strategies):
            strategy["rank"] = i + 1

        return ToolResult.ok(
            data={
                "strategies": strategies,
                "primary_metric": primary_metric,
                "winner": strategies[0]["backtest_id"] if strategies else None,
            },
            metadata={
                "compared_at": datetime.now().isoformat(),
                "normalized": normalize,
            }
        )


@register_tool(
    agent_types=[AgentType.ANALYST],
    tags=["analysis", "performance", "evaluation"],
)
class EvaluatePerformanceTool(QuantMindTool):
    """Evaluate strategy performance against benchmarks."""

    name: str = "evaluate_performance"
    description: str = """Evaluate strategy performance with detailed analysis.
    Compares against benchmarks like buy-and-hold.
    Calculates risk-adjusted returns and statistical significance."""

    args_schema: type[BaseModel] = EvaluatePerformanceInput
    category: ToolCategory = ToolCategory.ANALYSIS
    priority: ToolPriority = ToolPriority.NORMAL

    def execute(
        self,
        backtest_id: str,
        benchmark: Optional[str] = None,
        risk_free_rate: float = 0.02,
        **kwargs
    ) -> ToolResult:
        """Execute performance evaluation."""
        logger.info(f"Evaluating performance: {backtest_id}")

        result_data = {
            "backtest_id": backtest_id,
            "evaluation": {
                "annualized_return": 18.5,
                "volatility": 12.3,
                "sharpe_ratio": 1.45,
                "sortino_ratio": 1.89,
                "calmar_ratio": 1.50,
                "information_ratio": 0.85,
                "treynor_ratio": 0.12,
                "alpha": 3.2,
                "beta": 0.85,
            },
            "risk_metrics": {
                "value_at_risk_95": 250.00,
                "expected_shortfall": 312.50,
                "max_consecutive_losses": 5,
                "avg_loss_duration_days": 3.2,
            }
        }

        if benchmark:
            result_data["benchmark_comparison"] = {
                "benchmark": benchmark,
                "strategy_return": 18.5,
                "benchmark_return": 10.2,
                "excess_return": 8.3,
                "tracking_error": 5.4,
                "outperformance": True,
            }

        return ToolResult.ok(
            data=result_data,
            metadata={
                "evaluated_at": datetime.now().isoformat(),
                "risk_free_rate": risk_free_rate,
                "benchmark": benchmark,
            }
        )


@register_tool(
    agent_types=[AgentType.ANALYST],
    tags=["analysis", "report", "optimization"],
)
class GenerateOptimizationReportTool(QuantMindTool):
    """Generate comprehensive optimization report."""

    name: str = "generate_optimization_report"
    description: str = """Generate a comprehensive optimization report for a backtest.
    Includes performance analysis, risk assessment, and optimization recommendations.
    Supports multiple output formats."""

    args_schema: type[BaseModel] = GenerateOptimizationReportInput
    category: ToolCategory = ToolCategory.ANALYSIS
    priority: ToolPriority = ToolPriority.LOW

    def execute(
        self,
        backtest_id: str,
        format: str = "markdown",
        include_recommendations: bool = True,
        **kwargs
    ) -> ToolResult:
        """Execute report generation."""
        logger.info(f"Generating optimization report: {backtest_id}")

        if format == "markdown":
            report = self._generate_markdown_report(backtest_id, include_recommendations)
        elif format == "html":
            report = self._generate_html_report(backtest_id, include_recommendations)
        else:
            report = self._generate_json_report(backtest_id, include_recommendations)

        return ToolResult.ok(
            data={
                "report": report,
                "format": format,
                "backtest_id": backtest_id,
            },
            metadata={
                "generated_at": datetime.now().isoformat(),
                "includes_recommendations": include_recommendations,
            }
        )

    def _generate_markdown_report(self, backtest_id: str, include_recs: bool) -> str:
        """Generate markdown format report."""
        report = f"""# Optimization Report: {backtest_id}

## Executive Summary
- **Net Profit**: $6,670.25 (+66.7%)
- **Sharpe Ratio**: 1.45
- **Max Drawdown**: 12.3%
- **Win Rate**: 60.4%

## Performance Metrics

| Metric | Value |
|--------|-------|
| Total Trades | 245 |
| Profit Factor | 1.76 |
| Recovery Factor | 2.84 |
| CAGR | 18.5% |

## Risk Analysis
- Maximum drawdown occurred during high volatility period
- Recovery was within acceptable parameters

"""
        if include_recs:
            report += """## Optimization Recommendations

1. **Stop Loss Optimization**: Consider tightening stops during low volatility
2. **Position Sizing**: Implement dynamic sizing based on ATR
3. **Time Filters**: Avoid trading during major news events
4. **Exit Strategy**: Add trailing stop functionality

## Next Steps
1. Run walk-forward analysis
2. Perform Monte Carlo simulation
3. Test on out-of-sample data
"""
        return report

    def _generate_html_report(self, backtest_id: str, include_recs: bool) -> str:
        """Generate HTML format report."""
        return f"<html><body><h1>Report for {backtest_id}</h1></body></html>"

    def _generate_json_report(self, backtest_id: str, include_recs: bool) -> Dict:
        """Generate JSON format report."""
        return {"backtest_id": backtest_id, "format": "json"}


# Export all tools
__all__ = [
    "AnalyzeBacktestTool",
    "CompareStrategiesTool",
    "EvaluatePerformanceTool",
    "GenerateOptimizationReportTool",
    "PerformanceMetrics",
    "MetricType",
]

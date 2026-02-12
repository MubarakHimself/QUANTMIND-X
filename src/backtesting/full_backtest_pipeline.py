"""
Full Backtest Pipeline - Orchestrator for All 4 Variants

Task Group 3.7: Create full_backtest_pipeline.py orchestrator

Orchestrates all 4 backtest variants with comparison report generation
and overfitting detection.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, field
import logging
import json

# Import existing components
from backtesting.mt5_engine import MQL5Timeframe, MT5BacktestResult
from backtesting.mode_runner import (
    BacktestMode,
    run_vanilla_backtest,
    run_spiced_backtest,
    SpicedBacktestResult
)
from backtesting.walk_forward import WalkForwardOptimizer, WalkForwardResult
from backtesting.monte_carlo import MonteCarloSimulator, MonteCarloResult

logger = logging.getLogger(__name__)


@dataclass
class BacktestComparison:
    """Comparison result across all 4 variants."""
    vanilla_result: Optional[MT5BacktestResult] = None
    spiced_result: Optional[SpicedBacktestResult] = None
    vanilla_full_result: Optional[MT5BacktestResult] = None
    spiced_full_result: Optional[SpicedBacktestResult] = None

    # Monte Carlo results
    vanilla_mc_result: Optional[MonteCarloResult] = None
    spiced_mc_result: Optional[MonteCarloResult] = None

    # Comparison metrics
    overfitting_detected: bool = False
    overfitting_score: float = 0.0
    robustness_score: float = 0.0

    # Comparison analysis
    regime_impact: float = 0.0
    walk_forward_validation: float = 0.0

    # Recommendation
    recommendation: str = ""
    risk_level: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'vanilla_result': self.vanilla_result.to_dict() if self.vanilla_result else None,
            'spiced_result': self.spiced_result.to_dict() if self.spiced_result else None,
            'vanilla_full_result': self.vanilla_full_result.to_dict() if self.vanilla_full_result else None,
            'spiced_full_result': self.spiced_full_result.to_dict() if self.spiced_full_result else None,
            'vanilla_mc_result': self.vanilla_mc_result.to_dict() if self.vanilla_mc_result else None,
            'spiced_mc_result': self.spiced_mc_result.to_dict() if self.spiced_mc_result else None,
            'overfitting_detected': self.overfitting_detected,
            'overfitting_score': self.overfitting_score,
            'robustness_score': self.robustness_score,
            'regime_impact': self.regime_impact,
            'walk_forward_validation': self.walk_forward_validation,
            'recommendation': self.recommendation,
            'risk_level': self.risk_level
        }


class FullBacktestPipeline:
    """Orchestrator for all 4 backtest variants.

    Runs Vanilla, Spiced, Vanilla+Full, Spiced+Full backtests
    with comparison analysis and overfitting detection.

    Example:
        >>> pipeline = FullBacktestPipeline()
        >>> comparison = pipeline.run_all_variants(data, symbol, timeframe, strategy_code)
        >>> print(f"Robustness: {comparison.robustness_score:.2%}")
        >>> print(f"Recommendation: {comparison.recommendation}")
    """

    def __init__(
        self,
        initial_cash: float = 10000.0,
        commission: float = 0.001,
        slippage: float = 0.0,
        run_monte_carlo: bool = True,
        mc_simulations: int = 1000
    ):
        """Initialize full backtest pipeline.

        Args:
            initial_cash: Starting account balance
            commission: Commission per trade
            slippage: Slippage in price points
            run_monte_carlo: Whether to run Monte Carlo simulations
            mc_simulations: Number of Monte Carlo simulations
        """
        self.initial_cash = initial_cash
        self.commission = commission
        self.slippage = slippage
        self.run_monte_carlo = run_monte_carlo
        self.mc_simulations = mc_simulations

        logger.info("FullBacktestPipeline initialized")

    def run_all_variants(
        self,
        data: pd.DataFrame,
        symbol: str,
        timeframe: int,
        strategy_code: str
    ) -> BacktestComparison:
        """Run all 4 backtest variants with comparison analysis.

        Args:
            data: OHLCV data
            symbol: Trading symbol
            timeframe: MQL5 timeframe constant
            strategy_code: Python strategy code

        Returns:
            BacktestComparison with all results and analysis
        """
        logger.info(f"Running all 4 backtest variants for {symbol}")

        comparison = BacktestComparison()

        # 1. Run Vanilla backtest
        logger.info("Running Vanilla backtest...")
        try:
            comparison.vanilla_result = run_vanilla_backtest(
                strategy_code=strategy_code,
                data=data,
                symbol=symbol,
                timeframe=timeframe,
                initial_cash=self.initial_cash,
                commission=self.commission,
                slippage=self.slippage
            )

            # Run Monte Carlo on Vanilla result
            if self.run_monte_carlo and comparison.vanilla_result:
                mc_sim = MonteCarloSimulator(num_simulations=self.mc_simulations)
                comparison.vanilla_mc_result = mc_sim.simulate(comparison.vanilla_result, data)

        except Exception as e:
            logger.error(f"Vanilla backtest failed: {e}")

        # 2. Run Spiced backtest
        logger.info("Running Spiced backtest...")
        try:
            comparison.spiced_result = run_spiced_backtest(
                strategy_code=strategy_code,
                data=data,
                symbol=symbol,
                timeframe=timeframe,
                initial_cash=self.initial_cash,
                commission=self.commission,
                slippage=self.slippage
            )

            # Run Monte Carlo on Spiced result
            if self.run_monte_carlo and comparison.spiced_result:
                mc_sim = MonteCarloSimulator(num_simulations=self.mc_simulations)
                comparison.spiced_mc_result = mc_sim.simulate(comparison.spiced_result, data)

        except Exception as e:
            logger.error(f"Spiced backtest failed: {e}")

        # 3. Run Vanilla+Full (Walk-Forward)
        logger.info("Running Vanilla+Full (Walk-Forward) backtest...")
        try:
            wf_optimizer = WalkForwardOptimizer(train_pct=0.5, test_pct=0.2, gap_pct=0.1)
            wf_result = wf_optimizer.optimize(
                data=data,
                symbol=symbol,
                timeframe=timeframe,
                strategy_code=strategy_code,
                initial_cash=self.initial_cash,
                commission=self.commission,
                slippage=self.slippage
            )

            # Convert to MT5BacktestResult format
            comparison.vanilla_full_result = MT5BacktestResult(
                sharpe=wf_result.aggregate_metrics.get('sharpe_mean', 0.0),
                return_pct=wf_result.aggregate_metrics.get('return_pct_mean', 0.0),
                drawdown=wf_result.aggregate_metrics.get('drawdown_mean', 0.0),
                trades=wf_result.aggregate_metrics.get('total_trades', 0),
                log=f"Walk-Forward: {wf_result.total_windows} windows, win rate: {wf_result.win_rate:.2%}",
                initial_cash=self.initial_cash,
                final_cash=self.initial_cash * (1 + wf_result.aggregate_metrics.get('return_pct_mean', 0.0) / 100),
                equity_curve=wf_result.aggregate_equity_curve,
                trade_history=wf_result.all_trade_history
            )

        except Exception as e:
            logger.error(f"Vanilla+Full backtest failed: {e}")

        # 4. Run Spiced+Full (Walk-Forward + Regime Filter)
        logger.info("Running Spiced+Full (Walk-Forward + Regime Filter) backtest...")
        try:
            wf_optimizer_spiced = WalkForwardOptimizer(train_pct=0.5, test_pct=0.2, gap_pct=0.1)
            wf_result_spiced = wf_optimizer_spiced.optimize(
                data=data,
                symbol=symbol,
                timeframe=timeframe,
                strategy_code=strategy_code,
                initial_cash=self.initial_cash,
                commission=self.commission,
                slippage=self.slippage,
                use_regime_filter=True
            )

            # Convert to SpicedBacktestResult format
            comparison.spiced_full_result = SpicedBacktestResult(
                sharpe=wf_result_spiced.aggregate_metrics.get('sharpe_mean', 0.0),
                return_pct=wf_result_spiced.aggregate_metrics.get('return_pct_mean', 0.0),
                drawdown=wf_result_spiced.aggregate_metrics.get('drawdown_mean', 0.0),
                trades=wf_result_spiced.aggregate_metrics.get('total_trades', 0),
                log=f"Walk-Forward + Regime: {wf_result_spiced.total_windows} windows, win rate: {wf_result_spiced.win_rate:.2%}",
                initial_cash=self.initial_cash,
                final_cash=self.initial_cash * (1 + wf_result_spiced.aggregate_metrics.get('return_pct_mean', 0.0) / 100),
                equity_curve=wf_result_spiced.aggregate_equity_curve,
                trade_history=wf_result_spiced.all_trade_history,
                regime_distribution=self._aggregate_regime_distribution(wf_result_spiced.window_regime_stats),
                filtered_trades=sum(w.get('filtered_trades', 0) for w in wf_result_spiced.window_regime_stats),
                avg_regime_quality=np.mean([w.get('avg_regime_quality', 0.0) for w in wf_result_spiced.window_regime_stats]) if wf_result_spiced.window_regime_stats else 0.0
            )

        except Exception as e:
            logger.error(f"Spiced+Full backtest failed: {e}")

        # 5. Run comparison analysis
        self._analyze_comparison(comparison)

        logger.info("All 4 backtest variants completed")
        return comparison

    def _aggregate_regime_distribution(self, window_stats: List[Dict[str, Any]]) -> Dict[str, int]:
        """Aggregate regime distribution across windows.

        Args:
            window_stats: List of window regime statistics

        Returns:
            Aggregated regime distribution
        """
        aggregated = {}

        for stats in window_stats:
            for regime, count in stats.get('regime_distribution', {}).items():
                aggregated[regime] = aggregated.get(regime, 0) + count

        return aggregated

    def _analyze_comparison(self, comparison: BacktestComparison):
        """Analyze comparison results and detect overfitting.

        Args:
            comparison: Comparison result to analyze
        """
        # Calculate overfitting score
        if comparison.vanilla_result and comparison.spiced_result:
            vanilla_return = comparison.vanilla_result.return_pct
            spiced_return = comparison.spiced_result.return_pct

            # Regime impact: difference between Vanilla and Spiced
            comparison.regime_impact = abs(vanilla_return - spiced_return)

            # Overfitting detected if Vanilla >> Spiced (performs poorly when filtered)
            if vanilla_return > 0 and spiced_return < 0:
                comparison.overfitting_detected = True
                comparison.overfitting_score = 1.0
            elif vanilla_return > spiced_return * 1.5:
                comparison.overfitting_detected = True
                comparison.overfitting_score = 0.5
            else:
                comparison.overfitting_score = 0.0

        # Calculate walk-forward validation
        if comparison.vanilla_full_result:
            comparison.walk_forward_validation = comparison.vanilla_full_result.return_pct

        # Calculate robustness score
        scores = []

        # 1. Regime robustness: Spiced should not be much worse than Vanilla
        if comparison.vanilla_result and comparison.spiced_result:
            ratio = comparison.spiced_result.return_pct / comparison.vanilla_result.return_pct if comparison.vanilla_result.return_pct != 0 else 0
            scores.append(min(abs(ratio), 1.0))

        # 2. Walk-forward validation: should be positive
        if comparison.vanilla_full_result:
            scores.append(1.0 if comparison.vanilla_full_result.return_pct > 0 else 0.0)

        # 3. Monte Carlo confidence: 95% CI should be positive
        if comparison.vanilla_mc_result:
            scores.append(1.0 if comparison.vanilla_mc_result.confidence_interval_5th > 0 else 0.0)

        comparison.robustness_score = np.mean(scores) if scores else 0.0

        # Generate recommendation
        if comparison.robustness_score >= 0.8:
            comparison.recommendation = "Deploy: Strategy shows robust performance across all variants"
            comparison.risk_level = "Low"
        elif comparison.robustness_score >= 0.5:
            comparison.recommendation = "Caution: Strategy shows moderate robustness, consider additional testing"
            comparison.risk_level = "Medium"
        else:
            comparison.recommendation = "Do Not Deploy: Strategy lacks robustness, likely overfitted"
            comparison.risk_level = "High"

        if comparison.overfitting_detected:
            comparison.recommendation += " (Overfitting Detected)"
            comparison.risk_level = "High"

    def generate_comparison_report(
        self,
        comparison: BacktestComparison,
        output_format: str = "text"
    ) -> str:
        """Generate comparison report.

        Args:
            comparison: Comparison result
            output_format: Output format ('text', 'json', 'markdown')

        Returns:
            Formatted report string
        """
        if output_format == "json":
            return json.dumps(comparison.to_dict(), indent=2)

        elif output_format == "markdown":
            return self._generate_markdown_report(comparison)

        else:  # text
            return self._generate_text_report(comparison)

    def _generate_text_report(self, comparison: BacktestComparison) -> str:
        """Generate text format report.

        Args:
            comparison: Comparison result

        Returns:
            Text report string
        """
        lines = []
        lines.append("=" * 70)
        lines.append("BACKTEST COMPARISON REPORT")
        lines.append("=" * 70)
        lines.append("")

        # Vanilla results
        if comparison.vanilla_result:
            lines.append("VANILLA BACKTEST:")
            lines.append(f"  Return: {comparison.vanilla_result.return_pct:.2%}")
            lines.append(f"  Sharpe: {comparison.vanilla_result.sharpe:.2f}")
            lines.append(f"  Drawdown: {comparison.vanilla_result.drawdown:.2%}")
            lines.append(f"  Trades: {comparison.vanilla_result.trades}")
            lines.append("")

        # Spiced results
        if comparison.spiced_result:
            lines.append("SPICED BACKTEST (with regime filtering):")
            lines.append(f"  Return: {comparison.spiced_result.return_pct:.2%}")
            lines.append(f"  Sharpe: {comparison.spiced_result.sharpe:.2f}")
            lines.append(f"  Drawdown: {comparison.spiced_result.drawdown:.2%}")
            lines.append(f"  Trades: {comparison.spiced_result.trades}")
            lines.append(f"  Filtered Trades: {comparison.spiced_result.filtered_trades}")
            lines.append(f"  Avg Regime Quality: {comparison.spiced_result.avg_regime_quality:.2%}")
            lines.append("")

        # Vanilla+Full results
        if comparison.vanilla_full_result:
            lines.append("VANILLA+FULL (Walk-Forward):")
            lines.append(f"  Return: {comparison.vanilla_full_result.return_pct:.2%}")
            lines.append(f"  Sharpe: {comparison.vanilla_full_result.sharpe:.2f}")
            lines.append(f"  Drawdown: {comparison.vanilla_full_result.drawdown:.2%}")
            lines.append("")

        # Monte Carlo results
        if comparison.vanilla_mc_result:
            lines.append("MONTE CARLO SIMULATION:")
            lines.append(f"  95% CI: [{comparison.vanilla_mc_result.confidence_interval_5th:.2%}, {comparison.vanilla_mc_result.confidence_interval_95th:.2%}]")
            lines.append(f"  VaR (95%): {comparison.vanilla_mc_result.value_at_risk_95:.2%}")
            lines.append(f"  Expected Shortfall (95%): {comparison.vanilla_mc_result.expected_shortfall_95:.2%}")
            lines.append(f"  Probability Profitable: {comparison.vanilla_mc_result.probability_profitable:.2%}")
            lines.append("")

        # Analysis
        lines.append("ANALYSIS:")
        lines.append(f"  Overfitting Detected: {comparison.overfitting_detected}")
        lines.append(f"  Overfitting Score: {comparison.overfitting_score:.2%}")
        lines.append(f"  Robustness Score: {comparison.robustness_score:.2%}")
        lines.append(f"  Regime Impact: {comparison.regime_impact:.2%}")
        lines.append(f"  Walk-Forward Validation: {comparison.walk_forward_validation:.2%}")
        lines.append("")

        # Recommendation
        lines.append("RECOMMENDATION:")
        lines.append(f"  {comparison.recommendation}")
        lines.append(f"  Risk Level: {comparison.risk_level}")
        lines.append("")

        lines.append("=" * 70)

        return "\n".join(lines)

    def _generate_markdown_report(self, comparison: BacktestComparison) -> str:
        """Generate markdown format report.

        Args:
            comparison: Comparison result

        Returns:
            Markdown report string
        """
        lines = []
        lines.append("# Backtest Comparison Report")
        lines.append("")
        lines.append("## Results Summary")
        lines.append("")
        lines.append("| Variant | Return | Sharpe | Drawdown | Trades |")
        lines.append("|---------|--------|--------|----------|--------|")

        if comparison.vanilla_result:
            lines.append(
                f"| Vanilla | {comparison.vanilla_result.return_pct:.2%} | "
                f"{comparison.vanilla_result.sharpe:.2f} | {comparison.vanilla_result.drawdown:.2%} | "
                f"{comparison.vanilla_result.trades} |"
            )

        if comparison.spiced_result:
            lines.append(
                f"| Spiced | {comparison.spiced_result.return_pct:.2%} | "
                f"{comparison.spiced_result.sharpe:.2f} | {comparison.spiced_result.drawdown:.2%} | "
                f"{comparison.spiced_result.trades} |"
            )

        if comparison.vanilla_full_result:
            lines.append(
                f"| Vanilla+Full | {comparison.vanilla_full_result.return_pct:.2%} | "
                f"{comparison.vanilla_full_result.sharpe:.2f} | {comparison.vanilla_full_result.drawdown:.2%} | "
                f"{comparison.vanilla_full_result.trades} |"
            )

        lines.append("")
        lines.append("## Analysis")
        lines.append("")
        lines.append(f"- **Overfitting Detected**: {comparison.overfitting_detected}")
        lines.append(f"- **Robustness Score**: {comparison.robustness_score:.2%}")
        lines.append(f"- **Regime Impact**: {comparison.regime_impact:.2%}")
        lines.append("")
        lines.append("## Recommendation")
        lines.append("")
        lines.append(f"**{comparison.recommendation}**")
        lines.append("")
        lines.append(f"Risk Level: **{comparison.risk_level}**")

        return "\n".join(lines)


__all__ = [
    'FullBacktestPipeline',
    'BacktestComparison',
]

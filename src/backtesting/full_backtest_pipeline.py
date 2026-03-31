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
from backtesting.pbo_calculator import PBOCalculator
from src.data.data_manager import DataManager

logger = logging.getLogger(__name__)


@dataclass
class DataSplitConfig:
    """Configuration for train/test/gap data split.

    Default 30/40/30 split:
    - 30% training data (in-sample)
    - 40% test data (out-of-sample)
    - 30% gap (excluded, prevents look-ahead bias)
    """
    train_pct: float = 0.30
    test_pct: float = 0.40
    gap_pct: float = 0.30

    def __post_init__(self):
        total = self.train_pct + self.test_pct + self.gap_pct
        if abs(total - 1.0) > 0.001:
            raise ValueError(
                f"DataSplitConfig percentages ({self.train_pct:.0%} + "
                f"{self.test_pct:.0%} + {self.gap_pct:.0%}) must sum to 100%"
            )

    def as_walk_forward_params(self) -> Dict[str, float]:
        """Return dict suitable for WalkForwardOptimizer constructor."""
        return {
            "train_pct": self.train_pct,
            "test_pct": self.test_pct,
            "gap_pct": self.gap_pct,
        }


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

    # PBO (Probability of Backtest Overfitting)
    pbo: float = 0.5
    pbo_recommendation: str = ""
    pbo_confidence_interval: tuple = (0.0, 0.0)

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
            'pbo': self.pbo,
            'pbo_recommendation': self.pbo_recommendation,
            'pbo_confidence_interval': self.pbo_confidence_interval,
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
        broker_id: str = "icmarkets_raw",
        run_monte_carlo: bool = True,
        mc_simulations: int = 1000,
        run_pbo: bool = True,
        pbo_blocks: int = 5,
        pbo_simulations: int = 100,
        data_split: Optional[DataSplitConfig] = None,
    ):
        """Initialize full backtest pipeline.

        Args:
            initial_cash: Starting account balance
            commission: Commission per trade
            slippage: Slippage in price points
            broker_id: Broker identifier for fee-aware Kelly
            run_monte_carlo: Whether to run Monte Carlo simulations
            mc_simulations: Number of Monte Carlo simulations
            run_pbo: Whether to run PBO calculation
            pbo_blocks: Number of blocks for PBO calculation
            pbo_simulations: Number of simulations for PBO
            data_split: Train/test/gap split config (default: 30/40/30)
        """
        self.initial_cash = initial_cash
        self.commission = commission
        self.slippage = slippage
        self.broker_id = broker_id
        self.run_monte_carlo = run_monte_carlo
        self.mc_simulations = mc_simulations
        self.run_pbo = run_pbo
        self.pbo_blocks = pbo_blocks
        self.pbo_simulations = pbo_simulations
        self.data_split = data_split or DataSplitConfig()

        # Initialize PBO calculator
        self._pbo_calculator = PBOCalculator(
            n_blocks=pbo_blocks,
            n_simulations=pbo_simulations
        )

        logger.info(
            f"FullBacktestPipeline initialized: data_split="
            f"{self.data_split.train_pct:.0%}/{self.data_split.test_pct:.0%}/"
            f"{self.data_split.gap_pct:.0%}"
        )

    def fetch_data(self, symbol: str, timeframe: int) -> pd.DataFrame:
        """Fetch OHLCV data from Dukascopy via DataManager.

        Args:
            symbol: Trading symbol (e.g., "EURUSD")
            timeframe: MQL5 timeframe constant

        Returns:
            DataFrame with OHLCV data (max 730 days)
        """
        logger.info(f"Fetching data for {symbol} from Dukascopy...")
        dm = DataManager(prefer_dukascopy=True)

        # Calculate bar count: 730 days * 24 hours for H1
        # Use a large count to get max data (approximately 730 days)
        lookback_days = 730
        timeframe_minutes = MQL5Timeframe.to_minutes(timeframe)
        bars_needed = (lookback_days * 24 * 60) // timeframe_minutes

        data = dm.fetch_data(symbol, timeframe, count=bars_needed)

        if len(data) == 0:
            logger.error(f"No data fetched for {symbol}")
        else:
            logger.info(f"Fetched {len(data)} bars for {symbol}")

        return data

    def run_all_variants(
        self,
        data: Optional[pd.DataFrame],
        symbol: str,
        timeframe: int,
        strategy_code: str
    ) -> BacktestComparison:
        """Run all 4 backtest variants with comparison analysis.

        Args:
            data: OHLCV data (optional, will fetch from Dukascopy if None)
            symbol: Trading symbol
            timeframe: MQL5 timeframe constant
            strategy_code: Python strategy code

        Returns:
            BacktestComparison with all results and analysis
        """
        # Fetch data if not provided
        if data is None:
            logger.info(f"No data provided, fetching from Dukascopy for {symbol}")
            data = self.fetch_data(symbol, timeframe)

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
                slippage=self.slippage,
                broker_id=self.broker_id
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
                slippage=self.slippage,
                broker_id=self.broker_id
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
            wf_params = self.data_split.as_walk_forward_params()
            wf_optimizer = WalkForwardOptimizer(**wf_params)
            wf_result = wf_optimizer.optimize(
                data=data,
                symbol=symbol,
                timeframe=timeframe,
                strategy_code=strategy_code,
                initial_cash=self.initial_cash,
                commission=self.commission,
                slippage=self.slippage,
                broker_id=self.broker_id
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
            wf_params_spiced = self.data_split.as_walk_forward_params()
            wf_optimizer_spiced = WalkForwardOptimizer(**wf_params_spiced)
            wf_result_spiced = wf_optimizer_spiced.optimize(
                data=data,
                symbol=symbol,
                timeframe=timeframe,
                strategy_code=strategy_code,
                initial_cash=self.initial_cash,
                commission=self.commission,
                slippage=self.slippage,
                use_regime_filter=True,
                broker_id=self.broker_id
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

        # Calculate PBO (Probability of Backtest Overfitting)
        if self.run_pbo:
            self._calculate_pbo(comparison)

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

        # 4. PBO score: lower PBO is better
        if self.run_pbo and comparison.pbo is not None:
            scores.append(1.0 - comparison.pbo)

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

        # Adjust recommendation based on PBO
        if self.run_pbo and comparison.pbo > 0.5:
            comparison.recommendation += " (High PBO detected)"
            comparison.risk_level = "High"

    def _calculate_pbo(self, comparison: BacktestComparison):
        """Calculate PBO from backtest results.

        Args:
            comparison: Comparison result with backtest data
        """
        try:
            # Extract returns from trade history if available
            returns = []

            if comparison.vanilla_result and comparison.vanilla_result.trade_history:
                for trade in comparison.vanilla_result.trade_history:
                    if isinstance(trade, dict) and 'profit' in trade:
                        returns.append(trade['profit'])
                    elif hasattr(trade, 'profit'):
                        returns.append(trade.profit)

            # If no trade history, try equity curve
            if not returns and comparison.vanilla_result and comparison.vanilla_result.equity_curve:
                equity = comparison.vanilla_result.equity_curve
                if isinstance(equity, list) and len(equity) > 1:
                    returns = [equity[i+1] - equity[i] for i in range(len(equity)-1)]

            # Fallback: use return percentage if available
            if not returns and comparison.vanilla_result:
                # Create synthetic returns from total return
                total_return = comparison.vanilla_result.return_pct / 100
                if comparison.vanilla_result.trades > 0:
                    avg_return = total_return / comparison.vanilla_result.trades
                    returns = [avg_return] * comparison.vanilla_result.trades

            if len(returns) >= 20:
                # Evaluate robustness using PBO calculator
                pbo_result = self._pbo_calculator.evaluate_strategy_robustness(returns)

                comparison.pbo = pbo_result.get('pbo', 0.5)
                comparison.pbo_recommendation = pbo_result.get('recommendation', 'UNKNOWN')
                comparison.pbo_confidence_interval = pbo_result.get('confidence_interval', (0.0, 0.0))

                # Update overfitting detection based on PBO
                if comparison.pbo > 0.5:
                    comparison.overfitting_detected = True
                    comparison.overfitting_score = max(comparison.overfitting_score, comparison.pbo)

                logger.info(f"PBO Calculation: {comparison.pbo:.3f} - {comparison.pbo_recommendation}")
            else:
                logger.warning(f"Insufficient trade data for PBO calculation: {len(returns)} trades")
                comparison.pbo = 0.5
                comparison.pbo_recommendation = "INSUFFICIENT_DATA"

        except Exception as e:
            logger.error(f"Error calculating PBO: {e}")
            comparison.pbo = 0.5
            comparison.pbo_recommendation = "ERROR"

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
    'DataSplitConfig',
]

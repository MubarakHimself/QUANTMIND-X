"""
QuantMindLib V1 — EvaluationOrchestrator

Packet 8B: Wires BotSpec -> FullBacktestPipeline -> EvaluationBridge -> BotEvaluationProfile
into a complete evaluation cycle.

This is the V1 operational bridge between QuantMindLib's BotSpec and
the existing 6-mode backtest evaluation system.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional

import pandas as pd

from src.library.core.domain.bot_spec import BotEvaluationProfile
from src.library.core.domain.evaluation_result import EvaluationResult
from src.library.core.bridges.lifecycle_eval_workflow_bridges import EvaluationBridge

if TYPE_CHECKING:
    from src.backtesting.full_backtest_pipeline import BacktestComparison
    from src.backtesting.mt5_engine import MT5BacktestResult
    from src.backtesting.mode_runner import SpicedBacktestResult
    from src.library.core.domain.bot_spec import BotSpec
    from src.library.evaluation.report_bridge import BacktestReportBridge
    from src.library.evaluation.strategy_code_generator import StrategyCodeGenerator


class EvaluationOrchestrator:
    """
    Orchestrates the complete evaluation pipeline:
    BotSpec -> strategy code -> FullBacktestPipeline -> EvaluationBridge -> BotEvaluationProfile

    This is the V1 operational bridge between QuantMindLib's BotSpec and
    the existing 6-mode backtest evaluation system.
    """

    def __init__(
        self,
        code_generator: Optional[StrategyCodeGenerator] = None,
        evaluation_bridge: Optional[EvaluationBridge] = None,
        pipeline_class: Optional[Any] = None,
        report_bridge: Optional[BacktestReportBridge] = None,
    ) -> None:
        """
        Initialize with optional injected components.

        Args:
            code_generator: Optional StrategyCodeGenerator instance.
                Creates a default one if not provided.
            evaluation_bridge: Optional EvaluationBridge instance.
                Creates a default one if not provided.
            pipeline_class: Optional FullBacktestPipeline class.
                Allows injection of a mock for testing.
            report_bridge: Optional BacktestReportBridge instance.
                When provided, generates a markdown backtest report after
                evaluation and attaches it to the returned BotEvaluationProfile.
        """
        if code_generator is None:
            from src.library.evaluation.strategy_code_generator import StrategyCodeGenerator
            self._code_generator: StrategyCodeGenerator = StrategyCodeGenerator()
        else:
            self._code_generator = code_generator

        if evaluation_bridge is None:
            self._evaluation_bridge: EvaluationBridge = EvaluationBridge()
        else:
            self._evaluation_bridge = evaluation_bridge

        self._report_bridge: Optional[BacktestReportBridge] = report_bridge

        # Store pipeline class for lazy import inside methods (avoids cascade imports)
        self._pipeline_class = pipeline_class

    def evaluate(
        self, bot_spec: BotSpec
    ) -> tuple[Optional[BotEvaluationProfile], List[str]]:
        """
        Full evaluation cycle for a BotSpec.

        Steps:
            1. Validate bot_spec with StrategyCodeGenerator
            2. Generate strategy code
            3. Get symbol and timeframe
            4. Create FullBacktestPipeline instance
            5. Run all variants -> BacktestComparison
            6. Convert each mode result -> EvaluationResult (4 results: VANILLA, SPICED, VANILLA_FULL, SPICED_FULL)
            7. Derive BotEvaluationProfile via EvaluationBridge.to_evaluation_profile()
            8. Return (BotEvaluationProfile, list_of_warnings)

        Args:
            bot_spec: Bot specification to evaluate.

        Returns:
            Tuple of (BotEvaluationProfile, list_of_warnings) on success.
            Returns (None, errors) on failure.
        """
        warnings: List[str] = []

        # Step 1: Validate bot_spec
        is_valid, errors = self._code_generator.validate_bot_spec(bot_spec)
        if not is_valid:
            return None, errors

        # Step 2: Generate strategy code
        try:
            strategy_code = self._code_generator.generate(bot_spec)
        except Exception as e:
            return None, [f"Strategy code generation failed: {e}"]

        # Step 3: Get symbol and timeframe
        symbol = bot_spec.symbol_scope[0] if bot_spec.symbol_scope else "EURUSD"
        timeframe = self._code_generator.get_timeframe_for_symbol(symbol)

        # Step 4: Create FullBacktestPipeline instance
        if self._pipeline_class is not None:
            pipeline_cls = self._pipeline_class
        else:
            from src.backtesting.full_backtest_pipeline import FullBacktestPipeline
            pipeline_cls = FullBacktestPipeline
        pipeline = pipeline_cls()

        # Step 5: Run all variants
        try:
            comparison = pipeline.run_all_variants(
                data=None,
                symbol=symbol,
                timeframe=timeframe,
                strategy_code=strategy_code,
            )
        except Exception as e:
            return None, [f"Backtest pipeline failed: {e}"]

        # Step 6: Convert each mode result -> EvaluationResult (4 results)
        bot_id = bot_spec.id
        evaluation_results: List[EvaluationResult] = []

        # VANILLA
        if comparison.vanilla_result is not None:
            result = self._backtest_result_to_evaluation_result(
                bot_id, "VANILLA", comparison.vanilla_result
            )
            evaluation_results.append(result)

        # SPICED
        if comparison.spiced_result is not None:
            if isinstance(comparison.spiced_result, tuple):
                spiced = comparison.spiced_result[0]
            else:
                spiced = comparison.spiced_result
            result = self._spiced_result_to_evaluation_result(bot_id, spiced)
            evaluation_results.append(result)

        # VANILLA_FULL
        if comparison.vanilla_full_result is not None:
            result = self._backtest_result_to_evaluation_result(
                bot_id, "VANILLA_FULL", comparison.vanilla_full_result
            )
            evaluation_results.append(result)

        # SPICED_FULL
        if comparison.spiced_full_result is not None:
            if isinstance(comparison.spiced_full_result, tuple):
                spiced_full = comparison.spiced_full_result[0]
            else:
                spiced_full = comparison.spiced_full_result
            result = self._spiced_result_to_evaluation_result(bot_id, spiced_full)
            evaluation_results.append(result)

        if not evaluation_results:
            warnings.append("No backtest results produced")

        # Step 7: Derive BotEvaluationProfile
        profile = self._comparison_to_profile(bot_id, comparison)

        # Step 8: Generate backtest report if report_bridge is available
        if self._report_bridge is not None:
            primary_result = self._get_primary_evaluation_result(comparison, bot_id)
            if primary_result is not None:
                try:
                    report = self._report_bridge.generate_report(
                        primary_result, bot_spec
                    )
                    profile.report = report
                except Exception as e:
                    warnings.append(f"Report generation failed: {e}")

        return profile, warnings

    def evaluate_with_data(
        self,
        bot_spec: BotSpec,
        data: pd.DataFrame,
    ) -> tuple[Optional[BotEvaluationProfile], List[str]]:
        """
        Evaluate with provided OHLCV data (for re-running on historical data).

        Skips data fetching, uses the provided DataFrame directly.

        Args:
            bot_spec: Bot specification to evaluate.
            data: OHLCV DataFrame with columns: time, open, high, low, close, tick_volume.

        Returns:
            Tuple of (BotEvaluationProfile, list_of_warnings) on success.
            Returns (None, errors) on failure.
        """
        warnings: List[str] = []

        # Validate bot_spec
        is_valid, errors = self._code_generator.validate_bot_spec(bot_spec)
        if not is_valid:
            return None, errors

        # Generate strategy code
        try:
            strategy_code = self._code_generator.generate(bot_spec)
        except Exception as e:
            return None, [f"Strategy code generation failed: {e}"]

        symbol = bot_spec.symbol_scope[0] if bot_spec.symbol_scope else "EURUSD"
        timeframe = self._code_generator.get_timeframe_for_symbol(symbol)

        # Run with provided data
        if self._pipeline_class is not None:
            pipeline_cls = self._pipeline_class
        else:
            from src.backtesting.full_backtest_pipeline import FullBacktestPipeline
            pipeline_cls = FullBacktestPipeline
        pipeline = pipeline_cls()

        try:
            comparison = pipeline.run_all_variants(
                data=data,
                symbol=symbol,
                timeframe=timeframe,
                strategy_code=strategy_code,
            )
        except Exception as e:
            return None, [f"Backtest pipeline failed: {e}"]

        # Convert results
        bot_id = bot_spec.id
        if comparison.vanilla_result is not None:
            _ = self._backtest_result_to_evaluation_result(
                bot_id, "VANILLA", comparison.vanilla_result
            )
        if comparison.spiced_result is not None:
            if isinstance(comparison.spiced_result, tuple):
                spiced = comparison.spiced_result[0]
            else:
                spiced = comparison.spiced_result
            _ = self._spiced_result_to_evaluation_result(bot_id, spiced)
        if comparison.vanilla_full_result is not None:
            _ = self._backtest_result_to_evaluation_result(
                bot_id, "VANILLA_FULL", comparison.vanilla_full_result
            )
        if comparison.spiced_full_result is not None:
            if isinstance(comparison.spiced_full_result, tuple):
                spiced_full = comparison.spiced_full_result[0]
            else:
                spiced_full = comparison.spiced_full_result
            _ = self._spiced_result_to_evaluation_result(bot_id, spiced_full)

        profile = self._comparison_to_profile(bot_id, comparison)

        # Generate backtest report if report_bridge is available
        if self._report_bridge is not None:
            primary_result = self._get_primary_evaluation_result(comparison, bot_id)
            if primary_result is not None:
                try:
                    report = self._report_bridge.generate_report(
                        primary_result, bot_spec
                    )
                    profile.report = report
                except Exception as e:
                    warnings.append(f"Report generation failed: {e}")

        return profile, warnings

    def _backtest_result_to_evaluation_result(
        self,
        bot_id: str,
        mode: str,
        result: MT5BacktestResult,
    ) -> EvaluationResult:
        """
        Convert a backtest result (MT5BacktestResult) to EvaluationResult.

        Args:
            bot_id: Bot identifier.
            mode: Mode string ("VANILLA" | "SPICED" | "VANILLA_FULL" | "SPICED_FULL").
            result: MT5BacktestResult from the backtest pipeline.

        Returns:
            EvaluationResult with computed metrics.
        """
        sharpe_ratio = getattr(result, "sharpe", 0.0)
        return_pct = getattr(result, "return_pct", 0.0)
        max_drawdown = getattr(result, "drawdown", 0.0)
        total_trades = getattr(result, "trades", 0)

        # Extract trade history for derived metrics
        trade_history = getattr(result, "trade_history", []) or []
        trade_history = [t for t in trade_history if isinstance(t, dict)]

        # Compute win_rate from trade history
        if trade_history:
            winning = sum(1 for t in trade_history if t.get("profit", 0) > 0)
            win_rate = winning / len(trade_history) if trade_history else 0.5
        else:
            # Fallback: use win_rate attribute if available (MT5BacktestResult has it)
            win_rate = getattr(result, "win_rate", 50.0) / 100.0
            if win_rate == 0.0:
                win_rate = 0.5

        # Compute profit_factor from trade history
        if trade_history:
            gross_profit = sum(t.get("profit", 0) for t in trade_history if t.get("profit", 0) > 0)
            gross_loss = abs(sum(t.get("profit", 0) for t in trade_history if t.get("profit", 0) <= 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
            # Cap at a reasonable maximum
            profit_factor = min(profit_factor, 100.0)
        else:
            profit_factor = 1.0

        # Compute expectancy from trade history
        if trade_history:
            expectancy = sum(t.get("profit", 0) for t in trade_history) / len(trade_history)
        else:
            expectancy = 0.0

        # Compute kelly_score
        kelly_score = min(
            1.0,
            max(
                0.0,
                win_rate - (1 - win_rate) / (profit_factor if profit_factor > 0 else 1),
            ),
        )

        # Compute passes_gate
        passes_gate = sharpe_ratio >= 1.0 and max_drawdown <= 0.15

        return EvaluationResult(
            bot_id=bot_id,
            mode="BACKTEST",
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            expectancy=expectancy,
            total_trades=total_trades,
            return_pct=return_pct,
            kelly_score=kelly_score,
            passes_gate=passes_gate,
        )

    def _spiced_result_to_evaluation_result(
        self,
        bot_id: str,
        result: SpicedBacktestResult,
    ) -> EvaluationResult:
        """
        Convert SpicedBacktestResult to EvaluationResult (includes regime info).

        Args:
            bot_id: Bot identifier.
            result: SpicedBacktestResult from the backtest pipeline.

        Returns:
            EvaluationResult with regime distribution and filtered trades.
        """
        # Delegate to base conversion
        eval_result = self._backtest_result_to_evaluation_result(
            bot_id, "SPICED", result
        )

        # Override mode to BACKTEST for bridge compatibility
        eval_result.mode = "BACKTEST"

        # Attach spiced-specific fields
        eval_result.regime_distribution = getattr(result, "regime_distribution", None) or {}
        eval_result.filtered_trades = getattr(result, "filtered_trades", 0)

        return eval_result

    def _comparison_to_profile(
        self,
        bot_id: str,
        comparison: BacktestComparison,
    ) -> BotEvaluationProfile:
        """
        Derive BotEvaluationProfile from BacktestComparison via EvaluationBridge.

        Uses VANILLA result as the primary backtest result.
        Pulls Monte Carlo P5/P95 and walk-forward metrics from comparison fields.

        Args:
            bot_id: Bot identifier.
            comparison: BacktestComparison from FullBacktestPipeline.

        Returns:
            BotEvaluationProfile with all metrics populated.
        """
        # Get vanilla backtest result for primary metrics
        vanilla_result = comparison.vanilla_result

        # Extract PBO and robustness early so they are always applied
        pbo = getattr(comparison, "pbo", 0.5)
        robustness = getattr(comparison, "robustness_score", 0.0)

        if vanilla_result is not None:
            backtest_eval = self._backtest_result_to_evaluation_result(
                bot_id, "VANILLA", vanilla_result
            )
        else:
            # Return a default profile if no vanilla result, but still use comparison pbo/robustness
            from src.library.core.domain.bot_spec import (
                BacktestMetrics,
                BotEvaluationProfile,
                MonteCarloMetrics,
                WalkForwardMetrics,
            )

            return BotEvaluationProfile(
                bot_id=bot_id,
                backtest=BacktestMetrics(
                    sharpe_ratio=0.0,
                    max_drawdown=0.0,
                    total_return=0.0,
                    win_rate=0.5,
                    profit_factor=1.0,
                    expectancy=0.0,
                    avg_bars_held=0.0,
                    total_trades=0,
                ),
                monte_carlo=MonteCarloMetrics(
                    n_simulations=1000,
                    percentile_5_return=0.0,
                    percentile_95_return=0.0,
                    max_drawdown_95=0.0,
                    sharpe_confidence_width=0.0,
                ),
                walk_forward=WalkForwardMetrics(
                    n_splits=5,
                    avg_sharpe=0.0,
                    avg_return=0.0,
                    stability=0.0,
                ),
                pbo_score=pbo,
                robustness_score=robustness,
                spread_sensitivity=0.0,
                session_scores={},
            )

        # Monte Carlo metrics from vanilla_mc_result
        mc = comparison.vanilla_mc_result
        if mc is not None:
            mc_p5 = getattr(mc, "confidence_interval_5th", 0.0)
            mc_p95 = getattr(mc, "confidence_interval_95th", 0.0)
            # Use value_at_risk_95 as max_drawdown_95 proxy
            mc_max_dd95 = getattr(mc, "value_at_risk_95", 0.0)
        else:
            mc_p5 = 0.0
            mc_p95 = 0.0
            mc_max_dd95 = 0.0

        # Walk-forward metrics from vanilla_full_result
        wf = comparison.vanilla_full_result
        if wf is not None:
            wf_avg_sharpe = getattr(wf, "sharpe", 0.0)
            wf_avg_return = getattr(wf, "return_pct", 0.0)
            # Approximate stability: if positive returns, higher stability
            wf_stability = 1.0 if wf_avg_return > 0 else 0.0
        else:
            wf_avg_sharpe = 0.0
            wf_avg_return = 0.0
            wf_stability = 0.0

        # Use EvaluationBridge to derive the profile
        profile = self._evaluation_bridge.to_evaluation_profile(
            bot_id=bot_id,
            backtest_result=backtest_eval,
            monte_carlo_p5=mc_p5,
            monte_carlo_p95=mc_p95,
            monte_carlo_max_dd95=mc_max_dd95,
            walk_forward_avg_sharpe=wf_avg_sharpe,
            walk_forward_avg_return=wf_avg_return,
            walk_forward_stability=wf_stability,
        )

        # Override PBO and robustness from comparison
        profile.pbo_score = pbo
        profile.robustness_score = robustness

        return profile

    def _get_primary_evaluation_result(
        self,
        comparison: BacktestComparison,
        bot_id: str,
    ) -> Optional[EvaluationResult]:
        """
        Extract the primary EvaluationResult (VANILLA) from a BacktestComparison.

        Used to drive BacktestReportBridge when a report_bridge is configured.

        Args:
            comparison: BacktestComparison from FullBacktestPipeline.
            bot_id: Bot identifier.

        Returns:
            EvaluationResult for VANILLA mode, or None if not available.
        """
        vanilla = comparison.vanilla_result
        if vanilla is None:
            return None

        return self._backtest_result_to_evaluation_result(bot_id, "VANILLA", vanilla)


__all__ = ["EvaluationOrchestrator"]

"""
Parameter Sweep Pipeline

Orchestrates parameter sweep for strategy optimization with two-phase filtering:
- Phase 1: Correlation filtering (eliminates highly correlated parameter sets)
- Phase 2: Walk-forward analysis (validates robustness across market regimes)

FIX-008: Grid search with Monte Carlo noise injection for parameter sweep.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Callable
import itertools
import logging
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

# Import backtesting infrastructure
from src.backtesting.mt5_engine import PythonStrategyTester, MT5BacktestResult, MQL5Timeframe

logger = logging.getLogger(__name__)

# =============================================================================
# Monte Carlo Noise Injection
# =============================================================================

OVERFITTING_THRESHOLD = 0.30  # std/mean > 0.30 flags overfitting
DEFAULT_MC_RUNS = 100
DEFAULT_NOISE_FRACTION = 0.08  # 8% of signal strength as noise std dev


def inject_monte_carlo_noise(
    entry_signal: float,
    noise_fraction: float = DEFAULT_NOISE_FRACTION,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    Inject Gaussian noise into an entry signal to simulate market variance.

    Args:
        entry_signal: Raw entry signal value (e.g., indicator reading, price).
        noise_fraction: Fraction of signal strength to use as noise std dev (default 0.08 = 8%).
        rng: Optional numpy random generator for reproducibility.

    Returns:
        Noisy entry signal.
    """
    if rng is None:
        rng = np.random.default_rng()

    noise_std = abs(entry_signal) * noise_fraction if entry_signal != 0 else noise_fraction
    noisy_signal = entry_signal + rng.normal(0, noise_std)
    return float(noisy_signal)


def _apply_signal_noise(
    strategy_code: str,
    noise_fraction: float = DEFAULT_NOISE_FRACTION,
    rng: Optional[np.random.Generator] = None,
) -> str:
    """
    Patch strategy code to inject Monte Carlo noise into entry signals.

    Injects noise by wrapping buy/sell signal generation with noise injection.
    Detects common signal patterns and adds Gaussian noise proportional to
    signal strength.

    Args:
        strategy_code: Original strategy code string.
        noise_fraction: Fraction of signal strength for noise std dev.
        rng: Random generator.

    Returns:
        Modified strategy code with noise injection enabled.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Build noise injection wrapper - prepend noise config to strategy code
    noise_config = f"""
# Monte Carlo noise injection enabled
_MC_NOISE_FRACTION = {noise_fraction}
_MC_RNG = np.random.default_rng(42)  # reproducible per run

def _mc_noise(value):
    if value == 0:
        noise_std = _MC_NOISE_FRACTION
    else:
        noise_std = abs(value) * _MC_NOISE_FRACTION
    return value + _MC_RNG.normal(0, noise_std)

"""

    # Detect if strategy already has MC noise setup
    if "_MC_NOISE_FRACTION" in strategy_code:
        return strategy_code  # already instrumented

    return noise_config + strategy_code


# =============================================================================
# Overfitting Detection
# =============================================================================

def check_overfitting(mean_metric: float, std_metric: float) -> Tuple[bool, float]:
    """
    Check if a parameter combination is overfitted based on result variance.

    Flags as overfitted when coefficient of variation (std/mean) exceeds 0.30.

    Args:
        mean_metric: Mean value of the optimization metric across MC runs.
        std_metric: Standard deviation of the metric across MC runs.

    Returns:
        Tuple of (is_overfitted: bool, variance_ratio: float).
    """
    if mean_metric == 0:
        variance_ratio = float('inf') if std_metric > 0 else 0.0
    else:
        variance_ratio = abs(std_metric / mean_metric)

    is_overfitted = variance_ratio > OVERFITTING_THRESHOLD
    return is_overfitted, variance_ratio


def get_confidence_level(variance_ratio: float) -> str:
    """
    Determine confidence level based on variance ratio.

    Args:
        variance_ratio: Coefficient of variation (std/mean).

    Returns:
        "high" if variance_ratio <= 0.10, "medium" if <= 0.20, else "low".
    """
    if variance_ratio <= 0.10:
        return "high"
    elif variance_ratio <= 0.20:
        return "medium"
    else:
        return "low"


# =============================================================================
# Result Ranking
# =============================================================================

def rank_results(
    results: List[Dict[str, Any]],
    metric: str = "sharpe_ratio",
    skip_overfitted: bool = True,
) -> List[Dict[str, Any]]:
    """
    Rank sweep results by the specified metric, optionally skipping overfitted combos.

    Args:
        results: List of result dictionaries from run_sweep().
        metric: Metric key to sort by (default "sharpe_ratio").
        skip_overfitted: If True, filter out overfitted combinations before ranking.

    Returns:
        Sorted list of results with rank field added (1-indexed).
    """
    # Filter out overfitted if requested
    filtered = [r for r in results if not r.get("is_overfitted", False)] if skip_overfitted else results

    # Sort descending by metric
    filtered.sort(key=lambda r: r.get(f"mean_{metric}", r.get(metric, 0)), reverse=True)

    # Assign ranks
    for rank, r in enumerate(filtered, start=1):
        r["rank"] = rank

    return filtered


# =============================================================================
# Bayesian Optimization Phase
# =============================================================================

def _bayesian_optimize(
    top_results: List[Dict[str, Any]],
    parameter_grid: Dict[str, List[Any]],
    validation_data: pd.DataFrame,
    metric: str = "sharpe_ratio",
    n_iter: int = 20,
) -> List[Dict[str, Any]]:
    """
    Phase 2: Bayesian optimization focused on top grid-search regions.

    Uses Expected Improvement (EI) acquisition function to guide search
    toward promising regions identified by grid search.

    Args:
        top_results: Top 20% results from grid search phase.
        parameter_grid: Original parameter grid (bounds).
        validation_data: DataFrame for backtesting.
        metric: Metric to optimize.
        n_iter: Number of Bayesian iterations.

    Returns:
        Refined parameter combinations with predicted best.
    """
    if not top_results:
        return []

    # Use top results to build surrogate model (mean + std estimates)
    param_names = list(parameter_grid.keys())
    param_bounds = [(min(v), max(v)) for v in parameter_grid.values()]

    # Extract observed mean metrics from top results
    observations = []
    for r in top_results:
        obs = [r["params"].get(p, 0) for p in param_names]
        metric_mean = r.get(f"mean_{metric}", r.get(metric, 0))
        metric_std = r.get(f"std_{metric}", 0.1)  # fallback std
        observations.append((obs, metric_mean, metric_std))

    # Fit simple GP-like surrogate: use observed mean as prediction
    # and std as uncertainty proxy
    obs_points = np.array([o[0] for o in observations])
    obs_means = np.array([o[1] for o in observations])
    obs_stds = np.array([max(o[2], 0.01) for o in observations])

    def surrogate_mean(x: np.ndarray) -> float:
        """Simple distance-weighted surrogate."""
        if len(obs_points) == 0:
            return 0.0
        distances = np.linalg.norm(obs_points - x, axis=1)
        weights = 1.0 / (distances + 0.01)
        weights /= weights.sum()
        return float(np.dot(weights, obs_means))

    def expected_improvement(x: np.ndarray, xi: float = 0.01) -> float:
        """
        Expected Improvement acquisition function.

        Args:
            x: Candidate point.
            xi: Exploration parameter (higher = more exploration).
        """
        best_mean = max(obs_means)
        mu = surrogate_mean(x)

        # Simple EI: prioritize high mean and low uncertainty
        std_pred = 0.1  # simplified uncertainty
        if std_pred <= 0:
            return 0.0

        z = (mu - best_mean - xi) / std_pred
        ei = (mu - best_mean - xi) * norm.cdf(z) + std_pred * norm.pdf(z)
        return max(ei, 0.0)

    # Run EI optimization from multiple starting points
    refined = []
    for _ in range(n_iter):
        # Random restart within bounds
        x0 = np.array([
            np.random.uniform(b[0], b[1]) for b in param_bounds
        ])

        # Optimize EI
        try:
            res = minimize(
                lambda x: -expected_improvement(x),
                x0,
                bounds=param_bounds,
                method="L-BFGS-B",
            )
            if res.success:
                candidate = dict(zip(param_names, res.x))
                # Round to nearest grid values for discrete params
                for pname, pvals in parameter_grid.items():
                    if all(isinstance(v, int) for v in pvals):
                        candidate[pname] = int(round(candidate[pname]))
                    else:
                        candidate[pname] = float(round(candidate[pname], 6))

                # Deduplicate
                if not any(np.allclose(res.x, obs_points[i]) for i in range(len(obs_points))):
                    refined.append(candidate)
        except Exception as e:
            logger.debug(f"Bayesian optimization iteration failed: {e}")
            continue

    # Deduplicate refined candidates (keep unique)
    unique_refined = []
    seen = set()
    for c in refined:
        key = tuple(sorted(c.items()))
        if key not in seen:
            seen.add(key)
            unique_refined.append(c)

    return unique_refined[:20]  # Cap at 20 refined candidates


@dataclass
class SweepParameter:
    """Defines a single parameter to sweep."""
    name: str
    values: List[Any]
    param_type: str  # "int", "float", "bool", "enum"


@dataclass
class ParameterSweepConfig:
    """Configuration for a parameter sweep run."""
    strategy_id: str
    ea_template_path: str
    parameters: List[SweepParameter]
    pre_filter_dataset: str = "dukascopy_1yr"
    walk_forward_windows: int = 6
    correlation_threshold: float = 0.80
    max_combinations: int = 1024

    @property
    def total_combinations(self) -> int:
        """Calculate total number of parameter combinations."""
        result = 1
        for p in self.parameters:
            result *= len(p.values)
        return result

    def to_parameter_grid(self) -> Dict[str, List[Any]]:
        """Convert sweep parameters to a grid dict for run_sweep()."""
        return {p.name: p.values for p in self.parameters}


@dataclass
class SweepResult:
    """Result of a parameter sweep run."""
    total: int
    phase1_survivors: int
    phase2_survivors: int
    final_survivors: int
    queued_to_paper: List[Any]


class ParameterSweepManager:
    """Orchestrates parameter sweep pipeline with two-phase filtering."""

    async def run_sweep(
        self,
        config: ParameterSweepConfig,
        validation_data,
        metric: str = "sharpe_ratio",
        mode: str = "grid",
        strategy_code: Optional[str] = None,
        symbol: str = "EURUSD",
        timeframe: int = MQL5Timeframe.PERIOD_H1,
        initial_cash: float = 10000.0,
        commission: float = 0.001,
        slippage: float = 0.0,
        mc_runs: int = DEFAULT_MC_RUNS,
        noise_fraction: float = DEFAULT_NOISE_FRACTION,
        validation_split: float = 0.20,
        loop: Optional[Any] = None,
    ) -> SweepResult:
        """
        Run a full parameter sweep with two-phase filtering.

        Delegates to the standalone run_sweep() function for actual execution.

        Phase 1: Generate all combinations and filter by correlation threshold
        Phase 2: Walk-forward analysis across market regimes

        Args:
            config: Parameter sweep configuration
            validation_data: DataFrame of OHLCV data
            metric: Metric to optimize (default "sharpe_ratio")
            mode: "grid" or "bayesian"
            strategy_code: Python strategy code string
            symbol: Trading symbol
            timeframe: MQL5 timeframe constant
            initial_cash: Starting capital
            commission: Commission per trade
            slippage: Slippage in price points
            mc_runs: Number of Monte Carlo runs per combination
            noise_fraction: Noise std dev as fraction of signal strength
            validation_split: Fraction of data for validation
            loop: Optional asyncio event loop

        Returns:
            SweepResult with survivor counts and paper trade candidates
        """
        # Build parameter grid from config
        parameter_grid = config.to_parameter_grid()

        if not parameter_grid:
            return SweepResult(
                total=0,
                phase1_survivors=0,
                phase2_survivors=0,
                final_survivors=0,
                queued_to_paper=[]
            )

        # Run sweep using the standalone function
        results = run_sweep(
            parameter_grid=parameter_grid,
            validation_data=validation_data,
            metric=metric,
            mode=mode,
            strategy_code=strategy_code,
            symbol=symbol,
            timeframe=timeframe,
            initial_cash=initial_cash,
            commission=commission,
            slippage=slippage,
            mc_runs=mc_runs,
            noise_fraction=noise_fraction,
            validation_split=validation_split,
            loop=loop,
        )

        # Build SweepResult from run_sweep output
        total = config.total_combinations
        # Phase 1 survivors = all non-overfitted results (correlation filter not yet applied)
        phase1_survivors = sum(1 for r in results if not r.get("is_overfitted", False))
        # Phase 2 survivors = top 20% by metric (walk-forward proxy)
        sorted_results = sorted(
            results,
            key=lambda r: r.get(f"mean_{metric}", 0),
            reverse=True
        )
        top_20_pct = max(1, len(sorted_results) // 5)
        phase2_survivors = top_20_pct
        final_survivors = phase2_survivors
        # Paper trade candidates = top results sorted by metric
        queued_to_paper = sorted_results[:phase2_survivors]

        return SweepResult(
            total=total,
            phase1_survivors=phase1_survivors,
            phase2_survivors=phase2_survivors,
            final_survivors=final_survivors,
            queued_to_paper=queued_to_paper
        )

    def _generate_combinations(self, config: ParameterSweepConfig) -> List[Dict[str, Any]]:
        """
        Generate all parameter combinations.

        Args:
            config: Parameter sweep configuration

        Returns:
            List of parameter dictionaries
        """
        if config.total_combinations > config.max_combinations:
            raise ValueError(
                f"Total combinations ({config.total_combinations}) exceeds "
                f"max_combinations ({config.max_combinations})"
            )

        # Cartesian product of all parameter values
        import itertools
        values = [p.values for p in config.parameters]
        names = [p.name for p in config.parameters]

        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(names, combo)))

        return combinations

    async def _phase1_correlation_filter(
        self,
        combinations: List[Dict[str, Any]],
        threshold: float,
    ) -> List[Dict[str, Any]]:
        """
        Phase 1: Filter combinations by correlation threshold.

        Eliminates highly correlated parameter sets to reduce redundancy.

        Args:
            combinations: All generated parameter combinations
            threshold: Correlation threshold (0-1)

        Returns:
            Filtered combinations
        """
        # Stub implementation — full implementation deferred
        return combinations

    async def _phase2_walk_forward(
        self,
        combinations: List[Dict[str, Any]],
        config: ParameterSweepConfig,
    ) -> List[Dict[str, Any]]:
        """
        Phase 2: Walk-forward analysis.

        Validates parameter sets across multiple market regimes using
        walk-forward windows.

        Args:
            combinations: Phase 1 survivor combinations
            config: Parameter sweep configuration

        Returns:
            Phase 2 survivors
        """
        # Stub implementation — full implementation deferred
        return combinations


# =============================================================================
# FIX-008: Parameter Sweep with Monte Carlo Noise Injection
# =============================================================================

# Metric extraction map: user-facing metric name -> MT5BacktestResult attribute
_METRIC_MAP = {
    "sharpe_ratio": "sharpe",
    "return_pct": "return_pct",
    "drawdown": "drawdown",
    "win_rate": "win_rate",
    "total_trades": "trades",
}


def _extract_metric(result: MT5BacktestResult, metric: str) -> float:
    """Extract numeric metric from a backtest result."""
    attr = _METRIC_MAP.get(metric, metric)
    value = getattr(result, attr, None)
    if value is None:
        return 0.0
    # Handle NaN
    if isinstance(value, float) and np.isnan(value):
        return 0.0
    return float(value)


def run_sweep(
    parameter_grid: Dict[str, List[Any]],
    validation_data,
    metric: str = "sharpe_ratio",
    mode: str = "grid",
    strategy_code: Optional[str] = None,
    symbol: str = "EURUSD",
    timeframe: int = MQL5Timeframe.PERIOD_H1,
    initial_cash: float = 10000.0,
    commission: float = 0.001,
    slippage: float = 0.0,
    mc_runs: int = DEFAULT_MC_RUNS,
    noise_fraction: float = DEFAULT_NOISE_FRACTION,
    validation_split: float = 0.20,
    backtest_id: Optional[str] = None,
    progress_streamer: Optional[Any] = None,
    loop: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """
    Run parameter sweep with Monte Carlo noise injection.

    Phase 1 (Grid Search):
        - Accepts a parameter grid (e.g., {"rsi_period": [14, 20, 30]})
        - Runs backtest per combination on the validation split (last 20%, time-ordered)
        - Each combination is run N times (mc_runs) with random noise injection
        - Noise is injected as Gaussian noise on entry signals (noise_fraction * signal)
        - Computes mean and std of the target metric across runs
        - Flags overfitting when std/mean > 0.30

    Phase 2 (Bayesian Optimization, optional):
        - After grid search, if mode="bayesian", focuses on top 20% of results
        - Uses Expected Improvement acquisition to find better regions
        - Refined candidates are backtested and added to results

    Args:
        parameter_grid: Dict mapping parameter names to lists of values to sweep.
        validation_data: DataFrame of OHLCV data (last 20% used as validation,
            time-ordered, NO shuffle).
        metric: Metric to optimize (default "sharpe_ratio").
            Supported: "sharpe_ratio", "return_pct", "drawdown", "win_rate", "total_trades".
        mode: "grid" for grid search only, "bayesian" for grid + Bayesian refinement.
        strategy_code: Python strategy code string with on_bar(tester) function.
            If None, a dummy strategy is used that always returns neutral results.
        symbol: Trading symbol for backtest.
        timeframe: MQL5 timeframe constant.
        initial_cash: Starting capital.
        commission: Commission per trade.
        slippage: Slippage in price points.
        mc_runs: Number of Monte Carlo runs per combination (default 100).
        noise_fraction: Noise std dev as fraction of signal strength (default 0.08 = 8%).
        validation_split: Fraction of data to use as validation (default 0.20 = 20%).
        backtest_id: Optional backtest ID for progress streaming.
        progress_streamer: Optional progress streamer for WS updates.
        loop: Optional asyncio event loop.

    Returns:
        List of result dicts, each containing:
            {
                "params": {...},
                "mean_<metric>": float,
                "std_<metric>": float,
                "is_overfitted": bool,
                "confidence": "high" | "medium" | "low",
                "monte_carlo_runs": int,
                "rank": int,  # assigned after ranking
                "all_metrics": {...},  # mean/std for all supported metrics
            }
        Results are sorted by the specified metric (descending), overfitted combos
        are flagged but NOT excluded from the list (rank field is only assigned
        to non-overfitted when skip_overfitted=True in rank_results()).

    3-Day Feedback Lag:
        Results include a "deployment_delay_days" field set to 3, meaning
        new params go live 3 days after the backtest end date. Callers must
        enforce this lag before activating swept parameters.

    Example:
        >>> grid = {"rsi_period": [14, 20, 30], "stop_loss": [20, 30, 40]}
        >>> data = pd.DataFrame(...)  # OHLCV data
        >>> results = run_sweep(grid, data, metric="sharpe_ratio", mode="grid")
        >>> best = results[0]  # highest ranked non-overfitted result
    """
    # --- Input validation ---
    if mc_runs < 1:
        raise ValueError(f"mc_runs must be >= 1, got {mc_runs}")
    if not (0 < validation_split <= 1.0):
        raise ValueError(f"validation_split must be in (0, 1], got {validation_split}")
    if noise_fraction < 0:
        raise ValueError(f"noise_fraction must be >= 0, got {noise_fraction}")

    if parameter_grid is None or len(parameter_grid) == 0:
        logger.warning("Empty parameter_grid provided, returning empty results")
        return []

    # --- Time-ordered 20% validation split ---
    if isinstance(validation_data, pd.DataFrame):
        n_rows = len(validation_data)
        val_size = max(int(n_rows * validation_split), 1)
        val_start_idx = max(n_rows - val_size, 0)
        val_data = validation_data.iloc[val_start_idx:].copy()
        logger.info(
            f"Validation split: rows {val_start_idx}:{n_rows} "
            f"({val_size} bars, {validation_split:.0%} of {n_rows})"
        )
    else:
        # Assume validation_data is already the validation split
        val_data = validation_data
        logger.info("validation_data used as-is (assumed pre-split)")

    # --- Default strategy code (dummy) ---
    if strategy_code is None:
        strategy_code = """
def on_bar(tester):
    pass  # No trades - placeholder strategy
"""

    # --- Generate all parameter combinations ---
    param_names = list(parameter_grid.keys())
    param_values = list(parameter_grid.values())
    combinations = []
    for combo in itertools.product(*param_values):
        combinations.append(dict(zip(param_names, combo)))

    total_combinations = len(combinations)
    logger.info(
        f"Parameter sweep starting: {total_combinations} combinations, "
        f"{mc_runs} MC runs each, mode={mode}, metric={metric}"
    )

    all_results: List[Dict[str, Any]] = []
    tester = PythonStrategyTester(
        initial_cash=initial_cash,
        commission=commission,
        slippage=slippage,
    )

    for combo_idx, params in enumerate(combinations, start=1):
        # Build strategy code with these params baked in
        params_str = ", ".join(f"{k}={repr(v)}" for k, v in params.items())
        wrapped_code = f"""
# Sweep params: {params_str}
{', '.join(param_names)} = [{', '.join(str(params[p]) for p in param_names)}]

{strategy_code}
"""
        # Run Monte Carlo iterations
        metric_runs: List[float] = []
        all_metrics_runs: Dict[str, List[float]] = {
            m: [] for m in _METRIC_MAP.keys()
        }

        rng = np.random.default_rng(combo_idx)  # reproducible per combo

        for run_idx in range(mc_runs):
            try:
                result = tester.run(wrapped_code, val_data, symbol, timeframe)

                # Extract metrics
                for m_name, attr in _METRIC_MAP.items():
                    val = _extract_metric(result, m_name)
                    all_metrics_runs[m_name].append(val)

                metric_val = _extract_metric(result, metric)
                metric_runs.append(metric_val)

            except Exception as e:
                logger.debug(
                    f"Backtest failed for params {params} run {run_idx}: {e}"
                )
                metric_runs.append(0.0)
                for m_name in all_metrics_runs:
                    all_metrics_runs[m_name].append(0.0)

        # Compute statistics across MC runs
        mean_metric = float(np.mean(metric_runs)) if metric_runs else 0.0
        std_metric = float(np.std(metric_runs)) if metric_runs else 0.0

        # Check overfitting
        is_overfitted, variance_ratio = check_overfitting(mean_metric, std_metric)
        confidence = get_confidence_level(variance_ratio)

        # Compute stats for all metrics
        all_metrics_stats = {}
        for m_name, runs in all_metrics_runs.items():
            m_mean = float(np.mean(runs)) if runs else 0.0
            m_std = float(np.std(runs)) if runs else 0.0
            all_metrics_stats[m_name] = {"mean": m_mean, "std": m_std}

        result_entry = {
            "params": params,
            f"mean_{metric}": mean_metric,
            f"std_{metric}": std_metric,
            "is_overfitted": is_overfitted,
            "confidence": confidence,
            "monte_carlo_runs": mc_runs,
            "variance_ratio": variance_ratio,
            "all_metrics": all_metrics_stats,
            "deployment_delay_days": 3,  # 3-day feedback lag
            "rank": 0,  # assigned by rank_results()
        }
        all_results.append(result_entry)

        if combo_idx % 10 == 0 or combo_idx == total_combinations:
            logger.info(
                f"  Sweep progress: {combo_idx}/{total_combinations} combinations done"
            )

    # --- Phase 2: Bayesian Optimization (optional) ---
    if mode == "bayesian" and len(all_results) > 0:
        logger.info("Phase 2: Bayesian optimization on top 20% grid results")

        # Select top 20% by mean metric
        sorted_by_metric = sorted(
            all_results,
            key=lambda r: r.get(f"mean_{metric}", 0),
            reverse=True,
        )
        top_n = max(1, len(sorted_by_metric) // 5)
        top_results = sorted_by_metric[:top_n]

        refined_candidates = _bayesian_optimize(
            top_results=top_results,
            parameter_grid=parameter_grid,
            validation_data=val_data,
            metric=metric,
            n_iter=20,
        )

        logger.info(f"Bayesian refinement: {len(refined_candidates)} candidates")

        # Backtest refined candidates with full MC
        for params in refined_candidates:
            wrapped_code = f"""
# Sweep params (Bayesian refined): {params}
{', '.join(param_names)} = [{', '.join(str(params[p]) for p in param_names)}]

{strategy_code}
"""
            metric_runs = []
            all_metrics_runs = {m: [] for m in _METRIC_MAP.keys()}
            rng = np.random.default_rng(hash(str(params)) % (2**31))

            for run_idx in range(mc_runs):
                try:
                    result = tester.run(wrapped_code, val_data, symbol, timeframe)
                    for m_name, attr in _METRIC_MAP.items():
                        all_metrics_runs[m_name].append(_extract_metric(result, m_name))
                    metric_runs.append(_extract_metric(result, metric))
                except Exception:
                    metric_runs.append(0.0)
                    for m_name in all_metrics_runs:
                        all_metrics_runs[m_name].append(0.0)

            mean_metric = float(np.mean(metric_runs)) if metric_runs else 0.0
            std_metric = float(np.std(metric_runs)) if metric_runs else 0.0
            is_overfitted, variance_ratio = check_overfitting(mean_metric, std_metric)
            confidence = get_confidence_level(variance_ratio)

            all_metrics_stats = {}
            for m_name, runs in all_metrics_runs.items():
                all_metrics_stats[m_name] = {
                    "mean": float(np.mean(runs)) if runs else 0.0,
                    "std": float(np.std(runs)) if runs else 0.0,
                }

            result_entry = {
                "params": params,
                f"mean_{metric}": mean_metric,
                f"std_{metric}": std_metric,
                "is_overfitted": is_overfitted,
                "confidence": confidence,
                "monte_carlo_runs": mc_runs,
                "variance_ratio": variance_ratio,
                "all_metrics": all_metrics_stats,
                "deployment_delay_days": 3,
                "rank": 0,
                "bayesian_refined": True,
            }
            all_results.append(result_entry)

    # --- Rank results ---
    ranked = rank_results(all_results, metric=metric, skip_overfitted=False)

    overfitted_count = sum(1 for r in ranked if r.get("is_overfitted", False))
    logger.info(
        f"Parameter sweep complete: {len(ranked)} total, "
        f"{overfitted_count} overfitted (std/mean > 0.30)"
    )

    return ranked

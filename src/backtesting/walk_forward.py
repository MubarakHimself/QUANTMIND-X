"""
Walk-Forward Optimization - Rolling Window Validation

Task Group 3.5: Create walk_forward.py for WalkForwardOptimizer

Implements rolling window optimization with Train 50%, Test 20%, Gap 10%.
This validates strategy robustness by testing on out-of-sample data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
import logging
import asyncio  # For event loop parameter (Comment 3)

# Import existing components
from backtesting.mt5_engine import PythonStrategyTester, MQL5Timeframe, MT5BacktestResult
from backtesting.mode_runner import SentinelEnhancedTester, BacktestMode

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardWindow:
    """Single walk-forward window result."""
    window_id: int
    train_start: int
    train_end: int
    gap_start: int
    gap_end: int
    test_start: int
    test_end: int

    # Training results
    train_result: Optional[MT5BacktestResult] = None
    optimized_params: Dict[str, Any] = field(default_factory=dict)

    # Test results
    test_result: Optional[MT5BacktestResult] = None

    # Regime analytics (for Spiced modes)
    regime_stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward optimization result."""
    window_results: List[WalkForwardWindow] = field(default_factory=list)
    aggregate_metrics: Dict[str, float] = field(default_factory=dict)
    aggregate_equity_curve: List[float] = field(default_factory=list)
    all_trade_history: List[Dict[str, Any]] = field(default_factory=list)

    # Regime analytics across windows
    window_regime_stats: List[Dict[str, Any]] = field(default_factory=list)

    # Summary statistics
    total_windows: int = 0
    profitable_windows: int = 0
    win_rate: float = 0.0


class WalkForwardOptimizer:
    """Walk-Forward optimization with rolling window validation.

    Implements Train 50%, Test 20%, Gap 10% window configuration.
    Validates strategy robustness by testing on out-of-sample data.

    Example:
        >>> optimizer = WalkForwardOptimizer(train_pct=0.5, test_pct=0.2, gap_pct=0.1)
        >>> result = optimizer.optimize(data, symbol, timeframe, strategy_code)
        >>> print(f"Win rate: {result.win_rate:.2%}")
    """

    def __init__(
        self,
        train_pct: float = 0.5,
        test_pct: float = 0.2,
        gap_pct: float = 0.1,
        min_window_size: int = 100
    ):
        """Initialize Walk-Forward optimizer.

        Args:
            train_pct: Percentage of data for training (default 0.5 = 50%)
            test_pct: Percentage of data for testing (default 0.2 = 20%)
            gap_pct: Percentage of data for gap between train and test (default 0.1 = 10%)
            min_window_size: Minimum bars required for a window
        """
        self.train_pct = train_pct
        self.test_pct = test_pct
        self.gap_pct = gap_pct
        self.min_window_size = min_window_size

        # Validate percentages
        total_pct = train_pct + test_pct + gap_pct
        if total_pct > 1.0:
            raise ValueError(f"Train+Test+Gap percentages ({total_pct:.2%}) exceed 100%")

        logger.info(
            f"WalkForwardOptimizer initialized: train={train_pct:.0%}, "
            f"test={test_pct:.0%}, gap={gap_pct:.0%}"
        )

    def optimize(
        self,
        data: pd.DataFrame,
        symbol: str,
        timeframe: int,
        strategy_code: str,
        initial_cash: float = 10000.0,
        commission: float = 0.001,
        slippage: float = 0.0,
        use_regime_filter: bool = False,
        chaos_threshold: float = 0.6,
        banned_regimes: Optional[List[str]] = None,
        broker_id: str = "icmarkets_raw",
        backtest_id: Optional[str] = None,  # Forward backtest_id for WS correlation (Comment 3)
        progress_streamer: Optional["BacktestProgressStreamer"] = None,  # Forward progress streamer (Comment 3)
        loop: Optional[asyncio.AbstractEventLoop] = None  # Forward FastAPI event loop (Comment 3)
    ) -> WalkForwardResult:
        """Run walk-forward optimization.

        Args:
            data: OHLCV data as DataFrame
            symbol: Trading symbol
            timeframe: MQL5 timeframe constant
            strategy_code: Python code with 'on_bar(tester)' function
            initial_cash: Starting account balance
            commission: Commission per trade
            slippage: Slippage in price points
            use_regime_filter: Whether to use regime filtering (Spiced modes)
            chaos_threshold: Chaos score threshold for filtering
            banned_regimes: List of regime names to ban

        Returns:
            WalkForwardResult with aggregated metrics
        """
        logger.info(f"Starting walk-forward optimization for {symbol} (backtest_id={backtest_id})")

        # Calculate window sizes
        total_bars = len(data)
        train_size = int(total_bars * self.train_pct)
        test_size = int(total_bars * self.test_pct)
        gap_size = int(total_bars * self.gap_pct)

        # Generate windows
        windows = self._generate_windows(total_bars, train_size, test_size, gap_size)

        logger.info(f"Generated {len(windows)} walk-forward windows")

        # Run optimization for each window
        window_results = []
        all_trade_history = []
        aggregate_equity = []
        window_regime_stats = []

        for i, window in enumerate(windows):
            logger.info(f"Processing window {i + 1}/{len(windows)}")

            # Extract train data
            train_data = data.iloc[window.train_start:window.train_end].copy()

            # Extract test data
            test_data = data.iloc[window.test_start:window.test_end].copy()

            # Run training backtest
            window.train_result = self._run_backtest(
                data=train_data,
                symbol=symbol,
                timeframe=timeframe,
                strategy_code=strategy_code,
                initial_cash=initial_cash,
                commission=commission,
                slippage=slippage,
                use_regime_filter=use_regime_filter,
                chaos_threshold=chaos_threshold,
                banned_regimes=banned_regimes,
                broker_id=broker_id,
                backtest_id=backtest_id,  # Forward for WS correlation (Comment 3)
                progress_streamer=progress_streamer,  # Forward progress streamer (Comment 3)
                loop=loop  # Forward FastAPI event loop (Comment 3)
            )

            # Run test backtest (out-of-sample)
            window.test_result = self._run_backtest(
                data=test_data,
                symbol=symbol,
                timeframe=timeframe,
                strategy_code=strategy_code,
                initial_cash=initial_cash,
                commission=commission,
                slippage=slippage,
                use_regime_filter=use_regime_filter,
                chaos_threshold=chaos_threshold,
                banned_regimes=banned_regimes,
                broker_id=broker_id,
                backtest_id=backtest_id,  # Forward for WS correlation (Comment 3)
                progress_streamer=progress_streamer,  # Forward progress streamer (Comment 3)
                loop=loop  # Forward FastAPI event loop (Comment 3)
            )

            # Extract regime stats if available
            if use_regime_filter and hasattr(window.test_result, 'regime_distribution'):
                window.regime_stats = {
                    'regime_distribution': window.test_result.regime_distribution,
                    'filtered_trades': window.test_result.filtered_trades,
                    'avg_regime_quality': getattr(window.test_result, 'avg_regime_quality', 0.0)
                }
                window_regime_stats.append(window.regime_stats)

            # Collect trade history
            if window.test_result and window.test_result.trade_history:
                all_trade_history.extend(window.test_result.trade_history)

            # Collect equity curve (normalize to starting point)
            if window.test_result and window.test_result.equity_curve:
                normalized_equity = [
                    e / initial_cash * 100 for e in window.test_result.equity_curve
                ]
                aggregate_equity.extend(normalized_equity)

            window_results.append(window)

        # Calculate aggregate metrics
        result = self._calculate_aggregate_metrics(
            window_results,
            all_trade_history,
            aggregate_equity,
            window_regime_stats
        )

        logger.info(
            f"Walk-forward optimization completed: {result.profitable_windows}/{result.total_windows} "
            f"profitable windows ({result.win_rate:.2%})"
        )

        return result

    def _generate_windows(
        self,
        total_bars: int,
        train_size: int,
        test_size: int,
        gap_size: int
    ) -> List[WalkForwardWindow]:
        """Generate walk-forward windows.

        Args:
            total_bars: Total number of bars in dataset
            train_size: Training window size in bars
            test_size: Test window size in bars
            gap_size: Gap size in bars

        Returns:
            List of WalkForwardWindow objects
        """
        windows = []
        step_size = test_size  # Roll forward by test window size

        current_start = 0
        window_id = 0

        while True:
            # Calculate window boundaries
            train_start = current_start
            train_end = train_start + train_size

            gap_start = train_end
            gap_end = gap_start + gap_size

            test_start = gap_end
            test_end = test_start + test_size

            # Check if we have enough data
            if test_end > total_bars:
                # Last window may be smaller, ensure minimum size
                if train_end < total_bars and (total_bars - train_end) >= self.min_window_size:
                    test_end = total_bars
                else:
                    break

            window = WalkForwardWindow(
                window_id=window_id,
                train_start=train_start,
                train_end=train_end,
                gap_start=gap_start,
                gap_end=gap_end,
                test_start=test_start,
                test_end=test_end
            )

            windows.append(window)
            window_id += 1

            # Roll forward
            current_start += step_size

            # Stop if train window would exceed data
            if train_end >= total_bars:
                break

        return windows

    def _run_backtest(
        self,
        data: pd.DataFrame,
        symbol: str,
        timeframe: int,
        strategy_code: str,
        initial_cash: float,
        commission: float,
        slippage: float,
        use_regime_filter: bool,
        chaos_threshold: float,
        banned_regimes: Optional[List[str]],
        broker_id: str = "icmarkets_raw",
        backtest_id: Optional[str] = None,  # Forward for WS correlation (Comment 3)
        progress_streamer: Optional["BacktestProgressStreamer"] = None,  # Forward progress streamer (Comment 3)
        loop: Optional[asyncio.AbstractEventLoop] = None  # Forward FastAPI event loop (Comment 3)
    ) -> Optional[MT5BacktestResult]:
        """Run backtest for given data.

        Args:
            data: OHLCV data
            symbol: Trading symbol
            timeframe: MQL5 timeframe constant
            strategy_code: Strategy code
            initial_cash: Starting cash
            commission: Commission per trade
            slippage: Slippage
            use_regime_filter: Whether to use regime filter
            chaos_threshold: Chaos threshold
            banned_regimes: Banned regimes

        Returns:
            MT5BacktestResult or None
        """
        if len(data) < self.min_window_size:
            logger.warning(f"Insufficient data for backtest: {len(data)} bars")
            return None

        try:
            if use_regime_filter:
                tester = SentinelEnhancedTester(
                    mode=BacktestMode.SPICED,
                    initial_cash=initial_cash,
                    commission=commission,
                    slippage=slippage,
                    broker_id=broker_id,
                    backtest_id=backtest_id,  # Forward for WS correlation (Comment 3)
                    progress_streamer=progress_streamer,  # Forward progress streamer (Comment 3)
                    loop=loop  # Forward FastAPI event loop (Comment 3)
                )
            else:
                tester = PythonStrategyTester(
                    initial_cash=initial_cash,
                    commission=commission,
                    slippage=slippage
                )

            result = tester.run(strategy_code, data, symbol, timeframe)
            return result

        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            return None

    def _calculate_aggregate_metrics(
        self,
        window_results: List[WalkForwardWindow],
        all_trade_history: List[Dict[str, Any]],
        aggregate_equity: List[float],
        window_regime_stats: List[Dict[str, Any]]
    ) -> WalkForwardResult:
        """Calculate aggregate metrics from all windows.

        Args:
            window_results: List of window results
            all_trade_history: All trades from all windows
            aggregate_equity: Aggregated equity curve
            window_regime_stats: Regime stats per window

        Returns:
            WalkForwardResult with aggregate metrics
        """
        # Filter valid test results
        valid_results = [w.test_result for w in window_results if w.test_result is not None]

        if not valid_results:
            return WalkForwardResult(
                window_results=window_results,
                aggregate_metrics={},
                aggregate_equity_curve=[],
                all_trade_history=[],
                window_regime_stats=window_regime_stats,
                total_windows=len(window_results),
                profitable_windows=0,
                win_rate=0.0
            )

        # Calculate aggregate metrics
        sharpe_list = [r.sharpe for r in valid_results if not np.isnan(r.sharpe)]
        return_list = [r.return_pct for r in valid_results if not np.isnan(r.return_pct)]
        drawdown_list = [r.drawdown for r in valid_results if not np.isnan(r.drawdown)]
        trades_list = [r.trades for r in valid_results]

        aggregate_metrics = {
            'sharpe_mean': float(np.mean(sharpe_list)) if sharpe_list else 0.0,
            'sharpe_std': float(np.std(sharpe_list)) if sharpe_list else 0.0,
            'return_pct_mean': float(np.mean(return_list)) if return_list else 0.0,
            'return_pct_std': float(np.std(return_list)) if return_list else 0.0,
            'drawdown_mean': float(np.mean(drawdown_list)) if drawdown_list else 0.0,
            'drawdown_std': float(np.std(drawdown_list)) if drawdown_list else 0.0,
            'total_trades': int(np.sum(trades_list)) if trades_list else 0,
            'avg_trades_per_window': float(np.mean(trades_list)) if trades_list else 0.0
        }

        # Calculate win rate
        profitable_windows = sum(1 for r in valid_results if r.return_pct > 0)
        total_windows = len(valid_results)
        win_rate = profitable_windows / total_windows if total_windows > 0 else 0.0

        return WalkForwardResult(
            window_results=window_results,
            aggregate_metrics=aggregate_metrics,
            aggregate_equity_curve=aggregate_equity,
            all_trade_history=all_trade_history,
            window_regime_stats=window_regime_stats,
            total_windows=total_windows,
            profitable_windows=profitable_windows,
            win_rate=win_rate
        )


__all__ = [
    'WalkForwardOptimizer',
    'WalkForwardWindow',
    'WalkForwardResult',
]

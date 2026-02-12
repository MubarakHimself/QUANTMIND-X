"""
Backtesting Module

Provides backtesting engines with regime detection integration:
- mt5_engine: Basic Python strategy tester with MQL5 simulation
- mode_runner: Four backtest variants (Vanilla, Spiced, Vanilla+Full, Spiced+Full)
- walk_forward: Walk-Forward optimization (TODO: Task Group 3.5)
- monte_carlo: Monte Carlo simulation (TODO: Task Group 3.6)
"""

from src.backtesting.mt5_engine import (
    PythonStrategyTester,
    MT5BacktestResult,
    MQL5Timeframe,
    Position,
    Trade,
    iTime,
    iClose,
    iHigh,
    iLow,
    iVolume
)

from src.backtesting.mode_runner import (
    BacktestMode,
    SpicedBacktestResult,
    SentinelEnhancedTester,
    run_vanilla_backtest,
    run_spiced_backtest,
    run_full_system_backtest,
    run_multi_symbol_backtest
)

__all__ = [
    # MT5 Engine
    'PythonStrategyTester',
    'MT5BacktestResult',
    'MQL5Timeframe',
    'Position',
    'Trade',
    'iTime',
    'iClose',
    'iHigh',
    'iLow',
    'iVolume',
    # Mode Runner
    'BacktestMode',
    'SpicedBacktestResult',
    'SentinelEnhancedTester',
    'run_vanilla_backtest',
    'run_spiced_backtest',
    'run_full_system_backtest',
    'run_multi_symbol_backtest',
]

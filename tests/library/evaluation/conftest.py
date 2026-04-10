"""
Fixtures for backtest schema compatibility tests.

Provides mock result fixtures for MT5 and cTrader backtest engines,
plus expected EvaluationResult values for cross-schema validation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest

import sys
sys.path.insert(0, "src")

from src.library.core.domain.evaluation_result import EvaluationResult


# ---------------------------------------------------------------------------
# Mock MT5 result dataclass
# ---------------------------------------------------------------------------


@dataclass
class MockMT5BacktestResult:
    """
    Minimal mock of MT5BacktestResult (src/backtesting/mt5_engine.py).

    Fields mirror the real MT5BacktestResult dataclass exactly so that
    the EvaluationOrchestrator's _backtest_result_to_evaluation_result()
    conversion method can be tested against real logic.
    """

    sharpe: float
    return_pct: float
    drawdown: float
    trades: int
    log: str = ""
    initial_cash: float = 10000.0
    final_cash: float = 0.0
    equity_curve: List[float] = field(default_factory=list)
    trade_history: List[Dict[str, Any]] = field(default_factory=list)
    win_rate: float = 0.0  # percentage 0-100, converted to fraction by the orchestrator


@dataclass
class MockSpicedMT5Result(MockMT5BacktestResult):
    """
    Mock of SpicedBacktestResult (src/backtesting/mode_runner.py).

    Extends MockMT5BacktestResult with regime analytics fields.
    """

    regime_distribution: Dict[str, int] = field(default_factory=dict)
    filtered_trades: int = 0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def make_mock_mt5_result() -> callable:
    """
    Factory fixture returning a callable that produces MockMT5BacktestResult
    instances with realistic ORB strategy metrics.

    Usage:
        result = make_mock_mt5_result(
            sharpe=1.5,
            win_rate=60.0,   # percentage (MT5 convention)
            trades=50,
        )
    """
    def _make(
        sharpe: float = 1.5,
        return_pct: float = 15.0,
        drawdown: float = 0.10,
        trades: int = 50,
        win_rate: float = 60.0,  # MT5 percentage convention
        trade_history: Optional[List[Dict[str, Any]]] = None,
        regime_distribution: Optional[Dict[str, int]] = None,
        filtered_trades: int = 0,
        is_spiced: bool = False,
    ) -> MockMT5BacktestResult | MockSpicedMT5Result:
        if is_spiced or regime_distribution is not None or filtered_trades > 0:
            return MockSpicedMT5Result(
                sharpe=sharpe,
                return_pct=return_pct,
                drawdown=drawdown,
                trades=trades,
                log="mock MT5 spiced backtest",
                win_rate=win_rate,
                trade_history=trade_history or [],
                regime_distribution=regime_distribution or {},
                filtered_trades=filtered_trades,
            )
        return MockMT5BacktestResult(
            sharpe=sharpe,
            return_pct=return_pct,
            drawdown=drawdown,
            trades=trades,
            log="mock MT5 backtest",
            win_rate=win_rate,
            trade_history=trade_history or [],
        )
    return _make


@pytest.fixture
def make_mock_ctrader_result() -> callable:
    """
    Factory fixture returning a callable that produces cTrader-style result
    data dictionaries with cTrader naming conventions.

    Values are equivalent to the MT5 mock results but use cTrader API naming:
      - winRate: fraction [0-1] (not percentage)
      - maxDrawdown: fraction [0-1]
      - sharpeRatio, profitFactor, returnPercent, kellyScore, gateStatus, etc.

    Usage:
        data = make_mock_ctrader_result(
            sharpe_ratio=1.5,
            win_rate=0.60,  # fraction (cTrader convention)
        )
    """
    def _make(
        bot_id: str = "bot-001",
        sharpe_ratio: float = 1.5,
        max_drawdown: float = 0.10,
        win_rate: float = 0.60,  # fraction [0-1]
        profit_factor: float = 2.5,
        expectancy: float = 25.0,
        total_trades: int = 50,
        return_pct: float = 15.0,
        kelly_score: float = 0.4,
        passes_gate: bool = True,
        regime_distribution: Optional[Dict[str, int]] = None,
        filtered_trades: Optional[int] = None,
        trade_history: Optional[List[Dict[str, Any]]] = None,
        is_regime_distribution: Optional[Dict[str, int]] = None,
        oos_regime_distribution: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        return {
            "botId": bot_id,
            "mode": "BACKTEST",
            "sharpeRatio": sharpe_ratio,
            "maxDrawdown": max_drawdown,
            "winRate": win_rate,
            "profitFactor": profit_factor,
            "expectancy": expectancy,
            "totalTrades": total_trades,
            "returnPercent": return_pct,
            "kellyScore": kelly_score,
            "gateStatus": passes_gate,
            "regimeDistribution": regime_distribution,
            "filteredTrades": filtered_trades,
            "tradeHistory": trade_history,
            "isRegimeDistribution": is_regime_distribution,
            "oosRegimeDistribution": oos_regime_distribution,
        }
    return _make


@pytest.fixture
def make_evaluation_result() -> callable:
    """
    Factory fixture returning a callable that produces the expected
    EvaluationResult for a given set of metrics.

    This is the canonical output both MT5 and cTrader adapters must produce.
    The kelly_score is computed using the standard formula:
        kelly = win_rate - (1 - win_rate) / profit_factor, clamped [0, 1]

    Usage:
        expected = make_evaluation_result(
            bot_id="bot-001",
            sharpe_ratio=1.5,
            win_rate=0.60,
            profit_factor=2.5,
        )
    """
    def _make(
        bot_id: str = "bot-001",
        sharpe_ratio: float = 1.5,
        max_drawdown: float = 0.10,
        win_rate: float = 0.60,
        profit_factor: float = 2.5,
        expectancy: float = 25.0,
        total_trades: int = 50,
        return_pct: float = 15.0,
        kelly_score: Optional[float] = None,
        passes_gate: Optional[bool] = None,
        regime_distribution: Optional[Dict[str, int]] = None,
        filtered_trades: Optional[int] = None,
    ) -> EvaluationResult:
        # Default kelly: win_rate - (1 - win_rate) / profit_factor
        if kelly_score is None:
            kelly_score = min(
                1.0,
                max(0.0, win_rate - (1 - win_rate) / (profit_factor if profit_factor > 0 else 1)),
            )

        # Default passes_gate: sharpe >= 1.0 AND max_drawdown <= 0.15
        if passes_gate is None:
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
            regime_distribution=regime_distribution,
            filtered_trades=filtered_trades,
        )
    return _make


@pytest.fixture
def realistic_trade_history() -> List[Dict[str, Any]]:
    """
    Realistic trade history for 10 trades: 6 winners, 4 losers.

    Win profits:   +100, +80, +120, +90, +110, +60   = 560
    Loss profits:  -50, -30, -20, -40                  = -140
    Net profit:    420

    Derived:
      - win_rate = 6/10 = 0.6
      - profit_factor = 560/140 = 4.0
      - expectancy = 420/10 = 42.0
    """
    return [
        {"profit": 100.0},
        {"profit": -50.0},
        {"profit": 80.0},
        {"profit": -30.0},
        {"profit": 120.0},
        {"profit": -20.0},
        {"profit": 90.0},
        {"profit": -40.0},
        {"profit": 110.0},
        {"profit": 60.0},
    ]


@pytest.fixture
def regime_distributions() -> dict:
    """
    Realistic regime distributions for IS and OOS periods.

    In-sample (IS):  30 TRENDING, 15 RANGING, 5 HIGH_CHAOS
    Out-of-sample(OOS): 20 TRENDING, 10 RANGING, 2 HIGH_CHAOS
    Merged:         50 TRENDING, 25 RANGING, 7 HIGH_CHAOS
    """
    return {
        "is": {"TRENDING": 30, "RANGING": 15, "HIGH_CHAOS": 5},
        "oos": {"TRENDING": 20, "RANGING": 10, "HIGH_CHAOS": 2},
        "merged": {"TRENDING": 50, "RANGING": 25, "HIGH_CHAOS": 7},
    }

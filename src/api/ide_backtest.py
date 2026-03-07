"""
QuantMind IDE Backtest Endpoints

API endpoints for backtesting.
"""

import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException

from src.api.ide_models import BacktestRunRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/backtest", tags=["backtest"])

# In-memory storage for backtest sessions and results
_backtest_sessions: Dict[str, Dict[str, Any]] = {}
_backtest_results: Dict[str, Dict[str, Any]] = {}


@router.post("/run")
async def run_backtest(request: BacktestRunRequest):
    """
    Run a backtest with the specified parameters.

    Supports 4 backtest variants:
    - vanilla: Historical backtest with static parameters
    - spiced: Vanilla + regime filtering (skip HIGH_CHAOS, NEWS_EVENT)
    - vanilla_full: Vanilla + Walk-Forward optimization
    - spiced_full: Spiced + Walk-Forward optimization

    Uses fee-aware Kelly position sizing when broker_id is provided.
    """
    import uuid
    import numpy as np
    import pandas as pd

    try:
        from src.backtesting.mode_runner import run_full_system_backtest, BacktestMode
        from src.backtesting.mt5_engine import MQL5Timeframe
        from src.api.ws_logger import setup_backtest_logging
    except ImportError as e:
        logger.warning(f"Backtest modules not available, using mock: {e}")
        # Return mock response for development
        import uuid
        backtest_id = str(uuid.uuid4())
        return {
            "backtest_id": backtest_id,
            "status": "running",
            "message": f"Backtest {backtest_id} started (mock mode)"
        }

    # Generate unique backtest ID
    backtest_id = str(uuid.uuid4())

    # Map timeframe string to MQL5 constant
    timeframe_map = {
        "M1": MQL5Timeframe.PERIOD_M1,
        "M5": MQL5Timeframe.PERIOD_M5,
        "M15": MQL5Timeframe.PERIOD_M15,
        "M30": MQL5Timeframe.PERIOD_M30,
        "H1": MQL5Timeframe.PERIOD_H1,
        "H4": MQL5Timeframe.PERIOD_H4,
        "D1": MQL5Timeframe.PERIOD_D1
    }
    timeframe_int = timeframe_map.get(request.timeframe, MQL5Timeframe.PERIOD_H1)

    # Default strategy code if not provided
    if request.strategy_code is None:
        request.strategy_code = '''
def on_bar(tester):
    """Simple moving average crossover strategy for backtest demo."""
    symbol = tester.symbol
    if not symbol:
        return

    close = tester.iClose(symbol, 0, 0)
    if close is None:
        return

    prev_close = tester.iClose(symbol, 0, 1)
    if prev_close is None:
        return

    if tester.iMA(symbol, 0, 0, 20, 0) is None:
        return

    ma_fast = tester.iMA(symbol, 0, 0, 20, 0)
    ma_slow = tester.iMA(symbol, 0, 0, 50, 0)

    if ma_fast is None or ma_slow is None:
        return

    tester.equity
    if ma_fast > ma_slow and prev_close <= ma_fast:
        tester.buy(symbol, 0.1)
    elif ma_fast < ma_slow and prev_close >= ma_fast:
        tester.sell(symbol, 0.1)
'''

    # Create mock data for demo
    date_range = pd.date_range(
        start=request.start_date,
        end=request.end_date,
        freq='H' if 'H' in request.timeframe else 'D'
    )
    data = pd.DataFrame({
        'time': date_range,
        'open': np.random.uniform(1.08, 1.10, len(date_range)),
        'high': np.random.uniform(1.08, 1.11, len(date_range)),
        'low': np.random.uniform(1.07, 1.10, len(date_range)),
        'close': np.random.uniform(1.08, 1.10, len(date_range)),
        'volume': np.random.randint(1000, 10000, len(date_range))
    })

    # Store backtest session
    _backtest_sessions[backtest_id] = {
        "status": "running",
        "backtest_id": backtest_id,
        "symbol": request.symbol,
        "timeframe": request.timeframe,
        "variant": request.variant,
        "start_date": request.start_date,
        "end_date": request.end_date,
        "created_at": datetime.now().isoformat()
    }

    # Get the running event loop for thread-safe WebSocket broadcasts
    loop = asyncio.get_running_loop()

    # Setup WebSocket logging if enabled
    if request.enable_ws_streaming:
        ws_logger, progress_streamer = setup_backtest_logging(backtest_id, loop=loop)
    else:
        ws_logger = None
        progress_streamer = None

    async def run_backtest_task():
        try:
            # Offload sync CPU-heavy backtest to thread pool
            result = await loop.run_in_executor(
                None,
                lambda: run_full_system_backtest(
                    mode=request.variant,
                    data=data,
                    symbol=request.symbol,
                    timeframe=timeframe_int,
                    strategy_code=request.strategy_code or "",
                    initial_cash=request.initial_cash if request.initial_cash is not None else 10000.0,
                    commission=request.commission if request.commission is not None else 0.001,
                    broker_id=request.broker_id if request.broker_id is not None else "icmarkets_raw",
                    backtest_id=backtest_id,
                    progress_streamer=progress_streamer,
                    ws_logger=ws_logger,
                    enable_ws_streaming=request.enable_ws_streaming or False,
                    loop=loop
                )
            )

            # Handle result
            if isinstance(result, dict):
                first_key = next(iter(result))
                backtest_result = result[first_key]
            else:
                backtest_result = result

            # Store results
            _backtest_results[backtest_id] = {
                "backtest_id": backtest_id,
                "final_balance": backtest_result.final_cash,
                "total_trades": backtest_result.trades,
                "win_rate": getattr(backtest_result, 'win_rate', None),
                "sharpe_ratio": backtest_result.sharpe,
                "drawdown": backtest_result.drawdown,
                "return_pct": backtest_result.return_pct,
                "duration_seconds": None,
                "results": backtest_result.to_dict() if hasattr(backtest_result, 'to_dict') else {}
            }

            _backtest_sessions[backtest_id]["status"] = "completed"
            _backtest_sessions[backtest_id]["completed_at"] = datetime.now().isoformat()

            logger.info(f"Backtest {backtest_id} completed")

        except Exception as e:
            logger.error(f"Backtest {backtest_id} failed: {e}")
            _backtest_sessions[backtest_id]["status"] = "error"
            _backtest_sessions[backtest_id]["error"] = str(e)

    asyncio.create_task(run_backtest_task())

    return {
        "backtest_id": backtest_id,
        "status": "running",
        "message": f"Backtest {backtest_id} started"
    }


@router.get("/status/{backtest_id}")
async def get_backtest_status(backtest_id: str):
    """
    Get the status of a running or completed backtest.

    Returns the current status including progress if running.
    """
    session = _backtest_sessions.get(backtest_id)

    if session is None:
        raise HTTPException(404, f"Backtest {backtest_id} not found")

    return {
        "backtest_id": backtest_id,
        "status": session.get("status", "unknown"),
        "progress": 100 if session.get("status") == "completed" else 0
    }


@router.get("/results")
async def list_backtest_results():
    """
    List all available backtest results.
    Returns a list of all stored backtest results.
    """
    return list(_backtest_results.values())


@router.get("/results/{backtest_id}")
async def get_backtest_results(backtest_id: str):
    """
    Get the results of a completed backtest.
    """
    result = _backtest_results.get(backtest_id)

    if result is None:
        raise HTTPException(404, f"Results for backtest {backtest_id} not found")

    return result


@router.post("/pbo/calculate")
async def calculate_pbo(request: dict):
    """
    Calculate Probability of Backtest Overfitting (PBO).

    Uses Combinatorially Symmetric Cross-Validation (CSCV) method to detect
    whether a strategy is likely overfitted to historical data.

    Request body:
        returns_series: List of period returns (required)
        n_blocks: Number of blocks for CSCV (optional, default: 5)
        n_simulations: Number of bootstrap simulations (optional, default: 100)
        drawdown_series: Optional list of drawdowns

    Returns:
        Dict with pbo, recommendation, and detailed metrics
    """
    try:
        from src.backtesting.pbo_calculator import PBOCalculator

        returns_series = request.get("returns_series", [])
        n_blocks = request.get("n_blocks", 5)
        n_simulations = request.get("n_simulations", 100)
        drawdown_series = request.get("drawdown_series")

        if not returns_series or len(returns_series) < 10:
            return {
                "pbo": 0.5,
                "recommendation": "INSUFFICIENT_DATA",
                "reason": "Need at least 10 data points for PBO calculation",
                "first_half_return": 0.0,
                "second_half_return": 0.0,
                "return_drift": 0.0,
                "confidence_interval": (0.0, 0.0)
            }

        calculator = PBOCalculator(n_blocks=n_blocks, n_simulations=n_simulations)

        result = calculator.evaluate_strategy_robustness(
            returns_series=returns_series,
            drawdown_series=drawdown_series
        )

        return result

    except ImportError as e:
        logger.warning(f"PBO calculator not available: {e}")
        return {
            "pbo": 0.5,
            "recommendation": "ERROR",
            "reason": f"PBO calculator not available: {str(e)}",
            "first_half_return": 0.0,
            "second_half_return": 0.0,
            "return_drift": 0.0,
            "confidence_interval": (0.0, 0.0)
        }
    except Exception as e:
        logger.error(f"PBO calculation error: {e}")
        return {
            "pbo": 0.5,
            "recommendation": "ERROR",
            "reason": str(e),
            "first_half_return": 0.0,
            "second_half_return": 0.0,
            "return_drift": 0.0,
            "confidence_interval": (0.0, 0.0)
        }


@router.post("/pbo/evaluate-parameters")
async def evaluate_parameter_sets(request: dict):
    """
    Evaluate robustness across multiple parameter sets.

    Request body:
        parameter_results: Dict mapping parameter names to their
                          in-sample and out-of-sample performance

    Returns:
        Dict with PBO and best parameter set recommendation
    """
    try:
        from src.backtesting.pbo_calculator import PBOCalculator

        parameter_results = request.get("parameter_results", {})
        n_blocks = request.get("n_blocks", 5)
        n_simulations = request.get("n_simulations", 100)

        if not parameter_results:
            return {
                "pbo": 0.5,
                "recommendation": "NO_DATA",
                "reason": "No parameter results provided"
            }

        calculator = PBOCalculator(n_blocks=n_blocks, n_simulations=n_simulations)
        result = calculator.evaluate_parameter_sets(parameter_results)

        return result

    except ImportError as e:
        logger.warning(f"PBO calculator not available: {e}")
        return {
            "pbo": 0.5,
            "recommendation": "ERROR",
            "reason": f"PBO calculator not available: {str(e)}"
        }
    except Exception as e:
        logger.error(f"Parameter evaluation error: {e}")
        return {
            "pbo": 0.5,
            "recommendation": "ERROR",
            "reason": str(e)
        }

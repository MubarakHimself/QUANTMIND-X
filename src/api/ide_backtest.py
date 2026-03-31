"""
QuantMind IDE Backtest Endpoints

API endpoints for backtesting.
"""

import logging
import asyncio
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException

from src.api.ide_models import BacktestRunRequest
from src.data.data_manager import DataManager
from src.database.engine import get_session
from src.database.models import StrategyPerformance
from src.database.models.base import TradingMode

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
    try:
        from src.backtesting.full_backtest_pipeline import FullBacktestPipeline
        from src.backtesting.mt5_engine import MQL5Timeframe
        from src.api.ws_logger import setup_backtest_logging
    except ImportError as e:
        raise ImportError(
            f"Backtest modules not available: {e}. "
            "Cannot run backtest without backtesting infrastructure."
        ) from e

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

    # Fetch real OHLCV data from DataManager (Dukascopy)
    dm = DataManager(prefer_dukascopy=True)
    timeframe_minutes = MQL5Timeframe.to_minutes(timeframe_int)

    # Calculate bar count from date range
    start_dt = datetime.strptime(request.start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(request.end_date, '%Y-%m-%d')
    delta = end_dt - start_dt
    bars_needed = int(delta.total_seconds() / (timeframe_minutes * 60))
    bars_needed = max(bars_needed, 100)  # minimum 100 bars

    data = dm.fetch_data(
        symbol=request.symbol,
        timeframe=timeframe_int,
        count=bars_needed,
        start_date=start_dt,
        end_date=end_dt,
        prefer_dukascopy=True
    )

    if len(data) == 0:
        raise HTTPException(
            status_code=503,
            detail=f"No data available for {request.symbol} on {request.timeframe} "
                   f"from {request.start_date} to {request.end_date}"
        )

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
            # Create pipeline with all options
            pipeline = FullBacktestPipeline(
                initial_cash=request.initial_cash if request.initial_cash is not None else 10000.0,
                commission=request.commission if request.commission is not None else 0.001,
                broker_id=request.broker_id if request.broker_id is not None else "icmarkets_raw",
                run_monte_carlo=request.run_monte_carlo if request.run_monte_carlo is not None else True,
                mc_simulations=request.mc_simulations if request.mc_simulations is not None else 1000,
                run_pbo=request.run_pbo if request.run_pbo is not None else True,
                pbo_blocks=request.pbo_blocks if request.pbo_blocks is not None else 5,
                pbo_simulations=request.pbo_simulations if request.pbo_simulations is not None else 100,
            )

            # Run all 4 variants (sync, off event loop via run_in_executor)
            comparison = await loop.run_in_executor(
                None,
                lambda: pipeline.run_all_variants(
                    data=data,
                    symbol=request.symbol,
                    timeframe=timeframe_int,
                    strategy_code=request.strategy_code or ""
                )
            )

            # Extract requested variant's metrics for scalar DB fields
            variant_map = {
                'vanilla': comparison.vanilla_result,
                'spiced': comparison.spiced_result,
                'vanilla_full': comparison.vanilla_full_result,
                'spiced_full': comparison.spiced_full_result
            }
            requested_result = variant_map.get(request.variant, comparison.vanilla_result)

            if requested_result is None:
                requested_result = comparison.vanilla_result  # fallback

            # Extract scalar metrics from requested variant
            final_balance = getattr(requested_result, 'final_cash', 0.0) or 0.0
            initial_cash = request.initial_cash if request.initial_cash is not None else 10000.0
            net_profit = final_balance - initial_cash
            total_trades = getattr(requested_result, 'trades', 0) or 0
            win_rate = getattr(requested_result, 'win_rate', 0.0) or 0.0
            sharpe_ratio = getattr(requested_result, 'sharpe', 0.0) or 0.0
            drawdown = getattr(requested_result, 'drawdown', 0.0) or 0.0
            return_pct = getattr(requested_result, 'return_pct', 0.0) or 0.0

            # Calculate Kelly score (simplified: win_rate * profit_factor - 1)
            profit_factor = getattr(requested_result, 'profit_factor', 0.0) or 0.0
            kelly_score = (win_rate / 100.0 * profit_factor) - 1.0 if profit_factor > 0 else 0.0

            # Build full backtest_results JSON with all variant data
            backtest_results_json = {
                "backtest_id": backtest_id,
                "all_variants": comparison.to_dict(),
                "comparison_summary": {
                    "robustness_score": comparison.robustness_score,
                    "recommendation": comparison.recommendation,
                    "risk_level": comparison.risk_level,
                    "pbo": comparison.pbo,
                    "pbo_recommendation": comparison.pbo_recommendation,
                    "regime_impact": comparison.regime_impact,
                    "walk_forward_validation": comparison.walk_forward_validation
                },
                "requested_variant": request.variant,
                "symbol": request.symbol,
                "timeframe": request.timeframe,
                "start_date": request.start_date,
                "end_date": request.end_date,
                "strategy_code": request.strategy_code,
                "final_result": requested_result.to_dict() if hasattr(requested_result, 'to_dict') else {}
            }

            # Persist to database
            try:
                session = get_session()
                perf_record = StrategyPerformance(
                    strategy_name=request.strategy_name if request.strategy_name else f"{request.symbol}_{request.variant}",
                    backtest_results=backtest_results_json,
                    kelly_score=round(kelly_score, 4),
                    sharpe_ratio=round(sharpe_ratio, 4),
                    max_drawdown=round(drawdown, 4),
                    win_rate=round(win_rate, 4),
                    profit_factor=round(profit_factor, 4),
                    total_trades=total_trades,
                    mode=TradingMode.DEMO,
                    variant=request.variant,
                    symbol=request.symbol,
                    timeframe=request.timeframe,
                )
                session.add(perf_record)
                session.commit()
                logger.info(f"Persisted backtest result to database: {perf_record.id}")
            except Exception as db_err:
                logger.error(f"Failed to persist backtest result: {db_err}")
                session.rollback()
            finally:
                session.close()

            # Store results in memory
            _backtest_results[backtest_id] = {
                "backtest_id": backtest_id,
                "status": "completed",
                "final_balance": final_balance,
                "total_trades": total_trades,
                "win_rate": win_rate,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": drawdown,
                "return_pct": return_pct,
                "comparison_summary": {
                    "robustness_score": comparison.robustness_score,
                    "recommendation": comparison.recommendation,
                    "risk_level": comparison.risk_level
                },
                "all_variants": comparison.to_dict()
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

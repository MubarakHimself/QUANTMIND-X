"""
Backtest MCP Tools Module.

Provides tools for strategy backtesting via Backtest MCP server.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from src.agents.tools.mcp.manager import get_mcp_manager

logger = logging.getLogger(__name__)


async def run_backtest(
    code: str,
    config: Dict[str, Any],
    strategy_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run a backtest for MQL5 strategy code.

    This tool submits a backtest job to the Backtest MCP server.

    Args:
        code: MQL5 strategy code to backtest
        config: Backtest configuration including:
            - symbol: Trading symbol (e.g., "EURUSD")
            - timeframe: Timeframe (e.g., "H1")
            - start_date: Start date for backtest
            - end_date: End date for backtest
            - initial_deposit: Initial deposit amount
        strategy_name: Optional name for the strategy

    Returns:
        Dictionary containing:
        - backtest_id: Backtest job identifier
        - status: Job status
        - estimated_time: Estimated completion time
    """
    logger.info(f"Running backtest for strategy: {strategy_name or 'unnamed'}")

    manager = await get_mcp_manager()

    try:
        result = await manager.call_tool(
            "backtest",
            "run",
            {
                "code": code,
                "config": config,
                "strategy_name": strategy_name
            }
        )

        if isinstance(result, dict):
            return {
                "backtest_id": result.get("backtest_id", f"bt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
                "status": result.get("status", "queued"),
                "strategy_name": strategy_name,
                "config": config,
                "estimated_time": result.get("estimated_time", "2 minutes")
            }
        return {
            "backtest_id": f"bt_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "status": "unknown",
            "strategy_name": strategy_name,
            "config": config,
            "estimated_time": "unknown"
        }

    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise RuntimeError(f"Failed to run backtest: {e}")


async def get_backtest_status(backtest_id: str) -> Dict[str, Any]:
    """
    Get status of a backtest job.

    Args:
        backtest_id: Backtest job identifier

    Returns:
        Dictionary containing job status and progress
    """
    logger.info(f"Getting backtest status: {backtest_id}")

    manager = await get_mcp_manager()

    try:
        result = await manager.call_tool(
            "backtest",
            "status",
            {"backtest_id": backtest_id}
        )

        if isinstance(result, dict):
            return {
                "backtest_id": backtest_id,
                "status": result.get("status", "unknown"),
                "progress": result.get("progress", 0),
                "started_at": result.get("started_at"),
                "estimated_completion": result.get("estimated_completion")
            }
        return {
            "backtest_id": backtest_id,
            "status": "unknown",
            "progress": 0,
            "started_at": None,
            "estimated_completion": None
        }

    except Exception as e:
        logger.error(f"Failed to get backtest status: {e}")
        raise RuntimeError(f"Failed to get backtest status: {e}")


async def get_backtest_results(backtest_id: str) -> Dict[str, Any]:
    """
    Get results of a completed backtest.

    Args:
        backtest_id: Backtest job identifier

    Returns:
        Dictionary containing backtest results:
        - metrics: Performance metrics
        - trades: List of trades
        - equity_curve: Equity curve data
    """
    logger.info(f"Getting backtest results: {backtest_id}")

    manager = await get_mcp_manager()

    try:
        result = await manager.call_tool(
            "backtest",
            "results",
            {"backtest_id": backtest_id}
        )

        if isinstance(result, dict):
            return {
                "backtest_id": backtest_id,
                "status": result.get("status", "completed"),
                "metrics": result.get("metrics", {}),
                "trades": result.get("trades", []),
                "equity_curve": result.get("equity_curve", []),
                "completed_at": result.get("completed_at", datetime.now().isoformat())
            }
        return {
            "backtest_id": backtest_id,
            "status": "unknown",
            "metrics": {},
            "trades": [],
            "equity_curve": [],
            "completed_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to get backtest results: {e}")
        raise RuntimeError(f"Failed to get backtest results: {e}")


async def compare_backtests(
    backtest_ids: List[str]
) -> Dict[str, Any]:
    """
    Compare results from multiple backtests.

    Args:
        backtest_ids: List of backtest identifiers to compare

    Returns:
        Dictionary containing comparison results
    """
    logger.info(f"Comparing {len(backtest_ids)} backtests")

    manager = await get_mcp_manager()

    try:
        result = await manager.call_tool(
            "backtest",
            "compare",
            {"backtest_ids": backtest_ids}
        )

        if isinstance(result, dict):
            return {
                "backtest_ids": backtest_ids,
                "comparison": result.get("comparison", {}),
                "metrics_table": result.get("metrics_table", [])
            }
        return {
            "backtest_ids": backtest_ids,
            "comparison": {},
            "metrics_table": []
        }

    except Exception as e:
        logger.error(f"Failed to compare backtests: {e}")
        raise RuntimeError(f"Failed to compare backtests: {e}")

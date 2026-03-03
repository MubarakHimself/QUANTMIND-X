"""
MCP Tool Integration for QuantCode agent.

Contains all MCP (Model Context Protocol) tool wrappers.
"""

import logging
import asyncio
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Lazy imports to avoid module not found errors
_PaperTradingValidator = None
_PaperTradingDeployer = None


def _get_paper_trading_validator():
    """Lazy load PaperTradingValidator to avoid import errors."""
    global _PaperTradingValidator
    if _PaperTradingValidator is None:
        try:
            from src.agents.paper_trading_validator import PaperTradingValidator
            _PaperTradingValidator = PaperTradingValidator
        except ImportError as e:
            logger.warning(f"PaperTradingValidator not available: {e}")
            _PaperTradingValidator = None
    return _PaperTradingValidator


def _get_paper_trading_deployer():
    """Lazy load PaperTradingDeployer to avoid import errors."""
    global _PaperTradingDeployer
    if _PaperTradingDeployer is None:
        try:
            from mcp_mt5.paper_trading.deployer import PaperTradingDeployer
            _PaperTradingDeployer = PaperTradingDeployer
        except ImportError as e:
            logger.warning(f"PaperTradingDeployer not available: {e}")
            _PaperTradingDeployer = None
    return _PaperTradingDeployer


async def compile_mql5(code: str, filename: str, code_type: str = "expert") -> Dict[str, Any]:
    """
    Compile MQL5 code using MT5 Compiler MCP.

    Args:
        code: MQL5 source code
        filename: Output filename (without extension)
        code_type: Type of code (expert, indicator, script, library)

    Returns:
        Compilation result with success status, errors, and warnings
    """
    try:
        from src.agents.tools.mcp_tools import compile_mql5_code
        result = await compile_mql5_code(
            code=code,
            filename=filename,
            code_type=code_type
        )
        return result
    except Exception as e:
        logger.error(f"MT5 Compiler MCP call failed: {e}")
        return {
            "success": False,
            "errors": [str(e)],
            "warnings": [],
            "output_path": None
        }


async def run_backtest_mcp(code: str, config: Dict[str, Any], strategy_name: str) -> Dict[str, Any]:
    """
    Run backtest using Backtest MCP.

    Args:
        code: MQL5 source code
        config: Backtest configuration
        strategy_name: Name of the strategy

    Returns:
        Backtest results
    """
    try:
        from src.agents.tools.mcp_tools import run_backtest, get_backtest_results, get_backtest_status

        # Submit backtest job
        job = await run_backtest(
            code=code,
            config=config,
            strategy_name=strategy_name
        )

        backtest_id = job.get("backtest_id")
        if not backtest_id:
            raise RuntimeError("No backtest ID returned")

        # Wait for completion
        max_wait = 300  # 5 minutes
        waited = 0
        while waited < max_wait:
            status = await get_backtest_status(backtest_id)
            if status.get("status") == "completed":
                break
            if status.get("status") == "failed":
                raise RuntimeError(f"Backtest failed: {status.get('error', 'Unknown error')}")
            await asyncio.sleep(5)
            waited += 5

        if waited >= max_wait:
            raise TimeoutError(f"Backtest timed out after {max_wait} seconds")

        # Get results
        results = await get_backtest_results(backtest_id)
        return results

    except Exception as e:
        logger.error(f"Backtest MCP call failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "metrics": {},
            "trades": []
        }


async def validate_syntax(code: str) -> Dict[str, Any]:
    """
    Validate MQL5 syntax using MT5 Compiler MCP.

    Args:
        code: MQL5 source code

    Returns:
        Validation result
    """
    try:
        from src.agents.tools.mcp_tools import validate_mql5_syntax
        result = await validate_mql5_syntax(code)
        return result
    except Exception as e:
        logger.error(f"Syntax validation failed: {e}")
        return {
            "valid": False,
            "errors": [str(e)],
            "warnings": []
        }

"""
Backtest MCP Server - Main Entry Point

FastMCP server for running trading strategy backtests.
Follows MT5 MCP pattern from /mcp-metatrader5-server/src/mcp_mt5/main.py

Spec: lines 39-52, 488-526
"""

import logging
from datetime import datetime
from typing import Any, Literal
from fastmcp import FastMCP
from pydantic import ValidationError

# Handle both relative and absolute imports
try:
    from .models import (
        BacktestConfig,
        BacktestResult,
        BacktestStatus,
        BacktestSyntaxError,
        BacktestDataError,
        BacktestRuntimeError,
        BacktestTimeoutError,
    )
    from .queue_manager import BacktestQueueManager
except ImportError:
    from models import (
        BacktestConfig,
        BacktestResult,
        BacktestStatus,
        BacktestSyntaxError,
        BacktestDataError,
        BacktestRuntimeError,
        BacktestTimeoutError,
    )
    from queue_manager import BacktestQueueManager

logger = logging.getLogger(__name__)

# Initialize FastMCP server following MT5 MCP pattern
mcp = FastMCP(
    "Backtest MCP Server",
    instructions="""
    This server is used to run trading strategy backtests using Backtrader and MetaTrader 5.

    Backtest Tools:
    - run_backtest: Submit a strategy for backtesting with configuration
    - get_backtest_status: Check the status of a running or completed backtest

    Supported Languages:
    - Python: Direct Backtrader framework execution
    - MQ5: MetaTrader 5 strategy backtesting via MT5 Python API

    Backtest Configuration:
    - symbol: Trading symbol (e.g., "EURUSD", "XAUUSD")
    - timeframe: Timeframe in minutes (60 = H1, 1440 = D1)
    - start_date: Backtest start date
    - end_date: Backtest end date
    - initial_capital: Starting capital (default: 10000.0)
    - commission: Commission per lot (default: 0.0002)
    - slippage: Slippage per trade (default: 0.0001)
    - position_size: Default position size in lots (default: 0.1)

    Performance Metrics:
    - sharpe_ratio: Risk-adjusted return measure
    - max_drawdown: Maximum peak-to-trough decline
    - total_return: Total percentage return
    - win_rate: Percentage of profitable trades
    - profit_factor: Ratio of gross profit to gross loss

    Queue Management:
    - Up to 10 simultaneous backtests
    - CPU-aware worker allocation
    - Status tracking via backtest_id
    - Result caching for identical configurations
    """,
)

# Initialize queue manager (will be fully implemented in Task Group 5)
_queue_manager = BacktestQueueManager(max_workers=10)


@mcp.tool()
async def run_backtest(
    code_content: str,
    language: Literal["python", "mq5"],
    config: dict[str, Any],
) -> str:
    """
    Run a backtest for a trading strategy.

    IMPORTANT: Pass config fields DIRECTLY as a dictionary, do NOT wrap in nested objects.

    Args:
        code_content: Strategy source code (Python or MQL5)
        language: Strategy language - "python" or "mq5"
        config: Backtest configuration as a flat dictionary with these keys:
            REQUIRED:
            - symbol (str): Trading symbol, e.g., "EURUSD", "XAUUSD"
            - timeframe (int): Timeframe in minutes, e.g., 60 for H1, 1440 for D1
            - start_date (str): Start date in ISO format, e.g., "2024-01-01"
            - end_date (str): End date in ISO format, e.g., "2024-12-31"

            OPTIONAL (use defaults if not specified):
            - initial_capital (float): Starting capital, default 10000.0
            - commission (float): Commission per lot, default 0.0002
            - slippage (float): Slippage per trade, default 0.0001
            - position_size (float): Position size in lots, default 0.1

    Returns:
        str: JSON string containing backtest_id for tracking

    Raises:
        ValueError: If config is invalid or code has syntax errors

    Example 1 - Minimal config (uses defaults):
        config = {
            "symbol": "EURUSD",
            "timeframe": 60,
            "start_date": "2024-01-01",
            "end_date": "2024-12-31"
        }

    Example 2 - Full config with all parameters:
        config = {
            "symbol": "XAUUSD",
            "timeframe": 240,
            "start_date": "2024-01-01",
            "end_date": "2024-06-30",
            "initial_capital": 50000.0,
            "commission": 0.0001,
            "slippage": 0.00005,
            "position_size": 0.5
        }

    Common Errors:
    - Invalid symbol: Must be alphanumeric with underscore
    - Invalid timeframe: Must be a valid MT5 timeframe (1, 5, 15, 30, 60, 240, 1440, etc.)
    - Invalid date range: end_date must be after start_date
    - Syntax error: Strategy code has syntax errors
    - Data error: Insufficient historical data for the symbol/period
    """
    try:
        # Validate and parse configuration
        try:
            backtest_config = BacktestConfig(**config)
        except ValidationError as e:
            error_details = e.errors()
            error_msg = "Invalid backtest configuration:\n"
            for err in error_details:
                field = " -> ".join(str(loc) for loc in err["loc"])
                error_msg += f"  - {field}: {err['msg']}\n"
            raise ValueError(error_msg)

        # Validate language
        if language not in ["python", "mq5"]:
            raise ValueError(
                f"Invalid language '{language}'. Must be 'python' or 'mq5'"
            )

        # Submit to queue manager
        backtest_id = _queue_manager.submit_backtest(
            code_content=code_content, language=language, config=config
        )

        logger.info(
            f"Backtest {backtest_id} submitted for {backtest_config.symbol} "
            f"({language}, {backtest_config.timeframe}min)"
        )

        return f'{{"backtest_id": "{backtest_id}", "status": "queued"}}'

    except (BacktestSyntaxError, BacktestDataError, BacktestRuntimeError) as e:
        logger.error(f"Backtest error: {e}")
        raise ValueError(str(e))
    except Exception as e:
        logger.error(f"Unexpected error in run_backtest: {e}")
        raise ValueError(f"Failed to run backtest: {str(e)}")


@mcp.tool()
async def get_backtest_status(backtest_id: str) -> str:
    """
    Get the status of a backtest.

    Args:
        backtest_id: Backtest identifier returned by run_backtest

    Returns:
        str: JSON string containing backtest status with fields:
            - backtest_id: Unique identifier
            - status: "queued", "running", "completed", or "failed"
            - progress_percent: Progress (0-100)
            - estimated_completion: ISO timestamp if running
            - result: Full BacktestResult if completed

    Example:
        # Check status
        status_json = await get_backtest_status(backtest_id="abc-123")
        # Returns: '{"backtest_id": "abc-123", "status": "completed", ...}'

    Example Response (completed):
        {
            "backtest_id": "abc-123",
            "status": "completed",
            "progress_percent": 100.0,
            "result": {
                "backtest_id": "abc-123",
                "status": "success",
                "metrics": {
                    "sharpe_ratio": 1.85,
                    "max_drawdown": 12.5,
                    "total_return": 25.3,
                    "win_rate": 52.0,
                    "profit_factor": 1.8
                },
                "equity_curve": [...],
                "trade_log": [...],
                "logs": "Backtest completed successfully",
                "execution_time_seconds": 45.2
            }
        }

    Example Response (running):
        {
            "backtest_id": "abc-123",
            "status": "running",
            "progress_percent": 65.0,
            "estimated_completion": "2024-01-28T12:30:00Z"
        }
    """
    try:
        status = _queue_manager.get_status(backtest_id)

        # Convert to dict for JSON serialization
        result = {
            "backtest_id": status.backtest_id,
            "status": status.status,
            "progress_percent": status.progress_percent,
        }

        if status.estimated_completion:
            result["estimated_completion"] = status.estimated_completion.isoformat()

        if status.result:
            result["result"] = status.result.model_dump()

        import json
        return json.dumps(result, indent=2)

    except Exception as e:
        logger.error(f"Error getting backtest status: {e}")
        raise ValueError(f"Failed to get backtest status: {str(e)}")


# Resource endpoint for backtest configuration schema (Task Group 6.4)
@mcp.resource("backtest://config")
@mcp.resource("resources/backtest-config")
def get_backtest_config_schema() -> str:
    """
    Get the backtest configuration schema documentation.

    Task Group 6.4: Expose /resources/backtest-config with schema.
    Accessible via both backtest://config and resources/backtest-config

    Returns:
        str: JSON schema documentation for BacktestConfig
    """
    return """
# Backtest Configuration Schema

## Required Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| symbol | string | Trading symbol (6 chars max, alphanumeric) | "EURUSD", "XAUUSD" |
| timeframe | integer | Timeframe in minutes | 60 (H1), 240 (H4), 1440 (D1) |
| start_date | string | Start date (ISO format) | "2024-01-01" |
| end_date | string | End date (ISO format) | "2024-12-31" |

## Optional Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| initial_capital | float | 10000.0 | Starting capital amount |
| commission | float | 0.0002 | Commission per lot traded |
| slippage | float | 0.0001 | Slippage per trade |
| position_size | float | 0.1 | Default position size in lots |

## Valid Timeframes (in minutes)

Minutes: 1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30
Hours: 60 (H1), 120 (H2), 180 (H3), 240 (H4), 360 (H6), 480 (H8), 720 (H12)
Days/Weeks/Months: 1440 (D1), 10080 (W1), 43200 (MN1)

## Validation Rules

1. Symbol must be non-empty and alphanumeric (underscore allowed)
2. Timeframe must be a valid MT5 timeframe constant
3. end_date must be after start_date
4. position_size must be between 0.01 and 100 lots
5. All numeric fields must be positive (or zero for commission/slippage)

## Error Classification (Task Group 6.3)

The server classifies errors into four categories:

| Error Type | Class | Description | Example |
|------------|-------|-------------|---------|
| Syntax Error | `BacktestSyntaxError` | Code has syntax errors | Missing colon, invalid syntax |
| Data Error | `BacktestDataError` | Insufficient/invalid data | < 50 data points, invalid symbol |
| Runtime Error | `BacktestRuntimeError` | Execution fails | Division by zero, invalid operations |
| Timeout Error | `BacktestTimeoutError` | Exceeds time limit | > 300 seconds execution |

## Performance Targets (Task Group 6.2)

- Simple backtest: < 2 minutes (120 seconds)
- Parallel execution: Up to 10 concurrent backtests
- Data loading: Optimized with caching
- Metrics calculation: Vectorized with NumPy

## Example Configuration

```python
config = {
    "symbol": "EURUSD",
    "timeframe": 60,
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "initial_capital": 10000.0,
    "commission": 0.0002,
    "slippage": 0.0001,
    "position_size": 0.1
}
```

## Usage Example

```python
# Submit backtest
result = await run_backtest(
    code_content=strategy_code,
    language="python",
    config=config
)

# Check status
status = await get_backtest_status(backtest_id)
```
"""


def main():
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()

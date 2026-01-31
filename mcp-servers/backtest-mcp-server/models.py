"""
Pydantic models for Backtest MCP Server.

Defines request/response schemas following spec lines 493-525.
Follows MT5 MCP pattern from /mcp-metatrader5-server/src/mcp_mt5/main.py lines 69-100.
"""

import logging
from datetime import date, datetime
from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class BacktestConfig(BaseModel):
    """
    Schema for backtest configuration.

    Spec: lines 493-502
    """

    symbol: str = Field(..., description="Trading symbol (e.g., 'EURUSD', 'XAUUSD')")
    timeframe: int = Field(
        ..., description="Timeframe in minutes (60 = H1, 1440 = D1)", gt=0
    )
    start_date: date = Field(..., description="Backtest start date")
    end_date: date = Field(..., description="Backtest end date")
    initial_capital: float = Field(
        default=10000.0, description="Initial capital for backtest", gt=0
    )
    commission: float = Field(
        default=0.0002, description="Commission per lot", ge=0
    )
    slippage: float = Field(default=0.0001, description="Slippage per trade", ge=0)
    position_size: float = Field(
        default=0.1, description="Default position size in lots", gt=0
    )

    @field_validator("symbol")
    @classmethod
    def _symbol_valid(cls, v: str) -> str:
        """Validate symbol is not empty and contains only alphanumeric characters."""
        if not v or not v.strip():
            raise ValueError("symbol must be a non-empty string")
        if not v.replace("_", "").isalnum():
            raise ValueError(
                f"symbol '{v}' must contain only alphanumeric characters and underscore"
            )
        return v.upper()

    @field_validator("timeframe")
    @classmethod
    def _timeframe_valid(cls, v: int) -> int:
        """Validate timeframe is a valid MT5 timeframe."""
        valid_timeframes = {
            1,
            2,
            3,
            4,
            5,
            6,
            10,
            12,
            15,
            20,
            30,
            60,
            120,
            180,
            240,
            360,
            480,
            720,
            1440,
            10080,
            43200,
        }
        if v not in valid_timeframes:
            raise ValueError(
                f"timeframe {v} is not valid. Supported: {sorted(valid_timeframes)}"
            )
        return v

    @field_validator("end_date")
    @classmethod
    def _date_range_valid(cls, v: date, info) -> date:
        """Validate end_date is after start_date."""
        if "start_date" in info.data and v <= info.data["start_date"]:
            raise ValueError("end_date must be after start_date")
        return v

    @field_validator("position_size")
    @classmethod
    def _position_size_valid(cls, v: float) -> float:
        """Validate position size is reasonable (0.01 to 100 lots)."""
        if v < 0.01 or v > 100:
            raise ValueError("position_size must be between 0.01 and 100 lots")
        return v


class BacktestResult(BaseModel):
    """
    Schema for backtest result.

    Spec: lines 504-513
    """

    backtest_id: str = Field(..., description="Unique backtest identifier")
    status: Literal["success", "error", "timeout"] = Field(
        ..., description="Backtest execution status"
    )
    metrics: dict[str, float] | None = Field(
        None,
        description="Performance metrics: sharpe_ratio, max_drawdown, total_return, win_rate, profit_factor",
    )
    equity_curve: list[dict[str, Any]] | None = Field(
        None, description="Equity curve data: [{'timestamp': '...', 'value': 123.45}, ...]"
    )
    trade_log: list[dict[str, Any]] | None = Field(
        None, description="Trade log with entry, exit, PnL for each trade"
    )
    logs: str = Field(default="", description="Execution logs")
    execution_time_seconds: float = Field(
        ..., description="Execution time in seconds", ge=0
    )


class BacktestStatus(BaseModel):
    """
    Schema for backtest status.

    Spec: lines 518-525
    """

    backtest_id: str = Field(..., description="Unique backtest identifier")
    status: Literal["queued", "running", "completed", "failed"] = Field(
        ..., description="Current backtest status"
    )
    progress_percent: float = Field(
        ..., description="Progress percentage (0-100)", ge=0, le=100
    )
    estimated_completion: datetime | None = Field(
        None, description="Estimated completion time"
    )
    result: BacktestResult | None = Field(None, description="Final result if completed")


# Backtest Error Classes
# Task Group 6.3: Comprehensive error classification

class BacktestSyntaxError(Exception):
    """
    Raised when strategy code has syntax errors.

    Task Group 6.3: Syntax error classification for Python and MQL5 code.

    Attributes:
        message: Error message describing the syntax issue
        line_number: Line number where error occurred (if available)
        error_code: Error classification code
    """

    # Error classification codes
    ERROR_MISSING_COLON = "SYN_001"
    ERROR_INVALID_INDENTATION = "SYN_002"
    ERROR_UNTERMINATED_STRING = "SYN_003"
    ERROR_INVALID_TOKEN = "SYN_004"
    ERROR_UNMATCHED_BRACKET = "SYN_005"

    def __init__(self, message: str, line_number: int | None = None, error_code: str | None = None):
        self.line_number = line_number
        self.error_code = error_code or "SYN_000"

        # Format message with line number
        if line_number:
            message = f"Syntax error at line {line_number}: {message}"

        # Add error code
        message = f"[{self.error_code}] {message}"

        super().__init__(message)


class BacktestDataError(Exception):
    """
    Raised when backtest data is insufficient or invalid.

    Task Group 6.3: Data error classification for market data issues.

    Attributes:
        message: Error message describing the data issue
        symbol: Trading symbol that caused the error
        error_code: Error classification code
    """

    # Error classification codes
    ERROR_INSUFFICIENT_DATA = "DAT_001"
    ERROR_MISSING_SYMBOL = "DAT_002"
    ERROR_INVALID_DATE_RANGE = "DAT_003"
    ERROR_DATA_QUALITY = "DAT_004"

    def __init__(self, message: str, symbol: str | None = None, error_code: str | None = None):
        self.symbol = symbol
        self.error_code = error_code or "DAT_000"

        # Format message with symbol
        if symbol:
            message = f"Data error for {symbol}: {message}"

        # Add error code
        message = f"[{self.error_code}] {message}"

        super().__init__(message)


class BacktestRuntimeError(Exception):
    """
    Raised when strategy execution fails at runtime.

    Task Group 6.3: Runtime error classification for execution failures.

    Attributes:
        message: Error message describing the runtime issue
        error_type: Type of runtime error (e.g., "ZeroDivisionError")
        error_code: Error classification code
    """

    # Error classification codes
    ERROR_DIVISION_BY_ZERO = "RUN_001"
    ERROR_INVALID_OPERATION = "RUN_002"
    ERROR_MEMORY_ERROR = "RUN_003"
    ERROR_IMPORT_ERROR = "RUN_004"
    ERROR_ATTRIBUTE_ERROR = "RUN_005"

    def __init__(self, message: str, error_type: str | None = None, error_code: str | None = None):
        self.error_type = error_type or "RuntimeError"
        self.error_code = error_code or "RUN_000"

        # Format message with error type
        if error_type:
            message = f"Runtime error ({error_type}): {message}"

        # Add error code
        message = f"[{self.error_code}] {message}"

        super().__init__(message)


class BacktestTimeoutError(Exception):
    """
    Raised when backtest execution exceeds timeout.

    Task Group 6.3: Timeout error classification for execution timeouts.

    Attributes:
        timeout_seconds: Timeout limit that was exceeded
        error_code: Error classification code
    """

    # Error classification codes
    ERROR_EXECUTION_TIMEOUT = "TIM_001"
    ERROR_DATA_LOAD_TIMEOUT = "TIM_002"

    def __init__(self, timeout_seconds: float, error_code: str | None = None):
        self.timeout_seconds = timeout_seconds
        self.error_code = error_code or "TIM_001"

        message = f"Backtest exceeded timeout of {timeout_seconds} seconds"
        message = f"[{self.error_code}] {message}"

        super().__init__(message)

"""
Trading API Models

Contains all request/response models and enums for the trading API.
"""

from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from enum import Enum

from pydantic import BaseModel, Field, validator


class BacktestVariant(str, Enum):
    """Supported backtest variants."""
    VANILLA = "vanilla"
    SPICED = "spiced"
    VANILLA_FULL = "vanilla_full"
    SPICED_FULL = "spiced_full"


class Timeframe(str, Enum):
    """Supported timeframes."""
    M1 = "M1"
    M5 = "M5"
    M15 = "M15"
    M30 = "M30"
    H1 = "H1"
    H4 = "H4"
    D1 = "D1"
    W1 = "W1"
    MN1 = "MN1"


# -----------------------------------------------------------------------------
# Backtest API Models
# -----------------------------------------------------------------------------

class BacktestRunRequest(BaseModel):
    """Request model for POST /api/v1/backtest/run"""

    symbol: str = Field(..., description="Trading symbol (e.g., EURUSD)", min_length=1)
    timeframe: Timeframe = Field(..., description="Timeframe for backtesting")
    variant: BacktestVariant = Field(default=BacktestVariant.VANILLA, description="Backtest variant")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    initial_cash: float = Field(default=10000.0, description="Initial account balance", gt=0)
    commission: float = Field(default=0.001, description="Commission per trade", ge=0)
    slippage: float = Field(default=0.0, description="Slippage in points", ge=0)

    # Regime filtering for spiced variants
    regime_filtering: bool = Field(default=False, description="Enable regime filtering")
    chaos_threshold: float = Field(default=0.6, description="Chaos score threshold", ge=0, le=1)
    banned_regimes: List[str] = Field(
        default_factory=lambda: ["NEWS_EVENT", "HIGH_CHAOS"],
        description="Regimes to filter out"
    )

    # Strategy parameters
    strategy_code: Optional[str] = Field(
        default=None,
        description="Custom strategy code (Python with on_bar function)"
    )

    @validator('symbol')
    def validate_symbol(cls, v):
        """Validate and normalize symbol format."""
        v = v.replace('.', '').replace('_', '').upper()
        if len(v) < 6:
            raise ValueError(f"Invalid symbol format: {v}")
        return v

    @validator('start_date', 'end_date')
    def validate_date_format(cls, v):
        """Validate date format YYYY-MM-DD."""
        try:
            datetime.strptime(v, '%Y-%m-%d')
        except ValueError:
            raise ValueError(f"Invalid date format: {v}. Use YYYY-MM-DD")
        return v


class BacktestRunResponse(BaseModel):
    """Response model for POST /api/v1/backtest/run"""

    success: bool = Field(..., description="Whether backtest was queued successfully")
    backtest_id: str = Field(..., description="Unique backtest identifier")
    status: str = Field(..., description="Initial status (queued/running/completed)")
    message: str = Field(..., description="Status message")
    estimated_time_seconds: Optional[int] = Field(default=None, description="Estimated completion time")


class BacktestResultResponse(BaseModel):
    """Response model for GET /api/v1/backtest/results/{id}"""

    backtest_id: str = Field(..., description="Backtest identifier")
    status: str = Field(..., description="Backtest status (queued/running/completed/failed)")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Performance metrics
    sharpe_ratio: Optional[float] = Field(default=None, description="Sharpe ratio")
    return_pct: Optional[float] = Field(default=None, description="Total return percentage")
    max_drawdown: Optional[float] = Field(default=None, description="Maximum drawdown percentage")
    total_trades: Optional[int] = Field(default=None, description="Total number of trades")
    win_rate: Optional[float] = Field(default=None, description="Win rate percentage")
    profit_factor: Optional[float] = Field(default=None, description="Profit factor")

    # Backtest parameters
    initial_cash: Optional[float] = Field(default=None, description="Initial account balance")
    final_cash: Optional[float] = Field(default=None, description="Final account balance")

    # Data for visualization
    equity_curve: List[float] = Field(default_factory=list, description="Equity curve data points")
    trade_history: List[Dict[str, Any]] = Field(default_factory=list, description="Trade history")

    # Error information if failed
    error_message: Optional[str] = Field(default=None, description="Error message if failed")

    # Regime analytics for spiced variants
    regime_distribution: Optional[Dict[str, int]] = Field(
        default=None,
        description="Distribution of regimes during backtest"
    )
    filtered_trades: Optional[int] = Field(
        default=None,
        description="Number of trades filtered by regime"
    )


# -----------------------------------------------------------------------------
# Data Management API Models
# -----------------------------------------------------------------------------

class DataUploadRequest(BaseModel):
    """Request model for POST /api/v1/data/upload"""

    symbol: str = Field(..., description="Trading symbol", min_length=1)
    timeframe: Timeframe = Field(..., description="Data timeframe")
    data_format: str = Field(default="csv", description="Data format (csv/parquet)")
    overwrite: bool = Field(default=False, description="Overwrite existing data")

    # File content (for direct upload)
    file_content: Optional[str] = Field(default=None, description="Base64 encoded file content")

    # URL for remote upload
    data_url: Optional[str] = Field(default=None, description="URL to fetch data from")

    @validator('data_format')
    def validate_format(cls, v):
        """Validate data format."""
        if v not in ['csv', 'parquet']:
            raise ValueError(f"Unsupported format: {v}. Use csv or parquet")
        return v


class DataUploadResponse(BaseModel):
    """Response model for POST /api/v1/data/upload"""

    success: bool = Field(..., description="Whether upload was successful")
    message: str = Field(..., description="Status message")
    symbol: str = Field(..., description="Symbol uploaded")
    timeframe: str = Field(..., description="Timeframe of data")
    rows_uploaded: int = Field(..., description="Number of rows uploaded")
    file_path: str = Field(..., description="Path to cached Parquet file")
    upload_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DataStatusResponse(BaseModel):
    """Response model for GET /api/v1/data/status"""

    symbols: List[str] = Field(..., description="Available symbols in cache")
    total_cached_files: int = Field(..., description="Total number of cached files")
    last_updated: datetime = Field(..., description="Last cache update timestamp")
    cache_size_mb: float = Field(..., description="Cache size in megabytes")
    data_quality_score: float = Field(..., description="Overall data quality score (0-1)")

    # Detailed symbol information
    symbol_details: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Per-symbol details (timeframes, rows, quality)"
    )


class DataRefreshRequest(BaseModel):
    """Request model for POST /api/v1/data/refresh"""

    symbols: Optional[List[str]] = Field(default=None, description="Symbols to refresh (all if None)")
    force: bool = Field(default=False, description="Force refresh even if data is recent")


# -----------------------------------------------------------------------------
# Trading Control API Models
# -----------------------------------------------------------------------------

class EmergencyStopRequest(BaseModel):
    """Request model for POST /api/v1/trading/emergency_stop"""

    reason: str = Field(default="Manual API request", description="Reason for emergency stop")
    force_close: bool = Field(default=True, description="Force close all positions")
    use_smart_exit: bool = Field(default=False, description="Use smart exit strategy")
    current_pnl_pct: float = Field(default=0.0, description="Current P&L percentage for smart exit")


class EmergencyStopResponse(BaseModel):
    """Response model for POST /api/v1/trading/emergency_stop"""

    success: bool = Field(..., description="Whether stop was triggered")
    message: str = Field(..., description="Status message")
    kill_switch_active: bool = Field(..., description="Whether kill switch is now active")
    positions_closed: int = Field(..., description="Number of positions closed")
    triggered_by: str = Field(..., description="Who triggered the stop")
    timestamp: datetime = Field(..., description="When stop was triggered")

    # Exit strategy details
    exit_strategy: Optional[str] = Field(default=None, description="Exit strategy used")
    accounts_affected: List[str] = Field(default_factory=list, description="Accounts affected")


class TradingStatusResponse(BaseModel):
    """Response model for GET /api/v1/trading/status"""

    trading_enabled: bool = Field(..., description="Whether trading is currently enabled")
    kill_switch_active: bool = Field(..., description="Whether kill switch is active")
    current_regime: str = Field(..., description="Current market regime")
    chaos_score: float = Field(..., description="Current chaos score (0-1)")
    regime_quality: float = Field(..., description="Regime quality score (0-1)")

    # Account state
    open_positions: int = Field(..., description="Number of open positions")
    daily_pnl_pct: float = Field(..., description="Daily P&L percentage")
    account_equity: float = Field(..., description="Current account equity")
    account_balance: float = Field(..., description="Current account balance")

    # Trading constraints
    risk_multiplier: float = Field(..., description="Current risk multiplier")
    daily_loss_limit_pct: float = Field(..., description="Daily loss limit percentage")
    max_drawdown_pct: float = Field(..., description="Maximum drawdown percentage")

    last_update: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BotStatusResponse(BaseModel):
    """Response model for GET /api/v1/trading/bots"""

    total_bots: int = Field(..., description="Total number of bots")
    active_bots: int = Field(..., description="Number of active bots")
    primal_bots: int = Field(..., description="Number of primal bots")
    pending_bots: int = Field(..., description="Number of pending bots")
    quarantine_bots: int = Field(..., description="Number of quarantined bots")

    # Bot details
    bots: List[Dict[str, Any]] = Field(default_factory=list, description="Bot status details")


# -----------------------------------------------------------------------------
# Broker Connection API Models (Phase 5)
# -----------------------------------------------------------------------------

class BrokerConnectRequest(BaseModel):
    """Request model for POST /api/v1/trading/broker/connect"""

    broker_id: str = Field(..., description="Broker identifier")
    login: int = Field(..., description="Account login number")
    password: str = Field(..., description="Account password")
    server: str = Field(..., description="Broker server name")


class BrokerConnectResponse(BaseModel):
    """Response model for broker connection"""

    success: bool = Field(..., description="Whether connection was successful")
    broker_id: str = Field(..., description="Broker identifier")
    login: int = Field(..., description="Account login number")
    server: str = Field(..., description="Broker server name")
    connected: bool = Field(..., description="Whether MT5 is connected")
    account_info: Optional[Dict[str, Any]] = Field(default=None, description="Account information from MT5")
    error: Optional[str] = Field(default=None, description="Error message if connection failed")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Connection timestamp")


# -----------------------------------------------------------------------------
# Broker Registry API Models (Task Group 7.5)
# -----------------------------------------------------------------------------

class BrokerCreateRequest(BaseModel):
    """Request model for POST /api/v1/brokers"""

    broker_id: str = Field(..., description="Unique broker identifier", min_length=1)
    broker_name: str = Field(..., description="Human-readable broker name", min_length=1)
    spread_avg: float = Field(default=0.0, description="Average spread in points")
    commission_per_lot: float = Field(default=0.0, description="Commission per standard lot")
    lot_step: float = Field(default=0.01, description="Minimum lot step increment", gt=0)
    min_lot: float = Field(default=0.01, description="Minimum lot size", gt=0)
    max_lot: float = Field(default=100.0, description="Maximum lot size", gt=0)
    pip_values: Dict[str, float] = Field(
        default_factory=lambda: {"EURUSD": 10.0, "GBPUSD": 10.0, "XAUUSD": 1.0},
        description="Symbol pip values"
    )
    preference_tags: List[str] = Field(
        default_factory=lambda: ["STANDARD"],
        description="Broker preference tags (RAW_ECN, LOW_SPREAD, etc.)"
    )


class BrokerResponse(BaseModel):
    """Response model for broker profile"""
    id: int
    broker_id: str
    broker_name: str
    spread_avg: float
    commission_per_lot: float
    lot_step: float
    min_lot: float
    max_lot: float
    pip_values: Dict[str, float]
    preference_tags: List[str]


class BrokersListResponse(BaseModel):
    """Response model for GET /api/v1/brokers"""
    brokers: List[Dict[str, Any]]
    count: int

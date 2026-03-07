# src/agents/departments/schemas/research.py
"""
Research Department Schemas

Pydantic models for structured outputs in the Research department.
"""
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class StrategyType(str, Enum):
    """Types of trading strategies."""
    TREND = "trend"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    MOMENTUM = "momentum"
    SCALPING = "scalping"
    SWING = "swing"


class TimeFrame(str, Enum):
    """Timeframe for analysis."""
    M1 = "M1"
    M5 = "M5"
    M15 = "M15"
    H1 = "H1"
    H4 = "H4"
    D1 = "D1"
    W1 = "W1"


class StrategyStatus(str, Enum):
    """Strategy development status."""
    DRAFT = "draft"
    TESTING = "testing"
    VALIDATED = "validated"
    REJECTED = "rejected"
    PRODUCTION = "production"


class IndicatorType(str, Enum):
    """Technical indicators."""
    RSI = "RSI"
    MACD = "MACD"
    BOLLINGER_BANDS = "Bollinger_Bands"
    SMA = "SMA"
    EMA = "EMA"
    ATR = "ATR"
    STOCHASTIC = "Stochastic"


class StrategyOutput(BaseModel):
    """Output model for strategy development."""
    strategy_id: str = Field(..., description="Unique strategy identifier")
    name: str = Field(..., description="Strategy name")
    strategy_type: StrategyType
    description: str = Field(..., description="Strategy description")
    symbols: List[str] = Field(..., description="Target trading symbols")
    timeframes: List[TimeFrame] = Field(default_factory=list)
    indicators: List[Dict[str, Any]] = Field(default_factory=list, description="Technical indicators used")
    entry_rules: str = Field(..., description="Entry signal rules")
    exit_rules: str = Field(..., description="Exit signal rules")
    risk_parameters: Dict[str, float] = Field(default_factory=dict)
    status: StrategyStatus = Field(default=StrategyStatus.DRAFT)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    backtest_results: Optional[Dict[str, Any]] = Field(default=None)

    class Config:
        use_enum_values = True
        json_schema_extra = {
            "example": {
                "strategy_id": "STRAT_001",
                "name": "Trend Follower",
                "strategy_type": "trend",
                "description": "Simple trend following strategy using SMA crossover",
                "symbols": ["EURUSD", "GBPUSD"],
                "timeframes": ["H1", "H4"],
                "entry_rules": "Enter long when fast SMA crosses above slow SMA",
                "exit_rules": "Exit when fast SMA crosses below slow SMA",
                "risk_parameters": {"stop_loss_pips": 50, "take_profit_pips": 100}
            }
        }


class BacktestResult(BaseModel):
    """Backtest results model."""
    backtest_id: str = Field(..., description="Unique backtest identifier")
    strategy_id: str = Field(..., description="Strategy tested")
    start_date: datetime
    end_date: datetime
    initial_balance: float = Field(..., gt=0, description="Initial account balance")
    final_balance: float = Field(..., description="Final account balance")
    total_return: float = Field(..., description="Total return percentage")
    max_drawdown: float = Field(..., description="Maximum drawdown percentage")
    sharpe_ratio: Optional[float] = Field(default=None, description="Sharpe ratio")
    win_rate: float = Field(..., ge=0, le=1, description="Win rate (0-1)")
    total_trades: int = Field(..., ge=0, description="Total number of trades")
    profitable_trades: int = Field(..., ge=0, description="Number of profitable trades")
    losing_trades: int = Field(..., ge=0, description="Number of losing trades")
    avg_profit: float = Field(..., description="Average profit per trade")
    avg_loss: float = Field(..., description="Average loss per trade")
    profit_factor: float = Field(..., description="Profit factor")
    running_time_ms: int = Field(..., description="Backtest execution time")
    symbol_results: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    monthly_returns: Dict[str, float] = Field(default_factory=dict)

    @field_validator("max_drawdown")
    @classmethod
    def validate_drawdown(cls, v: float) -> float:
        if v > 0:
            return -abs(v)  # Drawdown should be negative
        return v


class AlphaFactor(BaseModel):
    """Alpha factor research output."""
    factor_id: str = Field(..., description="Unique factor identifier")
    name: str = Field(..., description="Factor name")
    category: str = Field(..., description="Factor category (momentum, value, quality, etc.)")
    description: str = Field(..., description="Factor description")
    universe: List[str] = Field(..., description="Asset universe")
    calculation_method: str = Field(..., description="How the factor is calculated")
    historical_data_range: Dict[str, str] = Field(..., description="Data range used")
    performance_metrics: Dict[str, float] = Field(..., description="Factor performance metrics")
    correlation_with_existing: Dict[str, float] = Field(default_factory=dict)
    implementation_notes: str = Field(default="")
    status: str = Field(default="researching")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "factor_id": "ALPHA_001",
                "name": "RSI Momentum",
                "category": "momentum",
                "description": "RSI-based momentum factor",
                "universe": ["EURUSD", "GBPUSD", "USDJPY"],
                "calculation_method": "RSI(14) with z-score normalization",
                "historical_data_range": {"start": "2023-01-01", "end": "2024-01-01"},
                "performance_metrics": {"ic": 0.05, "ir": 1.2, "rank_ic": 0.03}
            }
        }


class MarketAnalysisRequest(BaseModel):
    """Market analysis request."""
    symbol: str = Field(..., description="Trading symbol")
    timeframe: TimeFrame = Field(default=TimeFrame.H1)
    indicators: List[str] = Field(default_factory=list)
    lookback_periods: int = Field(default=100, gt=0)
    include_fundamentals: bool = Field(default=False)


class MarketAnalysisResult(BaseModel):
    """Market analysis result."""
    symbol: str
    timeframe: TimeFrame
    trend: str = Field(..., description="Current trend direction")
    volatility: str = Field(..., description="Volatility level (low/medium/high)")
    key_levels: Dict[str, float] = Field(..., description="Key support/resistance levels")
    signals: List[Dict[str, Any]] = Field(default_factory=list)
    indicator_values: Dict[str, float] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SignalOutput(BaseModel):
    """Trading signal output."""
    signal_id: str = Field(..., description="Unique signal identifier")
    strategy_id: str = Field(..., description="Source strategy")
    symbol: str = Field(..., description="Trading symbol")
    direction: str = Field(..., description="Signal direction (long/short)")
    entry_price: Optional[float] = Field(default=None)
    stop_loss: Optional[float] = Field(default=None)
    take_profit: Optional[float] = Field(default=None)
    confidence: float = Field(..., ge=0, le=1, description="Signal confidence")
    rationale: str = Field(..., description="Signal rationale")
    expires_at: Optional[datetime] = Field(default=None)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

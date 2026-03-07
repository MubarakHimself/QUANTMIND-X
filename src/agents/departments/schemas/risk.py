# src/agents/departments/schemas/risk.py
"""
Risk Department Schemas

Pydantic models for structured outputs in the Risk department.
"""
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class RiskLevel(str, Enum):
    """Risk levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PositionSizeRequest(BaseModel):
    """Position size calculation request."""
    symbol: str = Field(..., description="Trading symbol")
    account_balance: float = Field(..., gt=0, description="Total account balance")
    risk_percent: float = Field(..., gt=0, le=100, description="Risk percentage per trade")
    entry_price: float = Field(..., gt=0, description="Entry price")
    stop_loss: float = Field(..., gt=0, description="Stop loss price")
    pip_value: float = Field(default=1.0, description="Pip value for the symbol")
    max_risk_amount: Optional[float] = Field(default=None, gt=0, description="Maximum risk amount")

    @field_validator("stop_loss")
    @classmethod
    def validate_stop_loss(cls, v: float, info) -> float:
        if "entry_price" in info.data:
            entry = info.data["entry_price"]
            if v >= entry:
                raise ValueError("Stop loss must be below entry for long positions")
        return v


class PositionSizeResponse(BaseModel):
    """Position size calculation response."""
    symbol: str
    quantity: float = Field(..., gt=0, description="Recommended position size")
    risk_amount: float = Field(..., description="Risk amount in account currency")
    risk_percent: float = Field(..., description="Actual risk percentage")
    pip_risk: float = Field(..., description="Pip risk")
    potential_loss: float = Field(..., description="Potential loss at stop loss")
    reward_risk_ratio: float = Field(..., description="Reward to risk ratio")
    risk_level: RiskLevel
    message: str = Field(default="", description="Additional message")
    calculations: Dict[str, float] = Field(default_factory=dict)

    class Config:
        use_enum_values = True
        json_schema_extra = {
            "example": {
                "symbol": "EURUSD",
                "quantity": 2.0,
                "risk_amount": 100.0,
                "risk_percent": 1.0,
                "pip_risk": 50,
                "potential_loss": 100.0,
                "reward_risk_ratio": 2.0,
                "risk_level": "low"
            }
        }


class DrawdownInfo(BaseModel):
    """Drawdown information."""
    account_id: str = Field(..., description="Account identifier")
    current_drawdown: float = Field(..., description="Current drawdown percentage (negative)")
    peak_balance: float = Field(..., description="Peak account balance")
    current_balance: float = Field(..., description="Current account balance")
    drawdown_start: Optional[datetime] = Field(default=None, description="Drawdown start time")
    drawdown_duration_hours: Optional[float] = Field(default=None, description="Duration in hours")
    max_drawdown: float = Field(..., description="Maximum historical drawdown")
    max_drawdown_duration_hours: Optional[float] = Field(default=None)
    risk_level: RiskLevel
    alerts_triggered: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("current_drawdown", "max_drawdown")
    @classmethod
    def validate_drawdown_negative(cls, v: float) -> float:
        if v > 0:
            return -abs(v)
        return v


class VaRResult(BaseModel):
    """Value at Risk calculation result."""
    portfolio: Dict[str, float] = Field(..., description="Portfolio holdings")
    confidence_level: float = Field(..., ge=0.5, le=0.99, description="Confidence level")
    timeframe_days: int = Field(..., gt=0, description="Timeframe in days")
    var_absolute: float = Field(..., description="VaR in absolute terms")
    var_percentage: float = Field(..., description="VaR as percentage of portfolio")
    cvar: float = Field(..., description="Conditional VaR (Expected Shortfall)")
    method: str = Field(..., description="Calculation method used")
    risk_level: RiskLevel
    confidence_interval: tuple[float, float] = Field(default=(0.0, 0.0))
    historical_data_points: int = Field(default=0)
    assumptions: Dict[str, str] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "portfolio": {"EURUSD": 10000, "GBPUSD": 5000},
                "confidence_level": 0.95,
                "timeframe_days": 1,
                "var_absolute": 250.0,
                "var_percentage": 1.67,
                "cvar": 375.0,
                "method": "historical",
                "risk_level": "medium"
            }
        }


class RiskValidationRequest(BaseModel):
    """Risk validation request for a trade."""
    symbol: str
    side: str
    quantity: float
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    account_balance: float
    current_exposure: Dict[str, float] = Field(default_factory=dict)
    strategy_id: Optional[str] = None


class RiskValidationResult(BaseModel):
    """Risk validation result."""
    approved: bool = Field(..., description="Whether trade is approved")
    risk_checks: Dict[str, bool] = Field(default_factory=dict, description="Individual check results")
    violations: List[str] = Field(default_factory=list, description="Failed check messages")
    warnings: List[str] = Field(default_factory=list)
    adjusted_parameters: Dict[str, float] = Field(default_factory=dict)
    final_risk_level: RiskLevel
    overall_score: float = Field(..., ge=0, le=1)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ExposureLimit(BaseModel):
    """Exposure limit configuration."""
    limit_type: str = Field(..., description="Type of limit")
    max_value: float = Field(..., gt=0, description="Maximum allowed value")
    current_value: float = Field(default=0.0, description="Current exposure")
    unit: str = Field(default="percent", description="Unit of measurement")
    is_active: bool = Field(default=True)


class RiskLimits(BaseModel):
    """Complete risk limits configuration."""
    account_id: str
    max_daily_loss_percent: float = Field(default=5.0, gt=0, le=100)
    max_position_size: float = Field(default=10.0, gt=0)
    max_exposure_per_symbol_percent: float = Field(default=20.0, gt=0, le=100)
    max_total_exposure_percent: float = Field(default=100.0, gt=0, le=200)
    max_correlated_exposure_percent: float = Field(default=30.0, gt=0, le=100)
    max_drawdown_percent: float = Field(default=15.0, gt=0, le=100)
    min_account_balance: float = Field(default=1000.0, gt=0)
    position_limits: Dict[str, ExposureLimit] = Field(default_factory=dict)


class RiskReport(BaseModel):
    """Comprehensive risk report."""
    account_id: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    current_exposure: Dict[str, float] = Field(default_factory=dict)
    total_exposure_percent: float = Field(default=0.0)
    drawdown_info: Optional[DrawdownInfo] = None
    var_result: Optional[VaRResult] = None
    limits: RiskLimits
    active_positions_count: int = Field(default=0)
    pending_orders_count: int = Field(default=0)
    daily_pnl: float = Field(default=0.0)
    overall_risk_score: float = Field(..., ge=0, le=1)
    risk_level: RiskLevel
    recommendations: List[str] = Field(default_factory=list)
    alerts: List[str] = Field(default_factory=list)

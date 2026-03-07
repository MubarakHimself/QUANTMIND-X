# src/agents/departments/schemas/trading.py
"""
Trading Department Schemas

Pydantic models for structured outputs in the Trading department.
"""
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class OrderSide(str, Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order type."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(str, Enum):
    """Order status."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class TimeInForce(str, Enum):
    """Time in force."""
    GTC = "GTC"  # Good Till Cancelled
    IOC = "IOC"  # Immediate Or Cancel
    FOK = "FOK"  # Fill Or Kill
    DAY = "DAY"


class OrderRequest(BaseModel):
    """Order request model."""
    symbol: str = Field(..., description="Trading symbol", min_length=1)
    side: OrderSide
    quantity: float = Field(..., gt=0, description="Order quantity")
    order_type: OrderType = Field(default=OrderType.MARKET)
    price: Optional[float] = Field(default=None, gt=0, description="Limit/stop price")
    stop_price: Optional[float] = Field(default=None, gt=0, description="Stop price for stop orders")
    time_in_force: TimeInForce = Field(default=TimeInForce.GTC)
    comment: Optional[str] = Field(default=None, description="Order comment")
    magic_number: Optional[int] = Field(default=None, description="EA magic number")
    strategy_id: Optional[str] = Field(default=None, description="Associated strategy ID")

    class Config:
        use_enum_values = True
        json_schema_extra = {
            "example": {
                "symbol": "EURUSD",
                "side": "buy",
                "quantity": 1.0,
                "order_type": "market",
                "time_in_force": "GTC",
                "strategy_id": "STRAT_001"
            }
        }


class FillInfo(BaseModel):
    """Order fill information."""
    fill_id: str = Field(..., description="Unique fill identifier")
    order_id: str = Field(..., description="Associated order ID")
    fill_price: float = Field(..., gt=0, description="Fill price")
    fill_quantity: float = Field(..., gt=0, description="Filled quantity")
    fill_time: datetime = Field(default_factory=datetime.utcnow)
    commission: float = Field(default=0.0, description="Commission charged")
    swap: float = Field(default=0.0, description="Overnight swap")
    slippage: float = Field(default=0.0, description="Slippage in pips")

    @field_validator("slippage")
    @classmethod
    def validate_slippage(cls, v: float) -> float:
        return round(v, 2)


class OrderResponse(BaseModel):
    """Order response model."""
    order_id: str = Field(..., description="Unique order identifier")
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    filled_quantity: float = Field(default=0.0, description="Total filled quantity")
    status: OrderStatus
    request_price: Optional[float] = Field(default=None, description="Requested price")
    avg_fill_price: Optional[float] = Field(default=None, description="Average fill price")
    fills: list[FillInfo] = Field(default_factory=list)
    message: str = Field(default="", description="Status message")
    error_code: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        use_enum_values = True
        json_schema_extra = {
            "example": {
                "order_id": "ORD_123456",
                "symbol": "EURUSD",
                "side": "buy",
                "order_type": "market",
                "quantity": 1.0,
                "filled_quantity": 1.0,
                "status": "filled",
                "avg_fill_price": 1.0850,
                "fills": []
            }
        }


class PositionInfo(BaseModel):
    """Position information."""
    position_id: str = Field(..., description="Unique position identifier")
    symbol: str
    side: OrderSide
    quantity: float = Field(..., gt=0)
    entry_price: float = Field(..., gt=0)
    current_price: float = Field(..., gt=0)
    unrealized_pnl: float = Field(default=0.0, description="Unrealized profit/loss")
    realized_pnl: float = Field(default=0.0, description="Realized profit/loss")
    swap: float = Field(default=0.0)
    commission: float = Field(default=0.0)
    open_time: datetime
    comment: Optional[str] = Field(default=None)
    magic_number: Optional[int] = Field(default=None)


class TradeMonitorRequest(BaseModel):
    """Trade monitoring request."""
    positions: list[PositionInfo] = Field(default_factory=list)
    alert_thresholds: dict[str, float] = Field(default_factory=dict)
    check_interval_seconds: int = Field(default=60, gt=0)


class SlippageAnalysis(BaseModel):
    """Slippage analysis result."""
    symbol: str
    period: str = Field(..., description="Analysis period")
    avg_slippage: float = Field(default=0.0, description="Average slippage in pips")
    max_slippage: float = Field(default=0.0, description="Maximum slippage in pips")
    slippage_distribution: dict[str, float] = Field(default_factory=dict)
    recommendations: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ExecutionQuality(BaseModel):
    """Execution quality metrics."""
    order_id: str
    symbol: str
    requested_price: float
    fill_price: float
    slippage_pips: float
    fill_time_ms: int = Field(..., description="Time from order to fill")
    partial_fills: int = Field(default=0)
    requotes: int = Field(default=0)
    rejections: int = Field(default=0)


class VenueRoutingRequest(BaseModel):
    """Venue routing optimization request."""
    symbol: str
    side: OrderSide
    quantity: float
    venues: list[str] = Field(default_factory=list, description="Available venues")
    venue_weights: dict[str, float] = Field(default_factory=dict, description="Venue preference weights")
    routing_algorithm: str = Field(default="smart", description="Routing algorithm to use")

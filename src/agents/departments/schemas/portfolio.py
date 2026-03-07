# src/agents/departments/schemas/portfolio.py
"""
Portfolio Department Schemas

Pydantic models for structured outputs in the Portfolio department.
"""
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class OptimizationObjective(str, Enum):
    """Portfolio optimization objectives."""
    MAXIMIZE_SHARPE = "maximize_sharpe"
    MINIMIZE_VOLATILITY = "minimize_volatility"
    MAXIMIZE_RETURN = "maximize_return"
    RISK_PARITY = "risk_parity"
    MAXIMUM_DIVERSIFICATION = "maximum_diversification"


class AllocationRequest(BaseModel):
    """Portfolio allocation optimization request."""
    assets: List[str] = Field(..., description="List of assets to allocate")
    target_return: Optional[float] = Field(default=None, description="Target return percentage")
    max_risk: Optional[float] = Field(default=None, gt=0, description="Maximum risk tolerance")
    constraints: Dict[str, float] = Field(default_factory=dict, description="Additional constraints")
    optimization_objective: OptimizationObjective = Field(default=OptimizationObjective.MAXIMIZE_SHARPE)
    risk_free_rate: float = Field(default=0.0, description="Risk-free rate for calculations")
    historical_data_periods: int = Field(default=252, description="Historical periods for analysis")

    class Config:
        use_enum_values = True
        json_schema_extra = {
            "example": {
                "assets": ["EURUSD", "GBPUSD", "USDJPY"],
                "target_return": 10.0,
                "max_risk": 15.0,
                "optimization_objective": "maximize_sharpe"
            }
        }


class AssetAllocation(BaseModel):
    """Individual asset allocation."""
    asset: str = Field(..., description="Asset symbol")
    weight: float = Field(..., ge=0, le=1, description="Weight (0-1)")
    expected_return: Optional[float] = Field(default=None, description="Expected return percentage")
    volatility: Optional[float] = Field(default=None, description="Volatility percentage")
    correlation_with_portfolio: Optional[float] = Field(default=None)


class AllocationResult(BaseModel):
    """Portfolio allocation optimization result."""
    allocation_id: str = Field(..., description="Unique allocation identifier")
    allocations: List[AssetAllocation] = Field(..., description="Asset allocations")
    expected_return: float = Field(..., description="Expected portfolio return")
    portfolio_volatility: float = Field(..., description="Portfolio volatility")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    optimization_objective: OptimizationObjective
    constraints_applied: Dict[str, float] = Field(default_factory=dict)
    optimization_status: str = Field(..., description="Optimization status")
    execution_time_ms: int = Field(..., description="Optimization execution time")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        use_enum_values = True
        json_schema_extra = {
            "example": {
                "allocation_id": "ALLOC_001",
                "allocations": [
                    {"asset": "EURUSD", "weight": 0.5, "expected_return": 8.0, "volatility": 10.0},
                    {"asset": "GBPUSD", "weight": 0.3, "expected_return": 6.0, "volatility": 12.0},
                    {"asset": "USDJPY", "weight": 0.2, "expected_return": 4.0, "volatility": 8.0}
                ],
                "expected_return": 6.8,
                "portfolio_volatility": 9.5,
                "sharpe_ratio": 0.72,
                "optimization_status": "optimal"
            }
        }


class RebalancePlan(BaseModel):
    """Portfolio rebalancing plan."""
    plan_id: str = Field(..., description="Unique rebalance plan identifier")
    current_allocation: List[AssetAllocation] = Field(..., description="Current allocation")
    target_allocation: List[AssetAllocation] = Field(..., description="Target allocation")
    rebalance_threshold: float = Field(..., gt=0, description="Threshold for rebalancing")
    trades_required: List[Dict[str, float]] = Field(
        default_factory=list,
        description="Required trades to rebalance"
    )
    estimated_cost: float = Field(default=0.0, description="Estimated transaction cost")
    rebalance_reason: str = Field(default="", description="Reason for rebalancing")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class PerformanceMetrics(BaseModel):
    """Portfolio performance metrics."""
    period: str = Field(..., description="Performance period")
    start_date: datetime
    end_date: datetime
    start_value: float = Field(..., gt=0)
    end_value: float = Field(..., gt=0)
    total_return: float = Field(..., description="Total return percentage")
    annualized_return: float = Field(..., description="Annualized return")
    volatility: float = Field(..., description="Volatility")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    sortino_ratio: Optional[float] = Field(default=None, description="Sortino ratio")
    max_drawdown: float = Field(..., description="Maximum drawdown")
    calmar_ratio: Optional[float] = Field(default=None, description="Calmar ratio")
    win_rate: Optional[float] = Field(default=None, description="Win rate")
    profit_factor: Optional[float] = Field(default=None, description="Profit factor")
    best_trade: Optional[float] = Field(default=None, description="Best trade return")
    worst_trade: Optional[float] = Field(default=None, description="Worst trade return")
    avg_trade: Optional[float] = Field(default=None, description="Average trade return")
    benchmark_return: Optional[float] = Field(default=None, description="Benchmark return")
    alpha: Optional[float] = Field(default=None, description="Alpha")
    beta: Optional[float] = Field(default=None, description="Beta")
    information_ratio: Optional[float] = Field(default=None, description="Information ratio")
    tracking_error: Optional[float] = Field(default=None, description="Tracking error")
    monthly_returns: Dict[str, float] = Field(default_factory=dict)
    daily_returns: List[float] = Field(default_factory=list)

    class Config:
        json_schema_extra = {
            "example": {
                "period": "YTD",
                "total_return": 15.5,
                "annualized_return": 18.2,
                "volatility": 12.3,
                "sharpe_ratio": 1.48,
                "max_drawdown": -8.5
            }
        }


class PortfolioSummary(BaseModel):
    """Portfolio summary."""
    portfolio_id: str = Field(..., description="Portfolio identifier")
    name: str = Field(default="")
    total_value: float = Field(..., gt=0, description="Total portfolio value")
    cash_balance: float = Field(default=0.0, description="Cash balance")
    positions_value: float = Field(..., description="Positions value")
    daily_pnl: float = Field(default=0.0, description="Daily P&L")
    weekly_pnl: float = Field(default=0.0, description="Weekly P&L")
    monthly_pnl: float = Field(default=0.0, description="Monthly P&L")
    total_pnl: float = Field(default=0.0, description="Total P&L")
    positions_count: int = Field(default=0)
    allocation: List[AssetAllocation] = Field(default_factory=list)
    performance: Optional[PerformanceMetrics] = None
    risk_metrics: Dict[str, float] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class RebalanceRequest(BaseModel):
    """Rebalance request."""
    portfolio_id: str
    target_allocation: List[AssetAllocation] = Field(..., description="Target allocation")
    threshold: float = Field(default=5.0, gt=0, le=50, description="Rebalance threshold percentage")
    include_cash: bool = Field(default=True, description="Include cash in rebalancing")
    rebalance_method: str = Field(default="threshold", description="Rebalance method")
    max_turnover: Optional[float] = Field(default=None, le=1.0, description="Maximum portfolio turnover")


class PortfolioOptimizationRequest(BaseModel):
    """Portfolio optimization request."""
    portfolio_id: str
    objective: OptimizationObjective
    constraints: Dict[str, float] = Field(default_factory=dict)
    risk_free_rate: float = Field(default=0.0)
    optimization_method: str = Field(default="mean_variance")
    allow_short: bool = Field(default=False)
    min_weight: float = Field(default=0.0, ge=0)
    max_weight: float = Field(default=1.0, le=1)

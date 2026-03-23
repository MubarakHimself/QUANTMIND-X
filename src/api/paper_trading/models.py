"""
Paper Trading API Models

Pydantic models for paper trading API requests and responses.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class PromotionRequest(BaseModel):
    """Request to promote a paper trading agent to live trading."""
    target_account: str = Field(
        default="account_b_sniper",
        description="Target account for live trading"
    )
    strategy_name: Optional[str] = Field(
        default=None,
        description="Name for the live bot (defaults to paper trading name)"
    )
    strategy_type: str = Field(
        default="STRUCTURAL",
        description="Strategy classification (SCALPER, STRUCTURAL, SWING, HFT)"
    )
    target_mode: Optional[str] = Field(
        default=None,
        description="Target trading mode for promotion (demo, live). If None, auto-promotes to DEMO first."
    )
    capital_allocation: Optional[float] = Field(
        default=None,
        description="Optional capital allocation override. If None, uses PromotionManager defaults."
    )


class PromotionResult(BaseModel):
    """Result of a promotion request."""
    promoted: bool
    bot_id: Optional[str] = None
    agent_id: str
    manifest: Optional[dict] = None
    registration_status: str
    target_account: Optional[str] = None
    live_trading_url: Optional[str] = None
    performance_summary: Optional[dict] = None
    error: Optional[str] = None


class AgentPerformanceResponse(BaseModel):
    """Enhanced performance response with validation status."""
    agent_id: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    average_pnl: float = 0.0
    max_drawdown: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: Optional[float] = None
    validation_status: str = "pending"
    days_validated: int = 0
    meets_criteria: bool = False
    validation_thresholds: dict = Field(
        default_factory=lambda: {
            "min_sharpe_ratio": 1.5,
            "min_win_rate": 0.55,
            "min_validation_days": 30
        }
    )


class TradeRecordRequest(BaseModel):
    """Request to record a trade for performance tracking."""
    symbol: str = Field(..., description="Trading symbol")
    direction: str = Field(..., description="Trade direction (BUY/SELL)")
    entry_price: float = Field(..., description="Entry price")
    exit_price: float = Field(..., description="Exit price")
    pnl: float = Field(..., description="Profit/loss amount")
    timestamp: Optional[str] = Field(None, description="Trade timestamp (ISO format)")


class AddDemoAccountRequest(BaseModel):
    """Request to add a demo account."""
    login: int
    password: str
    server: str
    broker: str = "generic"
    nickname: Optional[str] = None


class DemoAccountResponse(BaseModel):
    """Response for demo account operations."""
    login: int
    server: str
    broker: str
    nickname: str
    account_type: str = "demo"
    is_active: bool = True


class ActiveAgentItem(BaseModel):
    """Summary of a single active paper trading agent."""
    ea_name: str
    pair: str
    days_running: int
    win_rate: float
    pnl_current: float
    status: str
    started_at: str


class ActiveAgentsResponse(BaseModel):
    """Response for GET /api/paper-trading/active."""
    items: List[ActiveAgentItem]

"""
Data models for paper trading deployment.

Defines Pydantic models for agent status, deployment requests, performance metrics,
and API schemas for MCP tools.
"""

import logging
from datetime import datetime, UTC
from typing import Optional, Literal
from enum import Enum

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class PaperTradingConfig(BaseModel):
    """
    Configuration for paper trading mode.
    
    Clarifies the distinction between paper trading modes:
    - Paper Trading: Pure MT5 demo account (no broker connection)
    - Demo Trading: MT5 demo account WITH broker connection
    - Live Trading: Real broker account
    
    Paper trading uses MT5 demo accounts with live market data
    but executes trades virtually without broker involvement.
    """
    use_mt5_demo: bool = Field(
        default=True,
        description="Use MT5 demo account for paper trading"
    )
    broker_connection: bool = Field(
        default=False,
        description="Connect to broker (False for pure paper trading)"
    )
    virtual_balance: float = Field(
        default=10000.0,
        description="Virtual balance for paper trading",
        ge=0
    )
    use_live_data: bool = Field(
        default=True,
        description="Use live tick data from MT5"
    )
    simulate_slippage: bool = Field(
        default=True,
        description="Simulate realistic slippage on orders"
    )
    simulate_fees: bool = Field(
        default=False,
        description="Simulate trading fees (disabled for paper trading)"
    )
    fee_rate: float = Field(
        default=0.0,
        description="Fee rate to simulate (0 = no fees)",
        ge=0,
        le=0.01
    )
    slippage_range: tuple = Field(
        default=(0.0, 0.0002),
        description="Slippage range in price units (min, max)"
    )
    track_in_quantmind_db: bool = Field(
        default=True,
        description="Track virtual trades in QuantMind database"
    )
    enable_metrics: bool = Field(
        default=True,
        description="Enable Prometheus metrics for paper trading"
    )
    
    @property
    def trading_mode(self) -> str:
        """Get the trading mode description."""
        if self.broker_connection and self.use_mt5_demo:
            return "demo"  # Demo trading with broker
        elif self.use_mt5_demo:
            return "paper"  # Pure paper trading
        else:
            return "live"  # Live trading


class AgentStatus(str, Enum):
    """Agent lifecycle status."""
    PENDING = "pending"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    UNHEALTHY = "unhealthy"


class AgentHealth(str, Enum):
    """Agent health status based on heartbeat detection."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    STALE = "stale"
    DEAD = "dead"


class PaperAgentStatus(BaseModel):
    """
    Status of a paper trading agent container.

    Returned by list_paper_agents to show all deployed agents.
    """
    agent_id: str = Field(
        description="Unique agent identifier",
        examples=["strategy-rsi-eurusd-001"]
    )
    container_id: str = Field(
        description="Docker container ID",
        examples=["a1b2c3d4e5f6"]
    )
    container_name: str = Field(
        description="Docker container name",
        examples=["quantmindx-agent-strategy-rsi-eurusd-001"]
    )
    status: AgentStatus = Field(
        description="Agent lifecycle status",
        default=AgentStatus.PENDING
    )
    health: AgentHealth = Field(
        description="Agent health based on heartbeat",
        default=AgentHealth.HEALTHY
    )
    strategy_name: str = Field(
        description="Name of the trading strategy",
        examples=["RSI Reversal"]
    )
    symbol: Optional[str] = Field(
        description="Trading symbol for the agent",
        default=None,
        examples=["EURUSD", "GBPUSD"]
    )
    timeframe: Optional[str] = Field(
        description="Timeframe for the agent (e.g., M1, H1, D1)",
        default=None,
        examples=["M1", "H1", "D1"]
    )
    mt5_account: Optional[int] = Field(
        description="MT5 account number",
        default=None,
        examples=[12345678]
    )
    mt5_server: Optional[str] = Field(
        description="MT5 server name",
        default=None,
        examples=["MetaQuotes-Demo"]
    )
    magic_number: Optional[int] = Field(
        description="EA magic number for trade identification",
        default=None,
        examples=[98765432]
    )
    redis_channel: str = Field(
        description="Redis channel for agent events",
        examples=["agent:heartbeat:strategy-rsi-eurusd-001"]
    )
    created_at: datetime = Field(
        description="Container creation timestamp",
        default_factory=lambda: datetime.now(UTC)
    )
    started_at: Optional[datetime] = Field(
        description="Container start timestamp",
        default=None
    )
    uptime_seconds: Optional[int] = Field(
        description="Agent uptime in seconds",
        default=None
    )
    last_heartbeat: Optional[datetime] = Field(
        description="Last heartbeat timestamp",
        default=None
    )
    missed_heartbeats: int = Field(
        description="Number of consecutive missed heartbeats",
        default=0
    )
    image_name: str = Field(
        description="Docker image used",
        default="quantmindx/strategy-agent:latest"
    )

    @field_validator('status', mode='before')
    @classmethod
    def validate_status(cls, v):
        """Convert string to AgentStatus enum."""
        if isinstance(v, str):
            return AgentStatus(v.lower())
        return v

    @field_validator('health', mode='before')
    @classmethod
    def validate_health(cls, v):
        """Convert string to AgentHealth enum."""
        if isinstance(v, str):
            return AgentHealth(v.lower())
        return v


class AgentDeploymentRequest(BaseModel):
    """
    Request to deploy a new paper trading agent.

    Input to deploy_paper_agent tool.
    """
    strategy_name: str = Field(
        description="Name of the trading strategy",
        examples=["RSI Reversal", "MACD Crossover"]
    )
    strategy_code: str = Field(
        description="Strategy Python code or reference to template",
        examples=["template:rsi-reversal", "custom:<code>"]
    )
    config: dict = Field(
        description="Strategy configuration parameters",
        examples=[
            {
                "rsi_period": 14,
                "oversold": 30,
                "overbought": 70,
                "symbols": ["EURUSD", "GBPUSD"],
                "timeframe": "H1",
                "lot_size": 0.1
            }
        ]
    )
    mt5_credentials: Optional[dict] = Field(
        default=None,
        description="MT5 account credentials (optional for pure paper mode). Required for broker connection.",
        examples=[
            {
                "account": 12345678,
                "password": "demo_password",
                "server": "MetaQuotes-Demo"
            }
        ]
    )
    magic_number: int = Field(
        description="Unique magic number for this agent",
        examples=[98765432]
    )
    agent_id: Optional[str] = Field(
        description="Custom agent ID (auto-generated if not provided)",
        default=None
    )
    image_tag: str = Field(
        description="Docker image tag to use",
        default="latest"
    )
    redis_host: str = Field(
        description="Redis host for event publishing",
        default="localhost"
    )
    redis_port: int = Field(
        description="Redis port",
        default=6379
    )
    redis_db: int = Field(
        description="Redis database number",
        default=0
    )
    environment_vars: Optional[dict] = Field(
        description="Additional environment variables",
        default=None
    )
    resource_limits: Optional[dict] = Field(
        description="Container resource limits",
        default=None,
        examples=[
            {
                "memory": "512m",
                "cpus": "1.0"
            }
        ]
    )
    paper_config: Optional[PaperTradingConfig] = Field(
        default=None,
        description="Paper trading configuration. If provided with broker_connection=False, "
                    "enables pure paper mode without broker credentials."
    )

    @field_validator('mt5_credentials')
    @classmethod
    def validate_mt5_credentials(cls, v):
        """Ensure MT5 credentials have required fields when provided."""
        if v is None:
            return v  # Credentials optional for pure paper mode
        required = ['account', 'password', 'server']
        missing = [k for k in required if k not in v]
        if missing:
            raise ValueError(f"Missing MT5 credentials: {', '.join(missing)}")
        return v

    @field_validator('magic_number')
    @classmethod
    def validate_magic_number(cls, v):
        """Ensure magic number is in valid range."""
        if not 0 <= v <= 2147483647:  # Max int32
            raise ValueError("magic_number must be between 0 and 2147483647")
        return v
    
    def model_post_init(self, __context):
        """Validate that credentials are provided when broker connection is required."""
        # If paper_config specifies broker_connection=True, credentials are required
        if self.paper_config and self.paper_config.broker_connection:
            if not self.mt5_credentials:
                raise ValueError(
                    "mt5_credentials required when paper_config.broker_connection is True"
                )


class AgentDeploymentResult(BaseModel):
    """
    Result of agent deployment.

    Returned by deploy_paper_agent tool.
    """
    agent_id: str = Field(
        description="Unique agent identifier"
    )
    container_id: str = Field(
        description="Docker container ID"
    )
    container_name: str = Field(
        description="Docker container name"
    )
    status: AgentStatus = Field(
        description="Initial agent status"
    )
    redis_channel: str = Field(
        description="Redis channel for monitoring"
    )
    logs_url: Optional[str] = Field(
        description="URL to view agent logs",
        default=None
    )
    created_at: datetime = Field(
        description="Deployment timestamp",
        default_factory=lambda: datetime.now(UTC)
    )
    message: str = Field(
        description="Deployment status message",
        examples=["Agent deployed successfully"]
    )


class ValidationStatus(str, Enum):
    """Validation status for paper trading agents."""
    PENDING = "pending"       # Not enough data to validate
    VALIDATING = "validating" # Validation in progress
    VALIDATED = "validated"   # Passed all validation criteria
    FAILED = "failed"         # Failed validation criteria


class AgentPerformance(BaseModel):
    """
    Performance metrics for a paper trading agent.

    Returned by get_agent_performance tool.
    """
    agent_id: str = Field(
        description="Agent identifier"
    )
    total_trades: int = Field(
        description="Total number of trades executed",
        ge=0
    )
    winning_trades: int = Field(
        description="Number of winning trades",
        ge=0
    )
    losing_trades: int = Field(
        description="Number of losing trades",
        ge=0
    )
    win_rate: float = Field(
        description="Win rate as percentage (0-100)",
        ge=0,
        le=100
    )
    total_pnl: float = Field(
        description="Total profit/loss in account currency",
        examples=[1250.50, -500.25]
    )
    average_pnl: float = Field(
        description="Average profit/loss per trade",
        examples=[25.01, -10.00]
    )
    max_drawdown: float = Field(
        description="Maximum drawdown",
        ge=0
    )
    profit_factor: float = Field(
        description="Profit factor (gross wins / gross losses)",
        ge=0
    )
    sharpe_ratio: Optional[float] = Field(
        description="Sharpe ratio (if enough data)",
        default=None
    )
    symbols_traded: list[str] = Field(
        description="List of symbols traded",
        default_factory=list
    )
    first_trade_at: Optional[datetime] = Field(
        description="First trade timestamp",
        default=None
    )
    last_trade_at: Optional[datetime] = Field(
        description="Last trade timestamp",
        default=None
    )
    calculated_at: datetime = Field(
        description="When metrics were calculated",
        default_factory=lambda: datetime.now(UTC)
    )
    # Validation fields
    validation_status: ValidationStatus = Field(
        description="Current validation status",
        default=ValidationStatus.PENDING
    )
    days_validated: int = Field(
        description="Number of days since validation started",
        ge=0,
        default=0
    )
    meets_criteria: bool = Field(
        description="Whether agent meets promotion criteria",
        default=False
    )
    validation_start_time: Optional[datetime] = Field(
        description="Timestamp when validation began",
        default=None
    )
    validation_thresholds: dict = Field(
        description="Validation thresholds applied",
        default_factory=lambda: {
            "min_sharpe_ratio": 1.5,
            "min_win_rate": 0.55,
            "min_validation_days": 30
        }
    )


class AgentLogsResult(BaseModel):
    """
    Result of get_agent_logs tool.

    Contains logs from the agent container.
    """
    agent_id: str = Field(
        description="Agent identifier"
    )
    logs: list[str] = Field(
        description="Log lines"
    )
    line_count: int = Field(
        description="Number of log lines returned",
        ge=0
    )
    tail_lines: int = Field(
        description="Number of lines requested",
        ge=0
    )
    retrieved_at: datetime = Field(
        description="When logs were retrieved",
        default_factory=lambda: datetime.now(UTC)
    )
    has_more: bool = Field(
        description="True if more logs available",
        default=False
    )

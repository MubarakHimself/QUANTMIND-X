"""
Performance models.

Contains models for strategy performance, paper trading, house money state, and strategy family tracking.
"""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Index, Enum, UniqueConstraint
from sqlalchemy.orm import relationship
from ..models.base import Base, TradingMode


class StrategyPerformance(Base):
    """
    Strategy performance tracking for backtests and live trading results.

    Attributes:
        id: Primary key
        strategy_name: Name of the strategy
        backtest_results: JSON containing backtest metrics
        kelly_score: Kelly criterion score
        sharpe_ratio: Sharpe ratio
        max_drawdown: Maximum drawdown percentage
        win_rate: Win rate percentage
        profit_factor: Profit factor
        total_trades: Total number of trades
        mode: Trading mode (demo or live)
        created_at: Record creation timestamp
    """
    __tablename__ = 'strategy_performance'

    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_name = Column(String(200), nullable=False, index=True)
    backtest_results = Column(String, nullable=False)  # JSON string
    kelly_score = Column(Float, nullable=False, index=True)
    sharpe_ratio = Column(Float, nullable=False)
    max_drawdown = Column(Float, nullable=False)
    win_rate = Column(Float, nullable=True)
    profit_factor = Column(Float, nullable=True)
    total_trades = Column(Integer, nullable=True)
    mode = Column(Enum(TradingMode), nullable=False, default=TradingMode.LIVE, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False, index=True)

    __table_args__ = (
        Index('ix_strategy_performance_kelly', 'kelly_score'),
        Index('ix_strategy_performance_sharpe', 'sharpe_ratio'),
    )

    def __repr__(self):
        return f"<StrategyPerformance(id={self.id}, strategy={self.strategy_name}, kelly={self.kelly_score:.2f}, sharpe={self.sharpe_ratio:.2f})>"


class PaperTradingPerformance(Base):
    """
    Paper trading agent performance tracking for validation and promotion.

    Stores historical performance metrics for paper trading agents,
    enabling trend analysis and validation tracking.

    Attributes:
        id: Primary key
        agent_id: Paper trading agent identifier
        timestamp: When metrics were calculated
        total_trades: Total number of trades executed
        winning_trades: Number of winning trades
        losing_trades: Number of losing trades
        win_rate: Win rate as decimal (0-1)
        total_pnl: Total profit/loss
        average_pnl: Average PnL per trade
        sharpe_ratio: Risk-adjusted return metric
        max_drawdown: Maximum equity drawdown
        profit_factor: Gross wins / gross losses
        validation_status: Current validation state (pending, validating, validated)
        days_validated: Days since deployment
        meets_criteria: Whether agent meets promotion criteria
        mode: Trading mode (demo or live - defaults to demo for paper trading)
    """
    __tablename__ = 'paper_trading_performance'

    id = Column(Integer, primary_key=True, autoincrement=True)
    agent_id = Column(String(100), nullable=False, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False, index=True)
    total_trades = Column(Integer, nullable=False, default=0)
    winning_trades = Column(Integer, nullable=False, default=0)
    losing_trades = Column(Integer, nullable=False, default=0)
    win_rate = Column(Float, nullable=False, default=0.0)
    total_pnl = Column(Float, nullable=False, default=0.0)
    average_pnl = Column(Float, nullable=False, default=0.0)
    sharpe_ratio = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=False, default=0.0)
    profit_factor = Column(Float, nullable=False, default=0.0)
    validation_status = Column(String(20), nullable=False, default='pending', index=True)
    days_validated = Column(Integer, nullable=False, default=0)
    meets_criteria = Column(Boolean, nullable=False, default=False)
    extra_data = Column(String, nullable=True)  # For additional metrics (JSON string)
    mode = Column(Enum(TradingMode), nullable=False, default=TradingMode.DEMO, index=True)

    __table_args__ = (
        Index('ix_paper_trading_agent_timestamp', 'agent_id', 'timestamp'),
        Index('ix_paper_trading_validation_status', 'validation_status'),
    )

    def __repr__(self):
        return f"<PaperTradingPerformance(id={self.id}, agent={self.agent_id}, status={self.validation_status}, sharpe={self.sharpe_ratio})>"

    def to_dict(self):
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "total_pnl": self.total_pnl,
            "average_pnl": self.average_pnl,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "profit_factor": self.profit_factor,
            "validation_status": self.validation_status,
            "days_validated": self.days_validated,
            "meets_criteria": self.meets_criteria,
            "extra_data": self.extra_data,
            "mode": self.mode.value if self.mode else None
        }


class HouseMoneyState(Base):
    """
    House Money State Tracking for dynamic risk adjustment.

    Tracks daily P&L and adjusts risk multiplier based on house money effect.
    Increases risk when trading with profits, decreases when down.

    Attributes:
        id: Primary key
        account_id: MT5 account number (string for flexibility)
        daily_start_balance: Balance at start of trading day
        current_pnl: Current daily profit/loss
        high_water_mark: Highest equity reached today
        risk_multiplier: Current risk multiplier (1.0=baseline, 1.5=up>5%, 0.5=down>3%)
        is_preservation_mode: Whether preservation mode is active
        date: Calendar date (YYYY-MM-DD format)
        created_at: Record creation timestamp
        updated_at: Last update timestamp
    """
    __tablename__ = 'house_money_state'

    id = Column(Integer, primary_key=True, autoincrement=True)
    account_id = Column(String(50), nullable=False, index=True)
    daily_start_balance = Column(Float, nullable=False, default=0.0)
    current_pnl = Column(Float, nullable=False, default=0.0)
    high_water_mark = Column(Float, nullable=False, default=0.0)
    risk_multiplier = Column(Float, nullable=False, default=1.0)
    is_preservation_mode = Column(Boolean, nullable=False, default=False)
    date = Column(String(10), nullable=False)  # YYYY-MM-DD format
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)

    __table_args__ = (
        UniqueConstraint('account_id', 'date', name='uq_house_money_account_date'),
        Index('idx_house_money_date', 'date'),
    )

    def __repr__(self):
        return f"<HouseMoneyState(id={self.id}, account={self.account_id}, date={self.date}, pnl={self.current_pnl}, multiplier={self.risk_multiplier})>"


class StrategyFamilyState(Base):
    """
    Strategy Family State for Tier 2 protection.

    Tracks the state of each strategy family for quarantine management.

    Attributes:
        id: Primary key
        family: Strategy type (SCALPER, STRUCTURAL, SWING, HFT)
        failed_bots: JSON list of failed bot IDs
        total_pnl: Combined P&L for the family
        initial_capital: Starting capital allocation
        is_quarantined: Whether family is quarantined
        quarantine_time: When quarantine started
        quarantine_reason: Why family was quarantined
        created_at: Record creation timestamp
        updated_at: Last update timestamp
    """
    __tablename__ = 'strategy_family_states'

    id = Column(Integer, primary_key=True, autoincrement=True)
    family = Column(String(20), nullable=False, unique=True, index=True)
    failed_bots = Column(String, nullable=False, default='[]')  # JSON string
    total_pnl = Column(Float, nullable=False, default=0.0)
    initial_capital = Column(Float, nullable=False, default=10000.0)
    is_quarantined = Column(Boolean, nullable=False, default=False, index=True)
    quarantine_time = Column(DateTime, nullable=True)
    quarantine_reason = Column(String(200), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)

    __table_args__ = (
        Index('idx_strategy_family_quarantined', 'is_quarantined'),
    )

    def __repr__(self):
        return f"<StrategyFamilyState(id={self.id}, family={self.family}, quarantined={self.is_quarantined})>"

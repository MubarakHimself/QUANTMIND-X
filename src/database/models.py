"""
SQLAlchemy Models for QuantMind Hybrid Core

Defines the database schema for prop firm account management,
daily snapshots, trade proposals, agent tasks, and strategy performance.
"""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime,
    ForeignKey, UniqueConstraint, Index, JSON, Text
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class PropFirmAccount(Base):
    """
    Represents a prop firm trading account with risk parameters.

    Attributes:
        id: Primary key
        firm_name: Name of the prop firm (e.g., "MyForexFunds")
        account_id: MT5 account number (string for flexibility)
        daily_loss_limit_pct: Maximum daily loss as percentage (e.g., 5.0 = 5%)
        hard_stop_buffer_pct: Safety buffer below daily limit (e.g., 1.0 = 1%)
        target_profit_pct: Profit target to trigger preservation mode (e.g., 8.0 = 8%)
        min_trading_days: Minimum trading days required (e.g., 5 days)
        created_at: Account creation timestamp
        updated_at: Last update timestamp
    """
    __tablename__ = 'prop_firm_accounts'

    id = Column(Integer, primary_key=True, autoincrement=True)
    firm_name = Column(String(100), nullable=False)
    account_id = Column(String(50), nullable=False, unique=True, index=True)
    daily_loss_limit_pct = Column(Float, nullable=False, default=5.0)
    hard_stop_buffer_pct = Column(Float, nullable=False, default=1.0)
    target_profit_pct = Column(Float, nullable=False, default=8.0)
    min_trading_days = Column(Integer, nullable=False, default=5)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    daily_snapshots = relationship("DailySnapshot", back_populates="prop_account", cascade="all, delete-orphan")
    trade_proposals = relationship("TradeProposal", back_populates="prop_account")

    def __repr__(self):
        return f"<PropFirmAccount(id={self.id}, firm={self.firm_name}, account={self.account_id})>"


class DailySnapshot(Base):
    """
    Daily account state snapshot for tracking drawdown and high water marks.

    Attributes:
        id: Primary key
        account_id: Foreign key to PropFirmAccount
        date: Calendar date (YYYY-MM-DD format)
        daily_start_balance: Balance at start of trading day
        high_water_mark: Highest equity reached during the day
        current_equity: Current equity value
        daily_drawdown_pct: Drawdown percentage from daily start
        is_breached: Whether daily loss limit was breached
        snapshot_timestamp: When snapshot was recorded
    """
    __tablename__ = 'daily_snapshots'

    id = Column(Integer, primary_key=True, autoincrement=True)
    account_id = Column(Integer, ForeignKey('prop_firm_accounts.id', ondelete='CASCADE'), nullable=False, index=True)
    date = Column(String(10), nullable=False)  # YYYY-MM-DD format
    daily_start_balance = Column(Float, nullable=False, default=0.0)
    high_water_mark = Column(Float, nullable=False, default=0.0)
    current_equity = Column(Float, nullable=False, default=0.0)
    daily_drawdown_pct = Column(Float, nullable=False, default=0.0)
    is_breached = Column(Boolean, nullable=False, default=False)
    snapshot_timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationship to account
    prop_account = relationship("PropFirmAccount", back_populates="daily_snapshots")

    # Unique constraint to prevent duplicate snapshots per account per day
    __table_args__ = (
        UniqueConstraint('account_id', 'date', name='uq_account_date'),
        Index('idx_snapshot_date', 'date'),
    )

    def __repr__(self):
        return f"<DailySnapshot(id={self.id}, account_id={self.account_id}, date={self.date}, equity={self.current_equity})>"


class TradeProposal(Base):
    """
    Trade proposal from bots for analyst/quant review and approval.

    Attributes:
        id: Primary key
        account_id: Foreign key to PropFirmAccount (optional for standalone proposals)
        bot_id: Identifier of the proposing bot/strategy
        symbol: Trading symbol (e.g., "EURUSD")
        kelly_score: Kelly criterion score for the trade
        regime: Market regime classification
        proposed_lot_size: Suggested position size
        status: Proposal status (pending/approved/rejected)
        created_at: Proposal creation timestamp
        reviewed_at: When proposal was reviewed
    """
    __tablename__ = 'trade_proposals'

    id = Column(Integer, primary_key=True, autoincrement=True)
    account_id = Column(Integer, ForeignKey('prop_firm_accounts.id', ondelete='SET NULL'), nullable=True, index=True)
    bot_id = Column(String(100), nullable=False, index=True)
    symbol = Column(String(20), nullable=False)
    kelly_score = Column(Float, nullable=False)
    regime = Column(String(50))
    proposed_lot_size = Column(Float, nullable=False)
    status = Column(String(20), server_default='pending', nullable=False)  # pending, approved, rejected
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    reviewed_at = Column(DateTime, nullable=True)

    # Relationship to account
    prop_account = relationship("PropFirmAccount", back_populates="trade_proposals")

    __table_args__ = (
        Index('ix_trade_proposals_status', 'status'),
    )

    def __repr__(self):
        return f"<TradeProposal(id={self.id}, bot={self.bot_id}, symbol={self.symbol}, status={self.status})>"


class AgentTasks(Base):
    """
    Agent task history for tracking agent operations and coordination.

    Attributes:
        id: Primary key
        agent_type: Type of agent (analyst/quant/copilot)
        task_type: Type of task being performed
        task_data: JSON data containing task details
        status: Task status (pending/in_progress/completed/failed)
        created_at: Task creation timestamp
        completed_at: Task completion timestamp
    """
    __tablename__ = 'agent_tasks'

    id = Column(Integer, primary_key=True, autoincrement=True)
    agent_type = Column(String(50), nullable=False, index=True)
    task_type = Column(String(100), nullable=False, index=True)
    task_data = Column(JSON, nullable=False)
    status = Column(String(20), server_default='pending', nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    completed_at = Column(DateTime, nullable=True)

    __table_args__ = (
        Index('ix_agent_tasks_agent_status', 'agent_type', 'status'),
    )

    def __repr__(self):
        return f"<AgentTasks(id={self.id}, agent={self.agent_type}, type={self.task_type}, status={self.status})>"


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
        created_at: Record creation timestamp
    """
    __tablename__ = 'strategy_performance'

    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_name = Column(String(200), nullable=False, index=True)
    backtest_results = Column(JSON, nullable=False)
    kelly_score = Column(Float, nullable=False, index=True)
    sharpe_ratio = Column(Float, nullable=False)
    max_drawdown = Column(Float, nullable=False)
    win_rate = Column(Float, nullable=True)
    profit_factor = Column(Float, nullable=True)
    total_trades = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    __table_args__ = (
        Index('ix_strategy_performance_kelly', 'kelly_score'),
        Index('ix_strategy_performance_sharpe', 'sharpe_ratio'),
    )

    def __repr__(self):
        return f"<StrategyPerformance(id={self.id}, strategy={self.strategy_name}, kelly={self.kelly_score:.2f}, sharpe={self.sharpe_ratio:.2f})>"

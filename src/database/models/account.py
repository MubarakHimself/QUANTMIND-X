"""
Account models.

Contains models for prop firm accounts, snapshots, broker registry, and loss tracking.
"""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, UniqueConstraint, Index, Enum
from sqlalchemy.orm import relationship
from ..models.base import Base, TradingMode


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
        risk_mode: V8 Tiered Risk Engine mode ('growth', 'scaling', 'guardian')
        mode: Trading mode (demo or live)
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
    risk_mode = Column(String(20), nullable=False, default='growth', index=True)  # V8: 'growth', 'scaling', 'guardian'
    mode = Column(Enum(TradingMode), nullable=False, default=TradingMode.LIVE, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)

    # Relationships
    daily_snapshots = relationship("DailySnapshot", back_populates="prop_account", cascade="all, delete-orphan")
    trade_proposals = relationship("TradeProposal", back_populates="prop_account")
    tier_transitions = relationship("RiskTierTransition", back_populates="prop_account", cascade="all, delete-orphan")  # V8

    def __repr__(self):
        return f"<PropFirmAccount(id={self.id}, firm={self.firm_name}, account={self.account_id}, risk_mode={self.risk_mode})>"


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
        mode: Trading mode (demo or live)
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
    snapshot_timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    mode = Column(Enum(TradingMode), nullable=False, default=TradingMode.LIVE, index=True)

    # Relationship to account
    prop_account = relationship("PropFirmAccount", back_populates="daily_snapshots")

    # Unique constraint to prevent duplicate snapshots per account per day
    __table_args__ = (
        UniqueConstraint('account_id', 'date', name='uq_account_date'),
        Index('idx_snapshot_date', 'date'),
    )

    def __repr__(self):
        return f"<DailySnapshot(id={self.id}, account_id={self.account_id}, date={self.date}, equity={self.current_equity})>"


class BrokerRegistry(Base):
    """
    Broker Registry for fee structures and trading parameters.

    Stores broker profiles with spread, commission, lot sizes, and pip values.
    Used for fee-aware position sizing and dynamic pip value calculation.

    Attributes:
        id: Primary key
        broker_id: Unique broker identifier (e.g., "icmarkets_raw")
        broker_name: Human-readable broker name
        spread_avg: Average spread in points
        commission_per_lot: Commission per standard lot
        lot_step: Minimum lot step increment
        min_lot: Minimum lot size
        max_lot: Maximum lot size
        pip_values: JSON dict mapping symbols to pip values
        preference_tags: List of broker tags (RAW_ECN, STANDARD, etc.)
        created_at: Record creation timestamp
        updated_at: Last update timestamp
    """
    __tablename__ = 'broker_registry'

    id = Column(Integer, primary_key=True, autoincrement=True)
    broker_id = Column(String(100), nullable=False, unique=True, index=True)
    broker_name = Column(String(200), nullable=False)
    spread_avg = Column(Float, nullable=False, default=0.0)
    commission_per_lot = Column(Float, nullable=False, default=0.0)
    lot_step = Column(Float, nullable=False, default=0.01)
    min_lot = Column(Float, nullable=False, default=0.01)
    max_lot = Column(Float, nullable=False, default=100.0)
    pip_values = Column(String, nullable=False, default='{}')  # {"EURUSD": 10.0, "XAUUSD": 1.0}
    preference_tags = Column(String, nullable=False, default='[]')  # ["RAW_ECN", "LOW_SPREAD"]
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)

    __table_args__ = (
        Index('idx_broker_registry_id', 'broker_id'),
    )

    def __repr__(self):
        return f"<BrokerRegistry(id={self.id}, broker_id={self.broker_id}, name={self.broker_name})>"


class AccountLossState(Base):
    """
    Account Loss State for Tier 3 protection.

    Tracks daily and weekly P&L for account-level loss limits.

    Attributes:
        id: Primary key
        account_id: Account identifier
        daily_pnl: Today's profit/loss
        weekly_pnl: This week's profit/loss
        last_reset_date: When daily counters were last reset
        week_start: Start of the current week
        daily_stop_triggered: Whether daily stop is active
        weekly_stop_triggered: Whether weekly stop is active
        created_at: Record creation timestamp
        updated_at: Last update timestamp
    """
    __tablename__ = 'account_loss_states'

    id = Column(Integer, primary_key=True, autoincrement=True)
    account_id = Column(String(50), nullable=False, unique=True, index=True)
    initial_balance = Column(Float, nullable=False, default=10000.0)
    daily_pnl = Column(Float, nullable=False, default=0.0)
    weekly_pnl = Column(Float, nullable=False, default=0.0)
    last_reset_date = Column(String(10), nullable=True)  # YYYY-MM-DD
    week_start = Column(String(10), nullable=True)  # YYYY-MM-DD
    daily_stop_triggered = Column(Boolean, nullable=False, default=False)
    weekly_stop_triggered = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)

    __table_args__ = (
        Index('idx_account_loss_stops', 'daily_stop_triggered', 'weekly_stop_triggered'),
    )

    def __repr__(self):
        return f"<AccountLossState(id={self.id}, account={self.account_id}, daily_stop={self.daily_stop_triggered})>"

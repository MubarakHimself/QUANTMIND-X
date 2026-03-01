"""
SQLAlchemy Models for QuantMind Hybrid Core

Defines the database schema for prop firm account management,
daily snapshots, trade proposals, agent tasks, and strategy performance.
"""

from datetime import datetime, timezone
from enum import Enum as PyEnum

from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime,
    ForeignKey, UniqueConstraint, Index, JSON, Text, Enum
)
from sqlalchemy.orm import declarative_base, relationship, Session
from typing import Generator

# Import session from engine for FastAPI dependency injection
from .engine import get_session

Base = declarative_base()


def get_db_session() -> Generator[Session, None, None]:
    """
    FastAPI dependency for database session management.

    Yields a database session and ensures cleanup after request.

    Usage:
        @router.get("/items")
        async def get_items(db: Session = Depends(get_db_session)):
            items = db.query(Item).all()
            return items
    """
    session = get_session()
    try:
        yield session
    finally:
        session.close()


class TradingMode(PyEnum):
    """Trading mode enum for demo/live distinction."""
    DEMO = "demo"
    LIVE = "live"


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
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False, index=True)
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
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False, index=True)
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
        mode: Trading mode (demo or live)
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
    extra_data = Column(JSON, nullable=True)  # For additional metrics
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


class RiskTierTransition(Base):
    """
    V8 Tiered Risk Engine: Tracks risk tier transitions for audit purposes.

    Attributes:
        id: Primary key
        account_id: Foreign key to PropFirmAccount
        from_tier: Previous risk tier ('growth', 'scaling', 'guardian')
        to_tier: New risk tier ('growth', 'scaling', 'guardian')
        equity_at_transition: Account equity when transition occurred
        transition_timestamp: When the transition happened
    """
    __tablename__ = 'risk_tier_transitions'

    id = Column(Integer, primary_key=True, autoincrement=True)
    account_id = Column(Integer, ForeignKey('prop_firm_accounts.id', ondelete='CASCADE'), nullable=False, index=True)
    from_tier = Column(String(20), nullable=False)
    to_tier = Column(String(20), nullable=False)
    equity_at_transition = Column(Float, nullable=False)
    transition_timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False, index=True)

    # Relationship to account
    prop_account = relationship("PropFirmAccount", back_populates="tier_transitions")

    __table_args__ = (
        Index('idx_tier_transition_account_timestamp', 'account_id', 'transition_timestamp'),
    )

    def __repr__(self):
        return f"<RiskTierTransition(id={self.id}, account_id={self.account_id}, {self.from_tier}->{self.to_tier}, equity={self.equity_at_transition})>"


class CryptoTrade(Base):
    """
    V8 Crypto Module: Tracks crypto trades from Binance and other exchanges.
    
    Uses same structure as MT5 trades for unified reporting and analysis.
    
    Attributes:
        id: Primary key
        broker_type: Broker type ('binance_spot', 'binance_futures', 'mt5')
        broker_id: Broker identifier from registry
        order_id: Exchange order ID
        symbol: Trading symbol (e.g., "BTCUSDT", "ETHUSDT")
        direction: Trade direction ('buy' or 'sell')
        volume: Position size (quantity for crypto, lots for MT5)
        entry_price: Entry price
        exit_price: Exit price (null if still open)
        stop_loss: Stop loss price (optional)
        take_profit: Take profit price (optional)
        profit: Realized profit/loss (null if still open)
        status: Trade status ('open', 'closed', 'cancelled')
        mode: Trading mode (demo or live)
        open_timestamp: When trade was opened
        close_timestamp: When trade was closed (null if still open)
        metadata: JSON field for additional trade data (fees, slippage, etc.)
    """
    __tablename__ = 'crypto_trades'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    broker_type = Column(String(50), nullable=False, index=True)  # 'binance_spot', 'binance_futures', 'mt5'
    broker_id = Column(String(100), nullable=False, index=True)  # From broker registry
    order_id = Column(String(100), nullable=False, unique=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    direction = Column(String(10), nullable=False)  # 'buy' or 'sell'
    volume = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=True)
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)
    profit = Column(Float, nullable=True)
    status = Column(String(20), nullable=False, default='open', index=True)  # 'open', 'closed', 'cancelled'
    mode = Column(Enum(TradingMode), nullable=False, default=TradingMode.LIVE, index=True)
    open_timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False, index=True)
    close_timestamp = Column(DateTime, nullable=True)
    trade_metadata = Column(JSON, nullable=True)  # Fees, slippage, shadow stops, etc. (renamed from 'metadata' to avoid SQLAlchemy conflict)
    
    __table_args__ = (
        Index('idx_crypto_trades_broker_symbol', 'broker_id', 'symbol'),
        Index('idx_crypto_trades_status_timestamp', 'status', 'open_timestamp'),
    )
    
    def __repr__(self):
        return f"<CryptoTrade(id={self.id}, broker={self.broker_type}, symbol={self.symbol}, direction={self.direction}, status={self.status})>"


class StrategyFolder(Base):
    """
    Strategy Folder for linking NPRD → TRD → EA.

    Tracks the complete strategy development pipeline from Non-Processing
    Research Data (NPRD) through Technical Requirements Document (TRD) to
    Expert Advisor (EA) code.

    Attributes:
        id: Primary key
        folder_name: Unique folder name (e.g., "ict_orderblock")
        description: Strategy description
        nprd_path: Path to NPRD document
        trd_vanilla_path: Path to vanilla TRD document
        trd_enhanced_path: Path to enhanced TRD document
        ea_vanilla_path: Path to vanilla EA code
        ea_enhanced_path: Path to enhanced EA code
        preferred_conditions: JSON with trading preferences
        status: Folder status (draft, backtested, approved, deployed)
        created_at: Record creation timestamp
        updated_at: Last update timestamp
    """
    __tablename__ = 'strategy_folders'

    id = Column(Integer, primary_key=True, autoincrement=True)
    folder_name = Column(String(200), nullable=False, unique=True, index=True)
    description = Column(Text, nullable=True)
    nprd_path = Column(String(500), nullable=True)
    trd_vanilla_path = Column(String(500), nullable=True)
    trd_enhanced_path = Column(String(500), nullable=True)
    ea_vanilla_path = Column(String(500), nullable=True)
    ea_enhanced_path = Column(String(500), nullable=True)
    preferred_conditions = Column(JSON, nullable=True)  # sessions, timeframes, instruments, regimes
    status = Column(String(50), nullable=False, default='draft', index=True)  # draft, backtested, approved, deployed
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)

    __table_args__ = (
        Index('idx_strategy_folders_name', 'folder_name'),
        Index('idx_strategy_folders_status', 'status'),
    )

    def __repr__(self):
        return f"<StrategyFolder(id={self.id}, name={self.folder_name}, status={self.status})>"


class SharedAsset(Base):
    """
    Shared Asset Library for reusable code components.

    Stores shared indicators, libraries, and utility code that can be
    integrated into multiple EAs.

    Attributes:
        id: Primary key
        asset_name: Unique asset name (e.g., "adaptive_rsi")
        asset_type: Type of asset (indicator, library, utility)
        language: Programming language (mql5, python, etc.)
        file_path: Path to asset file
        description: Asset description
        parameters: JSON list of configurable parameters
        dependencies: JSON list of required dependencies
        version: Asset version
        is_active: Whether asset is currently usable
        created_at: Record creation timestamp
        updated_at: Last update timestamp
    """
    __tablename__ = 'shared_assets'

    id = Column(Integer, primary_key=True, autoincrement=True)
    asset_name = Column(String(200), nullable=False, unique=True, index=True)
    asset_type = Column(String(50), nullable=False, index=True)  # indicator, library, utility
    language = Column(String(20), nullable=False, default='mql5')
    file_path = Column(String(500), nullable=False)
    description = Column(Text, nullable=True)
    parameters = Column(JSON, nullable=True)  # [{"name": "period", "type": "int", "default": 14}]
    dependencies = Column(JSON, nullable=True)  # [{"asset": "indicator", "version": "1.0"}]
    version = Column(String(20), nullable=False, default='1.0')
    is_active = Column(Boolean, nullable=False, default=True, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)

    __table_args__ = (
        Index('idx_shared_assets_name', 'asset_name'),
        Index('idx_shared_assets_type', 'asset_type'),
        Index('idx_shared_assets_active', 'is_active'),
    )

    def __repr__(self):
        return f"<SharedAsset(id={self.id}, name={self.asset_name}, type={self.asset_type}, lang={self.language})>"


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
    pip_values = Column(JSON, nullable=False, default=dict)  # {"EURUSD": 10.0, "XAUUSD": 1.0}
    preference_tags = Column(JSON, nullable=False, default=list)  # ["RAW_ECN", "LOW_SPREAD"]
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)

    __table_args__ = (
        Index('idx_broker_registry_id', 'broker_id'),
    )

    def __repr__(self):
        return f"<BrokerRegistry(id={self.id}, broker_id={self.broker_id}, name={self.broker_name})>"


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


class BotCircuitBreaker(Base):
    """
    Bot Circuit Breaker for automatic quarantine on poor performance.

    Tracks bot-level performance and auto-quarantines underperforming bots.
    Prevents catastrophic losses from malfunctioning strategies.

    Attributes:
        id: Primary key
        bot_id: Unique bot identifier
        consecutive_losses: Number of consecutive losses
        daily_trade_count: Number of trades today
        last_trade_time: Timestamp of last trade
        is_quarantined: Whether bot is currently quarantined
        quarantine_reason: Reason for quarantine
        quarantine_start: When quarantine started
        mode: Trading mode (demo or live)
        created_at: Record creation timestamp
        updated_at: Last update timestamp
    """
    __tablename__ = 'bot_circuit_breaker'

    id = Column(Integer, primary_key=True, autoincrement=True)
    bot_id = Column(String(100), nullable=False, unique=True, index=True)
    consecutive_losses = Column(Integer, nullable=False, default=0)
    daily_trade_count = Column(Integer, nullable=False, default=0)
    last_trade_time = Column(DateTime, nullable=True)
    is_quarantined = Column(Boolean, nullable=False, default=False, index=True)
    quarantine_reason = Column(String(200), nullable=True)
    quarantine_start = Column(DateTime, nullable=True)
    mode = Column(Enum(TradingMode), nullable=False, default=TradingMode.LIVE, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)

    __table_args__ = (
        Index('idx_bot_circuit_breaker_quarantined', 'is_quarantined'),
    )


    def __repr__(self):
        return f"<BotCircuitBreaker(id={self.id}, bot={self.bot_id}, quarantined={self.is_quarantined}, losses={self.consecutive_losses})>"


class TradeJournal(Base):
    """
    Trade Journal for comprehensive trade context logging.
    
    Stores the "Why?" for every trade: regime, governor state, Kelly recommendation,
    balance zone, house money adjustment, and execution details.
    
    Attributes:
        id: Primary key
        timestamp: When the trade was executed
        symbol: Trading symbol
        bot_id: Which bot executed the trade
        direction: BUY or SELL
        regime: Market regime from Sentinel
        chaos_score: Chaos score from regime detection
        governor_throttle: Governor throttle applied
        balance_zone: DANGER, GROWTH, SCALING, or GUARDIAN
        kelly_recommendation: Kelly calculator recommendation
        actual_risk: Actual risk taken (USD)
        house_money_adjustment: Whether house money adjustment was applied
        mode: Trading mode (demo or live)
        broker: Broker used for execution
        spread_at_entry: Spread at entry in pips
        commission: Commission paid
        pnl: Profit/Loss (filled after close)
        duration_minutes: Trade duration in minutes
        exit_reason: Why the trade was closed
        strategy_folder_id: Link to strategy folder
        created_at: Record creation timestamp
    """
    __tablename__ = 'trade_journal'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    bot_id = Column(String(100), nullable=False, index=True)
    direction = Column(String(10), nullable=False)  # BUY or SELL
    
    # The "Why?" Context
    regime = Column(String(50), nullable=True)
    chaos_score = Column(Float, nullable=True)
    governor_throttle = Column(Float, nullable=True)
    balance_zone = Column(String(20), nullable=True, index=True)
    kelly_recommendation = Column(Float, nullable=True)
    actual_risk = Column(Float, nullable=False)
    house_money_adjustment = Column(Boolean, nullable=False, default=False)
    mode = Column(Enum(TradingMode), nullable=False, default=TradingMode.LIVE, index=True)
    
    # Execution Details
    broker = Column(String(100), nullable=True)
    spread_at_entry = Column(Float, nullable=True)
    commission = Column(Float, nullable=True, default=0.0)
    entry_price = Column(Float, nullable=True)
    exit_price = Column(Float, nullable=True)
    lot_size = Column(Float, nullable=False)
    
    # Outcome
    pnl = Column(Float, nullable=True)
    duration_minutes = Column(Integer, nullable=True)
    exit_reason = Column(String(100), nullable=True)
    
    # Link to strategy
    strategy_folder_id = Column(Integer, ForeignKey('strategy_folders.id'), nullable=True, index=True)
    
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    
    __table_args__ = (
        Index('idx_trade_journal_timestamp', 'timestamp'),
        Index('idx_trade_journal_bot', 'bot_id'),
        Index('idx_trade_journal_symbol', 'symbol'),
        Index('idx_trade_journal_zone', 'balance_zone'),
    )
    
    def __repr__(self):
        return f"<TradeJournal(id={self.id}, bot={self.bot_id}, symbol={self.symbol}, pnl={self.pnl}, zone={self.balance_zone})>"


class BotCloneHistory(Base):
    __tablename__ = 'bot_clone_history'
    
    id = Column(Integer, primary_key=True)
    original_bot_id = Column(String(100), nullable=False, index=True)
    clone_bot_id = Column(String(100), nullable=False, index=True)
    original_symbol = Column(String(20), nullable=False)
    clone_symbol = Column(String(20), nullable=False)
    performance_at_clone = Column(JSON, nullable=False)  # Sharpe, win rate, etc.
    allocation_strategy = Column(String(50), nullable=False)  # 'adaptive' or 'equal'
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class DailyFeeTracking(Base):
    """
    Daily Fee Tracking for monitoring trading costs and fee burn.

    Tracks daily fees, trade counts, and fee burn percentage to trigger
    kill switch when fees exceed acceptable thresholds.

    Attributes:
        id: Primary key
        account_id: Account identifier for fee tracking
        date: Calendar date (YYYY-MM-DD format)
        total_fees: Total fees incurred for the day
        total_trades: Number of trades executed
        fee_burn_pct: Fee burn percentage (fees/balance * 100)
        account_balance: Account balance at time of tracking
        kill_switch_activated: Whether kill switch was triggered
        created_at: Record creation timestamp
    """
    __tablename__ = 'daily_fee_tracking'

    id = Column(Integer, primary_key=True, autoincrement=True)
    account_id = Column(String(50), nullable=False, index=True)
    date = Column(String(10), nullable=False)  # YYYY-MM-DD format
    total_fees = Column(Float, nullable=False, default=0.0)
    total_trades = Column(Integer, nullable=False, default=0)
    fee_burn_pct = Column(Float, nullable=False, default=0.0)
    account_balance = Column(Float, nullable=False, default=0.0)
    kill_switch_activated = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    __table_args__ = (
        UniqueConstraint('account_id', 'date', name='uq_daily_fee_account_date'),
        Index('idx_daily_fee_date', 'date'),
    )

    def __repr__(self):
        return f"<DailyFeeTracking(id={self.id}, account_id={self.account_id}, date={self.date}, fees={self.total_fees}, trades={self.total_trades})>"


# =============================================================================
# Progressive Kill Switch Models (Phase 9)
# =============================================================================

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
    failed_bots = Column(JSON, nullable=False, default=list)
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


class AlertHistory(Base):
    """
    Alert History for Progressive Kill Switch System.

    Stores all raised alerts for audit and analysis.

    Attributes:
        id: Primary key
        level: Alert level (GREEN, YELLOW, ORANGE, RED, BLACK)
        tier: Protection tier (1-5)
        message: Alert description
        threshold_pct: Threshold percentage when raised
        triggered_at: When alert was raised
        source: Component that raised the alert
        alert_metadata: JSON with additional context
        is_active: Whether alert is still active
        cleared_at: When alert was cleared
        created_at: Record creation timestamp
    """
    __tablename__ = 'alert_history'

    id = Column(Integer, primary_key=True, autoincrement=True)
    level = Column(String(10), nullable=False, index=True)
    tier = Column(Integer, nullable=False, index=True)
    message = Column(Text, nullable=False)
    threshold_pct = Column(Float, nullable=False)
    triggered_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False, index=True)
    source = Column(String(50), nullable=False, index=True)
    alert_metadata = Column(JSON, nullable=True)  # Renamed from 'metadata' to avoid SQLAlchemy conflict
    is_active = Column(Boolean, nullable=False, default=True, index=True)
    cleared_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    __table_args__ = (
        Index('idx_alert_history_level_tier', 'level', 'tier'),
        Index('idx_alert_history_triggered', 'triggered_at'),
    )

    def __repr__(self):
        return f"<AlertHistory(id={self.id}, level={self.level}, tier={self.tier}, source={self.source})>"


# =============================================================================
# HMM Regime Detection Models
# =============================================================================

class HMMModel(Base):
    """
    HMM Model for Regime Detection.

    Stores trained Hidden Markov Model metadata for regime detection.
    Models are trained on Contabo (training server) and synced to Cloudzy.

    Attributes:
        id: Primary key
        version: Model version string (e.g., "1.0.0", "1.1.0")
        model_type: Training hierarchy level ('universal', 'per_symbol', 'per_symbol_timeframe')
        symbol: Trading symbol (e.g., "EURUSD", "GBPUSD") - null for universal
        timeframe: Timeframe (e.g., "M5", "H1", "H4") - null for universal/per_symbol
        n_states: Number of hidden states (typically 4)
        log_likelihood: Training log-likelihood score
        state_distribution: JSON with state distribution percentages
        transition_matrix: JSON with transition probability matrix
        training_samples: Number of samples used for training
        training_date: When model was trained
        checksum: SHA256 checksum of model file for integrity
        file_path: Path to .pkl model file
        is_active: Whether model is currently active
        validation_status: Validation status (pending, validated, rejected)
        validation_notes: Notes from validation process
        created_at: Record creation timestamp
    """
    __tablename__ = 'hmm_models'

    id = Column(Integer, primary_key=True, autoincrement=True)
    version = Column(String(20), nullable=False, index=True)
    model_type = Column(String(30), nullable=False, index=True)
    symbol = Column(String(20), nullable=True, index=True)
    timeframe = Column(String(10), nullable=True, index=True)
    n_states = Column(Integer, nullable=False, default=4)
    log_likelihood = Column(Float, nullable=True)
    state_distribution = Column(JSON, nullable=True)  # {"state_0": 0.25, "state_1": 0.28, ...}
    transition_matrix = Column(JSON, nullable=True)  # [[0.85, 0.10, ...], ...]
    training_samples = Column(Integer, nullable=False, default=0)
    training_date = Column(DateTime, nullable=True)
    checksum = Column(String(64), nullable=True)
    file_path = Column(String(500), nullable=True)
    is_active = Column(Boolean, nullable=False, default=True, index=True)
    validation_status = Column(String(20), nullable=False, default='pending', index=True)
    validation_notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    # Relationships
    deployments = relationship("HMMDeployment", back_populates="model", cascade="all, delete-orphan")
    shadow_logs = relationship("HMMShadowLog", back_populates="model")

    __table_args__ = (
        Index('idx_hmm_models_version', 'version'),
        Index('idx_hmm_models_type_symbol', 'model_type', 'symbol'),
        Index('idx_hmm_models_active', 'is_active'),
        UniqueConstraint('version', 'symbol', 'timeframe', name='uq_hmm_model_version_symbol_tf'),
    )

    def __repr__(self):
        return f"<HMMModel(id={self.id}, version={self.version}, type={self.model_type}, symbol={self.symbol}, tf={self.timeframe})>"

    def to_dict(self):
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "version": self.version,
            "model_type": self.model_type,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "n_states": self.n_states,
            "log_likelihood": self.log_likelihood,
            "state_distribution": self.state_distribution,
            "transition_matrix": self.transition_matrix,
            "training_samples": self.training_samples,
            "training_date": self.training_date.isoformat() if self.training_date else None,
            "checksum": self.checksum,
            "file_path": self.file_path,
            "is_active": self.is_active,
            "validation_status": self.validation_status,
            "validation_notes": self.validation_notes,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class HMMShadowLog(Base):
    """
    HMM Shadow Mode Log for prediction comparison.

    Records HMM vs Ising predictions during shadow mode for validation
    and performance comparison.

    Attributes:
        id: Primary key
        model_id: Foreign key to HMMModel
        timestamp: When the prediction was made
        symbol: Trading symbol
        timeframe: Timeframe
        ising_regime: Ising model regime prediction
        ising_confidence: Ising model confidence (0-1)
        hmm_regime: HMM regime prediction
        hmm_state: HMM state ID (0-3)
        hmm_confidence: HMM confidence (0-1)
        agreement: Whether Ising and HMM agree
        decision_source: Which model was used for final decision
        market_context: JSON with market data at prediction time
        created_at: Record creation timestamp
    """
    __tablename__ = 'hmm_shadow_logs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(Integer, ForeignKey('hmm_models.id'), nullable=True, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False, index=True)
    ising_regime = Column(String(50), nullable=False)
    ising_confidence = Column(Float, nullable=False, default=0.0)
    hmm_regime = Column(String(50), nullable=False)
    hmm_state = Column(Integer, nullable=False)
    hmm_confidence = Column(Float, nullable=False, default=0.0)
    agreement = Column(Boolean, nullable=False, default=False)
    decision_source = Column(String(20), nullable=False, default='ising')  # 'ising', 'hmm', 'weighted'
    market_context = Column(JSON, nullable=True)  # volatility, price, etc.
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    # Relationship
    model = relationship("HMMModel", back_populates="shadow_logs")

    __table_args__ = (
        Index('idx_hmm_shadow_timestamp', 'timestamp'),
        Index('idx_hmm_shadow_symbol_tf', 'symbol', 'timeframe'),
        Index('idx_hmm_shadow_agreement', 'agreement'),
    )

    def __repr__(self):
        return f"<HMMShadowLog(id={self.id}, symbol={self.symbol}, ising={self.ising_regime}, hmm={self.hmm_regime}, agree={self.agreement})>"

    def to_dict(self):
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "model_id": self.model_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "ising_regime": self.ising_regime,
            "ising_confidence": self.ising_confidence,
            "hmm_regime": self.hmm_regime,
            "hmm_state": self.hmm_state,
            "hmm_confidence": self.hmm_confidence,
            "agreement": self.agreement,
            "decision_source": self.decision_source,
            "market_context": self.market_context,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class HMMDeployment(Base):
    """
    HMM Deployment State for tracking deployment history.

    Tracks the state transitions and deployment history for HMM models
    from shadow mode through hybrid to production.

    Attributes:
        id: Primary key
        model_id: Foreign key to HMMModel
        mode: Current deployment mode ('ising_only', 'hmm_shadow', 'hmm_hybrid_20', 'hmm_hybrid_50', 'hmm_hybrid_80', 'hmm_only')
        previous_mode: Previous deployment mode
        transition_date: When the mode transition occurred
        approved_by: Who approved the transition (user ID or 'auto')
        approval_token: Token used for approval
        performance_metrics: JSON with performance metrics at transition
        rollback_count: Number of rollbacks for this deployment
        is_active: Whether this is the current active deployment
        notes: Deployment notes
        created_at: Record creation timestamp
    """
    __tablename__ = 'hmm_deployments'

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(Integer, ForeignKey('hmm_models.id'), nullable=False, index=True)
    mode = Column(String(20), nullable=False, index=True)
    previous_mode = Column(String(20), nullable=True)
    transition_date = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    approved_by = Column(String(100), nullable=True)
    approval_token = Column(String(64), nullable=True)
    performance_metrics = Column(JSON, nullable=True)  # agreement_pct, sharpe, pnl, etc.
    rollback_count = Column(Integer, nullable=False, default=0)
    is_active = Column(Boolean, nullable=False, default=True, index=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    # Relationship
    model = relationship("HMMModel", back_populates="deployments")

    __table_args__ = (
        Index('idx_hmm_deployments_mode', 'mode'),
        Index('idx_hmm_deployments_active', 'is_active'),
    )

    def __repr__(self):
        return f"<HMMDeployment(id={self.id}, model_id={self.model_id}, mode={self.mode}, active={self.is_active})>"

    def to_dict(self):
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "model_id": self.model_id,
            "mode": self.mode,
            "previous_mode": self.previous_mode,
            "transition_date": self.transition_date.isoformat() if self.transition_date else None,
            "approved_by": self.approved_by,
            "performance_metrics": self.performance_metrics,
            "rollback_count": self.rollback_count,
            "is_active": self.is_active,
            "notes": self.notes,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class HMMSyncStatus(Base):
    """
    HMM Sync Status for tracking model synchronization between servers.

    Tracks the synchronization state between Contabo (training) and
    Cloudzy (trading) servers for HMM models.

    Attributes:
        id: Primary key
        contabo_version: Current model version on Contabo
        contabo_last_trained: Last training date on Contabo
        cloudzy_version: Current model version on Cloudzy
        cloudzy_last_deployed: Last deployment date on Cloudzy
        version_mismatch: Whether versions are out of sync
        last_sync_attempt: When last sync was attempted
        last_sync_status: Status of last sync ('success', 'failed', 'in_progress')
        sync_progress: Sync progress percentage (0-100)
        sync_message: Current sync status message
        created_at: Record creation timestamp
        updated_at: Last update timestamp
    """
    __tablename__ = 'hmm_sync_status'

    id = Column(Integer, primary_key=True, autoincrement=True)
    contabo_version = Column(String(20), nullable=True)
    contabo_last_trained = Column(DateTime, nullable=True)
    cloudzy_version = Column(String(20), nullable=True)
    cloudzy_last_deployed = Column(DateTime, nullable=True)
    version_mismatch = Column(Boolean, nullable=False, default=False, index=True)
    last_sync_attempt = Column(DateTime, nullable=True)
    last_sync_status = Column(String(20), nullable=True)
    sync_progress = Column(Float, nullable=False, default=0.0)
    sync_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)

    def __repr__(self):
        return f"<HMMSyncStatus(id={self.id}, contabo={self.contabo_version}, cloudzy={self.cloudzy_version}, mismatch={self.version_mismatch})>"

    def to_dict(self):
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "contabo_version": self.contabo_version,
            "contabo_last_trained": self.contabo_last_trained.isoformat() if self.contabo_last_trained else None,
            "cloudzy_version": self.cloudzy_version,
            "cloudzy_last_deployed": self.cloudzy_last_deployed.isoformat() if self.cloudzy_last_deployed else None,
            "version_mismatch": self.version_mismatch,
            "last_sync_attempt": self.last_sync_attempt.isoformat() if self.last_sync_attempt else None,
            "last_sync_status": self.last_sync_status,
            "sync_progress": self.sync_progress,
            "sync_message": self.sync_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


# =============================================================================
# HOT TIER TABLES (PostgreSQL) - Real-time tick storage with 1-hour retention
# =============================================================================

class SymbolSubscription(Base):
    """
    Symbol Subscription tracking for tick streaming.

    Tracks which bots are subscribed to which symbols for priority-based
    routing and single ZMQ subscription aggregation.

    Attributes:
        id: Primary key
        symbol: Trading symbol (e.g., EURUSD)
        timeframe: Timeframe for the subscription (e.g., M1, M5, H1)
        bot_id: Bot identifier
        priority: Priority level (1=LIVE, 2=DEMO, 3=PAPER)
        subscribed_at: When subscription was created
    """
    __tablename__ = 'symbol_subscriptions'

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False, default='M1', index=True)  # M1, M5, M15, H1, H4, D1
    bot_id = Column(String(100), nullable=False, index=True)
    priority = Column(Integer, nullable=False, default=3)  # 1=LIVE, 2=DEMO, 3=PAPER
    subscribed_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    __table_args__ = (
        Index('idx_symbol_subscriptions_symbol', 'symbol'),
        Index('idx_symbol_subscriptions_bot', 'bot_id'),
        Index('idx_symbol_subscriptions_timeframe', 'timeframe'),
        UniqueConstraint('symbol', 'timeframe', 'bot_id', name='uq_symbol_timeframe_bot'),
    )

    def __repr__(self):
        return f"<SymbolSubscription(id={self.id}, symbol={self.symbol}, timeframe={self.timeframe}, bot_id={self.bot_id}, priority={self.priority})>"


class TickCache(Base):
    """
    HOT tier tick cache for real-time tick storage.
    
    Stores real-time tick data with 1-hour retention.
    Auto-cleanup job runs every 5 minutes to delete old ticks.
    
    Attributes:
        id: Primary key
        symbol: Trading symbol
        bid: Bid price
        ask: Ask price
        timestamp: Tick timestamp
        sequence: Sequence number for ordering
        flags: Tick flags
        created_at: Record creation time
    """
    __tablename__ = 'tick_cache'

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    bid = Column(Float, nullable=False)
    ask = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    sequence = Column(Integer, nullable=False, default=0)
    flags = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    __table_args__ = (
        Index('idx_tick_cache_symbol_timestamp', 'symbol', 'timestamp'),
    )

    def __repr__(self):
        return f"<TickCache(id={self.id}, symbol={self.symbol}, time={self.timestamp})>"


# =============================================================================
# TradingView Integration Models
# =============================================================================

class WebhookLog(Base):
    """
    Webhook Log for TradingView alert tracking.

    Records all incoming webhook alerts from TradingView for audit,
    debugging, and performance analysis.

    Attributes:
        id: Primary key
        timestamp: When the webhook was received
        source_ip: IP address of the sender
        alert_payload: JSON with the full alert data
        signature_valid: Whether HMAC signature was valid
        bot_triggered: Whether a bot was successfully triggered
        order_id: MT5 order ID if trade was placed
        execution_time_ms: Processing time in milliseconds
        error_message: Error message if processing failed
        created_at: Record creation timestamp
    """
    __tablename__ = 'webhook_logs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False, index=True)
    source_ip = Column(String(50), nullable=False, index=True)
    alert_payload = Column(JSON, nullable=False)
    signature_valid = Column(Boolean, nullable=False, default=False)
    bot_triggered = Column(Boolean, nullable=False, default=False, index=True)
    order_id = Column(String(100), nullable=True)
    execution_time_ms = Column(Float, nullable=False, default=0.0)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    __table_args__ = (
        Index('idx_webhook_logs_timestamp', 'timestamp'),
        Index('idx_webhook_logs_bot_triggered', 'bot_triggered'),
        Index('idx_webhook_logs_source_ip', 'source_ip'),
    )

    def __repr__(self):
        return f"<WebhookLog(id={self.id}, ip={self.source_ip}, triggered={self.bot_triggered}, time={self.execution_time_ms}ms)>"

    def to_dict(self):
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "source_ip": self.source_ip,
            "alert_payload": self.alert_payload,
            "signature_valid": self.signature_valid,
            "bot_triggered": self.bot_triggered,
            "order_id": self.order_id,
            "execution_time_ms": self.execution_time_ms,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


# =============================================================================
# GitHub EA Sync Models
# =============================================================================

class ImportedEA(Base):
    """
    Imported EA from GitHub Repository.

    Tracks Expert Advisors imported from GitHub repositories for
    automated bot manifest generation and deployment.

    Attributes:
        id: Primary key
        ea_filename: Name of the EA file (e.g., "ICTSilverBullet.mq5")
        github_path: Full path in GitHub repository
        bot_manifest_id: Generated bot manifest ID (if imported)
        imported_at: When EA was first imported
        last_synced: Last sync timestamp
        version: EA version string
        checksum: SHA256 checksum for change detection
        status: Import status (new, updated, unchanged, error)
        input_parameters: JSON list of extracted input parameters
        strategy_type: Detected strategy type
        timeframe: Detected or specified timeframe
        symbols: JSON list of compatible symbols
        lines_of_code: Total lines of code
        metadata: Additional metadata JSON
        error_message: Error message if import failed
        created_at: Record creation timestamp
        updated_at: Last update timestamp
    """
    __tablename__ = 'imported_eas'

    id = Column(Integer, primary_key=True, autoincrement=True)
    ea_filename = Column(String(200), nullable=False, index=True)
    github_path = Column(String(500), nullable=False)
    bot_manifest_id = Column(Integer, ForeignKey('bot_manifests.id'), nullable=True, index=True)
    imported_at = Column(DateTime, nullable=True)
    last_synced = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    version = Column(String(50), nullable=True)
    checksum = Column(String(64), nullable=False)
    status = Column(String(20), nullable=False, default='new', index=True)  # new, updated, unchanged, error
    input_parameters = Column(JSON, nullable=True)  # [{"name": "LotSize", "type": "double", "default": 0.1}]
    strategy_type = Column(String(100), nullable=True)
    timeframe = Column(String(20), nullable=True)
    symbols = Column(JSON, nullable=True)  # ["EURUSD", "GBPUSD"]
    lines_of_code = Column(Integer, nullable=True)
    ea_metadata = Column(JSON, nullable=True)  # Additional metadata
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)

    # Relationship to BotManifest
    bot_manifest = relationship("BotManifest", back_populates="imported_eas")

    __table_args__ = (
        Index('idx_imported_eas_filename', 'ea_filename'),
        Index('idx_imported_eas_status', 'status'),
        Index('idx_imported_eas_checksum', 'checksum'),
        UniqueConstraint('github_path', name='uq_imported_eas_github_path'),
    )

    def __repr__(self):
        return f"<ImportedEA(id={self.id}, filename={self.ea_filename}, status={self.status})>"

    def to_dict(self):
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "ea_filename": self.ea_filename,
            "github_path": self.github_path,
            "bot_manifest_id": self.bot_manifest_id,
            "imported_at": self.imported_at.isoformat() if self.imported_at else None,
            "last_synced": self.last_synced.isoformat() if self.last_synced else None,
            "version": self.version,
            "checksum": self.checksum,
            "status": self.status,
            "input_parameters": self.input_parameters,
            "strategy_type": self.strategy_type,
            "timeframe": self.timeframe,
            "symbols": self.symbols,
            "lines_of_code": self.lines_of_code,
            "metadata": self.ea_metadata,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


class BotManifest(Base):
    """
    Bot Manifest for EA routing and lifecycle management.

    Defines the characteristics and requirements for a trading bot,
    enabling automatic routing to appropriate accounts.

    Attributes:
        id: Primary key
        bot_name: Name of the bot
        bot_type: Bot type (scalper, structural, swing, hft)
        strategy_type: Strategy classification
        trade_frequency: Expected trading frequency
        broker_type: Preferred broker type
        required_margin: Minimum margin required
        max_drawdown_pct: Maximum acceptable drawdown
        min_win_rate: Minimum win rate required
        target_sharpe: Target Sharpe ratio
        trading_mode: Current trading mode (demo, live)
        status: Bot status (active, paused, stopped)
        bot_metadata: Additional bot configuration
        created_at: Manifest creation timestamp
        updated_at: Last update timestamp
    """
    __tablename__ = 'bot_manifests'

    id = Column(Integer, primary_key=True, autoincrement=True)
    bot_name = Column(String(255), nullable=False, unique=True, index=True)
    bot_type = Column(String(50), nullable=False)
    strategy_type = Column(String(50), nullable=False)
    trade_frequency = Column(String(50), nullable=True)
    broker_type = Column(String(50), nullable=False, default='STANDARD')
    required_margin = Column(Float, nullable=False, default=100.0)
    max_drawdown_pct = Column(Float, nullable=False, default=10.0)
    min_win_rate = Column(Float, nullable=False, default=0.55)
    target_sharpe = Column(Float, nullable=True)
    trading_mode = Column(Enum(TradingMode), nullable=False, default=TradingMode.DEMO, index=True)
    status = Column(String(20), nullable=False, default='active', index=True)
    bot_metadata = Column('metadata', JSON, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    # Relationship to ImportedEAs
    imported_eas = relationship("ImportedEA", back_populates="bot_manifest")

    __table_args__ = (
        Index('idx_bot_manifests_status', 'status'),
        Index('idx_bot_manifests_mode', 'trading_mode'),
    )

    def __repr__(self):
        return f"<BotManifest(id={self.id}, name={self.bot_name}, type={self.bot_type}, mode={self.trading_mode})>"

    def to_dict(self):
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "bot_name": self.bot_name,
            "bot_type": self.bot_type,
            "strategy_type": self.strategy_type,
            "trade_frequency": self.trade_frequency,
            "broker_type": self.broker_type,
            "required_margin": self.required_margin,
            "max_drawdown_pct": self.max_drawdown_pct,
            "min_win_rate": self.min_win_rate,
            "target_sharpe": self.target_sharpe,
            "trading_mode": self.trading_mode.value if self.trading_mode else None,
            "status": self.status,
            "metadata": self.bot_metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


# =============================================================================
# Lifecycle Management Models
# =============================================================================

class BotLifecycleLog(Base):
    """
    Bot Lifecycle Log for tracking tag transitions.
    
    Records all bot tag transitions (promotions, quarantines, deaths)
    for audit trail and analytics.
    
    Attributes:
        id: Primary key
        bot_id: Bot identifier
        from_tag: Source tag
        to_tag: Destination tag
        reason: Reason for transition
        timestamp: When transition occurred
        triggered_by: System or manual
        performance_stats: JSON snapshot of performance at transition
        notes: Additional notes
    """
    __tablename__ = 'bot_lifecycle_log'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    bot_id = Column(String(255), nullable=False, index=True)
    from_tag = Column(String(50), nullable=False)
    to_tag = Column(String(50), nullable=False)
    reason = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False, index=True)
    triggered_by = Column(String(50), default='system')
    performance_stats = Column(JSON, nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    
    __table_args__ = (
        Index('idx_lifecycle_bot_id', 'bot_id'),
        Index('idx_lifecycle_timestamp', 'timestamp'),
        Index('idx_lifecycle_from_tag', 'from_tag'),
        Index('idx_lifecycle_to_tag', 'to_tag'),
    )
    
    def __repr__(self):
        return f"<BotLifecycleLog(id={self.id}, bot_id={self.bot_id}, {self.from_tag}->{self.to_tag})>"
    
    def to_dict(self):
        return {
            "id": self.id,
            "bot_id": self.bot_id,
            "from_tag": self.from_tag,
            "to_tag": self.to_tag,
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "triggered_by": self.triggered_by,
            "performance_stats": self.performance_stats,
            "notes": self.notes,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class MarketOpportunity(Base):
    """
    Market Opportunity detected by MarketScanner.
    
    Stores detected trading opportunities from session breakouts,
    volatility spikes, news events, and ICT setups.
    
    Attributes:
        id: Primary key
        scan_type: Type of scan (SESSION_BREAKOUT, VOLATILITY_SPIKE, etc.)
        symbol: Trading symbol
        session: Market session
        setup: Specific setup detected
        confidence: Confidence score 0-1
        recommended_bots: JSON list of recommended bot IDs
        metadata: Additional details JSON
        timestamp: When detected
        expires_at: When opportunity expires
        status: active, expired, triggered
        triggered_by: Bot ID activated for this opportunity
    """
    __tablename__ = 'market_opportunities'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    scan_type = Column(String(50), nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    session = Column(String(20), nullable=True)
    setup = Column(String(100), nullable=True)
    confidence = Column(Float, default=0.0)
    recommended_bots = Column(JSON, nullable=True)
    opportunity_metadata = Column(JSON, nullable=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False, index=True)
    expires_at = Column(DateTime, nullable=True)
    status = Column(String(20), default='active', index=True)
    triggered_by = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    
    __table_args__ = (
        Index('idx_opportunities_symbol', 'symbol'),
        Index('idx_opportunities_timestamp', 'timestamp'),
        Index('idx_opportunities_scan_type', 'scan_type'),
        Index('idx_opportunities_status', 'status'),
        Index('idx_opportunities_session', 'session'),
    )
    
    def __repr__(self):
        return f"<MarketOpportunity(id={self.id}, type={self.scan_type}, symbol={self.symbol}, confidence={self.confidence})>"
    
    def to_dict(self):
        return {
            "id": self.id,
            "scan_type": self.scan_type,
            "symbol": self.symbol,
            "session": self.session,
            "setup": self.setup,
            "confidence": self.confidence,
            "recommended_bots": self.recommended_bots,
            "metadata": self.opportunity_metadata,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "status": self.status,
            "triggered_by": self.triggered_by,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


# =============================================================================
# Session Factory
# =============================================================================

from sqlalchemy.orm import sessionmaker
from src.database import engine

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

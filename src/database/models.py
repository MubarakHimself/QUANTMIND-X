"""
SQLAlchemy Models for QuantMind Hybrid Core

Defines the database schema for prop firm account management,
daily snapshots, trade proposals, agent tasks, and strategy performance.
"""

from datetime import datetime, timezone
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime,
    ForeignKey, UniqueConstraint, Index, JSON, Text
)
from sqlalchemy.orm import declarative_base, relationship

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
        risk_mode: V8 Tiered Risk Engine mode ('growth', 'scaling', 'guardian')
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
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False, index=True)

    __table_args__ = (
        Index('ix_strategy_performance_kelly', 'kelly_score'),
        Index('ix_strategy_performance_sharpe', 'sharpe_ratio'),
    )

    def __repr__(self):
        return f"<StrategyPerformance(id={self.id}, strategy={self.strategy_name}, kelly={self.kelly_score:.2f}, sharpe={self.sharpe_ratio:.2f})>"


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

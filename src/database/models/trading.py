"""
Trading models.

Contains models for trade proposals, crypto trades, trade journals, and risk tier transitions.
"""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Index, Enum, Text
from sqlalchemy.orm import relationship
from ..models.base import Base, TradingMode


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
    trade_metadata = Column(String, nullable=True)  # Fees, slippage, shadow stops, etc. (stored as JSON string)

    __table_args__ = (
        Index('idx_crypto_trades_broker_symbol', 'broker_id', 'symbol'),
        Index('idx_crypto_trades_status_timestamp', 'status', 'open_timestamp'),
    )

    def __repr__(self):
        return f"<CryptoTrade(id={self.id}, broker={self.broker_type}, symbol={self.symbol}, direction={self.direction}, status={self.status})>"


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

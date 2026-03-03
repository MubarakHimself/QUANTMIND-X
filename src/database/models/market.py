"""
Market models.

Contains models for market data, subscriptions, opportunities, and strategy folders.
"""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Index, Text, UniqueConstraint
from sqlalchemy.orm import relationship
from ..models.base import Base, TradingMode


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


class StrategyFolder(Base):
    """
    Strategy Folder for linking NPRD -> TRD -> EA.

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
        ea_enhanced_path EA code
       : Path to enhanced preferred_conditions: JSON with trading preferences
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
    preferred_conditions = Column(String, nullable=True)  # JSON string
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
    parameters = Column(String, nullable=True)  # JSON string
    dependencies = Column(String, nullable=True)  # JSON string
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
    recommended_bots = Column(String, nullable=True)  # JSON string
    opportunity_metadata = Column(String, nullable=True)  # JSON string
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

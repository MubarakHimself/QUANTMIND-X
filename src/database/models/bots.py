"""
Bot models.

Contains models for bot management including circuit breakers, cloning, manifests, and lifecycle tracking.
"""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Index, Enum, Text, UniqueConstraint, JSON
from sqlalchemy.orm import relationship, attributes
from ..models.base import Base, TradingMode


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


class BotCloneHistory(Base):
    """
    Bot Clone History for tracking bot cloning operations.

    Attributes:
        id: Primary key
        original_bot_id: Original bot identifier
        clone_bot_id: Cloned bot identifier
        original_symbol: Original trading symbol
        clone_symbol: Cloned trading symbol
        performance_at_clone: JSON with performance metrics at clone time
        allocation_strategy: Allocation strategy used
        created_at: Record creation timestamp
    """
    __tablename__ = 'bot_clone_history'

    id = Column(Integer, primary_key=True)
    original_bot_id = Column(String(100), nullable=False, index=True)
    clone_bot_id = Column(String(100), nullable=False, index=True)
    original_symbol = Column(String(20), nullable=False)
    clone_symbol = Column(String(20), nullable=False)
    performance_at_clone = Column(JSON, nullable=False)
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
    input_parameters = Column(JSON, nullable=True)
    strategy_type = Column(String(100), nullable=True)
    timeframe = Column(String(20), nullable=True)
    symbols = Column(JSON, nullable=True)
    lines_of_code = Column(Integer, nullable=True)
    ea_metadata = Column(JSON, nullable=True)
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

    @property
    def prop_firm_tags(self) -> list:
        """Get prop firm tags from bot metadata."""
        if not self.bot_metadata:
            return []
        return list(self.bot_metadata.get('prop_firm_tags', []))

    @prop_firm_tags.setter
    def prop_firm_tags(self, tags: list):
        """Set prop firm tags in bot metadata."""
        if self.bot_metadata is None:
            self.bot_metadata = {}
        self.bot_metadata['prop_firm_tags'] = list(tags)
        attributes.flag_modified(self, 'bot_metadata')

    @property
    def firm_config(self) -> dict:
        """Get firm configuration from bot metadata."""
        if not self.bot_metadata:
            return {}
        return self.bot_metadata.get('firm_config', {})

    @firm_config.setter
    def firm_config(self, config: dict):
        """Set firm configuration in bot metadata."""
        if self.bot_metadata is None:
            self.bot_metadata = {}
        self.bot_metadata['firm_config'] = config
        attributes.flag_modified(self, 'bot_metadata')

    def add_prop_firm_tag(self, tag: str) -> None:
        """Add a prop firm tag to the bot."""
        tags = self.prop_firm_tags
        if tag not in tags:
            tags.append(tag)
            self.prop_firm_tags = tags

    def remove_prop_firm_tag(self, tag: str) -> None:
        """Remove a prop firm tag from the bot."""
        tags = self.prop_firm_tags
        if tag in tags:
            tags.remove(tag)
            self.prop_firm_tags = tags

    def has_prop_firm_tag(self, tag: str) -> bool:
        """Check if bot has a specific prop firm tag."""
        return tag in self.prop_firm_tags

    @classmethod
    def get_valid_prop_firm_tags(cls) -> list:
        """Get list of valid prop firm tags."""
        return ['@fundednext', '@challenge_active', '@funded']


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

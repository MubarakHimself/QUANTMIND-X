"""
Broker Account and Routing Rule models.

Extends the existing BrokerRegistry with additional fields for multi-broker management,
Islamic compliance, and routing matrix support.

Story 9.1: Broker Account Registry & Routing Matrix API
"""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, UniqueConstraint, Index, Enum, JSON, Text
from sqlalchemy.orm import relationship
from enum import Enum as PyEnum

from ..models.base import Base


class BrokerAccountType(PyEnum):
    """Broker account types including Islamic compliance."""
    STANDARD = "standard"
    ISLAMIC = "islamic"  # Swap-free
    PROP_FIRM = "prop_firm"
    PERSONAL = "personal"


class RegimeType(PyEnum):
    """Market regime types for routing rules."""
    TREND = "trend"
    RANGE = "range"
    BREAKOUT = "breakout"
    CHAOS = "chaos"


class StrategyTypeEnum(PyEnum):
    """Strategy types for routing rules."""
    SCALPER = "scalper"
    HFT = "hft"
    STRUCTURAL = "structural"
    SWING = "swing"
    MACRO = "macro"


class BrokerAccount(Base):
    """
    Broker account with full configuration for multi-broker management.

    Stores broker account details with MT5 auto-detection, Islamic compliance,
    and routing configuration.

    Attributes:
        id: Primary key
        broker_name: Name of the broker (e.g., "IC Markets", "RoboForex")
        account_number: MT5 account number
        account_type: Account type (standard, islamic, prop_firm, personal)
        account_tag: Optional tag for grouping (e.g., "hft", "swing", "prop")
        mt5_server: MT5 server hostname
        login_encrypted: Encrypted MT5 login
        swap_free: Whether account is swap-free (Islamic)
        leverage: Account leverage (e.g., 100, 200, 500)
        currency: Account base currency (e.g., "USD", "EUR")

        # MT5 Auto-detection fields
        detected_broker: Auto-detected broker name
        detected_account_type: Auto-detected account type
        detected_leverage: Auto-detected leverage
        detected_currency: Auto-detected currency

        # Status
        is_active: Whether account is active
        is_demo: Whether this is a demo account

        # Timestamps
        created_at: Account creation timestamp
        updated_at: Last update timestamp
    """
    __tablename__ = 'broker_accounts'

    id = Column(Integer, primary_key=True, autoincrement=True)
    broker_name = Column(String(200), nullable=False)
    account_number = Column(String(50), nullable=False, unique=True, index=True)
    account_type = Column(Enum(BrokerAccountType), nullable=False, default=BrokerAccountType.STANDARD, index=True)
    account_tag = Column(String(50), nullable=True, index=True)
    mt5_server = Column(String(200), nullable=False)
    login_encrypted = Column(Text, nullable=False)  # Encrypted storage
    swap_free = Column(Boolean, nullable=False, default=False, index=True)
    leverage = Column(Integer, nullable=False, default=100, index=True)
    currency = Column(String(10), nullable=False, default="USD")

    # MT5 Auto-detection results
    detected_broker = Column(String(200), nullable=True)
    detected_account_type = Column(String(50), nullable=True)
    detected_leverage = Column(Integer, nullable=True)
    detected_currency = Column(String(10), nullable=True)

    # Status
    is_active = Column(Boolean, nullable=False, default=True, index=True)
    is_demo = Column(Boolean, nullable=False, default=False, index=True)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)

    # Relationships
    routing_rules = relationship("RoutingRule", back_populates="broker_account", cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_broker_accounts_active', 'is_active', 'account_type'),
        Index('idx_broker_accounts_tag', 'account_tag', 'is_active'),
    )

    def __repr__(self):
        return f"<BrokerAccount(id={self.id}, broker={self.broker_name}, account={self.account_number}, type={self.account_type.value})>"


class RoutingRule(Base):
    """
    Routing rule for strategy-to-account assignment.

    Defines rules for assigning strategies to broker accounts based on:
    - account_tag: Grouping tag (e.g., "hft", "swing")
    - regime_filter: Market regime (trend, range, breakout, chaos)
    - strategy_type: Strategy type (scalper, hft, structural, swing, macro)

    Attributes:
        id: Primary key
        broker_account_id: Foreign key to BrokerAccount
        account_tag: Account tag to match
        regime_filter: Market regime to match (null = all regimes)
        strategy_type: Strategy type to match
        priority: Rule priority (higher = more important)
        is_active: Whether rule is active
        created_at: Rule creation timestamp
        updated_at: Last update timestamp
    """
    __tablename__ = 'routing_rules'

    id = Column(Integer, primary_key=True, autoincrement=True)
    broker_account_id = Column(Integer, ForeignKey('broker_accounts.id', ondelete='CASCADE'), nullable=False, index=True)
    account_tag = Column(String(50), nullable=True, index=True)  # null = match all
    regime_filter = Column(Enum(RegimeType), nullable=True, index=True)  # null = match all regimes
    strategy_type = Column(Enum(StrategyTypeEnum), nullable=False, index=True)
    priority = Column(Integer, nullable=False, default=100)  # Higher = more important
    is_active = Column(Boolean, nullable=False, default=True, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)

    # Relationship
    broker_account = relationship("BrokerAccount", back_populates="routing_rules")

    __table_args__ = (
        UniqueConstraint('broker_account_id', 'account_tag', 'regime_filter', 'strategy_type', name='uq_routing_rule_unique'),
        Index('idx_routing_rules_active', 'is_active', 'strategy_type'),
    )

    def __repr__(self):
        return f"<RoutingRule(id={self.id}, strategy={self.strategy_type.value}, account_tag={self.account_tag}, regime={self.regime_filter})>"
"""
Database models package.

This package provides modular SQLAlchemy models for QuantMind Hybrid Core.
Models are organized by domain: account, trading, performance, bots, agents, hmm, market, monitoring.

For backward compatibility, all models are also re-exported from this package.
Import directly from here: from src.database.models import PropFirmAccount
"""

# Base utilities
from .base import Base, TradingMode, AccountType, get_db_session

# Account models
from .account import (
    PropFirmAccount,
    DailySnapshot,
    BrokerRegistry,
    AccountLossState,
)

# Trading models
from .trading import (
    TradeProposal,
    RiskTierTransition,
    CryptoTrade,
    TradeJournal,
)

# Trade record model (Story 14.1: Layer 1 EA Hard Safety SL/TP)
from .trade_record import TradeRecord

# Performance models
from .performance import (
    StrategyPerformance,
    PaperTradingPerformance,
    HouseMoneyState,
    StrategyFamilyState,
)

# Bot models
from .bots import (
    BotCircuitBreaker,
    BotCloneHistory,
    DailyFeeTracking,
    ImportedEA,
    BotManifest,
    BotLifecycleLog,
)

# Agent models
from .agents import AgentTasks
from .agent_session import AgentSession
from .session_checkpoint import SessionCheckpoint

# Approval request model (HITL persistence + resume)
from .approval_request import ApprovalRequestModel, init_approval_requests_table

# HMM models
from .hmm import (
    HMMModel,
    HMMShadowLog,
    HMMDeployment,
    HMMSyncStatus,
)

# Market models
from .market import (
    SymbolSubscription,
    TickCache,
    StrategyFolder,
    SharedAsset,
    MarketOpportunity,
)

# Monitoring models
from .monitoring import (
    AlertHistory,
    WebhookLog,
)

# Activity models
from .activity import ActivityEvent

# Chat models
from .chat import ChatSession, ChatMessage

# Provider config models
from .provider_config import ProviderConfig

# Server config models
from .server_config import ServerConfig, ServerType

# Risk params models
from .risk_params import RiskParams, RiskParamsAudit

# News feed models (Story 6.3)
from .news_items import NewsItem

# Broker account models (Story 9.1)
from .broker_account import (
    BrokerAccount,
    RoutingRule,
    BrokerAccountType,
    RegimeType,
    StrategyTypeEnum,
)

# Notification config models (Story 10.3, 10.5)
from .notification_config import NotificationConfig, LogRetentionPolicy

# Audit log models (Story 10.1)
from .audit_log import AuditLogEntry, AuditQueryResult, AuditLayer, TradeEventType, StrategyLifecycleEventType, RiskParamEventType, AgentActionEventType, SystemHealthEventType

# Session factory (import from engine package)
from sqlalchemy.orm import sessionmaker
from src.database import engine

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Re-export Base for convenience
__all__ = [
    # Base
    'Base',
    'TradingMode',
    'AccountType',
    'get_db_session',
    'SessionLocal',

    # Account
    'PropFirmAccount',
    'DailySnapshot',
    'BrokerRegistry',
    'AccountLossState',

    # Trading
    'TradeProposal',
    'RiskTierTransition',
    'CryptoTrade',
    'TradeJournal',

    # Trade record (Story 14.1)
    'TradeRecord',

    # Performance
    'StrategyPerformance',
    'PaperTradingPerformance',
    'HouseMoneyState',
    'StrategyFamilyState',

    # Bots
    'BotCircuitBreaker',
    'BotCloneHistory',
    'DailyFeeTracking',
    'ImportedEA',
    'BotManifest',
    'BotLifecycleLog',

    # Agents
    'AgentTasks',
    'AgentSession',
    'SessionCheckpoint',

    # Approval requests (HITL)
    'ApprovalRequestModel',
    'init_approval_requests_table',

    # HMM
    'HMMModel',
    'HMMShadowLog',
    'HMMDeployment',
    'HMMSyncStatus',

    # Market
    'SymbolSubscription',
    'TickCache',
    'StrategyFolder',
    'SharedAsset',
    'MarketOpportunity',

    # Monitoring
    'AlertHistory',
    'WebhookLog',

    # Activity
    'ActivityEvent',

    # Chat
    'ChatSession',
    'ChatMessage',

    # Provider config
    'ProviderConfig',
    'ServerConfig',

    # Risk params
    'RiskParams',
    'RiskParamsAudit',

    # News feed (Story 6.3)
    'NewsItem',

    # Broker account models (Story 9.1)
    'BrokerAccount',
    'RoutingRule',
    'BrokerAccountType',
    'RegimeType',
    'StrategyTypeEnum',

    # Notification config models (Story 10.3, 10.5)
    'NotificationConfig',
    'LogRetentionPolicy',

    # Audit log models (Story 10.1)
    'AuditLogEntry',
    'AuditQueryResult',
    'AuditLayer',
    'TradeEventType',
    'StrategyLifecycleEventType',
    'RiskParamEventType',
    'AgentActionEventType',
    'SystemHealthEventType',
]

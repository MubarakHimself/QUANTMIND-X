"""
Database models package.

This package provides modular SQLAlchemy models for QuantMind Hybrid Core.
Models are organized by domain: account, trading, performance, bots, agents, hmm, market, monitoring.

For backward compatibility, all models are also re-exported from this package.
Import directly from here: from src.database.models import PropFirmAccount
"""

# Base utilities
from .base import Base, TradingMode, get_db_session

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

# Session factory (import from engine package)
from sqlalchemy.orm import sessionmaker
from src.database import engine

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Re-export Base for convenience
__all__ = [
    # Base
    'Base',
    'TradingMode',
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
]

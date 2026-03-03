"""
Test for models.py modular structure.

This test ensures backward compatibility by verifying that
all models can be imported from src.database.models.
"""

from src.database.models import (
    # Core models that should exist
    PropFirmAccount,
    DailySnapshot,
    TradeProposal,
    AgentTasks,
    StrategyPerformance,
    PaperTradingPerformance,
    CryptoTrade,
    StrategyFolder,
    SharedAsset,
    BrokerRegistry,
    HouseMoneyState,
    BotCircuitBreaker,
    TradeJournal,
    BotCloneHistory,
    DailyFeeTracking,
    StrategyFamilyState,
    AccountLossState,
    AlertHistory,
    HMMModel,
    HMMShadowLog,
    HMMDeployment,
    HMMSyncStatus,
    SymbolSubscription,
    TickCache,
    WebhookLog,
    ImportedEA,
    BotManifest,
    BotLifecycleLog,
    MarketOpportunity,
    RiskTierTransition,
    TradingMode,
    get_db_session,
    Base,
)


def test_model_imports():
    """Test that all major models are importable from src.database.models."""
    # Prop firm models
    assert PropFirmAccount is not None
    assert DailySnapshot is not None
    assert BrokerRegistry is not None
    assert AccountLossState is not None

    # Trading models
    assert TradeProposal is not None
    assert CryptoTrade is not None
    assert TradeJournal is not None
    assert RiskTierTransition is not None

    # Performance models
    assert StrategyPerformance is not None
    assert PaperTradingPerformance is not None
    assert HouseMoneyState is not None
    assert StrategyFamilyState is not None

    # Bot models
    assert BotCircuitBreaker is not None
    assert BotCloneHistory is not None
    assert BotManifest is not None
    assert BotLifecycleLog is not None
    assert DailyFeeTracking is not None

    # Agent models
    assert AgentTasks is not None

    # HMM models
    assert HMMModel is not None
    assert HMMShadowLog is not None
    assert HMMDeployment is not None
    assert HMMSyncStatus is not None

    # Market models
    assert SymbolSubscription is not None
    assert TickCache is not None
    assert MarketOpportunity is not None

    # Other models
    assert StrategyFolder is not None
    assert SharedAsset is not None
    assert AlertHistory is not None
    assert WebhookLog is not None
    assert ImportedEA is not None

    # Utilities
    assert TradingMode is not None
    assert get_db_session is not None
    assert Base is not None


def test_trading_mode_enum():
    """Test TradingMode enum values."""
    assert TradingMode.DEMO.value == "demo"
    assert TradingMode.LIVE.value == "live"


def test_get_db_session_callable():
    """Test that get_db_session is callable."""
    assert callable(get_db_session)

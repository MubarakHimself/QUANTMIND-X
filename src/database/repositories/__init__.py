"""
Database Repositories Module.

Modular structure for database access:
- account_repository: Prop firm account operations
- snapshot_repository: Daily snapshot operations
- proposal_repository: Trade proposal operations
- task_repository: Agent task operations
- broker_repository: Broker registry operations
- bot_repository: Bot manifest operations
- account_loss_state_repository: Account loss state (Tier 3)
- circuit_breaker_repository: Bot circuit breaker operations
- market_repository: Market opportunities and shared assets
- trading_repository: Trading operations (RiskTierTransition, CryptoTrade, TradeJournal)
- performance_repository: Performance tracking (StrategyPerformance, PaperTradingPerformance, etc.)
- bot_management_repository: Bot management (BotCloneHistory, DailyFeeTracking, ImportedEA, BotLifecycleLog)
- hmm_repository: HMM regime detection models
- monitoring_repository: Monitoring (SymbolSubscription, TickCache, AlertHistory, WebhookLog)
"""

from .account_repository import AccountRepository
from .snapshot_repository import SnapshotRepository
from .proposal_repository import ProposalRepository
from .task_repository import TaskRepository
from .broker_repository import BrokerRepository
from .bot_repository import BotRepository
from .account_loss_state_repository import AccountLossStateRepository
from .circuit_breaker_repository import CircuitBreakerRepository
from .market_repository import MarketRepository, SharedAssetRepository, StrategyFolderRepository
from .strategy_repository import StrategyRepository

# Trading repositories
from .trading_repository import (
    RiskTierTransitionRepository,
    CryptoTradeRepository,
    TradeJournalRepository,
)

# Performance repositories
from .performance_repository import (
    StrategyPerformanceRepository,
    PaperTradingPerformanceRepository,
    HouseMoneyStateRepository,
    StrategyFamilyStateRepository,
)

# Bot management repositories
from .bot_management_repository import (
    BotCloneHistoryRepository,
    DailyFeeTrackingRepository,
    ImportedEARepository,
    BotLifecycleLogRepository,
)

# HMM repositories
from .hmm_repository import (
    HMMModelRepository,
    HMMShadowLogRepository,
    HMMDeploymentRepository,
    HMMSyncStatusRepository,
)

# Monitoring repositories
from .monitoring_repository import (
    SymbolSubscriptionRepository,
    TickCacheRepository,
    AlertHistoryRepository,
    WebhookLogRepository,
)

# Activity repositories
from .activity_event_repository import ActivityEventRepository, activity_event_repository

# Repository instances (lazy initialization)
_account_repository = None
_snapshot_repository = None
_proposal_repository = None
_task_repository = None
_broker_repository = None
_bot_repository = None
_account_loss_state_repository = None
_circuit_breaker_repository = None
_market_repository = None

# Trading repositories
_risk_tier_transition_repository = None
_crypto_trade_repository = None
_trade_journal_repository = None

# Performance repositories
_strategy_performance_repository = None
_paper_trading_performance_repository = None
_house_money_state_repository = None
_strategy_family_state_repository = None

# Bot management repositories
_bot_clone_history_repository = None
_daily_fee_tracking_repository = None
_imported_ea_repository = None
_bot_lifecycle_log_repository = None

# HMM repositories
_hmm_model_repository = None
_hmm_shadow_log_repository = None
_hmm_deployment_repository = None
_hmm_sync_status_repository = None

# Monitoring repositories
_symbol_subscription_repository = None
_tick_cache_repository = None
_alert_history_repository = None
_webhook_log_repository = None

# Activity repositories
_activity_event_repository = None


def get_account_repository() -> AccountRepository:
    """Get or create account repository instance."""
    global _account_repository
    if _account_repository is None:
        _account_repository = AccountRepository()
    return _account_repository


def get_snapshot_repository() -> SnapshotRepository:
    """Get or create snapshot repository instance."""
    global _snapshot_repository
    if _snapshot_repository is None:
        _snapshot_repository = SnapshotRepository()
    return _snapshot_repository


def get_proposal_repository() -> ProposalRepository:
    """Get or create proposal repository instance."""
    global _proposal_repository
    if _proposal_repository is None:
        _proposal_repository = ProposalRepository()
    return _proposal_repository


def get_task_repository() -> TaskRepository:
    """Get or create task repository instance."""
    global _task_repository
    if _task_repository is None:
        _task_repository = TaskRepository()
    return _task_repository


def get_broker_repository() -> BrokerRepository:
    """Get or create broker repository instance."""
    global _broker_repository
    if _broker_repository is None:
        _broker_repository = BrokerRepository()
    return _broker_repository


def get_bot_repository() -> BotRepository:
    """Get or create bot repository instance."""
    global _bot_repository
    if _bot_repository is None:
        _bot_repository = BotRepository()
    return _bot_repository


def get_account_loss_state_repository() -> AccountLossStateRepository:
    """Get or create account loss state repository instance."""
    global _account_loss_state_repository
    if _account_loss_state_repository is None:
        _account_loss_state_repository = AccountLossStateRepository()
    return _account_loss_state_repository


def get_circuit_breaker_repository() -> CircuitBreakerRepository:
    """Get or create circuit breaker repository instance."""
    global _circuit_breaker_repository
    if _circuit_breaker_repository is None:
        _circuit_breaker_repository = CircuitBreakerRepository()
    return _circuit_breaker_repository


def get_market_repository() -> MarketRepository:
    """Get or create market repository instance."""
    global _market_repository
    if _market_repository is None:
        _market_repository = MarketRepository()
    return _market_repository


# Trading repository getters
def get_risk_tier_transition_repository() -> RiskTierTransitionRepository:
    """Get or create risk tier transition repository instance."""
    global _risk_tier_transition_repository
    if _risk_tier_transition_repository is None:
        _risk_tier_transition_repository = RiskTierTransitionRepository()
    return _risk_tier_transition_repository


def get_crypto_trade_repository() -> CryptoTradeRepository:
    """Get or create crypto trade repository instance."""
    global _crypto_trade_repository
    if _crypto_trade_repository is None:
        _crypto_trade_repository = CryptoTradeRepository()
    return _crypto_trade_repository


def get_trade_journal_repository() -> TradeJournalRepository:
    """Get or create trade journal repository instance."""
    global _trade_journal_repository
    if _trade_journal_repository is None:
        _trade_journal_repository = TradeJournalRepository()
    return _trade_journal_repository


# Performance repository getters
def get_strategy_performance_repository() -> StrategyPerformanceRepository:
    """Get or create strategy performance repository instance."""
    global _strategy_performance_repository
    if _strategy_performance_repository is None:
        _strategy_performance_repository = StrategyPerformanceRepository()
    return _strategy_performance_repository


def get_paper_trading_performance_repository() -> PaperTradingPerformanceRepository:
    """Get or create paper trading performance repository instance."""
    global _paper_trading_performance_repository
    if _paper_trading_performance_repository is None:
        _paper_trading_performance_repository = PaperTradingPerformanceRepository()
    return _paper_trading_performance_repository


def get_house_money_state_repository() -> HouseMoneyStateRepository:
    """Get or create house money state repository instance."""
    global _house_money_state_repository
    if _house_money_state_repository is None:
        _house_money_state_repository = HouseMoneyStateRepository()
    return _house_money_state_repository


def get_strategy_family_state_repository() -> StrategyFamilyStateRepository:
    """Get or create strategy family state repository instance."""
    global _strategy_family_state_repository
    if _strategy_family_state_repository is None:
        _strategy_family_state_repository = StrategyFamilyStateRepository()
    return _strategy_family_state_repository


# Bot management repository getters
def get_bot_clone_history_repository() -> BotCloneHistoryRepository:
    """Get or create bot clone history repository instance."""
    global _bot_clone_history_repository
    if _bot_clone_history_repository is None:
        _bot_clone_history_repository = BotCloneHistoryRepository()
    return _bot_clone_history_repository


def get_daily_fee_tracking_repository() -> DailyFeeTrackingRepository:
    """Get or create daily fee tracking repository instance."""
    global _daily_fee_tracking_repository
    if _daily_fee_tracking_repository is None:
        _daily_fee_tracking_repository = DailyFeeTrackingRepository()
    return _daily_fee_tracking_repository


def get_imported_ea_repository() -> ImportedEARepository:
    """Get or create imported EA repository instance."""
    global _imported_ea_repository
    if _imported_ea_repository is None:
        _imported_ea_repository = ImportedEARepository()
    return _imported_ea_repository


def get_bot_lifecycle_log_repository() -> BotLifecycleLogRepository:
    """Get or create bot lifecycle log repository instance."""
    global _bot_lifecycle_log_repository
    if _bot_lifecycle_log_repository is None:
        _bot_lifecycle_log_repository = BotLifecycleLogRepository()
    return _bot_lifecycle_log_repository


# HMM repository getters
def get_hmm_model_repository() -> HMMModelRepository:
    """Get or create HMM model repository instance."""
    global _hmm_model_repository
    if _hmm_model_repository is None:
        _hmm_model_repository = HMMModelRepository()
    return _hmm_model_repository


def get_hmm_shadow_log_repository() -> HMMShadowLogRepository:
    """Get or create HMM shadow log repository instance."""
    global _hmm_shadow_log_repository
    if _hmm_shadow_log_repository is None:
        _hmm_shadow_log_repository = HMMShadowLogRepository()
    return _hmm_shadow_log_repository


def get_hmm_deployment_repository() -> HMMDeploymentRepository:
    """Get or create HMM deployment repository instance."""
    global _hmm_deployment_repository
    if _hmm_deployment_repository is None:
        _hmm_deployment_repository = HMMDeploymentRepository()
    return _hmm_deployment_repository


def get_hmm_sync_status_repository() -> HMMSyncStatusRepository:
    """Get or create HMM sync status repository instance."""
    global _hmm_sync_status_repository
    if _hmm_sync_status_repository is None:
        _hmm_sync_status_repository = HMMSyncStatusRepository()
    return _hmm_sync_status_repository


# Monitoring repository getters
def get_symbol_subscription_repository() -> SymbolSubscriptionRepository:
    """Get or create symbol subscription repository instance."""
    global _symbol_subscription_repository
    if _symbol_subscription_repository is None:
        _symbol_subscription_repository = SymbolSubscriptionRepository()
    return _symbol_subscription_repository


def get_tick_cache_repository() -> TickCacheRepository:
    """Get or create tick cache repository instance."""
    global _tick_cache_repository
    if _tick_cache_repository is None:
        _tick_cache_repository = TickCacheRepository()
    return _tick_cache_repository


def get_alert_history_repository() -> AlertHistoryRepository:
    """Get or create alert history repository instance."""
    global _alert_history_repository
    if _alert_history_repository is None:
        _alert_history_repository = AlertHistoryRepository()
    return _alert_history_repository


def get_webhook_log_repository() -> WebhookLogRepository:
    """Get or create webhook log repository instance."""
    global _webhook_log_repository
    if _webhook_log_repository is None:
        _webhook_log_repository = WebhookLogRepository()
    return _webhook_log_repository


# Activity repository getters
def get_activity_event_repository() -> ActivityEventRepository:
    """Get or create activity event repository instance."""
    global _activity_event_repository
    if _activity_event_repository is None:
        _activity_event_repository = ActivityEventRepository()
    return _activity_event_repository


# Aliases for backward compatibility
account_repository = get_account_repository()
snapshot_repository = get_snapshot_repository()
proposal_repository = get_proposal_repository()
task_repository = get_task_repository()
broker_repository = get_broker_repository()
bot_repository = get_bot_repository()
account_loss_state_repository = get_account_loss_state_repository()
circuit_breaker_repository = get_circuit_breaker_repository()
market_repository = get_market_repository()
activity_event_repository = get_activity_event_repository()

__all__ = [
    # Repository classes
    'AccountRepository',
    'SnapshotRepository',
    'ProposalRepository',
    'TaskRepository',
    'BrokerRepository',
    'BotRepository',
    'AccountLossStateRepository',
    'CircuitBreakerRepository',
    'MarketRepository',
    'SharedAssetRepository',
    'StrategyFolderRepository',
    'StrategyRepository',
    # Trading repositories
    'RiskTierTransitionRepository',
    'CryptoTradeRepository',
    'TradeJournalRepository',
    # Performance repositories
    'StrategyPerformanceRepository',
    'PaperTradingPerformanceRepository',
    'HouseMoneyStateRepository',
    'StrategyFamilyStateRepository',
    # Bot management repositories
    'BotCloneHistoryRepository',
    'DailyFeeTrackingRepository',
    'ImportedEARepository',
    'BotLifecycleLogRepository',
    # HMM repositories
    'HMMModelRepository',
    'HMMShadowLogRepository',
    'HMMDeploymentRepository',
    'HMMSyncStatusRepository',
    # Monitoring repositories
    'SymbolSubscriptionRepository',
    'TickCacheRepository',
    'AlertHistoryRepository',
    'WebhookLogRepository',
    # Repository instances
    'account_repository',
    'snapshot_repository',
    'proposal_repository',
    'task_repository',
    'broker_repository',
    'bot_repository',
    'account_loss_state_repository',
    'circuit_breaker_repository',
    'market_repository',
    # Getter functions
    'get_account_repository',
    'get_snapshot_repository',
    'get_proposal_repository',
    'get_task_repository',
    'get_broker_repository',
    'get_bot_repository',
    'get_account_loss_state_repository',
    'get_circuit_breaker_repository',
    'get_market_repository',
    # Trading getters
    'get_risk_tier_transition_repository',
    'get_crypto_trade_repository',
    'get_trade_journal_repository',
    # Performance getters
    'get_strategy_performance_repository',
    'get_paper_trading_performance_repository',
    'get_house_money_state_repository',
    'get_strategy_family_state_repository',
    # Bot management getters
    'get_bot_clone_history_repository',
    'get_daily_fee_tracking_repository',
    'get_imported_ea_repository',
    'get_bot_lifecycle_log_repository',
    # HMM getters
    'get_hmm_model_repository',
    'get_hmm_shadow_log_repository',
    'get_hmm_deployment_repository',
    'get_hmm_sync_status_repository',
    # Monitoring getters
    'get_symbol_subscription_repository',
    'get_tick_cache_repository',
    'get_alert_history_repository',
    'get_webhook_log_repository',
    # Activity getters
    'get_activity_event_repository',
    'activity_event_repository',
]

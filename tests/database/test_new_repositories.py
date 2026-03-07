"""
Tests for new Database Repositories.

Tests the new repository implementations added to complete the repository pattern.
"""

import pytest


class TestNewRepositories:
    """Test new repository classes exist."""

    def test_broker_repository_class(self):
        """Test BrokerRepository class exists."""
        from src.database.repositories import BrokerRepository
        assert BrokerRepository is not None

    def test_bot_repository_class(self):
        """Test BotRepository class exists."""
        from src.database.repositories import BotRepository
        assert BotRepository is not None

    def test_account_loss_state_repository_class(self):
        """Test AccountLossStateRepository class exists."""
        from src.database.repositories import AccountLossStateRepository
        assert AccountLossStateRepository is not None

    def test_circuit_breaker_repository_class(self):
        """Test CircuitBreakerRepository class exists."""
        from src.database.repositories import CircuitBreakerRepository
        assert CircuitBreakerRepository is not None

    def test_market_repository_class(self):
        """Test MarketRepository class exists."""
        from src.database.repositories import MarketRepository
        assert MarketRepository is not None

    def test_shared_asset_repository_class(self):
        """Test SharedAssetRepository class exists."""
        from src.database.repositories import SharedAssetRepository
        assert SharedAssetRepository is not None

    def test_strategy_folder_repository_class(self):
        """Test StrategyFolderRepository class exists."""
        from src.database.repositories import StrategyFolderRepository
        assert StrategyFolderRepository is not None


class TestNewRepositoryInstances:
    """Test new repository instances exist."""

    def test_broker_repository_instance(self):
        """Test broker_repository instance exists."""
        from src.database.repositories import broker_repository
        assert broker_repository is not None

    def test_bot_repository_instance(self):
        """Test bot_repository instance exists."""
        from src.database.repositories import bot_repository
        assert bot_repository is not None

    def test_account_loss_state_repository_instance(self):
        """Test account_loss_state_repository instance exists."""
        from src.database.repositories import account_loss_state_repository
        assert account_loss_state_repository is not None

    def test_circuit_breaker_repository_instance(self):
        """Test circuit_breaker_repository instance exists."""
        from src.database.repositories import circuit_breaker_repository
        assert circuit_breaker_repository is not None

    def test_market_repository_instance(self):
        """Test market_repository instance exists."""
        from src.database.repositories import market_repository
        assert market_repository is not None


class TestBaseRepository:
    """Test base repository class."""

    def test_base_repository_importable(self):
        """Test BaseRepository can be imported."""
        from src.database.repositories.base_repository import BaseRepository
        assert BaseRepository is not None


class TestTradingRepositories:
    """Test trading repository classes exist."""

    def test_risk_tier_transition_repository_class(self):
        """Test RiskTierTransitionRepository class exists."""
        from src.database.repositories import RiskTierTransitionRepository
        assert RiskTierTransitionRepository is not None

    def test_crypto_trade_repository_class(self):
        """Test CryptoTradeRepository class exists."""
        from src.database.repositories import CryptoTradeRepository
        assert CryptoTradeRepository is not None

    def test_trade_journal_repository_class(self):
        """Test TradeJournalRepository class exists."""
        from src.database.repositories import TradeJournalRepository
        assert TradeJournalRepository is not None


class TestPerformanceRepositories:
    """Test performance repository classes exist."""

    def test_strategy_performance_repository_class(self):
        """Test StrategyPerformanceRepository class exists."""
        from src.database.repositories import StrategyPerformanceRepository
        assert StrategyPerformanceRepository is not None

    def test_paper_trading_performance_repository_class(self):
        """Test PaperTradingPerformanceRepository class exists."""
        from src.database.repositories import PaperTradingPerformanceRepository
        assert PaperTradingPerformanceRepository is not None

    def test_house_money_state_repository_class(self):
        """Test HouseMoneyStateRepository class exists."""
        from src.database.repositories import HouseMoneyStateRepository
        assert HouseMoneyStateRepository is not None

    def test_strategy_family_state_repository_class(self):
        """Test StrategyFamilyStateRepository class exists."""
        from src.database.repositories import StrategyFamilyStateRepository
        assert StrategyFamilyStateRepository is not None


class TestBotManagementRepositories:
    """Test bot management repository classes exist."""

    def test_bot_clone_history_repository_class(self):
        """Test BotCloneHistoryRepository class exists."""
        from src.database.repositories import BotCloneHistoryRepository
        assert BotCloneHistoryRepository is not None

    def test_daily_fee_tracking_repository_class(self):
        """Test DailyFeeTrackingRepository class exists."""
        from src.database.repositories import DailyFeeTrackingRepository
        assert DailyFeeTrackingRepository is not None

    def test_imported_ea_repository_class(self):
        """Test ImportedEARepository class exists."""
        from src.database.repositories import ImportedEARepository
        assert ImportedEARepository is not None

    def test_bot_lifecycle_log_repository_class(self):
        """Test BotLifecycleLogRepository class exists."""
        from src.database.repositories import BotLifecycleLogRepository
        assert BotLifecycleLogRepository is not None


class TestHMMRepositories:
    """Test HMM repository classes exist."""

    def test_hmm_model_repository_class(self):
        """Test HMMModelRepository class exists."""
        from src.database.repositories import HMMModelRepository
        assert HMMModelRepository is not None

    def test_hmm_shadow_log_repository_class(self):
        """Test HMMShadowLogRepository class exists."""
        from src.database.repositories import HMMShadowLogRepository
        assert HMMShadowLogRepository is not None

    def test_hmm_deployment_repository_class(self):
        """Test HMMDeploymentRepository class exists."""
        from src.database.repositories import HMMDeploymentRepository
        assert HMMDeploymentRepository is not None

    def test_hmm_sync_status_repository_class(self):
        """Test HMMSyncStatusRepository class exists."""
        from src.database.repositories import HMMSyncStatusRepository
        assert HMMSyncStatusRepository is not None


class TestMonitoringRepositories:
    """Test monitoring repository classes exist."""

    def test_symbol_subscription_repository_class(self):
        """Test SymbolSubscriptionRepository class exists."""
        from src.database.repositories import SymbolSubscriptionRepository
        assert SymbolSubscriptionRepository is not None

    def test_tick_cache_repository_class(self):
        """Test TickCacheRepository class exists."""
        from src.database.repositories import TickCacheRepository
        assert TickCacheRepository is not None

    def test_alert_history_repository_class(self):
        """Test AlertHistoryRepository class exists."""
        from src.database.repositories import AlertHistoryRepository
        assert AlertHistoryRepository is not None

    def test_webhook_log_repository_class(self):
        """Test WebhookLogRepository class exists."""
        from src.database.repositories import WebhookLogRepository
        assert WebhookLogRepository is not None


class TestRepositoryGetters:
    """Test repository getter functions exist."""

    def test_get_risk_tier_transition_repository(self):
        """Test get_risk_tier_transition_repository function exists."""
        from src.database.repositories import get_risk_tier_transition_repository
        assert get_risk_tier_transition_repository is not None

    def test_get_crypto_trade_repository(self):
        """Test get_crypto_trade_repository function exists."""
        from src.database.repositories import get_crypto_trade_repository
        assert get_crypto_trade_repository is not None

    def test_get_trade_journal_repository(self):
        """Test get_trade_journal_repository function exists."""
        from src.database.repositories import get_trade_journal_repository
        assert get_trade_journal_repository is not None

    def test_get_strategy_performance_repository(self):
        """Test get_strategy_performance_repository function exists."""
        from src.database.repositories import get_strategy_performance_repository
        assert get_strategy_performance_repository is not None

    def test_get_paper_trading_performance_repository(self):
        """Test get_paper_trading_performance_repository function exists."""
        from src.database.repositories import get_paper_trading_performance_repository
        assert get_paper_trading_performance_repository is not None

    def test_get_house_money_state_repository(self):
        """Test get_house_money_state_repository function exists."""
        from src.database.repositories import get_house_money_state_repository
        assert get_house_money_state_repository is not None

    def test_get_strategy_family_state_repository(self):
        """Test get_strategy_family_state_repository function exists."""
        from src.database.repositories import get_strategy_family_state_repository
        assert get_strategy_family_state_repository is not None

    def test_get_bot_clone_history_repository(self):
        """Test get_bot_clone_history_repository function exists."""
        from src.database.repositories import get_bot_clone_history_repository
        assert get_bot_clone_history_repository is not None

    def test_get_daily_fee_tracking_repository(self):
        """Test get_daily_fee_tracking_repository function exists."""
        from src.database.repositories import get_daily_fee_tracking_repository
        assert get_daily_fee_tracking_repository is not None

    def test_get_imported_ea_repository(self):
        """Test get_imported_ea_repository function exists."""
        from src.database.repositories import get_imported_ea_repository
        assert get_imported_ea_repository is not None

    def test_get_bot_lifecycle_log_repository(self):
        """Test get_bot_lifecycle_log_repository function exists."""
        from src.database.repositories import get_bot_lifecycle_log_repository
        assert get_bot_lifecycle_log_repository is not None

    def test_get_hmm_model_repository(self):
        """Test get_hmm_model_repository function exists."""
        from src.database.repositories import get_hmm_model_repository
        assert get_hmm_model_repository is not None

    def test_get_hmm_shadow_log_repository(self):
        """Test get_hmm_shadow_log_repository function exists."""
        from src.database.repositories import get_hmm_shadow_log_repository
        assert get_hmm_shadow_log_repository is not None

    def test_get_hmm_deployment_repository(self):
        """Test get_hmm_deployment_repository function exists."""
        from src.database.repositories import get_hmm_deployment_repository
        assert get_hmm_deployment_repository is not None

    def test_get_hmm_sync_status_repository(self):
        """Test get_hmm_sync_status_repository function exists."""
        from src.database.repositories import get_hmm_sync_status_repository
        assert get_hmm_sync_status_repository is not None

    def test_get_symbol_subscription_repository(self):
        """Test get_symbol_subscription_repository function exists."""
        from src.database.repositories import get_symbol_subscription_repository
        assert get_symbol_subscription_repository is not None

    def test_get_tick_cache_repository(self):
        """Test get_tick_cache_repository function exists."""
        from src.database.repositories import get_tick_cache_repository
        assert get_tick_cache_repository is not None

    def test_get_alert_history_repository(self):
        """Test get_alert_history_repository function exists."""
        from src.database.repositories import get_alert_history_repository
        assert get_alert_history_repository is not None

    def test_get_webhook_log_repository(self):
        """Test get_webhook_log_repository function exists."""
        from src.database.repositories import get_webhook_log_repository
        assert get_webhook_log_repository is not None

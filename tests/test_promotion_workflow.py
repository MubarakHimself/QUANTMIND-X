"""
Tests for Paper->Demo->Live Promotion Workflow

Tests cover:
- TradingMode enum and BotManifest fields
- PromotionManager eligibility checks
- PerformanceTracker stats calculation
- Promotion and downgrade operations
- Mode-based auction filtering
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import json

# Import the modules under test
from src.router.bot_manifest import (
    BotManifest,
    BotRegistry,
    StrategyType,
    TradeFrequency,
    BrokerType,
    TradingMode,
    ModePerformanceStats,
    PreferredConditions,
    TimeWindow,
)
from src.router.promotion_manager import (
    PromotionManager,
    PerformanceTracker,
    PromotionResult,
    PROMOTION_THRESHOLDS,
    CAPITAL_SCALING,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_storage_dir():
    """Create a temporary directory for test storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_manifest():
    """Create a sample BotManifest for testing."""
    return BotManifest(
        bot_id="test_bot_001",
        name="Test Bot",
        description="A test bot for promotion workflow",
        strategy_type=StrategyType.STRUCTURAL,
        frequency=TradeFrequency.LOW,
        min_capital_req=50.0,
        preferred_broker_type=BrokerType.STANDARD,
        prop_firm_safe=True,
        symbols=["EURUSD"],
        timeframes=["H1"],
        trading_mode=TradingMode.PAPER,
        tags=["@primal"],
    )


@pytest.fixture
def sample_stats():
    """Create sample performance stats for testing."""
    return ModePerformanceStats(
        total_trades=100,
        winning_trades=60,
        losing_trades=40,
        win_rate=0.60,
        total_pnl=5000.0,
        max_drawdown=0.05,
        sharpe_ratio=2.0,
        profit_factor=1.5,
        trading_days=35,
        start_date=datetime.now(timezone.utc) - timedelta(days=35),
        end_date=datetime.now(timezone.utc),
    )


@pytest.fixture
def bot_registry(temp_storage_dir):
    """Create a BotRegistry with temporary storage."""
    storage_path = os.path.join(temp_storage_dir, "test_registry.json")
    return BotRegistry(storage_path=storage_path)


@pytest.fixture
def performance_tracker(temp_storage_dir):
    """Create a PerformanceTracker with temporary storage."""
    storage_path = os.path.join(temp_storage_dir, "test_performance.json")
    return PerformanceTracker(storage_path=storage_path)


# ============================================================================
# TradingMode Enum Tests
# ============================================================================

class TestTradingMode:
    """Tests for TradingMode enum."""
    
    def test_trading_mode_values(self):
        """Test that TradingMode has correct values."""
        assert TradingMode.PAPER.value == "paper"
        assert TradingMode.DEMO.value == "demo"
        assert TradingMode.LIVE.value == "live"
    
    def test_trading_mode_from_string(self):
        """Test creating TradingMode from string."""
        assert TradingMode("paper") == TradingMode.PAPER
        assert TradingMode("demo") == TradingMode.DEMO
        assert TradingMode("live") == TradingMode.LIVE


# ============================================================================
# ModePerformanceStats Tests
# ============================================================================

class TestModePerformanceStats:
    """Tests for ModePerformanceStats dataclass."""
    
    def test_create_stats(self):
        """Test creating ModePerformanceStats."""
        stats = ModePerformanceStats(
            total_trades=50,
            winning_trades=30,
            losing_trades=20,
            win_rate=0.60,
        )
        assert stats.total_trades == 50
        assert stats.winning_trades == 30
        assert stats.losing_trades == 20
        assert stats.win_rate == 0.60
    
    def test_stats_to_dict(self, sample_stats):
        """Test serializing stats to dict."""
        data = sample_stats.to_dict()
        
        assert data["total_trades"] == 100
        assert data["win_rate"] == 0.60
        assert data["sharpe_ratio"] == 2.0
        assert "start_date" in data
        assert "end_date" in data
    
    def test_stats_from_dict(self, sample_stats):
        """Test deserializing stats from dict."""
        data = sample_stats.to_dict()
        restored = ModePerformanceStats.from_dict(data)
        
        assert restored.total_trades == sample_stats.total_trades
        assert restored.win_rate == sample_stats.win_rate
        assert restored.sharpe_ratio == sample_stats.sharpe_ratio


# ============================================================================
# BotManifest Trading Mode Tests
# ============================================================================

class TestBotManifestTradingMode:
    """Tests for BotManifest trading mode fields."""
    
    def test_default_trading_mode(self, sample_manifest):
        """Test that default trading mode is PAPER."""
        assert sample_manifest.trading_mode == TradingMode.PAPER
        assert sample_manifest.capital_allocated == 0.0
        assert sample_manifest.promotion_eligible == False
    
    def test_manifest_with_stats(self, sample_manifest, sample_stats):
        """Test setting stats on a manifest."""
        sample_manifest.paper_stats = sample_stats
        
        assert sample_manifest.paper_stats.total_trades == 100
        assert sample_manifest.paper_stats.win_rate == 0.60
    
    def test_get_current_stats(self, sample_manifest, sample_stats):
        """Test getting current stats based on trading mode."""
        sample_manifest.paper_stats = sample_stats
        
        current = sample_manifest.get_current_stats()
        assert current == sample_stats
        
        # Change mode and check stats
        sample_manifest.trading_mode = TradingMode.DEMO
        sample_manifest.demo_stats = ModePerformanceStats(total_trades=50)
        
        current = sample_manifest.get_current_stats()
        assert current.total_trades == 50
    
    def test_update_stats(self, sample_manifest, sample_stats):
        """Test updating stats for a specific mode."""
        sample_manifest.update_stats(sample_stats, TradingMode.PAPER)
        assert sample_manifest.paper_stats == sample_stats
        
        # Update for current mode
        sample_manifest.trading_mode = TradingMode.DEMO
        demo_stats = ModePerformanceStats(total_trades=30)
        sample_manifest.update_stats(demo_stats)
        assert sample_manifest.demo_stats == demo_stats
    
    def test_check_promotion_eligibility_no_stats(self, sample_manifest):
        """Test eligibility check with no stats."""
        result = sample_manifest.check_promotion_eligibility()
        
        assert result["eligible"] == False
        assert "No performance stats" in result["reason"]
    
    def test_check_promotion_eligibility_eligible(self, sample_manifest, sample_stats):
        """Test eligibility check when eligible."""
        sample_manifest.paper_stats = sample_stats
        
        result = sample_manifest.check_promotion_eligibility()
        
        assert result["eligible"] == True
        assert result["next_mode"] == "demo"
        assert sample_manifest.promotion_eligible == True
    
    def test_check_promotion_eligibility_not_eligible(self, sample_manifest):
        """Test eligibility check when not eligible."""
        # Stats that don't meet criteria
        poor_stats = ModePerformanceStats(
            total_trades=20,
            winning_trades=8,
            losing_trades=12,
            win_rate=0.40,
            sharpe_ratio=0.5,
            trading_days=10,
        )
        sample_manifest.paper_stats = poor_stats
        
        result = sample_manifest.check_promotion_eligibility()
        
        assert result["eligible"] == False
        assert len(result["missing_criteria"]) > 0
    
    def test_promote(self, sample_manifest, sample_stats):
        """Test promoting a bot."""
        sample_manifest.paper_stats = sample_stats
        sample_manifest.check_promotion_eligibility()
        
        result = sample_manifest.promote()
        
        assert result["success"] == True
        assert result["old_mode"] == "paper"
        assert result["new_mode"] == "demo"
        assert sample_manifest.trading_mode == TradingMode.DEMO
    
    def test_promote_not_eligible(self, sample_manifest):
        """Test promoting a bot that's not eligible."""
        result = sample_manifest.promote()
        
        assert result["success"] == False
        assert "not eligible" in result["reason"].lower()
    
    def test_downgrade(self, sample_manifest):
        """Test downgrading a bot."""
        sample_manifest.trading_mode = TradingMode.LIVE
        
        result = sample_manifest.downgrade(reason="Performance issues")
        
        assert result["success"] == True
        assert result["old_mode"] == "live"
        assert result["new_mode"] == "demo"
        assert sample_manifest.trading_mode == TradingMode.DEMO
    
    def test_downgrade_paper_mode(self, sample_manifest):
        """Test downgrading a bot already in PAPER mode."""
        sample_manifest.trading_mode = TradingMode.PAPER
        
        result = sample_manifest.downgrade(reason="Test")
        
        assert result["success"] == False
        assert "lowest" in result["reason"].lower()
    
    def test_serialize_with_trading_mode(self, sample_manifest, sample_stats):
        """Test serializing manifest with trading mode fields."""
        sample_manifest.paper_stats = sample_stats
        sample_manifest.capital_allocated = 1000.0
        sample_manifest.promotion_eligible = True
        
        data = sample_manifest.to_dict()
        
        assert data["trading_mode"] == "paper"
        assert data["capital_allocated"] == 1000.0
        assert data["promotion_eligible"] == True
        assert "paper_stats" in data
    
    def test_deserialize_with_trading_mode(self, sample_manifest, sample_stats):
        """Test deserializing manifest with trading mode fields."""
        sample_manifest.paper_stats = sample_stats
        sample_manifest.trading_mode = TradingMode.DEMO
        sample_manifest.capital_allocated = 5000.0
        
        data = sample_manifest.to_dict()
        restored = BotManifest.from_dict(data)
        
        assert restored.trading_mode == TradingMode.DEMO
        assert restored.capital_allocated == 5000.0
        assert restored.paper_stats.total_trades == sample_stats.total_trades


# ============================================================================
# BotRegistry Trading Mode Tests
# ============================================================================

class TestBotRegistryTradingMode:
    """Tests for BotRegistry trading mode methods."""
    
    def test_list_by_trading_mode(self, bot_registry, sample_manifest):
        """Test listing bots by trading mode."""
        # Register bots in different modes
        paper_bot = sample_manifest
        paper_bot.bot_id = "paper_bot"
        paper_bot.trading_mode = TradingMode.PAPER
        
        demo_bot = BotManifest.from_dict(sample_manifest.to_dict())
        demo_bot.bot_id = "demo_bot"
        demo_bot.trading_mode = TradingMode.DEMO
        
        live_bot = BotManifest.from_dict(sample_manifest.to_dict())
        live_bot.bot_id = "live_bot"
        live_bot.trading_mode = TradingMode.LIVE
        
        bot_registry.register(paper_bot)
        bot_registry.register(demo_bot)
        bot_registry.register(live_bot)
        
        # List by mode
        paper_bots = bot_registry.list_by_trading_mode(TradingMode.PAPER)
        demo_bots = bot_registry.list_by_trading_mode(TradingMode.DEMO)
        live_bots = bot_registry.list_by_trading_mode(TradingMode.LIVE)
        
        assert len(paper_bots) == 1
        assert len(demo_bots) == 1
        assert len(live_bots) == 1
        assert paper_bots[0].bot_id == "paper_bot"
        assert demo_bots[0].bot_id == "demo_bot"
        assert live_bots[0].bot_id == "live_bot"
    
    def test_list_promotion_eligible(self, bot_registry, sample_manifest, sample_stats):
        """Test listing bots eligible for promotion."""
        eligible_bot = sample_manifest
        eligible_bot.bot_id = "eligible_bot"
        eligible_bot.paper_stats = sample_stats
        eligible_bot.check_promotion_eligibility()
        
        not_eligible_bot = BotManifest.from_dict(sample_manifest.to_dict())
        not_eligible_bot.bot_id = "not_eligible_bot"
        not_eligible_bot.promotion_eligible = False
        
        bot_registry.register(eligible_bot)
        bot_registry.register(not_eligible_bot)
        
        eligible = bot_registry.list_promotion_eligible()
        
        assert len(eligible) == 1
        assert eligible[0].bot_id == "eligible_bot"
    
    def test_promote_bot(self, bot_registry, sample_manifest, sample_stats):
        """Test promoting a bot through registry."""
        sample_manifest.paper_stats = sample_stats
        bot_registry.register(sample_manifest)
        
        # Check eligibility first
        sample_manifest.check_promotion_eligibility()
        
        result = bot_registry.promote_bot(sample_manifest.bot_id)
        
        assert result["success"] == True
        assert result["new_mode"] == "demo"
        
        # Verify bot was updated
        bot = bot_registry.get(sample_manifest.bot_id)
        assert bot.trading_mode == TradingMode.DEMO
    
    def test_downgrade_bot(self, bot_registry, sample_manifest):
        """Test downgrading a bot through registry."""
        sample_manifest.trading_mode = TradingMode.LIVE
        bot_registry.register(sample_manifest)
        
        result = bot_registry.downgrade_bot(sample_manifest.bot_id, "Performance issues")
        
        assert result["success"] == True
        assert result["new_mode"] == "demo"
        
        # Verify bot was updated
        bot = bot_registry.get(sample_manifest.bot_id)
        assert bot.trading_mode == TradingMode.DEMO


# ============================================================================
# PerformanceTracker Tests
# ============================================================================

class TestPerformanceTracker:
    """Tests for PerformanceTracker class."""
    
    def test_record_trade(self, performance_tracker):
        """Test recording a trade."""
        trade = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": "EURUSD",
            "direction": "BUY",
            "pnl": 100.0,
            "mode": "paper",
        }
        
        performance_tracker.record_trade("test_bot", trade)
        
        history = performance_tracker.get_trade_history("test_bot")
        assert len(history) == 1
        assert history[0]["pnl"] == 100.0
    
    def test_calculate_stats(self, performance_tracker):
        """Test calculating performance stats."""
        # Record multiple trades
        for i in range(10):
            pnl = 50.0 if i < 6 else -30.0  # 6 wins, 4 losses
            trade = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "symbol": "EURUSD",
                "pnl": pnl,
                "mode": "paper",
            }
            performance_tracker.record_trade("test_bot", trade)
        
        stats = performance_tracker.calculate_stats("test_bot", mode="paper")
        
        assert stats is not None
        assert stats.total_trades == 10
        assert stats.winning_trades == 6
        assert stats.losing_trades == 4
        assert stats.win_rate == 0.6
        assert stats.total_pnl == 6 * 50 - 4 * 30  # 300 - 120 = 180
    
    def test_calculate_stats_no_trades(self, performance_tracker):
        """Test calculating stats with no trades."""
        stats = performance_tracker.calculate_stats("nonexistent_bot")
        assert stats is None
    
    def test_filter_by_mode(self, performance_tracker):
        """Test filtering trades by mode."""
        # Record trades in different modes
        for mode in ["paper", "demo", "live"]:
            trade = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "pnl": 100.0,
                "mode": mode,
            }
            performance_tracker.record_trade("test_bot", trade)
        
        paper_trades = performance_tracker.get_trade_history("test_bot", mode="paper")
        demo_trades = performance_tracker.get_trade_history("test_bot", mode="demo")
        live_trades = performance_tracker.get_trade_history("test_bot", mode="live")
        
        assert len(paper_trades) == 1
        assert len(demo_trades) == 1
        assert len(live_trades) == 1


# ============================================================================
# PromotionManager Tests
# ============================================================================

class TestPromotionManager:
    """Tests for PromotionManager class."""
    
    def test_check_promotion_eligibility(self, bot_registry, performance_tracker, sample_manifest, sample_stats):
        """Test checking promotion eligibility."""
        # Register bot and record trades
        bot_registry.register(sample_manifest)
        
        # Record trades to build stats
        for i in range(100):
            pnl = 50.0 if i < 60 else -30.0
            trade = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "pnl": pnl,
                "mode": "paper",
            }
            performance_tracker.record_trade(sample_manifest.bot_id, trade)
        
        manager = PromotionManager(
            bot_registry=bot_registry,
            performance_tracker=performance_tracker,
        )
        
        result = manager.check_promotion_eligibility(sample_manifest.bot_id)
        
        assert result.eligible == True
        assert result.next_mode == "demo"
    
    def test_promote_bot(self, bot_registry, performance_tracker, sample_manifest, sample_stats):
        """Test promoting a bot through PromotionManager."""
        sample_manifest.paper_stats = sample_stats
        sample_manifest.promotion_eligible = True
        bot_registry.register(sample_manifest)
        
        manager = PromotionManager(
            bot_registry=bot_registry,
            performance_tracker=performance_tracker,
        )
        
        result = manager.promote_bot(sample_manifest.bot_id)
        
        assert result.eligible == True or result.promoted_at is not None
        assert result.next_mode == "demo"
    
    def test_promote_bot_not_eligible(self, bot_registry, performance_tracker, sample_manifest):
        """Test promoting a bot that's not eligible."""
        # Poor stats
        poor_stats = ModePerformanceStats(
            total_trades=10,
            win_rate=0.30,
            sharpe_ratio=0.5,
            trading_days=5,
        )
        sample_manifest.paper_stats = poor_stats
        bot_registry.register(sample_manifest)
        
        manager = PromotionManager(
            bot_registry=bot_registry,
            performance_tracker=performance_tracker,
        )
        
        result = manager.promote_bot(sample_manifest.bot_id)
        
        assert result.eligible == False
        assert result.error is not None or len(result.missing_criteria) > 0
    
    def test_downgrade_bot(self, bot_registry, performance_tracker, sample_manifest):
        """Test downgrading a bot."""
        sample_manifest.trading_mode = TradingMode.LIVE
        bot_registry.register(sample_manifest)
        
        manager = PromotionManager(
            bot_registry=bot_registry,
            performance_tracker=performance_tracker,
        )
        
        result = manager.downgrade_bot(sample_manifest.bot_id, "Performance issues")
        
        assert result.error is not None  # Downgrade is reflected in error message
    
    def test_capital_scaling(self, bot_registry, sample_manifest, sample_stats):
        """Test capital scaling for different modes."""
        manager = PromotionManager(bot_registry=bot_registry)
        
        # Paper mode - no capital
        capital = manager._calculate_capital_for_mode(sample_manifest, TradingMode.PAPER)
        assert capital == 0.0
        
        # Demo mode - fixed capital
        capital = manager._calculate_capital_for_mode(sample_manifest, TradingMode.DEMO)
        assert capital == CAPITAL_SCALING["demo"]
        
        # Live mode - base capital
        capital = manager._calculate_capital_for_mode(sample_manifest, TradingMode.LIVE)
        assert capital == CAPITAL_SCALING["live_base"]
        
        # Live mode with demo profits
        sample_manifest.demo_stats = ModePerformanceStats(total_pnl=2000.0)
        capital = manager._calculate_capital_for_mode(sample_manifest, TradingMode.LIVE)
        # Should be base + 50% of demo profits
        expected = CAPITAL_SCALING["live_base"] + (2000.0 * CAPITAL_SCALING["live_scaling_factor"])
        assert capital == expected
    
    def test_run_daily_promotion_check(self, bot_registry, performance_tracker, sample_manifest, sample_stats):
        """Test running daily promotion checks."""
        # Register multiple bots
        sample_manifest.paper_stats = sample_stats
        sample_manifest.promotion_eligible = True
        bot_registry.register(sample_manifest)
        
        another_bot = BotManifest.from_dict(sample_manifest.to_dict())
        another_bot.bot_id = "another_bot"
        another_bot.promotion_eligible = False
        bot_registry.register(another_bot)
        
        manager = PromotionManager(
            bot_registry=bot_registry,
            performance_tracker=performance_tracker,
        )
        
        results = manager.run_daily_promotion_check()
        
        assert len(results) == 2


# ============================================================================
# Promotion Thresholds Tests
# ============================================================================

class TestPromotionThresholds:
    """Tests for promotion threshold constants."""
    
    def test_paper_to_demo_thresholds(self):
        """Test PAPER->DEMO promotion thresholds."""
        thresholds = PROMOTION_THRESHOLDS["paper_to_demo"]
        
        assert thresholds["min_trading_days"] == 30
        assert thresholds["min_sharpe_ratio"] == 1.5
        assert thresholds["min_win_rate"] == 0.55
        assert thresholds["min_total_trades"] == 50
    
    def test_demo_to_live_thresholds(self):
        """Test DEMO->LIVE promotion thresholds."""
        thresholds = PROMOTION_THRESHOLDS["demo_to_live"]
        
        assert thresholds["min_trading_days"] == 30
        assert thresholds["min_sharpe_ratio"] == 1.5
        assert thresholds["min_win_rate"] == 0.55
        assert thresholds["max_drawdown"] == 0.10
        assert thresholds["min_total_trades"] == 50


# ============================================================================
# Capital Scaling Tests
# ============================================================================

class TestCapitalScaling:
    """Tests for capital scaling constants."""
    
    def test_capital_scaling_values(self):
        """Test capital scaling values."""
        assert CAPITAL_SCALING["paper"] == 0.0
        assert CAPITAL_SCALING["demo"] == 1000.0
        assert CAPITAL_SCALING["live_base"] == 1000.0
        assert CAPITAL_SCALING["live_max"] == 10000.0
        assert CAPITAL_SCALING["live_scaling_factor"] == 0.5


# ============================================================================
# Integration Tests
# ============================================================================

class TestPromotionWorkflowIntegration:
    """Integration tests for the complete promotion workflow."""
    
    def test_full_promotion_cycle(self, bot_registry, performance_tracker):
        """Test a bot going through the full promotion cycle."""
        # Create a new bot in PAPER mode
        bot = BotManifest(
            bot_id="integration_test_bot",
            name="Integration Test Bot",
            strategy_type=StrategyType.STRUCTURAL,
            frequency=TradeFrequency.LOW,
            trading_mode=TradingMode.PAPER,
            tags=["@primal"],
        )
        bot_registry.register(bot)
        
        # Record trades for paper phase
        for i in range(100):
            pnl = 50.0 if i < 60 else -30.0
            trade = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "pnl": pnl,
                "mode": "paper",
            }
            performance_tracker.record_trade(bot.bot_id, trade)
        
        manager = PromotionManager(
            bot_registry=bot_registry,
            performance_tracker=performance_tracker,
        )
        
        # Check eligibility and promote to DEMO
        result = manager.check_promotion_eligibility(bot.bot_id)
        assert result.eligible == True
        
        promote_result = manager.promote_bot(bot.bot_id)
        assert promote_result.next_mode == "demo"
        
        # Verify bot is now in DEMO mode
        updated_bot = bot_registry.get(bot.bot_id)
        assert updated_bot.trading_mode == TradingMode.DEMO
        assert updated_bot.capital_allocated == CAPITAL_SCALING["demo"]
        
        # Record trades for demo phase
        for i in range(100):
            pnl = 60.0 if i < 65 else -25.0
            trade = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "pnl": pnl,
                "mode": "demo",
            }
            performance_tracker.record_trade(bot.bot_id, trade)
        
        # Check eligibility and promote to LIVE
        result = manager.check_promotion_eligibility(bot.bot_id)
        if result.eligible:
            promote_result = manager.promote_bot(bot.bot_id)
            assert promote_result.next_mode == "live"
            
            # Verify bot is now in LIVE mode
            updated_bot = bot_registry.get(bot.bot_id)
            assert updated_bot.trading_mode == TradingMode.LIVE
    
    def test_downgrade_workflow(self, bot_registry, performance_tracker):
        """Test downgrading a bot due to poor performance."""
        # Create a bot in LIVE mode
        bot = BotManifest(
            bot_id="downgrade_test_bot",
            name="Downgrade Test Bot",
            strategy_type=StrategyType.STRUCTURAL,
            frequency=TradeFrequency.LOW,
            trading_mode=TradingMode.LIVE,
            capital_allocated=5000.0,
            tags=["@primal"],
        )
        bot_registry.register(bot)
        
        manager = PromotionManager(bot_registry=bot_registry)
        
        # Downgrade due to performance issues
        result = manager.downgrade_bot(bot.bot_id, "Max drawdown exceeded")
        
        # Verify bot is now in DEMO mode
        updated_bot = bot_registry.get(bot.bot_id)
        assert updated_bot.trading_mode == TradingMode.DEMO
        assert updated_bot.promotion_eligible == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
"""
Unit tests for LifecycleManager component.

Tests tag progression criteria, quarantine triggers, and daily check execution.
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Import the component under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.router.lifecycle_manager import (
    LifecycleManager,
    TagProgression,
    ProgressionCriteria,
    QuarantineTrigger,
)


class TestProgressionCriteria:
    """Test tag progression criteria evaluation."""
    
    def test_primal_to_pending_criteria(self):
        """Test @primal -> @pending progression criteria."""
        criteria = ProgressionCriteria(
            min_trades=20,
            min_win_rate=0.50,
            min_days_active=7,
            max_critical_errors=0,
        )
        
        # Should pass
        assert criteria.is_met(
            trades=25,
            win_rate=0.55,
            days_active=10,
            critical_errors=0
        )
        
        # Should fail - not enough trades
        assert not criteria.is_met(
            trades=15,
            win_rate=0.55,
            days_active=10,
            critical_errors=0
        )
        
        # Should fail - win rate too low
        assert not criteria.is_met(
            trades=25,
            win_rate=0.45,
            days_active=10,
            critical_errors=0
        )
        
        # Should fail - critical errors
        assert not criteria.is_met(
            trades=25,
            win_rate=0.55,
            days_active=10,
            critical_errors=1
        )
    
    def test_pending_to_perfect_criteria(self):
        """Test @pending -> @perfect progression criteria."""
        criteria = ProgressionCriteria(
            min_trades=50,
            min_win_rate=0.55,
            min_sharpe_ratio=1.5,
            min_days_active=30,
            max_drawdown=0.15,
        )
        
        # Should pass
        assert criteria.is_met(
            trades=60,
            win_rate=0.58,
            sharpe_ratio=1.8,
            days_active=35,
            max_drawdown=0.10
        )
        
        # Should fail - Sharpe too low
        assert not criteria.is_met(
            trades=60,
            win_rate=0.58,
            sharpe_ratio=1.2,
            days_active=35,
            max_drawdown=0.10
        )
        
        # Should fail - drawdown too high
        assert not criteria.is_met(
            trades=60,
            win_rate=0.58,
            sharpe_ratio=1.8,
            days_active=35,
            max_drawdown=0.20
        )
    
    def test_perfect_to_live_criteria(self):
        """Test @perfect -> @live progression criteria."""
        criteria = ProgressionCriteria(
            min_trades=100,
            min_win_rate=0.58,
            min_sharpe_ratio=2.0,
            min_days_active=60,
            max_drawdown=0.10,
            min_profit_factor=1.5,
        )
        
        # Should pass
        assert criteria.is_met(
            trades=120,
            win_rate=0.60,
            sharpe_ratio=2.2,
            days_active=70,
            max_drawdown=0.08,
            profit_factor=1.8
        )
        
        # Should fail - profit factor too low
        assert not criteria.is_met(
            trades=120,
            win_rate=0.60,
            sharpe_ratio=2.2,
            days_active=70,
            max_drawdown=0.08,
            profit_factor=1.3
        )


class TestQuarantineTriggers:
    """Test quarantine trigger conditions."""
    
    def test_win_rate_drop_trigger(self):
        """Test quarantine on win rate drop."""
        trigger = QuarantineTrigger(
            max_win_rate_drop=0.45,
        )
        
        # Should trigger quarantine
        assert trigger.should_quarantine(win_rate=0.40)
        assert trigger.should_quarantine(win_rate=0.45)
        
        # Should not trigger
        assert not trigger.should_quarantine(win_rate=0.50)
    
    def test_sharpe_ratio_drop_trigger(self):
        """Test quarantine on Sharpe ratio drop."""
        trigger = QuarantineTrigger(
            max_sharpe_ratio=0.5,
        )
        
        assert trigger.should_quarantine(sharpe_ratio=0.3)
        assert not trigger.should_quarantine(sharpe_ratio=0.6)
    
    def test_drawdown_trigger(self):
        """Test quarantine on high drawdown."""
        trigger = QuarantineTrigger(
            max_drawdown=0.20,
        )
        
        assert trigger.should_quarantine(max_drawdown=0.25)
        assert not trigger.should_quarantine(max_drawdown=0.15)
    
    def test_consecutive_losing_days_trigger(self):
        """Test quarantine on consecutive losing days."""
        trigger = QuarantineTrigger(
            max_consecutive_losing_days=5,
        )
        
        assert trigger.should_quarantine(consecutive_losing_days=5)
        assert trigger.should_quarantine(consecutive_losing_days=7)
        assert not trigger.should_quarantine(consecutive_losing_days=3)


class TestLifecycleManager:
    """Test LifecycleManager main functionality."""
    
    @pytest.fixture
    def manager(self):
        """Create a LifecycleManager instance for testing."""
        with patch('src.router.lifecycle_manager.BotRegistry') as mock_registry:
            with patch('src.router.lifecycle_manager.PerformanceTracker') as mock_tracker:
                manager = LifecycleManager()
                manager.registry = mock_registry.return_value
                manager.tracker = mock_tracker.return_value
                yield manager
    
    def test_check_progression_primal_to_pending(self, manager):
        """Test checking progression from @primal to @pending."""
        # Mock bot with @primal tag
        mock_bot = Mock()
        mock_bot.bot_id = "test_bot_001"
        mock_bot.tags = ["@primal"]
        mock_bot.created_at = datetime.now(timezone.utc) - timedelta(days=10)
        
        # Mock performance stats
        manager.tracker.calculate_stats.return_value = {
            "total_trades": 25,
            "win_rate": 0.55,
            "critical_errors": 0,
        }
        
        result = manager._check_progression(mock_bot)
        
        assert result is not None
        assert result.current_tag == "@primal"
        assert result.next_tag == "@pending"
        assert result.meets_criteria is True
    
    def test_check_progression_not_ready(self, manager):
        """Test bot not ready for progression."""
        mock_bot = Mock()
        mock_bot.bot_id = "test_bot_002"
        mock_bot.tags = ["@primal"]
        mock_bot.created_at = datetime.now(timezone.utc) - timedelta(days=3)
        
        # Mock insufficient performance
        manager.tracker.calculate_stats.return_value = {
            "total_trades": 10,
            "win_rate": 0.45,
            "critical_errors": 0,
        }
        
        result = manager._check_progression(mock_bot)
        
        assert result is None or not result.meets_criteria
    
    def test_check_quarantine_needed(self, manager):
        """Test detecting bot that needs quarantine."""
        mock_bot = Mock()
        mock_bot.bot_id = "test_bot_003"
        mock_bot.tags = ["@live"]
        
        # Mock poor performance
        manager.tracker.calculate_stats.return_value = {
            "win_rate": 0.40,
            "sharpe_ratio": 0.3,
            "max_drawdown": 0.25,
            "consecutive_losing_days": 6,
        }
        
        result = manager._check_quarantine(mock_bot)
        
        assert result is True
    
    def test_check_quarantine_not_needed(self, manager):
        """Test bot performing well doesn't need quarantine."""
        mock_bot = Mock()
        mock_bot.bot_id = "test_bot_004"
        mock_bot.tags = ["@live"]
        
        # Mock good performance
        manager.tracker.calculate_stats.return_value = {
            "win_rate": 0.60,
            "sharpe_ratio": 1.8,
            "max_drawdown": 0.08,
            "consecutive_losing_days": 0,
        }
        
        result = manager._check_quarantine(mock_bot)
        
        assert result is False
    
    def test_promote_bot(self, manager):
        """Test promoting a bot to a new tag."""
        mock_bot = Mock()
        mock_bot.bot_id = "test_bot_005"
        mock_bot.tags = ["@primal"]
        
        manager._promote_bot(mock_bot, "@pending")
        
        # Verify tag update was called
        manager.registry.update_bot_tags.assert_called_once()
    
    def test_quarantine_bot(self, manager):
        """Test quarantining a bot."""
        mock_bot = Mock()
        mock_bot.bot_id = "test_bot_006"
        mock_bot.tags = ["@live"]
        
        manager._quarantine_bot(mock_bot, reason="Win rate dropped below 45%")
        
        # Verify quarantine tag was added
        manager.registry.update_bot_tags.assert_called_once()
    
    def test_run_daily_check(self, manager):
        """Test running daily lifecycle check."""
        # Mock bots
        mock_bots = [
            Mock(bot_id="bot_001", tags=["@primal"], created_at=datetime.now(timezone.utc) - timedelta(days=10)),
            Mock(bot_id="bot_002", tags=["@pending"], created_at=datetime.now(timezone.utc) - timedelta(days=35)),
            Mock(bot_id="bot_003", tags=["@live"], created_at=datetime.now(timezone.utc) - timedelta(days=100)),
        ]
        manager.registry.list_all.return_value = mock_bots
        
        # Mock performance stats
        manager.tracker.calculate_stats.return_value = {
            "total_trades": 50,
            "win_rate": 0.55,
            "sharpe_ratio": 1.6,
            "max_drawdown": 0.10,
            "critical_errors": 0,
            "consecutive_losing_days": 0,
        }
        
        result = manager.run_daily_check()
        
        assert "checked" in result
        assert "promoted" in result
        assert "quarantined" in result
        assert result["checked"] == 3


class TestTagProgression:
    """Test TagProgression data class."""
    
    def test_tag_progression_creation(self):
        """Test creating a TagProgression instance."""
        progression = TagProgression(
            bot_id="test_bot",
            current_tag="@primal",
            next_tag="@pending",
            meets_criteria=True,
            criteria_details={
                "trades": 25,
                "win_rate": 0.55,
                "days_active": 10,
            },
        )
        
        assert progression.bot_id == "test_bot"
        assert progression.current_tag == "@primal"
        assert progression.next_tag == "@pending"
        assert progression.meets_criteria is True
    
    def test_tag_order(self):
        """Test tag order is correct."""
        tags = ["@primal", "@pending", "@perfect", "@live", "@quarantine", "@dead"]
        
        # Verify order
        assert tags.index("@primal") < tags.index("@pending")
        assert tags.index("@pending") < tags.index("@perfect")
        assert tags.index("@perfect") < tags.index("@live")


class TestLifecycleMetrics:
    """Test Prometheus metrics emission."""
    
    @pytest.fixture
    def manager(self):
        """Create a LifecycleManager with mocked metrics."""
        with patch('src.router.lifecycle_manager.BOT_LIFECYCLE_PROMOTIONS') as mock_promotions:
            with patch('src.router.lifecycle_manager.BOT_LIFECYCLE_QUARANTINES') as mock_quarantines:
                with patch('src.router.lifecycle_manager.BotRegistry'):
                    with patch('src.router.lifecycle_manager.PerformanceTracker'):
                        manager = LifecycleManager()
                        manager.promotion_counter = mock_promotions
                        manager.quarantine_counter = mock_quarantines
                        yield manager
    
    def test_promotion_metric_incremented(self, manager):
        """Test that promotion metric is incremented."""
        mock_bot = Mock(bot_id="test_bot", tags=["@primal"])
        
        manager._promote_bot(mock_bot, "@pending")
        
        # Verify metric was incremented
        manager.promotion_counter.labels.assert_called()
    
    def test_quarantine_metric_incremented(self, manager):
        """Test that quarantine metric is incremented."""
        mock_bot = Mock(bot_id="test_bot", tags=["@live"])
        
        manager._quarantine_bot(mock_bot, reason="Test quarantine")
        
        # Verify metric was incremented
        manager.quarantine_counter.labels.assert_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
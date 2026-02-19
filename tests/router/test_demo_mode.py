"""
Unit Tests for Demo Mode Implementation

Tests EA registration, virtual balance tracking, mode-aware risk calculations,
and mode-aware execution routing.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock

# Test EA Registry
class TestEARegistry:
    """Tests for EA Registry module."""
    
    def test_ea_config_creation_live_mode(self):
        """Test creating EA config in live mode."""
        from src.router.ea_registry import EAConfig, EAMode
        
        config = EAConfig(
            ea_id="test_ea_001",
            name="Test EA",
            symbol="EURUSD",
            timeframe="H1",
            magic_number=12345,
            mode=EAMode.LIVE
        )
        
        assert config.ea_id == "test_ea_001"
        assert config.mode == EAMode.LIVE
        assert "@live" in config.tags
        assert "@demo" not in config.tags
    
    def test_ea_config_creation_demo_mode(self):
        """Test creating EA config in demo mode."""
        from src.router.ea_registry import EAConfig, EAMode
        
        config = EAConfig(
            ea_id="test_ea_002",
            name="Demo EA",
            symbol="GBPUSD",
            timeframe="M15",
            magic_number=67890,
            mode=EAMode.DEMO,
            virtual_balance=5000.0
        )
        
        assert config.ea_id == "test_ea_002"
        assert config.mode == EAMode.DEMO
        assert config.virtual_balance == 5000.0
        assert "@demo" in config.tags
        assert "@live" not in config.tags
    
    def test_ea_registry_register(self):
        """Test registering an EA."""
        from src.router.ea_registry import EARegistry, EAConfig, EAMode
        
        registry = EARegistry()
        config = EAConfig(
            ea_id="test_001",
            name="Test EA",
            symbol="EURUSD",
            timeframe="H1",
            magic_number=11111,
            mode=EAMode.DEMO
        )
        
        result = registry.register(config)
        assert result is True
        assert registry.get("test_001") == config
    
    def test_ea_registry_get_by_mode(self):
        """Test filtering EAs by mode."""
        from src.router.ea_registry import EARegistry, EAConfig, EAMode
        
        registry = EARegistry()
        
        # Register demo EA
        demo_config = EAConfig(
            ea_id="demo_001",
            name="Demo EA",
            symbol="EURUSD",
            timeframe="H1",
            magic_number=11111,
            mode=EAMode.DEMO
        )
        registry.register(demo_config)
        
        # Register live EA
        live_config = EAConfig(
            ea_id="live_001",
            name="Live EA",
            symbol="GBPUSD",
            timeframe="H1",
            magic_number=22222,
            mode=EAMode.LIVE
        )
        registry.register(live_config)
        
        demo_eas = registry.get_demo_eas()
        live_eas = registry.get_live_eas()
        
        assert len(demo_eas) == 1
        assert len(live_eas) == 1
        assert demo_eas[0].ea_id == "demo_001"
        assert live_eas[0].ea_id == "live_001"
    
    def test_ea_promotion(self):
        """Test promoting EA from demo to live."""
        from src.router.ea_registry import EARegistry, EAConfig, EAMode
        
        registry = EARegistry()
        config = EAConfig(
            ea_id="promo_001",
            name="Promotion Test",
            symbol="EURUSD",
            timeframe="H1",
            magic_number=33333,
            mode=EAMode.DEMO
        )
        registry.register(config)
        
        # Promote to live
        result = registry.promote_to_live("promo_001")
        assert result is True
        
        updated = registry.get("promo_001")
        assert updated.mode == EAMode.LIVE
        assert "@live" in updated.tags
        assert "@demo" not in updated.tags
    
    def test_ea_demotion(self):
        """Test demoting EA from live to demo."""
        from src.router.ea_registry import EARegistry, EAConfig, EAMode
        
        registry = EARegistry()
        config = EAConfig(
            ea_id="demote_001",
            name="Demotion Test",
            symbol="EURUSD",
            timeframe="H1",
            magic_number=44444,
            mode=EAMode.LIVE
        )
        registry.register(config)
        
        # Demote to demo
        result = registry.demote_to_demo("demote_001", virtual_balance=2000.0)
        assert result is True
        
        updated = registry.get("demote_001")
        assert updated.mode == EAMode.DEMO
        assert updated.virtual_balance == 2000.0
        assert "@demo" in updated.tags


# Test Virtual Balance Manager
class TestVirtualBalanceManager:
    """Tests for Virtual Balance Manager module."""
    
    def test_virtual_account_creation(self):
        """Test creating a virtual account."""
        from src.router.virtual_balance import VirtualBalanceManager
        
        manager = VirtualBalanceManager()
        account = manager.create_account("ea_001", initial_balance=1000.0)
        
        assert account.ea_id == "ea_001"
        assert account.initial_balance == 1000.0
        assert account.current_balance == 1000.0
        assert account.equity == 1000.0
        assert account.free_margin == 1000.0
    
    def test_virtual_account_balance_update(self):
        """Test updating virtual account balance."""
        from src.router.virtual_balance import VirtualBalanceManager
        
        manager = VirtualBalanceManager()
        manager.create_account("ea_002", initial_balance=1000.0)
        
        # Profit trade
        new_balance = manager.update_after_trade("ea_002", profit_loss=100.0)
        assert new_balance == 1100.0
        
        # Loss trade
        new_balance = manager.update_after_trade("ea_002", profit_loss=-50.0)
        assert new_balance == 1050.0
    
    def test_virtual_account_margin_tracking(self):
        """Test margin tracking in virtual account."""
        from src.router.virtual_balance import VirtualBalanceManager, VirtualAccount
        
        manager = VirtualBalanceManager()
        account = manager.create_account("ea_003", initial_balance=1000.0)
        
        # Use margin
        can_trade = account.use_margin(200.0)
        assert can_trade is True
        assert account.margin_used == 200.0
        assert account.free_margin == 800.0
        
        # Release margin
        account.release_margin(100.0)
        assert account.margin_used == 100.0
        assert account.free_margin == 900.0
    
    def test_virtual_account_can_trade(self):
        """Test checking if account can trade."""
        from src.router.virtual_balance import VirtualBalanceManager
        
        manager = VirtualBalanceManager()
        account = manager.create_account("ea_004", initial_balance=1000.0)
        
        # Should be able to trade with available margin
        assert account.can_trade(500.0) is True
        assert account.can_trade(1500.0) is False
    
    def test_virtual_account_reset(self):
        """Test resetting virtual account."""
        from src.router.virtual_balance import VirtualBalanceManager
        
        manager = VirtualBalanceManager()
        manager.create_account("ea_005", initial_balance=1000.0)
        
        # Make some trades
        manager.update_after_trade("ea_005", profit_loss=200.0)
        
        # Reset
        manager.reset_account("ea_005")
        account = manager.get_account("ea_005")
        
        assert account.current_balance == 1000.0  # Reset to initial
        assert account.equity == 1000.0
    
    def test_virtual_account_summary(self):
        """Test getting virtual account summary."""
        from src.router.virtual_balance import VirtualBalanceManager
        
        manager = VirtualBalanceManager()
        manager.create_account("ea_006", initial_balance=1000.0)
        manager.create_account("ea_007", initial_balance=2000.0)
        
        manager.update_after_trade("ea_006", profit_loss=100.0)
        manager.update_after_trade("ea_007", profit_loss=-50.0)
        
        summary = manager.get_summary()
        
        assert summary["total_accounts"] == 2
        assert summary["total_balance"] == 3050.0  # 1100 + 1950
        assert summary["total_pnl"] == 50.0  # 100 - 50


# Test Governor Mode Awareness
class TestGovernorModeAwareness:
    """Tests for Governor mode awareness."""
    
    def test_risk_mandate_mode_field(self):
        """Test RiskMandate has mode field."""
        from src.router.governor import RiskMandate
        
        mandate = RiskMandate(
            allocation_scalar=0.5,
            risk_mode="CLAMPED",
            mode="demo"
        )
        
        assert mandate.mode == "demo"
    
    def test_governor_calculate_risk_with_demo_mode(self):
        """Test Governor calculates risk with demo mode."""
        from src.router.governor import Governor, RiskMandate
        from src.router.sentinel import RegimeReport
        
        governor = Governor()
        
        # Create mock regime report
        regime_report = Mock(spec=RegimeReport)
        regime_report.chaos_score = 0.4
        regime_report.is_systemic_risk = False
        
        trade_proposal = {
            'bot_id': 'test_bot',
            'symbol': 'EURUSD'
        }
        
        mandate = governor.calculate_risk(
            regime_report=regime_report,
            trade_proposal=trade_proposal,
            mode="demo"
        )
        
        assert mandate.mode == "demo"
        assert "[DEMO]" in mandate.notes
    
    def test_governor_calculate_risk_with_live_mode(self):
        """Test Governor calculates risk with live mode."""
        from src.router.governor import Governor
        from src.router.sentinel import RegimeReport
        
        governor = Governor()
        
        # Create mock regime report
        regime_report = Mock(spec=RegimeReport)
        regime_report.chaos_score = 0.4
        regime_report.is_systemic_risk = False
        
        trade_proposal = {
            'bot_id': 'test_bot',
            'symbol': 'EURUSD'
        }
        
        mandate = governor.calculate_risk(
            regime_report=regime_report,
            trade_proposal=trade_proposal,
            mode="live"
        )
        
        assert mandate.mode == "live"
        assert "[LIVE]" in mandate.notes
    
    def test_governor_demo_mode_higher_risk(self):
        """Test demo mode allows higher risk."""
        from src.router.governor import Governor
        from src.router.sentinel import RegimeReport
        
        governor = Governor()
        
        regime_report = Mock(spec=RegimeReport)
        regime_report.chaos_score = 0.5
        regime_report.is_systemic_risk = False
        
        trade_proposal = {'bot_id': 'test', 'symbol': 'EURUSD'}
        
        demo_mandate = governor.calculate_risk(
            regime_report=regime_report,
            trade_proposal=trade_proposal,
            mode="demo"
        )
        
        live_mandate = governor.calculate_risk(
            regime_report=regime_report,
            trade_proposal=trade_proposal,
            mode="live"
        )
        
        # Demo should have higher allocation (1.5x multiplier)
        assert demo_mandate.allocation_scalar > live_mandate.allocation_scalar


# Test Enhanced Governor Mode Awareness
class TestEnhancedGovernorModeAwareness:
    """Tests for Enhanced Governor mode awareness."""
    
    @patch('src.router.enhanced_governor.DB_AVAILABLE', False)
    def test_enhanced_governor_accepts_mode_parameter(self):
        """Test Enhanced Governor accepts mode parameter."""
        from src.router.enhanced_governor import EnhancedGovernor
        from src.router.sentinel import RegimeReport
        
        governor = EnhancedGovernor(account_id="test_account")
        
        regime_report = Mock(spec=RegimeReport)
        regime_report.chaos_score = 0.3
        regime_report.is_systemic_risk = False
        regime_report.regime_quality = 0.7
        
        trade_proposal = {
            'bot_id': 'test_bot',
            'symbol': 'EURUSD',
            'current_balance': 10000.0
        }
        
        # Should not raise error with mode parameter
        mandate = governor.calculate_risk(
            regime_report=regime_report,
            trade_proposal=trade_proposal,
            mode="demo"
        )
        
        assert mandate is not None


# Test Trading Mode Enum
class TestTradingModeEnum:
    """Tests for TradingMode enum in database models."""
    
    def test_trading_mode_values(self):
        """Test TradingMode enum values."""
        from src.database.models import TradingMode
        
        assert TradingMode.DEMO.value == "demo"
        assert TradingMode.LIVE.value == "live"
    
    def test_trading_mode_count(self):
        """Test TradingMode has exactly two values."""
        from src.database.models import TradingMode
        
        assert len(list(TradingMode)) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
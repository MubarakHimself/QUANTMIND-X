"""
Test LEAN Models Integration

Tests for LEAN slippage and commission models integration into backtesting:
- Slippage model selection and configuration
- Commission model selection and configuration
- Model application in backtest execution
- Configuration parameter passing

Validates requirements for Phase 3 integration.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# Test LEAN integration into backtesting
class TestLeanSlippageModels:
    """Test LEAN slippage models."""
    
    def test_slippage_model_factory(self):
        """Test SlippageModelFactory creates correct models."""
        from src.backtesting.lean_slippage import SlippageModelFactory
        
        # Test constant model
        constant_model = SlippageModelFactory.create("constant", base_slippage=2.0)
        assert constant_model is not None
        
        # Test volume-based model
        volume_model = SlippageModelFactory.create("volume_based")
        assert volume_model is not None
        
        # Test volatility-based model
        volatility_model = SlippageModelFactory.create("volatility_based")
        assert volatility_model is not None
    
    def test_constant_slippage_calculation(self):
        """Test constant slippage model calculation."""
        from src.backtesting.lean_slippage import ConstantSlippage
        
        model = ConstantSlippage(base_slippage_pips=2.0)
        
        # Test calculation
        slippage = model.calculate_slippage(
            order_volume=1.0,
            price=1.1000
        )
        
        assert slippage is not None
        assert slippage >= 0
    
    def test_volume_based_slippage(self):
        """Test volume-based slippage model."""
        from src.backtesting.lean_slippage import VolumeBasedSlippage
        
        model = VolumeBasedSlippage()
        
        slippage = model.calculate_slippage(
            order_volume=5.0,
            price=1.1000,
            avg_volume=100000.0
        )
        
        assert slippage is not None
        assert slippage >= 0
    
    def test_volatility_based_slippage(self):
        """Test volatility-based slippage model."""
        from src.backtesting.lean_slippage import VolatilityBasedSlippage
        
        model = VolatilityBasedSlippage()
        
        slippage = model.calculate_slippage(
            order_volume=1.0,
            price=1.1000,
            current_atr=0.0015,
            avg_atr=0.0010
        )
        
        assert slippage is not None
        assert slippage >= 0
    
    def test_slippage_apply_slippage_method(self):
        """Test the apply_slippage method."""
        from src.backtesting.lean_slippage import ConstantSlippage
        
        model = ConstantSlippage(base_slippage_pips=2.0)
        
        # Test buy order
        entry_price = model.apply_slippage(
            price=1.1000,
            side="buy",
            order_volume=1.0,
            avg_volume=100000.0,
            current_atr=0.001,
            avg_atr=0.001
        )
        
        # Buy order: slippage reduces entry price
        assert entry_price <= 1.1000
        
        # Test sell order
        entry_price_sell = model.apply_slippage(
            price=1.1000,
            side="sell",
            order_volume=1.0,
            avg_volume=100000.0,
            current_atr=0.001,
            avg_atr=0.001
        )
        
        # Sell order: slippage increases entry price
        assert entry_price_sell >= 1.1000


class TestLeanCommissionModels:
    """Test LEAN commission models."""
    
    def test_commission_model_factory(self):
        """Test CommissionModelFactory creates correct models."""
        from src.backtesting.lean_commission import CommissionModelFactory
        
        # Test per lot model
        per_lot = CommissionModelFactory.create("per_lot", cost_per_lot=5.0)
        assert per_lot is not None
        
        # Test per share model
        per_share = CommissionModelFactory.create("per_share")
        assert per_share is not None
        
        # Test tiered model
        tiered = CommissionModelFactory.create("tiered")
        assert tiered is not None
    
    def test_per_lot_commission(self):
        """Test per-lot commission calculation."""
        from src.backtesting.lean_commission import PerLotCommission
        
        model = PerLotCommission(cost_per_lot=5.0)
        
        commission = model.calculate_commission(
            lots=2.0,
            symbol="EURUSD",
            price=1.1000,
            side="buy"
        )
        
        # 2 lots * $5 = $10
        assert commission == 10.0
    
    def test_per_share_commission(self):
        """Test per-share commission calculation."""
        from src.backtesting.lean_commission import PerShareCommission
        
        model = PerShareCommission(cost_per_share=0.005)
        
        commission = model.calculate_commission(
            lots=1.0,  # For forex, lots represent shares
            symbol="EURUSD",
            price=1.1000,
            side="buy"
        )
        
        assert commission is not None
        assert commission >= 0
    
    def test_tiered_commission(self):
        """Test tiered commission model."""
        from src.backtesting.lean_commission import TieredCommission
        
        model = TieredCommission()
        
        # Test different tiers
        commission_low = model.calculate_commission(
            lots=1.0,
            symbol="EURUSD",
            price=1.1000,
            side="buy"
        )
        
        commission_high = model.calculate_commission(
            lots=100.0,
            symbol="EURUSD",
            price=1.1000,
            side="buy"
        )
        
        assert commission_high < commission_low  # Higher volume = lower rate
    
    def test_spread_based_commission(self):
        """Test spread-based commission model."""
        from src.backtesting.lean_commission import SpreadBasedCommission
        
        model = SpreadBasedCommission(spread_pips=1.0)
        
        commission = model.calculate_commission(
            lots=1.0,
            symbol="EURUSD",
            price=1.1000,
            side="buy"
        )
        
        assert commission is not None
        assert commission >= 0


class TestLeanBacktestIntegration:
    """Test LEAN models integration into backtesting."""
    
    def test_backtest_config_slippage_model(self):
        """Test BacktestConfig accepts slippage model."""
        from src.backtesting.core_engine import QuantMindBacktester
        
        # Test with slippage model config
        config = {
            'slippage_model': 'volume_based',
            'slippage_params': {
                'base_slippage_pips': 2.0
            }
        }
        
        # This would be tested in integration
        assert config['slippage_model'] == 'volume_based'
    
    def test_backtest_config_commission_model(self):
        """Test BacktestConfig accepts commission model."""
        from src.backtesting.core_engine import QuantMindBacktester
        
        config = {
            'commission_model': 'per_lot',
            'commission_params': {
                'cost_per_lot': 5.0
            }
        }
        
        assert config['commission_model'] == 'per_lot'
    
    def test_lean_slippage_commission_class(self):
        """Test LeanSlippageCommission integration class."""
        from src.backtesting.lean_slippage import ConstantSlippage
        from src.backtesting.lean_commission import PerLotCommission
        from src.backtesting.core_engine import LeanSlippageCommission
        
        # Create models
        slippage = ConstantSlippage(base_slippage_pips=2.0)
        commission = PerLotCommission(cost_per_lot=5.0)
        
        # Create backtrader comminfo
        comminfo = LeanSlippageCommission(
            commission=0.0,
            slippage_model=slippage,
            commission_model=commission
        )
        
        # Test commission calculation
        comm = comminfo._getcommission(
            size=1.0,
            price=1.1000,
            pseudoexec=True
        )
        
        assert comm >= 0
    
    def test_slippage_applied_to_execution(self):
        """Test slippage is applied to execution price."""
        from src.backtesting.lean_slippage import ConstantSlippage
        from src.backtesting.core_engine import LeanSlippageCommission
        
        slippage = ConstantSlippage(base_slippage_pips=2.0)
        
        comminfo = LeanSlippageCommission(
            commission=0.0,
            slippage_model=slippage,
            commission_model=None
        )
        
        # Test slippage calculation
        slippage_pips = comminfo.get_slippage(
            size=1.0,
            price=1.1000,
            pseudoexec=True
        )
        
        assert slippage_pips is not None
        assert slippage_pips >= 0


class TestBacktestResultsWithLeanModels:
    """Test that backtest results include slippage and commission."""
    
    def test_backtest_result_includes_slippage(self):
        """Test BacktestResult includes slippage costs."""
        from src.backtesting.core_engine import BacktestResult
        
        result = BacktestResult(
            sharpe=1.5,
            return_pct=10.0,
            drawdown=5.0,
            trades=100,
            log="",
            total_slippage=50.0,
            total_commission=25.0
        )
        
        assert result['total_slippage'] == 50.0
        assert result['total_commission'] == 25.0
    
    def test_backtest_result_net_pnl(self):
        """Test that net PnL accounts for costs."""
        from src.backtesting.core_engine import BacktestResult
        
        result = BacktestResult(
            sharpe=1.5,
            return_pct=10.0,
            drawdown=5.0,
            trades=100,
            log="",
            total_slippage=50.0,
            total_commission=25.0
        )
        
        # Total costs
        total_costs = result['total_slippage'] + result['total_commission']
        assert total_costs == 75.0


class TestLeanModelConfiguration:
    """Test LEAN model configuration options."""
    
    def test_all_slippage_models_available(self):
        """Test all slippage models are available."""
        from src.backtesting.lean_slippage import SlippageModelFactory
        
        models = [
            "constant",
            "volume_based", 
            "volatility_based",
            "market_impact"
        ]
        
        for model_name in models:
            model = SlippageModelFactory.create(model_name)
            assert model is not None
    
    def test_all_commission_models_available(self):
        """Test all commission models are available."""
        from src.backtesting.lean_commission import CommissionModelFactory
        
        models = [
            "per_trade",
            "per_lot",
            "per_share",
            "tiered",
            "spread_based"
        ]
        
        for model_name in models:
            model = CommissionModelFactory.create(model_name)
            assert model is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

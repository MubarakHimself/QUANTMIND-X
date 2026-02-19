"""
LEAN-Inspired Commission Models

Provides various commission models for realistic backtesting.
Based on QuantConnect LEAN engine concepts.

**Validates: Property 22: LEAN Commission Models**
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class CommissionType(Enum):
    """Supported commission model types."""
    PER_TRADE = "per_trade"
    PER_LOT = "per_lot"
    PER_SHARE = "per_share"
    TIERED = "tiered"
    SPREAD_BASED = "spread_based"


class CommissionModel(ABC):
    """
    Base class for commission models.
    
    Commission represents the trading fees charged by brokers.
    """
    
    @abstractmethod
    def calculate_commission(
        self,
        lots: float,
        symbol: str = "",
        price: float = 0,
        side: str = "buy"
    ) -> float:
        """
        Calculate commission for a trade.
        
        Args:
            lots: Number of lots traded
            symbol: Trading symbol
            price: Trade price
            side: 'buy' or 'sell'
            
        Returns:
            Commission amount in account currency
        """
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize model to dictionary."""
        return {
            'type': self.__class__.__name__,
            'params': {}
        }


class PerTradeCommission(CommissionModel):
    """
    Fixed commission per trade.
    
    Charges a fixed amount regardless of trade size.
    Common for flat-fee broker structures.
    """
    
    def __init__(self, fixed_amount: float = 7.0):
        """
        Initialize per-trade commission model.
        
        Args:
            fixed_amount: Fixed commission per trade in account currency
        """
        self.fixed_amount = fixed_amount
    
    def calculate_commission(
        self,
        lots: float,
        symbol: str = "",
        price: float = 0,
        side: str = "buy"
    ) -> float:
        """Calculate fixed per-trade commission."""
        return self.fixed_amount
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'PerTradeCommission',
            'params': {'fixed_amount': self.fixed_amount}
        }


class PerLotCommission(CommissionModel):
    """
    Commission per lot traded.
    
    Charges a fixed amount per lot, common in forex trading.
    Total commission = lots × cost_per_lot
    """
    
    def __init__(self, cost_per_lot: float = 7.0, min_commission: float = 0.0):
        """
        Initialize per-lot commission model.
        
        Args:
            cost_per_lot: Commission per lot in account currency
            min_commission: Minimum commission per trade
        """
        self.cost_per_lot = cost_per_lot
        self.min_commission = min_commission
    
    def calculate_commission(
        self,
        lots: float,
        symbol: str = "",
        price: float = 0,
        side: str = "buy"
    ) -> float:
        """Calculate per-lot commission."""
        commission = abs(lots) * self.cost_per_lot
        return max(commission, self.min_commission)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'PerLotCommission',
            'params': {
                'cost_per_lot': self.cost_per_lot,
                'min_commission': self.min_commission
            }
        }


class PerShareCommission(CommissionModel):
    """
    Commission per share traded.
    
    Common for equity trading. Total commission = shares × cost_per_share.
    """
    
    def __init__(
        self,
        cost_per_share: float = 0.01,
        min_commission: float = 1.0,
        max_commission_rate: float = 0.01
    ):
        """
        Initialize per-share commission model.
        
        Args:
            cost_per_share: Commission per share
            min_commission: Minimum commission per trade
            max_commission_rate: Maximum commission as % of trade value
        """
        self.cost_per_share = cost_per_share
        self.min_commission = min_commission
        self.max_commission_rate = max_commission_rate
    
    def calculate_commission(
        self,
        lots: float,
        symbol: str = "",
        price: float = 0,
        side: str = "buy"
    ) -> float:
        """Calculate per-share commission."""
        shares = abs(lots) * 100  # Assume 1 lot = 100 shares
        
        # Calculate per-share commission
        commission = shares * self.cost_per_share
        
        # Apply minimum
        commission = max(commission, self.min_commission)
        
        # Apply maximum rate cap
        if price > 0:
            trade_value = shares * price
            max_commission = trade_value * self.max_commission_rate
            commission = min(commission, max_commission)
        
        return commission
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'PerShareCommission',
            'params': {
                'cost_per_share': self.cost_per_share,
                'min_commission': self.min_commission,
                'max_commission_rate': self.max_commission_rate
            }
        }


class TieredCommission(CommissionModel):
    """
    Volume-tiered commission structure.
    
    Commission rate decreases as monthly volume increases.
    Common for professional forex brokers.
    """
    
    def __init__(
        self,
        tiers: List[Tuple[float, float]] = None,
        default_rate: float = 7.0
    ):
        """
        Initialize tiered commission model.
        
        Args:
            tiers: List of (volume_threshold, rate) tuples
                   e.g., [(0, 7.0), (100, 6.0), (500, 5.0), (1000, 4.0)]
            default_rate: Default rate if no tiers specified
        """
        self.tiers = tiers or [
            (0, 7.0),      # 0-99 lots: $7/lot
            (100, 6.0),    # 100-499 lots: $6/lot
            (500, 5.0),    # 500-999 lots: $5/lot
            (1000, 4.0),   # 1000+ lots: $4/lot
        ]
        self.default_rate = default_rate
        self._monthly_volume = 0.0
    
    def set_monthly_volume(self, volume: float):
        """Set current monthly volume for tier calculation."""
        self._monthly_volume = volume
    
    def get_current_rate(self) -> float:
        """Get current commission rate based on monthly volume."""
        rate = self.default_rate
        
        for threshold, tier_rate in sorted(self.tiers, key=lambda x: x[0]):
            if self._monthly_volume >= threshold:
                rate = tier_rate
        
        return rate
    
    def calculate_commission(
        self,
        lots: float,
        symbol: str = "",
        price: float = 0,
        side: str = "buy"
    ) -> float:
        """Calculate tiered commission."""
        rate = self.get_current_rate()
        return abs(lots) * rate
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'TieredCommission',
            'params': {
                'tiers': self.tiers,
                'monthly_volume': self._monthly_volume
            }
        }


class SpreadBasedCommission(CommissionModel):
    """
    Commission based on spread.
    
    For brokers that charge via widened spread instead of direct commission.
    Converts spread cost to commission equivalent.
    """
    
    def __init__(
        self,
        spread_multiplier: float = 1.0,
        pip_value: float = 10.0,
        lot_size: int = 100000
    ):
        """
        Initialize spread-based commission model.
        
        Args:
            spread_multiplier: Multiplier applied to spread cost
            pip_value: Value per pip per lot in account currency
            lot_size: Base lot size (default: 100,000 for forex)
        """
        self.spread_multiplier = spread_multiplier
        self.pip_value = pip_value
        self.lot_size = lot_size
        self._current_spread_pips = 1.0
    
    def set_spread(self, spread_pips: float):
        """Set current spread in pips."""
        self._current_spread_pips = spread_pips
    
    def calculate_commission(
        self,
        lots: float,
        symbol: str = "",
        price: float = 0,
        side: str = "buy"
    ) -> float:
        """Calculate spread-based commission."""
        # Cost = spread_pips × pip_value × lots × multiplier
        spread_cost = self._current_spread_pips * self.pip_value * abs(lots)
        return spread_cost * self.spread_multiplier
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'SpreadBasedCommission',
            'params': {
                'spread_multiplier': self.spread_multiplier,
                'pip_value': self.pip_value,
                'lot_size': self.lot_size,
                'current_spread_pips': self._current_spread_pips
            }
        }


class MakerTakerCommission(CommissionModel):
    """
    Maker/Taker commission structure.
    
    Different rates for liquidity makers (limit orders) vs takers (market orders).
    Common in crypto and some equity exchanges.
    """
    
    def __init__(
        self,
        maker_rate: float = 0.001,  # 0.1%
        taker_rate: float = 0.002,  # 0.2%
        min_commission: float = 0.0
    ):
        """
        Initialize maker/taker commission model.
        
        Args:
            maker_rate: Commission rate for maker orders (as decimal)
            taker_rate: Commission rate for taker orders (as decimal)
            min_commission: Minimum commission per trade
        """
        self.maker_rate = maker_rate
        self.taker_rate = taker_rate
        self.min_commission = min_commission
        self._is_maker = False
    
    def set_order_type(self, is_maker: bool):
        """Set whether the order is a maker (limit) or taker (market)."""
        self._is_maker = is_maker
    
    def calculate_commission(
        self,
        lots: float,
        symbol: str = "",
        price: float = 0,
        side: str = "buy"
    ) -> float:
        """Calculate maker/taker commission."""
        rate = self.maker_rate if self._is_maker else self.taker_rate
        
        if price > 0:
            trade_value = abs(lots) * 100 * price  # Assuming 100 units per lot
            commission = trade_value * rate
        else:
            # Fallback to per-lot calculation
            commission = abs(lots) * rate * 100
        
        return max(commission, self.min_commission)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'MakerTakerCommission',
            'params': {
                'maker_rate': self.maker_rate,
                'taker_rate': self.taker_rate,
                'min_commission': self.min_commission
            }
        }


class CommissionModelFactory:
    """
    Factory for creating commission models.
    """
    
    _models = {
        'per_trade': PerTradeCommission,
        'per_lot': PerLotCommission,
        'per_share': PerShareCommission,
        'tiered': TieredCommission,
        'spread_based': SpreadBasedCommission,
        'maker_taker': MakerTakerCommission,
    }
    
    @classmethod
    def create(cls, model_type: str, **kwargs) -> CommissionModel:
        """Create a commission model by type."""
        model_class = cls._models.get(model_type.lower())
        
        if not model_class:
            raise ValueError(f"Unknown commission model type: {model_type}")
        
        return model_class(**kwargs)
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> CommissionModel:
        """Create a commission model from configuration dictionary."""
        model_type = config.get('type', 'per_lot')
        params = config.get('params', {})
        return cls.create(model_type, **params)
    
    @classmethod
    def list_models(cls) -> list:
        """List available model types."""
        return list(cls._models.keys())


# Default commission instances
DEFAULT_COMMISSION = PerLotCommission(cost_per_lot=7.0)
FOREX_COMMISSION = PerLotCommission(cost_per_lot=7.0)
ECN_COMMISSION = PerLotCommission(cost_per_lot=3.5)  # Typical ECN commission
STOCKS_COMMISSION = PerShareCommission(cost_per_share=0.01, min_commission=1.0)
CRYPTO_COMMISSION = MakerTakerCommission(maker_rate=0.001, taker_rate=0.002)


if __name__ == '__main__':
    # Test commission models
    print("Testing Commission Models")
    print("=" * 50)
    
    test_cases = [
        (1.0, "EURUSD", 1.1000),   # 1 lot forex
        (0.1, "EURUSD", 1.1000),   # 0.1 lot forex
        (10.0, "AAPL", 150.0),     # 10 lots stocks
    ]
    
    models = [
        ('Per Trade', PerTradeCommission(fixed_amount=7.0)),
        ('Per Lot', PerLotCommission(cost_per_lot=7.0)),
        ('Per Share', PerShareCommission(cost_per_share=0.01)),
        ('Tiered', TieredCommission()),
        ('Spread Based', SpreadBasedCommission()),
        ('Maker/Taker', MakerTakerCommission()),
    ]
    
    for lots, symbol, price in test_cases:
        print(f"\nTrade: {lots} lots {symbol} @ {price}")
        print("-" * 40)
        
        for name, model in models:
            commission = model.calculate_commission(lots, symbol, price)
            print(f"  {name}: ${commission:.2f}")
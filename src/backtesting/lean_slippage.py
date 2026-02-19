"""
LEAN-Inspired Slippage Models

Provides various slippage models for realistic backtesting.
Based on QuantConnect LEAN engine concepts.

**Validates: Property 21: LEAN Slippage Models**
"""

import math
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class SlippageType(Enum):
    """Supported slippage model types."""
    CONSTANT = "constant"
    VOLUME_BASED = "volume_based"
    VOLATILITY_BASED = "volatility_based"
    MARKET_IMPACT = "market_impact"


class SlippageModel(ABC):
    """
    Base class for slippage models.
    
    Slippage represents the difference between the expected price
    of a trade and the actual execution price.
    """
    
    @abstractmethod
    def calculate_slippage(
        self,
        order_volume: float,
        avg_volume: float = 0,
        current_atr: float = 0,
        avg_atr: float = 0,
        price: float = 0
    ) -> float:
        """
        Calculate slippage in price units.
        
        Args:
            order_volume: Volume of the order (in lots or units)
            avg_volume: Average daily volume for the symbol
            current_atr: Current ATR value
            avg_atr: Average ATR value
            price: Current price (for percentage calculations)
            
        Returns:
            Slippage amount in price units
        """
        pass
    
    def apply_slippage(
        self,
        price: float,
        side: str,
        order_volume: float = 0,
        avg_volume: float = 0,
        current_atr: float = 0,
        avg_atr: float = 0
    ) -> float:
        """
        Apply slippage to a price.
        
        Args:
            price: Original price
            side: 'buy' or 'sell'
            order_volume: Volume of the order
            avg_volume: Average daily volume
            current_atr: Current ATR value
            avg_atr: Average ATR value
            
        Returns:
            Adjusted price with slippage applied
        """
        slippage = self.calculate_slippage(
            order_volume=order_volume,
            avg_volume=avg_volume,
            current_atr=current_atr,
            avg_atr=avg_atr,
            price=price
        )
        
        # Apply slippage based on order side
        # Buy: slippage increases price (pay more)
        # Sell: slippage decreases price (receive less)
        if side.lower() in ('buy', 'long'):
            return price + slippage
        else:
            return price - slippage
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize model to dictionary."""
        return {
            'type': self.__class__.__name__,
            'params': {}
        }


class ConstantSlippage(SlippageModel):
    """
    Constant slippage model.
    
    Applies a fixed slippage amount to all trades.
    Useful for simple estimations or when market impact is minimal.
    """
    
    def __init__(self, slippage_pips: float = 0.5):
        """
        Initialize constant slippage model.
        
        Args:
            slippage_pips: Fixed slippage in pips (default: 0.5 pips)
        """
        self.slippage_pips = slippage_pips
        self.pip_value = 0.0001  # Standard pip value for most forex pairs
    
    def calculate_slippage(
        self,
        order_volume: float = 0,
        avg_volume: float = 0,
        current_atr: float = 0,
        avg_atr: float = 0,
        price: float = 0
    ) -> float:
        """Calculate constant slippage."""
        return self.slippage_pips * self.pip_value
    
    def set_pip_value(self, pip_value: float):
        """Set pip value for the symbol."""
        self.pip_value = pip_value
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'ConstantSlippage',
            'params': {
                'slippage_pips': self.slippage_pips,
                'pip_value': self.pip_value
            }
        }


class VolumeBasedSlippage(SlippageModel):
    """
    Volume-based slippage model.
    
    Slippage increases with order volume relative to average volume.
    Based on the square root model commonly used in market impact estimation.
    
    Formula: slippage = base_slippage × (order_volume / avg_volume)^0.5
    """
    
    def __init__(
        self,
        base_slippage_pips: float = 0.5,
        max_slippage_multiplier: float = 5.0
    ):
        """
        Initialize volume-based slippage model.
        
        Args:
            base_slippage_pips: Base slippage in pips
            max_slippage_multiplier: Maximum multiplier for slippage
        """
        self.base_slippage_pips = base_slippage_pips
        self.max_slippage_multiplier = max_slippage_multiplier
        self.pip_value = 0.0001
    
    def calculate_slippage(
        self,
        order_volume: float,
        avg_volume: float = 0,
        current_atr: float = 0,
        avg_atr: float = 0,
        price: float = 0
    ) -> float:
        """Calculate volume-based slippage."""
        base_slippage = self.base_slippage_pips * self.pip_value
        
        if avg_volume <= 0 or order_volume <= 0:
            return base_slippage
        
        # Calculate volume ratio
        volume_ratio = order_volume / avg_volume
        
        # Apply square root model
        multiplier = math.sqrt(volume_ratio)
        
        # Cap the multiplier
        multiplier = min(multiplier, self.max_slippage_multiplier)
        
        return base_slippage * multiplier
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'VolumeBasedSlippage',
            'params': {
                'base_slippage_pips': self.base_slippage_pips,
                'max_slippage_multiplier': self.max_slippage_multiplier
            }
        }


class VolatilityBasedSlippage(SlippageModel):
    """
    Volatility-based slippage model.
    
    Slippage scales with current volatility relative to average volatility.
    Higher volatility leads to higher slippage.
    
    Formula: slippage = base_slippage × (current_atr / avg_atr)
    """
    
    def __init__(
        self,
        base_slippage_pips: float = 0.5,
        max_slippage_multiplier: float = 3.0
    ):
        """
        Initialize volatility-based slippage model.
        
        Args:
            base_slippage_pips: Base slippage in pips
            max_slippage_multiplier: Maximum multiplier for slippage
        """
        self.base_slippage_pips = base_slippage_pips
        self.max_slippage_multiplier = max_slippage_multiplier
        self.pip_value = 0.0001
    
    def calculate_slippage(
        self,
        order_volume: float,
        avg_volume: float = 0,
        current_atr: float = 0,
        avg_atr: float = 0,
        price: float = 0
    ) -> float:
        """Calculate volatility-based slippage."""
        base_slippage = self.base_slippage_pips * self.pip_value
        
        if avg_atr <= 0 or current_atr <= 0:
            return base_slippage
        
        # Calculate volatility ratio
        volatility_ratio = current_atr / avg_atr
        
        # Cap the multiplier
        multiplier = min(volatility_ratio, self.max_slippage_multiplier)
        
        return base_slippage * multiplier
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'VolatilityBasedSlippage',
            'params': {
                'base_slippage_pips': self.base_slippage_pips,
                'max_slippage_multiplier': self.max_slippage_multiplier
            }
        }


class MarketImpactSlippage(SlippageModel):
    """
    Market impact slippage model (Almgren-Chriss inspired).
    
    Estimates price impact based on order size relative to daily volume.
    Uses a power law model commonly used in institutional trading.
    
    Formula: impact = k × (order_volume / daily_volume)^0.6 × price
    """
    
    def __init__(
        self,
        impact_coefficient: float = 0.1,
        impact_exponent: float = 0.6,
        min_impact_bps: float = 1.0,
        max_impact_bps: float = 50.0
    ):
        """
        Initialize market impact slippage model.
        
        Args:
            impact_coefficient: Impact coefficient (k)
            impact_exponent: Impact exponent (default: 0.6)
            min_impact_bps: Minimum impact in basis points
            max_impact_bps: Maximum impact in basis points
        """
        self.impact_coefficient = impact_coefficient
        self.impact_exponent = impact_exponent
        self.min_impact_bps = min_impact_bps
        self.max_impact_bps = max_impact_bps
    
    def calculate_slippage(
        self,
        order_volume: float,
        avg_volume: float = 0,
        current_atr: float = 0,
        avg_atr: float = 0,
        price: float = 0
    ) -> float:
        """Calculate market impact slippage."""
        if avg_volume <= 0 or order_volume <= 0 or price <= 0:
            return price * self.min_impact_bps / 10000
        
        # Calculate participation rate
        participation_rate = order_volume / avg_volume
        
        # Apply power law model
        impact = self.impact_coefficient * (participation_rate ** self.impact_exponent)
        
        # Convert to basis points
        impact_bps = impact * 10000
        
        # Apply bounds
        impact_bps = max(self.min_impact_bps, min(impact_bps, self.max_impact_bps))
        
        # Convert to price units
        return price * impact_bps / 10000
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'MarketImpactSlippage',
            'params': {
                'impact_coefficient': self.impact_coefficient,
                'impact_exponent': self.impact_exponent,
                'min_impact_bps': self.min_impact_bps,
                'max_impact_bps': self.max_impact_bps
            }
        }


class SlippageModelFactory:
    """
    Factory for creating slippage models.
    """
    
    _models = {
        'constant': ConstantSlippage,
        'volume_based': VolumeBasedSlippage,
        'volatility_based': VolatilityBasedSlippage,
        'market_impact': MarketImpactSlippage,
    }
    
    @classmethod
    def create(
        cls,
        model_type: str,
        **kwargs
    ) -> SlippageModel:
        """
        Create a slippage model by type.
        
        Args:
            model_type: Type of slippage model
            **kwargs: Model-specific parameters
            
        Returns:
            Configured SlippageModel instance
        """
        model_class = cls._models.get(model_type.lower())
        
        if not model_class:
            raise ValueError(f"Unknown slippage model type: {model_type}")
        
        return model_class(**kwargs)
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> SlippageModel:
        """
        Create a slippage model from configuration dictionary.
        
        Args:
            config: Configuration with 'type' and 'params' keys
            
        Returns:
            Configured SlippageModel instance
        """
        model_type = config.get('type', 'constant')
        params = config.get('params', {})
        
        return cls.create(model_type, **params)
    
    @classmethod
    def list_models(cls) -> list:
        """List available model types."""
        return list(cls._models.keys())


# Default slippage model instances
DEFAULT_SLIPPAGE = ConstantSlippage(slippage_pips=0.5)
FOREX_SLIPPAGE = ConstantSlippage(slippage_pips=0.2)
STOCKS_SLIPPAGE = VolumeBasedSlippage(base_slippage_pips=1.0)
CRYPTO_SLIPPAGE = VolatilityBasedSlippage(base_slippage_pips=2.0)


if __name__ == '__main__':
    # Test the slippage models
    print("Testing Slippage Models")
    print("=" * 50)
    
    price = 1.1000  # EURUSD price
    order_volume = 1.0  # 1 lot
    avg_volume = 10.0  # Average 10 lots daily
    current_atr = 0.0080
    avg_atr = 0.0050
    
    models = [
        ('Constant', ConstantSlippage(slippage_pips=0.5)),
        ('Volume Based', VolumeBasedSlippage(base_slippage_pips=0.5)),
        ('Volatility Based', VolatilityBasedSlippage(base_slippage_pips=0.5)),
        ('Market Impact', MarketImpactSlippage())
    ]
    
    for name, model in models:
        buy_price = model.apply_slippage(
            price=price,
            side='buy',
            order_volume=order_volume,
            avg_volume=avg_volume,
            current_atr=current_atr,
            avg_atr=avg_atr
        )
        
        sell_price = model.apply_slippage(
            price=price,
            side='sell',
            order_volume=order_volume,
            avg_volume=avg_volume,
            current_atr=current_atr,
            avg_atr=avg_atr
        )
        
        slippage = model.calculate_slippage(
            order_volume=order_volume,
            avg_volume=avg_volume,
            current_atr=current_atr,
            avg_atr=avg_atr,
            price=price
        )
        
        print(f"\n{name}:")
        print(f"  Original price: {price:.5f}")
        print(f"  Buy price:      {buy_price:.5f}")
        print(f"  Sell price:     {sell_price:.5f}")
        print(f"  Slippage:       {slippage:.6f} ({slippage * 10000:.1f} pips)")
    
    # Test factory
    print("\n\nFactory Test:")
    model = SlippageModelFactory.create('volume_based', base_slippage_pips=1.0)
    print(f"Created: {model.to_dict()}")
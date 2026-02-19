"""
Multi-Asset Backtesting Framework

Provides asset class-specific handlers for backtesting across
different markets (Forex, Stocks, Crypto, Futures).

Based on QuantConnect LEAN engine architecture.

**Validates: Property 23: Multi-Asset Support**
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime, time, timedelta
from typing import Dict, Any, Optional, List
from enum import Enum

logger = logging.getLogger(__name__)


class AssetClass(Enum):
    """Supported asset classes."""
    FOREX = "forex"
    STOCKS = "stocks"
    CRYPTO = "crypto"
    FUTURES = "futures"
    INDICES = "indices"
    COMMODITIES = "commodities"


class MarketHours:
    """Market hours configuration."""
    
    # Forex: 24/5 (Sunday 5 PM ET - Friday 5 PM ET)
    FOREX_OPEN = time(17, 0)  # Sunday 5 PM ET
    FOREX_CLOSE = time(17, 0)  # Friday 5 PM ET
    
    # US Stock Market: 9:30 AM - 4:00 PM ET
    STOCK_OPEN = time(9, 30)
    STOCK_CLOSE = time(16, 0)
    
    # Crypto: 24/7
    CRYPTO_ALWAYS_OPEN = True
    
    # Futures: Varies by contract, typically 6 PM - 5 PM ET
    FUTURES_OPEN = time(18, 0)
    FUTURES_CLOSE = time(17, 0)


class AssetHandler(ABC):
    """
    Base class for asset-specific handling.
    
    Provides common interface for different asset classes.
    """
    
    asset_class: AssetClass = None
    
    @abstractmethod
    def calculate_pip_value(
        self,
        symbol: str,
        price: float,
        lot_size: float = 1.0
    ) -> float:
        """
        Calculate pip value for the asset.
        
        Args:
            symbol: Trading symbol
            price: Current price
            lot_size: Position size in lots
            
        Returns:
            Pip value in account currency
        """
        pass
    
    @abstractmethod
    def calculate_margin(
        self,
        symbol: str,
        price: float,
        lots: float,
        leverage: float = 1.0
    ) -> float:
        """
        Calculate required margin for a position.
        
        Args:
            symbol: Trading symbol
            price: Current price
            lots: Position size in lots
            leverage: Account leverage
            
        Returns:
            Required margin in account currency
        """
        pass
    
    @abstractmethod
    def apply_fees(
        self,
        symbol: str,
        price: float,
        lots: float,
        side: str
    ) -> float:
        """
        Calculate trading fees (commission, spread, etc.).
        
        Args:
            symbol: Trading symbol
            price: Trade price
            lots: Position size
            side: 'buy' or 'sell'
            
        Returns:
            Total fees in account currency
        """
        pass
    
    @abstractmethod
    def is_market_open(self, dt: datetime) -> bool:
        """
        Check if market is open at given time.
        
        Args:
            dt: Datetime to check
            
        Returns:
            True if market is open
        """
        pass
    
    @abstractmethod
    def normalize_price(self, symbol: str, price: float) -> float:
        """
        Normalize price to symbol's tick size.
        
        Args:
            symbol: Trading symbol
            price: Raw price
            
        Returns:
            Normalized price
        """
        pass
    
    @abstractmethod
    def get_contract_size(self, symbol: str) -> float:
        """
        Get contract/lot size for the symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Contract size
        """
        pass


class ForexHandler(AssetHandler):
    """
    Handler for Forex currency pairs.
    
    Features:
    - 24/5 trading
    - Standard lot sizes (100,000 units)
    - Leverage typically 50:1 to 500:1
    - Pip-based pricing
    """
    
    asset_class = AssetClass.FOREX
    
    # Pip values for different currency pairs
    PIP_VALUES = {
        'default': 0.0001,  # 4 decimal places
        'JPY': 0.01,        # 2 decimal places for JPY pairs
    }
    
    CONTRACT_SIZES = {
        'standard': 100000,
        'mini': 10000,
        'micro': 1000,
    }
    
    def __init__(
        self,
        default_leverage: float = 100.0,
        commission_per_lot: float = 7.0,
        default_spread_pips: float = 1.0
    ):
        self.default_leverage = default_leverage
        self.commission_per_lot = commission_per_lot
        self.default_spread_pips = default_spread_pips
        
        # Symbol-specific configurations
        self._symbol_configs: Dict[str, Dict] = {}
    
    def configure_symbol(self, symbol: str, config: Dict[str, Any]):
        """Configure symbol-specific settings."""
        self._symbol_configs[symbol.upper()] = config
    
    def get_pip_size(self, symbol: str) -> float:
        """Get pip size for symbol."""
        symbol = symbol.upper()
        if 'JPY' in symbol:
            return self.PIP_VALUES['JPY']
        return self.PIP_VALUES['default']
    
    def calculate_pip_value(
        self,
        symbol: str,
        price: float,
        lot_size: float = 1.0
    ) -> float:
        """Calculate pip value for forex pair."""
        pip_size = self.get_pip_size(symbol)
        contract_size = self.get_contract_size(symbol)
        
        # Pip value = pip_size × contract_size × lots
        pip_value = pip_size * contract_size * lot_size
        
        # Adjust for cross-currency pairs (simplified)
        # In reality, this depends on account currency
        return pip_value
    
    def calculate_margin(
        self,
        symbol: str,
        price: float,
        lots: float,
        leverage: float = None
    ) -> float:
        """Calculate required margin."""
        leverage = leverage or self.default_leverage
        contract_size = self.get_contract_size(symbol)
        position_value = price * contract_size * lots
        return position_value / leverage
    
    def apply_fees(
        self,
        symbol: str,
        price: float,
        lots: float,
        side: str
    ) -> float:
        """Calculate forex trading fees."""
        # Commission
        commission = abs(lots) * self.commission_per_lot
        
        # Spread cost
        config = self._symbol_configs.get(symbol.upper(), {})
        spread_pips = config.get('spread_pips', self.default_spread_pips)
        pip_value = self.calculate_pip_value(symbol, price, lots)
        spread_cost = spread_pips * pip_value / lots if lots > 0 else 0
        
        return commission + (spread_cost * abs(lots))
    
    def is_market_open(self, dt: datetime) -> bool:
        """Check if forex market is open."""
        # Forex is closed on weekends
        if dt.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        return True
    
    def normalize_price(self, symbol: str, price: float) -> float:
        """Normalize price to pip precision."""
        pip_size = self.get_pip_size(symbol)
        return round(price / pip_size) * pip_size
    
    def get_contract_size(self, symbol: str) -> float:
        """Get standard forex lot size."""
        return self.CONTRACT_SIZES['standard']


class StockHandler(AssetHandler):
    """
    Handler for stock/equity trading.
    
    Features:
    - Market hours: 9:30 AM - 4:00 PM ET
    - Share-based sizing
    - Commission per share or flat fee
    - No leverage (margin accounts differ)
    """
    
    asset_class = AssetClass.STOCKS
    
    def __init__(
        self,
        commission_per_share: float = 0.01,
        min_commission: float = 1.0,
        margin_rate: float = 0.5
    ):
        self.commission_per_share = commission_per_share
        self.min_commission = min_commission
        self.margin_rate = margin_rate  # 50% for long, 150% for short
        
        self._symbol_configs: Dict[str, Dict] = {}
    
    def calculate_pip_value(
        self,
        symbol: str,
        price: float,
        lot_size: float = 1.0
    ) -> float:
        """Stocks don't use pips - return minimum tick value."""
        tick_size = 0.01  # $0.01 minimum tick
        shares = lot_size * 100  # 1 lot = 100 shares
        return tick_size * shares
    
    def calculate_margin(
        self,
        symbol: str,
        price: float,
        lots: float,
        leverage: float = None
    ) -> float:
        """Calculate stock margin requirement."""
        shares = lots * 100
        position_value = price * shares
        return position_value * self.margin_rate
    
    def apply_fees(
        self,
        symbol: str,
        price: float,
        lots: float,
        side: str
    ) -> float:
        """Calculate stock trading fees."""
        shares = abs(lots) * 100
        commission = shares * self.commission_per_share
        return max(commission, self.min_commission)
    
    def is_market_open(self, dt: datetime) -> bool:
        """Check if US stock market is open."""
        # Weekend check
        if dt.weekday() >= 5:
            return False
        
        # Market hours check (9:30 AM - 4:00 PM ET)
        current_time = dt.time()
        return MarketHours.STOCK_OPEN <= current_time <= MarketHours.STOCK_CLOSE
    
    def normalize_price(self, symbol: str, price: float) -> float:
        """Normalize to cent precision."""
        return round(price, 2)
    
    def get_contract_size(self, symbol: str) -> float:
        """Stocks trade in shares, not lots."""
        return 1.0  # 1 share
    
    def calculate_dividend_adjustment(
        self,
        symbol: str,
        shares: float,
        ex_dividend_date: datetime
    ) -> float:
        """Calculate dividend adjustment for backtest."""
        config = self._symbol_configs.get(symbol.upper(), {})
        dividend_per_share = config.get('dividend_per_share', 0)
        return shares * dividend_per_share


class CryptoHandler(AssetHandler):
    """
    Handler for cryptocurrency trading.
    
    Features:
    - 24/7 trading
    - Maker/taker fee structure
    - Variable contract sizes
    - Funding rates for perpetuals
    """
    
    asset_class = AssetClass.CRYPTO
    
    def __init__(
        self,
        maker_fee: float = 0.001,  # 0.1%
        taker_fee: float = 0.002,  # 0.2%
        funding_rate: float = 0.0001  # 0.01% per 8 hours
    ):
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.funding_rate = funding_rate
        
        self._symbol_configs: Dict[str, Dict] = {}
    
    def calculate_pip_value(
        self,
        symbol: str,
        price: float,
        lot_size: float = 1.0
    ) -> float:
        """Calculate tick value for crypto."""
        config = self._symbol_configs.get(symbol.upper(), {})
        tick_size = config.get('tick_size', 0.01)
        contract_size = config.get('contract_size', 1.0)
        return tick_size * contract_size * lot_size
    
    def calculate_margin(
        self,
        symbol: str,
        price: float,
        lots: float,
        leverage: float = 10.0
    ) -> float:
        """Calculate crypto margin (typically higher leverage)."""
        contract_size = self.get_contract_size(symbol)
        position_value = price * contract_size * lots
        return position_value / leverage
    
    def apply_fees(
        self,
        symbol: str,
        price: float,
        lots: float,
        side: str,
        is_maker: bool = False
    ) -> float:
        """Calculate crypto trading fees (maker/taker)."""
        contract_size = self.get_contract_size(symbol)
        trade_value = price * abs(lots) * contract_size
        
        fee_rate = self.maker_fee if is_maker else self.taker_fee
        return trade_value * fee_rate
    
    def calculate_funding(
        self,
        symbol: str,
        price: float,
        lots: float,
        hours: float = 8
    ) -> float:
        """Calculate funding rate cost for perpetual positions."""
        contract_size = self.get_contract_size(symbol)
        position_value = price * lots * contract_size
        
        # Funding is typically charged every 8 hours
        periods = hours / 8
        return position_value * self.funding_rate * periods
    
    def is_market_open(self, dt: datetime) -> bool:
        """Crypto markets are always open."""
        return True
    
    def normalize_price(self, symbol: str, price: float) -> float:
        """Normalize to tick precision."""
        config = self._symbol_configs.get(symbol.upper(), {})
        tick_size = config.get('tick_size', 0.01)
        return round(price / tick_size) * tick_size
    
    def get_contract_size(self, symbol: str) -> float:
        """Get crypto contract size."""
        config = self._symbol_configs.get(symbol.upper(), {})
        return config.get('contract_size', 1.0)


class FuturesHandler(AssetHandler):
    """
    Handler for futures contracts.
    
    Features:
    - Variable trading hours by contract
    - Standard contract sizes
    - Initial and maintenance margin
    - Contract expiration handling
    """
    
    asset_class = AssetClass.FUTURES
    
    # Standard contract specifications
    CONTRACT_SPECS = {
        'ES': {'size': 50, 'tick': 0.25, 'margin': 12500},       # S&P 500
        'NQ': {'size': 20, 'tick': 0.25, 'margin': 15000},       # Nasdaq 100
        'YM': {'size': 5, 'tick': 1.0, 'margin': 7500},          # Dow Jones
        'CL': {'size': 1000, 'tick': 0.01, 'margin': 7500},      # Crude Oil
        'GC': {'size': 100, 'tick': 0.1, 'margin': 10000},       # Gold
        'SI': {'size': 5000, 'tick': 0.005, 'margin': 10000},    # Silver
        'ZC': {'size': 5000, 'tick': 0.0025, 'margin': 2000},    # Corn
        'ZW': {'size': 5000, 'tick': 0.0025, 'margin': 2500},    # Wheat
    }
    
    def __init__(
        self,
        commission_per_contract: float = 2.50,
        exchange_fees: float = 1.50
    ):
        self.commission_per_contract = commission_per_contract
        self.exchange_fees = exchange_fees
        
        self._contract_months = ['H', 'M', 'U', 'Z']  # Mar, Jun, Sep, Dec
    
    def calculate_pip_value(
        self,
        symbol: str,
        price: float,
        lot_size: float = 1.0
    ) -> float:
        """Calculate tick value for futures."""
        base_symbol = self._get_base_symbol(symbol)
        specs = self.CONTRACT_SPECS.get(base_symbol, {'size': 1, 'tick': 0.01})
        return specs['size'] * specs['tick'] * lot_size
    
    def calculate_margin(
        self,
        symbol: str,
        price: float,
        lots: float,
        leverage: float = None
    ) -> float:
        """Calculate futures margin requirement."""
        base_symbol = self._get_base_symbol(symbol)
        specs = self.CONTRACT_SPECS.get(base_symbol, {'margin': 5000})
        return specs['margin'] * abs(lots)
    
    def apply_fees(
        self,
        symbol: str,
        price: float,
        lots: float,
        side: str
    ) -> float:
        """Calculate futures trading fees."""
        contracts = abs(lots)
        return contracts * (self.commission_per_contract + self.exchange_fees)
    
    def is_market_open(self, dt: datetime) -> bool:
        """Check if futures market is open."""
        # Most futures trade nearly 24 hours with breaks
        current_time = dt.time()
        
        # Daily maintenance window: 5:00 PM - 6:00 PM ET
        if time(17, 0) <= current_time <= time(18, 0):
            return False
        
        # Weekend close: Friday 5 PM - Sunday 6 PM ET
        if dt.weekday() == 4 and current_time >= time(17, 0):
            return False
        if dt.weekday() == 5:
            return False
        if dt.weekday() == 6 and current_time < time(18, 0):
            return False
        
        return True
    
    def normalize_price(self, symbol: str, price: float) -> float:
        """Normalize to tick precision."""
        base_symbol = self._get_base_symbol(symbol)
        specs = self.CONTRACT_SPECS.get(base_symbol, {'tick': 0.01})
        return round(price / specs['tick']) * specs['tick']
    
    def get_contract_size(self, symbol: str) -> float:
        """Get futures contract size."""
        base_symbol = self._get_base_symbol(symbol)
        specs = self.CONTRACT_SPECS.get(base_symbol, {'size': 1})
        return specs['size']
    
    def _get_base_symbol(self, symbol: str) -> str:
        """Extract base symbol from full contract code."""
        # e.g., 'ESU23' -> 'ES', 'GCZ24' -> 'GC'
        for i, c in enumerate(symbol):
            if c.isdigit():
                return symbol[:i]
        return symbol
    
    def get_expiry_date(self, symbol: str) -> Optional[datetime]:
        """Get contract expiration date."""
        # Parse contract month/year from symbol
        base = self._get_base_symbol(symbol)
        suffix = symbol[len(base):]
        
        if len(suffix) >= 2:
            month_code = suffix[0]
            year_code = suffix[1:]
            
            if month_code in self._contract_months:
                month = (self._contract_months.index(month_code) + 1) * 3
                year = 2000 + int(year_code) if len(year_code) == 2 else int(year_code)
                
                # Third Friday of the month
                return self._get_third_friday(year, month)
        
        return None
    
    def _get_third_friday(self, year: int, month: int) -> datetime:
        """Get the third Friday of a month."""
        first_day = datetime(year, month, 1)
        offset_days = (4 - first_day.weekday()) % 7
        first_friday = first_day + timedelta(days=offset_days)
        third_friday = first_friday + timedelta(days=14)
        return third_friday


class AssetHandlerFactory:
    """
    Factory for creating asset handlers.
    """
    
    _handlers = {
        AssetClass.FOREX: ForexHandler,
        AssetClass.STOCKS: StockHandler,
        AssetClass.CRYPTO: CryptoHandler,
        AssetClass.FUTURES: FuturesHandler,
    }
    
    @classmethod
    def create(cls, asset_class: AssetClass, **kwargs) -> AssetHandler:
        """Create an asset handler for the specified class."""
        handler_class = cls._handlers.get(asset_class)
        
        if not handler_class:
            raise ValueError(f"Unknown asset class: {asset_class}")
        
        return handler_class(**kwargs)
    
    @classmethod
    def get_handler_for_symbol(cls, symbol: str) -> AssetHandler:
        """Determine asset class and return appropriate handler."""
        asset_class = cls.detect_asset_class(symbol)
        return cls.create(asset_class)
    
    @classmethod
    def detect_asset_class(cls, symbol: str) -> AssetClass:
        """Detect asset class from symbol format."""
        symbol = symbol.upper()
        
        # Forex pairs (6 characters, currency codes)
        forex_pairs = ['EUR', 'USD', 'GBP', 'JPY', 'CHF', 'AUD', 'NZD', 'CAD']
        if len(symbol) == 6:
            base, quote = symbol[:3], symbol[3:]
            if base in forex_pairs and quote in forex_pairs:
                return AssetClass.FOREX
        
        # Crypto (contains X, BTC, ETH, etc.)
        crypto_symbols = ['BTC', 'ETH', 'XRP', 'LTC', 'ADA', 'DOT', 'SOL', 'XBT']
        for crypto in crypto_symbols:
            if crypto in symbol:
                return AssetClass.CRYPTO
        
        # Futures (contains month codes)
        futures_codes = ['ES', 'NQ', 'YM', 'CL', 'GC', 'SI', 'ZC', 'ZW']
        for code in futures_codes:
            if symbol.startswith(code):
                return AssetClass.FUTURES
        
        # Default to stocks
        return AssetClass.STOCKS
    
    @classmethod
    def list_asset_classes(cls) -> List[str]:
        """List supported asset classes."""
        return [ac.value for ac in cls._handlers.keys()]


if __name__ == '__main__':
    # Test multi-asset handling
    print("Testing Multi-Asset Framework")
    print("=" * 50)
    
    # Test symbols
    test_cases = [
        ('EURUSD', 1.1000, 1.0),   # Forex
        ('BTCUSD', 45000, 0.1),    # Crypto
        ('AAPL', 150, 10),         # Stock
        ('ESU24', 4500, 1),        # Futures
    ]
    
    for symbol, price, lots in test_cases:
        print(f"\n{symbol} @ {price} ({lots} lots)")
        print("-" * 40)
        
        handler = AssetHandlerFactory.get_handler_for_symbol(symbol)
        print(f"  Asset Class: {handler.asset_class.value}")
        print(f"  Pip Value: ${handler.calculate_pip_value(symbol, price, lots):.2f}")
        print(f"  Margin: ${handler.calculate_margin(symbol, price, lots):.2f}")
        print(f"  Fees: ${handler.apply_fees(symbol, price, lots, 'buy'):.2f}")
        print(f"  Market Open: {handler.is_market_open(datetime.now())}")
        print(f"  Contract Size: {handler.get_contract_size(symbol)}")
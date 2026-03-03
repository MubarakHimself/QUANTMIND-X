"""Symbol data models for MT5 integration."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict


@dataclass
class SymbolInfo:
    """Symbol information model."""
    name: str
    digits: int
    point: float
    tick_value: float
    tick_size: float
    contract_size: float
    currency_base: str
    currency_profit: str
    currency_margin: str
    volume_min: float
    volume_max: float
    volume_step: float
    pip_location: int  # Position of pip (e.g., 4 for EURUSD, 2 for XAUUSD)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "digits": self.digits,
            "point": self.point,
            "tick_value": self.tick_value,
            "tick_size": self.tick_size,
            "contract_size": self.contract_size,
            "currency_base": self.currency_base,
            "currency_profit": self.currency_profit,
            "currency_margin": self.currency_margin,
            "volume_min": self.volume_min,
            "volume_max": self.volume_max,
            "volume_step": self.volume_step,
            "pip_location": self.pip_location
        }


@dataclass
class TickData:
    """Tick data model."""
    symbol: str
    bid: float
    ask: float
    spread: float
    timestamp: datetime
    volume: int = 0
    source: str = "live"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "bid": self.bid,
            "ask": self.ask,
            "spread": self.spread,
            "timestamp": self.timestamp.isoformat(),
            "volume": self.volume,
            "source": self.source
        }

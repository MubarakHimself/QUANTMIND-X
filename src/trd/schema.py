"""
TRD Document Schema Definition

Defines the data structures for Trading Strategy Documents (TRD).
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any


class PositionSizingMethod(str, Enum):
    """Position sizing method types."""
    FIXED_LOT = "fixed_lot"
    DYNAMIC = "dynamic"
    KELLY = "kelly"
    MARTINGALE = "martingale"


class TradingSession(str, Enum):
    """Trading session identifiers."""
    SYDNEY = "Sydney"
    TOKYO = "Tokyo"
    LONDON = "London"
    NEW_YORK = "NewYork"
    ASIA = "Asia"
    EUROPE = "Europe"
    US = "US"
    UK = "UK"


class StrategyType(str, Enum):
    """Strategy type classification."""
    TREND = "trend"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    SCALPING = "scalping"
    SWING = "swing"


@dataclass
class PositionSizing:
    """Position sizing configuration."""
    method: PositionSizingMethod
    risk_percent: float = 1.0
    max_lots: float = 1.0
    fixed_lot_size: float = 0.01

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method.value,
            "risk_percent": self.risk_percent,
            "max_lots": self.max_lots,
            "fixed_lot_size": self.fixed_lot_size,
        }


@dataclass
class TRDParameter:
    """Individual TRD parameter with validation metadata."""
    name: str
    value: Any
    required: bool = True
    param_type: str = "string"
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[List[str]] = None
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "required": self.required,
            "type": self.param_type,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "allowed_values": self.allowed_values,
            "description": self.description,
        }


@dataclass
class TRDDocument:
    """
    Complete Trading Strategy Document (TRD) schema.

    This document captures all strategy parameters needed to generate
    a production-ready MQL5 EA.
    """
    # Core identifiers
    strategy_id: str
    strategy_name: str
    version: int = 1

    # Market context
    symbol: str = "EURUSD"
    timeframe: str = "H4"

    # Strategy classification
    strategy_type: StrategyType = StrategyType.TREND
    description: str = ""

    # Entry and exit rules
    entry_conditions: List[str] = field(default_factory=list)
    exit_conditions: List[str] = field(default_factory=list)

    # Position sizing
    position_sizing: Optional[PositionSizing] = None

    # Trading parameters
    parameters: Dict[str, Any] = field(default_factory=dict)

    # EA template fields
    bot_type: str = "scalping"  # Bot strategy type (scalping/orb/structural/swing)
    session_tags: List[str] = field(default_factory=list)  # Trading session tags (LONDON, ASIA, NEW_YORK)
    news_blackout: bool = True  # Enable news blackout kill zones
    atr_stop_period: int = 14   # ATR period for ATR-based stop loss
    atr_stop_multiplier: float = 1.5  # ATR multiplier for stop distance (1.5x = 1.5R)

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    author: str = "Research Department"
    source: str = "manual"

    # Ambiguity tracking (populated by validator)
    ambiguous_parameters: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "strategy_id": self.strategy_id,
            "strategy_name": self.strategy_name,
            "version": self.version,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "strategy_type": self.strategy_type.value if self.strategy_type else None,
            "description": self.description,
            "entry_conditions": self.entry_conditions,
            "exit_conditions": self.exit_conditions,
            "position_sizing": self.position_sizing.to_dict() if self.position_sizing else None,
            "parameters": self.parameters,
            "bot_type": self.bot_type,
            "session_tags": self.session_tags,
            "news_blackout": self.news_blackout,
            "atr_stop_period": self.atr_stop_period,
            "atr_stop_multiplier": self.atr_stop_multiplier,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "author": self.author,
            "source": self.source,
            "ambiguous_parameters": self.ambiguous_parameters,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TRDDocument":
        """Create TRDDocument from dictionary."""
        # Handle position_sizing
        position_sizing = None
        if data.get("position_sizing"):
            ps_data = data["position_sizing"]
            method = PositionSizingMethod(ps_data.get("method", "fixed_lot"))
            position_sizing = PositionSizing(
                method=method,
                risk_percent=ps_data.get("risk_percent", 1.0),
                max_lots=ps_data.get("max_lots", 1.0),
                fixed_lot_size=ps_data.get("fixed_lot_size", 0.01),
            )

        # Handle strategy_type
        strategy_type = None
        if data.get("strategy_type"):
            strategy_type = StrategyType(data["strategy_type"])

        # Handle datetime fields
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.now()

        updated_at = data.get("updated_at")
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)
        elif updated_at is None:
            updated_at = datetime.now()

        return cls(
            strategy_id=data["strategy_id"],
            strategy_name=data["strategy_name"],
            version=data.get("version", 1),
            symbol=data.get("symbol", "EURUSD"),
            timeframe=data.get("timeframe", "H4"),
            strategy_type=strategy_type,
            description=data.get("description", ""),
            entry_conditions=data.get("entry_conditions", []),
            exit_conditions=data.get("exit_conditions", []),
            position_sizing=position_sizing,
            parameters=data.get("parameters", {}),
            bot_type=data.get("bot_type", "scalping"),
            session_tags=data.get("session_tags", []),
            news_blackout=data.get("news_blackout", True),
            atr_stop_period=data.get("atr_stop_period", 14),
            atr_stop_multiplier=data.get("atr_stop_multiplier", 1.5),
            created_at=created_at,
            updated_at=updated_at,
            author=data.get("author", "Research Department"),
            source=data.get("source", "manual"),
            ambiguous_parameters=data.get("ambiguous_parameters", []),
        )

    def get_parameter(self, name: str, default: Any = None) -> Any:
        """Get parameter value by name."""
        return self.parameters.get(name, default)

    def set_parameter(self, name: str, value: Any) -> None:
        """Set parameter value."""
        self.parameters[name] = value
        self.updated_at = datetime.now()

    def get_all_parameters(self) -> List[TRDParameter]:
        """Get all parameters as TRDParameter objects."""
        params = []
        for name, value in self.parameters.items():
            param = TRDParameter(name=name, value=value)
            params.append(param)
        return params


# Standard TRD parameters with their expected types and ranges
STANDARD_PARAMETERS: Dict[str, Dict[str, Any]] = {
    "session_mask": {
        "type": "string",
        "description": "Trading session mask (e.g., 'UK/US', 'Asia')",
        "allowed_values": ["UK", "US", "Asia", "Europe", "Tokyo", "London", "NewYork", "Sydney"],
    },
    "force_close_hour": {
        "type": "integer",
        "description": "Hour to force close all positions (0-23)",
        "min_value": 0,
        "max_value": 23,
    },
    "overnight_hold": {
        "type": "boolean",
        "description": "Whether to hold positions overnight",
    },
    "daily_loss_cap": {
        "type": "float",
        "description": "Daily loss limit as percentage or absolute",
        "min_value": 0,
    },
    "spread_filter": {
        "type": "float",
        "description": "Maximum spread to open trades (in points)",
        "min_value": 0,
    },
    "max_spread_entry": {
        "type": "float",
        "description": "Maximum spread for entry (points)",
        "min_value": 0,
    },
    "slippage": {
        "type": "integer",
        "description": "Allowed slippage in points",
        "min_value": 0,
    },
    "magic_number": {
        "type": "integer",
        "description": "Magic number for EA trades",
        "min_value": 0,
    },
    "max_orders": {
        "type": "integer",
        "description": "Maximum concurrent orders",
        "min_value": 1,
        "max_value": 100,
    },
    "max_lots": {
        "type": "float",
        "description": "Maximum lot size per trade",
        "min_value": 0.01,
    },
    "use_martingale": {
        "type": "boolean",
        "description": "Whether to use martingale position sizing",
    },
    "martingale_multiplier": {
        "type": "float",
        "description": "Martingale multiplier after loss",
        "min_value": 1.0,
    },
    "trailing_stop": {
        "type": "boolean",
        "description": "Whether to use trailing stop",
    },
    "trailing_distance": {
        "type": "integer",
        "description": "Trailing stop distance in points",
        "min_value": 0,
    },
    "use_grid": {
        "type": "boolean",
        "description": "Whether to use grid trading",
    },
    "grid_spacing": {
        "type": "integer",
        "description": "Grid order spacing in points",
        "min_value": 0,
    },
    "max_grid_levels": {
        "type": "integer",
        "description": "Maximum grid levels",
        "min_value": 0,
    },
    "break_even_trigger": {
        "type": "integer",
        "description": "Profit in points to trigger break-even",
        "min_value": 0,
    },
    "break_even_distance": {
        "type": "integer",
        "description": "Break-even distance in points",
        "min_value": 0,
    },
    "close_on_opposite": {
        "type": "boolean",
        "description": "Close position on opposite signal",
    },
    "news_filter": {
        "type": "boolean",
        "description": "Whether to filter trades during news",
    },
    "news_offset_minutes": {
        "type": "integer",
        "description": "Minutes before/after news to avoid",
        "min_value": 0,
    },
}

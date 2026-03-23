"""
Strategy Template Models for Fast-Track Event Workflow

Template schema for matching news events to strategy templates.
Used by Alpha Forge fast-track deployment (Story 8.3).
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid


class StrategyType(str, Enum):
    """Strategy type enumeration."""
    NEWS_EVENT_BREAKOUT = "news_event_breakout"
    RANGE_EXPANDSION = "range_expansion"
    VOLATILITY_SPIKE = "volatility_spike"


class EventType(str, Enum):
    """News event types that templates can respond to."""
    HIGH_IMPACT_NEWS = "HIGH_IMPACT_NEWS"
    CENTRAL_BANK = "CENTRAL_BANK"
    GEOPOLITICAL = "GEOPOLITICAL"
    ECONOMIC_DATA = "ECONOMIC_DATA"
    EARNINGS = "EARNINGS"


class RiskProfile(str, Enum):
    """Risk profile for template."""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class StrategyTemplate(BaseModel):
    """
    Strategy Template for fast-track event workflow.

    Matches against news events for rapid deployment (11-15 minutes).
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    strategy_type: StrategyType
    applicable_events: List[EventType]
    risk_profile: RiskProfile
    avg_deployment_time: int = Field(description="Estimated deployment time in minutes")

    # Template parameters (EA input parameters)
    parameters: Dict[str, Any] = Field(default_factory=dict)

    # Supported symbols and timeframes
    symbols: List[str] = Field(default_factory=lambda: ["EURUSD", "GBPUSD", "USDJPY"])
    timeframes: List[str] = Field(default_factory=lambda: ["M15", "H1", "H4"])

    # Matching criteria
    match_conditions: Dict[str, Any] = Field(
        default_factory=lambda: {
            "min_confidence": 0.6,
            "regime_preferences": [],
            "volatility_preferences": {}
        }
    )

    # Fast-track specific settings
    fast_track_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "enabled": True,
            "max_deployment_time_minutes": 15,
            "conservative_lot_sizing": True,  # 0.5x normal
            "auto_expiry_hours": 24,  # Event strategies expire in 24h
            "overnight_hold": False,  # No overnight holds for events
            "force_close_hour": 21,  # Force close at 21:00
            "islamic_compliance": True
        }
    )

    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    created_by: str = "system"
    version: str = "1.0.0"
    is_active: bool = True


class TemplateMatchResult(BaseModel):
    """Result of template matching against a news event."""
    template: StrategyTemplate
    confidence_score: float
    estimated_deployment_time: int
    match_reasons: List[str]
    symbol_match: Optional[str] = None
    regime_match: Optional[str] = None


class TemplateListResponse(BaseModel):
    """Response for listing templates."""
    templates: List[Dict[str, Any]]
    total: int


# Default template parameters for each strategy type
DEFAULT_PARAMS = {
    StrategyType.NEWS_EVENT_BREAKOUT: {
        "ma_fast": 20,
        "ma_slow": 50,
        "atr_period": 14,
        "breakout_threshold": 1.5,
        "sl_points": 500,
        "tp_rr": 2.0,
        "max_spread": 30,
        "session_mask": "UK,US",
        "max_orders": 2,
    },
    StrategyType.RANGE_EXPANSION: {
        "bb_period": 20,
        "bb_std": 2.0,
        "rsi_period": 14,
        "rsi_overbought": 70,
        "rsi_oversold": 30,
        "range_threshold": 0.8,
        "sl_points": 400,
        "tp_rr": 1.5,
        "max_spread": 25,
    },
    StrategyType.VOLATILITY_SPIKE: {
        "atr_multiplier": 2.0,
        "volatility_lookback": 20,
        "volatility_threshold": 1.5,
        "momentum_period": 10,
        "sl_points": 600,
        "tp_rr": 2.5,
        "max_spread": 40,
        "use_trailing": True,
        "trailing_distance": 300,
    },
}

# Default fast-track config for each strategy type
DEFAULT_FAST_TRACK_CONFIG = {
    StrategyType.NEWS_EVENT_BREAKOUT: {
        "enabled": True,
        "max_deployment_time_minutes": 11,
        "conservative_lot_sizing": True,
        "auto_expiry_hours": 24,
        "overnight_hold": False,
        "force_close_hour": 21,
        "islamic_compliance": True,
    },
    StrategyType.RANGE_EXPANSION: {
        "enabled": True,
        "max_deployment_time_minutes": 13,
        "conservative_lot_sizing": True,
        "auto_expiry_hours": 24,
        "overnight_hold": False,
        "force_close_hour": 21,
        "islamic_compliance": True,
    },
    StrategyType.VOLATILITY_SPIKE: {
        "enabled": True,
        "max_deployment_time_minutes": 15,
        "conservative_lot_sizing": True,
        "auto_expiry_hours": 24,
        "overnight_hold": False,
        "force_close_hour": 21,
        "islamic_compliance": True,
    },
}
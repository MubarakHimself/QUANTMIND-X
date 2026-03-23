"""
Risk Management Data Models

Pydantic models for Enhanced Kelly Position Sizing with market physics integration.

Exports:
    StrategyPerformance: Strategy performance metrics for Kelly calculation
    MarketPhysics: Market regime physics indicators (Lyapunov, Ising, RMT)
    SizingRecommendation: Position sizing recommendation with constraints
    PositionSizingResult: Final position sizing result with margin calculations
    ValidationError: Base validation exception for model errors
    ModelValidationError: Specific validation error with field details

Calendar Models (Story 4-1):
    NewsItem: Economic calendar event
    CalendarRule: Per-account calendar rule configuration
    CalendarState: Current calendar state tracking
    CalendarEventType: Event type enumeration
    NewsImpact: Impact level enumeration
    CalendarPhase: Calendar phase enumeration
"""

from .strategy_performance import StrategyPerformance
from .market_physics import MarketPhysics, RiskLevel
from .sizing_recommendation import SizingRecommendation
from .position_sizing_result import PositionSizingResult

# Calendar models (Story 4-1)
from .calendar import (
    NewsItem,
    CalendarRule,
    CalendarState,
    CalendarEventType,
    NewsImpact,
    CalendarPhase,
)

__all__ = [
    "StrategyPerformance",
    "MarketPhysics",
    "RiskLevel",
    "SizingRecommendation",
    "PositionSizingResult",
    # Calendar models
    "NewsItem",
    "CalendarRule",
    "CalendarState",
    "CalendarEventType",
    "NewsImpact",
    "CalendarPhase",
]

"""
Strategy Template Schema Definition

Data structures for the fast-track strategy template library.
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
import uuid


class EventType(str, Enum):
    """Types of news events that templates can respond to."""
    HIGH_IMPACT_NEWS = "HIGH_IMPACT_NEWS"
    CENTRAL_BANK = "CENTRAL_BANK"
    GEOPOLITICAL = "GEOPOLITICAL"
    ECONOMIC_DATA = "ECONOMIC_DATA"
    MARKET_SHOCK = "MARKET_SHOCK"


class StrategyTypeTemplate(str, Enum):
    """Strategy template types for event-driven trading."""
    NEWS_EVENT_BREAKOUT = "news_event_breakout"
    RANGE_EXPANSION = "range_expansion"
    VOLATILITY_SPIKE = "volatility_spike"


class RiskProfile(str, Enum):
    """Risk profile for templates."""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


@dataclass
class TemplateParameter:
    """Individual template parameter with metadata."""
    name: str
    value: Any
    param_type: str = "string"
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "type": self.param_type,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "description": self.description,
        }


@dataclass
class StrategyTemplate:
    """
    Strategy Template for fast-track event-driven deployment.

    Represents a pre-configured strategy that can be quickly deployed
    when matching news events occur.
    """
    # Core identifiers
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    strategy_type: StrategyTypeTemplate = StrategyTypeTemplate.NEWS_EVENT_BREAKOUT

    # Event matching
    applicable_events: List[str] = field(default_factory=list)  # EventType values
    applicable_symbols: List[str] = field(default_factory=list)  # e.g., ["EURUSD", "GBPUSD"]

    # Risk and timing
    risk_profile: RiskProfile = RiskProfile.CONSERVATIVE
    avg_deployment_time: int = 11  # minutes

    # Template parameters (EA input parameters)
    parameters: Dict[str, Any] = field(default_factory=dict)

    # Fast-track specific settings
    lot_sizing_multiplier: float = 0.5  # 0.5x = conservative (half normal)
    auto_expiry_hours: int = 24  # Auto-expire after 24 hours
    is_islamic_compliant: bool = True

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    author: str = "Research Department"
    is_active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "strategy_type": self.strategy_type.value if self.strategy_type else None,
            "applicable_events": self.applicable_events,
            "applicable_symbols": self.applicable_symbols,
            "risk_profile": self.risk_profile.value if self.risk_profile else None,
            "avg_deployment_time": self.avg_deployment_time,
            "parameters": self.parameters,
            "lot_sizing_multiplier": self.lot_sizing_multiplier,
            "auto_expiry_hours": self.auto_expiry_hours,
            "is_islamic_compliant": self.is_islamic_compliant,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "author": self.author,
            "is_active": self.is_active,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategyTemplate":
        """Create StrategyTemplate from dictionary."""
        # Parse enums
        strategy_type = data.get("strategy_type")
        if strategy_type:
            strategy_type = StrategyTypeTemplate(strategy_type)

        risk_profile = data.get("risk_profile")
        if risk_profile:
            risk_profile = RiskProfile(risk_profile)

        # Parse datetime fields
        created_at = data.get("created_at")
        if created_at and isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        updated_at = data.get("updated_at")
        if updated_at and isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)

        return cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data.get("name", ""),
            strategy_type=strategy_type or StrategyTypeTemplate.NEWS_EVENT_BREAKOUT,
            applicable_events=data.get("applicable_events", []),
            applicable_symbols=data.get("applicable_symbols", []),
            risk_profile=risk_profile or RiskProfile.CONSERVATIVE,
            avg_deployment_time=data.get("avg_deployment_time", 11),
            parameters=data.get("parameters", {}),
            lot_sizing_multiplier=data.get("lot_sizing_multiplier", 0.5),
            auto_expiry_hours=data.get("auto_expiry_hours", 24),
            is_islamic_compliant=data.get("is_islamic_compliant", True),
            created_at=created_at,
            updated_at=updated_at,
            author=data.get("author", "Research Department"),
            is_active=data.get("is_active", True),
        )


@dataclass
class TemplateMatchResult:
    """Result of matching a template against a news event."""
    template: StrategyTemplate
    confidence_score: float  # 0.0 - 1.0
    match_factors: Dict[str, float]  # Breakdown of matching factors
    estimated_deployment_time: int  # minutes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "template_name": self.template.name,
            "confidence_score": self.confidence_score,
            "match_factors": self.match_factors,
            "estimated_deployment_time": self.estimated_deployment_time,
            "template": self.template.to_dict(),
        }
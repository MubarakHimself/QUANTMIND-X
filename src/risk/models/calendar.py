"""
Calendar Data Models - Economic Calendar Events & Rules

Data models for calendar-aware trading rules:
- NewsItem: Economic calendar event (NFP, ECB, CPI, FOMC, etc.)
- CalendarRule: Per-account calendar rule configuration
- CalendarEventType: Enumeration of economic event types
- NewsImpact: Impact level (HIGH, MEDIUM, LOW)
- CalendarPhase: Current calendar phase for position sizing

Story: 4-1-calendargovernor-news-blackout-calendar-aware-trading-rules
"""

import logging
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class CalendarEventType(str, Enum):
    """Economic calendar event types."""

    NFP = "NFP"  # Non-Farm Payrolls
    ECB = "ECB"  # ECB Rate Decision
    CPI = "CPI"  # Consumer Price Index
    FOMC = "FOMC"  # Federal Open Market Committee
    PPI = "PPI"  # Producer Price Index
    GDP = "GDP"  # Gross Domestic Product
    RETAIL = "RETAIL"  # Retail Sales
    MANUFACTURING = "MANUFACTURING"  # Manufacturing PMI
    UNEMPLOYMENT = "UNEMPLOYMENT"  # Unemployment Rate
    HOUSING = "HOUSING"  # Housing Data
    CONSUMER = "CONSUMER"  # Consumer Confidence
    OTHER = "OTHER"  # Other economic events


class NewsImpact(str, Enum):
    """News event impact level."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class CalendarPhase(str, Enum):
    """Calendar phase for lot scaling determination."""

    NORMAL = "normal"
    PRE_EVENT = "pre_event"
    DURING_EVENT = "during_event"
    POST_EVENT_REGIME_CHECK = "post_event_regime_check"


# Tier 1 events - most impactful, require stricter rules
TIER1_EVENT_TYPES: Set[CalendarEventType] = {
    CalendarEventType.NFP,
    CalendarEventType.ECB,
    CalendarEventType.CPI,
    CalendarEventType.FOMC,
}

# Default blackout window in minutes
DEFAULT_BLACKOUT_MINUTES = 30

# Default post-event reactivation delay in minutes
DEFAULT_POST_EVENT_DELAY_MINUTES = 60

# Default lot scaling factors
DEFAULT_LOT_SCALING = {
    "pre_event": 0.5,
    "during_event": 0.0,
    "post_event_regime_check": 0.75,
    "normal": 1.0,
}


class NewsItem(BaseModel):
    """Economic calendar event / news item.

    Represents a scheduled economic event that may affect trading:
    - NFP (Non-Farm Payrolls) - first Friday of each month
    - ECB Rate Decision - monthly
    - CPI (Consumer Price Index) - monthly
    - FOMC Meeting Minutes - 8 times per year

    Attributes:
        event_id: Unique identifier for the event
        title: Human-readable event title
        event_type: Type of economic event
        impact: Impact level (HIGH/MEDIUM/LOW)
        event_time: UTC timestamp of the event
        currencies: List of affected currency pairs (e.g., ["EUR", "USD"])
        description: Optional event description
    """

    event_id: str = Field(..., description="Unique event identifier")
    title: str = Field(..., description="Event title")
    event_type: CalendarEventType = Field(..., description="Type of economic event")
    impact: NewsImpact = Field(..., description="Impact level")
    event_time: datetime = Field(..., description="Event timestamp in UTC")
    currencies: List[str] = Field(default_factory=list, description="Affected currencies")
    description: Optional[str] = Field(None, description="Optional description")
    source: Optional[str] = Field(None, description="Data source (e.g., 'forexfactory', 'manual')")

    @field_validator("currencies", mode="before")
    @classmethod
    def validate_currencies(cls, v):
        """Ensure currencies is a list."""
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return list(v)

    @property
    def is_tier1(self) -> bool:
        """Check if this is a Tier 1 (high impact) event."""
        return self.event_type in TIER1_EVENT_TYPES or self.impact == NewsImpact.HIGH

    def is_approaching(self, minutes_threshold: int = DEFAULT_BLACKOUT_MINUTES, now: Optional[datetime] = None) -> bool:
        """Check if event is approaching within the specified threshold.

        Args:
            minutes_threshold: Minutes before event to consider "approaching"
            now: Reference time (defaults to datetime.now(UTC) if not provided)

        Returns:
            True if event is within the threshold
        """
        if now is None:
            now = datetime.now(timezone.utc)
        time_until_event = self.event_time - now
        return timedelta(minutes=0) <= time_until_event <= timedelta(minutes=minutes_threshold)

    def is_active(self, now: Optional[datetime] = None) -> bool:
        """Check if event is currently active (within event window).

        Args:
            now: Reference time (defaults to datetime.now(UTC) if not provided)
        """
        if now is None:
            now = datetime.now(timezone.utc)
        # Event is active from 15 minutes before to 15 minutes after
        return (self.event_time - timedelta(minutes=15)) <= now <= (self.event_time + timedelta(minutes=15))

    def is_past(self, now: Optional[datetime] = None) -> bool:
        """Check if event has passed.

        Args:
            now: Reference time (defaults to datetime.now(UTC) if not provided)
        """
        if now is None:
            now = datetime.now(timezone.utc)
        return now > (self.event_time + timedelta(minutes=15))

    def get_affected_pairs(self) -> List[str]:
        """Get list of affected currency pairs based on currencies.

        Returns:
            List of currency pair symbols (e.g., ["EURUSD", "GBPUSD"])
        """
        pairs = []
        major_currencies = ["USD", "EUR", "GBP", "JPY", "CHF", "AUD", "CAD", "NZD"]

        for curr in self.currencies:
            if curr in major_currencies:
                for other in major_currencies:
                    if curr != other:
                        # Create pair (e.g., EURUSD)
                        pair = f"{curr}{other}"
                        if pair not in pairs:
                            pairs.append(pair)
                        # Also add reverse
                        reverse = f"{other}{curr}"
                        if reverse not in pairs:
                            pairs.append(reverse)

        return pairs if pairs else ["EURUSD", "GBPUSD", "USDJPY"]  # Default pairs

    class Config:
        use_enum_values = True


class CalendarRule(BaseModel):
    """Per-account calendar rule configuration.

    Configures how calendar events affect trading for a specific account:
    - Blackout window duration (how long before event to apply rules)
    - Lot scaling factors for each phase
    - Post-event reactivation delay
    - Regime check configuration

    Attributes:
        rule_id: Unique rule identifier
        account_id: Account this rule applies to
        blacklist_enabled: Whether to apply blackout rules
        blackout_minutes: Minutes before event to start blackout
        lot_scaling_factors: Position sizing multipliers per phase
        post_event_delay_minutes: Minutes after event before reactivation
        regime_check_enabled: Whether to check regime before full reactivation
        affected_symbols: Symbols this rule applies to (empty = all)
        enabled: Whether this rule is active
    """

    rule_id: str = Field(..., description="Unique rule identifier")
    account_id: str = Field(..., description="Account this rule applies to")
    blacklist_enabled: bool = Field(True, description="Enable blackout rules")
    blackout_minutes: int = Field(DEFAULT_BLACKOUT_MINUTES, description="Blackout window in minutes")
    lot_scaling_factors: Dict[str, float] = Field(
        default_factory=lambda: DEFAULT_LOT_SCALING.copy(),
        description="Lot scaling factors per calendar phase"
    )
    post_event_delay_minutes: int = Field(
        DEFAULT_POST_EVENT_DELAY_MINUTES,
        description="Post-event reactivation delay in minutes"
    )
    regime_check_enabled: bool = Field(True, description="Enable regime check for reactivation")
    affected_symbols: List[str] = Field(default_factory=list, description="Affected symbols (empty = all)")
    enabled: bool = Field(True, description="Whether rule is active")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Rule creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update timestamp"
    )

    def get_lot_scaling(self, phase: CalendarPhase) -> float:
        """Get lot scaling factor for the given phase.

        Args:
            phase: Calendar phase

        Returns:
            Lot scaling factor (0.0 to 1.0)
        """
        return self.lot_scaling_factors.get(phase.value, 1.0)

    def is_applicable_to_symbol(self, symbol: str) -> bool:
        """Check if this rule applies to the given symbol.

        Args:
            symbol: Trading symbol (e.g., "EURUSD")

        Returns:
            True if rule applies to symbol
        """
        if not self.affected_symbols:
            return True  # Empty means all symbols

        # Check both directions (EURUSD and USDEUR)
        base = symbol[:3]
        quote = symbol[3:]
        return symbol in self.affected_symbols or base in self.affected_symbols or quote in self.affected_symbols

    class Config:
        use_enum_values = True


class CalendarState(BaseModel):
    """Current calendar state for an account.

    Tracks the current phase and active events for decision-making.

    Attributes:
        account_id: Account identifier
        current_phase: Current calendar phase
        active_event: Currently active news event (if any)
        lot_scaling_current: Current lot scaling factor
        last_evaluation: Last evaluation timestamp
    """

    account_id: str = Field(..., description="Account identifier")
    current_phase: CalendarPhase = Field(CalendarPhase.NORMAL, description="Current calendar phase")
    active_event: Optional[NewsItem] = Field(None, description="Active news event")
    lot_scaling_current: float = Field(1.0, description="Current lot scaling factor")
    last_evaluation: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last evaluation timestamp"
    )

    class Config:
        use_enum_values = True
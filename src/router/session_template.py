"""
Session Template Module — 6 Canonical Trading Sessions.

FIX-010: Session Template Class — 6 Canonical Trading Sessions

Defines 6 canonical trading sessions in a 24-hour cycle as a configurable class.
Consumed by SessionDetector, TiltStateMachine, and GovernorKellyEngine.

Per NFR-M2: SessionTemplate is a data/configuration class — NO LLM calls.
Per NFR-M3: All Python backend files kept under 500 lines.

6 Canonical Sessions (ACTIVE TRADING):
- Asian: 22:00-07:00 GMT — MR 70%/Mom 30%, 18 bots, non-premium
- London Open: 07:00-10:00 GMT — ORB 60%/Mom 40%, 50 bots, PREMIUM
- London Mid: 10:00-11:30 GMT — MR 50%/Mom 50%, 15 bots, non-premium
- Inter-Session: 11:30-13:00 GMT — MR 80%/TC 20%, 6 bots, non-premium
- NY+Overlap: 13:00-16:00 GMT — ORB 55%/Mom 45%, 60 bots, PREMIUM
- NY Wind-Down: 16:00-17:00 GMT — TC 60%/MR 40%, 8 bots, non-premium

Dead Zone: 17:00-22:00 GMT — NO TRADING (special case in SessionDetector, not a session template)
"""

from datetime import datetime, time, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

if TYPE_CHECKING:
    from src.events.session_template import SessionTemplateEvent


# =============================================================================
# Constants
# =============================================================================

# Dead Zone: 17:00-22:00 GMT — no new scalping/ORB entries authorized
DEAD_ZONE_START: int = 17  # 17:00 GMT
DEAD_ZONE_END: int = 22    # 22:00 GMT

# Premium Kelly boost multiplier (applied at Governor layer)
PREMIUM_KELLY_MULTIPLIER: float = 1.4

# Base Kelly multiplier for non-premium windows
BASE_KELLY_MULTIPLIER: float = 1.0

# Premium sessions that receive Kelly boost (ORB boost of 6%)
PREMIUM_SESSION_NAMES: List[str] = [
    "London Open",
    "NY+Overlap",
]

# Session check order for get_session_at - canonical order for 6 sessions
# Check order prioritizes London Open before London Mid since they overlap (07:00-10:00 vs 10:00-11:30)
SESSION_CHECK_ORDER: List[str] = [
    "Asian",             # 22:00-07:00 (overnight)
    "London Open",       # 07:00-10:00 (premium)
    "London Mid",        # 10:00-11:30
    "Inter-Session",     # 11:30-13:00
    "NY+Overlap",        # 13:00-16:00 (premium)
    "NY Wind-Down",      # 16:00-17:00
]


# =============================================================================
# Enums
# =============================================================================

class CanonicalSession(str, Enum):
    """
    6 canonical trading sessions in the 24-hour trading cycle.

    These are the definitive session identifiers used throughout
    the trading system for session-aware bot filtering and Kelly sizing.
    Dead Zone (17:00-22:00) is NO TRADING — handled as special case.
    """
    ASIAN = "Asian"
    LONDON_OPEN = "London Open"
    LONDON_MID = "London Mid"
    INTER_SESSION = "Inter-Session"
    NY_PLUS_OVERLAP = "NY+Overlap"
    NY_WIND_DOWN = "NY Wind-Down"


class WindowIntensity(str, Enum):
    """
    Trading intensity levels for session windows.

    Used to configure bot type mix and execution aggressiveness.
    """
    VERY_LOW = "VERY_LOW"   # e.g., Sydney-Tokyo Overlap
    LOW = "LOW"             # e.g., Sydney Open, NY Wind-Down, Inter-Session Cooldown
    MODERATE = "MODERATE"   # e.g., Tokyo Open, London Mid
    HIGH = "HIGH"           # e.g., Tokyo-London Overlap
    PREMIUM = "PREMIUM"      # e.g., London Open, London-NY Overlap (implies Kelly boost)


# =============================================================================
# Pydantic Models
# =============================================================================

class BotTypeMix(BaseModel):
    """
    Bot type mix percentages for a session window.

    All percentages must sum to 100. Used to configure the proportion
    of different trading strategies active during each window.

    Attributes:
        orb_pct: Percentage of ORB (Opening Range Breakout) bots (0-100)
        momentum_pct: Percentage of momentum bots (0-100)
        mean_reversion_pct: Percentage of mean reversion bots (0-100)
        trend_continuation_pct: Percentage of trend continuation bots (0-100)
    """
    orb_pct: int = Field(default=0, ge=0, le=100, description="ORB bot percentage")
    momentum_pct: int = Field(default=0, ge=0, le=100, description="Momentum bot percentage")
    mean_reversion_pct: int = Field(default=0, ge=0, le=100, description="Mean reversion bot percentage")
    trend_continuation_pct: int = Field(default=0, ge=0, le=100, description="Trend continuation bot percentage")

    @field_validator("orb_pct", "momentum_pct", "mean_reversion_pct", "trend_continuation_pct")
    @classmethod
    def validate_percentages(cls, v: int) -> int:
        """Validate percentage is within valid range."""
        if v < 0 or v > 100:
            raise ValueError(f"Percentage must be between 0 and 100, got {v}")
        return v

    def total_pct(self) -> int:
        """Calculate total percentage across all bot types."""
        return self.orb_pct + self.momentum_pct + self.mean_reversion_pct + self.trend_continuation_pct

    def validate_sum(self) -> bool:
        """
        Validate that all percentages sum to 100 or 0.

        Valid sums:
        - 100: Normal trading windows with full bot allocation
        - 0: Dead Zone where no trading occurs

        Returns:
            True if sum is valid (0 or 100)
        """
        total = self.total_pct()
        return total == 0 or total == 100


class EventFilter(BaseModel):
    """
    Event filter configuration for a session window.

    Used to block or modify trading during specific market events
    (e.g., SNB event filter for Zurich session, NFP, FOMC, etc.).

    Attributes:
        filter_type: Type of event filter (e.g., "SNB", "FOMC", "NFP")
        affected_windows: List of window names this filter applies to
        description: Human-readable description of the filter
    """
    filter_type: str = Field(description="Type of event filter (e.g., SNB, FOMC, NFP)")
    affected_windows: List[str] = Field(default_factory=list, description="Windows this filter affects")
    description: str = Field(default="", description="Human-readable filter description")


class SessionWindow(BaseModel):
    """
    Configuration for a single trading session.

    Defines all parameters for one of the 6 canonical sessions including
    timing, eligible symbols, Kelly multiplier, bot type mix, and concurrency limits.

    Attributes:
        name: Canonical session name
        start_gmt: Session start time in "HH:MM" GMT format
        end_gmt: Session end time in "HH:MM" GMT format
        eligible_symbols: List of symbols eligible for trading during this session
        eligible_strategy_types: List of strategy types allowed in this session
        kelly_multiplier: Kelly multiplier for this session (1.0 baseline, premium for premium sessions)
        orb_boost_pct: ORB boost percentage for this session (0 for non-premium, 6 for premium)
        max_concurrent_bots: Maximum number of concurrent bots for this session
        house_money_threshold_override: Optional override for house money threshold (None = use global)
        event_filters: List of event filters applicable to this session
        bot_type_mix: Bot type mix configuration
        intensity: Trading intensity level
        notes: Additional notes about this session
    """
    name: str = Field(description="Canonical session name")
    start_gmt: str = Field(description="Session start in HH:MM GMT format (e.g., '22:00')")
    end_gmt: str = Field(description="Session end in HH:MM GMT format (e.g., '07:00')")
    eligible_symbols: List[str] = Field(default_factory=list, description="Eligible trading symbols")
    eligible_strategy_types: List[str] = Field(default_factory=list, description="Eligible strategy types")
    kelly_multiplier: float = Field(default=1.0, ge=0.0, description="Kelly multiplier for this session")
    orb_boost_pct: int = Field(default=0, ge=0, le=100, description="ORB boost percentage (0 or 6)")
    max_concurrent_bots: int = Field(default=10, ge=0, description="Maximum concurrent bots for this session")
    house_money_threshold_override: Optional[float] = Field(default=None, description="Override house money threshold")
    event_filters: List[EventFilter] = Field(default_factory=list, description="Event filters for this session")
    bot_type_mix: BotTypeMix = Field(default_factory=BotTypeMix, description="Bot type mix configuration")
    intensity: WindowIntensity = Field(default=WindowIntensity.LOW, description="Trading intensity")
    notes: str = Field(default="", description="Additional notes")

    def get_start_time(self) -> time:
        """Parse start_gmt to time object."""
        hour, minute = map(int, self.start_gmt.split(":"))
        return time(hour=hour, minute=minute)

    def get_end_time(self) -> time:
        """Parse end_gmt to time object."""
        hour, minute = map(int, self.end_gmt.split(":"))
        return time(hour=hour, minute=minute)

    def contains_gmt_time(self, gmt_time: time) -> bool:
        """
        Check if a GMT time falls within this window.

        Handles overnight windows where start > end (e.g., 22:00-07:00).

        Args:
            gmt_time: Time to check in GMT

        Returns:
            True if the time falls within the window
        """
        start = self.get_start_time()
        end = self.get_end_time()

        if start <= end:
            # Normal range (e.g., 08:00-12:00)
            return start <= gmt_time < end
        else:
            # Overnight range (e.g., 22:00-07:00)
            return gmt_time >= start or gmt_time < end


class SessionTemplateConfig(BaseModel):
    """
    Configuration for the 6-canonical-session template.

    Contains all 6 canonical trading sessions and metadata about the template.
    Dead Zone (17:00-22:00) is handled as a special case, not stored here.

    Attributes:
        name: Template name (e.g., "default", "high-activity")
        description: Human-readable description
        sessions: Dictionary of CanonicalSession -> SessionWindow
        dead_zone_start: Dead zone start hour in GMT (0-23)
        dead_zone_end: Dead zone end hour in GMT (0-23)
        premium_kelly_multiplier: Kelly multiplier for premium sessions
    """
    name: str = Field(default="default", description="Template name")
    description: str = Field(default="", description="Template description")
    sessions: Dict[str, SessionWindow] = Field(default_factory=dict, description="Session configurations")
    dead_zone_start: int = Field(default=DEAD_ZONE_START, ge=0, le=23, description="Dead zone start hour GMT")
    dead_zone_end: int = Field(default=DEAD_ZONE_END, ge=0, le=23, description="Dead zone end hour GMT")
    premium_kelly_multiplier: float = Field(default=PREMIUM_KELLY_MULTIPLIER, ge=0.0, description="Premium Kelly multiplier")

    def get_session(self, session_name: str) -> Optional[SessionWindow]:
        """Get a session configuration by name."""
        return self.sessions.get(session_name)

    def get_all_sessions(self) -> List[SessionWindow]:
        """Get all configured sessions sorted by canonical order."""
        ordered_sessions = []
        for canonical in CanonicalSession:
            session = self.sessions.get(canonical.value)
            if session:
                ordered_sessions.append(session)
        return ordered_sessions


# =============================================================================
# Default 6-Session Configuration
# =============================================================================

def _create_default_sessions() -> Dict[str, SessionWindow]:
    """
    Create the default 6-canonical-session configuration.

    This defines the canonical 24-hour trading cycle with per-session
    bot type mix, intensity, Kelly multiplier, and concurrency limits.

    Returns:
        Dictionary of session name -> SessionWindow configuration
    """
    return {
        CanonicalSession.ASIAN.value: SessionWindow(
            name=CanonicalSession.ASIAN.value,
            start_gmt="22:00",
            end_gmt="07:00",
            eligible_symbols=["AUDUSD", "NZDUSD", "AUDJPY"],
            eligible_strategy_types=["mean_reversion", "momentum"],
            kelly_multiplier=BASE_KELLY_MULTIPLIER,
            orb_boost_pct=0,
            max_concurrent_bots=18,
            house_money_threshold_override=None,
            event_filters=[],
            bot_type_mix=BotTypeMix(orb_pct=0, momentum_pct=30, mean_reversion_pct=70, trend_continuation_pct=0),
            intensity=WindowIntensity.LOW,
            notes="Asian session — MR dominant",
        ),
        CanonicalSession.LONDON_OPEN.value: SessionWindow(
            name=CanonicalSession.LONDON_OPEN.value,
            start_gmt="07:00",
            end_gmt="10:00",
            eligible_symbols=["EURUSD", "GBPUSD", "USDJPY", "EURGBP", "AUDUSD"],
            eligible_strategy_types=["ORB", "momentum"],
            kelly_multiplier=BASE_KELLY_MULTIPLIER,
            orb_boost_pct=6,
            max_concurrent_bots=50,
            house_money_threshold_override=0.06,
            event_filters=[],
            bot_type_mix=BotTypeMix(orb_pct=60, momentum_pct=40, mean_reversion_pct=0, trend_continuation_pct=0),
            intensity=WindowIntensity.PREMIUM,
            notes="London Open — PREMIUM, ORB dominant",
        ),
        CanonicalSession.LONDON_MID.value: SessionWindow(
            name=CanonicalSession.LONDON_MID.value,
            start_gmt="10:00",
            end_gmt="11:30",
            eligible_symbols=["EURUSD", "GBPUSD", "USDJPY"],
            eligible_strategy_types=["mean_reversion", "momentum"],
            kelly_multiplier=BASE_KELLY_MULTIPLIER,
            orb_boost_pct=0,
            max_concurrent_bots=15,
            house_money_threshold_override=None,
            event_filters=[],
            bot_type_mix=BotTypeMix(orb_pct=0, momentum_pct=50, mean_reversion_pct=50, trend_continuation_pct=0),
            intensity=WindowIntensity.MODERATE,
            notes="London Mid — balanced MR/Mom",
        ),
        CanonicalSession.INTER_SESSION.value: SessionWindow(
            name=CanonicalSession.INTER_SESSION.value,
            start_gmt="11:30",
            end_gmt="13:00",
            eligible_symbols=["EURUSD", "GBPUSD"],
            eligible_strategy_types=["mean_reversion", "trend_continuation"],
            kelly_multiplier=BASE_KELLY_MULTIPLIER,
            orb_boost_pct=0,
            max_concurrent_bots=6,
            house_money_threshold_override=None,
            event_filters=[],
            bot_type_mix=BotTypeMix(orb_pct=0, momentum_pct=0, mean_reversion_pct=80, trend_continuation_pct=20),
            intensity=WindowIntensity.LOW,
            notes="Inter-Session — MR dominant, lowest bot count",
        ),
        CanonicalSession.NY_PLUS_OVERLAP.value: SessionWindow(
            name=CanonicalSession.NY_PLUS_OVERLAP.value,
            start_gmt="13:00",
            end_gmt="16:00",
            eligible_symbols=["EURUSD", "GBPUSD", "USDCHF", "USDCAD"],
            eligible_strategy_types=["ORB", "momentum"],
            kelly_multiplier=BASE_KELLY_MULTIPLIER,
            orb_boost_pct=6,
            max_concurrent_bots=60,
            house_money_threshold_override=0.06,
            event_filters=[],
            bot_type_mix=BotTypeMix(orb_pct=55, momentum_pct=45, mean_reversion_pct=0, trend_continuation_pct=0),
            intensity=WindowIntensity.PREMIUM,
            notes="NY+Overlap — PREMIUM, highest bot count",
        ),
        CanonicalSession.NY_WIND_DOWN.value: SessionWindow(
            name=CanonicalSession.NY_WIND_DOWN.value,
            start_gmt="16:00",
            end_gmt="17:00",
            eligible_symbols=["EURUSD", "GBPUSD"],
            eligible_strategy_types=["trend_continuation", "mean_reversion"],
            kelly_multiplier=BASE_KELLY_MULTIPLIER,
            orb_boost_pct=0,
            max_concurrent_bots=8,
            house_money_threshold_override=None,
            event_filters=[],
            bot_type_mix=BotTypeMix(orb_pct=0, momentum_pct=40, mean_reversion_pct=40, trend_continuation_pct=60),
            intensity=WindowIntensity.LOW,
            notes="NY Wind-Down — TC dominant",
        ),
    }


def create_default_template() -> SessionTemplateConfig:
    """Create the default session template configuration."""
    return SessionTemplateConfig(
        name="default",
        description="Default 6-canonical-session trading cycle",
        sessions=_create_default_sessions(),
        dead_zone_start=DEAD_ZONE_START,
        dead_zone_end=DEAD_ZONE_END,
        premium_kelly_multiplier=PREMIUM_KELLY_MULTIPLIER,
    )


# =============================================================================
# SessionTemplate Class
# =============================================================================

class SessionTemplate:
    """
    Configurable 6-Canonical-Session Template.

    This is the single source of truth for all session definitions
    in the 24-hour trading cycle. Consumed by:
    - SessionDetector: for canonical session time detection
    - TiltStateMachine: for session boundary handoff context
    - GovernorKellyEngine: for Kelly multiplier decisions

    Key behaviors:
    1. Provides session at time lookup (get_session_at)
    2. Identifies premium sessions for Kelly boost (is_premium_session)
    3. Determines if trading is authorized (is_trading_authorised)
    4. Returns bot type mix, intensity, and concurrency limits per session

    Attributes:
        config: The session template configuration
        PREMIUM_SESSIONS: Class constant for premium session names
        DEAD_ZONE_START: Dead zone start hour GMT
        DEAD_ZONE_END: Dead zone end hour GMT
    """

    PREMIUM_SESSIONS: List[str] = PREMIUM_SESSION_NAMES
    DEAD_ZONE_START: int = DEAD_ZONE_START
    DEAD_ZONE_END: int = DEAD_ZONE_END

    def __init__(self, config: Optional[SessionTemplateConfig] = None):
        """
        Initialize SessionTemplate.

        Args:
            config: Optional configuration. If not provided, uses default.
        """
        self.config = config if config is not None else create_default_template()

    def get_session_at(self, gmt_time: datetime) -> SessionWindow:
        """
        Get the active session at the given GMT time.

        Iterates through all sessions and returns the first one that
        contains the given time. Sessions are checked in canonical order.

        Args:
            gmt_time: UTC datetime to check

        Returns:
            SessionWindow for the active session

        Raises:
            ValueError: If no session is found for the given time (Dead Zone)
        """
        # Ensure UTC
        if gmt_time.tzinfo is None:
            gmt_time = gmt_time.replace(tzinfo=timezone.utc)
        elif gmt_time.tzinfo != timezone.utc:
            gmt_time = gmt_time.astimezone(timezone.utc)

        gmt_time_only = gmt_time.time()

        # Check each session in canonical order
        for session_name in SESSION_CHECK_ORDER:
            session = self.config.sessions.get(session_name)
            if session and session.contains_gmt_time(gmt_time_only):
                return session

        # Dead Zone: 17:00-22:00 — no trading session, return special indicator
        raise ValueError(f"No active trading session for time {gmt_time} (Dead Zone 17:00-22:00)")

    def is_premium_session(self, session_name: str) -> bool:
        """
        Check if a session is a premium Kelly boost session.

        Premium sessions receive Kelly multiplier boost as defined in
        Story 4.10 (Session Kelly Modifiers).

        Args:
            session_name: Name of the session to check

        Returns:
            True if the session is a premium Kelly boost session
        """
        return session_name in self.PREMIUM_SESSIONS

    def is_active_trading_session(self, session_name: str) -> bool:
        """
        Check if a session is an active trading session.

        All 6 canonical sessions are active trading sessions.
        Dead Zone (17:00-22:00) is NOT a session in this template.

        Args:
            session_name: Name of the session to check

        Returns:
            True if the session is an active trading session
        """
        return session_name in self.config.sessions

    def is_dead_zone(self, gmt_time: datetime) -> bool:
        """
        Check if the given GMT time falls within the Dead Zone.

        Dead Zone: 16:00-22:00 GMT — no new scalping or ORB entries authorized.

        Args:
            gmt_time: UTC datetime to check

        Returns:
            True if the time is within the Dead Zone
        """
        # Ensure UTC
        if gmt_time.tzinfo is None:
            gmt_time = gmt_time.replace(tzinfo=timezone.utc)
        elif gmt_time.tzinfo != timezone.utc:
            gmt_time = gmt_time.astimezone(timezone.utc)

        hour = gmt_time.hour

        # Dead zone spans 16:00-22:00 (hour >= 16 and hour < 22)
        return hour >= self.DEAD_ZONE_START and hour < self.DEAD_ZONE_END

    def is_trading_authorised(self, gmt_time: datetime, active_events: Optional[List[str]] = None) -> bool:
        """
        Check if new trades are authorized at the given GMT time.

        Returns False during Dead Zone (17:00-22:00). Also checks event filters if
        active_events list is provided.

        Args:
            gmt_time: UTC datetime to check
            active_events: Optional list of currently active event types
                (e.g., ["SNB", "FOMC"]). If None, event filters are ignored.

        Returns:
            True if trading is authorized
        """
        # Check dead zone
        if self.is_dead_zone(gmt_time):
            return False

        # Get current session and check event filters
        try:
            session = self.get_session_at(gmt_time)
        except ValueError:
            # Dead Zone — should not reach here since is_dead_zone catches it
            return False

        # Check if any event filters block trading
        if active_events and session.event_filters:
            for event_filter in session.event_filters:
                if event_filter.filter_type in active_events:
                    return False

        return True

    def get_kelly_multiplier(self, session_name: str, base_kelly: float) -> float:
        """
        Get the Kelly multiplier for a session.

        Premium sessions receive the configured premium multiplier.
        All other active sessions receive base Kelly (1.0).

        Note: This returns the session's Kelly multiplier. The actual
        Kelly sizing is computed at the Governor layer which applies
        additional modifiers (HMM, Reverse HMM, etc.).

        Args:
            session_name: Name of the session
            base_kelly: Base Kelly value to adjust

        Returns:
            Adjusted Kelly multiplier
        """
        if self.is_premium_session(session_name):
            return self.config.premium_kelly_multiplier * base_kelly
        return base_kelly

    def get_bot_type_mix(self, session_name: str) -> BotTypeMix:
        """
        Get the bot type mix for a session.

        Args:
            session_name: Name of the session

        Returns:
            BotTypeMix configuration for the session

        Raises:
            ValueError: If session is not found
        """
        session = self.config.sessions.get(session_name)
        if session is None:
            raise ValueError(f"Session not found: {session_name}")
        return session.bot_type_mix

    def get_intensity(self, session_name: str) -> WindowIntensity:
        """
        Get the trading intensity for a session.

        Args:
            session_name: Name of the session

        Returns:
            WindowIntensity for the session

        Raises:
            ValueError: If session is not found
        """
        session = self.config.sessions.get(session_name)
        if session is None:
            raise ValueError(f"Session not found: {session_name}")
        return session.intensity

    def get_max_concurrent_bots(self, session_name: str) -> int:
        """
        Get the maximum concurrent bots for a session.

        Args:
            session_name: Name of the session

        Returns:
            Maximum number of concurrent bots for the session

        Raises:
            ValueError: If session is not found
        """
        session = self.config.sessions.get(session_name)
        if session is None:
            raise ValueError(f"Session not found: {session_name}")
        return session.max_concurrent_bots

    def get_house_money_threshold_override(self, session_name: str) -> Optional[float]:
        """
        Get the house money threshold override for a session.

        Returns None if the session does not override the global setting.

        Args:
            session_name: Name of the session

        Returns:
            House money threshold override or None if using global setting

        Raises:
            ValueError: If session is not found
        """
        session = self.config.sessions.get(session_name)
        if session is None:
            raise ValueError(f"Session not found: {session_name}")
        return session.house_money_threshold_override

    def is_filtered(self, session_name: str, active_events: List[str]) -> bool:
        """
        Check if trading is filtered for a session due to active events.

        Args:
            session_name: Name of the session to check
            active_events: List of currently active event types (e.g., ["SNB", "FOMC"])

        Returns:
            True if trading should be blocked by an active event filter
        """
        session = self.config.sessions.get(session_name)
        if session is None:
            return False

        for event_filter in session.event_filters:
            if event_filter.filter_type in active_events:
                return True

        return False

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to dictionary for JSON storage.

        Returns:
            Dictionary representation of the template
        """
        return {
            "name": self.config.name,
            "description": self.config.description,
            "sessions": {name: session.model_dump() for name, session in self.config.sessions.items()},
            "dead_zone_start": self.config.dead_zone_start,
            "dead_zone_end": self.config.dead_zone_end,
            "premium_kelly_multiplier": self.config.premium_kelly_multiplier,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionTemplate":
        """
        Deserialize from dictionary.

        Args:
            data: Dictionary representation of the template

        Returns:
            SessionTemplate instance
        """
        config = SessionTemplateConfig(
            name=data.get("name", "default"),
            description=data.get("description", ""),
            sessions={name: SessionWindow(**session_data) for name, session_data in data.get("sessions", {}).items()},
            dead_zone_start=data.get("dead_zone_start", DEAD_ZONE_START),
            dead_zone_end=data.get("dead_zone_end", DEAD_ZONE_END),
            premium_kelly_multiplier=data.get("premium_kelly_multiplier", PREMIUM_KELLY_MULTIPLIER),
        )
        return cls(config=config)

    def reload(self, new_config: Optional[SessionTemplateConfig] = None) -> None:
        """
        Reload configuration at runtime without restart.

        Args:
            new_config: New configuration to apply. If None, re-creates default.
        """
        if new_config is not None:
            self.config = new_config
        else:
            self.config = create_default_template()

    def validate_24_hour_coverage(self) -> bool:
        """
        Validate that sessions provide 24-hour coverage with no gaps.

        Checks that the union of all session time ranges covers every hour
        of the day except Dead Zone (17:00-22:00) which is intentionally
        a no-trading period.

        Returns:
            True if 24-hour coverage is complete and valid
        """
        # Build a 24-hour bitmask to track coverage
        hour_coverage = [False] * 24

        for session_name, session in self.config.sessions.items():
            start_hour = int(session.start_gmt.split(":")[0])
            end_hour = int(session.end_gmt.split(":")[0])

            if start_hour <= end_hour:
                # Normal range (e.g., 07:00-10:00)
                for hour in range(start_hour, end_hour):
                    hour_coverage[hour] = True
            else:
                # Overnight range (e.g., 22:00-07:00)
                for hour in range(start_hour, 24):
                    hour_coverage[hour] = True
                for hour in range(0, end_hour):
                    hour_coverage[hour] = True

        # Dead Zone hours (17:00-22:00) are intentionally not covered by sessions
        # These hours should be False in hour_coverage
        for hour in range(self.DEAD_ZONE_START, self.DEAD_ZONE_END):
            if hour_coverage[hour]:
                return False  # Session should not cover Dead Zone hours

        # All non-Dead Zone hours must be covered
        for hour in range(24):
            if hour < self.DEAD_ZONE_START or hour >= self.DEAD_ZONE_END:
                if not hour_coverage[hour]:
                    return False

        return True

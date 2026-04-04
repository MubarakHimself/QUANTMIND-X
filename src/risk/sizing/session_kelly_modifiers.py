"""
Session Kelly Modifiers Module

Manages session-scoped Kelly criterion modifiers:
- House Money Mode (HMM): Global daily P&L based Kelly multipliers
- Reverse HMM: Session-level consecutive loss penalties
- Premium Session: Lowered thresholds during high-value sessions

Multiplier Chain: Kelly_base --> [RVOL x SessionKelly x Correlation x HouseMoney]

Story 4.10: Session-Scoped Kelly Modifiers
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any

from src.router.sessions import SessionDetector, TradingSession

logger = logging.getLogger(__name__)


class PremiumSessionAssault(Enum):
    """Premium session assault windows with Kelly boost."""
    LONDON_OPEN = "london_open"          # London Open assault
    LONDON_NY_OVERLAP = "london_ny_overlap"  # London-NY Overlap assault
    NY_OPEN = "ny_open"                  # NY Open assault


@dataclass
class SessionKellyState:
    """Immutable snapshot of session Kelly modifier state."""
    # House Money Mode state
    hmm_multiplier: float = 1.0
    is_preservation_mode: bool = False
    is_house_money_active: bool = False

    # Reverse HMM state
    session_loss_counter: int = 0
    reverse_hmm_multiplier: float = 1.0
    premium_boost_active: bool = False

    # Premium session state
    is_premium_session: bool = False
    premium_assault: Optional[PremiumSessionAssault] = None

    # Composite modifier
    session_kelly_multiplier: float = 1.0

    # Metadata
    current_session: str = ""
    daily_pnl_pct: float = 0.0
    last_updated: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/debugging."""
        return {
            "hmm_multiplier": self.hmm_multiplier,
            "is_preservation_mode": self.is_preservation_mode,
            "is_house_money_active": self.is_house_money_active,
            "session_loss_counter": self.session_loss_counter,
            "reverse_hmm_multiplier": self.reverse_hmm_multiplier,
            "premium_boost_active": self.premium_boost_active,
            "is_premium_session": self.is_premium_session,
            "premium_assault": self.premium_assault.value if self.premium_assault else None,
            "session_kelly_multiplier": self.session_kelly_multiplier,
            "current_session": self.current_session,
            "daily_pnl_pct": self.daily_pnl_pct,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
        }


class SessionKellyModifiers:
    """
    Session-Scoped Kelly Modifier Engine.

    Applies global session-level modifiers to Kelly criterion sizing:
    1. House Money Mode (HMM): Daily P&L based Kelly multipliers
       - Daily P&L >= +8% --> Kelly x 1.5x (House Money effect)
       - Daily P&L <= -10% --> Kelly x 0.5x (Preservation mode)
       - Otherwise --> Kelly x 1.0x baseline

    2. Premium Session Threshold Lowering:
       - Premium sessions (London Open assault, London-NY Overlap, NY Open assault)
         have lowered equity thresholds for HMM activation

    3. Reverse House Money Effect:
       - 2 consecutive session losses --> 1.0x (premium boost removed)
       - 4 consecutive session losses --> 0.70x
       - 6 consecutive session losses --> 0.50x

    4. Win-Reset:
       - Winning trade resets session_loss_counter to 0
       - Premium boost requires 2 consecutive session-level wins to re-enable

    5. Session Auto-Reset:
       - On session boundary, all modifiers reset to 1.0x

    Usage:
        modifiers = SessionKellyModifiers()
        state = modifiers.compute_session_kelly_modifier(
            daily_pnl_pct=0.10,  # +10% daily P&L
            current_session=TradingSession.LONDON,
            utc_now=datetime.now(timezone.utc)
        )
        # state.session_kelly_multiplier = 1.5 (HMM active)

        modifiers.on_trade_result(is_win=True)  # Win resets loss counter
        modifiers.on_trade_result(is_win=False)  # Loss increments counter
        modifiers.on_session_close()  # Reset all modifiers
    """

    # HMM Thresholds (Story 4.10 / Addendum F-06)
    HMM_PROFIT_THRESHOLD_PCT: float = 0.08   # +8% triggers House Money (1.5x)
    HMM_LOSS_THRESHOLD_PCT: float = -0.10    # -10% triggers Preservation (0.5x)

    # HMM Multipliers (Story 4.10 / Addendum F-06)
    HMM_MULTIPLIER_HOUSE_MONEY: float = 1.5  # +8% daily P&L threshold
    HMM_MULTIPLIER_PRESERVATION: float = 0.5  # -10% daily P&L threshold
    HMM_MULTIPLIER_BASELINE: float = 1.0     # Normal trading

    # Reverse HMM Multipliers (Story 4.10)
    REVERSE_HMM_2_LOSSES: float = 1.0    # 2 losses: remove premium boost
    REVERSE_HMM_4_LOSSES: float = 0.70   # 4 losses: 0.70x
    REVERSE_HMM_6_LOSSES: float = 0.50   # 6 losses: 0.50x (minimum viable)

    # Premium session threshold boost (percentage points to lower threshold)
    PREMIUM_THRESHOLD_BOOST_PCT: float = 0.02  # Lower threshold by 2% during premium sessions

    # Win reset requirements
    WINS_TO_REENABLE_PREMIUM: int = 2  # 2 consecutive wins to re-enable premium boost

    # Premium session assault definitions
    PREMIUM_ASSAULTS: Dict[PremiumSessionAssault, Dict[str, Any]] = {
        PremiumSessionAssault.LONDON_OPEN: {
            "name": "London Open Assault",
            "description": "High-liquidity London session open",
        },
        PremiumSessionAssault.LONDON_NY_OVERLAP: {
            "name": "London-NY Overlap Assault",
            "description": "Highest-liquidity overlap period",
        },
        PremiumSessionAssault.NY_OPEN: {
            "name": "NY Open Assault",
            "description": "High-liquidity NY session open",
        },
    }

    def __init__(
        self,
        account_id: Optional[str] = None,
        hmm_profit_threshold: float = HMM_PROFIT_THRESHOLD_PCT,
        hmm_loss_threshold: float = HMM_LOSS_THRESHOLD_PCT,
        premium_threshold_boost: float = PREMIUM_THRESHOLD_BOOST_PCT,
    ):
        """
        Initialize SessionKellyModifiers.

        Args:
            account_id: Optional account ID for persistence
            hmm_profit_threshold: Daily P&L % to trigger House Money (default 8%)
            hmm_loss_threshold: Daily P&L % to trigger Preservation (default -10%)
            premium_threshold_boost: Percentage points to lower threshold during premium (default 2%)
        """
        self.account_id = account_id
        self.hmm_profit_threshold = hmm_profit_threshold
        self.hmm_loss_threshold = hmm_loss_threshold
        self.premium_threshold_boost = premium_threshold_boost

        # Session state
        self._current_session: Optional[TradingSession] = None
        self._session_start: Optional[datetime] = None

        # HMM state
        self._hmm_multiplier: float = self.HMM_MULTIPLIER_BASELINE
        self._is_preservation_mode: bool = False
        self._is_house_money_active: bool = False

        # Reverse HMM state
        self._session_loss_counter: int = 0
        self._reverse_hmm_multiplier: float = self.HMM_MULTIPLIER_BASELINE
        self._premium_boost_active: bool = False
        self._consecutive_session_wins: int = 0

        # London-NY overlap consecutive losses (for overlap-specific reverse HMM)
        self._overlap_consecutive_losses: int = 0

        # Premium session state
        self._is_premium_session: bool = False
        self._current_premium_assault: Optional[PremiumSessionAssault] = None

        logger.info(
            f"SessionKellyModifiers initialized: account_id={account_id}, "
            f"hmm_thresholds=[{hmm_profit_threshold}, {hmm_loss_threshold}], "
            f"premium_boost={premium_threshold_boost}"
        )

    def is_premium_session(
        self,
        session: TradingSession,
        utc_now: Optional[datetime] = None
    ) -> tuple[bool, Optional[PremiumSessionAssault]]:
        """
        Check if current session is a premium assault session.

        Premium sessions (Story 4.10):
        - London Open assault (08:00-09:00 London time)
        - London-NY Overlap assault (13:00-14:00 GMT)
        - NY Open assault (13:30-14:30 GMT)

        Args:
            session: Current trading session
            utc_now: Current UTC time (defaults to now)

        Returns:
            Tuple of (is_premium, premium_assault_type)
        """
        if utc_now is None:
            utc_now = datetime.now(timezone.utc)

        # Check each premium assault window
        for assault_type in PremiumSessionAssault:
            if self._is_in_assault_window(assault_type, utc_now):
                return True, assault_type

        return False, None

    def _is_in_assault_window(
        self,
        assault: PremiumSessionAssault,
        utc_now: datetime
    ) -> bool:
        """
        Check if UTC time falls within a premium assault window.

        Args:
            assault: Premium assault type
            utc_now: UTC time to check

        Returns:
            True if in assault window
        """
        from zoneinfo import ZoneInfo

        if assault == PremiumSessionAssault.LONDON_OPEN:
            # London Open assault: 07:00-10:00 GMT (07:00-09:59 London time)
            london_tz = ZoneInfo("Europe/London")
            london_time = utc_now.astimezone(london_tz).time()
            hour = london_time.hour
            # 07:00-09:59
            return 7 <= hour < 10

        elif assault == PremiumSessionAssault.LONDON_NY_OVERLAP:
            # London-NY Overlap assault: 13:00-16:00 GMT
            gmt_tz = ZoneInfo("GMT")
            gmt_time = utc_now.astimezone(gmt_tz).time()
            hour = gmt_time.hour
            # 13:00-15:59
            return 13 <= hour < 16

        elif assault == PremiumSessionAssault.NY_OPEN:
            # NY Open assault: 13:00-16:00 GMT (same as London-NY overlap per Phase 1 plan)
            gmt_tz = ZoneInfo("GMT")
            gmt_time = utc_now.astimezone(gmt_tz).time()
            hour = gmt_time.hour
            # 13:00-15:59
            return 13 <= hour < 16

        return False

    def compute_session_kelly_modifier(
        self,
        daily_pnl_pct: float,
        current_session: TradingSession,
        utc_now: Optional[datetime] = None
    ) -> SessionKellyState:
        """
        Compute the composite session Kelly modifier.

        This method evaluates all modifier components:
        1. House Money Mode (based on daily P&L %)
        2. Premium Session threshold lowering
        3. Reverse HMM (based on consecutive session losses)
        4. Premium boost status

        Args:
            daily_pnl_pct: Daily P&L as percentage (e.g., 0.08 = +8%)
            current_session: Current trading session
            utc_now: Current UTC time (defaults to now)

        Returns:
            SessionKellyState with all modifier values
        """
        if utc_now is None:
            utc_now = datetime.now(timezone.utc)

        # Update premium session detection
        self._is_premium_session, self._current_premium_assault = self.is_premium_session(
            current_session, utc_now
        )

        # Compute HMM multiplier
        self._compute_hmm_multiplier(daily_pnl_pct, utc_now)

        # Compute Reverse HMM multiplier (uses current loss counter state)
        self._compute_reverse_hmm_multiplier()

        # Compute composite session Kelly modifier
        # Chain: HMM x Reverse_HMM (premium boost applied within reverse)
        composite = (
            self._hmm_multiplier *
            self._reverse_hmm_multiplier
        )

        # Build state snapshot
        state = SessionKellyState(
            hmm_multiplier=self._hmm_multiplier,
            is_preservation_mode=self._is_preservation_mode,
            is_house_money_active=self._is_house_money_active,
            session_loss_counter=self._session_loss_counter,
            reverse_hmm_multiplier=self._reverse_hmm_multiplier,
            premium_boost_active=self._premium_boost_active,
            is_premium_session=self._is_premium_session,
            premium_assault=self._current_premium_assault,
            session_kelly_multiplier=composite,
            current_session=current_session.value,
            daily_pnl_pct=daily_pnl_pct,
            last_updated=utc_now,
        )

        logger.debug(
            f"SessionKelly modifier computed: "
            f"daily_pnl={daily_pnl_pct*100:.1f}%, "
            f"hmm={self._hmm_multiplier:.2f}, "
            f"reverse_hmm={self._reverse_hmm_multiplier:.2f}, "
            f"premium={self._premium_boost_active}, "
            f"composite={composite:.2f}"
        )

        return state

    def _in_london_ny_overlap(self, utc_now: datetime) -> bool:
        """
        Check if UTC time falls within London-NY overlap window (13:00-16:00 GMT).

        Args:
            utc_now: UTC time to check

        Returns:
            True if in London-NY overlap window
        """
        from zoneinfo import ZoneInfo
        gmt_tz = ZoneInfo("GMT")
        gmt_time = utc_now.astimezone(gmt_tz).time()
        hour = gmt_time.hour
        return 13 <= hour < 16

    def _compute_hmm_multiplier(
        self,
        daily_pnl_pct: float,
        utc_now: Optional[datetime] = None
    ) -> None:
        """
        Compute House Money Mode multiplier based on daily P&L.

        Continuous scaling formula (Phase 1 plan):
        - pnl_pct > 0.08: multiplier = min(1.0 + (pnl_pct / 0.10), 2.5) up to 2.5x
        - pnl_pct > 0.04 (and <= 0.08): multiplier = 1.25 (early house money)
        - pnl_pct < -0.10: multiplier = 0.5 (preservation)
        - Otherwise: multiplier = 1.0

        During London-NY overlap: if overlap_consecutive_losses >= 3, reset multiplier to 1.0.

        Args:
            daily_pnl_pct: Daily P&L as percentage
            utc_now: Current UTC time for overlap detection
        """
        if utc_now is None:
            utc_now = datetime.now(timezone.utc)
        # Check Preservation (loss) condition first - highest priority
        if daily_pnl_pct <= self.hmm_loss_threshold:
            self._hmm_multiplier = self.HMM_MULTIPLIER_PRESERVATION
            self._is_preservation_mode = True
            self._is_house_money_active = False
            logger.warning(f"Preservation Mode ACTIVE: {daily_pnl_pct*100:.1f}% -> {self._hmm_multiplier}x")
            return

        # During London-NY overlap: use lower 0.04 threshold with pnl_pct/0.08 scaling
        in_overlap = self._in_london_ny_overlap(utc_now)
        profit_threshold = 0.04 if in_overlap else self.hmm_profit_threshold
        scaling_denominator = 0.08 if in_overlap else 0.10

        # Check House Money (profit) conditions with continuous scaling
        if daily_pnl_pct > profit_threshold:
            # Continuous scaling: min(1.0 + (pnl_pct / scaling_denominator), 2.5)
            self._hmm_multiplier = min(1.0 + (daily_pnl_pct / scaling_denominator), 2.5)
            self._is_preservation_mode = False
            self._is_house_money_active = True
            logger.info(f"House Money Mode ACTIVE: +{daily_pnl_pct*100:.1f}% -> {self._hmm_multiplier:.2f}x")
        elif daily_pnl_pct > 0.04:
            # Early house money: 1.25x (between 4% and threshold)
            # During overlap, this branch is skipped due to threshold above
            self._hmm_multiplier = 1.25
            self._is_preservation_mode = False
            self._is_house_money_active = True
            logger.info(f"Early House Money: +{daily_pnl_pct*100:.1f}% -> {self._hmm_multiplier:.2f}x")
        else:
            # Baseline (no modifier)
            self._hmm_multiplier = self.HMM_MULTIPLIER_BASELINE
            self._is_preservation_mode = False
            self._is_house_money_active = False

        # During London-NY overlap: if overlap_consecutive_losses >= 3, reset multiplier to 1.0
        if in_overlap and self._overlap_consecutive_losses >= 3:
            self._hmm_multiplier = self.HMM_MULTIPLIER_BASELINE
            self._is_house_money_active = False
            self._premium_boost_active = False
            logger.warning(f"London-NY Overlap reverse HMM: {self._overlap_consecutive_losses} consecutive losses -> multiplier reset to 1.0")

    def _compute_reverse_hmm_multiplier(self) -> None:
        """
        Compute Reverse House Money multiplier based on session loss counter.

        Applies penalty multipliers based on consecutive losses:
        - 2 losses: 1.0x (remove premium boost, no penalty)
        - 4 losses: 0.70x (session under stress)
        - 6 losses: 0.50x (minimum viable sizing)
        """
        if self._session_loss_counter >= 6:
            self._reverse_hmm_multiplier = self.REVERSE_HMM_6_LOSSES
            self._premium_boost_active = False
        elif self._session_loss_counter >= 4:
            self._reverse_hmm_multiplier = self.REVERSE_HMM_4_LOSSES
            self._premium_boost_active = False
        elif self._session_loss_counter >= 2:
            self._reverse_hmm_multiplier = self.REVERSE_HMM_2_LOSSES
            self._premium_boost_active = False  # Premium boost removed
        else:
            self._reverse_hmm_multiplier = self.HMM_MULTIPLIER_BASELINE
            # Premium boost remains active only if we have 2 consecutive session wins
            if self._consecutive_session_wins >= self.WINS_TO_REENABLE_PREMIUM:
                self._premium_boost_active = True

    def on_trade_result(
        self,
        is_win: bool,
        utc_now: Optional[datetime] = None
    ) -> SessionKellyState:
        """
        Handle trade result for session Kelly modifier tracking.

        Args:
            is_win: True if trade was a winner, False if loser
            utc_now: Current UTC time (for overlap detection)

        Returns:
            Updated SessionKellyState
        """
        if utc_now is None:
            utc_now = datetime.now(timezone.utc)

        in_overlap = self._in_london_ny_overlap(utc_now)

        if is_win:
            # Win: reset loss counter, increment win streak
            self._session_loss_counter = 0
            self._consecutive_session_wins += 1
            if in_overlap:
                self._overlap_consecutive_losses = 0
            logger.debug(f"Win recorded: session_loss_counter=0, consecutive_wins={self._consecutive_session_wins}")
        else:
            # Loss: increment loss counter, reset win streak
            self._session_loss_counter += 1
            self._consecutive_session_wins = 0
            if in_overlap:
                self._overlap_consecutive_losses += 1
            logger.debug(f"Loss recorded: session_loss_counter={self._session_loss_counter}, consecutive_wins=0")

        # Recompute reverse HMM multiplier
        self._compute_reverse_hmm_multiplier()

        return self.get_current_state()

    def on_session_close(
        self,
        utc_now: Optional[datetime] = None
    ) -> None:
        """
        Reset all session-level modifiers on session boundary.

        Called when a session ends (e.g., London session ends at 12:00 GMT).

        Args:
            utc_now: Current UTC time (defaults to now)
        """
        if utc_now is None:
            utc_now = datetime.now(timezone.utc)

        # Reset all session-level state
        self._session_loss_counter = 0
        self._consecutive_session_wins = 0
        self._premium_boost_active = False
        self._reverse_hmm_multiplier = self.HMM_MULTIPLIER_BASELINE
        self._is_premium_session = False
        self._current_premium_assault = None
        self._overlap_consecutive_losses = 0

        # Note: HMM (daily P&L based) does NOT reset on session close
        # Only session-scoped modifiers reset

        logger.info(
            f"Session boundary reset at {utc_now.isoformat()}: "
            f"session_loss_counter=0, premium_boost=False"
        )

    def on_session_start(
        self,
        session: TradingSession,
        utc_now: Optional[datetime] = None
    ) -> None:
        """
        Initialize state for new session.

        Args:
            session: The new trading session
            utc_now: Current UTC time (defaults to now)
        """
        if utc_now is None:
            utc_now = datetime.now(timezone.utc)

        self._current_session = session
        self._session_start = utc_now

        # Session-level modifiers reset on session start (same as session close)
        self.on_session_close(utc_now)

        logger.info(f"Session started: {session.value} at {utc_now.isoformat()}")

    def get_current_state(self) -> SessionKellyState:
        """
        Get current session Kelly modifier state.

        Returns:
            Current SessionKellyState snapshot
        """
        return SessionKellyState(
            hmm_multiplier=self._hmm_multiplier,
            is_preservation_mode=self._is_preservation_mode,
            is_house_money_active=self._is_house_money_active,
            session_loss_counter=self._session_loss_counter,
            reverse_hmm_multiplier=self._reverse_hmm_multiplier,
            premium_boost_active=self._premium_boost_active,
            is_premium_session=self._is_premium_session,
            premium_assault=self._current_premium_assault,
            session_kelly_multiplier=self._hmm_multiplier * self._reverse_hmm_multiplier,
            current_session=self._current_session.value if self._current_session else "",
            daily_pnl_pct=0.0,  # P&L tracked externally
            last_updated=datetime.now(timezone.utc),
        )

    def get_modifier_chain_components(self) -> Dict[str, float]:
        """
        Get individual modifier chain components for display.

        Returns:
            Dictionary with named modifier components
        """
        return {
            "hmm_multiplier": self._hmm_multiplier,
            "reverse_hmm_multiplier": self._reverse_hmm_multiplier,
            "premium_boost_active": 1.2 if self._premium_boost_active else 1.0,  # Premium adds 20% boost
            "session_kelly_multiplier": self._hmm_multiplier * self._reverse_hmm_multiplier,
        }

    def get_multiplier_chain_display(self) -> str:
        """
        Get human-readable multiplier chain display string.

        Returns:
            Formatted string showing all active modifiers
        """
        components = []

        if self._is_house_money_active:
            components.append(f"HMM: {self._hmm_multiplier:.2f}x")
        elif self._is_preservation_mode:
            components.append(f"Preservation: {self._hmm_multiplier:.2f}x")

        if self._session_loss_counter >= 6:
            components.append(f"UnderStress: {self._reverse_hmm_multiplier:.2f}x")
        elif self._session_loss_counter >= 4:
            components.append(f"Stress: {self._reverse_hmm_multiplier:.2f}x")
        elif self._session_loss_counter >= 2:
            components.append("PremiumReset")

        if self._is_premium_session and self._premium_boost_active:
            assault_name = self._current_premium_assault.value if self._current_premium_assault else "premium"
            components.append(f"{assault_name.title()}: Active")

        return " | ".join(components) if components else "Baseline"

    @property
    def session_loss_counter(self) -> int:
        """Get current session loss counter value."""
        return self._session_loss_counter

    @property
    def is_premium_boost_active(self) -> bool:
        """Check if premium boost is currently active."""
        return self._premium_boost_active

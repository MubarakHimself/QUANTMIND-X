"""
CalendarGovernor - News Blackout & Calendar-Aware Trading Rules

A mixin that extends EnhancedGovernor to apply economic calendar rules:
- Lot scaling during high-impact news events
- Entry pauses during blackout windows
- Post-event reactivation with regime checks

Story: 4-1-calendargovernor-news-blackout-calendar-aware-trading-rules

Architecture: CalendarGovernor is a mixin that extends EnhancedGovernor.
DO NOT modify the base Governor class - extend EnhancedGovernor with CalendarGovernor mixin.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, TYPE_CHECKING

from src.router.enhanced_governor import EnhancedGovernor
from src.router.governor import RiskMandate

# Import calendar models
from src.risk.models.calendar import (
    NewsItem,
    CalendarRule,
    CalendarPhase,
    DEFAULT_BLACKOUT_MINUTES,
    DEFAULT_POST_EVENT_DELAY_MINUTES,
)

if TYPE_CHECKING:
    from src.router.sentinel import RegimeReport

logger = logging.getLogger(__name__)


class CalendarGovernor(EnhancedGovernor):
    """
    CalendarGovernor Mixin - Extends EnhancedGovernor with calendar-aware rules.

    Applies economic calendar rules to position sizing:
    - Pre-event: Lot scaling (e.g., 0.5x for 30 min before high-impact events)
    - During-event: Full pause (0.0x) for Tier 1 events
    - Post-event: Regime-check reactivation after delay

    Features:
    - Per-account calendar rule configuration
    - Blackout window evaluation
    - Lot scaling factors per calendar phase
    - Post-event reactivation with regime quality check
    - Audit trail logging for rule activations/resumptions

    Usage:
        governor = CalendarGovernor(account_id="FTMO-12345")
        governor.register_calendar_rule(calendar_rule)
        governor.add_calendar_event(news_item)
        mandate = governor.calculate_risk(regime_report, trade_proposal)
    """

    def __init__(
        self,
        account_id: Optional[str] = None,
        config: Optional[object] = None,
        settings: Optional[object] = None
    ):
        """
        Initialize CalendarGovernor with calendar-specific state.

        Args:
            account_id: Account identifier for calendar rules
            config: Optional EnhancedKellyConfig
            settings: Optional RiskSettings
        """
        # Call EnhancedGovernor __init__
        super().__init__(account_id=account_id, config=config, settings=settings)

        # Calendar-specific state
        self._calendar_rules: Dict[str, CalendarRule] = {}
        self._active_events: List[NewsItem] = []
        self._reactivation_timers: Dict[str, datetime] = {}

        logger.info(
            f"CalendarGovernor initialized for account {account_id}, "
            f"calendar rules: 0, active events: 0"
        )

    def register_calendar_rule(self, rule: CalendarRule) -> None:
        """
        Register a calendar rule for an account.

        Args:
            rule: CalendarRule configuration
        """
        self._calendar_rules[rule.account_id] = rule
        logger.info(
            f"Calendar rule registered: {rule.rule_id} for account {rule.account_id}"
        )

    def add_calendar_event(self, event: NewsItem) -> None:
        """
        Add a calendar event to the active events list.

        Args:
            event: NewsItem representing an economic event
        """
        # Avoid duplicates
        existing_ids = {e.event_id for e in self._active_events}
        if event.event_id not in existing_ids:
            self._active_events.append(event)
            logger.info(
                f"Calendar event added: {event.event_id} ({event.event_type}) "
                f"at {event.event_time.isoformat()}"
            )

    def remove_calendar_event(self, event_id: str) -> None:
        """
        Remove a calendar event from active events.

        Args:
            event_id: Event ID to remove
        """
        self._active_events = [
            e for e in self._active_events if e.event_id != event_id
        ]

    def clear_past_events(self, now: Optional[datetime] = None) -> int:
        """
        Clear all past events from active list.

        Args:
            now: Reference time for determining past events (defaults to current UTC time)

        Returns:
            Number of events removed
        """
        before_count = len(self._active_events)
        self._active_events = [
            e for e in self._active_events if not e.is_past(now=now)
        ]
        removed = before_count - len(self._active_events)
        if removed > 0:
            logger.info(f"Cleared {removed} past calendar events")
        return removed

    def is_within_blackout_window(
        self,
        account_id: str,
        check_time: datetime,
        rule: Optional[CalendarRule] = None
    ) -> bool:
        """
        Check if the given time is within a blackout window for the account.

        Args:
            account_id: Account identifier
            check_time: Time to check
            rule: CalendarRule (optional, will look up if not provided)

        Returns:
            True if within blackout window
        """
        if rule is None:
            rule = self._calendar_rules.get(account_id)

        if rule is None or not rule.blacklist_enabled:
            return False

        blackout_minutes = rule.blackout_minutes
        blackout_start = check_time - timedelta(minutes=blackout_minutes)

        for event in self._active_events:
            # Check if event is upcoming and within blackout window
            if event.event_time > check_time:  # Event is in the future
                if blackout_start <= event.event_time:
                    # Within blackout window
                    return True

        return False

    def evaluate_lot_scaling(
        self,
        account_id: str,
        check_time: datetime,
        event: Optional[NewsItem] = None,
        rule: Optional[CalendarRule] = None
    ) -> float:
        """
        Evaluate lot scaling factor based on current calendar state.

        Args:
            account_id: Account identifier
            check_time: Current time
            event: Active event (optional)
            rule: CalendarRule (optional)

        Returns:
            Lot scaling factor (0.0 to 1.0)
        """
        if rule is None:
            rule = self._calendar_rules.get(account_id)

        if rule is None:
            return 1.0  # No rule = normal operation

        if not rule.blacklist_enabled:
            return 1.0  # Blacklist disabled = normal operation

        # Determine current phase
        phase = self._determine_phase(check_time, event, rule)
        scaling = rule.get_lot_scaling(phase)

        logger.debug(
            f"Lot scaling for {account_id} at {check_time.isoformat()}: "
            f"phase={phase}, scaling={scaling}"
        )

        return scaling

    def evaluate_lot_scaling_normal(self, account_id: str, check_time: datetime) -> float:
        """
        Evaluate lot scaling when no events are active (normal operation).

        Args:
            account_id: Account identifier
            check_time: Current time

        Returns:
            Lot scaling factor (normally 1.0)
        """
        rule = self._calendar_rules.get(account_id)

        # Check if we should return to normal after post-event delay
        for event in self._active_events:
            if event.is_past():
                # Check if post-event delay has passed
                if self.is_post_event_reactivation_time(account_id, check_time, event, rule):
                    return 1.0

        # If no active events affecting us, return normal scaling
        if not self.is_within_blackout_window(account_id, check_time, rule):
            return 1.0

        return rule.get_lot_scaling(CalendarPhase.PRE_EVENT) if rule else 1.0

    def _determine_phase(
        self,
        check_time: datetime,
        event: Optional[NewsItem],
        rule: Optional[CalendarRule]
    ) -> CalendarPhase:
        """
        Determine the current calendar phase for an event.

        Args:
            check_time: Current time
            event: Active event
            rule: Calendar rule

        Returns:
            Current CalendarPhase
        """
        if event is None:
            return CalendarPhase.NORMAL

        time_until_event = event.event_time - check_time

        # During event (15 min before to 15 min after) — "all pause" window
        if timedelta(minutes=-15) <= time_until_event <= timedelta(minutes=15):
            return CalendarPhase.DURING_EVENT

        # Post-event regime check phase
        if time_until_event < timedelta(minutes=0):
            if rule and rule.regime_check_enabled:
                post_event_delay = timedelta(minutes=rule.post_event_delay_minutes)
                if time_until_event > -post_event_delay:
                    return CalendarPhase.POST_EVENT_REGIME_CHECK

        # Pre-event phase (within blackout window)
        if rule and time_until_event > timedelta(minutes=0):
            blackout = timedelta(minutes=rule.blackout_minutes)
            if time_until_event <= blackout:
                return CalendarPhase.PRE_EVENT

        return CalendarPhase.NORMAL

    def is_post_event_reactivation_time(
        self,
        account_id: str,
        check_time: datetime,
        event: NewsItem,
        rule: Optional[CalendarRule] = None
    ) -> bool:
        """
        Check if it's time to reactivate after an event.

        For regime_check_enabled: Returns True when in regime-check window
        (after event time but before post_event_delay) OR after post_event_delay.

        Args:
            account_id: Account identifier
            check_time: Current time
            event: Past event
            rule: Calendar rule

        Returns:
            True if reactivation is allowed
        """
        if rule is None:
            rule = self._calendar_rules.get(account_id)

        if rule is None:
            return True  # No rule = allow reactivation

        time_since_event = check_time - event.event_time
        post_event_delay = timedelta(minutes=rule.post_event_delay_minutes)

        # Case 1: Full reactivation - after post_event_delay
        if time_since_event >= post_event_delay:
            return True

        # Case 2: Regime check phase - with regime_check_enabled
        # Returns True for regime check window (after event, before full delay)
        if rule.regime_check_enabled and time_since_event > timedelta(minutes=0):
            return True

        return False

    def check_regime_for_reactivation(
        self,
        regime_report: 'RegimeReport',
        min_regime_quality: float = 0.6
    ) -> bool:
        """
        Check if regime is suitable for reactivation after event.

        Args:
            regime_report: Current regime report
            min_regime_quality: Minimum regime quality threshold

        Returns:
            True if regime is stable enough for reactivation
        """
        if regime_report is None:
            return True  # No report = allow reactivation

        # Check regime quality
        if regime_report.regime_quality < min_regime_quality:
            logger.warning(
                f"Regime check failed for reactivation: "
                f"quality={regime_report.regime_quality:.2f} < {min_regime_quality}"
            )
            return False

        # Check chaos score
        if regime_report.chaos_score > 0.5:
            logger.warning(
                f"Regime check failed for reactivation: "
                f"chaos={regime_report.chaos_score:.2f} > 0.5"
            )
            return False

        return True

    def calculate_risk(
        self,
        regime_report: 'RegimeReport',
        trade_proposal: dict,
        account_balance: Optional[float] = None,
        broker_id: Optional[str] = None,
        account_id: Optional[str] = None,
        mode: str = "live",
        **kwargs
    ) -> RiskMandate:
        """
        Calculate risk with calendar-aware position sizing.

        Extends EnhancedGovernor's calculate_risk to apply calendar rules:
        - Scales position sizes based on calendar phase
        - Applies pauses during high-impact events
        - Logs rule activations to audit trail

        Args:
            regime_report: Current market regime from Sentinel
            trade_proposal: Trade proposal dict with symbol, balance, etc.
            account_balance: Account balance (optional)
            broker_id: Broker identifier (optional)
            account_id: Account identifier (optional)
            mode: Trading mode ('demo' or 'live')
            **kwargs: Additional parameters

        Returns:
            RiskMandate with calendar-aware position sizing
        """
        # Get account_id from trade_proposal or self.account_id
        account_id = account_id or trade_proposal.get('account_id', self.account_id)

        # Get current time first so it can be passed to clear_past_events
        check_time = self._get_current_utc_time()

        # Clear any past events using the same reference time (enables deterministic testing)
        self.clear_past_events(now=check_time)

        # Get calendar rule for this account
        rule = self._calendar_rules.get(account_id) if account_id else None

        # Determine lot scaling based on calendar
        calendar_scaling = 1.0

        if rule and rule.blacklist_enabled and self._active_events:
            # Find active event for this symbol
            symbol = trade_proposal.get('symbol', '')
            affected_currencies = self._get_currencies_from_symbol(symbol)

            # Find relevant active event
            relevant_event = None
            for event in self._active_events:
                if any(curr in event.currencies for curr in affected_currencies):
                    relevant_event = event
                    break

            if relevant_event:
                calendar_scaling = self.evaluate_lot_scaling(
                    account_id, check_time, relevant_event, rule
                )

                # Log rule activation if scaling changed
                if calendar_scaling != 1.0:
                    phase = self._determine_phase(check_time, relevant_event, rule)
                    self._log_rule_activation(account_id, phase.value, calendar_scaling)

                    # Log resumption when entering post-event regime-check phase (AC2)
                    if phase == CalendarPhase.POST_EVENT_REGIME_CHECK:
                        regime_name = str(getattr(regime_report, 'regime', 'UNKNOWN'))
                        regime_quality = float(getattr(regime_report, 'regime_quality', 0.0))
                        self._log_resumption(account_id or 'unknown', regime_name, regime_quality)
            else:
                # No relevant events - check normal scaling
                if account_id:
                    calendar_scaling = self.evaluate_lot_scaling_normal(account_id, check_time)

        # Call EnhancedGovernor's calculate_risk for base calculations
        base_mandate = super().calculate_risk(
            regime_report=regime_report,
            trade_proposal=trade_proposal,
            account_balance=account_balance,
            broker_id=broker_id,
            account_id=account_id,
            mode=mode,
            **kwargs
        )

        # Apply calendar scaling to the base mandate
        if calendar_scaling != 1.0:
            base_mandate.allocation_scalar *= calendar_scaling
            base_mandate.position_size *= calendar_scaling
            base_mandate.risk_amount *= calendar_scaling

            # Add calendar adjustment to notes
            calendar_note = f"CalendarGovernor: {calendar_scaling}x scaling applied. "
            if base_mandate.notes:
                base_mandate.notes = calendar_note + base_mandate.notes
            else:
                base_mandate.notes = calendar_note

            logger.info(
                f"CalendarGovernor: Applied {calendar_scaling}x scaling for account {account_id}"
            )

        return base_mandate

    def _get_current_utc_time(self) -> datetime:
        """Get current UTC time. Can be overridden for testing."""
        return datetime.now(timezone.utc)

    def _get_currencies_from_symbol(self, symbol: str) -> List[str]:
        """Extract currencies from a symbol like EURUSD."""
        if len(symbol) >= 6:
            return [symbol[:3], symbol[3:6]]
        return []

    def _log_rule_activation(
        self,
        account_id: str,
        phase: str,
        scaling: float
    ) -> None:
        """Log calendar rule activation to audit trail."""
        logger.info(
            f"[AUDIT] Calendar rule activated: account={account_id}, "
            f"phase={phase}, scaling={scaling}, time={datetime.now(timezone.utc).isoformat()}"
        )

    def _log_resumption(
        self,
        account_id: str,
        regime: str,
        regime_quality: float
    ) -> None:
        """Log calendar resumption to audit trail."""
        logger.info(
            f"[AUDIT] Calendar resumption: account={account_id}, "
            f"regime={regime}, quality={regime_quality:.2f}, "
            f"time={datetime.now(timezone.utc).isoformat()}"
        )


# Alias for convenience
__all__ = ["CalendarGovernor"]
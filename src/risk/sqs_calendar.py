"""
SQS Calendar Integration - CalendarGovernor News Window Override

Integrates SQS engine with CalendarGovernor to raise thresholds during news events.

Story: 4-7-spread-quality-score-sqs-system
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, Literal
from enum import Enum
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class NewsImpact(str, Enum):
    """News event impact levels."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class NewsWindowState(BaseModel):
    """State of news window for a symbol."""
    active: bool = False
    impact: Optional[NewsImpact] = None
    minutes_to_event: int = 0


class SQSCalendarIntegration:
    """
    Integration between SQS Engine and CalendarGovernor.

    Raises SQS entry thresholds during news blackout windows:
    - High impact (T-15/T+15): raise threshold to ORB level (>0.80)
    - Medium impact (T-2/T+1): apply 0.10 threshold bump
    """

    # Threshold adjustments during news
    HIGH_IMPACT_THRESHOLD_BUMP = 0.10  # Raises to ORB level (0.80)
    MEDIUM_IMPACT_THRESHOLD_BUMP = 0.10  # Adds 0.10 to threshold

    # News window thresholds (minutes before/after event)
    HIGH_IMPACT_WINDOW_MINUTES = 15
    MEDIUM_IMPACT_WINDOW_MINUTES = 2

    def __init__(self, calendar_governor=None):
        """
        Initialize SQS Calendar integration.

        Args:
            calendar_governor: CalendarGovernor instance for news state queries
        """
        self._governor = calendar_governor
        logger.info("SQS Calendar Integration initialized")

    def get_news_window_state(self, symbol: str) -> NewsWindowState:
        """
        Get current news window state for symbol.

        Queries CalendarGovernor for active news events affecting the symbol.

        Returns:
            NewsWindowState with active=True if within news window
        """
        if self._governor is None:
            return NewsWindowState(active=False)

        try:
            # Try to get news state from CalendarGovernor
            news_state = self._governor.get_news_window_state(symbol)

            if news_state is None:
                return NewsWindowState(active=False)

            return news_state

        except AttributeError:
            # CalendarGovernor doesn't have get_news_window_state - try alternative
            logger.debug("CalendarGovernor doesn't have get_news_window_state, checking alternative method")
            return self._get_news_state_alternative(symbol)

        except Exception as e:
            logger.warning(f"Error getting news window state for {symbol}: {e}")
            return NewsWindowState(active=False)

    def _get_news_state_alternative(self, symbol: str) -> NewsWindowState:
        """Alternative method to get news state when CalendarGovernor not fully available."""
        try:
            # Try to access via risk_endpoints calendar events
            from src.api.risk_endpoints import _calendar_events

            now = datetime.now(timezone.utc)

            # Check for active or approaching high-impact events
            for event in _calendar_events:
                if event.impact != NewsImpact.HIGH:
                    continue

                # Check if within window
                minutes_to_event = int((event.event_time - now).total_seconds() / 60)

                if abs(minutes_to_event) <= self.HIGH_IMPACT_WINDOW_MINUTES:
                    return NewsWindowState(
                        active=True,
                        impact=NewsImpact.HIGH,
                        minutes_to_event=minutes_to_event
                    )

            return NewsWindowState(active=False)

        except Exception as e:
            logger.warning(f"Error in alternative news state method: {e}")
            return NewsWindowState(active=False)

    def get_threshold_override(self, symbol: str) -> Optional[float]:
        """
        Get threshold override modifier based on news window state.

        Args:
            symbol: Trading symbol

        Returns:
            Threshold modifier (e.g., +0.10) if news override active, None otherwise
        """
        state = self.get_news_window_state(symbol)

        if not state.active:
            return None

        if state.impact == NewsImpact.HIGH:
            # High impact: raise to ORB level regardless of strategy
            logger.info(
                f"SQS Calendar: High impact news for {symbol}, "
                f"T-{state.minutes_to_event} minutes, applying +{self.HIGH_IMPACT_THRESHOLD_BUMP:.2f} bump"
            )
            return self.HIGH_IMPACT_THRESHOLD_BUMP

        elif state.impact == NewsImpact.MEDIUM:
            # Medium impact: apply 0.10 bump
            logger.info(
                f"SQS Calendar: Medium impact news for {symbol}, "
                f"T-{state.minutes_to_event} minutes, applying +{self.MEDIUM_IMPACT_THRESHOLD_BUMP:.2f} bump"
            )
            return self.MEDIUM_IMPACT_THRESHOLD_BUMP

        return None

    def is_within_high_impact_window(self, symbol: str) -> bool:
        """Check if currently within high-impact news window for symbol."""
        state = self.get_news_window_state(symbol)
        return state.active and state.impact == NewsImpact.HIGH

    def is_within_medium_impact_window(self, symbol: str) -> bool:
        """Check if currently within medium-impact news window for symbol."""
        state = self.get_news_window_state(symbol)
        return state.active and state.impact == NewsImpact.MEDIUM


def create_calendar_integration() -> SQSCalendarIntegration:
    """
    Factory function to create SQS calendar integration.

    Attempts to get CalendarGovernor instance for news state queries.
    """
    governor = None

    try:
        # Try to get CalendarGovernor from router
        from src.router.governor import Governor
        governor = Governor()
        logger.debug("SQS Calendar: Using Governor for news state")
    except ImportError:
        logger.warning("Could not import Governor for calendar integration")
    except Exception as e:
        logger.warning(f"Could not initialize Governor for calendar integration: {e}")

    return SQSCalendarIntegration(calendar_governor=governor)

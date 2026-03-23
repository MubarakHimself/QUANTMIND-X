"""
News Event Integration for Fast-Track Template Matching

Integrates with Story 6.3 news system to trigger template matching
on HIGH-impact news events.
"""
import logging
from typing import Dict, Any, List, Optional

from src.mql5.templates.matcher import get_template_matcher
from src.mql5.templates.storage import get_template_storage

logger = logging.getLogger(__name__)


class FastTrackEventListener:
    """
    Listens for HIGH-impact news events and triggers template matching.

    Integration points:
    - WebSocket broadcast from Story 6.3 news system
    - POST /api/news/alert endpoint callbacks
    """

    def __init__(self):
        self.matcher = get_template_matcher()
        self.storage = get_template_storage()

    def on_high_impact_news(
        self,
        event_type: str,
        headline: str,
        affected_symbols: List[str],
        impact_tier: str = "HIGH",
    ) -> Dict[str, Any]:
        """
        Handle a HIGH-impact news event.

        Called when GeopoliticalSubAgent classifies an event as HIGH impact
        with action_type FAST_TRACK or ALERT.

        Args:
            event_type: Type of news event
            headline: News headline
            affected_symbols: List of affected symbols
            impact_tier: Impact tier (HIGH/MEDIUM/LOW)

        Returns:
            Fast-track recommendation with matched templates
        """
        logger.info(
            f"High-impact news detected: {event_type} - {headline[:50]}..."
        )

        # Get top matching templates
        matches = self.matcher.get_top_matches(
            event_type=event_type,
            affected_symbols=affected_symbols,
            impact_tier=impact_tier,
            min_confidence=0.3,
            limit=3,
        )

        if not matches:
            logger.info("No matching templates found for event")
            return {
                "event_type": event_type,
                "headline": headline,
                "has_fast_track": False,
                "message": "No matching templates available",
            }

        # Get top match
        top_match = matches[0]
        template = top_match.template

        # Build Copilot message
        message = self._build_copilot_message(top_match, affected_symbols)

        logger.info(
            f"Fast-track available: {template.name} "
            f"(confidence: {top_match.confidence_score:.1%})"
        )

        return {
            "event_type": event_type,
            "headline": headline,
            "has_fast_track": True,
            "template_name": template.name,
            "confidence_score": top_match.confidence_score,
            "estimated_deployment_time": template.avg_deployment_time,
            "symbol": affected_symbols[0] if affected_symbols else None,
            "copilot_message": message,
            "all_matches": [m.to_dict() for m in matches],
        }

    def _build_copilot_message(
        self,
        match_result: Any,
        affected_symbols: List[str],
    ) -> str:
        """Build the Copilot message for fast-track suggestion."""
        template = match_result.template
        confidence = match_result.confidence_score
        deploy_time = template.avg_deployment_time
        symbol = affected_symbols[0] if affected_symbols else "pair"

        return (
            f"Fast-track available — {symbol} {template.name} template matches. "
            f"Deploy? [{deploy_time} min]"
        )


# Module-level listener instance
_listener: Optional[FastTrackEventListener] = None


def get_fast_track_listener() -> FastTrackEventListener:
    """Get the global fast-track event listener."""
    global _listener
    if _listener is None:
        _listener = FastTrackEventListener()
    return _listener


def check_and_suggest_fast_track(
    event_type: str,
    affected_symbols: List[str],
    impact_tier: str = "HIGH",
) -> Optional[str]:
    """
    Convenience function to check for fast-track suggestions.

    Returns Copilot suggestion message if fast-track is available,
    None otherwise.
    """
    if impact_tier != "HIGH":
        return None

    listener = get_fast_track_listener()

    # Create a dummy headline for matching
    result = listener.on_high_impact_news(
        event_type=event_type,
        headline="",
        affected_symbols=affected_symbols,
        impact_tier=impact_tier,
    )

    if result.get("has_fast_track"):
        return result.get("copilot_message")

    return None
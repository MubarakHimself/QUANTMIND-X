"""
Template Matching Service

Scores and ranks strategy templates against news events.
"""
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from src.mql5.templates.schema import (
    StrategyTemplate,
    TemplateMatchResult,
    EventType,
)
from src.mql5.templates.storage import get_template_storage

logger = logging.getLogger(__name__)


class TemplateMatcher:
    """
    Matches strategy templates against news events.

    Scoring algorithm:
    - Event type match: 40% weight
    - Symbol match: 30% weight
    - Risk profile alignment: 20% weight
    - Deployment time: 10% weight
    """

    # Scoring weights
    WEIGHTS = {
        "event_match": 0.40,
        "symbol_match": 0.30,
        "risk_alignment": 0.20,
        "deployment_time": 0.10,
    }

    def __init__(self):
        self.storage = get_template_storage()

    def match_event(
        self,
        event_type: str,
        affected_symbols: List[str],
        impact_tier: str = "HIGH",
    ) -> List[TemplateMatchResult]:
        """
        Match templates against a news event.

        Args:
            event_type: Type of news event (from GeopoliticalSubAgent)
            affected_symbols: List of affected trading symbols
            impact_tier: HIGH, MEDIUM, or LOW

        Returns:
            List of matched templates ranked by confidence score
        """
        logger.info(
            f"Matching templates for event: {event_type}, "
            f"symbols: {affected_symbols}, impact: {impact_tier}"
        )

        # Get all active templates
        templates = self.storage.get_active()

        if not templates:
            logger.warning("No active templates found")
            return []

        results = []

        for template in templates:
            match_result = self._score_template(
                template=template,
                event_type=event_type,
                affected_symbols=affected_symbols,
                impact_tier=impact_tier,
            )
            results.append(match_result)

        # Sort by confidence score (descending)
        results.sort(key=lambda r: r.confidence_score, reverse=True)

        logger.info(f"Found {len(results)} matching templates")
        return results

    def _score_template(
        self,
        template: StrategyTemplate,
        event_type: str,
        affected_symbols: List[str],
        impact_tier: str,
    ) -> TemplateMatchResult:
        """Score a single template against the event."""

        match_factors = {}

        # 1. Event type match (40%)
        event_score = self._calculate_event_match(template, event_type)
        match_factors["event_match"] = event_score

        # 2. Symbol match (30%)
        symbol_score = self._calculate_symbol_match(template, affected_symbols)
        match_factors["symbol_match"] = symbol_score

        # 3. Risk alignment (20%)
        risk_score = self._calculate_risk_alignment(template, impact_tier)
        match_factors["risk_alignment"] = risk_score

        # 4. Deployment time (10%)
        time_score = self._calculate_deployment_time_score(template)
        match_factors["deployment_time"] = time_score

        # Calculate weighted total
        total_score = (
            event_score * self.WEIGHTS["event_match"]
            + symbol_score * self.WEIGHTS["symbol_match"]
            + risk_score * self.WEIGHTS["risk_alignment"]
            + time_score * self.WEIGHTS["deployment_time"]
        )

        return TemplateMatchResult(
            template=template,
            confidence_score=round(total_score, 3),
            match_factors={k: round(v, 3) for k, v in match_factors.items()},
            estimated_deployment_time=template.avg_deployment_time,
        )

    def _calculate_event_match(
        self,
        template: StrategyTemplate,
        event_type: str,
    ) -> float:
        """Calculate event type match score."""
        if not template.applicable_events:
            return 0.0

        # Direct match
        if event_type in template.applicable_events:
            return 1.0

        # Partial match - check if event category is covered
        # Map event types to broader categories
        event_category_map = {
            EventType.HIGH_IMPACT_NEWS.value: [
                EventType.HIGH_IMPACT_NEWS.value,
                EventType.MARKET_SHOCK.value,
            ],
            EventType.CENTRAL_BANK.value: [EventType.CENTRAL_BANK.value],
            EventType.GEOPOLITICAL.value: [
                EventType.GEOPOLITICAL.value,
                EventType.MARKET_SHOCK.value,
            ],
            EventType.ECONOMIC_DATA.value: [EventType.ECONOMIC_DATA.value],
            EventType.MARKET_SHOCK.value: [EventType.MARKET_SHOCK.value],
        }

        covered_events = event_category_map.get(event_type, [event_type])
        matched = set(covered_events) & set(template.applicable_events)

        if matched:
            return 0.7  # Partial match

        return 0.0

    def _calculate_symbol_match(
        self,
        template: StrategyTemplate,
        affected_symbols: List[str],
    ) -> float:
        """Calculate symbol match score."""
        if not template.applicable_symbols or not affected_symbols:
            return 0.0

        matched = set(affected_symbols) & set(template.applicable_symbols)

        if not matched:
            return 0.0

        # Score based on proportion of symbols covered
        return len(matched) / len(affected_symbols)

    def _calculate_risk_alignment(
        self,
        template: StrategyTemplate,
        impact_tier: str,
    ) -> float:
        """Calculate risk profile alignment score."""
        # HIGH impact events should use conservative templates
        if impact_tier == "HIGH":
            if template.risk_profile.value == "conservative":
                return 1.0
            elif template.risk_profile.value == "moderate":
                return 0.7
            else:
                return 0.3
        elif impact_tier == "MEDIUM":
            if template.risk_profile.value in ["conservative", "moderate"]:
                return 0.9
            else:
                return 0.5
        else:  # LOW
            return 0.8  # Any risk profile works for low impact

    def _calculate_deployment_time_score(
        self,
        template: StrategyTemplate,
    ) -> float:
        """Calculate deployment time score."""
        # Fast deployment is better - score decreases with time
        # Target is 11-15 minutes
        deployment_time = template.avg_deployment_time

        if deployment_time <= 11:
            return 1.0
        elif deployment_time <= 13:
            return 0.9
        elif deployment_time <= 15:
            return 0.7
        else:
            return 0.5

    def get_top_matches(
        self,
        event_type: str,
        affected_symbols: List[str],
        impact_tier: str = "HIGH",
        min_confidence: float = 0.3,
        limit: int = 5,
    ) -> List[TemplateMatchResult]:
        """
        Get top matching templates above a confidence threshold.

        Args:
            event_type: Type of news event
            affected_symbols: List of affected symbols
            impact_tier: HIGH, MEDIUM, or LOW
            min_confidence: Minimum confidence threshold
            limit: Maximum number of results

        Returns:
            List of top matching templates
        """
        all_matches = self.match_event(event_type, affected_symbols, impact_tier)

        # Filter by confidence threshold
        filtered = [m for m in all_matches if m.confidence_score >= min_confidence]

        # Return top N
        return filtered[:limit]


# Module-level matcher instance
_matcher: Optional[TemplateMatcher] = None


def get_template_matcher() -> TemplateMatcher:
    """Get the global template matcher instance."""
    global _matcher
    if _matcher is None:
        _matcher = TemplateMatcher()
    return _matcher
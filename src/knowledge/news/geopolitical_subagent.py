"""
GeopoliticalSubAgent — Haiku-tier news classifier.

Accepts a NewsItem headline + summary and returns a classification dict with:
  - impact_tier: HIGH | MEDIUM | LOW
  - affected_symbols: List[str]
  - event_type: str
  - action_type: FAST_TRACK | ALERT | MONITOR

Uses claude-3-haiku-20240307 for sub-30-second latency per article.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict

import anthropic

from src.database.models.news_items import NewsItem

logger = logging.getLogger(__name__)

# Model selection: Haiku-tier per architecture §13.3
_MODEL = "claude-3-haiku-20240307"
_MAX_TOKENS = 400

# Classification prompt — instructs Haiku to return JSON only
CLASSIFICATION_PROMPT = """You are a geopolitical and macroeconomic news classifier for a forex trading system.

Classify the following news article and respond ONLY with valid JSON, no other text:

Headline: {headline}
Summary: {summary}

Return JSON with exactly these fields:
{{
  "impact_tier": "HIGH" or "MEDIUM" or "LOW",
  "affected_symbols": ["EURUSD", "GBPUSD"],
  "event_type": "central_bank" or "geopolitical" or "economic_data" or "market_shock" or "other",
  "action_type": "FAST_TRACK" or "ALERT" or "MONITOR"
}}

Use:
- HIGH + FAST_TRACK: Unexpected central bank actions, geopolitical shocks, major market disruptions
- HIGH + ALERT: Significant economic data surprises, scheduled high-impact events
- MEDIUM + MONITOR: Regular economic data, earnings, moderate news
- LOW + MONITOR: Background news, low market relevance"""

# Fallback when classification fails
_FALLBACK_CLASSIFICATION: Dict[str, Any] = {
    "impact_tier": "LOW",
    "affected_symbols": [],
    "event_type": "unknown",
    "action_type": "MONITOR",
}


class GeopoliticalSubAgent:
    """
    Lightweight Haiku-tier classifier for live news items.

    This is NOT a full department sub-agent — it does not use DepartmentMailService.
    It is called directly from NewsFeedPoller for each new article.
    """

    def __init__(self):
        self._client = anthropic.Anthropic()

    def classify(self, item: NewsItem) -> Dict[str, Any]:
        """
        Classify a news item and return structured impact data.

        Updates the item's severity, action_type, and classified_at_utc fields
        in-place so the caller can persist the update.

        Args:
            item: NewsItem ORM instance (already persisted or fresh).

        Returns:
            Classification dict: {impact_tier, affected_symbols, event_type, action_type}
        """
        headline = item.headline or ""
        summary = item.summary or ""

        prompt = CLASSIFICATION_PROMPT.format(
            headline=headline,
            summary=summary,
        )

        try:
            response = self._client.messages.create(
                model=_MODEL,
                max_tokens=_MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )
            raw_text = response.content[0].text.strip()
            logger.debug("GeopoliticalSubAgent raw response: %s", raw_text)

            classification = json.loads(raw_text)

            # Validate required fields are present
            impact_tier = classification.get("impact_tier", "LOW").upper()
            affected_symbols = classification.get("affected_symbols") or []
            event_type = classification.get("event_type", "other")
            action_type = classification.get("action_type", "MONITOR").upper()

            # Normalise to known values
            if impact_tier not in ("HIGH", "MEDIUM", "LOW"):
                impact_tier = "LOW"
            if action_type not in ("FAST_TRACK", "ALERT", "MONITOR"):
                action_type = "MONITOR"

            result: Dict[str, Any] = {
                "impact_tier": impact_tier,
                "affected_symbols": [str(s) for s in affected_symbols],
                "event_type": event_type,
                "action_type": action_type,
            }

        except (json.JSONDecodeError, KeyError, IndexError, ValueError) as exc:
            logger.error(
                "GeopoliticalSubAgent JSON parse failure for item_id=%s: %s",
                item.item_id, exc,
            )
            result = _FALLBACK_CLASSIFICATION.copy()
        except Exception as exc:
            logger.error(
                "GeopoliticalSubAgent LLM failure for item_id=%s: %s",
                item.item_id, exc,
            )
            result = _FALLBACK_CLASSIFICATION.copy()

        # Update the ORM item in-place
        item.severity = result["impact_tier"]
        item.action_type = result["action_type"]
        item.classified_at_utc = datetime.now(timezone.utc)

        return result

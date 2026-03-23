"""
FinnhubProvider — NewsProvider implementation backed by the Finnhub SDK.

Fetches general and forex news articles and maps them to NewsItem ORM objects.
Authenticates via FINNHUB_API_KEY environment variable.
"""

import os
import logging
import asyncio
from datetime import datetime, timezone
from typing import List

from src.knowledge.news.provider import NewsProvider
from src.database.models.news_items import NewsItem

logger = logging.getLogger(__name__)

# Categories to fetch — 'general' for macro/geopolitical, 'forex' for FX-specific
_CATEGORIES = ("general", "forex")

# Import finnhub at module level — tests can patch 'src.knowledge.news.finnhub_provider.finnhub'
try:
    import finnhub  # type: ignore[import]
except ImportError:
    finnhub = None  # type: ignore[assignment]


class FinnhubProvider(NewsProvider):
    """
    Concrete NewsProvider using the finnhub-python SDK.

    Fetches from `general_news(category=...)` for each category.
    Raises RuntimeError if FINNHUB_API_KEY is not set (suitable for both
    background service and HTTP contexts).  HTTP endpoints that call this
    should catch RuntimeError and re-raise as HTTPException(503).
    """

    def __init__(self):
        api_key = os.environ.get("FINNHUB_API_KEY")
        if not api_key:
            raise RuntimeError("FINNHUB_API_KEY not configured")
        self._api_key = api_key

    def _make_client(self):
        """Create a Finnhub client using the module-level finnhub import."""
        if finnhub is None:
            raise RuntimeError(
                "finnhub-python package is required. Install with: pip install finnhub-python>=2.4.0"
            )
        return finnhub.Client(api_key=self._api_key)

    def _fetch_category_sync(self, category: str) -> list:
        """Synchronous Finnhub SDK call — must be wrapped in run_in_executor."""
        client = self._make_client()
        return client.general_news(category=category, min_id=0) or []

    async def fetch_latest(self, since_utc: datetime) -> List[NewsItem]:
        """
        Fetch news articles published since `since_utc` from Finnhub.

        Queries both 'general' and 'forex' categories.
        Deduplication by item_id is handled by the caller (poller), but
        we also deduplicate within this batch to avoid returning duplicates
        across categories.

        Args:
            since_utc: Only return articles published after this timestamp.

        Returns:
            List of (unsaved) NewsItem instances.
        """
        loop = asyncio.get_running_loop()

        all_raw: list = []
        for category in _CATEGORIES:
            try:
                raw_items = await loop.run_in_executor(
                    None, self._fetch_category_sync, category
                )
                all_raw.extend(raw_items)
                logger.debug(
                    "Finnhub category=%s returned %d items", category, len(raw_items)
                )
            except HTTPException:
                raise
            except Exception as exc:
                logger.error("FinnhubProvider fetch error (category=%s): %s", category, exc)
                raise

        # Map raw items to NewsItem objects, filtering by since_utc
        seen_ids: set = set()
        news_items: List[NewsItem] = []

        for raw in all_raw:
            item_id = str(raw.get("id", ""))
            if not item_id or item_id in seen_ids:
                continue
            seen_ids.add(item_id)

            # Parse published_utc from UNIX timestamp
            unix_ts = raw.get("datetime", 0)
            try:
                published_utc = datetime.utcfromtimestamp(unix_ts).replace(
                    tzinfo=timezone.utc
                )
            except (OSError, ValueError, OverflowError):
                logger.warning("Could not parse datetime for item_id=%s, skipping", item_id)
                continue

            # Filter by since_utc
            if published_utc <= since_utc:
                continue

            # Map related field (comma-separated string) to list
            related_str = raw.get("related", "") or ""
            related_instruments = [
                s.strip() for s in related_str.split(",") if s.strip()
            ]

            news_item = NewsItem(
                item_id=item_id,
                headline=raw.get("headline", ""),
                summary=raw.get("summary") or None,
                source=raw.get("source") or None,
                published_utc=published_utc,
                url=raw.get("url") or None,
                related_instruments=related_instruments if related_instruments else None,
                severity=None,
                action_type=None,
                classified_at_utc=None,
            )
            news_items.append(news_item)

        logger.debug(
            "FinnhubProvider: %d new items after since_utc filter", len(news_items)
        )
        return news_items

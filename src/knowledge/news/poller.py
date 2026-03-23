"""
NewsFeedPoller — APScheduler-based background service.

Polls Finnhub every 60 seconds, persists new NewsItem rows to SQLite,
classifies each via GeopoliticalSubAgent, and broadcasts HIGH+ALERT events
via WebSocket.

On FAST_TRACK events, logs a WARNING only — actual Fast-Track workflow is
Epic 8 scope and is NOT implemented here.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from src.knowledge.news.finnhub_provider import FinnhubProvider
from src.knowledge.news.geopolitical_subagent import GeopoliticalSubAgent
from src.database.models.news_items import NewsItem

logger = logging.getLogger(__name__)

# Poll every 60 seconds — not configurable per story scope (FR50 / Journey 46)
POLL_INTERVAL_SECONDS = 60

# Exponential backoff config (max 3 attempts, delays: 10s / 20s / 40s)
_MAX_RETRY_ATTEMPTS = 3
_BASE_RETRY_DELAY_SECONDS = 10


class NewsFeedPoller:
    """
    Background polling service for live news feed.

    Lifecycle:
        poller = NewsFeedPoller()
        poller.start()     # registers APScheduler job, starts scheduler
        poller.stop()      # shuts down scheduler cleanly
    """

    def __init__(self):
        self._scheduler = AsyncIOScheduler()
        self._provider = FinnhubProvider()
        self._sub_agent = GeopoliticalSubAgent()
        # Start from 5 minutes ago to catch articles published just before startup
        self._last_polled_utc: datetime = datetime.now(timezone.utc) - timedelta(minutes=5)

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Register the poll job and start the APScheduler."""
        self._scheduler.add_job(
            self._poll_cycle,
            trigger=IntervalTrigger(seconds=POLL_INTERVAL_SECONDS),
            id="news_feed_poll",
            replace_existing=True,
        )
        self._scheduler.start()
        logger.info("NewsFeedPoller started (interval=%ds)", POLL_INTERVAL_SECONDS)

    def stop(self) -> None:
        """Shutdown the APScheduler without waiting for running jobs."""
        self._scheduler.shutdown(wait=False)
        logger.info("NewsFeedPoller stopped")

    # ------------------------------------------------------------------
    # Internal poll cycle (executed by APScheduler)
    # ------------------------------------------------------------------

    async def _poll_cycle(self) -> None:
        """Main poll cycle — fetch, persist, classify, broadcast."""
        logger.debug("NewsFeedPoller: starting poll cycle (since=%s)", self._last_polled_utc)
        cycle_start = datetime.now(timezone.utc)

        # Attempt fetch with exponential backoff
        new_items = await self._fetch_with_retry()
        if new_items is None:
            # All retries exhausted — skip this cycle
            return

        self._last_polled_utc = cycle_start

        if not new_items:
            logger.debug("NewsFeedPoller: no new items this cycle")
            return

        logger.info("NewsFeedPoller: %d new items fetched", len(new_items))

        # Persist and classify each item
        for item in new_items:
            await self._persist_and_classify(item)

    async def _fetch_with_retry(self) -> Optional[list]:
        """
        Fetch latest news with exponential backoff.

        Returns:
            List of new NewsItem instances, or None if all retries failed.
        """
        last_exc: Optional[Exception] = None

        for attempt in range(_MAX_RETRY_ATTEMPTS):
            try:
                items = await self._provider.fetch_latest(since_utc=self._last_polled_utc)
                return items
            except Exception as exc:
                last_exc = exc
                delay = _BASE_RETRY_DELAY_SECONDS * (2 ** attempt)  # 10s, 20s, 40s
                logger.warning(
                    "NewsFeedPoller: Finnhub fetch failed (attempt %d/%d), "
                    "retrying in %ds: %s",
                    attempt + 1, _MAX_RETRY_ATTEMPTS, delay, exc,
                )
                if attempt < _MAX_RETRY_ATTEMPTS - 1:
                    await asyncio.sleep(delay)

        logger.error(
            "NewsFeedPoller: Finnhub fetch failed after %d attempts — skipping cycle. "
            "Last error: %s",
            _MAX_RETRY_ATTEMPTS, last_exc,
        )
        return None

    async def _persist_and_classify(self, item: NewsItem) -> None:
        """
        Persist a single NewsItem to DB (deduplicating by item_id),
        then classify via GeopoliticalSubAgent and update the row.
        On HIGH+ALERT: broadcast WebSocket event.
        On HIGH+FAST_TRACK: log WARNING only.
        """
        from src.database.engine import get_session

        # --- Persist ---
        db = get_session()
        try:
            # Deduplicate: skip if already in DB
            existing = db.query(NewsItem).filter_by(item_id=item.item_id).first()
            if existing:
                logger.debug(
                    "NewsFeedPoller: item_id=%s already in DB, skipping", item.item_id
                )
                return

            db.add(item)
            db.commit()
            db.refresh(item)
            logger.debug("NewsFeedPoller: persisted item_id=%s", item.item_id)
        except Exception as exc:
            db.rollback()
            logger.error(
                "NewsFeedPoller: DB persist failed for item_id=%s: %s",
                item.item_id, exc,
            )
            return
        finally:
            db.close()

        # --- Classify ---
        try:
            classification = self._sub_agent.classify(item)
        except Exception as exc:
            logger.error(
                "NewsFeedPoller: classification failed for item_id=%s: %s",
                item.item_id, exc,
            )
            return

        # --- Persist classification update ---
        db = get_session()
        try:
            db_item = db.query(NewsItem).filter_by(item_id=item.item_id).first()
            if db_item:
                db_item.severity = item.severity
                db_item.action_type = item.action_type
                db_item.classified_at_utc = item.classified_at_utc
                db.commit()
        except Exception as exc:
            db.rollback()
            logger.error(
                "NewsFeedPoller: classification DB update failed for item_id=%s: %s",
                item.item_id, exc,
            )
        finally:
            db.close()

        impact_tier = classification.get("impact_tier", "LOW")
        action_type = classification.get("action_type", "MONITOR")
        affected_symbols = classification.get("affected_symbols", [])

        # HIGH + FAST_TRACK: log warning, do NOT implement Fast-Track (Epic 8 scope)
        if impact_tier == "HIGH" and action_type == "FAST_TRACK":
            logger.warning(
                "FAST_TRACK event detected for item_id=%s — manual fast-track "
                "workflow required (headline: %s)",
                item.item_id, item.headline,
            )

        # HIGH + ALERT: broadcast via WebSocket topic="news" AND send department mail to Copilot (AC8)
        if impact_tier == "HIGH" and action_type == "ALERT":
            await self._broadcast_alert(item, affected_symbols)
            self._send_copilot_mail(item, affected_symbols)

    def _send_copilot_mail(self, item: NewsItem, affected_symbols: list) -> None:
        """Send department mail to Copilot system for HIGH+ALERT news events (AC8)."""
        try:
            from src.agents.departments.department_mail import (
                DepartmentMailService,
                MessageType,
                Priority,
            )

            mail_service = DepartmentMailService()

            subject = f"HIGH ALERT: {item.headline[:80]}"
            body = f"""# Geopolitical News Alert

**Headline:** {item.headline}

**Source:** {item.source or 'Unknown'}
**Published:** {item.published_utc.isoformat() if item.published_utc else 'Unknown'}
**Severity:** {item.severity}
**Action:** {item.action_type}

**Affected Symbols:** {', '.join(affected_symbols) if affected_symbols else 'None'}

**Summary:** {item.summary or 'No summary available'}

**URL:** {item.url or 'N/A'}
"""

            mail_service.send(
                from_dept="news_poller",
                to_dept="copilot",
                type=MessageType.ALERT,
                subject=subject,
                body=body,
                priority=Priority.HIGH,
            )
            mail_service.close()
            logger.info(
                "NewsFeedPoller: sent Copilot mail for item_id=%s", item.item_id
            )
        except Exception as exc:
            logger.error(
                "NewsFeedPoller: Copilot mail failed for item_id=%s: %s",
                item.item_id, exc,
            )

    async def _broadcast_alert(self, item: NewsItem, affected_symbols: list) -> None:
        """Broadcast a HIGH+ALERT news event via WebSocket topic='news'."""
        try:
            from src.api.websocket_endpoints import manager

            payload = {
                "type": "news_alert",
                "data": {
                    "item_id": item.item_id,
                    "headline": item.headline,
                    "severity": item.severity,
                    "action_type": item.action_type,
                    "affected_symbols": affected_symbols,
                    "published_utc": (
                        item.published_utc.isoformat()
                        if item.published_utc
                        else None
                    ),
                },
            }
            await manager.broadcast(payload, topic="news")
            logger.info(
                "NewsFeedPoller: broadcast HIGH+ALERT for item_id=%s", item.item_id
            )
        except Exception as exc:
            logger.error(
                "NewsFeedPoller: WebSocket broadcast failed for item_id=%s: %s",
                item.item_id, exc,
            )

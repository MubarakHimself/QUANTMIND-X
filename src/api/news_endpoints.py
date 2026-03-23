"""
News Feed API Endpoints (Story 6.3).

Provides:
    GET  /api/news/feed    — Latest 20 news items from SQLite (ordered by published_utc DESC)
    POST /api/news/alert   — Store a pre-classified alert + broadcast via WebSocket topic='news'

Registered in server.py under INCLUDE_CONTABO block.
"""

import logging
from typing import List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, field_validator
from sqlalchemy.orm import Session

from src.database.models.news_items import NewsItem
from src.database.models import get_db_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/news", tags=["news"])


# =============================================================================
# Pydantic models
# =============================================================================

class NewsFeedItem(BaseModel):
    item_id: str
    headline: str
    summary: Optional[str] = None
    source: Optional[str] = None
    published_utc: str       # ISO 8601 string
    url: Optional[str] = None
    related_instruments: List[str] = []
    severity: Optional[str] = None      # LOW / MEDIUM / HIGH
    action_type: Optional[str] = None   # MONITOR / ALERT / FAST_TRACK

    class Config:
        from_attributes = True


class NewsAlertRequest(BaseModel):
    item_id: str
    headline: str
    severity: Literal["HIGH", "MEDIUM", "LOW"]
    action_type: Literal["ALERT", "FAST_TRACK", "MONITOR"]
    affected_symbols: List[str] = []
    published_utc: str


class NewsAlertResponse(BaseModel):
    stored: bool
    broadcast: bool
    item_id: str


# =============================================================================
# Endpoints
# =============================================================================

@router.get("/feed", response_model=List[NewsFeedItem])
async def get_news_feed(db: Session = Depends(get_db_session)):
    """
    Return the latest 20 NewsItem rows from SQLite ordered by published_utc DESC.

    Includes severity and action_type as set by the GeopoliticalSubAgent.
    Items not yet classified will have severity=null, action_type=null.
    """
    items = (
        db.query(NewsItem)
        .order_by(NewsItem.published_utc.desc())
        .limit(20)
        .all()
    )

    result = []
    for item in items:
        result.append(
            NewsFeedItem(
                item_id=item.item_id,
                headline=item.headline,
                summary=item.summary,
                source=item.source,
                published_utc=(
                    item.published_utc.isoformat()
                    if item.published_utc
                    else ""
                ),
                url=item.url,
                related_instruments=item.related_instruments or [],
                severity=item.severity,
                action_type=item.action_type,
            )
        )

    return result


@router.post("/alert", response_model=NewsAlertResponse)
async def post_news_alert(
    alert: NewsAlertRequest,
    db: Session = Depends(get_db_session),
):
    """
    Store a pre-classified NewsAlert payload and broadcast to WebSocket topic='news'.

    If the item_id already exists in DB, updates severity and action_type.
    If not, inserts a new minimal NewsItem row.

    Broadcast payload shape:
        { item_id, headline, severity, action_type, affected_symbols, published_utc }
    """
    from datetime import datetime, timezone

    stored = False
    broadcast = False

    try:
        # Parse published_utc from ISO string once for use in both branches
        try:
            incoming_published_dt = datetime.fromisoformat(alert.published_utc.replace("Z", "+00:00"))
        except ValueError:
            incoming_published_dt = datetime.now(timezone.utc)

        # Upsert: find existing or create new
        existing = db.query(NewsItem).filter_by(item_id=alert.item_id).first()
        if existing:
            # BUG 2/3 fix: only update if incoming timestamp is same or newer
            # existing.published_utc may be naive (SQLite strips tzinfo); normalize both to UTC timestamps
            existing_ts = existing.published_utc
            if existing_ts is not None and existing_ts.tzinfo is None:
                existing_ts = existing_ts.replace(tzinfo=timezone.utc)
            incoming_ts = incoming_published_dt
            if incoming_ts.tzinfo is None:
                incoming_ts = incoming_ts.replace(tzinfo=timezone.utc)
            if existing_ts is None or incoming_ts >= existing_ts:
                existing.headline = alert.headline
                existing.summary = None
                existing.published_utc = incoming_published_dt
                existing.severity = alert.severity
                existing.action_type = alert.action_type
                existing.classified_at_utc = datetime.now(timezone.utc)
                db.commit()
                stored = True
                logger.info("news/alert: updated item_id=%s (severity=%s)", alert.item_id, alert.severity)
            else:
                # BUG 3 fix: incoming is older, skip update but still return success
                db.commit()
                stored = True
                logger.info("news/alert: skipped item_id=%s (incoming timestamp older)", alert.item_id)
        else:
            new_item = NewsItem(
                item_id=alert.item_id,
                headline=alert.headline,
                published_utc=incoming_published_dt,
                severity=alert.severity,
                action_type=alert.action_type,
                classified_at_utc=datetime.now(timezone.utc),
                related_instruments=alert.affected_symbols or None,
            )
            db.add(new_item)
            db.commit()
            stored = True
            logger.info("news/alert: inserted item_id=%s (severity=%s)", alert.item_id, alert.severity)
    except Exception as exc:
        db.rollback()
        logger.error("news/alert: DB error for item_id=%s: %s", alert.item_id, exc)
        raise HTTPException(status_code=500, detail=f"Database error: {exc}") from exc

    # Broadcast via WebSocket
    try:
        from src.api.websocket_endpoints import manager

        payload = {
            "type": "news_alert",
            "data": {
                "item_id": alert.item_id,
                "headline": alert.headline,
                "severity": alert.severity,
                "action_type": alert.action_type,
                "affected_symbols": alert.affected_symbols,
                "published_utc": alert.published_utc,
            },
        }
        await manager.broadcast(payload, topic="news")
        broadcast = True
        logger.info("news/alert: broadcast item_id=%s to WebSocket topic=news", alert.item_id)
    except Exception as exc:
        logger.error("news/alert: WebSocket broadcast failed for item_id=%s: %s", alert.item_id, exc)
        # Do not fail the request if broadcast fails — stored is the critical part

    return NewsAlertResponse(
        stored=stored,
        broadcast=broadcast,
        item_id=alert.item_id,
    )

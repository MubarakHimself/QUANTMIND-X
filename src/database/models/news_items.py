"""
News Items Database Model.

Stores live news feed items from Finnhub with geopolitical classification
performed by the GeopoliticalSubAgent (Story 6.3).
"""

from datetime import datetime, timezone
from sqlalchemy import Column, String, Text, DateTime, JSON, Index
from ..models.base import Base


class NewsItem(Base):
    """
    Live news item from Finnhub, enriched by GeopoliticalSubAgent classification.

    Attributes:
        item_id: UUID string primary key (Finnhub id cast to string)
        headline: Article headline text
        summary: Article summary (may be None for short articles)
        source: News source name (e.g., "Reuters", "Bloomberg")
        published_utc: When the article was published (UTC)
        url: Article URL
        related_instruments: JSON list of forex/equity symbols (e.g., ["EURUSD", "GBPUSD"])
        severity: GeopoliticalSubAgent classification — LOW / MEDIUM / HIGH (nullable until classified)
        action_type: GeopoliticalSubAgent action — MONITOR / ALERT / FAST_TRACK (nullable until classified)
        classified_at_utc: When the sub-agent classified this item (nullable)
        created_at_utc: When the row was inserted into the DB
    """
    __tablename__ = 'news_items'

    item_id = Column(String(100), primary_key=True)
    headline = Column(Text, nullable=False)
    summary = Column(Text, nullable=True)
    source = Column(String(200), nullable=True)
    published_utc = Column(DateTime, nullable=False)
    url = Column(String(500), nullable=True)
    related_instruments = Column(JSON, nullable=True)          # ["EURUSD", "GBPUSD"]
    severity = Column(String(10), nullable=True)               # LOW / MEDIUM / HIGH
    action_type = Column(String(20), nullable=True)            # MONITOR / ALERT / FAST_TRACK
    classified_at_utc = Column(DateTime, nullable=True)
    created_at_utc = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )

    __table_args__ = (
        Index('idx_news_items_published_utc', 'published_utc'),
        Index('idx_news_items_severity', 'severity'),
    )

    def __repr__(self):
        return (
            f"<NewsItem(item_id={self.item_id!r}, "
            f"headline={self.headline[:40]!r}, "
            f"severity={self.severity!r})>"
        )

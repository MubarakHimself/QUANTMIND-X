"""
NewsProvider Abstract Base Class.

Defines the contract for all news data providers used by the NewsFeedPoller.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from src.database.models.news_items import NewsItem


class NewsProvider(ABC):
    """
    Abstract base class for news data providers.

    Implementors must provide `fetch_latest` which returns a list of
    `NewsItem` instances representing articles published since `since_utc`.
    """

    @abstractmethod
    async def fetch_latest(self, since_utc: datetime) -> List["NewsItem"]:
        """
        Fetch news articles published since the given UTC datetime.

        Args:
            since_utc: Fetch articles published after this UTC timestamp.

        Returns:
            List of NewsItem ORM instances (not yet persisted to DB).

        Raises:
            HTTPException(503): If the provider is unavailable or misconfigured.
        """
        ...

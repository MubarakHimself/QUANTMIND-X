"""
Live News Feed package (Story 6.3).

Provides:
    - NewsProvider: Abstract base class for news data providers
    - FinnhubProvider: Finnhub SDK implementation
    - GeopoliticalSubAgent: Haiku-tier news impact classifier
    - NewsFeedPoller: APScheduler-based polling service
"""

from src.knowledge.news.provider import NewsProvider
from src.knowledge.news.finnhub_provider import FinnhubProvider
from src.knowledge.news.geopolitical_subagent import GeopoliticalSubAgent
from src.knowledge.news.poller import NewsFeedPoller

__all__ = [
    "NewsProvider",
    "FinnhubProvider",
    "GeopoliticalSubAgent",
    "NewsFeedPoller",
]

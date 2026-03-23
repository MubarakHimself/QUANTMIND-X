"""
Tests for Live News Feed API Endpoints (Story 6.3).

Covers:
    - GET /api/news/feed (AC1, AC6)
    - POST /api/news/alert (AC5)
    - FinnhubProvider field mapping and deduplication (AC2)
    - GeopoliticalSubAgent JSON parsing + fallback behavior (AC3)
    - NewsFeedPoller retry logic (AC4)
"""

import json
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch, call
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.database.models.base import Base
from src.database.models.news_items import NewsItem


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def in_memory_engine():
    """
    Create an in-memory SQLite engine with all tables.

    Uses StaticPool so all connections within the test share the same
    in-memory database (required for TestClient running in a thread pool).
    """
    # Ensure NewsItem is registered in Base.metadata before create_all
    from src.database.models.news_items import NewsItem as _NewsItem  # noqa: F401

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    return engine


@pytest.fixture
def db_session(in_memory_engine):
    """Provide a transactional DB session for tests."""
    Session = sessionmaker(bind=in_memory_engine)
    session = Session()
    yield session
    session.rollback()
    session.close()


@pytest.fixture
def news_app(in_memory_engine):
    """
    Create a test FastAPI app with the news router and in-memory DB.

    Overrides get_db_session to use in-memory SQLite.
    """
    from src.api.news_endpoints import router

    app = FastAPI()
    app.include_router(router)

    Session = sessionmaker(bind=in_memory_engine)

    def override_get_db():
        session = Session()
        try:
            yield session
        finally:
            session.close()

    from src.database.models import get_db_session
    app.dependency_overrides[get_db_session] = override_get_db

    return app


@pytest.fixture
def client(news_app):
    """TestClient for the news test app."""
    return TestClient(news_app)


def _make_news_item(
    item_id: str,
    headline: str,
    published_utc: datetime = None,
    severity: str = None,
    action_type: str = None,
) -> NewsItem:
    """Helper: create a NewsItem instance for seeding tests."""
    if published_utc is None:
        published_utc = datetime.now(timezone.utc)
    return NewsItem(
        item_id=item_id,
        headline=headline,
        summary="A test summary",
        source="TestSource",
        published_utc=published_utc,
        url="https://example.com/news/1",
        related_instruments=["EURUSD"],
        severity=severity,
        action_type=action_type,
    )


# =============================================================================
# Tests: GET /api/news/feed
# =============================================================================

class TestGetNewsFeed:
    """Tests for GET /api/news/feed (AC1, AC6)."""

    def test_returns_empty_list_when_no_items(self, client):
        """Should return 200 with empty list when DB is empty."""
        response = client.get("/api/news/feed")
        assert response.status_code == 200
        assert response.json() == []

    def test_returns_latest_20_ordered_by_published_utc_desc(self, client, news_app, in_memory_engine):
        """Should return latest 20 items ordered by published_utc DESC."""
        Session = sessionmaker(bind=in_memory_engine)
        session = Session()

        base_time = datetime.now(timezone.utc)
        items = []
        for i in range(25):
            item = _make_news_item(
                item_id=str(i),
                headline=f"Headline {i}",
                published_utc=base_time - timedelta(minutes=i),
            )
            items.append(item)
        session.add_all(items)
        session.commit()
        session.close()

        response = client.get("/api/news/feed")
        assert response.status_code == 200
        data = response.json()

        # Only 20 items returned
        assert len(data) == 20

        # Ordered by published_utc DESC (most recent first = item_id 0)
        assert data[0]["item_id"] == "0"
        assert data[-1]["item_id"] == "19"

    def test_returns_severity_and_action_type(self, client, in_memory_engine):
        """Should include severity and action_type from sub-agent classification."""
        Session = sessionmaker(bind=in_memory_engine)
        session = Session()

        item = _make_news_item(
            item_id="classified-1",
            headline="ECB raises rates unexpectedly",
            severity="HIGH",
            action_type="ALERT",
        )
        session.add(item)
        session.commit()
        session.close()

        response = client.get("/api/news/feed")
        assert response.status_code == 200
        data = response.json()

        assert len(data) == 1
        assert data[0]["severity"] == "HIGH"
        assert data[0]["action_type"] == "ALERT"
        assert data[0]["item_id"] == "classified-1"

    def test_returns_null_severity_for_unclassified_items(self, client, in_memory_engine):
        """Unclassified items should have severity=null and action_type=null."""
        Session = sessionmaker(bind=in_memory_engine)
        session = Session()

        item = _make_news_item(item_id="unclassified-1", headline="Some routine news")
        session.add(item)
        session.commit()
        session.close()

        response = client.get("/api/news/feed")
        assert response.status_code == 200
        data = response.json()
        assert data[0]["severity"] is None
        assert data[0]["action_type"] is None


# =============================================================================
# Tests: POST /api/news/alert
# =============================================================================

class TestPostNewsAlert:
    """Tests for POST /api/news/alert (AC5)."""

    def test_stores_alert_and_broadcasts_websocket(self, client, in_memory_engine):
        """Should store alert in DB and broadcast via WebSocket topic='news'."""
        with patch("src.api.websocket_endpoints.manager") as mock_manager:
            mock_manager.broadcast = AsyncMock()

            payload = {
                "item_id": "alert-001",
                "headline": "Fed surprise rate cut",
                "severity": "HIGH",
                "action_type": "ALERT",
                "affected_symbols": ["EURUSD", "GBPUSD"],
                "published_utc": "2026-03-19T10:00:00+00:00",
            }

            response = client.post("/api/news/alert", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["stored"] is True
        assert data["broadcast"] is True
        assert data["item_id"] == "alert-001"

        # Verify WebSocket broadcast was called with topic='news'
        mock_manager.broadcast.assert_called_once()
        call_args = mock_manager.broadcast.call_args
        assert call_args[1]["topic"] == "news" or call_args[0][1] == "news"

    def test_updates_existing_item_on_duplicate_item_id(self, client, in_memory_engine):
        """Should update severity/action_type if item_id already exists."""
        Session = sessionmaker(bind=in_memory_engine)
        session = Session()

        existing = _make_news_item(
            item_id="existing-001",
            headline="Existing headline",
            severity="LOW",
            action_type="MONITOR",
        )
        session.add(existing)
        session.commit()
        session.close()

        with patch("src.api.websocket_endpoints.manager") as mock_manager:
            mock_manager.broadcast = AsyncMock()

            response = client.post("/api/news/alert", json={
                "item_id": "existing-001",
                "headline": "Existing headline",
                "severity": "HIGH",
                "action_type": "ALERT",
                "affected_symbols": ["USDJPY"],
                "published_utc": "2026-03-19T10:00:00+00:00",
            })

        assert response.status_code == 200
        assert response.json()["stored"] is True

        # Verify DB was updated
        Session2 = sessionmaker(bind=in_memory_engine)
        session2 = Session2()
        updated = session2.query(NewsItem).filter_by(item_id="existing-001").first()
        session2.close()

        assert updated.severity == "HIGH"
        assert updated.action_type == "ALERT"

    def test_broadcast_payload_contains_required_fields(self, client, in_memory_engine):
        """Broadcast payload must contain: item_id, headline, severity, action_type, affected_symbols, published_utc."""
        broadcast_calls = []

        async def capture_broadcast(payload, topic=None):
            broadcast_calls.append((payload, topic))

        with patch("src.api.websocket_endpoints.manager") as mock_manager:
            mock_manager.broadcast = capture_broadcast

            response = client.post("/api/news/alert", json={
                "item_id": "broadcast-test-001",
                "headline": "Test headline",
                "severity": "MEDIUM",
                "action_type": "MONITOR",
                "affected_symbols": ["EURUSD"],
                "published_utc": "2026-03-19T11:00:00+00:00",
            })

        assert response.status_code == 200
        assert len(broadcast_calls) == 1

        payload, topic = broadcast_calls[0]
        assert topic == "news"
        assert payload["type"] == "news_alert"
        data = payload["data"]
        assert "item_id" in data
        assert "headline" in data
        assert "severity" in data
        assert "action_type" in data
        assert "affected_symbols" in data
        assert "published_utc" in data


# =============================================================================
# Tests: FinnhubProvider
# =============================================================================

class TestFinnhubProvider:
    """Tests for FinnhubProvider field mapping and deduplication (AC2)."""

    def test_raises_runtime_error_when_finnhub_api_key_not_set(self):
        """Should raise RuntimeError if FINNHUB_API_KEY not in env."""
        with patch.dict("os.environ", {}, clear=False):
            import os
            original = os.environ.pop("FINNHUB_API_KEY", None)
            try:
                with pytest.raises(RuntimeError, match="FINNHUB_API_KEY not configured"):
                    from src.knowledge.news.finnhub_provider import FinnhubProvider
                    FinnhubProvider()
            finally:
                if original:
                    os.environ["FINNHUB_API_KEY"] = original

    def test_fetch_latest_maps_fields_correctly(self):
        """Should map Finnhub API response fields to NewsItem correctly."""
        mock_response = [
            {
                "id": 12345,
                "headline": "ECB Policy Decision",
                "summary": "ECB raises interest rates by 25bps",
                "source": "Reuters",
                "datetime": 1710000000,
                "url": "https://reuters.com/article/1",
                "related": "EURUSD,GBPUSD",
                "category": "forex",
            }
        ]

        with patch.dict("os.environ", {"FINNHUB_API_KEY": "test-key-abc"}):
            with patch("src.knowledge.news.finnhub_provider.finnhub") as mock_finnhub_module:
                mock_client = MagicMock()
                mock_client.general_news.return_value = mock_response
                mock_finnhub_module.Client.return_value = mock_client

                from src.knowledge.news.finnhub_provider import FinnhubProvider
                import asyncio

                provider = FinnhubProvider()

                # since_utc in the past to allow all items through
                since_utc = datetime.utcfromtimestamp(0).replace(tzinfo=timezone.utc)
                items = asyncio.get_event_loop().run_until_complete(
                    provider.fetch_latest(since_utc=since_utc)
                )

        assert len(items) == 1
        item = items[0]
        assert item.item_id == "12345"
        assert item.headline == "ECB Policy Decision"
        assert item.summary == "ECB raises interest rates by 25bps"
        assert item.source == "Reuters"
        assert item.url == "https://reuters.com/article/1"
        assert "EURUSD" in item.related_instruments
        assert "GBPUSD" in item.related_instruments
        # published_utc should be UTC datetime from unix timestamp
        assert item.published_utc == datetime.utcfromtimestamp(1710000000).replace(tzinfo=timezone.utc)

    def test_fetch_latest_deduplicates_across_categories(self):
        """Items with duplicate id should appear only once in the result."""
        # Same item returned in both 'general' and 'forex' categories
        duplicate_item = {
            "id": 99999,
            "headline": "Duplicate headline",
            "summary": "Summary",
            "source": "Bloomberg",
            "datetime": 1710000000,
            "url": "https://bloomberg.com/1",
            "related": "",
        }

        with patch.dict("os.environ", {"FINNHUB_API_KEY": "test-key-abc"}):
            with patch("src.knowledge.news.finnhub_provider.finnhub") as mock_finnhub_module:
                mock_client = MagicMock()
                # Both categories return the same item
                mock_client.general_news.return_value = [duplicate_item]
                mock_finnhub_module.Client.return_value = mock_client

                from src.knowledge.news.finnhub_provider import FinnhubProvider
                import asyncio

                provider = FinnhubProvider()
                since_utc = datetime.utcfromtimestamp(0).replace(tzinfo=timezone.utc)
                items = asyncio.get_event_loop().run_until_complete(
                    provider.fetch_latest(since_utc=since_utc)
                )

        # Should only have one, not two
        assert len(items) == 1
        assert items[0].item_id == "99999"

    def test_fetch_latest_filters_by_since_utc(self):
        """Items published before since_utc should be filtered out."""
        old_item = {
            "id": 11111,
            "headline": "Old news",
            "summary": "Old",
            "source": "Old source",
            "datetime": 1000000,   # Unix timestamp far in the past
            "url": "https://old.com",
            "related": "",
        }

        with patch.dict("os.environ", {"FINNHUB_API_KEY": "test-key-abc"}):
            with patch("src.knowledge.news.finnhub_provider.finnhub") as mock_finnhub_module:
                mock_client = MagicMock()
                mock_client.general_news.return_value = [old_item]
                mock_finnhub_module.Client.return_value = mock_client

                from src.knowledge.news.finnhub_provider import FinnhubProvider
                import asyncio

                provider = FinnhubProvider()
                # since_utc is well after the item's published_utc
                since_utc = datetime.now(timezone.utc) - timedelta(minutes=5)
                items = asyncio.get_event_loop().run_until_complete(
                    provider.fetch_latest(since_utc=since_utc)
                )

        assert items == []


# =============================================================================
# Tests: GeopoliticalSubAgent
# =============================================================================

class TestGeopoliticalSubAgent:
    """Tests for GeopoliticalSubAgent classification and fallback (AC3)."""

    def _make_item(self) -> NewsItem:
        return _make_news_item(
            item_id="test-geo-001",
            headline="ECB raises rates by 75bps unexpectedly",
            published_utc=datetime.now(timezone.utc),
        )

    def test_classify_returns_valid_classification(self):
        """Should parse and return valid classification JSON from LLM."""
        mock_response_text = json.dumps({
            "impact_tier": "HIGH",
            "affected_symbols": ["EURUSD", "EURGBP"],
            "event_type": "central_bank",
            "action_type": "FAST_TRACK",
        })

        mock_content = MagicMock()
        mock_content.text = mock_response_text
        mock_response = MagicMock()
        mock_response.content = [mock_content]

        with patch("src.knowledge.news.geopolitical_subagent.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_anthropic.return_value = mock_client

            from src.knowledge.news.geopolitical_subagent import GeopoliticalSubAgent
            agent = GeopoliticalSubAgent()
            item = self._make_item()
            result = agent.classify(item)

        assert result["impact_tier"] == "HIGH"
        assert "EURUSD" in result["affected_symbols"]
        assert result["event_type"] == "central_bank"
        assert result["action_type"] == "FAST_TRACK"

        # Item fields should be updated in-place
        assert item.severity == "HIGH"
        assert item.action_type == "FAST_TRACK"
        assert item.classified_at_utc is not None

    def test_classify_returns_fallback_on_malformed_json(self):
        """Should return fallback classification when LLM returns malformed JSON."""
        mock_content = MagicMock()
        mock_content.text = "This is not JSON at all"
        mock_response = MagicMock()
        mock_response.content = [mock_content]

        with patch("src.knowledge.news.geopolitical_subagent.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_anthropic.return_value = mock_client

            from src.knowledge.news.geopolitical_subagent import GeopoliticalSubAgent
            agent = GeopoliticalSubAgent()
            item = self._make_item()
            result = agent.classify(item)

        # Fallback values
        assert result["impact_tier"] == "LOW"
        assert result["affected_symbols"] == []
        assert result["action_type"] == "MONITOR"

    def test_classify_returns_fallback_on_llm_exception(self):
        """Should return fallback classification when LLM call raises an exception."""
        with patch("src.knowledge.news.geopolitical_subagent.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.side_effect = Exception("LLM API error")
            mock_anthropic.return_value = mock_client

            from src.knowledge.news.geopolitical_subagent import GeopoliticalSubAgent
            agent = GeopoliticalSubAgent()
            item = self._make_item()
            result = agent.classify(item)

        assert result["impact_tier"] == "LOW"
        assert result["action_type"] == "MONITOR"

    def test_classify_normalises_unknown_impact_tier(self):
        """Should normalise unexpected impact_tier values to LOW."""
        mock_content = MagicMock()
        mock_content.text = json.dumps({
            "impact_tier": "CRITICAL",   # Not a valid value
            "affected_symbols": [],
            "event_type": "other",
            "action_type": "MONITOR",
        })
        mock_response = MagicMock()
        mock_response.content = [mock_content]

        with patch("src.knowledge.news.geopolitical_subagent.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_anthropic.return_value = mock_client

            from src.knowledge.news.geopolitical_subagent import GeopoliticalSubAgent
            agent = GeopoliticalSubAgent()
            item = self._make_item()
            result = agent.classify(item)

        assert result["impact_tier"] == "LOW"


# =============================================================================
# Tests: NewsFeedPoller retry logic
# =============================================================================

class TestNewsFeedPollerRetryLogic:
    """Tests for NewsFeedPoller exponential backoff retry (AC4)."""

    @pytest.mark.asyncio
    async def test_retry_succeeds_on_third_attempt(self):
        """
        Provider raises exception twice, succeeds on third attempt.
        Poller should return the successful result.
        """
        success_items = [
            _make_news_item(
                item_id="retry-001",
                headline="Retry test headline",
                published_utc=datetime.now(timezone.utc),
            )
        ]

        fetch_call_count = 0

        async def mock_fetch_latest(since_utc):
            nonlocal fetch_call_count
            fetch_call_count += 1
            if fetch_call_count < 3:
                raise ConnectionError(f"Simulated Finnhub offline (attempt {fetch_call_count})")
            return success_items

        with patch.dict("os.environ", {"FINNHUB_API_KEY": "test-key"}):
            with patch("src.knowledge.news.poller.FinnhubProvider") as mock_provider_cls:
                with patch("src.knowledge.news.poller.GeopoliticalSubAgent"):
                    with patch("asyncio.sleep", new_callable=AsyncMock):
                        mock_provider = MagicMock()
                        mock_provider.fetch_latest = mock_fetch_latest
                        mock_provider_cls.return_value = mock_provider

                        from src.knowledge.news.poller import NewsFeedPoller
                        poller = NewsFeedPoller()

                        result = await poller._fetch_with_retry()

        assert result is not None
        assert len(result) == 1
        assert result[0].item_id == "retry-001"
        assert fetch_call_count == 3

    @pytest.mark.asyncio
    async def test_returns_none_after_max_retries_exhausted(self):
        """Should return None after all 3 retry attempts fail."""
        async def always_fail(since_utc):
            raise ConnectionError("Finnhub is down")

        with patch.dict("os.environ", {"FINNHUB_API_KEY": "test-key"}):
            with patch("src.knowledge.news.poller.FinnhubProvider") as mock_provider_cls:
                with patch("src.knowledge.news.poller.GeopoliticalSubAgent"):
                    with patch("asyncio.sleep", new_callable=AsyncMock):
                        mock_provider = MagicMock()
                        mock_provider.fetch_latest = always_fail
                        mock_provider_cls.return_value = mock_provider

                        from src.knowledge.news.poller import NewsFeedPoller
                        poller = NewsFeedPoller()

                        result = await poller._fetch_with_retry()

        assert result is None

    @pytest.mark.asyncio
    async def test_retry_uses_exponential_backoff_delays(self):
        """Should sleep for 10s, 20s (exponential backoff) between retry attempts."""
        sleep_calls = []

        async def mock_sleep(seconds):
            sleep_calls.append(seconds)

        async def always_fail(since_utc):
            raise ConnectionError("down")

        with patch.dict("os.environ", {"FINNHUB_API_KEY": "test-key"}):
            with patch("src.knowledge.news.poller.FinnhubProvider") as mock_provider_cls:
                with patch("src.knowledge.news.poller.GeopoliticalSubAgent"):
                    with patch("asyncio.sleep", side_effect=mock_sleep):
                        mock_provider = MagicMock()
                        mock_provider.fetch_latest = always_fail
                        mock_provider_cls.return_value = mock_provider

                        from src.knowledge.news.poller import NewsFeedPoller
                        poller = NewsFeedPoller()
                        await poller._fetch_with_retry()

        # Should sleep twice (after attempt 1 and after attempt 2; no sleep after attempt 3)
        assert len(sleep_calls) == 2
        assert sleep_calls[0] == 10   # 10 * 2^0
        assert sleep_calls[1] == 20   # 10 * 2^1

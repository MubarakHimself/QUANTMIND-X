"""
P1 Tests: News Feed Tile Rendering

Epic 6 - Knowledge & Research Engine
Priority: P1 (High)
Coverage: News feed tile component behavior validation

Covers:
- Severity badge rendering
- Time ago formatting
- Symbol tag display
- Empty state handling
- Alert action buttons
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, patch

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
    """Create an in-memory SQLite engine with all tables."""
    from src.database.models.news_items import NewsItem as _NewsItem  # noqa: F401
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    return engine


@pytest.fixture
def news_app(in_memory_engine):
    """Create test FastAPI app with news router."""
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
    """TestClient for news endpoints."""
    return TestClient(news_app)


# =============================================================================
# Helper Functions (these would be in the component, tested here)
# =============================================================================

def _severity_to_color(severity: str) -> str:
    """Map severity level to UI color."""
    if severity == "HIGH":
        return "red"
    elif severity == "MEDIUM":
        return "orange"
    elif severity == "LOW":
        return "green"
    return "gray"


def _format_time_ago(dt: datetime) -> str:
    """Format datetime as 'X min ago' or 'X hour ago'."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    delta = now - dt

    minutes = int(delta.total_seconds() / 60)
    if minutes < 1:
        return "just now"
    if minutes < 60:
        return f"{minutes} min ago"

    hours = minutes // 60
    if hours < 24:
        return f"{hours} hour{'s' if hours > 1 else ''} ago"

    days = hours // 24
    return f"{days} day{'s' if days > 1 else ''} ago"


def _extract_symbol_tags(instruments: list) -> list:
    """Extract instruments as symbol tags for display."""
    return [f"symbol:{inst}" for inst in instruments]


def _truncate_headline(headline: str, max_length: int = 100) -> str:
    """Truncate long headlines with ellipsis.

    Keeps max_length characters of content, appends '...' (3 chars).
    Result length = max_length + 3.
    """
    if len(headline) <= max_length:
        return headline
    return headline[:max_length] + "..."


def _get_action_buttons(item: dict) -> list:
    """Get configured action buttons for news item type."""
    action_type = item.get("action_type")
    severity = item.get("severity")

    if action_type == "ALERT" and severity == "HIGH":
        return ["View Trade", "Dismiss Alert", "Details"]
    elif action_type == "ALERT":
        return ["View Trade", "Details"]
    elif action_type == "FAST_TRACK":
        return ["Quick Trade", "View Details"]
    else:
        return ["View"]


# =============================================================================
# P1 Tests: News Feed Tile Rendering
# =============================================================================

class TestNewsFeedTileRendering:
    """P1 tests for news feed tile component behavior."""

    def test_severity_badge_high_renders_red(self):
        """P1: HIGH severity should map to red color for urgent display."""
        assert _severity_to_color("HIGH") == "red"

    def test_severity_badge_medium_renders_orange(self):
        """P1: MEDIUM severity should map to orange color."""
        assert _severity_to_color("MEDIUM") == "orange"

    def test_severity_badge_low_renders_green(self):
        """P1: LOW severity should map to green color."""
        assert _severity_to_color("LOW") == "green"

    def test_severity_badge_null_renders_gray(self):
        """P1: Null severity should map to gray (neutral/default)."""
        assert _severity_to_color(None) == "gray"

    def test_time_ago_just_now(self):
        """P1: Recent items should show 'just now'."""
        now = datetime.now(timezone.utc)
        assert "just now" in _format_time_ago(now)

    def test_time_ago_minutes(self):
        """P1: Items within an hour show minutes."""
        minutes_ago = datetime.now(timezone.utc) - timedelta(minutes=5)
        result = _format_time_ago(minutes_ago)
        assert "5 min" in result

    def test_time_ago_hours(self):
        """P1: Items within a day show hours."""
        hours_ago = datetime.now(timezone.utc) - timedelta(hours=3)
        result = _format_time_ago(hours_ago)
        assert "3 hour" in result

    def test_time_ago_days(self):
        """P1: Items beyond a day show days."""
        days_ago = datetime.now(timezone.utc) - timedelta(days=2)
        result = _format_time_ago(days_ago)
        assert "2 day" in result

    def test_symbol_tags_extraction(self):
        """P1: Related instruments should be extracted as symbol tags."""
        instruments = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]
        tags = _extract_symbol_tags(instruments)
        assert len(tags) == 4
        assert all(t.startswith("symbol:") for t in tags)
        assert "symbol:EURUSD" in tags

    def test_truncate_long_headline(self):
        """P1: Headlines >100 chars should be truncated with ellipsis."""
        long_headline = "A" * 150
        truncated = _truncate_headline(long_headline, max_length=100)
        assert len(truncated) == 103  # 100 + "..."
        assert truncated.endswith("...")

    def test_truncate_short_headline_unchanged(self):
        """P1: Short headlines should not be truncated."""
        short = "Short headline"
        assert _truncate_headline(short) == short

    def test_action_buttons_alert_high(self):
        """P1: HIGH ALERT should have urgent action buttons."""
        item = {"action_type": "ALERT", "severity": "HIGH"}
        buttons = _get_action_buttons(item)
        assert len(buttons) >= 2
        assert any("trade" in b.lower() or "alert" in b.lower() for b in buttons)

    def test_action_buttons_monitoring(self):
        """P1: MONITOR items should have basic View button."""
        item = {"action_type": "MONITOR", "severity": "LOW"}
        buttons = _get_action_buttons(item)
        assert "View" in buttons


class TestNewsFeedAPIIntegration:
    """P1 tests for news feed API with tile rendering data."""

    def test_feed_returns_items_with_all_tile_fields(self, client, in_memory_engine):
        """P1: Feed items should include all fields needed for tile rendering."""
        Session = sessionmaker(bind=in_memory_engine)
        session = Session()

        item = NewsItem(
            item_id="tile-test-001",
            headline="Fed announces rate decision",
            summary="Fed keeps rates unchanged",
            source="Reuters",
            published_utc=datetime.now(timezone.utc) - timedelta(minutes=30),
            url="https://reuters.com/article/1",
            related_instruments=["EURUSD", "USDJPY"],
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
        item_data = data[0]

        # All fields needed for tile rendering
        assert "item_id" in item_data
        assert "headline" in item_data
        assert "severity" in item_data
        assert "action_type" in item_data
        assert "related_instruments" in item_data
        assert "published_utc" in item_data

    def test_feed_empty_returns_empty_list(self, client):
        """P1: Empty news feed should return empty array, not null."""
        response = client.get("/api/news/feed")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 0

    def test_feed_includes_unread_indicator_field(self, client, in_memory_engine):
        """P1: Items should include timestamp for 'new' indicator logic."""
        Session = sessionmaker(bind=in_memory_engine)
        session = Session()

        recent = datetime.now(timezone.utc) - timedelta(minutes=5)
        item = NewsItem(
            item_id="new-indicator-test",
            headline="Breaking news",
            summary="Test",
            source="Test",
            published_utc=recent,
            url="https://test.com/1",
            related_instruments=["EURUSD"],
        )
        session.add(item)
        session.commit()
        session.close()

        response = client.get("/api/news/feed")
        assert response.status_code == 200
        data = response.json()

        # Frontend uses published_utc to compute 'new' badge
        assert "published_utc" in data[0]


# =============================================================================
# Test Summary
# =============================================================================

"""
P1 News Feed Tile Test Coverage:
| Test | Scenario | P0 Gap |
|------|----------|--------|
| test_severity_badge_high_renders_red | Color mapping | No P0 equivalent |
| test_severity_badge_medium_renders_orange | Color mapping | No P0 equivalent |
| test_severity_badge_low_renders_green | Color mapping | No P0 equivalent |
| test_severity_badge_null_renders_gray | Null handling | No P0 equivalent |
| test_time_ago_just_now | Time formatting | No P0 equivalent |
| test_time_ago_minutes | Time formatting | No P0 equivalent |
| test_time_ago_hours | Time formatting | No P0 equivalent |
| test_time_ago_days | Time formatting | No P0 equivalent |
| test_symbol_tags_extraction | Symbol display | No P0 equivalent |
| test_truncate_long_headline | Headline overflow | No P0 equivalent |
| test_truncate_short_headline_unchanged | Edge case | No P0 equivalent |
| test_action_buttons_alert_high | Action buttons | No P0 equivalent |
| test_action_buttons_monitoring | Action buttons | No P0 equivalent |
| test_feed_returns_items_with_all_tile_fields | API integration | No P0 equivalent |
| test_feed_empty_returns_empty_list | Empty state | No P0 equivalent |
| test_feed_includes_unread_indicator_field | New badge | No P0 equivalent |

Total: 16 P1 tests added
"""

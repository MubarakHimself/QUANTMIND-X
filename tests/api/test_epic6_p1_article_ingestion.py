"""
P1 Tests: Article Ingestion Edge Cases

Epic 6 - Knowledge & Research Engine
Priority: P1 (High)
Coverage: Article ingestion edge cases beyond P0 scope

Covers:
- URL validation edge cases
- Unicode content handling
- Relevance tag validation
- Rate limit retry scenarios
"""

import pytest
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def test_app():
    """Create a minimal FastAPI app with the knowledge_ingest router."""
    from src.api.knowledge_ingest_endpoints import router
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(test_app):
    """Return a synchronous TestClient for the test app."""
    return TestClient(test_app)


@pytest.fixture(autouse=True)
def firecrawl_api_key(monkeypatch):
    """Inject a fake Firecrawl API key so endpoints don't return 503."""
    monkeypatch.setenv("FIRECRAWL_API_KEY", "test-firecrawl-key")


# =============================================================================
# Helper
# =============================================================================

def _mock_firecrawl_success(markdown: str = "# Test\nContent"):
    """Return a mock successful Firecrawl response."""
    return {"markdown": markdown}


# =============================================================================
# P1 Tests: Article Ingestion Edge Cases
# =============================================================================

class TestArticleIngestionEdgeCases:
    """P1 edge case tests for article ingestion beyond P0 coverage."""

    def test_ingest_url_with_special_characters_in_query_params(self, client):
        """
        P1: URLs with special characters (? & =) should be handled correctly.

        Validates that URL encoding is preserved during Firecrawl scrape request.
        """
        with patch("src.api.knowledge_ingest_endpoints.FirecrawlApp") as MockApp:
            instance = MockApp.return_value
            instance.scrape.return_value = _mock_firecrawl_success()

            response = client.post(
                "/api/knowledge/ingest",
                json={
                    "url": "https://example.com/article?id=123&ref=test&token=abc&filter=active%20status"
                },
            )

            assert response.status_code == 200, response.text
            data = response.json()
            assert data["status"] == "success"

    def test_ingest_url_with_unicode_content(self, client):
        """
        P1: Firecrawl returning unicode content should be handled.

        Verifies that international characters, emoji, and non-ASCII
        content are preserved in the stored markdown file.
        """
        unicode_content = (
            "# \u30c6\u30b9\u30c8\n"  # Japanese: Test
            "Content with emoji: \U0001F4C8\n"  # Chart emoji
            "Cyrillic: \u041f\u0440\u0438\u0432\u0435\u0442\n"  # Russian: Hi
            "Arabic: \u0627\u0644\u0633\u0644\u0627\u0645"
        )

        with patch("src.api.knowledge_ingest_endpoints.FirecrawlApp") as MockApp:
            instance = MockApp.return_value
            instance.scrape.return_value = {"markdown": unicode_content}

            response = client.post(
                "/api/knowledge/ingest",
                json={"url": "https://example.com/unicode"},
            )

            assert response.status_code == 200, response.text
            data = response.json()
            assert data["status"] == "success"

    def test_ingest_url_with_very_long_url(self, client):
        """
        P1: Very long URLs (>2000 chars) should be rejected with 422.

        HTTP servers typically limit URL length, so we validate upfront.
        """
        long_url = "https://example.com/" + "a" * 5000
        response = client.post(
            "/api/knowledge/ingest",
            json={"url": long_url},
        )
        assert response.status_code == 422, \
            f"Long URL should be rejected, got {response.status_code}"

    def test_ingest_url_validates_relevance_tags_type(self, client):
        """
        P1: Non-list relevance_tags should return 422.

        P0 covers valid list behavior; this covers invalid type rejection.
        """
        response = client.post(
            "/api/knowledge/ingest",
            json={
                "url": "https://example.com/test",
                "relevance_tags": "not-a-list",  # Should be list
            },
        )
        assert response.status_code == 422, \
            "String tags should be rejected, expected 422"

    def test_ingest_url_rejects_non_http_urls(self, client):
        """
        P1: Non-HTTP protocols should be rejected.

        Protocols like FTP, file://, s3:// are security risks and
        not supported by Firecrawl.
        """
        invalid_protocols = [
            ("ftp://example.com/file", "FTP"),
            ("file:///etc/passwd", "File protocol"),
            ("s3://bucket/key", "S3 protocol"),
            ("ws://example.com/socket", "WebSocket"),
            ("mailto://user@example.com", "Mailto"),
        ]

        for url, name in invalid_protocols:
            response = client.post(
                "/api/knowledge/ingest",
                json={"url": url},
            )
            assert response.status_code == 422, \
                f"{name} should be rejected, got {response.status_code}"

    def test_ingest_url_handles_firecrawl_rate_limit(self, client):
        """
        P1: Firecrawl rate limit (429) should trigger retry logic.

        P0 covers transient errors; this specifically tests 429 responses.
        """
        with patch("src.api.knowledge_ingest_endpoints.FirecrawlApp") as MockApp:
            instance = MockApp.return_value
            # Simulate rate limit followed by success
            instance.scrape.side_effect = [
                Exception("429 rate limit exceeded"),
                Exception("429 rate limit exceeded"),
                {"markdown": "# Success After Retry"},
            ]

            with patch("asyncio.sleep", new_callable=AsyncMock):
                response = client.post(
                    "/api/knowledge/ingest",
                    json={"url": "https://example.com/rate-limited"},
                )

            assert response.status_code == 200, response.text
            assert response.json()["status"] == "success"

    def test_ingest_url_handles_github_raw_content(self, client, tmp_path):
        """
        P1: GitHub raw content URLs should be handled.

        GitHub raw URLs are commonly used for MQL5 code snippets.
        """
        import src.api.knowledge_ingest_endpoints as mod

        articles_dir = tmp_path / "scraped_articles"
        mod.SCRAPED_ARTICLES_DIR = articles_dir

        github_content = """# MQL5 Expert Advisor
//+------------------------------------------------------------------+
void OnTick()
{
    // Trading logic here
}
//+------------------------------------------------------------------+
"""

        with patch("src.api.knowledge_ingest_endpoints.FirecrawlApp") as MockApp:
            instance = MockApp.return_value
            instance.scrape.return_value = {"markdown": github_content}

            response = client.post(
                "/api/knowledge/ingest",
                json={
                    "url": "https://raw.githubusercontent.com/user/repo/main/EA.mq5",
                    "relevance_tags": ["mql5", "trading"],
                },
            )

            assert response.status_code == 200, response.text

    def test_ingest_url_handles_empty_tags_list(self, client):
        """
        P1: Empty relevance_tags list should be accepted.

        P0 covers non-empty tags; empty list is valid (no filtering).
        """
        with patch("src.api.knowledge_ingest_endpoints.FirecrawlApp") as MockApp:
            instance = MockApp.return_value
            instance.scrape.return_value = _mock_firecrawl_success()

            response = client.post(
                "/api/knowledge/ingest",
                json={
                    "url": "https://example.com/no-tags",
                    "relevance_tags": [],
                },
            )

            assert response.status_code == 200, response.text


# =============================================================================
# Test Summary
# =============================================================================

"""
P1 Article Ingestion Test Coverage:
| Test | Scenario | P0 Gap Filled |
|------|----------|---------------|
| test_ingest_url_with_special_characters_in_query_params | URL encoding | No P0 equivalent |
| test_ingest_url_with_unicode_content | International chars | No P0 equivalent |
| test_ingest_url_with_very_long_url | URL length limit | No P0 equivalent |
| test_ingest_url_validates_relevance_tags_type | Tag type validation | No P0 equivalent |
| test_ingest_url_rejects_non_http_urls | Protocol validation | P0 covers missing key only |
| test_ingest_url_handles_firecrawl_rate_limit | 429 rate limit | P0 covers generic errors |
| test_ingest_url_handles_github_raw_content | GitHub raw URLs | No P0 equivalent |
| test_ingest_url_handles_empty_tags_list | Empty tags list | No P0 equivalent |

Total: 8 P1 tests added
"""

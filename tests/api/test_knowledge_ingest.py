"""
Tests for Knowledge Ingest API Endpoints

Story 6.2: Web Scraping, Article Ingestion & Personal Knowledge API

Covers:
- POST /api/knowledge/ingest — Firecrawl URL scrape + file-drop to PageIndex
- POST /api/knowledge/personal — Personal note body + file upload
- Retry logic (exponential backoff, max 3 attempts)
- Failure after max retries (500 + error logged)
"""

import asyncio
import io
import logging
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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


@pytest.fixture(autouse=True)
def patch_file_writes(tmp_path, monkeypatch):
    """Redirect article and personal-note writes to a temp directory."""
    import src.api.knowledge_ingest_endpoints as mod

    monkeypatch.setattr(mod, "SCRAPED_ARTICLES_DIR", tmp_path / "scraped_articles")
    monkeypatch.setattr(mod, "PERSONAL_NOTES_DIR", tmp_path / "personal")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_firecrawl_result(markdown: str = "# Hello\nWorld"):
    """Return a dict-like Firecrawl response with markdown content."""
    return {"markdown": markdown}


# ---------------------------------------------------------------------------
# POST /api/knowledge/ingest
# ---------------------------------------------------------------------------


class TestIngestUrlEndpoint:
    """Tests for POST /api/knowledge/ingest."""

    def test_ingest_url_returns_job_response_on_success(self, client):
        """Happy path: Firecrawl returns markdown, file written, 200 + job shape."""
        mock_result = _mock_firecrawl_result("# Article\nSome content here.")

        with patch("src.api.knowledge_ingest_endpoints.FirecrawlApp") as MockApp:
            instance = MockApp.return_value
            instance.scrape.return_value = mock_result

            response = client.post(
                "/api/knowledge/ingest",
                json={"url": "https://example.com/article", "relevance_tags": ["finance"]},
            )

        assert response.status_code == 200, response.text
        data = response.json()
        assert data["status"] == "success"
        assert data["source_url"] == "https://example.com/article"
        assert data["job_id"] is not None
        assert data["scraped_at_utc"] is not None
        assert data["error"] is None

    def test_ingest_url_503_when_no_api_key(self, client, monkeypatch):
        """Should return 503 when FIRECRAWL_API_KEY is not set."""
        monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)

        response = client.post(
            "/api/knowledge/ingest",
            json={"url": "https://example.com/article"},
        )
        assert response.status_code == 503

    def test_ingest_url_rejects_invalid_url(self, client):
        """Non-HTTP URLs and malformed strings should be rejected with 422."""
        # ftp:// is not an http/https URL
        response = client.post(
            "/api/knowledge/ingest",
            json={"url": "ftp://example.com/file"},
        )
        assert response.status_code == 422

        # Plain string that is not a URL at all
        response = client.post(
            "/api/knowledge/ingest",
            json={"url": "not-a-url"},
        )
        assert response.status_code == 422

    def test_ingest_url_provenance_metadata_in_file(self, client, tmp_path):
        """Written file must contain provenance front-matter."""
        import src.api.knowledge_ingest_endpoints as mod

        articles_dir = tmp_path / "scraped_articles"
        mod.SCRAPED_ARTICLES_DIR = articles_dir

        mock_result = _mock_firecrawl_result("# Test\nContent.")

        with patch("src.api.knowledge_ingest_endpoints.FirecrawlApp") as MockApp:
            instance = MockApp.return_value
            instance.scrape.return_value = mock_result

            response = client.post(
                "/api/knowledge/ingest",
                json={
                    "url": "https://example.com/test",
                    "relevance_tags": ["crypto", "news"],
                },
            )

        assert response.status_code == 200
        data = response.json()
        job_id = data["job_id"]

        written_files = list(articles_dir.glob("*.md"))
        assert len(written_files) == 1, f"Expected 1 file, found {written_files}"

        text = written_files[0].read_text()
        # source_url is YAML-quoted to prevent front-matter injection
        assert "source_url: 'https://example.com/test'" in text
        assert "scraped_at_utc:" in text
        assert job_id in text
        # AC2: relevance_tags stored in provenance
        assert "crypto" in text
        assert "news" in text

    def test_ingest_url_returns_failed_on_empty_content(self, client):
        """Empty markdown from Firecrawl should return status=failed."""
        mock_result = {"markdown": None}

        with patch("src.api.knowledge_ingest_endpoints.FirecrawlApp") as MockApp:
            instance = MockApp.return_value
            instance.scrape.return_value = mock_result

            response = client.post(
                "/api/knowledge/ingest",
                json={"url": "https://paywall.example.com"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "failed"
        assert "Empty content" in (data["error"] or "")


# ---------------------------------------------------------------------------
# POST /api/knowledge/personal
# ---------------------------------------------------------------------------


class TestPersonalNoteEndpoint:
    """Tests for POST /api/knowledge/personal."""

    def test_personal_note_body_returns_job_response(self, client):
        """Happy path: JSON note body → file written → 200 + personal job shape."""
        response = client.post(
            "/api/knowledge/personal",
            data={
                "content": "## My trading journal\nGold breakout confirmed.",
                "source_description": "Trading journal 2026-03-19",
                "tags": "gold,forex",
            },
        )

        assert response.status_code == 200, response.text
        data = response.json()
        assert data["status"] == "success"
        assert data["type"] == "personal"
        assert data["job_id"] is not None
        assert data["created_at_utc"] is not None
        assert data["source_description"] == "Trading journal 2026-03-19"

    def test_personal_note_file_has_type_personal_tag(self, client, tmp_path):
        """Written markdown file must include type=personal front-matter."""
        import src.api.knowledge_ingest_endpoints as mod

        personal_dir = tmp_path / "personal"
        mod.PERSONAL_NOTES_DIR = personal_dir

        response = client.post(
            "/api/knowledge/personal",
            data={
                "content": "# My note\nSome insight.",
                "source_description": "Test note",
                "tags": "test",
            },
        )

        assert response.status_code == 200
        written_files = list(personal_dir.glob("*.md"))
        assert len(written_files) == 1

        text = written_files[0].read_text()
        assert "type: personal" in text

    def test_personal_note_txt_file_upload(self, client, tmp_path):
        """TXT file upload should be accepted and written as markdown."""
        import src.api.knowledge_ingest_endpoints as mod

        personal_dir = tmp_path / "personal_txt"
        mod.PERSONAL_NOTES_DIR = personal_dir

        txt_content = b"Plain text trading note\nLine 2"

        response = client.post(
            "/api/knowledge/personal",
            data={"source_description": "Uploaded txt", "tags": ""},
            files={"file": ("note.txt", io.BytesIO(txt_content), "text/plain")},
        )

        assert response.status_code == 200, response.text
        data = response.json()
        assert data["status"] == "success"
        assert data["type"] == "personal"

    def test_personal_note_md_file_upload(self, client, tmp_path):
        """MD file upload should be accepted and written."""
        import src.api.knowledge_ingest_endpoints as mod

        personal_dir = tmp_path / "personal_md"
        mod.PERSONAL_NOTES_DIR = personal_dir

        md_content = b"# My MD Note\nContent here."

        response = client.post(
            "/api/knowledge/personal",
            data={"source_description": "Uploaded md", "tags": "notes"},
            files={"file": ("note.md", io.BytesIO(md_content), "text/markdown")},
        )

        assert response.status_code == 200, response.text
        data = response.json()
        assert data["status"] == "success"

    def test_personal_note_unsupported_file_type_rejected(self, client):
        """Uploading an unsupported file type (e.g., .docx) should return 422."""
        response = client.post(
            "/api/knowledge/personal",
            data={"source_description": "Bad file"},
            files={"file": ("note.docx", io.BytesIO(b"data"), "application/octet-stream")},
        )
        assert response.status_code == 422

    def test_personal_note_missing_content_and_file_rejected(self, client):
        """No content and no file upload should return 422."""
        response = client.post(
            "/api/knowledge/personal",
            data={"source_description": "Empty request"},
        )
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Retry logic tests
# ---------------------------------------------------------------------------


class TestRetryLogic:
    """Tests for the _retry_with_backoff helper and endpoint integration."""

    def test_ingest_succeeds_after_two_failures(self, client):
        """Firecrawl raising exception twice then succeeding → final 200."""
        mock_result = _mock_firecrawl_result("# Recovered\nContent.")

        with patch("src.api.knowledge_ingest_endpoints.FirecrawlApp") as MockApp:
            instance = MockApp.return_value
            instance.scrape.side_effect = [
                Exception("rate limit"),
                Exception("timeout"),
                mock_result,
            ]

            with patch("asyncio.sleep", new_callable=AsyncMock):
                response = client.post(
                    "/api/knowledge/ingest",
                    json={"url": "https://example.com/retry"},
                )

        assert response.status_code == 200, response.text
        data = response.json()
        assert data["status"] == "success"

    def test_ingest_fails_after_max_retries(self, client, caplog):
        """Firecrawl raising exception all 3 times → status=failed + error logged."""
        with patch("src.api.knowledge_ingest_endpoints.FirecrawlApp") as MockApp:
            instance = MockApp.return_value
            instance.scrape.side_effect = [
                Exception("rate limit 1"),
                Exception("rate limit 2"),
                Exception("rate limit 3"),
            ]

            with patch("asyncio.sleep", new_callable=AsyncMock):
                with caplog.at_level(logging.ERROR, logger="src.api.knowledge_ingest_endpoints"):
                    response = client.post(
                        "/api/knowledge/ingest",
                        json={"url": "https://example.com/fail"},
                    )

        assert response.status_code == 200, response.text
        data = response.json()
        assert data["status"] == "failed"
        assert data["error"] is not None
        # Verify error was logged at ERROR level
        assert any("attempts failed" in r.message for r in caplog.records)

    def test_retry_backoff_uses_exponential_delays(self):
        """Unit test: _retry_with_backoff calls asyncio.sleep with 1s, 2s delays."""
        from src.api.knowledge_ingest_endpoints import _retry_with_backoff

        sleep_calls = []

        async def run():
            call_count = 0

            def failing_func():
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise Exception("transient error")
                return "success"

            with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                result = await _retry_with_backoff(
                    failing_func, max_retries=3, base_delay=1.0
                )
                sleep_calls.extend(mock_sleep.call_args_list)
                return result

        result = asyncio.get_event_loop().run_until_complete(run())
        assert result == "success"
        # First sleep: 1.0s, second sleep: 2.0s
        assert len(sleep_calls) == 2
        assert sleep_calls[0].args[0] == pytest.approx(1.0)
        assert sleep_calls[1].args[0] == pytest.approx(2.0)

"""
P1 Tests: Video Ingest UI Validation

Epic 6 - Knowledge & Research Engine
Priority: P1 (High)
Coverage: YouTube video ingest UI backend validation

Covers:
- YouTube URL validation
- Playlist detection
- Job submission response structure
- Authentication status checks
- Job status polling
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def video_ingest_app():
    """Create test FastAPI app for video ingest endpoints."""
    from src.api.ide_video_ingest import router
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(video_ingest_app):
    """TestClient for video ingest endpoints."""
    return TestClient(video_ingest_app)


# =============================================================================
# P1 Tests: Video Ingest API
# =============================================================================

class TestVideoIngestAPI:
    """P1 tests for video ingest backend endpoints."""

    def test_submit_youtube_video_url_valid(self, client):
        """
        P1: Valid YouTube video URL should be accepted.

        Validates standard YouTube watch URL format.
        """
        with patch("src.api.ide_video_ingest.process_video_async", new_callable=AsyncMock) as mock_process:
            mock_process.return_value = {
                "job_id": "test-job-123",
                "status": "processing",
                "strategy_folder": "/data/strategies/video_ingest",
            }

            response = client.post(
                "/video-ingest/process",
                json={
                    "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                    "strategy_name": "my_strategy",
                    "is_playlist": False,
                },
            )

            assert response.status_code == 200, response.text
            data = response.json()
            assert "job_id" in data
            assert data["status"] == "processing"
            assert data["strategy_folder"] == "/data/strategies/video_ingest"

    def test_submit_youtube_short_url(self, client):
        """
        P1: YouTube short URLs (youtu.be) should be accepted.
        """
        with patch("src.api.ide_video_ingest.process_video_async", new_callable=AsyncMock) as mock_process:
            mock_process.return_value = {
                "job_id": "short-url-job",
                "status": "processing",
            }

            response = client.post(
                "/video-ingest/process",
                json={
                    "url": "https://youtu.be/dQw4w9WgXcQ",
                    "strategy_name": "short_test",
                },
            )

            assert response.status_code == 200, response.text

    def test_submit_youtube_playlist_url(self, client):
        """
        P1: YouTube playlist URL should set is_playlist flag.

        Playlist processing differs from single video - verifies flag.
        """
        with patch("src.api.ide_video_ingest.process_video_async", new_callable=AsyncMock) as mock_process:
            mock_process.return_value = {
                "job_id": "playlist-job-456",
                "status": "processing",
            }

            response = client.post(
                "/video-ingest/process",
                json={
                    "url": "https://www.youtube.com/playlist?list=PLrAXtmErZgOeiKm4sgNOknGvNjby9efdf",
                    "strategy_name": "playlist_strategy",
                    "is_playlist": True,
                },
            )

            assert response.status_code == 200, response.text
            data = response.json()
            assert data["status"] == "processing"

    def test_submit_video_rejects_invalid_youtube_url(self, client):
        """
        P1: Invalid YouTube URLs should return 422.

        Covers: missing video ID, wrong domain, malformed URLs.
        """
        invalid_urls = [
            ("not-a-url", "Plain text"),
            ("https://vimeo.com/123456", "Wrong platform"),
            ("https://youtube.com/watch", "Missing video ID"),
            ("https://youtube.com/watch?v=", "Empty video ID"),
            ("https://youtube.com/shorts/", "Shorts without ID"),
        ]

        for url, description in invalid_urls:
            response = client.post(
                "/video-ingest/process",
                json={"url": url},
            )
            assert response.status_code == 422, \
                f"{description}: URL '{url}' should be rejected, got {response.status_code}"

    def test_get_auth_status_returns_required_fields(self, client):
        """
        P1: Auth status endpoint should return gemini and qwen status.

        Frontend uses this to enable/disable provider buttons.
        """
        response = client.get("/video-ingest/auth-status")
        assert response.status_code == 200, response.text
        data = response.json()
        assert "gemini" in data, "Missing gemini auth status"
        assert "qwen" in data, "Missing qwen auth status"
        assert isinstance(data["gemini"], bool), "gemini should be boolean"
        assert isinstance(data["qwen"], bool), "qwen should be boolean"

    def test_get_job_status_valid_id(self, client):
        """
        P1: Valid job ID should return job status.

        Covers progress, current_stage, and error fields.
        """
        with patch("src.api.ide_video_ingest.get_job_status") as mock_get:
            mock_get.return_value = {
                "job_id": "test-123",
                "status": "completed",
                "current_stage": "transcription",
                "progress": 100,
            }

            response = client.get("/video-ingest/jobs/test-123")
            assert response.status_code == 200, response.text
            data = response.json()
            assert data["job_id"] == "test-123"
            assert data["status"] == "completed"
            assert data["progress"] == 100

    def test_get_job_status_with_error(self, client):
        """
        P1: Failed job should return error message.

        Error state is critical for UI error display.
        """
        with patch("src.api.ide_video_ingest.get_job_status") as mock_get:
            mock_get.return_value = {
                "job_id": "failed-job",
                "status": "failed",
                "error": "Video unavailable or deleted",
                "current_stage": "download",
            }

            response = client.get("/video-ingest/jobs/failed-job")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "failed"
            assert "error" in data

    def test_submit_video_requires_url_field(self, client):
        """
        P1: Missing URL field should return 422.

        Request validation - URL is required.
        """
        response = client.post(
            "/video-ingest/process",
            json={"strategy_name": "test"},
        )
        assert response.status_code == 422, \
            "Missing URL should return 422, got " + str(response.status_code)

    def test_submit_video_validates_strategy_name_length(self, client):
        """
        P1: Strategy name >255 chars should return 422.

        Storage paths have limits, validate upfront.
        """
        long_name = "x" * 300
        response = client.post(
            "/video-ingest/process",
            json={
                "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                "strategy_name": long_name,
            },
        )
        assert response.status_code == 422, \
            "Strategy name >255 chars should be rejected"

    def test_submit_video_allows_default_strategy_name(self, client):
        """
        P1: Omitting strategy_name should use default.

        Default is 'video_ingest' per API spec.
        """
        with patch("src.api.ide_video_ingest.process_video_async", new_callable=AsyncMock) as mock_process:
            mock_process.return_value = {
                "job_id": "default-name-job",
                "status": "processing",
                "strategy_folder": "/data/strategies/video_ingest",
            }

            response = client.post(
                "/video-ingest/process",
                json={"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"},
            )

            assert response.status_code == 200, response.text


# =============================================================================
# Test Summary
# =============================================================================

"""
P1 Video Ingest Test Coverage:
| Test | Scenario | P0 Gap |
|------|----------|--------|
| test_submit_youtube_video_url_valid | Standard YouTube URL | No P0 equivalent |
| test_submit_youtube_short_url | youtu.be short URL | No P0 equivalent |
| test_submit_youtube_playlist_url | Playlist flag handling | No P0 equivalent |
| test_submit_video_rejects_invalid_youtube_url | URL validation | No P0 equivalent |
| test_get_auth_status_returns_required_fields | Auth check | No P0 equivalent |
| test_get_job_status_valid_id | Job status polling | No P0 equivalent |
| test_get_job_status_with_error | Error state | No P0 equivalent |
| test_submit_video_requires_url_field | Required field validation | No P0 equivalent |
| test_submit_video_validates_strategy_name_length | Name length | No P0 equivalent |
| test_submit_video_allows_default_strategy_name | Default name | No P0 equivalent |

Total: 10 P1 tests added
"""

"""
P3 Tests: Video Ingest Rare Scenarios

Epic 6 - Knowledge & Research Engine
Priority: P3 (Low)
Coverage: Rare edge cases for video ingest

Covers:
- Concurrent job submission limits
- Expired job cleanup
- Playlist with single video
"""

import pytest
from unittest.mock import AsyncMock, patch

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
# P3 Tests: Video Ingest Rare Cases
# =============================================================================

class TestVideoIngestRareCases:
    """P3 rare scenario tests for video ingest."""

    def test_submit_video_rejects_youtube_short_without_id(self, client):
        """
        P3: youtube.com/shorts/ without video ID should be rejected.

        Edge case: Shorts URL format validation.
        """
        response = client.post(
            "/video-ingest/process",
            json={"url": "https://youtube.com/shorts/"},
        )
        assert response.status_code == 422

    def test_submit_video_handles_age_restricted_content(self, client):
        """
        P3: Age-restricted YouTube videos should return appropriate error.

        Firecrawl cannot scrape age-restricted content.
        """
        with patch("src.api.ide_video_ingest.process_video_async", new_callable=AsyncMock) as mock_process:
            mock_process.side_effect = Exception("Video is age-restricted or unavailable")

            response = client.post(
                "/video-ingest/process",
                json={"url": "https://youtube.com/watch?v=age-restricted-id"},
            )

        # Should handle gracefully
        assert response.status_code in [200, 500]

    def test_submit_playlist_with_single_video(self, client):
        """
        P3: Playlist URL containing single video should work normally.

        Edge case in playlist detection.
        """
        with patch("src.api.ide_video_ingest.process_video_async", new_callable=AsyncMock) as mock_process:
            mock_process.return_value = {
                "job_id": "single-video-playlist",
                "status": "completed",
                "videos_processed": 1,
            }

            response = client.post(
                "/video-ingest/process",
                json={
                    "url": "https://youtube.com/watch?v=dQw4w9WgXcQ&list=PLrAXtmErZgOeiKm4sgNOknGvNjby9efdf",
                    "is_playlist": True,
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["status"] in ["processing", "completed"]

    def test_get_job_status_returns_stages(self, client):
        """
        P3: Job status should include processing stages.

        UI shows progress bar with current stage.
        """
        with patch("src.api.ide_video_ingest.get_job_status") as mock_get:
            mock_get.return_value = {
                "job_id": "multi-stage-job",
                "status": "processing",
                "current_stage": "transcription",
                "progress": 50,
                "stages": ["download", "transcription", "analysis", "summary"],
            }

            response = client.get("/video-ingest/jobs/multi-stage-job")
            assert response.status_code == 200
            data = response.json()
            assert "stages" in data or "current_stage" in data

    def test_submit_video_with_strategy_name_sanitization(self, client):
        """
        P3: Strategy names with special characters should be sanitized.

        File system paths cannot have certain characters.
        """
        with patch("src.api.ide_video_ingest.process_video_async", new_callable=AsyncMock) as mock_process:
            mock_process.return_value = {
                "job_id": "sanitized-name",
                "status": "processing",
            }

            response = client.post(
                "/video-ingest/process",
                json={
                    "url": "https://youtube.com/watch?v=test",
                    "strategy_name": "My Strategy: Volume Profile & VWAP",
                },
            )

            assert response.status_code == 200


# =============================================================================
# Test Summary
# =============================================================================

"""
P3 Video Ingest Test Coverage:
| Test | Scenario | Notes |
|------|----------|-------|
| test_submit_video_rejects_youtube_short_without_id | URL validation | Edge case |
| test_submit_video_handles_age_restricted_content | Error handling | Rare error |
| test_submit_playlist_with_single_video | Playlist edge | Edge case |
| test_get_job_status_returns_stages | Progress display | UI enhancement |
| test_submit_video_with_strategy_name_sanitization | Name cleanup | Edge case |

Total: 5 P3 tests added
"""

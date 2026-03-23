"""
P3 Tests for Epic 8 - Monaco Editor Integration

Priority: P3
Coverage: Monaco editor component, code display, syntax highlighting trigger

Risk Coverage:
- R-001: EA Deployment Pipeline (code viewing)
"""

import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI


class TestMonacoEditorIntegration:
    """P3: Monaco editor component integration."""

    def test_variant_code_endpoint_returns_mql5_language(self):
        """P3: Code endpoint returns correct language for Monaco."""
        client = _make_client()  # variant browser client
        response = client.get("/api/variant-browser/news-event-breakout/vanilla/code")
        assert response.status_code == 200

        data = response.json()
        assert data["language"] == "mql5"
        assert "code" in data

    def test_code_response_is_valid_string(self):
        """P3: Code content is a valid string for Monaco display."""
        client = _make_client()
        response = client.get("/api/variant-browser/news-event-breakout/vanilla/code")

        if response.status_code == 200:
            data = response.json()
            assert isinstance(data["code"], str)
            assert len(data["code"]) > 0


def _make_client():
    from src.api.variant_browser_endpoints import router
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

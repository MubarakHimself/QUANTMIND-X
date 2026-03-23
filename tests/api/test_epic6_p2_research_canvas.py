"""
P2 Tests: Research Canvas Knowledge Query

Epic 6 - Knowledge & Research Engine
Priority: P2 (Medium)
Coverage: Research canvas knowledge query functionality

Covers:
- Collection-specific search
- Source filtering
- Result ranking validation
- Empty query handling
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def knowledge_app():
    """Create a minimal FastAPI app with the unified knowledge router."""
    from src.api.knowledge_endpoints import router
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def knowledge_client(knowledge_app):
    """TestClient for the knowledge endpoints app."""
    return TestClient(knowledge_app)


# =============================================================================
# Helper to patch the router's client search_async
# =============================================================================

def _patch_router_search_async(mock_results):
    """
    Patch kb_router.client.search_async with the given results.

    kb_router is a singleton instantiated at module load time, so we must
    patch the instance method directly (not the PageIndexClient class).

    Args:
        mock_results: Either a single value (returned for every call) or a list
                      (each call returns the next item via side_effect).
    """
    from src.api import knowledge_endpoints
    kb_router = knowledge_endpoints._kb_router
    if isinstance(mock_results, list) and mock_results and isinstance(mock_results[0], list):
        # Nested list - use side_effect to iterate across calls
        mock_method = AsyncMock(side_effect=iter(mock_results))
    else:
        # Single result or empty list - return the same value each call
        mock_method = AsyncMock(return_value=mock_results)
    return patch.object(kb_router.client, 'search_async', mock_method)


# =============================================================================
# P2 Tests: Research Canvas Knowledge Query
# =============================================================================

class TestResearchCanvasKnowledgeQuery:
    """P2 tests for research canvas knowledge query beyond P0 scope."""

    def test_search_articles_collection_only(self, knowledge_client):
        """
        P2: Searching only articles collection should return only those results.

        Research canvas allows filtering by source type.
        """
        mock_results = [
            {
                "content": "Research paper content about trading",
                "source": "articles/trading-paper.pdf",
                "score": 0.95,
            }
        ]
        with _patch_router_search_async(mock_results) as mock_search:
            response = knowledge_client.post(
                "/api/knowledge/search",
                json={
                    "query": "trading strategy backtesting",
                    "sources": ["articles"],
                    "limit": 5,
                },
            )

            assert response.status_code == 200, response.text
            data = response.json()

            # Should only query articles
            mock_search.assert_called_once_with(
                "trading strategy backtesting", "articles", 5
            )
            assert len(data["results"]) == 1

    def test_search_returns_results_sorted_by_relevance(self, knowledge_client):
        """
        P2: Search results should be sorted by relevance score descending.

        Canvas displays highest relevance first.
        """
        # Return out of order results from different sources
        mock_results = [
            [{"content": "Low", "source": "low.pdf", "score": 0.3}],
            [{"content": "High", "source": "high.pdf", "score": 0.95}],
            [{"content": "Medium", "source": "med.pdf", "score": 0.6}],
        ]
        with _patch_router_search_async(mock_results) as mock_search:
            response = knowledge_client.post(
                "/api/knowledge/search",
                json={"query": "trading"},
            )

            assert response.status_code == 200
            results = response.json()["results"]

            # Verify sorted by score descending
            scores = [r["relevance_score"] for r in results]
            assert scores == sorted(scores, reverse=True)

    def test_search_with_empty_query_returns_error(self, knowledge_client):
        """
        P2: Empty query string should return 422.

        Validation before hitting search backend.
        """
        response = knowledge_client.post(
            "/api/knowledge/search",
            json={"query": ""},
        )
        assert response.status_code == 422, \
            f"Empty query should return 422, got {response.status_code}"

    def test_search_unknown_collection_returns_warning(self, knowledge_client):
        """
        P2: Unknown collection should be warned and filtered out.

        Graceful handling of typos in source names.
        """
        with _patch_router_search_async([]) as mock_search:
            response = knowledge_client.post(
                "/api/knowledge/search",
                json={
                    "query": "test query",
                    "sources": ["articles", "typo_source", "books"],
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert len(data["warnings"]) >= 1
            assert any("typo_source" in w for w in data["warnings"])

    def test_search_limit_validation_max_100(self, knowledge_client):
        """
        P2: Limit > 100 should be rejected with 422.

        Prevents excessive result sets.
        """
        response = knowledge_client.post(
            "/api/knowledge/search",
            json={"query": "test", "limit": 150},
        )
        assert response.status_code == 422, \
            f"Limit > 100 should return 422, got {response.status_code}"

    def test_search_limit_validation_min_1(self, knowledge_client):
        """
        P2: Limit < 1 should be rejected with 422.

        Must request at least 1 result.
        """
        response = knowledge_client.post(
            "/api/knowledge/search",
            json={"query": "test", "limit": 0},
        )
        assert response.status_code == 422

    def test_search_respects_limit_per_source(self, knowledge_client):
        """
        P2: Each source should return up to limit results.

        Total results = limit * sources queried.
        """
        # Return 3 items from articles and 3 from books (matching limit=3)
        mock_results = [
            [
                {"content": f"art{i}", "source": f"a{i}.pdf", "score": 0.9 - i * 0.1}
                for i in range(3)
            ],
            [
                {"content": f"book{i}", "source": f"b{i}.pdf", "score": 0.9 - i * 0.1}
                for i in range(3)
            ],
        ]
        with _patch_router_search_async(mock_results) as mock_search:
            response = knowledge_client.post(
                "/api/knowledge/search",
                json={"query": "test", "sources": ["articles", "books"], "limit": 3},
            )

            assert response.status_code == 200
            # Total should be 6 (3 from each of 2 sources)
            assert len(response.json()["results"]) <= 6

    def test_search_all_three_collections(self, knowledge_client):
        """
        P2: Searching without sources should query all 3 collections.

        Default behavior for research canvas.
        """
        with _patch_router_search_async([]) as mock_search:
            response = knowledge_client.post(
                "/api/knowledge/search",
                json={"query": "trading strategy"},
            )

            assert response.status_code == 200
            # Should have been called 3 times (articles, books, logs)
            assert mock_search.call_count == 3

    def test_search_books_and_logs_only(self, knowledge_client):
        """
        P2: Can filter to just books and logs collections.

        Used when articles are known to be irrelevant.
        """
        with _patch_router_search_async([]) as mock_search:
            response = knowledge_client.post(
                "/api/knowledge/search",
                json={
                    "query": "backtesting methods",
                    "sources": ["books", "logs"],
                },
            )

            assert response.status_code == 200
            # Should have been called 2 times (books, logs)
            assert mock_search.call_count == 2

    def test_search_includes_provenance_data(self, knowledge_client):
        """
        P2: Results should include provenance for source verification.

        Research canvas shows source attribution.
        """
        mock_results = [
            {
                "content": "Content from PDF",
                "source": "articles/strategy.pdf",
                "score": 0.9,
            }
        ]
        with _patch_router_search_async(mock_results) as mock_search:
            response = knowledge_client.post(
                "/api/knowledge/search",
                json={"query": "strategy", "sources": ["articles"]},
            )

            assert response.status_code == 200
            data = response.json()
            assert len(data["results"]) == 1

            result = data["results"][0]
            assert "provenance" in result
            assert "source_url" in result["provenance"]


# =============================================================================
# Test Summary
# =============================================================================

"""
P2 Research Canvas Test Coverage:
| Test | Scenario | P0 Gap |
|------|----------|--------|
| test_search_articles_collection_only | Source filter | P0 covers all 3 |
| test_search_returns_results_sorted_by_relevance | Sorting | Not covered |
| test_search_with_empty_query_returns_error | Validation | Not covered |
| test_search_unknown_collection_returns_warning | Error handling | Not covered |
| test_search_limit_validation_max_100 | Validation | Not covered |
| test_search_limit_validation_min_1 | Validation | Not covered |
| test_search_respects_limit_per_source | Limit | Not covered |
| test_search_all_three_collections | Default behavior | P0 covers fanout |
| test_search_books_and_logs_only | Source filter | Not covered |
| test_search_includes_provenance_data | Attribution | Not covered |

Total: 10 P2 tests added
"""

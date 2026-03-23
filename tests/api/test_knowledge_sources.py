"""
Tests for Unified Knowledge API Endpoints

Story 6.1: PageIndex Integration & Knowledge API

Covers:
- AC1: GET /api/knowledge/sources returns all 3 sources with required fields
- AC2: POST /api/knowledge/search fans out to all instances
- AC3: Results are merged and ranked by relevance_score
- AC4: Each result has { source_type, title, excerpt, relevance_score, provenance }
- AC5: Offline instance is skipped with warning; remaining results returned
"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def test_app():
    """Create a minimal FastAPI app with only the unified knowledge router."""
    from src.api.knowledge_endpoints import router

    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(test_app):
    """TestClient for the knowledge endpoints app."""
    return TestClient(test_app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_health_ok() -> dict:
    return {"status": "healthy"}


def _make_stats(count: int) -> dict:
    return {"document_count": count, "status": "ok"}


def _fake_search_results(collection: str, score: float = 0.7) -> list:
    return [
        {
            "content": f"content from {collection}",
            "source": f"{collection}_doc.pdf",
            "score": score,
            "collection": collection,
        }
    ]


# ---------------------------------------------------------------------------
# Tests: GET /api/knowledge/sources  (AC1)
# ---------------------------------------------------------------------------


class TestGetKnowledgeSources:
    """Test GET /api/knowledge/sources — AC1, AC5."""

    def test_sources_returns_list_of_three(self, client):
        """Should return a list with exactly 3 entries."""
        health_mock = AsyncMock(return_value=_make_health_ok())
        stats_mock = AsyncMock(return_value=_make_stats(10))
        with patch.object(
            __import__(
                "src.agents.knowledge.router", fromlist=["PageIndexClient"]
            ).PageIndexClient,
            "health_check_async",
            health_mock,
        ), patch.object(
            __import__(
                "src.agents.knowledge.router", fromlist=["PageIndexClient"]
            ).PageIndexClient,
            "get_stats_async",
            stats_mock,
        ):
            response = client.get("/api/knowledge/sources")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 3

    def test_sources_contains_required_fields(self, client):
        """Each source entry must have id, type, status, document_count."""
        from src.agents.knowledge.router import PageIndexClient

        health_mock = AsyncMock(return_value=_make_health_ok())
        stats_mock = AsyncMock(return_value=_make_stats(42))
        with patch.object(PageIndexClient, "health_check_async", health_mock), \
             patch.object(PageIndexClient, "get_stats_async", stats_mock):
            response = client.get("/api/knowledge/sources")

        assert response.status_code == 200
        data = response.json()
        expected_ids = {"articles", "books", "logs"}
        returned_ids = {item["id"] for item in data}
        assert returned_ids == expected_ids

        for item in data:
            assert "id" in item
            assert "type" in item
            assert "status" in item
            assert "document_count" in item

    def test_sources_document_count_from_stats(self, client):
        """document_count should reflect the value from /stats response."""
        from src.agents.knowledge.router import PageIndexClient

        counts = {"articles": 100, "books": 200, "logs": 300}

        async def _mock_stats(self_arg, collection):
            return _make_stats(counts[collection])

        health_mock = AsyncMock(return_value=_make_health_ok())
        with patch.object(PageIndexClient, "health_check_async", health_mock), \
             patch.object(PageIndexClient, "get_stats_async", _mock_stats):
            response = client.get("/api/knowledge/sources")

        assert response.status_code == 200
        data = response.json()
        result_map = {item["id"]: item for item in data}
        assert result_map["articles"]["document_count"] == 100
        assert result_map["books"]["document_count"] == 200
        assert result_map["logs"]["document_count"] == 300

    def test_sources_online_status_from_health(self, client):
        """Instances with healthy response should report status='online'."""
        from src.agents.knowledge.router import PageIndexClient

        health_mock = AsyncMock(return_value=_make_health_ok())
        stats_mock = AsyncMock(return_value=_make_stats(5))
        with patch.object(PageIndexClient, "health_check_async", health_mock), \
             patch.object(PageIndexClient, "get_stats_async", stats_mock):
            response = client.get("/api/knowledge/sources")

        assert response.status_code == 200
        data = response.json()
        for item in data:
            assert item["status"] == "online"

    def test_sources_offline_when_raises(self, client):
        """An instance that raises an exception should appear as offline. (AC5)"""
        from src.agents.knowledge.router import PageIndexClient

        async def _failing_health(self_arg, collection):
            if collection == "books":
                raise ConnectionError("books instance unreachable")
            return _make_health_ok()

        async def _failing_stats(self_arg, collection):
            if collection == "books":
                raise ConnectionError("books instance unreachable")
            return _make_stats(10)

        with patch.object(PageIndexClient, "health_check_async", _failing_health), \
             patch.object(PageIndexClient, "get_stats_async", _failing_stats):
            response = client.get("/api/knowledge/sources")

        assert response.status_code == 200
        data = response.json()
        result_map = {item["id"]: item for item in data}
        assert result_map["books"]["status"] == "offline"
        assert result_map["books"]["document_count"] == 0
        # Online instances should still show up correctly
        assert result_map["articles"]["status"] == "online"
        assert result_map["logs"]["status"] == "online"


# ---------------------------------------------------------------------------
# Tests: POST /api/knowledge/search  (AC2, AC3, AC4, AC5)
# ---------------------------------------------------------------------------


class TestSearchKnowledge:
    """Test POST /api/knowledge/search — AC2, AC3, AC4, AC5."""

    def test_search_fans_out_to_all_instances(self, client):
        """Search should query all 3 collections by default. (AC2)"""
        from src.agents.knowledge.router import PageIndexClient

        called_collections = []

        async def _mock_search(self_arg, query, collection, limit=5):
            called_collections.append(collection)
            return []

        with patch.object(PageIndexClient, "search_async", _mock_search):
            response = client.post(
                "/api/knowledge/search",
                json={"query": "momentum strategy"},
            )

        assert response.status_code == 200
        assert set(called_collections) == {"articles", "books", "logs"}

    def test_search_returns_required_fields(self, client):
        """Each result must have source_type, title, excerpt, relevance_score, provenance. (AC4)"""
        from src.agents.knowledge.router import PageIndexClient

        async def _mock_search(self_arg, query, collection, limit=5):
            if collection == "articles":
                return _fake_search_results("articles", 0.92)
            return []

        with patch.object(PageIndexClient, "search_async", _mock_search):
            response = client.post(
                "/api/knowledge/search",
                json={"query": "momentum"},
            )

        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) >= 1
        result = data["results"][0]
        assert "source_type" in result
        assert "title" in result
        assert "excerpt" in result
        assert "relevance_score" in result
        assert "provenance" in result

    def test_search_results_sorted_by_relevance_descending(self, client):
        """Merged results must be sorted by relevance_score descending. (AC3)"""
        from src.agents.knowledge.router import PageIndexClient

        score_map = {"articles": 0.5, "books": 0.9, "logs": 0.3}

        async def _mock_search(self_arg, query, collection, limit=5):
            return _fake_search_results(collection, score_map[collection])

        with patch.object(PageIndexClient, "search_async", _mock_search):
            response = client.post(
                "/api/knowledge/search",
                json={"query": "test query"},
            )

        assert response.status_code == 200
        data = response.json()
        scores = [r["relevance_score"] for r in data["results"]]
        assert scores == sorted(scores, reverse=True), "Results must be sorted descending"
        assert scores[0] == pytest.approx(0.9)

    def test_search_provenance_structure(self, client):
        """Provenance block must contain source_url, source_type, indexed_at_utc. (AC4)"""
        from src.agents.knowledge.router import PageIndexClient

        async def _mock_search(self_arg, query, collection, limit=5):
            return _fake_search_results(collection, 0.7)

        with patch.object(PageIndexClient, "search_async", _mock_search):
            response = client.post(
                "/api/knowledge/search",
                json={"query": "test"},
            )

        assert response.status_code == 200
        data = response.json()
        for result in data["results"]:
            provenance = result["provenance"]
            assert "source_url" in provenance
            assert "source_type" in provenance
            assert "indexed_at_utc" in provenance

    def test_search_graceful_degradation_on_offline_instance(self, client):
        """Offline instance skipped with warning; remaining results returned. (AC5)"""
        from src.agents.knowledge.router import PageIndexClient

        async def _mock_search(self_arg, query, collection, limit=5):
            if collection == "articles":
                raise ConnectionError("articles service down")
            return _fake_search_results(collection, 0.6)

        with patch.object(PageIndexClient, "search_async", _mock_search):
            response = client.post(
                "/api/knowledge/search",
                json={"query": "test"},
            )

        assert response.status_code == 200
        data = response.json()

        # Warnings must mention the offline instance
        assert len(data["warnings"]) == 1
        assert "articles" in data["warnings"][0].lower()

        # Results should still include books and logs
        returned_sources = {r["source_type"] for r in data["results"]}
        assert "books" in returned_sources
        assert "logs" in returned_sources
        assert "articles" not in returned_sources

    def test_search_all_instances_offline_returns_empty_with_warnings(self, client):
        """When all instances are offline, empty results with 3 warnings returned."""
        from src.agents.knowledge.router import PageIndexClient

        async def _mock_search(self_arg, query, collection, limit=5):
            raise ConnectionError(f"{collection} is down")

        with patch.object(PageIndexClient, "search_async", _mock_search):
            response = client.post(
                "/api/knowledge/search",
                json={"query": "test"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["results"] == []
        assert data["total"] == 0
        assert len(data["warnings"]) == 3

    def test_search_with_sources_filter(self, client):
        """When sources filter is provided, only those collections are queried."""
        from src.agents.knowledge.router import PageIndexClient

        called_collections = []

        async def _mock_search(self_arg, query, collection, limit=5):
            called_collections.append(collection)
            return []

        with patch.object(PageIndexClient, "search_async", _mock_search):
            response = client.post(
                "/api/knowledge/search",
                json={"query": "test", "sources": ["articles", "books"]},
            )

        assert response.status_code == 200
        assert set(called_collections) == {"articles", "books"}
        assert "logs" not in called_collections

    def test_search_response_contains_query_echo(self, client):
        """Response should echo back the original query string."""
        from src.agents.knowledge.router import PageIndexClient

        async def _mock_search(self_arg, query, collection, limit=5):
            return []

        with patch.object(PageIndexClient, "search_async", _mock_search):
            response = client.post(
                "/api/knowledge/search",
                json={"query": "specific query text"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "specific query text"

    def test_search_requires_query_field(self, client):
        """Missing query field should return a 422 validation error."""
        response = client.post(
            "/api/knowledge/search",
            json={"limit": 5},
        )
        assert response.status_code == 422

    def test_search_total_reflects_merged_result_count(self, client):
        """total field should equal len(results)."""
        from src.agents.knowledge.router import PageIndexClient

        async def _mock_search(self_arg, query, collection, limit=5):
            return _fake_search_results(collection, 0.5)

        with patch.object(PageIndexClient, "search_async", _mock_search):
            response = client.post(
                "/api/knowledge/search",
                json={"query": "test"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == len(data["results"])

    def test_search_unknown_source_produces_warning(self, client):
        """Unknown source names in the 'sources' filter should produce a warning."""
        from src.agents.knowledge.router import PageIndexClient

        async def _mock_search(self_arg, query, collection, limit=5):
            return _fake_search_results(collection, 0.5)

        with patch.object(PageIndexClient, "search_async", _mock_search):
            response = client.post(
                "/api/knowledge/search",
                json={"query": "test", "sources": ["articles", "nonexistent"]},
            )

        assert response.status_code == 200
        data = response.json()
        # nonexistent should appear in warnings
        assert any("nonexistent" in w for w in data["warnings"])
        # articles should still be queried
        returned_sources = {r["source_type"] for r in data["results"]}
        assert "articles" in returned_sources

    def test_search_empty_query_rejected(self, client):
        """Empty query string should return a 422 validation error."""
        response = client.post(
            "/api/knowledge/search",
            json={"query": ""},
        )
        assert response.status_code == 422

    def test_search_limit_zero_rejected(self, client):
        """limit=0 should return a 422 validation error."""
        response = client.post(
            "/api/knowledge/search",
            json={"query": "test", "limit": 0},
        )
        assert response.status_code == 422

    def test_search_limit_above_max_rejected(self, client):
        """limit=101 should return a 422 validation error."""
        response = client.post(
            "/api/knowledge/search",
            json={"query": "test", "limit": 101},
        )
        assert response.status_code == 422


class TestGetKnowledgeSourcesExtra:
    """Additional edge case tests for GET /api/knowledge/sources."""

    def test_sources_document_count_fallback_to_count_key(self, client):
        """document_count should fall back to 'count' key when 'document_count' absent."""
        from src.agents.knowledge.router import PageIndexClient

        async def _mock_stats_with_count_key(self_arg, collection):
            # Return stats using 'count' key instead of 'document_count'
            return {"count": 77, "status": "ok"}

        health_mock = AsyncMock(return_value=_make_health_ok())
        with patch.object(PageIndexClient, "health_check_async", health_mock), \
             patch.object(PageIndexClient, "get_stats_async", _mock_stats_with_count_key):
            response = client.get("/api/knowledge/sources")

        assert response.status_code == 200
        data = response.json()
        for item in data:
            assert item["document_count"] == 77

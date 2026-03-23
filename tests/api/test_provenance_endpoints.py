"""
Tests for Provenance Chain API Endpoints

Tests provenance model structures and endpoint logic for EA origin tracking.
"""
import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import FastAPI


def _make_client():
    from src.api.provenance_endpoints import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


class TestProvenanceModels:
    """Test provenance chain model structures."""

    def test_provenance_node_required_fields(self):
        from src.api.provenance_endpoints import ProvenanceNode

        node = ProvenanceNode(
            stage="research",
            timestamp="2026-03-21T10:00:00Z",
            actor="Research Department",
            status="completed",
        )
        assert node.stage == "research"
        assert node.status == "completed"
        assert node.details == {}

    def test_provenance_node_with_details(self):
        from src.api.provenance_endpoints import ProvenanceNode

        node = ProvenanceNode(
            stage="source",
            timestamp="2026-03-21T09:00:00Z",
            actor="Video Ingest",
            status="completed",
            details={"url": "https://youtube.com/watch?v=test", "source_type": "youtube"},
        )
        assert node.details["source_type"] == "youtube"

    def test_provenance_chain_structure(self):
        from src.api.provenance_endpoints import ProvenanceChain, ProvenanceNode

        chain = ProvenanceChain(
            strategy_id="strat-001",
            version_tag="1.0.0",
            chain=[
                ProvenanceNode(
                    stage="source",
                    timestamp="2026-03-21T09:00:00Z",
                    actor="YouTube",
                    status="completed",
                ),
                ProvenanceNode(
                    stage="research",
                    timestamp="2026-03-21T10:00:00Z",
                    actor="Research Dept",
                    status="completed",
                ),
            ],
            total_stages=2,
        )
        assert chain.strategy_id == "strat-001"
        assert chain.total_stages == 2
        assert chain.source_url is None

    def test_provenance_query_request(self):
        from src.api.provenance_endpoints import ProvenanceQueryRequest

        req = ProvenanceQueryRequest(
            query="Where did this EA come from?",
            strategy_id="strat-001",
        )
        assert req.query == "Where did this EA come from?"
        assert req.strategy_id == "strat-001"

    def test_provenance_query_response_structure(self):
        from src.api.provenance_endpoints import ProvenanceQueryResponse

        resp = ProvenanceQueryResponse(
            answer="This EA originated from YouTube...",
            chain=None,
            confidence=0.9,
        )
        assert resp.confidence == 0.9
        assert resp.chain is None


class TestProvenanceService:
    """Test ProvenanceService logic."""

    @pytest.mark.asyncio
    async def test_get_provenance_chain_returns_5_stages(self):
        from src.api.provenance_endpoints import ProvenanceService

        svc = ProvenanceService()
        chain = await svc.get_provenance_chain("strat-001")

        assert chain.strategy_id == "strat-001"
        assert chain.total_stages == 5
        stages = [n.stage for n in chain.chain]
        assert "source" in stages
        assert "research" in stages
        assert "dev" in stages
        assert "review" in stages
        assert "approval" in stages

    @pytest.mark.asyncio
    async def test_provenance_chain_default_version(self):
        from src.api.provenance_endpoints import ProvenanceService

        svc = ProvenanceService()
        chain = await svc.get_provenance_chain("strat-001")
        assert chain.version_tag == "1.0.0"

    @pytest.mark.asyncio
    async def test_provenance_chain_custom_version(self):
        from src.api.provenance_endpoints import ProvenanceService

        svc = ProvenanceService()
        chain = await svc.get_provenance_chain("strat-001", version_tag="2.0.0")
        assert chain.version_tag == "2.0.0"

    @pytest.mark.asyncio
    async def test_query_provenance_origin(self):
        from src.api.provenance_endpoints import ProvenanceService

        svc = ProvenanceService()
        result = await svc.query_provenance("strat-001", "What is the origin of this EA?")
        assert len(result.answer) > 0
        assert result.confidence > 0

    @pytest.mark.asyncio
    async def test_query_provenance_research_score(self):
        from src.api.provenance_endpoints import ProvenanceService

        svc = ProvenanceService()
        result = await svc.query_provenance("strat-001", "What is the research score?")
        assert "research" in result.answer.lower() or "score" in result.answer.lower()

    @pytest.mark.asyncio
    async def test_query_provenance_review_status(self):
        from src.api.provenance_endpoints import ProvenanceService

        svc = ProvenanceService()
        result = await svc.query_provenance("strat-001", "Was this reviewed and approved?")
        assert len(result.answer) > 0


class TestProvenanceEndpoints:
    """Test provenance REST endpoints."""

    def test_get_provenance_chain_endpoint(self):
        client = _make_client()
        response = client.get("/api/strategies/strat-001/provenance")
        assert response.status_code == 200
        data = response.json()
        assert data["strategy_id"] == "strat-001"
        assert data["total_stages"] == 5
        assert len(data["chain"]) == 5

    def test_get_provenance_with_version_tag(self):
        client = _make_client()
        response = client.get(
            "/api/strategies/strat-001/provenance",
            params={"version_tag": "2.1.0"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["version_tag"] == "2.1.0"

    def test_query_provenance_endpoint(self):
        client = _make_client()
        response = client.post(
            "/api/strategies/provenance/query",
            json={
                "strategy_id": "strat-001",
                "query": "Where did this EA originate?",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "confidence" in data
        assert data["confidence"] > 0

    def test_get_version_specific_provenance(self):
        client = _make_client()
        response = client.get(
            "/api/strategies/strat-001/versions/1.2.0/provenance"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["version_tag"] == "1.2.0"
        assert data["strategy_id"] == "strat-001"

    def test_provenance_chain_has_all_stage_fields(self):
        client = _make_client()
        response = client.get("/api/strategies/any-strategy/provenance")
        assert response.status_code == 200
        data = response.json()
        for node in data["chain"]:
            assert "stage" in node
            assert "timestamp" in node
            assert "actor" in node
            assert "status" in node

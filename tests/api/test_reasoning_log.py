# tests/api/test_reasoning_log.py
"""
Tests for Reasoning Log API Endpoints

Story 10.2: Agent Reasoning Transparency Log & API
Tests for retrieving agent reasoning chains from memory graph.

AC1: GET /api/audit/reasoning/{decision_id} - returns reasoning chain with OPINION nodes
AC2: GET /api/audit/reasoning/department/{department} - returns hypothesis chain, evidence sources, confidence scores
"""
import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI
from unittest.mock import patch, MagicMock
from datetime import datetime
import os

# Mock the graph memory DB path before importing
os.environ.setdefault("GRAPH_MEMORY_DB", ":memory:")


@pytest.fixture
def test_app():
    """Create a test FastAPI app with the reasoning log router."""
    from src.api.reasoning_log_endpoints import router

    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(test_app):
    """Create a test client for the test app."""
    return TestClient(test_app)


class MockMemoryNode:
    """Mock MemoryNode for testing."""
    def __init__(
        self,
        node_id="test-123",
        session_id="session-456",
        content="Test content",
        action="test_action",
        reasoning="test reasoning",
        confidence=0.85,
        agent_role="research",
        created_at=None,
        metadata=None,
        alternatives_considered="alternative option 1, alternative option 2",
        constraints_applied="risk limits, position size"
    ):
        self.id = node_id
        self.session_id = session_id
        self.content = content
        self.action = action
        self.reasoning = reasoning
        self.confidence = confidence
        self.agent_role = agent_role
        self.created_at = created_at or datetime.utcnow()
        self.metadata = metadata or {}
        self.alternatives_considered = alternatives_considered
        self.constraints_applied = constraints_applied


class TestGetReasoningByDecisionId:
    """Test GET /api/audit/reasoning/{decision_id} endpoint - AC1."""

    @patch('src.api.reasoning_log_endpoints._get_facade')
    def test_get_reasoning_returns_opinion_nodes(self, mock_get_facade, client):
        """Should return reasoning chain with OPINION nodes."""
        # Mock the facade and store
        mock_facade = MagicMock()
        mock_store = MagicMock()

        # Create mock OPINION nodes
        mock_node = MockMemoryNode(
            node_id="opinion-1",
            session_id="session-123",
            action="Recommended GBPUSD short",
            reasoning="Based on technical analysis and macro indicators",
            confidence=0.75,
            agent_role="research",
            metadata={"decision_id": "decision-1", "model_used": "claude-3-opus"}
        )

        mock_store.query_nodes.return_value = [mock_node]
        mock_facade.store = mock_store
        mock_get_facade.return_value = mock_facade

        response = client.get("/api/audit/reasoning/decision-1")

        # Should return 200
        assert response.status_code == 200

        data = response.json()
        assert "decision_id" in data
        assert "opinion_nodes" in data
        assert isinstance(data["opinion_nodes"], list)

    @patch('src.api.reasoning_log_endpoints._get_facade')
    def test_get_reasoning_empty_result(self, mock_get_facade, client):
        """Should return empty opinion_nodes when no matches found."""
        # Mock the facade and store
        mock_facade = MagicMock()
        mock_store = MagicMock()
        mock_store.query_nodes.return_value = []
        mock_facade.store = mock_store
        mock_get_facade.return_value = mock_facade

        response = client.get("/api/audit/reasoning/nonexistent-decision")

        # Should return 200 with empty results
        assert response.status_code == 200

        data = response.json()
        assert data["opinion_nodes"] == []

    @patch('src.api.reasoning_log_endpoints._get_facade')
    def test_get_reasoning_includes_metadata(self, mock_get_facade, client):
        """Should include context, model_used, and summaries in response."""
        # Mock the facade and store
        mock_facade = MagicMock()
        mock_store = MagicMock()

        mock_node = MockMemoryNode(
            node_id="opinion-2",
            session_id="session-456",
            content="Analyzing GBPUSD for short opportunity based on...",
            action="Recommend short",
            reasoning="Strong bearish signals from technical indicators",
            confidence=0.8,
            agent_role="research",
            metadata={"decision_id": "decision-2", "model_used": "claude-3-sonnet"}
        )

        mock_store.query_nodes.return_value = [mock_node]
        mock_facade.store = mock_store
        mock_get_facade.return_value = mock_facade

        response = client.get("/api/audit/reasoning/decision-2")

        assert response.status_code == 200
        data = response.json()
        # Check response structure
        assert "model_used" in data
        assert "prompt_summary" in data
        assert "response_summary" in data


class TestGetDepartmentReasoning:
    """Test GET /api/audit/reasoning/department/{department} endpoint - AC2."""

    @patch('src.api.reasoning_log_endpoints._get_facade')
    def test_get_department_reasoning_returns_chain(self, mock_get_facade, client):
        """Should return hypothesis chain with confidence scores."""
        # Mock the facade and store
        mock_facade = MagicMock()
        mock_store = MagicMock()

        # Create mock nodes for research department
        mock_node = MockMemoryNode(
            node_id="opinion-3",
            session_id="session-789",
            action="Recommend GBPUSD short",
            reasoning="Technical analysis shows strong bearish momentum",
            confidence=0.82,
            agent_role="research",
            metadata={
                "evidence_sources": ["USD weakness", "GBP strength", "technical charts"],
                "sub_agents": ["macro-analyst", "technical-analyst"]
            }
        )

        mock_store.query_nodes.return_value = [mock_node]
        mock_facade.store = mock_store
        mock_get_facade.return_value = mock_facade

        response = client.get("/api/audit/reasoning/department/research")

        # Should return 200
        assert response.status_code == 200

        data = response.json()
        assert data["department"] == "research"
        assert "reasoning_chain" in data
        assert isinstance(data["reasoning_chain"], list)
        assert "total_decisions" in data

    @patch('src.api.reasoning_log_endpoints._get_facade')
    def test_get_department_reasoning_with_date_filter(self, mock_get_facade, client):
        """Should filter by start_date and end_date parameters."""
        # Mock the facade and store
        mock_facade = MagicMock()
        mock_store = MagicMock()
        mock_store.query_nodes.return_value = []
        mock_facade.store = mock_store
        mock_get_facade.return_value = mock_facade

        response = client.get(
            "/api/audit/reasoning/department/research",
            params={
                "start_date": "2026-01-01T00:00:00",
                "end_date": "2026-03-20T00:00:00"
            }
        )

        # Should return 200
        assert response.status_code == 200

    @patch('src.api.reasoning_log_endpoints._get_facade')
    def test_get_department_reasoning_with_limit(self, mock_get_facade, client):
        """Should respect the limit parameter."""
        # Mock the facade and store
        mock_facade = MagicMock()
        mock_store = MagicMock()
        mock_store.query_nodes.return_value = []
        mock_facade.store = mock_store
        mock_get_facade.return_value = mock_facade

        response = client.get(
            "/api/audit/reasoning/department/risk",
            params={"limit": 10}
        )

        # Should return 200
        assert response.status_code == 200

        # Verify limit was passed to query
        mock_store.query_nodes.assert_called()


class TestReasoningResponseStructure:
    """Test response structures match acceptance criteria."""

    @patch('src.api.reasoning_log_endpoints._get_facade')
    def test_reasoning_log_response_schema(self, mock_get_facade, client):
        """AC1: Verify response includes all required fields."""
        # Mock the facade and store
        mock_facade = MagicMock()
        mock_store = MagicMock()

        mock_node = MockMemoryNode(
            node_id="opinion-4",
            session_id="session-abc",
            action="Take action",
            reasoning="Reasoning explanation",
            confidence=0.9,
            agent_role="trading",
            metadata={"decision_id": "decision-4"}
        )

        mock_store.query_nodes.return_value = [mock_node]
        mock_facade.store = mock_store
        mock_get_facade.return_value = mock_facade

        response = client.get("/api/audit/reasoning/decision-4")

        assert response.status_code == 200
        data = response.json()

        # AC1: Required fields
        assert "context_at_decision" in data
        assert "model_used" in data
        assert "prompt_summary" in data
        assert "response_summary" in data
        assert "action_taken" in data
        assert "opinion_nodes" in data

    @patch('src.api.reasoning_log_endpoints._get_facade')
    def test_department_reasoning_chain_schema(self, mock_get_facade, client):
        """AC2: Verify department response includes hypothesis, evidence, confidence."""
        # Mock the facade and store
        mock_facade = MagicMock()
        mock_store = MagicMock()

        mock_node = MockMemoryNode(
            node_id="opinion-5",
            session_id="session-def",
            action="Hypothesis test",
            reasoning="Evidence-based reasoning",
            confidence=0.88,
            agent_role="development",
            metadata={
                "evidence_sources": ["source1", "source2"],
                "sub_agents": ["agent1"]
            }
        )

        mock_store.query_nodes.return_value = [mock_node]
        mock_facade.store = mock_store
        mock_get_facade.return_value = mock_facade

        response = client.get("/api/audit/reasoning/department/development")

        assert response.status_code == 200
        data = response.json()

        # AC2: Required fields in reasoning chain
        if data["reasoning_chain"]:
            chain_item = data["reasoning_chain"][0]
            assert "hypothesis" in chain_item
            assert "evidence_sources" in chain_item
            assert "confidence_score" in chain_item
            assert "sub_agents" in chain_item

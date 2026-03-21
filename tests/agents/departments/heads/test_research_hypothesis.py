"""
Tests for ResearchHead Hypothesis Generation

Story 7.1: Research Department - Real Hypothesis Generation
Tests the hypothesis generation, confidence scoring, and opinion writing.
"""
import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock


class TestResearchHeadHypothesisGeneration:
    """Test ResearchHead hypothesis generation functionality."""

    def test_hypothesis_schema(self):
        """Test Hypothesis dataclass has correct schema."""
        from src.agents.departments.heads.research_head import Hypothesis

        hypothesis = Hypothesis(
            symbol="EURUSD",
            timeframe="H4",
            hypothesis="Test hypothesis",
            supporting_evidence=["evidence1", "evidence2"],
            confidence_score=0.8,
            recommended_next_steps=["step1", "step2"]
        )

        assert hypothesis.symbol == "EURUSD"
        assert hypothesis.timeframe == "H4"
        assert hypothesis.confidence_score == 0.8

        # Test to_dict method
        h_dict = hypothesis.to_dict()
        assert h_dict["symbol"] == "EURUSD"
        assert h_dict["confidence_score"] == 0.8

    def test_research_task_schema(self):
        """Test ResearchTask dataclass has correct schema."""
        from src.agents.departments.heads.research_head import ResearchTask

        task = ResearchTask(
            query="Test research query",
            symbols=["EURUSD", "GBPUSD"],
            timeframes=["H4", "D1"],
            session_id="test-session-123"
        )

        assert task.query == "Test research query"
        assert task.symbols == ["EURUSD", "GBPUSD"]
        assert task.timeframes == ["H4", "D1"]
        assert task.session_id == "test-session-123"

    def test_hypothesis_to_dict_complete(self):
        """Test Hypothesis to_dict returns all fields."""
        from src.agents.departments.heads.research_head import Hypothesis

        hypothesis = Hypothesis(
            symbol="GBPUSD",
            timeframe="D1",
            hypothesis="GBPUSD will decline due to BoE policy",
            supporting_evidence=["evidence1", "evidence2", "evidence3"],
            confidence_score=0.85,
            recommended_next_steps=["Backtest", "Risk review"]
        )

        h_dict = hypothesis.to_dict()

        assert h_dict["symbol"] == "GBPUSD"
        assert h_dict["timeframe"] == "D1"
        assert h_dict["hypothesis"] == "GBPUSD will decline due to BoE policy"
        assert len(h_dict["supporting_evidence"]) == 3
        assert h_dict["confidence_score"] == 0.85
        assert len(h_dict["recommended_next_steps"]) == 2


class TestResearchHeadConfidenceScoring:
    """Test confidence scoring functionality."""

    @pytest.fixture
    def research_head(self):
        """Create a ResearchHead instance with mocked dependencies."""
        # Import after mocking to avoid initialization issues
        with patch('src.agents.departments.heads.research_head.DepartmentHead.__init__'):
            from src.agents.departments.heads.research_head import ResearchHead

            head = ResearchHead.__new__(ResearchHead)
            head.department = MagicMock()
            head.agent_type = "research_head"
            head._current_session_id = "test-session"
            head.pageindex_client = None
            head.embedding_service = None
            head.mcp_integration = None
            return head

    def test_combine_evidence(self, research_head):
        """Test evidence combination from multiple sources."""
        knowledge_results = [
            {"source": "pageindex", "collection": "articles", "content": "Article about EURUSD", "score": 0.9}
        ]
        semantic_results = [
            {"source": "chroma", "content": "Memory of previous analysis", "score": 0.7}
        ]
        web_results = [
            {"source": "web", "content": "Latest EURUSD news", "score": 0.85}
        ]

        evidence = research_head._combine_evidence(
            knowledge_results, semantic_results, web_results
        )

        assert len(evidence) > 0
        assert any("KB-articles" in e for e in evidence)
        assert any("Memory" in e for e in evidence)
        assert any("Web" in e for e in evidence)

    def test_calculate_confidence_with_no_evidence(self, research_head):
        """Test confidence calculation with no evidence."""
        confidence = research_head._calculate_confidence([])
        assert confidence == 0.1

    def test_calculate_confidence_with_evidence(self, research_head):
        """Test confidence calculation with evidence."""
        evidence = ["evidence1", "evidence2", "evidence3", "evidence4", "evidence5",
                    "evidence6", "evidence7", "evidence8", "evidence9", "evidence10"]
        confidence = research_head._calculate_confidence(evidence)

        assert 0.1 < confidence <= 0.95

    def test_calculate_confidence_max_cap(self, research_head):
        """Test confidence calculation is capped at 0.95."""
        # Create many evidence items
        evidence = ["e" + str(i) for i in range(20)]
        confidence = research_head._calculate_confidence(evidence)

        assert confidence <= 0.95

    def test_generate_next_steps_high_confidence(self, research_head):
        """Test next steps generation for high confidence."""
        next_steps = research_head._generate_next_steps(0.8, ["evidence1"])

        assert any("TRD" in step or "Development" in step for step in next_steps)
        assert any("backtest" in step.lower() for step in next_steps)

    def test_generate_next_steps_low_confidence(self, research_head):
        """Test next steps generation for low confidence."""
        next_steps = research_head._generate_next_steps(0.3, ["evidence1"])

        assert any("more evidence" in step.lower() for step in next_steps)
        assert any("alternative" in step.lower() for step in next_steps)

    def test_should_escalate_to_trd_high_confidence(self, research_head):
        """Test TRD escalation decision for high confidence."""
        from src.agents.departments.heads.research_head import Hypothesis

        high_confidence_hypothesis = Hypothesis(
            symbol="EURUSD",
            timeframe="H4",
            hypothesis="Test",
            supporting_evidence=[],
            confidence_score=0.8,
            recommended_next_steps=[]
        )

        assert research_head.should_escalate_to_trd(high_confidence_hypothesis) is True

    def test_should_escalate_to_trd_low_confidence(self, research_head):
        """Test TRD escalation decision for low confidence."""
        from src.agents.departments.heads.research_head import Hypothesis

        low_confidence_hypothesis = Hypothesis(
            symbol="EURUSD",
            timeframe="H4",
            hypothesis="Test",
            supporting_evidence=[],
            confidence_score=0.5,
            recommended_next_steps=[]
        )

        assert research_head.should_escalate_to_trd(low_confidence_hypothesis) is False

    def test_should_escalate_to_trd_boundary(self, research_head):
        """Test TRD escalation decision at boundary (0.75)."""
        from src.agents.departments.heads.research_head import Hypothesis

        boundary_hypothesis = Hypothesis(
            symbol="EURUSD",
            timeframe="H4",
            hypothesis="Test",
            supporting_evidence=[],
            confidence_score=0.75,
            recommended_next_steps=[]
        )

        # Should escalate at exactly threshold
        assert research_head.should_escalate_to_trd(boundary_hypothesis) is True

    def test_get_escalation_prompt(self, research_head):
        """Test TRD escalation prompt generation."""
        from src.agents.departments.heads.research_head import Hypothesis

        hypothesis = Hypothesis(
            symbol="EURUSD",
            timeframe="H4",
            hypothesis="EURUSD will trend upward on H4 timeframe",
            supporting_evidence=["evidence1", "evidence2"],
            confidence_score=0.8,
            recommended_next_steps=["Run backtest", "Escalate to Development"]
        )

        prompt = research_head.get_escalation_prompt(hypothesis)

        assert "EURUSD" in prompt
        assert "80%" in prompt
        assert "Proceed to TRD?" in prompt
        assert "H4" in prompt

    def test_get_escalation_prompt_low_confidence(self, research_head):
        """Test escalation prompt returns empty for low confidence."""
        from src.agents.departments.heads.research_head import Hypothesis

        hypothesis = Hypothesis(
            symbol="EURUSD",
            timeframe="H4",
            hypothesis="Test",
            supporting_evidence=[],
            confidence_score=0.5,
            recommended_next_steps=[]
        )

        prompt = research_head.get_escalation_prompt(hypothesis)
        assert prompt == ""

    def test_get_escalation_prompt_includes_next_steps(self, research_head):
        """Test escalation prompt includes next steps."""
        from src.agents.departments.heads.research_head import Hypothesis

        hypothesis = Hypothesis(
            symbol="GBPUSD",
            timeframe="D1",
            hypothesis="Test hypothesis",
            supporting_evidence=[],
            confidence_score=0.9,
            recommended_next_steps=["Step 1: Validate", "Step 2: Implement"]
        )

        prompt = research_head.get_escalation_prompt(hypothesis)

        assert "Step 1" in prompt
        assert "Step 2" in prompt


class TestResearchHeadOpinionWriting:
    """Test OPINION node writing functionality."""

    @pytest.fixture
    def research_head(self):
        """Create a ResearchHead instance with mocked dependencies."""
        with patch('src.agents.departments.heads.research_head.DepartmentHead.__init__'):
            from src.agents.departments.heads.research_head import ResearchHead

            head = ResearchHead.__new__(ResearchHead)
            head.department = MagicMock()
            head.department.value = "research"
            head.agent_type = "research_head"
            head._current_session_id = "test-session"
            return head

    @patch('src.memory.graph.store.GraphMemoryStore')
    def test_write_research_opinion(self, mock_store_class, research_head):
        """Test writing OPINION node to memory graph."""
        from src.agents.departments.heads.research_head import Hypothesis

        mock_store = MagicMock()
        mock_store_class.return_value = mock_store

        hypothesis = Hypothesis(
            symbol="EURUSD",
            timeframe="H4",
            hypothesis="Test hypothesis",
            supporting_evidence=["evidence1"],
            confidence_score=0.8,
            recommended_next_steps=["step1"]
        )

        # This should not raise an exception
        research_head._write_research_opinion(hypothesis, "EURUSD analysis")

        # Verify store was called
        mock_store.create_node.assert_called_once()


class TestResearchHeadIntegration:
    """Integration tests for ResearchHead."""

    def test_trd_escalation_threshold_constant(self):
        """Test that TRD escalation threshold is correctly defined."""
        from src.agents.departments.heads.research_head import ResearchHead

        assert ResearchHead.TRD_ESCALATION_THRESHOLD == 0.75

    def test_hypothesis_preserves_evidence_order(self):
        """Test that hypothesis preserves evidence order."""
        from src.agents.departments.heads.research_head import Hypothesis

        evidence = ["first", "second", "third"]
        hypothesis = Hypothesis(
            symbol="EURUSD",
            timeframe="H4",
            hypothesis="Test",
            supporting_evidence=evidence,
            confidence_score=0.8,
            recommended_next_steps=[]
        )

        assert hypothesis.supporting_evidence == evidence

    def test_hypothesis_default_values(self):
        """Test Hypothesis default values."""
        from src.agents.departments.heads.research_head import Hypothesis

        hypothesis = Hypothesis(
            symbol="USDJPY",
            timeframe="H1",
            hypothesis="Minimal hypothesis",
            supporting_evidence=[],
            confidence_score=0.5,
            recommended_next_steps=[]
        )

        assert hypothesis.symbol == "USDJPY"
        assert hypothesis.timeframe == "H1"
        assert len(hypothesis.supporting_evidence) == 0

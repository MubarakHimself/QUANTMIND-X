# tests/agents/departments/heads/test_analysis_head.py
"""
Tests for Analysis Department Head

Task Group: Department Heads - Analysis
"""
import pytest
import tempfile
import os


class TestAnalysisHead:
    """Test Analysis Department Head."""

    def test_analysis_head_initialization(self):
        """Analysis Head should initialize with correct config."""
        from src.agents.departments.heads.analysis_head import AnalysisHead

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_mail.db")
            head = AnalysisHead(mail_db_path=db_path)

            assert head.department.value == "analysis"
            assert head.agent_type == "analyst_head"
            assert "market_analyst" in head.sub_agents
            assert "sentiment_scanner" in head.sub_agents
            assert "news_monitor" in head.sub_agents

            head.close()

    def test_analysis_head_has_analysis_tools(self):
        """Analysis Head should have analysis-related tools."""
        from src.agents.departments.heads.analysis_head import AnalysisHead

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_mail.db")
            head = AnalysisHead(mail_db_path=db_path)

            tools = head.get_tools()
            tool_names = [t.get("name", "") for t in tools]

            assert "analyze_market" in tool_names
            assert "scan_sentiment" in tool_names

            head.close()

    def test_analysis_head_process_market_analysis(self):
        """Analysis Head should process market analysis tasks."""
        from src.agents.departments.heads.analysis_head import AnalysisHead

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_mail.db")
            head = AnalysisHead(mail_db_path=db_path)

            result = head.analyze_market(symbol="EURUSD", timeframe="H1")

            assert result["status"] == "analyzed"
            assert result["symbol"] == "EURUSD"
            assert "analysis" in result

            head.close()

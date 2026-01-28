"""
pytest configuration for Analyst Agent tests.

Provides fixtures and test utilities for the analyst agent test suite.
"""

import pytest
from unittest.mock import Mock, patch
import tempfile
import json
from pathlib import Path

# Import test modules
from tools.analyst_agent.tests.test_kb_client import *
from tools.analyst_agent.tests.test_chains import *
from tools.analyst_agent.tests.test_graph import *
from tools.analyst_agent.tests.test_cli import *


@pytest.fixture
def test_data_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_nprd_data():
    """Sample NPRD data for testing."""
    return {
        "title": "Test Trading Strategy",
        "description": "A sample trading strategy description",
        "concepts": ["Moving Average Crossover", "RSI Divergence"],
        "indicators": ["MA", "RSI"],
        "timeframes": ["H1", "D1"],
        "risk_management": {
            "stop_loss": "1.5%",
            "take_profit": "3.0%",
            "position_size": "2% per trade"
        }
    }


@pytest.fixture
def sample_trd_data():
    """Sample TRD data for testing."""
    return {
        "title": "Test TRD",
        "overview": "Test trading strategy overview",
        "trading_strategy": {
            "concepts": ["Moving Average Crossover"],
            "indicators": ["MA", "RSI"],
            "timeframes": ["H1", "D1"]
        },
        "risk_management": {
            "stop_loss": "1.5%",
            "take_profit": "3.0%",
            "position_size": "2% per trade"
        }
    }


@pytest.fixture
def mock_chroma_db():
    """Mock ChromaDB for testing."""
    with patch('tools.analyst_agent.kb.client.chromadb') as mock_chroma:
        mock_client = Mock()
        mock_chroma.PersistentClient.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_langchain():
    """Mock LangChain components."""
    with patch('tools.analyst_agent.chains.ChatOpenAI') as mock_llm_class:
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        yield mock_llm


@pytest.fixture
def mock_workflow():
    """Mock workflow components."""
    with patch('tools.analyst_agent.graph.workflow.create_analyst_graph') as mock_create_graph, \
         patch('tools.analyst_agent.graph.workflow.run_analyst_workflow') as mock_run_workflow:

        mock_graph = Mock()
        mock_create_graph.return_value = mock_graph
        mock_run_workflow.return_value = {"success": True, "trd_output": "Test TRD"}
        yield mock_graph, mock_run_workflow


@pytest.fixture
def mock_cli(runner):
    """Mock CLI runner."""
    with patch('tools.analyst_agent.cli.commands.typer.testing.CliRunner') as mock_runner_class:
        mock_runner = Mock()
        mock_runner_class.return_value = mock_runner
        yield mock_runner


# Test markers
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "kb: mark test as knowledge base related"
    )
    config.addinivalue_line(
        "markers", "chains: mark test as LangChain chain related"
    )
    config.addinivalue_line(
        "markers", "graph: mark test as LangGraph workflow related"
    )
    config.addinivalue_line(
        "markers", "cli: mark test as CLI related"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
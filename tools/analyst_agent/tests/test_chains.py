"""
Test suite for LangChain chains in Analyst Agent.

Tests:
- extraction_chain with mock NPRD data
- search_chain with mock KB
- generation_chain with mock inputs
- Chain error handling
- Chain integration
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# Import the chains module
from tools.analyst_agent.chains import extraction_chain, search_chain, generation_chain


@pytest.fixture
def mock_llm():
    """Mock LLM for testing chains."""
    with patch('tools.analyst_agent.chains.ChatOpenAI') as mock_llm_class:
        mock_llm_instance = Mock()
        mock_llm_class.return_value = mock_llm_instance
        yield mock_llm_instance


@pytest.fixture
def mock_kb_client():
    """Mock ChromaKBClient for testing search chain."""
    with patch('tools.analyst_agent.chains.ChromaKBClient') as mock_client_class:
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        yield mock_client


def test_extraction_chain_basic_functionality(mock_llm):
    """Test extraction_chain basic functionality."""
    # Mock LLM response
    mock_llm.invoke.return_value.content = json.dumps({
        "concepts": ["Moving Average Crossover", "RSI Divergence"],
        "indicators": ["MA", "RSI"],
        "timeframes": ["H1", "D1"],
        "risk_management": {
            "stop_loss": "1.5%",
            "take_profit": "3.0%",
            "position_size": "2% per trade"
        }
    })

    # Test extraction chain
    nprd_data = {
        "title": "Test Strategy",
        "description": "A test trading strategy description"
    }

    result = extraction_chain.invoke({"nprd_data": nprd_data})

    assert "concepts" in result
    assert "indicators" in result
    assert "timeframes" in result
    assert "risk_management" in result

    # Verify LLM was called with correct prompt
    mock_llm.invoke.assert_called_once()


def test_extraction_chain_with_empty_input(mock_llm):
    """Test extraction_chain with empty input."""
    mock_llm.invoke.return_value.content = json.dumps({})

    nprd_data = {}

    result = extraction_chain.invoke({"nprd_data": nprd_data})

    assert "concepts" in result
    assert "indicators" in result
    assert "timeframes" in result
    assert "risk_management" in result


def test_extraction_chain_error_handling(mock_llm):
    """Test extraction_chain error handling."""
    mock_llm.invoke.side_effect = Exception("LLM error")

    nprd_data = {"title": "Test Strategy"}

    with pytest.raises(Exception, match="LLM error"):
        extraction_chain.invoke({"nprd_data": nprd_data})


def test_search_chain_basic_functionality(mock_kb_client, mock_llm):
    """Test search_chain basic functionality."""
    # Mock KB client search results
    mock_kb_client.search.return_value = [
        {
            "title": "Moving Average Crossover Strategy",
            "file_path": "ma_crossover.md",
            "categories": "Trading Systems",
            "score": 0.95,
            "preview": "This article discusses..."
        },
        {
            "title": "RSI Divergence Trading",
            "file_path": "rsi_divergence.md",
            "categories": "Indicators",
            "score": 0.88,
            "preview": "Relative Strength Index..."
        }
    ]

    # Mock LLM response for search chain
    mock_llm.invoke.return_value.content = json.dumps({
        "search_results": [
            {"title": "Moving Average Crossover Strategy", "relevance": "high"},
            {"title": "RSI Divergence Trading", "relevance": "medium"}
        ]
    })

    # Test search chain
    extraction_result = {
        "concepts": ["Moving Average Crossover", "RSI Divergence"],
        "indicators": ["MA", "RSI"]
    }

    result = search_chain.invoke({"extraction_result": extraction_result})

    assert "search_results" in result
    assert len(result["search_results"]) > 0

    # Verify KB client was called with correct query
    mock_kb_client.search.assert_called_once()


def test_search_chain_with_category_filter(mock_kb_client):
    """Test search_chain with category filter."""
    mock_kb_client.search.return_value = [
        {"title": "Trading Systems Article", "categories": "Trading Systems", "score": 0.9}
    ]

    extraction_result = {
        "concepts": ["Trading Systems"],
        "indicators": []
    }

    result = search_chain.invoke({
        "extraction_result": extraction_result,
        "category_filter": "Trading Systems"
    })

    assert len(result["search_results"]) == 1
    assert "Trading Systems" in result["search_results"][0].get("categories", "")


def test_search_chain_no_results(mock_kb_client):
    """Test search_chain with no search results."""
    mock_kb_client.search.return_value = []

    extraction_result = {"concepts": ["Nonexistent Concept"], "indicators": []}

    result = search_chain.invoke({"extraction_result": extraction_result})

    assert "search_results" in result
    assert len(result["search_results"]) == 0


def test_search_chain_error_handling(mock_kb_client):
    """Test search_chain error handling."""
    mock_kb_client.search.side_effect = Exception("KB error")

    extraction_result = {"concepts": ["Test"], "indicators": []}

    with pytest.raises(Exception, match="KB error"):
        search_chain.invoke({"extraction_result": extraction_result})


def test_generation_chain_basic_functionality(mock_llm):
    """Test generation_chain basic functionality."""
    # Mock LLM response for generation
    mock_llm.invoke.return_value.content = """# Test TRD
## Overview
This is a test TRD document.

## Trading Strategy
- Concept: Moving Average Crossover
- Indicators: MA, RSI

## Risk Management
- Stop Loss: 1.5%
- Take Profit: 3.0%"""

    # Test generation chain
    search_result = {
        "search_results": [
            {"title": "Moving Average Crossover Strategy", "relevance": "high"}
        ],
        "extraction_result": {
            "concepts": ["Moving Average Crossover"],
            "indicators": ["MA", "RSI"]
        }
    }

    result = generation_chain.invoke(search_result)

    assert "trd_output" in result
    assert "trd_path" in result["metadata"]
    assert "status" in result["metadata"]

    # Verify LLM was called with correct prompt
    mock_llm.invoke.assert_called_once()


def test_generation_chain_with_missing_info(mock_llm):
    """Test generation_chain with missing information."""
    mock_llm.invoke.return_value.content = """# Test TRD
## Overview
This TRD has some missing information.

## Trading Strategy
- Concept: Moving Average Crossover
- Indicators: MA (RSI information missing)

## Risk Management
- Stop Loss: 1.5% (Take Profit information missing)"""

    search_result = {
        "search_results": [],
        "extraction_result": {
            "concepts": ["Moving Average Crossover"],
            "indicators": ["MA"]
        },
        "missing_info": [
            {"field_name": "risk_management.take_profit", "description": "Take profit percentage"}
        ]
    }

    result = generation_chain.invoke(search_result)

    assert "trd_output" in result
    assert "missing_info" in result


def test_generation_chain_error_handling(mock_llm):
    """Test generation_chain error handling."""
    mock_llm.invoke.side_effect = Exception("LLM generation error")

    search_result = {
        "search_results": [],
        "extraction_result": {"concepts": ["Test"], "indicators": []}
    }

    with pytest.raises(Exception, match="LLM generation error"):
        generation_chain.invoke(search_result)


def test_chain_integration():
    """Test end-to-end chain integration."""
    # Mock all chain components
    with patch('tools.analyst_agent.chains.ChatOpenAI') as mock_llm_class, \
         patch('tools.analyst_agent.chains.ChromaKBClient') as mock_kb_class:

        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        mock_kb = Mock()
        mock_kb_class.return_value = mock_kb

        # Configure mock responses
        mock_llm.invoke.side_effect = [
            Mock(content=json.dumps({
                "concepts": ["Test Concept"],
                "indicators": ["Test Indicator"]
            })),
            Mock(content=json.dumps({
                "search_results": [{"title": "Test Article", "relevance": "high"}]
            })),
            Mock(content="Test TRD content")
        ]

        mock_kb.search.return_value = [
            {"title": "Test Article", "score": 0.9, "categories": "Trading Systems"}
        ]

        # Test the full chain
        nprd_data = {"title": "Test NPRD", "description": "Test description"}

        # Extraction chain
        extraction_result = extraction_chain.invoke({"nprd_data": nprd_data})

        # Search chain
        search_result = search_chain.invoke({"extraction_result": extraction_result})

        # Generation chain
        generation_result = generation_chain.invoke(search_result)

        assert "trd_output" in generation_result
        assert "status" in generation_result["metadata"]


def test_chain_with_user_answers():
    """Test generation_chain with user answers."""
    mock_llm = Mock()
    mock_llm.invoke.return_value.content = """# Test TRD with User Answers
## Risk Management
- Stop Loss: 1.5% (provided by user)
- Take Profit: 3.0% (provided by user)"""

    search_result = {
        "search_results": [],
        "extraction_result": {"concepts": ["Test"], "indicators": []},
        "user_answers": {
            "risk_management.stop_loss": "1.5%",
            "risk_management.take_profit": "3.0%"
        }
    }

    result = generation_chain.invoke(search_result)

    assert "trd_output" in result
    assert "user_answers" in result


def test_chain_output_format():
    """Test chain output format and structure."""
    mock_llm = Mock()
    mock_llm.invoke.return_value.content = """# Test TRD
## Overview
Test document.

## Trading Strategy
- Concept: Test Concept
- Indicators: Test Indicator"""

    search_result = {
        "search_results": [],
        "extraction_result": {"concepts": ["Test Concept"], "indicators": ["Test Indicator"]}
    }

    result = generation_chain.invoke(search_result)

    trd_output = result["trd_output"]
    assert "# Test TRD" in trd_output
    assert "## Overview" in trd_output
    assert "## Trading Strategy" in trd_output
    assert "Test Concept" in trd_output
    assert "Test Indicator" in trd_output


def test_chain_performance():
    """Test chain performance with large inputs."""
    mock_llm = Mock()
    mock_llm.invoke.return_value.content = "Test TRD content"

    # Large extraction result
    large_extraction_result = {
        "concepts": ["Concept " + str(i) for i in range(100)],
        "indicators": ["Indicator " + str(i) for i in range(50)],
        "timeframes": ["TF " + str(i) for i in range(10)],
        "risk_management": {
            "stop_loss": "1.5%",
            "take_profit": "3.0%",
            "position_size": "2% per trade"
        }
    }

    # Test that chain handles large inputs without crashing
    result = generation_chain.invoke({
        "search_results": [],
        "extraction_result": large_extraction_result
    })

    assert "trd_output" in result


def test_chain_error_recovery():
    """Test chain error recovery and retry logic."""
    mock_llm = Mock()

    # First call fails, second succeeds
    mock_llm.invoke.side_effect = [
        Exception("Temporary LLM error"),
        Mock(content="Test TRD content")
    ]

    search_result = {"search_results": [], "extraction_result": {"concepts": ["Test"]}}

    # Should retry and succeed
    result = generation_chain.invoke(search_result)

    assert "trd_output" in result
    assert mock_llm.invoke.call_count == 2  # Should retry once
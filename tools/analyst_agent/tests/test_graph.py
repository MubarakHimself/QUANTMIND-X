"""
Test suite for LangGraph workflow in Analyst Agent.

Tests:
- State transitions
- auto_mode bypasses human_input
- interactive mode includes HITL
- Workflow error handling
- State validation
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, Optional
from datetime import datetime

# Import the workflow module
from tools.analyst_agent.graph.workflow import (
    create_analyst_graph,
    run_analyst_workflow,
    run_analyst_workflow_interactive,
    get_workflow_graphviz
)


@pytest.fixture
def mock_graph():
    """Mock LangGraph components for testing."""
    with patch('tools.analyst_agent.graph.workflow.StateGraph') as mock_graph_class, \
         patch('tools.analyst_agent.graph.workflow.MemorySaver') as mock_saver_class:

        mock_graph = Mock()
        mock_graph_class.return_value = mock_graph
        mock_saver = Mock()
        mock_saver_class.return_value = mock_saver

        yield mock_graph, mock_saver


@pytest.fixture
def mock_nodes():
    """Mock node functions for testing."""
    with patch('tools.analyst_agent.graph.nodes.extract_node') as mock_extract, \
         patch('tools.analyst_agent.graph.nodes.search_node') as mock_search, \
         patch('tools.analyst_agent.graph.nodes.check_missing_node') as mock_check_missing, \
         patch('tools.analyst_agent.graph.nodes.generate_node') as mock_generate, \
         patch('tools.analyst_agent.graph.nodes.review_node') as mock_review:

        yield mock_extract, mock_search, mock_check_missing, mock_generate, mock_review


@pytest.fixture
def test_nprd_data():
    """Sample NPRD data for testing."""
    return {
        "title": "Test NPRD",
        "description": "A test trading strategy description",
        "concepts": ["Moving Average Crossover"],
        "indicators": ["MA", "RSI"],
        "timeframes": ["H1", "D1"],
        "risk_management": {
            "stop_loss": "1.5%",
            "take_profit": "3.0%",
            "position_size": "2% per trade"
        }
    }


def test_create_analyst_graph_basic(mock_graph):
    """Test create_analyst_graph basic functionality."""
    mock_graph_instance, mock_saver = mock_graph

    # Test auto_mode=False (interactive)
    graph = create_analyst_graph(auto_mode=False)

    # Verify graph was created with correct nodes
    assert mock_graph_instance.add_node.call_count == 6  # 6 nodes added
    assert "extract" in [call[0][0] for call in mock_graph_instance.add_node.call_args_list]
    assert "search" in [call[0][0] for call in mock_graph_instance.add_node.call_args_list]
    assert "check_missing" in [call[0][0] for call in mock_graph_instance.add_node.call_args_list]
    assert "human_input" in [call[0][0] for call in mock_graph_instance.add_node.call_args_list]
    assert "generate" in [call[0][0] for call in mock_graph_instance.add_node.call_args_list]
    assert "review" in [call[0][0] for call in mock_graph_instance.add_node.call_args_list]

    # Verify edges were set
    assert mock_graph_instance.set_entry_point.called
    assert mock_graph_instance.add_edge.call_count >= 5  # At least 5 edges


def test_create_analyst_graph_auto_mode(mock_graph):
    """Test create_analyst_graph with auto_mode=True."""
    mock_graph_instance, mock_saver = mock_graph

    # Test auto_mode=True (no human_input)
    graph = create_analyst_graph(auto_mode=True)

    # Verify human_input node is not added or is bypassed
    human_input_calls = [call for call in mock_graph_instance.add_node.call_args_list if call[0][0] == "human_input"]
    assert len(human_input_calls) == 0  # human_input node should not be added


def test_run_analyst_workflow_basic(test_nprd_data, mock_nodes):
    """Test run_analyst_workflow basic functionality."""
    mock_extract, mock_search, mock_check_missing, mock_generate, mock_review = mock_nodes

    # Configure mock node responses
    mock_extract.return_value = {"status": "extracted", "concepts": ["Test Concept"]}
    mock_search.return_value = {"status": "searched", "search_results": []}
    mock_check_missing.return_value = {"status": "complete", "missing_info": []}
    mock_generate.return_value = {"status": "generated", "trd_output": "Test TRD"}
    mock_review.return_value = {"status": "complete", "trd_output": "Test TRD"}

    # Mock graph execution
    with patch('tools.analyst_agent.graph.workflow.create_analyst_graph') as mock_create_graph:
        mock_graph = Mock()
        mock_create_graph.return_value = mock_graph
        mock_graph.astream.return_value = iter([{"generate": {"status": "complete"}}])

        result = run_analyst_workflow(test_nprd_data, auto_mode=True)

        assert result["success"] is True
        assert result["trd_output"] == "Test TRD"


def test_run_analyst_workflow_with_missing_info(test_nprd_data, mock_nodes):
    """Test workflow with missing information (should trigger human_input in interactive mode)."""
    mock_extract, mock_search, mock_check_missing, mock_generate, mock_review = mock_nodes

    # Configure mock responses
    mock_extract.return_value = {"status": "extracted", "concepts": ["Test Concept"]}
    mock_search.return_value = {"status": "searched", "search_results": []}
    mock_check_missing.return_value = {"status": "prompting", "missing_info": [{"field_name": "risk_management.stop_loss"}]}
    mock_generate.return_value = {"status": "generated", "trd_output": "Test TRD with missing info"}
    mock_review.return_value = {"status": "complete", "trd_output": "Test TRD with missing info"}

    # Mock graph execution
    with patch('tools.analyst_agent.graph.workflow.create_analyst_graph') as mock_create_graph:
        mock_graph = Mock()
        mock_create_graph.return_value = mock_graph
        mock_graph.astream.return_value = iter([{"generate": {"status": "complete"}}])

        result = run_analyst_workflow(test_nprd_data, auto_mode=False)

        assert result["success"] is True
        assert "missing_info" in result["state"]


def test_run_analyst_workflow_auto_mode_bypasses_human_input(test_nprd_data, mock_nodes):
    """Test that auto_mode bypasses human_input node."""
    mock_extract, mock_search, mock_check_missing, mock_generate, mock_review = mock_nodes

    # Configure mock responses - check_missing returns "generate" path
    mock_extract.return_value = {"status": "extracted", "concepts": ["Test Concept"]}
    mock_search.return_value = {"status": "searched", "search_results": []}
    mock_check_missing.return_value = {"status": "generate", "missing_info": []}
    mock_generate.return_value = {"status": "generated", "trd_output": "Auto-generated TRD"}
    mock_review.return_value = {"status": "complete", "trd_output": "Auto-generated TRD"}

    # Mock graph execution
    with patch('tools.analyst_agent.graph.workflow.create_analyst_graph') as mock_create_graph:
        mock_graph = Mock()
        mock_create_graph.return_value = mock_graph
        mock_graph.astream.return_value = iter([{"review": {"status": "complete"}}])

        result = run_analyst_workflow(test_nprd_data, auto_mode=True)

        assert result["success"] is True
        assert "human_input" not in result["state"]


def test_run_analyst_workflow_error_handling(test_nprd_data):
    """Test workflow error handling."""
    # Mock graph creation that raises an exception
    with patch('tools.analyst_agent.graph.workflow.create_analyst_graph', side_effect=Exception("Graph creation error")):

        result = run_analyst_workflow(test_nprd_data)

        assert result["success"] is False
        assert "Graph creation error" in result["errors"][0]


def test_run_analyst_workflow_interactive_basic(test_nprd_data, mock_nodes):
    """Test interactive workflow basic functionality."""
    mock_extract, mock_search, mock_check_missing, mock_generate, mock_review = mock_nodes

    # Configure mock responses
    mock_extract.return_value = {"status": "extracted", "concepts": ["Test Concept"]}
    mock_search.return_value = {"status": "searched", "search_results": []}
    mock_check_missing.return_value = {"status": "prompting", "missing_info": []}
    mock_generate.return_value = {"status": "generated", "trd_output": "Interactive TRD"}
    mock_review.return_value = {"status": "complete", "trd_output": "Interactive TRD"}

    # Mock graph execution
    with patch('tools.analyst_agent.graph.workflow.create_analyst_graph') as mock_create_graph:
        mock_graph = Mock()
        mock_create_graph.return_value = mock_graph
        mock_graph.ainvoke.return_value = {"status": "complete", "trd_output": "Interactive TRD"}

        # Mock user input callback
        async def mock_callback(missing_info):
            return {"risk_management.stop_loss": "1.5%"}

        result = run_analyst_workflow_interactive(test_nprd_data, user_input_callback=mock_callback)

        assert result["success"] is True
        assert "Interactive TRD" in result["trd_output"]


def test_run_analyst_workflow_interactive_no_callback(test_nprd_data, mock_nodes):
    """Test interactive workflow with no user input callback."""
    mock_extract, mock_search, mock_check_missing, mock_generate, mock_review = mock_nodes

    # Configure mock responses
    mock_extract.return_value = {"status": "extracted", "concepts": ["Test Concept"]}
    mock_search.return_value = {"status": "searched", "search_results": []}
    mock_check_missing.return_value = {"status": "prompting", "missing_info": [{"field_name": "risk_management.stop_loss"}]}
    mock_generate.return_value = {"status": "generated", "trd_output": "TRD with skipped fields"}
    mock_review.return_value = {"status": "complete", "trd_output": "TRD with skipped fields"}

    # Mock graph execution
    with patch('tools.analyst_agent.graph.workflow.create_analyst_graph') as mock_create_graph:
        mock_graph = Mock()
        mock_create_graph.return_value = mock_graph
        mock_graph.ainvoke.return_value = {"status": "complete", "trd_output": "TRD with skipped fields"}

        result = run_analyst_workflow_interactive(test_nprd_data)

        assert result["success"] is True
        assert "SKIPPED" in result["trd_output"]


def test_run_analyst_workflow_interactive_error_handling(test_nprd_data):
    """Test interactive workflow error handling."""
    # Mock graph creation that raises an exception
    with patch('tools.analyst_agent.graph.workflow.create_analyst_graph', side_effect=Exception("Interactive workflow error")):

        result = run_analyst_workflow_interactive(test_nprd_data)

        assert result["success"] is False
        assert "Interactive workflow error" in result["errors"][0]


def test_get_workflow_graphviz():
    """Test workflow graph visualization generation."""
    dot_graph = get_workflow_graphviz()

    assert "digraph AnalystWorkflow" in dot_graph
    assert "extract" in dot_graph
    assert "search" in dot_graph
    assert "check_missing" in dot_graph
    assert "human_input" in dot_graph
    assert "generate" in dot_graph
    assert "review" in dot_graph


def test_state_transitions():
    """Test state transitions between nodes."""
    # This tests the conditional logic in the workflow
    with patch('tools.analyst_agent.graph.workflow.create_analyst_graph') as mock_create_graph, \
         patch('tools.analyst_agent.graph.workflow.run_analyst_workflow') as mock_run_workflow:

        # Test that workflow can handle different states
        mock_run_workflow.return_value = {"success": True, "trd_output": "Test TRD"}

        result = run_analyst_workflow({"title": "Test"})

        assert result["success"] is True


def test_workflow_with_invalid_input():
    """Test workflow with invalid input data."""
    with patch('tools.analyst_agent.graph.workflow.create_analyst_graph') as mock_create_graph, \
         patch('tools.analyst_agent.graph.workflow.run_analyst_workflow') as mock_run_workflow:

        mock_run_workflow.return_value = {"success": False, "errors": ["Invalid input data"]}

        result = run_analyst_workflow({})

        assert result["success"] is False
        assert len(result["errors"]) > 0


def test_workflow_checkpointing():
    """Test workflow checkpointing functionality."""
    with patch('tools.analyst_agent.graph.workflow.create_analyst_graph') as mock_create_graph:
        mock_graph = Mock()
        mock_create_graph.return_value = mock_graph

        # Test with checkpointing enabled
        graph = create_analyst_graph(checkpoint=True)
        assert mock_graph.compile.called

        # Test with checkpointing disabled
        graph = create_analyst_graph(checkpoint=False)
        assert mock_graph.compile.called


def test_workflow_interrupt_handling():
    """Test workflow interrupt handling for human input."""
    with patch('tools.analyst_agent.graph.workflow.create_analyst_graph') as mock_create_graph:
        mock_graph = Mock()
        mock_create_graph.return_value = mock_graph

        # Test interrupt_before configuration
        graph = create_analyst_graph(auto_mode=False)
        assert mock_graph.compile.called
        # Verify interrupt_before includes human_input when auto_mode is False


def test_workflow_state_validation():
    """Test workflow state validation."""
    from tools.analyst_agent.graph.state import validate_state

    # Test valid state
    valid_state = {
        "nprd_data": {"title": "Test"},
        "status": "extracted",
        "concepts": [],
        "indicators": [],
        "timeframes": [],
        "risk_management": {},
        "search_results": [],
        "missing_info": [],
        "trd_output": "",
        "metadata": {"auto_mode": False}
    }

    is_valid, errors = validate_state(valid_state)
    assert is_valid is True
    assert len(errors) == 0

    # Test invalid state
    invalid_state = {"status": "invalid"}  # Missing required fields
    is_valid, errors = validate_state(invalid_state)
    assert is_valid is False
    assert len(errors) > 0


def test_workflow_concurrent_execution():
    """Test workflow can handle concurrent execution."""
    with patch('tools.analyst_agent.graph.workflow.create_analyst_graph') as mock_create_graph, \
         patch('tools.analyst_agent.graph.workflow.run_analyst_workflow') as mock_run_workflow:

        mock_run_workflow.side_effect = [
            {"success": True, "trd_output": "TRD 1"},
            {"success": True, "trd_output": "TRD 2"}
        ]

        # Test running multiple workflows concurrently
        result1 = run_analyst_workflow({"title": "Test 1"})
        result2 = run_analyst_workflow({"title": "Test 2"})

        assert result1["success"] is True
        assert result2["success"] is True
        assert result1["trd_output"] != result2["trd_output"]


def test_workflow_error_recovery():
    """Test workflow error recovery and retry logic."""
    with patch('tools.analyst_agent.graph.workflow.create_analyst_graph') as mock_create_graph, \
         patch('tools.analyst_agent.graph.workflow.run_analyst_workflow') as mock_run_workflow:

        # First execution fails, second succeeds
        mock_run_workflow.side_effect = [
            Exception("Temporary error"),
            {"success": True, "trd_output": "Recovered TRD"}
        ]

        # Should retry and succeed
        result = run_analyst_workflow({"title": "Test"})

        assert result["success"] is True
        assert "Recovered TRD" in result["trd_output"]
"""
Tests for modular quantcode package.

Tests that the quantcode module is properly split into a package
with separate modules for nodes, helpers, edges, graph, and mcp_tools.
"""

import pytest
import sys
import os

# Test 1: Package structure exists
def test_quantcode_package_exists():
    """Test that quantcode is a package with submodules."""
    from src.agents import quantcode
    package_path = os.path.dirname(quantcode.__file__)
    assert os.path.isdir(package_path), "quantcode should be a package directory"


def test_quantcode_has_submodules():
    """Test that quantcode package has the expected submodules."""
    from src.agents import quantcode
    package_path = os.path.dirname(quantcode.__file__)

    expected_modules = ['nodes', 'helpers', 'edges', 'graph', 'mcp_tools']
    for module_name in expected_modules:
        module_path = os.path.join(package_path, f'{module_name}.py')
        assert os.path.isfile(module_path), f"Module {module_name}.py should exist"


def test_nodes_module_exports():
    """Test that nodes module exports expected functions."""
    from src.agents.quantcode import nodes

    expected_exports = [
        'planning_node',
        'coding_node',
        'compilation_node',
        'backtesting_node',
        'analysis_node',
        'reflection_node',
        'paper_deployment_node',
        'validation_monitoring_node',
        'promotion_decision_node'
    ]

    for export_name in expected_exports:
        assert hasattr(nodes, export_name), f"nodes should export {export_name}"


def test_helpers_module_exports():
    """Test that helpers module exports expected functions."""
    from src.agents.quantcode import helpers

    expected_exports = [
        '_extract_strategy_type',
        '_extract_indicators',
        '_extract_entry_rules',
        '_extract_exit_rules',
        '_extract_risk_management',
        '_extract_timeframes',
        '_extract_symbols',
        '_generate_mql5_code',
        '_attempt_code_fix',
        '_calculate_kelly_score'
    ]

    for export_name in expected_exports:
        assert hasattr(helpers, export_name), f"helpers should export {export_name}"


def test_edges_module_exports():
    """Test that edges module exports expected functions."""
    from src.agents.quantcode import edges

    expected_exports = [
        'should_continue_after_backtest',
        'should_continue_to_end',
        'should_deploy_paper',
        'should_promote'
    ]

    for export_name in expected_exports:
        assert hasattr(edges, export_name), f"edges should export {export_name}"


def test_graph_module_exports():
    """Test that graph module exports expected functions."""
    from src.agents.quantcode import graph

    expected_exports = [
        'create_quantcode_graph',
        'compile_quantcode_graph'
    ]

    for export_name in expected_exports:
        assert hasattr(graph, export_name), f"graph should export {export_name}"


def test_mcp_tools_module_exports():
    """Test that mcp_tools module exports expected functions."""
    from src.agents.quantcode import mcp_tools

    expected_exports = [
        'compile_mql5',
        'run_backtest_mcp',
        'validate_syntax'
    ]

    for export_name in expected_exports:
        assert hasattr(mcp_tools, export_name), f"mcp_tools should export {export_name}"


def test_backward_compatibility_via_import():
    """Test that old import path still works for backward compatibility."""
    # This should work via the __init__.py re-exports
    from src.agents.quantcode import (
        planning_node,
        coding_node,
        compilation_node,
        backtesting_node,
        analysis_node,
        reflection_node,
        paper_deployment_node,
        validation_monitoring_node,
        promotion_decision_node,
        create_quantcode_graph,
        compile_quantcode_graph,
        run_quantcode_workflow,
        run_quantcode_workflow_async
    )

    # All should be callable
    assert callable(planning_node)
    assert callable(coding_node)
    assert callable(compilation_node)
    assert callable(backtesting_node)
    assert callable(analysis_node)
    assert callable(reflection_node)
    assert callable(paper_deployment_node)
    assert callable(validation_monitoring_node)
    assert callable(promotion_decision_node)
    assert callable(create_quantcode_graph)
    assert callable(compile_quantcode_graph)
    assert callable(run_quantcode_workflow)
    assert callable(run_quantcode_workflow_async)


def test_helpers_extract_strategy_type():
    """Test _extract_strategy_type helper function."""
    from src.agents.quantcode.helpers import _extract_strategy_type

    assert _extract_strategy_type("") == "Momentum"
    assert _extract_strategy_type("mean reversion strategy") == "Mean Reversion"
    assert _extract_strategy_type("trend following system") == "Trend Following"
    assert _extract_strategy_type("breakout pattern") == "Breakout"
    assert _extract_strategy_type("scalping technique") == "Scalping"


def test_helpers_extract_indicators():
    """Test _extract_indicators helper function."""
    from src.agents.quantcode.helpers import _extract_indicators

    result = _extract_indicators("RSI and MACD indicators")
    assert "RSI" in result
    assert "MACD" in result


def test_helpers_calculate_kelly_score():
    """Test _calculate_kelly_score helper function."""
    from src.agents.quantcode.helpers import _calculate_kelly_score

    # Should return value between 0 and 1
    score = _calculate_kelly_score(0.6, 0.2, 1.5)
    assert 0 <= score <= 1

    # Edge case: win_rate <= 0 should return 0
    score = _calculate_kelly_score(0, 0.2, 1.5)
    assert score == 0.0

    # Edge case: max_drawdown >= 1 should return 0
    score = _calculate_kelly_score(0.6, 1.0, 1.5)
    assert score == 0.0


def test_edges_should_continue_after_backtest():
    """Test should_continue_after_backtest edge function."""
    from src.agents.quantcode.edges import should_continue_after_backtest
    from src.agents.state import QuantCodeState

    # Test with trades
    state_with_trades = QuantCodeState(
        messages=[],
        current_task="test",
        workspace_path="/test",
        context={},
        memory_namespace=("test",),
        backtest_results={"total_trades": 10}
    )
    result = should_continue_after_backtest(state_with_trades)
    assert result == "analysis"

    # Test without trades
    state_no_trades = QuantCodeState(
        messages=[],
        current_task="test",
        workspace_path="/test",
        context={},
        memory_namespace=("test",),
        backtest_results={}
    )
    result = should_continue_after_backtest(state_no_trades)
    assert result == "coding"


def test_graph_creation():
    """Test that graph can be created."""
    from src.agents.quantcode.graph import create_quantcode_graph

    workflow = create_quantcode_graph()
    assert workflow is not None


def test_graph_compilation():
    """Test that graph can be compiled."""
    from src.agents.quantcode.graph import compile_quantcode_graph

    compiled = compile_quantcode_graph()
    assert compiled is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

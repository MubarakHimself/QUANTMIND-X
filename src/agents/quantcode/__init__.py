"""
QuantCode Agent Package

Modular package structure for QuantCode agent workflow.
Contains separate modules for nodes, helpers, edges, graph, and mcp_tools.

This package provides backward compatibility with the original quantcode.py module
by re-exporting all functions from their respective submodules.

Module Structure:
- nodes: LangGraph node functions for the workflow
- helpers: Helper functions for extraction and code generation
- edges: Conditional edge functions for workflow control
- graph: Graph construction and compilation
- mcp_tools: MCP tool integration wrappers
"""

# Re-export from submodules for backward compatibility
from src.agents.quantcode.nodes import (
    planning_node,
    coding_node,
    compilation_node,
    backtesting_node,
    analysis_node,
    reflection_node,
    paper_deployment_node,
    validation_monitoring_node,
    promotion_decision_node
)

from src.agents.quantcode.helpers import (
    _extract_strategy_type,
    _extract_indicators,
    _extract_entry_rules,
    _extract_exit_rules,
    _extract_risk_management,
    _extract_timeframes,
    _extract_symbols,
    _generate_mql5_code,
    _attempt_code_fix,
    _calculate_kelly_score
)

from src.agents.quantcode.edges import (
    should_continue_after_backtest,
    should_continue_to_end,
    should_deploy_paper,
    should_promote
)

from src.agents.quantcode.graph import (
    create_quantcode_graph,
    compile_quantcode_graph
)

from src.agents.quantcode.mcp_tools import (
    compile_mql5,
    run_backtest_mcp,
    validate_syntax,
    _get_paper_trading_validator,
    _get_paper_trading_deployer
)

# Re-export execution functions from original module
from src.agents.quantcode.execution import (
    run_quantcode_workflow,
    run_quantcode_workflow_async
)

__all__ = [
    # Nodes
    'planning_node',
    'coding_node',
    'compilation_node',
    'backtesting_node',
    'analysis_node',
    'reflection_node',
    'paper_deployment_node',
    'validation_monitoring_node',
    'promotion_decision_node',
    # Helpers
    '_extract_strategy_type',
    '_extract_indicators',
    '_extract_entry_rules',
    '_extract_exit_rules',
    '_extract_risk_management',
    '_extract_timeframes',
    '_extract_symbols',
    '_generate_mql5_code',
    '_attempt_code_fix',
    '_calculate_kelly_score',
    # Edges
    'should_continue_after_backtest',
    'should_continue_to_end',
    'should_deploy_paper',
    'should_promote',
    # Graph
    'create_quantcode_graph',
    'compile_quantcode_graph',
    # MCP Tools
    'compile_mql5',
    'run_backtest_mcp',
    'validate_syntax',
    '_get_paper_trading_validator',
    '_get_paper_trading_deployer',
    # Execution
    'run_quantcode_workflow',
    'run_quantcode_workflow_async',
]

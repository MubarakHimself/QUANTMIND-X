"""
DEPRECATED: This agent is deprecated.

The department-based system (floor_manager + department heads) is now canonical.
Use /api/floor-manager/* endpoints instead.

This file will be removed in a future release.
"""
"""
QuantCode Agent Workflow

Implements the QuantCode agent using LangGraph for strategy development and backtesting.

This agent:
1. Ingests TRD content
2. Generates MQL5 code
3. Invokes MT5 compiler MCP with error-handling/retry
4. Runs Backtest MCP
5. Passes results through the backtest validator

**Validates: Requirements 8.3**

NOTE: This module is maintained for backward compatibility.
The implementation has been moved to the src/agents/quantcode/ package.
"""

# Re-export everything from the new package for backward compatibility
from src.agents.quantcode import (
    # Nodes
    planning_node,
    coding_node,
    compilation_node,
    backtesting_node,
    analysis_node,
    reflection_node,
    paper_deployment_node,
    validation_monitoring_node,
    promotion_decision_node,
    # Helpers
    _extract_strategy_type,
    _extract_indicators,
    _extract_entry_rules,
    _extract_exit_rules,
    _extract_risk_management,
    _extract_timeframes,
    _extract_symbols,
    _generate_mql5_code,
    _attempt_code_fix,
    _calculate_kelly_score,
    # Edges
    should_continue_after_backtest,
    should_continue_to_end,
    should_deploy_paper,
    should_promote,
    # Graph
    create_quantcode_graph,
    compile_quantcode_graph,
    # MCP Tools
    compile_mql5,
    run_backtest_mcp,
    validate_syntax,
    _get_paper_trading_validator,
    _get_paper_trading_deployer,
    # Execution
    run_quantcode_workflow,
    run_quantcode_workflow_async,
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

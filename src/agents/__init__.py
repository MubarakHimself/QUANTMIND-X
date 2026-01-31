"""
Agent implementations for QuantMindX Unified Backend.

Exports all agent workflows and state definitions.
"""

from src.agents.state import (
    AgentState,
    AnalystState,
    QuantCodeState,
    ExecutorState,
    RouterState
)

from src.agents.analyst import (
    create_analyst_graph,
    compile_analyst_graph,
    run_analyst_workflow
)

from src.agents.quantcode import (
    create_quantcode_graph,
    compile_quantcode_graph,
    run_quantcode_workflow
)

from src.agents.executor import (
    create_executor_graph,
    compile_executor_graph,
    run_executor_workflow
)

from src.agents.router import (
    create_router_graph,
    compile_router_graph,
    run_router_workflow
)

__all__ = [
    # State definitions
    "AgentState",
    "AnalystState",
    "QuantCodeState",
    "ExecutorState",
    "RouterState",
    
    # Analyst agent
    "create_analyst_graph",
    "compile_analyst_graph",
    "run_analyst_workflow",
    
    # QuantCode agent
    "create_quantcode_graph",
    "compile_quantcode_graph",
    "run_quantcode_workflow",
    
    # Executor agent
    "create_executor_graph",
    "compile_executor_graph",
    "run_executor_workflow",
    
    # Router agent
    "create_router_graph",
    "compile_router_graph",
    "run_router_workflow",
]

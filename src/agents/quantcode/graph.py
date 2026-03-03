"""
Graph Construction for QuantCode workflow.

Contains all graph construction and compilation functions.
"""

import logging
from typing import Any, Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.agents.state import QuantCodeState
from src.agents.quantcode import nodes
from src.agents.quantcode import edges

logger = logging.getLogger(__name__)


def create_quantcode_graph() -> StateGraph:
    """Create the QuantCode agent workflow graph."""
    workflow = StateGraph(QuantCodeState)

    # Add nodes
    workflow.add_node("planning", nodes.planning_node)
    workflow.add_node("coding", nodes.coding_node)
    workflow.add_node("compilation", nodes.compilation_node)
    workflow.add_node("backtesting", nodes.backtesting_node)
    workflow.add_node("analysis", nodes.analysis_node)
    workflow.add_node("reflection", nodes.reflection_node)
    workflow.add_node("paper_deployment", nodes.paper_deployment_node)
    workflow.add_node("validation_monitoring", nodes.validation_monitoring_node)
    workflow.add_node("promotion_decision", nodes.promotion_decision_node)

    # Set entry point
    workflow.set_entry_point("planning")

    # Add edges
    workflow.add_edge("planning", "coding")
    workflow.add_edge("coding", "compilation")
    workflow.add_conditional_edges(
        "compilation",
        lambda state: "backtesting" if state.get('context', {}).get('compilation_result', {}).get('success') else "coding",
        {
            "backtesting": "backtesting",
            "coding": "coding"
        }
    )
    workflow.add_conditional_edges(
        "backtesting",
        edges.should_continue_after_backtest,
        {
            "analysis": "analysis",
            "coding": "coding"
        }
    )
    workflow.add_edge("analysis", "reflection")
    workflow.add_conditional_edges(
        "reflection",
        edges.should_deploy_paper,
        {
            "paper_deployment": "paper_deployment",
            END: END
        }
    )
    workflow.add_edge("paper_deployment", "validation_monitoring")
    workflow.add_conditional_edges(
        "validation_monitoring",
        edges.should_promote,
        {
            "promotion_decision": "promotion_decision",
            "planning": "planning"
        }
    )
    workflow.add_edge("promotion_decision", END)

    return workflow


def compile_quantcode_graph(checkpointer: Optional[MemorySaver] = None) -> Any:
    """Compile the QuantCode agent graph with optional checkpointing."""
    workflow = create_quantcode_graph()

    if checkpointer is None:
        checkpointer = MemorySaver()

    compiled_graph = workflow.compile(checkpointer=checkpointer)

    logger.info("QuantCode agent graph compiled successfully")

    return compiled_graph

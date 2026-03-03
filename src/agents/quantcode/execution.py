"""
Execution Interface for QuantCode workflow.

Contains the run functions for executing the QuantCode agent workflow.
"""

import logging
import uuid
from typing import Dict, Any, Optional

from langchain_core.messages import HumanMessage

from src.agents.state import QuantCodeState
from src.agents.quantcode.graph import compile_quantcode_graph

logger = logging.getLogger(__name__)


def run_quantcode_workflow(
    strategy_request: str,
    trd_content: Optional[str] = None,
    workspace_path: str = "workspaces/quant",
    memory_namespace: tuple = ("memories", "quantcode", "default")
) -> Dict[str, Any]:
    """Execute the QuantCode agent workflow."""
    initial_state: QuantCodeState = {
        "messages": [HumanMessage(content=strategy_request)],
        "current_task": "strategy_development",
        "workspace_path": workspace_path,
        "context": {"trd_content": trd_content} if trd_content else {},
        "memory_namespace": memory_namespace,
        "strategy_plan": None,
        "code_implementation": None,
        "backtest_results": None,
        "analysis_report": None,
        "reflection_notes": None,
        "paper_agent_id": None,
        "validation_start_time": None,
        "validation_period_days": 30,
        "paper_trading_metrics": None,
        "promotion_approved": False,
        "trd_content": trd_content,
        "compilation_result": None
    }

    graph = compile_quantcode_graph()
    config = {"configurable": {"thread_id": "quantcode_001"}}
    final_state = graph.invoke(initial_state, config)

    logger.info(f"QuantCode workflow completed")

    return final_state


async def run_quantcode_workflow_async(
    strategy_request: str,
    trd_content: Optional[str] = None,
    workspace_path: str = "workspaces/quant",
    memory_namespace: tuple = ("memories", "quantcode", "default")
) -> Dict[str, Any]:
    """Execute the QuantCode agent workflow asynchronously."""
    initial_state: QuantCodeState = {
        "messages": [HumanMessage(content=strategy_request)],
        "current_task": "strategy_development",
        "workspace_path": workspace_path,
        "context": {"trd_content": trd_content} if trd_content else {},
        "memory_namespace": memory_namespace,
        "strategy_plan": None,
        "code_implementation": None,
        "backtest_results": None,
        "analysis_report": None,
        "reflection_notes": None,
        "paper_agent_id": None,
        "validation_start_time": None,
        "validation_period_days": 30,
        "paper_trading_metrics": None,
        "promotion_approved": False,
        "trd_content": trd_content,
        "compilation_result": None
    }

    graph = compile_quantcode_graph()
    config = {"configurable": {"thread_id": f"quantcode_{uuid.uuid4().hex[:8]}"}}
    final_state = await graph.ainvoke(initial_state, config)

    logger.info(f"QuantCode workflow completed")

    return final_state

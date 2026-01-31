"""
Router Agent

Implements the Router agent for task delegation to specialized agents.

**Validates: Requirements 8.5**
"""

import logging
from typing import Dict, Any
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.agents.state import RouterState

logger = logging.getLogger(__name__)


def classify_task_node(state: RouterState) -> Dict[str, Any]:
    """Classify incoming task and determine target agent."""
    logger.info("Router classifying task")
    
    messages = state.get('messages', [])
    last_message = messages[-1].content if messages else ""
    
    # Simple keyword-based classification
    if any(keyword in last_message.lower() for keyword in ['research', 'analysis', 'market']):
        task_type = "research"
        target_agent = "analyst"
    elif any(keyword in last_message.lower() for keyword in ['strategy', 'backtest', 'code']):
        task_type = "strategy_development"
        target_agent = "quantcode"
    elif any(keyword in last_message.lower() for keyword in ['deploy', 'execute', 'monitor']):
        task_type = "deployment"
        target_agent = "copilot"
    else:
        task_type = "general"
        target_agent = "analyst"  # Default to analyst
    
    delegation_entry = {
        "task": last_message,
        "classified_as": task_type,
        "delegated_to": target_agent,
        "timestamp": "2024-01-31T00:00:00Z"
    }
    
    message = AIMessage(content=f"Task classified as '{task_type}', delegating to {target_agent}")
    
    return {
        "messages": [message],
        "task_type": task_type,
        "target_agent": target_agent,
        "delegation_history": state.get('delegation_history', []) + [delegation_entry],
        "context": {**state.get('context', {}), "delegation": delegation_entry}
    }


def delegate_node(state: RouterState) -> Dict[str, Any]:
    """Delegate task to target agent."""
    logger.info(f"Router delegating to {state.get('target_agent')}")
    
    target_agent = state.get('target_agent')
    task_type = state.get('task_type')
    
    # In production, this would actually invoke the target agent
    delegation_result = {
        "agent": target_agent,
        "status": "delegated",
        "task_type": task_type
    }
    
    message = AIMessage(content=f"Task delegated to {target_agent} agent")
    
    return {
        "messages": [message],
        "context": {**state.get('context', {}), "delegation_result": delegation_result}
    }


def create_router_graph() -> StateGraph:
    """Create the Router agent workflow graph."""
    workflow = StateGraph(RouterState)
    
    workflow.add_node("classify", classify_task_node)
    workflow.add_node("delegate", delegate_node)
    
    workflow.set_entry_point("classify")
    workflow.add_edge("classify", "delegate")
    workflow.add_edge("delegate", END)
    
    return workflow


def compile_router_graph(checkpointer: MemorySaver = None) -> Any:
    """Compile the Router agent graph."""
    workflow = create_router_graph()
    
    if checkpointer is None:
        checkpointer = MemorySaver()
    
    compiled_graph = workflow.compile(checkpointer=checkpointer)
    logger.info("Router agent graph compiled successfully")
    
    return compiled_graph


def run_router_workflow(
    task_request: str,
    workspace_path: str = "workspaces",
    memory_namespace: tuple = ("memories", "router", "default")
) -> Dict[str, Any]:
    """Execute the Router agent workflow."""
    initial_state = RouterState(
        messages=[HumanMessage(content=task_request)],
        current_task="task_routing",
        workspace_path=workspace_path,
        context={},
        memory_namespace=memory_namespace,
        task_type=None,
        target_agent=None,
        delegation_history=[]
    )
    
    graph = compile_router_graph()
    config = {"configurable": {"thread_id": "router_001"}}
    final_state = graph.invoke(initial_state, config)
    
    logger.info(f"Router workflow completed, delegated to {final_state.get('target_agent')}")
    
    return final_state

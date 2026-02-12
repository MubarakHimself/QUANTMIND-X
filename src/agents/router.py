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
    
    # V8: Check for crypto-related tasks
    is_crypto = any(keyword in last_message.lower() for keyword in ['crypto', 'binance', 'bitcoin', 'btc', 'eth', 'ethereum'])
    
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
    
    # V8: Add broker type to context
    broker_type = "crypto" if is_crypto else "mt5"
    
    delegation_entry = {
        "task": last_message,
        "classified_as": task_type,
        "delegated_to": target_agent,
        "broker_type": broker_type,  # V8
        "timestamp": "2024-01-31T00:00:00Z"
    }
    
    message = AIMessage(content=f"Task classified as '{task_type}' ({broker_type}), delegating to {target_agent}")
    
    return {
        "messages": [message],
        "task_type": task_type,
        "target_agent": target_agent,
        "delegation_history": state.get('delegation_history', []) + [delegation_entry],
        "context": {**state.get('context', {}), "delegation": delegation_entry, "broker_type": broker_type}
    }


def delegate_node(state: RouterState) -> Dict[str, Any]:
    """
    Delegate task to target agent.
    
    V8: Supports crypto strategy deployment via BrokerRegistry.
    """
    logger.info(f"Router delegating to {state.get('target_agent')}")
    
    target_agent = state.get('target_agent')
    task_type = state.get('task_type')
    broker_type = state.get('context', {}).get('broker_type', 'mt5')
    
    # V8: If deployment task, prepare broker configuration
    if task_type == "deployment":
        delegation_result = _prepare_deployment(state, broker_type)
    else:
        # Standard delegation
        delegation_result = {
            "agent": target_agent,
            "status": "delegated",
            "task_type": task_type,
            "broker_type": broker_type
        }
    
    message = AIMessage(content=f"Task delegated to {target_agent} agent (broker: {broker_type})")
    
    return {
        "messages": [message],
        "context": {**state.get('context', {}), "delegation_result": delegation_result}
    }


def _prepare_deployment(state: RouterState, broker_type: str) -> Dict[str, Any]:
    """
    V8: Prepare deployment configuration based on broker type.
    
    Args:
        state: Router state
        broker_type: 'mt5' or 'crypto'
        
    Returns:
        Deployment configuration dictionary
    """
    try:
        from src.data.brokers import BrokerRegistry
        
        # Initialize broker registry
        registry = BrokerRegistry()
        
        # Get appropriate broker
        if broker_type == "crypto":
            # Use Binance for crypto strategies
            broker_id = "binance_spot_main"  # Default crypto broker
            logger.info(f"Preparing crypto deployment via {broker_id}")
        else:
            # Use MT5 for forex strategies
            broker_id = "exness_demo_mock"  # Default MT5 broker
            logger.info(f"Preparing MT5 deployment via {broker_id}")
        
        # Validate broker connection
        try:
            broker = registry.get_broker(broker_id)
            is_connected = broker.validate_connection()
            
            return {
                "agent": state.get('target_agent'),
                "status": "ready" if is_connected else "broker_unavailable",
                "task_type": "deployment",
                "broker_type": broker_type,
                "broker_id": broker_id,
                "broker_connected": is_connected
            }
        except Exception as e:
            logger.warning(f"Broker {broker_id} not available: {e}")
            return {
                "agent": state.get('target_agent'),
                "status": "broker_unavailable",
                "task_type": "deployment",
                "broker_type": broker_type,
                "broker_id": broker_id,
                "broker_connected": False,
                "error": str(e)
            }
            
    except Exception as e:
        logger.error(f"Error preparing deployment: {e}")
        return {
            "agent": state.get('target_agent'),
            "status": "error",
            "task_type": "deployment",
            "broker_type": broker_type,
            "error": str(e)
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

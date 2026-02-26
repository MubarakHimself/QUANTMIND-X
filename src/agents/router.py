"""
Router Agent

Implements the Router agent for task delegation to specialized agents.
Uses simple routing logic instead of LangGraph.

**Validates: Requirements 8.5**

DEPRECATED: LangGraph imports removed. Use ClaudeOrchestrator instead.
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

from src.agents.state import RouterState, HumanMessage, AIMessage

# Lazy imports for backward compatibility
AgentConfig = None
CompiledAgent = None

def _get_agent_config():
    global AgentConfig
    if AgentConfig is None:
        from src.agents.config import AgentConfig as _AgentConfig
        AgentConfig = _AgentConfig
    return AgentConfig

def _get_compiled_agent():
    global CompiledAgent
    if CompiledAgent is None:
        from src.agents.compiled_agent import CompiledAgent as _CompiledAgent
        CompiledAgent = _CompiledAgent
    return CompiledAgent

logger = logging.getLogger(__name__)


# Simple in-memory state store for checkpointing
class MemorySaver:
    """Simple in-memory state persistence."""
    def __init__(self):
        self._states: Dict[str, Dict[str, Any]] = {}

    def save(self, thread_id: str, state: Dict[str, Any]) -> None:
        self._states[thread_id] = state

    def load(self, thread_id: str) -> Optional[Dict[str, Any]]:
        return self._states.get(thread_id)


def classify_task_node(state: RouterState) -> Dict[str, Any]:
    """Classify incoming task and determine target agent."""
    logger.info("Router classifying task")

    messages = state.get('messages', [])
    last_message = messages[-1].get("content", "") if messages else ""
    
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


def create_router_workflow() -> Dict[str, Any]:
    """
    Create a simple router workflow definition.

    Returns a workflow definition that can be executed step by step.
    """
    return {
        "nodes": {
            "classify": classify_task_node,
            "delegate": delegate_node,
        },
        "edges": [
            ("classify", "delegate"),
            ("delegate", END),
        ],
        "entry_point": "classify",
    }


def run_router_workflow(
    task_request: str,
    workspace_path: str = "workspaces",
    memory_namespace: tuple = ("memories", "router", "default"),
    checkpointer: Optional[MemorySaver] = None,
) -> Dict[str, Any]:
    """
    Execute the Router agent workflow using simple sequential execution.

    Args:
        task_request: The task request string
        workspace_path: Path to workspace
        memory_namespace: Memory namespace tuple
        checkpointer: Optional checkpointer for state persistence

    Returns:
        Final state dictionary
    """
    # Initialize state
    state: Dict[str, Any] = {
        "messages": [{"content": task_request, "role": "user"}],
        "current_task": "task_routing",
        "workspace_path": workspace_path,
        "context": {},
        "memory_namespace": memory_namespace,
        "task_type": None,
        "target_agent": None,
        "delegation_history": [],
    }

    # Load previous state if checkpointer provided
    if checkpointer:
        saved_state = checkpointer.load("router_001")
        if saved_state:
            state = saved_state

    try:
        # Step 1: Classify the task
        logger.info("Router classifying task")
        classify_result = classify_task_node(state)
        state.update(classify_result)

        # Step 2: Delegate to target agent
        logger.info(f"Router delegating to {state.get('target_agent')}")
        delegate_result = delegate_node(state)
        state.update(delegate_result)

        # Save final state if checkpointer provided
        if checkpointer:
            checkpointer.save("router_001", state)

        logger.info(f"Router workflow completed, delegated to {state.get('target_agent')}")

        return state

    except Exception as e:
        logger.error(f"Router workflow failed: {e}")
        raise


# Backward compatibility: compile_router_graph now returns a simple wrapper
class RouterGraphWrapper:
    """Simple wrapper that mimics compiled graph interface."""

    def __init__(self, checkpointer: Optional[MemorySaver] = None):
        self.checkpointer = checkpointer or MemorySaver()

    def invoke(self, state: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute the workflow."""
        thread_id = (config or {}).get("configurable", {}).get("thread_id", "router_001")

        # Load previous state
        saved_state = self.checkpointer.load(thread_id)
        if saved_state:
            state = {**saved_state, **state}

        try:
            # Execute workflow steps
            classify_result = classify_task_node(state)
            state.update(classify_result)

            delegate_result = delegate_node(state)
            state.update(delegate_result)

            # Save state
            self.checkpointer.save(thread_id, state)

            return state
        except Exception as e:
            logger.error(f"Router workflow failed: {e}")
            raise


def create_router_graph(checkpointer: MemorySaver = None) -> RouterGraphWrapper:
    """
    Create a router graph wrapper for backward compatibility.

    Alias for compile_router_graph() to maintain API compatibility.

    Args:
        checkpointer: Optional MemorySaver instance

    Returns:
        RouterGraphWrapper instance
    """
    return compile_router_graph(checkpointer)


def compile_router_graph(checkpointer: MemorySaver = None) -> RouterGraphWrapper:
    """
    Create a router graph wrapper for backward compatibility.

    Args:
        checkpointer: Optional MemorySaver instance

    Returns:
        RouterGraphWrapper instance
    """
    logger.info("Router agent workflow wrapper created")
    return RouterGraphWrapper(checkpointer or MemorySaver())


# =============================================================================
# Factory-Based Agent Creation (Phase 4)
# =============================================================================

def create_router_from_config(config) -> "CompiledAgent":
    """
    Create a Router agent from configuration using the factory pattern.

    Args:
        config: AgentConfig instance with router configuration

    Returns:
        CompiledAgent instance

    Raises:
        ValueError: If config is not for a router agent

    DEPRECATED: Use ClaudeOrchestrator instead.
    """
    import warnings

    if config.agent_type != "router":
        raise ValueError(f"Expected agent_type='router', got '{config.agent_type}'")

    warnings.warn(
        "create_router_from_config() uses the factory pattern. "
        "Consider using ClaudeOrchestrator instead.",
        DeprecationWarning,
        stacklevel=2
    )

    from src.agents.factory import get_factory
    from src.agents.di_container import get_container

    container = get_container()
    factory = get_factory(container)

    # Try to add observers if available
    try:
        from src.agents.observers.logging_observer import LoggingObserver
        from src.agents.observers.prometheus_observer import PrometheusObserver
        if not container.get_observers():
            container.add_observer(LoggingObserver())
            container.add_observer(PrometheusObserver())
    except ImportError:
        pass  # Observers not available

    agent = factory.create(config)

    logger.info(f"Created router agent from config: {config.agent_id}")

    return agent


def create_router_agent(
    agent_id: str = "router_001",
    name: str = "Router Agent",
    llm_model: str = "anthropic/claude-sonnet-4",
    temperature: float = 0.0,
    **kwargs
) -> "CompiledAgent":
    """
    Convenience function to create a Router agent.

    DEPRECATED: Use ClaudeOrchestrator instead.
    """
    AgentConfig = _get_agent_config()
    config = AgentConfig(
        agent_id=agent_id,
        agent_type="router",
        name=name,
        llm_model=llm_model,
        temperature=temperature,
        tools=[],  # Router uses classification, not direct tools
        **kwargs
    )

    return create_router_from_config(config)

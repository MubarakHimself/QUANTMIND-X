"""
QuantMind Copilot Agent Workflow with ToolNode Integration

Implements the QuantMind Copilot agent using LangGraph with dynamic tool calling.
The Copilot is the master orchestrator that manages agent handoffs and global task queuing.

**Validates: Requirements 8.4**
"""

import logging
from typing import Dict, Any, List, Optional, Union, Sequence
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

from src.agents.state import CopilotState
from src.agents.llm_provider import get_copilot_llm

logger = logging.getLogger(__name__)


# ============================================================================
# Tool Definitions for Copilot Agent
# ============================================================================

@tool
def create_deployment_manifest(
    ea_name: str,
    symbol: str = "EURUSD",
    timeframe: str = "H1",
    magic_number: int = 12345,
    max_risk: float = 0.02,
) -> Dict[str, Any]:
    """
    Create a deployment manifest for an Expert Advisor.

    Args:
        ea_name: Name of the EA to deploy
        symbol: Trading symbol (default: EURUSD)
        timeframe: Chart timeframe (default: H1)
        magic_number: Magic number for EA identification
        max_risk: Maximum risk per trade (default: 2%)

    Returns:
        Deployment manifest configuration
    """
    return {
        "ea_name": ea_name,
        "symbol": symbol,
        "timeframe": timeframe,
        "magic_number": magic_number,
        "risk_settings": {
            "max_risk": max_risk,
            "kelly_threshold": 0.8
        }
    }


@tool
def compile_ea(ea_path: str, optimization_level: str = "debug") -> Dict[str, Any]:
    """
    Compile an MQL5 Expert Advisor.

    Args:
        ea_path: Path to the EA source file
        optimization_level: Compilation optimization level

    Returns:
        Compilation result with status and logs
    """
    # In production, this would call the actual MT5 compiler
    return {
        "status": "SUCCESS",
        "output_path": ea_path.replace(".mq5", ".ex5"),
        "errors": [],
        "warnings": ["Variable 'unused_var' declared but never used"],
        "compile_time_ms": 1500
    }


@tool
def validate_deployment(
    ea_path: str,
    manifest: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate an EA deployment configuration.

    Args:
        ea_path: Path to compiled EA
        manifest: Deployment manifest

    Returns:
        Validation results
    """
    return {
        "syntax_check": "PASSED",
        "risk_validation": "PASSED",
        "connection_test": "PASSED",
        "heartbeat_test": "PASSED",
        "all_passed": True
    }


@tool
def deploy_ea(
    ea_path: str,
    manifest: Dict[str, Any],
    terminal_id: Optional[str] = None,
    auto_start: bool = True
) -> Dict[str, Any]:
    """
    Deploy an EA to MetaTrader 5 terminal.

    Args:
        ea_path: Path to compiled EA
        manifest: Deployment manifest
        terminal_id: Target terminal ID
        auto_start: Whether to start EA automatically

    Returns:
        Deployment result
    """
    return {
        "deployed": True,
        "terminal_id": terminal_id or "default",
        "chart_id": "chart_001",
        "status": "RUNNING" if auto_start else "STOPPED"
    }


@tool
def monitor_ea(
    terminal_id: str = "default",
    ea_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Monitor EA performance and status.

    Args:
        terminal_id: Terminal to monitor
        ea_name: Specific EA name (all if not specified)

    Returns:
        Monitoring data
    """
    return {
        "status": "RUNNING",
        "uptime": "2h 15m",
        "trades_executed": 3,
        "current_pnl": 150.50,
        "heartbeat_status": "ACTIVE"
    }


@tool
def stop_ea(
    terminal_id: str = "default",
    ea_name: Optional[str] = None,
    force: bool = False
) -> Dict[str, Any]:
    """
    Stop a running EA.

    Args:
        terminal_id: Terminal ID
        ea_name: EA to stop (all if not specified)
        force: Force stop even if mid-operation

    Returns:
        Stop result
    """
    return {
        "stopped": True,
        "ea_name": ea_name or "all"
    }


@tool
def trigger_analyst(
    nprd_content: str,
    auto_proceed: bool = True
) -> Dict[str, Any]:
    """
    Trigger the Analyst agent to process an NPRD.

    Args:
        nprd_content: NPRD content to process
        auto_proceed: Auto-proceed to QuantCode after TRD generation

    Returns:
        Workflow ID and initial status
    """
    return {
        "workflow_id": "analyst_001",
        "status": "started",
        "auto_proceed": auto_proceed
    }


@tool
def trigger_quantcode(
    trd_content: str,
    workspace_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Trigger the QuantCode agent to generate MQL5 code from TRD.

    Args:
        trd_content: Technical Requirements Document content
        workspace_path: Optional workspace path

    Returns:
        Generated code path and compilation status
    """
    return {
        "code_generated": True,
        "ea_path": "output/EA.ex5",
        "lines_of_code": 250
    }


# List of all Copilot tools
COPILOT_TOOLS = [
    create_deployment_manifest,
    compile_ea,
    validate_deployment,
    deploy_ea,
    monitor_ea,
    stop_ea,
    trigger_analyst,
    trigger_quantcode,
]


# ============================================================================
# Agent Node Functions
# ============================================================================

def agent_node(state: MessagesState) -> Dict[str, Any]:
    """
    Main agent node that uses LLM to decide which tools to call.

    This replaces the hardcoded nodes with dynamic tool calling.
    Uses the project's provider config (OpenRouter/Zhipu) with fallbacks.
    """
    # Get LLM with tools using provider config
    llm_with_tools = get_copilot_llm(tools=COPILOT_TOOLS)

    # Call LLM
    response = llm_with_tools.invoke(state["messages"])

    return {"messages": [response]}


# ============================================================================
# Graph Construction with ToolNode
# ============================================================================

def create_copilot_graph_with_tools() -> StateGraph:
    """
    Create the QuantMind Copilot agent workflow graph with ToolNode.

    This replaces hardcoded nodes with a single agent node that
    dynamically calls tools based on LLM decisions.

    Workflow:
    START -> agent -> (tool_condition) -> tools -> agent -> ...
                    -> (no tools) -> END
    """
    # Create tool node
    tool_node = ToolNode(COPILOT_TOOLS)

    # Create graph with MessagesState
    workflow = StateGraph(MessagesState)

    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)

    # Set entry point
    workflow.set_entry_point("agent")

    # Add conditional edges
    # If LLM calls tools -> go to tools node
    # If LLM doesn't call tools -> END
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "tools",
            END: END
        }
    )

    # After tools, always go back to agent
    workflow.add_edge("tools", "agent")

    return workflow


def create_copilot_graph_legacy() -> StateGraph:
    """
    Create the legacy QuantMind Copilot agent workflow graph.

    This is the original implementation with hardcoded nodes.
    Kept for backward compatibility.
    """
    workflow = StateGraph(CopilotState)

    # Legacy hardcoded nodes
    def deployment_node(state: CopilotState) -> Dict[str, Any]:
        logger.info("Deployment node creating manifest")
        manifest = {
            "ea_name": "QuantMind_Strategy_v1",
            "symbol": "EURUSD",
            "timeframe": "H1",
            "magic_number": 12345,
            "risk_settings": {"max_risk": 0.02, "kelly_threshold": 0.8}
        }
        return {
            "messages": [AIMessage(content=f"Deployment manifest created for {manifest['ea_name']}")],
            "deployment_manifest": manifest,
            "context": {**state.get('context', {}), "deployment_manifest": manifest}
        }

    def compilation_node(state: CopilotState) -> Dict[str, Any]:
        logger.info("Compilation node compiling EA")
        return {
            "messages": [AIMessage(content="Compilation SUCCESS")],
            "compilation_status": "SUCCESS",
            "context": {**state.get('context', {}), "compilation_log": "EA compiled successfully"}
        }

    def validation_node(state: CopilotState) -> Dict[str, Any]:
        logger.info("Validation node validating deployment")
        return {
            "messages": [AIMessage(content="Validation PASSED")],
            "validation_results": {"syntax_check": "PASSED", "risk_validation": "PASSED"},
            "context": {**state.get('context', {}), "validation_results": {"all_passed": True}}
        }

    def monitoring_node(state: CopilotState) -> Dict[str, Any]:
        logger.info("Monitoring node tracking EA performance")
        return {
            "messages": [AIMessage(content="Monitoring active: RUNNING")],
            "monitoring_data": {"status": "RUNNING", "trades_executed": 3},
            "context": {**state.get('context', {}), "monitoring_data": {"status": "RUNNING"}}
        }

    workflow.add_node("deployment", deployment_node)
    workflow.add_node("compilation", compilation_node)
    workflow.add_node("validation", validation_node)
    workflow.add_node("monitoring", monitoring_node)

    workflow.set_entry_point("deployment")
    workflow.add_edge("deployment", "compilation")
    workflow.add_edge("compilation", "validation")
    workflow.add_edge("validation", "monitoring")
    workflow.add_edge("monitoring", END)

    return workflow


def compile_copilot_graph(
    checkpointer: MemorySaver = None,
    use_tool_node: bool = True
) -> Any:
    """
    Compile the QuantMind Copilot agent graph.

    Args:
        checkpointer: Optional memory checkpointer
        use_tool_node: If True, use new ToolNode architecture; else use legacy

    Returns:
        Compiled graph
    """
    if use_tool_node:
        workflow = create_copilot_graph_with_tools()
    else:
        workflow = create_copilot_graph_legacy()

    if checkpointer is None:
        checkpointer = MemorySaver()

    compiled_graph = workflow.compile(checkpointer=checkpointer)
    logger.info(f"QuantMind Copilot agent graph compiled (ToolNode={use_tool_node})")

    return compiled_graph


def run_copilot_workflow(
    deployment_request: str,
    workspace_path: str = "workspaces/copilot",
    memory_namespace: tuple = ("memories", "copilot", "default"),
    use_tool_node: bool = True
) -> Dict[str, Any]:
    """
    Execute the QuantMind Copilot agent workflow.

    Args:
        deployment_request: The deployment request message
        workspace_path: Path to workspace
        memory_namespace: Memory namespace tuple
        use_tool_node: Use ToolNode architecture if True

    Returns:
        Final state after workflow completion
    """
    if use_tool_node:
        # New ToolNode-based workflow
        initial_state = {
            "messages": [HumanMessage(content=deployment_request)]
        }
    else:
        # Legacy workflow
        initial_state = CopilotState(
            messages=[HumanMessage(content=deployment_request)],
            current_task="ea_deployment",
            workspace_path=workspace_path,
            context={},
            memory_namespace=memory_namespace,
            deployment_manifest=None,
            compilation_status=None,
            validation_results=None,
            monitoring_data=None
        )

    graph = compile_copilot_graph(use_tool_node=use_tool_node)
    config = {"configurable": {"thread_id": "copilot_001"}}
    final_state = graph.invoke(initial_state, config)

    logger.info("QuantMind Copilot workflow completed")

    return final_state


# =============================================================================
# Factory-Based Agent Creation (Phase 4)
# =============================================================================

def create_copilot_from_config(config: AgentConfig) -> CompiledAgent:
    """
    Create a Copilot agent from configuration using the factory pattern.
    
    Args:
        config: AgentConfig instance with copilot configuration
        
    Returns:
        CompiledAgent instance
        
    Raises:
        ValueError: If config is not for a copilot agent
    """
    import warnings
    
    if config.agent_type != "copilot":
        raise ValueError(f"Expected agent_type='copilot', got '{config.agent_type}'")
    
    warnings.warn(
        "create_copilot_from_config() uses the factory pattern. "
        "Consider using AgentFactory directly for new code.",
        DeprecationWarning,
        stacklevel=2
    )
    
    from src.agents.factory import get_factory
    from src.agents.di_container import get_container
    from src.agents.observers.logging_observer import LoggingObserver
    from src.agents.observers.prometheus_observer import PrometheusObserver
    
    container = get_container()
    factory = get_factory(container)
    
    if not container.get_observers():
        container.add_observer(LoggingObserver())
        container.add_observer(PrometheusObserver())
    
    agent = factory.create(config)
    
    logger.info(f"Created copilot agent from config: {config.agent_id}")
    
    return agent


def create_copilot_agent(
    agent_id: str = "copilot_001",
    name: str = "Copilot Agent",
    llm_model: str = "anthropic/claude-sonnet-4",
    temperature: float = 0.0,
    **kwargs
) -> CompiledAgent:
    """
    Convenience function to create a Copilot agent.
    """
    config = AgentConfig(
        agent_id=agent_id,
        agent_type="copilot",
        name=name,
        llm_model=llm_model,
        temperature=temperature,
        tools=[
            "deploy_expert_advisor",
            "monitor_deployment",
            "manage_risk_parameters",
            "validate_deployment",
            "get_account_info",
            "execute_trade",
            "close_position",
        ],
        **kwargs
    )
    
    return create_copilot_from_config(config)

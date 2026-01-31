"""
Agent State Management for LangGraph

Defines the state structure for all agents in the system.

**Validates: Property 16: Agent State Persistence**
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional, Tuple
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    LangGraph agent state with message accumulation.
    
    **Validates: Requirements 8.1**
    
    The add_messages annotation ensures messages are accumulated rather than replaced,
    allowing agents to maintain conversation history across state updates.
    """
    messages: Annotated[List[BaseMessage], add_messages]
    current_task: Optional[str]
    workspace_path: str
    context: Dict[str, Any]
    memory_namespace: Tuple[str, ...]


class AnalystState(AgentState):
    """
    State for Analyst agent workflow.
    
    **Validates: Requirements 8.2**
    """
    research_query: Optional[str]
    extracted_data: Optional[Dict[str, Any]]
    synthesis_result: Optional[str]
    validation_status: Optional[str]


class QuantCodeState(AgentState):
    """
    State for QuantCode agent workflow.
    
    **Validates: Requirements 8.3**
    """
    strategy_plan: Optional[str]
    code_implementation: Optional[str]
    backtest_results: Optional[Dict[str, Any]]
    analysis_report: Optional[str]
    reflection_notes: Optional[str]


class ExecutorState(AgentState):
    """
    State for Executor agent workflow.
    
    **Validates: Requirements 8.4**
    """
    deployment_manifest: Optional[Dict[str, Any]]
    compilation_status: Optional[str]
    validation_results: Optional[Dict[str, Any]]
    monitoring_data: Optional[Dict[str, Any]]


class RouterState(AgentState):
    """
    State for Router agent.
    
    **Validates: Requirements 8.5**
    """
    task_type: Optional[str]
    target_agent: Optional[str]
    delegation_history: List[Dict[str, Any]]

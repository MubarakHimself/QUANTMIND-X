"""
Agent State Management

Defines the state structure for all agents in the system.
Uses simple dataclasses instead of LangGraph for cleaner architecture.

**Validates: Property 16: Agent State Persistence**

DEPRECATED: LangGraph imports removed. Use ClaudeOrchestrator instead.
"""

from typing import TypedDict, List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime


# Simple message type for portability
@dataclass
class BaseMessage:
    """Base message class for agent communication."""
    content: str
    role: str = "user"

    def to_dict(self) -> Dict[str, Any]:
        return {"content": self.content, "role": self.role}


@dataclass
class HumanMessage(BaseMessage):
    """Human message."""
    role: str = "user"


@dataclass
class AIMessage(BaseMessage):
    """AI message."""
    role: str = "assistant"


class MessagesState(TypedDict):
    """
    Simple messages state for basic agent workflows.

    Contains only the messages field.
    """
    messages: List[Dict[str, Any]]


class AgentState(TypedDict):
    """
    Agent state with message history.

    **Validates: Requirements 8.1**
    """
    messages: List[Dict[str, Any]]
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
    paper_agent_id: Optional[str]
    validation_start_time: Optional[datetime]
    validation_period_days: int
    paper_trading_metrics: Optional[Dict[str, Any]]
    promotion_approved: bool


class CopilotState(AgentState):
    """
    State for QuantMind Copilot agent workflow.
    
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


class ExecutorState(AgentState):
    """
    State for Executor agent workflow.
    
    Handles trade execution, position management, and PropCommander integration.
    
    **Validates: Requirements 8.6**
    """
    trade_proposal: Optional[Dict[str, Any]]
    execution_status: Optional[str]
    position_updates: List[Dict[str, Any]]
    risk_approval: Optional[bool]
    broker_id: Optional[str]


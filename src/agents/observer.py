"""
Agent Observer Abstract Base Class

Defines the interface for agent lifecycle and invocation observers.

**Validates: Phase 2.1 - Observer Base Class**
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from datetime import datetime

from src.agents.config import AgentConfig

logger = logging.getLogger(__name__)


class AgentObserver(ABC):
    """
    Abstract base class for agent observers.
    
    Observers can track agent lifecycle events, invocations,
    tool calls, and errors for monitoring and debugging.
    """
    
    @abstractmethod
    def on_agent_created(self, agent_id: str, config: AgentConfig) -> None:
        """
        Called when an agent is created.
        
        Args:
            agent_id: Unique agent identifier
            config: Agent configuration
        """
        pass
    
    @abstractmethod
    def on_agent_destroyed(self, agent_id: str) -> None:
        """
        Called when an agent is destroyed/cleaned up.
        
        Args:
            agent_id: Unique agent identifier
        """
        pass
    
    @abstractmethod
    def on_invocation_start(
        self,
        agent_id: str,
        invocation_id: str
    ) -> None:
        """
        Called when an agent invocation starts.
        
        Args:
            agent_id: Unique agent identifier
            invocation_id: Unique invocation identifier
        """
        pass
    
    @abstractmethod
    def on_invocation_complete(
        self,
        agent_id: str,
        invocation_id: str,
        duration: float
    ) -> None:
        """
        Called when an agent invocation completes successfully.
        
        Args:
            agent_id: Unique agent identifier
            invocation_id: Unique invocation identifier
            duration: Invocation duration in seconds
        """
        pass
    
    @abstractmethod
    def on_invocation_error(
        self,
        agent_id: str,
        invocation_id: str,
        error: Exception
    ) -> None:
        """
        Called when an agent invocation fails.
        
        Args:
            agent_id: Unique agent identifier
            invocation_id: Unique invocation identifier
            error: Exception that occurred
        """
        pass
    
    @abstractmethod
    def on_tool_call(
        self,
        agent_id: str,
        tool_name: str,
        args: Dict[str, Any]
    ) -> None:
        """
        Called when an agent calls a tool.
        
        Args:
            agent_id: Unique agent identifier
            tool_name: Name of the tool called
            args: Tool arguments
        """
        pass
    
    @abstractmethod
    def on_agent_invoke(
        self,
        agent_id: str,
        state: Dict[str, Any]
    ) -> None:
        """
        Called when the agent LLM is invoked.
        
        Args:
            agent_id: Unique agent identifier
            state: Current agent state
        """
        pass
    
    @abstractmethod
    def on_agent_response(
        self,
        agent_id: str,
        response: Any
    ) -> None:
        """
        Called when the agent LLM returns a response.
        
        Args:
            agent_id: Unique agent identifier
            response: LLM response
        """
        pass
    
    @abstractmethod
    def on_agent_error(
        self,
        agent_id: str,
        error: Exception
    ) -> None:
        """
        Called when an error occurs during agent execution.
        
        Args:
            agent_id: Unique agent identifier
            error: Exception that occurred
        """
        pass


class EventLogger:
    """
    Helper class for logging agent events with structured data.
    """
    
    def __init__(self, observer_name: str):
        """
        Initialize the event logger.
        
        Args:
            observer_name: Name of the observer
        """
        self.observer_name = observer_name
        self.logger = logging.getLogger(f"{__name__}.{observer_name}")
    
    def log_event(
        self,
        event_type: str,
        agent_id: str,
        **kwargs
    ) -> None:
        """
        Log an event with structured data.
        
        Args:
            event_type: Type of event
            agent_id: Agent identifier
            **kwargs: Additional event data
        """
        extra = {
            "observer": self.observer_name,
            "event_type": event_type,
            "agent_id": agent_id,
            "timestamp": datetime.utcnow().isoformat(),
            **kwargs
        }
        
        self.logger.debug(
            f"Agent event: {event_type}",
            extra=extra
        )

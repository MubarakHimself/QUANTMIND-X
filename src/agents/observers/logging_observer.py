"""
Logging Observer for Agent Factory

Provides structured logging for agent lifecycle and invocation events.

**Validates: Phase 2.3 - Logging Observer**
"""

import logging
from typing import Any, Dict
from datetime import datetime

from src.agents.observer import AgentObserver, EventLogger
from src.agents.config import AgentConfig

logger = logging.getLogger(__name__)


class LoggingObserver(AgentObserver):
    """
    Structured logging observer for agents.
    
    Logs agent lifecycle and invocation events with structured data.
    """
    
    def __init__(
        self,
        log_level: str = "INFO",
        logger_name: str = "quantmind.agents"
    ):
        """
        Initialize the logging observer.
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            logger_name: Name of the logger to use
        """
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)
        self.logger = logging.getLogger(logger_name)
        self.event_logger = EventLogger(self.__class__.__name__)
        
        # Track active invocations
        self._active_invocations: Dict[str, datetime] = {}
        
        logger.info(f"LoggingObserver initialized with level: {log_level}")
    
    def on_agent_created(self, agent_id: str, config: AgentConfig) -> None:
        """
        Log agent creation.
        
        Args:
            agent_id: Agent identifier
            config: Agent configuration
        """
        self.logger.log(
            self.log_level,
            f"Agent created: {agent_id}",
            extra={
                "event": "agent_created",
                "agent_id": agent_id,
                "agent_type": config.agent_type,
                "agent_name": config.name,
                "llm_model": config.llm_model,
                "tools_count": len(config.tools),
            }
        )
    
    def on_agent_destroyed(self, agent_id: str) -> None:
        """
        Log agent destruction.
        
        Args:
            agent_id: Agent identifier
        """
        self.logger.log(
            self.log_level,
            f"Agent destroyed: {agent_id}",
            extra={
                "event": "agent_destroyed",
                "agent_id": agent_id,
            }
        )
    
    def on_invocation_start(
        self,
        agent_id: str,
        invocation_id: str
    ) -> None:
        """
        Log invocation start.
        
        Args:
            agent_id: Agent identifier
            invocation_id: Invocation identifier
        """
        # Track start time
        self._active_invocations[invocation_id] = datetime.utcnow()
        
        self.logger.debug(
            f"Invocation started: {invocation_id}",
            extra={
                "event": "invocation_start",
                "agent_id": agent_id,
                "invocation_id": invocation_id,
            }
        )
    
    def on_invocation_complete(
        self,
        agent_id: str,
        invocation_id: str,
        duration: float
    ) -> None:
        """
        Log invocation completion.
        
        Args:
            agent_id: Agent identifier
            invocation_id: Invocation identifier
            duration: Invocation duration
        """
        # Clean up tracking
        if invocation_id in self._active_invocations:
            del self._active_invocations[invocation_id]
        
        self.logger.log(
            self.log_level,
            f"Invocation completed: {invocation_id} ({duration:.3f}s)",
            extra={
                "event": "invocation_complete",
                "agent_id": agent_id,
                "invocation_id": invocation_id,
                "duration": duration,
            }
        )
    
    def on_invocation_error(
        self,
        agent_id: str,
        invocation_id: str,
        error: Exception
    ) -> None:
        """
        Log invocation error.
        
        Args:
            agent_id: Agent identifier
            invocation_id: Invocation identifier
            error: Exception that occurred
        """
        # Clean up tracking
        if invocation_id in self._active_invocations:
            del self._active_invocations[invocation_id]
        
        self.logger.error(
            f"Invocation error: {invocation_id} - {error}",
            extra={
                "event": "invocation_error",
                "agent_id": agent_id,
                "invocation_id": invocation_id,
                "error_type": type(error).__name__,
                "error_message": str(error),
            },
            exc_info=True
        )
    
    def on_tool_call(
        self,
        agent_id: str,
        tool_name: str,
        args: Dict[str, Any]
    ) -> None:
        """
        Log tool call.
        
        Args:
            agent_id: Agent identifier
            tool_name: Name of the tool
            args: Tool arguments
        """
        self.logger.debug(
            f"Tool call: {tool_name}",
            extra={
                "event": "tool_call",
                "agent_id": agent_id,
                "tool_name": tool_name,
                "args": args,
            }
        )
    
    def on_agent_invoke(
        self,
        agent_id: str,
        state: Dict[str, Any]
    ) -> None:
        """
        Log agent invoke (LLM call).
        
        Args:
            agent_id: Agent identifier
            state: Current state
        """
        # Extract message count for logging
        messages = state.get("messages", [])
        message_count = len(messages)
        
        self.logger.debug(
            f"Agent invoke: {agent_id}",
            extra={
                "event": "agent_invoke",
                "agent_id": agent_id,
                "message_count": message_count,
            }
        )
    
    def on_agent_response(
        self,
        agent_id: str,
        response: Any
    ) -> None:
        """
        Log agent response.
        
        Args:
            agent_id: Agent identifier
            response: LLM response
        """
        # Extract response type and content info
        response_type = type(response).__name__
        has_tool_calls = hasattr(response, 'tool_calls') and response.tool_calls
        
        self.logger.debug(
            f"Agent response: {agent_id}",
            extra={
                "event": "agent_response",
                "agent_id": agent_id,
                "response_type": response_type,
                "has_tool_calls": has_tool_calls,
            }
        )
    
    def on_agent_error(
        self,
        agent_id: str,
        error: Exception
    ) -> None:
        """
        Log agent error.
        
        Args:
            agent_id: Agent identifier
            error: Exception that occurred
        """
        self.logger.error(
            f"Agent error: {agent_id} - {error}",
            extra={
                "event": "agent_error",
                "agent_id": agent_id,
                "error_type": type(error).__name__,
                "error_message": str(error),
            },
            exc_info=True
        )

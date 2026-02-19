"""
Prometheus Observer for Agent Factory

Provides Prometheus metrics integration for agent lifecycle and invocation events.

**Validates: Phase 2.2 - Prometheus Observer**
"""

import logging
from typing import Any, Dict

from prometheus_client import Counter, Histogram

from src.agents.observer import AgentObserver
from src.agents.config import AgentConfig

logger = logging.getLogger(__name__)


# Agent-specific metrics
agent_creations_total = Counter(
    'quantmind_agent_creations_total',
    'Total number of agent creations',
    ['agent_id', 'agent_type']
)

agent_tool_calls_total = Counter(
    'quantmind_agent_tool_calls_total',
    'Total number of agent tool calls',
    ['agent_id', 'tool']
)

agent_invocations_total = Counter(
    'quantmind_agent_invocations_total',
    'Total number of agent invocations',
    ['agent_id', 'status']
)

agent_invocation_duration_seconds = Histogram(
    'quantmind_agent_invocation_duration_seconds',
    'Agent invocation duration in seconds',
    ['agent_id'],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)
)


class PrometheusObserver(AgentObserver):
    """
    Prometheus metrics observer for agents.
    
    Tracks agent creations, invocations, tool calls, and durations.
    """
    
    def __init__(self):
        """Initialize the Prometheus observer."""
        logger.info("PrometheusObserver initialized")
    
    def on_agent_created(self, agent_id: str, config: AgentConfig) -> None:
        """
        Track agent creation.
        
        Args:
            agent_id: Agent identifier
            config: Agent configuration
        """
        agent_creations_total.labels(
            agent_id=agent_id,
            agent_type=config.agent_type
        ).inc()
        
        logger.debug(f"Tracked agent creation: {agent_id}")
    
    def on_agent_destroyed(self, agent_id: str) -> None:
        """
        Track agent destruction.
        
        Args:
            agent_id: Agent identifier
        """
        # Could track destruction if needed
        logger.debug(f"Agent destroyed: {agent_id}")
    
    def on_invocation_start(
        self,
        agent_id: str,
        invocation_id: str
    ) -> None:
        """
        Track invocation start.
        
        Args:
            agent_id: Agent identifier
            invocation_id: Invocation identifier
        """
        logger.debug(f"Invocation started: {invocation_id}")
    
    def on_invocation_complete(
        self,
        agent_id: str,
        invocation_id: str,
        duration: float
    ) -> None:
        """
        Track invocation completion.
        
        Args:
            agent_id: Agent identifier
            invocation_id: Invocation identifier
            duration: Invocation duration
        """
        # Track successful invocation
        agent_invocations_total.labels(
            agent_id=agent_id,
            status='success'
        ).inc()
        
        # Track duration
        agent_invocation_duration_seconds.labels(
            agent_id=agent_id
        ).observe(duration)
        
        logger.debug(f"Invocation completed: {invocation_id} ({duration:.3f}s)")
    
    def on_invocation_error(
        self,
        agent_id: str,
        invocation_id: str,
        error: Exception
    ) -> None:
        """
        Track invocation error.
        
        Args:
            agent_id: Agent identifier
            invocation_id: Invocation identifier
            error: Exception that occurred
        """
        # Track failed invocation
        agent_invocations_total.labels(
            agent_id=agent_id,
            status='error'
        ).inc()
        
        logger.debug(f"Invocation error: {invocation_id} - {error}")
    
    def on_tool_call(
        self,
        agent_id: str,
        tool_name: str,
        args: Dict[str, Any]
    ) -> None:
        """
        Track tool call.
        
        Args:
            agent_id: Agent identifier
            tool_name: Name of the tool
            args: Tool arguments
        """
        agent_tool_calls_total.labels(
            agent_id=agent_id,
            tool=tool_name
        ).inc()
        
        logger.debug(f"Tool call: {tool_name} by {agent_id}")
    
    def on_agent_invoke(
        self,
        agent_id: str,
        state: Dict[str, Any]
    ) -> None:
        """
        Track agent invoke (LLM call).
        
        Args:
            agent_id: Agent identifier
            state: Current state
        """
        logger.debug(f"Agent invoke: {agent_id}")
    
    def on_agent_response(
        self,
        agent_id: str,
        response: Any
    ) -> None:
        """
        Track agent response.
        
        Args:
            agent_id: Agent identifier
            response: LLM response
        """
        logger.debug(f"Agent response: {agent_id}")
    
    def on_agent_error(
        self,
        agent_id: str,
        error: Exception
    ) -> None:
        """
        Track agent error.
        
        Args:
            agent_id: Agent identifier
            error: Exception that occurred
        """
        logger.debug(f"Agent error: {agent_id} - {error}")

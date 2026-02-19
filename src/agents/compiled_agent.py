"""
Compiled Agent Wrapper

Wraps compiled LangGraph agents with observability, metrics tracking,
and additional utility methods.

**Validates: Phase 1.5 - Compiled Agent Wrapper**
"""

import logging
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime

from src.agents.config import AgentConfig
from src.agents.metrics_collector import MetricsCollector
from src.agents.di_container import DependencyContainer

logger = logging.getLogger(__name__)


class CompiledAgent:
    """
    Wrapper for compiled LangGraph agents.
    
    Provides observability, metrics tracking, and utility methods
    for factory-created agents.
    """
    
    def __init__(
        self,
        config: AgentConfig,
        graph: Any,
        metrics: MetricsCollector,
        observers: List[Any],
        container: DependencyContainer,
    ):
        """
        Initialize the compiled agent.
        
        Args:
            config: Agent configuration
            graph: Compiled LangGraph
            metrics: Metrics collector
            observers: List of observers
            container: Dependency container
        """
        self.config = config
        self.graph = graph
        self.metrics = metrics
        self.observers = observers
        self.container = container
        
        # State
        self._is_running = False
        self._invocation_count = 0
        
        logger.info(f"CompiledAgent initialized: {config.agent_id}")
    
    def invoke(
        self,
        input_state: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Invoke the agent synchronously.
        
        Args:
            input_state: Input state dictionary
            config: Optional runtime configuration
            
        Returns:
            Output state dictionary
        """
        invocation_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        # Build runtime config
        runtime_config = config or {}
        runtime_config.setdefault("configurable", {})
        runtime_config["configurable"]["thread_id"] = runtime_config.get(
            "configurable", {}
        ).get("thread_id", f"{self.config.agent_id}_{self._invocation_count}")
        
        logger.info(f"Invoking agent: {self.config.agent_id} (invocation: {invocation_id})")
        
        try:
            # Notify observers
            for observer in self.observers:
                try:
                    observer.on_invocation_start(self.config.agent_id, invocation_id)
                except Exception as e:
                    logger.warning(f"Observer error: {e}")
            
            # Invoke the graph
            result = self.graph.invoke(input_state, runtime_config)
            
            # Track metrics
            duration = (datetime.utcnow() - start_time).total_seconds()
            self.metrics.track_invocation(duration, success=True)
            self._invocation_count += 1
            
            # Notify observers
            for observer in self.observers:
                try:
                    observer.on_invocation_complete(
                        self.config.agent_id,
                        invocation_id,
                        duration
                    )
                except Exception as e:
                    logger.warning(f"Observer error: {e}")
            
            logger.info(
                f"Agent invocation complete: {self.config.agent_id} "
                f"(duration: {duration:.3f}s)"
            )
            
            return result
            
        except Exception as e:
            # Track error
            duration = (datetime.utcnow() - start_time).total_seconds()
            self.metrics.track_invocation(duration, success=False, error=e)
            
            # Notify observers
            for observer in self.observers:
                try:
                    observer.on_invocation_error(
                        self.config.agent_id,
                        invocation_id,
                        e
                    )
                except Exception as obs_error:
                    logger.warning(f"Observer error: {obs_error}")
            
            logger.error(f"Agent invocation failed: {self.config.agent_id} - {e}")
            raise
    
    async def ainvoke(
        self,
        input_state: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Invoke the agent asynchronously.
        
        Args:
            input_state: Input state dictionary
            config: Optional runtime configuration
            
        Returns:
            Output state dictionary
        """
        invocation_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        # Build runtime config
        runtime_config = config or {}
        runtime_config.setdefault("configurable", {})
        runtime_config["configurable"]["thread_id"] = runtime_config.get(
            "configurable", {}
        ).get("thread_id", f"{self.config.agent_id}_{self._invocation_count}")
        
        logger.info(f"Async invoking agent: {self.config.agent_id}")
        
        try:
            # Notify observers
            for observer in self.observers:
                try:
                    observer.on_invocation_start(self.config.agent_id, invocation_id)
                except Exception as e:
                    logger.warning(f"Observer error: {e}")
            
            # Invoke the graph asynchronously
            result = await self.graph.ainvoke(input_state, runtime_config)
            
            # Track metrics
            duration = (datetime.utcnow() - start_time).total_seconds()
            self.metrics.track_invocation(duration, success=True)
            self._invocation_count += 1
            
            # Notify observers
            for observer in self.observers:
                try:
                    observer.on_invocation_complete(
                        self.config.agent_id,
                        invocation_id,
                        duration
                    )
                except Exception as e:
                    logger.warning(f"Observer error: {e}")
            
            return result
            
        except Exception as e:
            # Track error
            duration = (datetime.utcnow() - start_time).total_seconds()
            self.metrics.track_invocation(duration, success=False, error=e)
            
            # Notify observers
            for observer in self.observers:
                try:
                    observer.on_invocation_error(
                        self.config.agent_id,
                        invocation_id,
                        e
                    )
                except Exception as obs_error:
                    logger.warning(f"Observer error: {obs_error}")
            
            raise
    
    def stream(
        self,
        input_state: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Stream agent responses.
        
        Only available if enable_streaming is True in config.
        
        Args:
            input_state: Input state dictionary
            config: Optional runtime configuration
            
        Yields:
            Streamed output chunks
        """
        if not self.config.enable_streaming:
            logger.warning(
                f"Streaming not enabled for agent: {self.config.agent_id}"
            )
            # Fall back to regular invoke
            yield self.invoke(input_state, config)
            return
        
        # Build runtime config
        runtime_config = config or {}
        runtime_config.setdefault("configurable", {})
        runtime_config["configurable"]["thread_id"] = runtime_config.get(
            "configurable", {}
        ).get("thread_id", f"{self.config.agent_id}_{self._invocation_count}")
        
        logger.info(f"Streaming agent: {self.config.agent_id}")
        
        try:
            for chunk in self.graph.stream(input_state, runtime_config):
                yield chunk
                
        except Exception as e:
            logger.error(f"Streaming failed for agent: {self.config.agent_id} - {e}")
            raise
    
    def get_state(
        self,
        thread_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get the current state of the agent.
        
        Args:
            thread_id: Optional thread ID (uses agent_id if not provided)
            
        Returns:
            Current state dictionary or None
        """
        thread_id = thread_id or self.config.agent_id
        
        try:
            config = {"configurable": {"thread_id": thread_id}}
            state = self.graph.get_state(config)
            return state.values if state else None
        except Exception as e:
            logger.warning(f"Failed to get state: {e}")
            return None
    
    def update_state(
        self,
        values: Dict[str, Any],
        thread_id: Optional[str] = None,
    ) -> None:
        """
        Update the agent state.
        
        Args:
            values: Values to update
            thread_id: Optional thread ID (uses agent_id if not provided)
        """
        thread_id = thread_id or self.config.agent_id
        
        try:
            config = {"configurable": {"thread_id": thread_id}}
            self.graph.update_state(config, values)
            logger.info(f"State updated for agent: {self.config.agent_id}")
        except Exception as e:
            logger.error(f"Failed to update state: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get agent statistics.
        
        Returns:
            Statistics dictionary
        """
        return self.metrics.get_stats()
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get agent health status.
        
        Returns:
            Health status dictionary
        """
        return self.metrics.get_health_status()
    
    def cleanup(self) -> None:
        """
        Clean up the agent resources.
        
        Notifies observers and clears state.
        """
        logger.info(f"Cleaning up agent: {self.config.agent_id}")
        
        # Notify observers
        for observer in self.observers:
            try:
                observer.on_agent_destroyed(self.config.agent_id)
            except Exception as e:
                logger.warning(f"Observer cleanup error: {e}")
        
        # Reset state
        self._is_running = False
        
        logger.info(f"Agent cleaned up: {self.config.agent_id}")
    
    @property
    def agent_id(self) -> str:
        """Get agent ID."""
        return self.config.agent_id
    
    @property
    def agent_type(self) -> str:
        """Get agent type."""
        return self.config.agent_type
    
    @property
    def name(self) -> str:
        """Get agent name."""
        return self.config.name
    
    @property
    def is_running(self) -> bool:
        """Check if agent is running."""
        return self._is_running
    
    def __repr__(self) -> str:
        return (
            f"CompiledAgent(id={self.config.agent_id}, "
            f"type={self.config.agent_type}, "
            f"name={self.config.name})"
        )

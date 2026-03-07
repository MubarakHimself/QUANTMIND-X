"""
Agent Factory for Factory-Based Agent Creation.

Provides centralized agent creation with dependency injection,
observability hooks, and compatibility wrappers for older factory-driven
callers.

Canonical runtime path: prefer the ClaudeOrchestrator-based stack for all new
agent flows. This module is retained as a legacy compatibility layer while the
remaining call sites are migrated.
"""

import logging
from typing import Dict, Any, Optional, List, Callable

from src.agents.config import AgentConfig
from src.agents.di_container import DependencyContainer, get_container
from src.agents.compiled_agent import CompiledAgent
from src.agents.state import (
    AgentState,
    AnalystState,
    QuantCodeState,
    CopilotState,
    RouterState,
    ExecutorState,
    MessagesState,
)

logger = logging.getLogger(__name__)


# Simple workflow state constant
END = "__END__"


class SimpleWorkflow:
    """
    Simple workflow wrapper that mimics LangGraph interface.

    Provides basic invoke/ainvoke/stream methods without LangGraph dependency.
    """

    def __init__(
        self,
        agent_node: Callable,
        tools: List[Any],
        state_class: Any,
    ):
        """Initialize the workflow."""
        self.agent_node = agent_node
        self.tools = tools
        self.state_class = state_class
        self._checkpointer = None

    def compile(self, checkpointer: Any = None) -> "SimpleWorkflow":
        """Compile the workflow with optional checkpointer."""
        self._checkpointer = checkpointer
        return self

    def invoke(
        self,
        state: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute the workflow synchronously."""
        thread_id = (config or {}).get("configurable", {}).get("thread_id", "default")

        # Load previous state if checkpointer available
        if self._checkpointer:
            saved_state = self._checkpointer.load(thread_id)
            if saved_state:
                state = {**saved_state, **state}

        try:
            # Execute agent node
            result = self.agent_node(state)
            state.update(result)

            # Handle tool calls if present
            messages = state.get("messages", [])
            if messages and hasattr(messages[-1], "tool_calls") and messages[-1].tool_calls:
                # Execute tools (simplified - just returns tool results)
                for tool_call in messages[-1].tool_calls:
                    tool_name = tool_call.get("name", "")
                    tool_args = tool_call.get("args", {})
                    # Find and execute tool
                    for tool in self.tools:
                        if hasattr(tool, "name") and tool.name == tool_name:
                            try:
                                tool_result = tool.invoke(tool_args)
                                # Add tool result to messages
                                state["messages"].append({
                                    "role": "tool",
                                    "content": str(tool_result),
                                    "tool_call_id": tool_call.get("id", ""),
                                })
                            except Exception as e:
                                logger.warning(f"Tool {tool_name} failed: {e}")

            # Save state if checkpointer available
            if self._checkpointer:
                self._checkpointer.save(thread_id, state)

            return state

        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            raise

    async def ainvoke(
        self,
        state: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute the workflow asynchronously."""
        # For now, just call sync version
        return self.invoke(state, config)

    def stream(
        self,
        state: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
    ):
        """Stream workflow responses (yields single result)."""
        yield self.invoke(state, config)

    def get_state(self, config: Optional[Dict[str, Any]] = None) -> Any:
        """Get current state from checkpointer."""
        if not self._checkpointer:
            return None

        thread_id = (config or {}).get("configurable", {}).get("thread_id", "default")

        class StateWrapper:
            def __init__(self, values):
                self.values = values

        return StateWrapper(self._checkpointer.load(thread_id))

    def update_state(
        self,
        config: Optional[Dict[str, Any]] = None,
        values: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update state in checkpointer."""
        if not self._checkpointer or not values:
            return

        thread_id = (config or {}).get("configurable", {}).get("thread_id", "default")
        current = self._checkpointer.load(thread_id) or {}
        current.update(values)
        self._checkpointer.save(thread_id, current)


# State class mapping
STATE_CLASS_MAP = {
    "AgentState": AgentState,
    "MessagesState": MessagesState,
    "AnalystState": AnalystState,
    "QuantCodeState": QuantCodeState,
    "CopilotState": CopilotState,
    "RouterState": RouterState,
    "ExecutorState": ExecutorState,
}


class AgentFactory:
    """
    Factory for creating and configuring agents.

    Handles dependency resolution, graph building, and compilation.
    Now uses simple workflow wrappers instead of LangGraph.

    Legacy compatibility layer: prefer ClaudeOrchestrator for new code.
    """

    def __init__(self, container: Optional[DependencyContainer] = None):
        """
        Initialize the agent factory.

        Args:
            container: Optional dependency container (uses global if not provided)
        """
        self.container = container or get_container()
        logger.info("AgentFactory initialized")

    def create(self, config: AgentConfig) -> CompiledAgent:
        """
        Create a compiled agent from configuration.

        Args:
            config: Agent configuration

        Returns:
            CompiledAgent instance

        Raises:
            ValueError: If configuration is invalid
            RuntimeError: If agent creation fails
        """
        if not config.is_valid:
            raise ValueError(f"Invalid agent config: {config}")

        logger.info(f"Creating agent: {config.agent_id} (type: {config.agent_type})")

        try:
            # Step 1: Get dependencies from container
            llm = self.container.get_llm_provider(config)
            tool_registry = self.container.get_tool_registry(config)
            checkpointer = self.container.get_checkpointer(config)
            metrics = self.container.get_metrics_collector(config)
            observers = self.container.get_observers()

            # Step 2: Get state class
            state_class = self._get_state_class(config.state_class)

            # Step 3: Build the workflow wrapper
            workflow = self._build_workflow(
                config=config,
                llm=llm,
                tool_registry=tool_registry,
                state_class=state_class,
            )

            # Step 4: Compile the workflow with checkpointer
            compiled_graph = workflow.compile(checkpointer=checkpointer)

            # Step 5: Create CompiledAgent wrapper
            agent = CompiledAgent(
                config=config,
                graph=compiled_graph,
                metrics=metrics,
                observers=observers,
                container=self.container,
            )

            # Step 6: Notify observers
            for observer in observers:
                try:
                    observer.on_agent_created(config.agent_id, config)
                except Exception as e:
                    logger.warning(f"Observer {observer.__class__.__name__} failed: {e}")

            logger.info(f"Agent created successfully: {config.agent_id}")
            return agent

        except Exception as e:
            logger.error(f"Failed to create agent {config.agent_id}: {e}")
            raise RuntimeError(f"Agent creation failed: {e}") from e

    def _get_state_class(self, class_name: str):
        """
        Get the state class by name.

        Args:
            class_name: Name of the state class

        Returns:
            State class

        Raises:
            ValueError: If state class not found
        """
        if class_name not in STATE_CLASS_MAP:
            logger.warning(
                f"Unknown state class: {class_name}, using MessagesState"
            )
            return MessagesState

        return STATE_CLASS_MAP[class_name]

    def _build_workflow(
        self,
        config: AgentConfig,
        llm: Any,
        tool_registry: Any,
        state_class: Any,
    ) -> "SimpleWorkflow":
        """
        Build a simple workflow wrapper instead of LangGraph.

        Args:
            config: Agent configuration
            llm: Configured LLM
            tool_registry: Tool registry
            state_class: State class

        Returns:
            SimpleWorkflow instance
        """
        # Get tools from registry
        tools = tool_registry.get_all()

        # Create agent node with observability
        agent_node = self._create_agent_node(config, llm, tools)

        # Create simple workflow wrapper
        return SimpleWorkflow(
            agent_node=agent_node,
            tools=tools,
            state_class=state_class,
        )

    def _create_agent_node(
        self,
        config: AgentConfig,
        llm: Any,
        tools: List[Any],
    ) -> Callable:
        """
        Create the agent node function with observability hooks.

        Note: Invocation-level metrics and observers are handled by
        CompiledAgent.invoke/ainvoke to avoid double-counting.
        This function only handles agent-specific observability.

        Args:
            config: Agent configuration
            llm: Configured LLM
            tools: List of tools

        Returns:
            Agent node function
        """
        # Get observers for this agent
        observers = self.container.get_observers()
        metrics = self.container.get_metrics_collector(config)

        def agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
            """
            Agent node that invokes the LLM with observability.
            """
            try:
                # Notify agent invoke
                for observer in observers:
                    try:
                        observer.on_agent_invoke(config.agent_id, state)
                    except Exception as e:
                        logger.warning(f"Observer error: {e}")

                # Invoke LLM
                response = llm.invoke(state.get("messages", []))

                # Track tool calls if any
                if hasattr(response, "tool_calls") and response.tool_calls:
                    for tool_call in response.tool_calls:
                        tool_name = tool_call.get("name", "unknown")
                        metrics.track_tool_call(tool_name)

                        for observer in observers:
                            try:
                                observer.on_tool_call(
                                    config.agent_id,
                                    tool_name,
                                    tool_call.get("args", {})
                                )
                            except Exception as e:
                                logger.warning(f"Observer error: {e}")

                # Notify response
                for observer in observers:
                    try:
                        observer.on_agent_response(config.agent_id, response)
                    except Exception as e:
                        logger.warning(f"Observer error: {e}")

                return {"messages": [response]}

            except Exception as e:
                # Notify error
                for observer in observers:
                    try:
                        observer.on_agent_error(config.agent_id, e)
                    except Exception as obs_error:
                        logger.warning(f"Observer error: {obs_error}")

                # Re-raise
                raise

        return agent_node
    
    def create_from_yaml(
        self,
        yaml_path: str,
        **overrides
    ) -> CompiledAgent:
        """
        Create an agent from a YAML configuration file.
        
        Args:
            yaml_path: Path to YAML configuration
            **overrides: Optional configuration overrides
            
        Returns:
            CompiledAgent instance
        """
        config = AgentConfig.from_yaml(yaml_path)
        
        if overrides:
            config = config.with_overrides(**overrides)
        
        return self.create(config)


# Global factory instance
_factory: Optional[AgentFactory] = None


def get_factory(container: Optional[DependencyContainer] = None) -> AgentFactory:
    """
    Get the global agent factory.
    
    Args:
        container: Optional dependency container
        
    Returns:
        AgentFactory instance
    """
    global _factory
    if _factory is None:
        _factory = AgentFactory(container or get_container())
    return _factory


def create_agent(config: AgentConfig) -> CompiledAgent:
    """
    Convenience function to create an agent.
    
    Args:
        config: Agent configuration
        
    Returns:
        CompiledAgent instance
    """
    factory = get_factory()
    return factory.create(config)


def create_agent_from_yaml(
    yaml_path: str,
    **overrides
) -> CompiledAgent:
    """
    Convenience function to create an agent from YAML.
    
    Args:
        yaml_path: Path to YAML configuration
        **overrides: Optional configuration overrides
        
    Returns:
        CompiledAgent instance
    """
    factory = get_factory()
    return factory.create_from_yaml(yaml_path, **overrides)

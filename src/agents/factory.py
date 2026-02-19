"""
Agent Factory for Factory-Based Agent Creation

Provides centralized agent creation with dependency injection,
observability hooks, and LangGraph compilation.

**Validates: Phase 1.4 - Agent Factory**
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
    Factory for creating and configuring LangGraph agents.
    
    Handles dependency resolution, graph building, and compilation.
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
            
            # Step 3: Build the graph
            graph = self._build_graph(
                config=config,
                llm=llm,
                tool_registry=tool_registry,
                state_class=state_class,
            )
            
            # Step 4: Compile the graph
            compiled_graph = graph.compile(checkpointer=checkpointer)
            
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
    
    def _build_graph(
        self,
        config: AgentConfig,
        llm: Any,
        tool_registry: Any,
        state_class: Any,
    ) -> Any:
        """
        Build the LangGraph workflow.
        
        Args:
            config: Agent configuration
            llm: Configured LLM
            tool_registry: Tool registry
            state_class: State class
            
        Returns:
            Compiled StateGraph
        """
        from langgraph.graph import StateGraph, END
        from langgraph.prebuilt import ToolNode, tools_condition
        
        # Get tools from registry
        tools = tool_registry.get_all()
        
        # Create tool node if tools available
        tool_node = None
        if tools:
            tool_node = ToolNode(tools)
        
        # Create workflow
        workflow = StateGraph(state_class)
        
        # Add agent node with observability
        agent_node = self._create_agent_node(config, llm, tools)
        workflow.add_node("agent", agent_node)
        
        # Add tools node if tools available
        if tool_node:
            workflow.add_node("tools", tool_node)
        
        # Set entry point
        workflow.set_entry_point("agent")
        
        # Add conditional edges
        if tool_node:
            workflow.add_conditional_edges(
                "agent",
                tools_condition,
                {
                    "tools": "tools",
                    END: END
                }
            )
            workflow.add_edge("tools", "agent")
        else:
            workflow.add_edge("agent", END)
        
        return workflow
    
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

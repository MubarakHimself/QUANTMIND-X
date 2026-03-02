"""
Agent Management API Endpoints

Provides REST API endpoints for managing factory-created agents.

**Validates: Phase 5.1 - Agent Management Endpoints**
"""

import logging
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from src.agents.config import AgentConfig
from src.agents.factory import get_factory
from src.agents.registry import get_registry
from src.agents.health import get_health_checker, AgentHealthChecker
from src.agents.di_container import get_container
from src.agents.compiled_agent import CompiledAgent
from src.agents.observers.logging_observer import LoggingObserver
from src.agents.observers.prometheus_observer import PrometheusObserver

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/agent-management", tags=["agent-management"])


# =============================================================================
# Request/Response Models
# =============================================================================

class CreateAgentRequest(BaseModel):
    """Request model for creating an agent."""
    agent_id: str
    agent_type: str
    name: str
    llm_provider: str = "openrouter"
    llm_model: str = ""
    temperature: float = 0.0
    max_tokens: int = 4096
    tools: List[str] = []
    custom: Dict[str, Any] = {}


class UpdateAgentConfigRequest(BaseModel):
    """Request model for updating agent config."""
    llm_model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    custom: Optional[Dict[str, Any]] = None


class InvokeAgentRequest(BaseModel):
    """Request model for invoking an agent."""
    messages: List[Dict[str, Any]]
    config: Dict[str, Any] = {}


# =============================================================================
# Dependencies
# =============================================================================

def get_agent_registry():
    """Get the agent registry."""
    return get_registry()


def get_factory_instance():
    """Get the agent factory."""
    return get_factory()


def get_health_checker_instance():
    """Get the health checker."""
    return get_health_checker()


# =============================================================================
# Agent Endpoints
# =============================================================================

@router.get("")
async def list_agents(
    registry = Depends(get_agent_registry)
) -> Dict[str, Any]:
    """
    List all registered agents.
    
    Returns:
        Dictionary with agent list and summary
    """
    try:
        summary = registry.get_summary()
        return {
            "success": True,
            "data": summary,
        }
    except Exception as e:
        logger.error(f"Error listing agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def get_agents_health(
    health_checker = Depends(get_health_checker_instance)
) -> Dict[str, Any]:
    """
    Get health status of all agents.
    
    Returns:
        Health status for all agents
    """
    try:
        health = health_checker.check_all()
        return {
            "success": True,
            "data": health,
        }
    except Exception as e:
        logger.error(f"Error checking agent health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("")
async def create_agent(
    request: CreateAgentRequest,
    factory = Depends(get_factory_instance),
    registry = Depends(get_agent_registry)
) -> Dict[str, Any]:
    """
    Create a new agent from configuration.
    
    Args:
        request: Agent creation request
        
    Returns:
        Created agent information
    """
    try:
        # Create config from request
        config = AgentConfig(
            agent_id=request.agent_id,
            agent_type=request.agent_type,
            name=request.name,
            llm_provider=request.llm_provider,
            llm_model=request.llm_model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            tools=request.tools,
            custom=request.custom,
        )
        
        # Add default observers if none exist
        container = get_container()
        if not container.get_observers():
            container.add_observer(LoggingObserver())
            container.add_observer(PrometheusObserver())
        
        # Create agent
        agent = factory.create(config)
        
        # Register agent
        registry.register(agent)
        
        return {
            "success": True,
            "data": {
                "agent_id": agent.agent_id,
                "agent_type": agent.agent_type,
                "name": agent.name,
                "message": f"Agent {agent.agent_id} created successfully",
            },
        }
        
    except Exception as e:
        logger.error(f"Error creating agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{agent_id}")
async def get_agent(
    agent_id: str,
    registry = Depends(get_agent_registry)
) -> Dict[str, Any]:
    """
    Get details of a specific agent.
    
    Args:
        agent_id: Agent identifier
        
    Returns:
        Agent details
    """
    agent = registry.get(agent_id)
    
    if agent is None:
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")
    
    try:
        stats = agent.get_stats()
        
        return {
            "success": True,
            "data": {
                "agent_id": agent.agent_id,
                "agent_type": agent.agent_type,
                "name": agent.name,
                "config": agent.config.to_dict(),
                "stats": stats,
            },
        }
        
    except Exception as e:
        logger.error(f"Error getting agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{agent_id}/metrics")
async def get_agent_metrics(
    agent_id: str,
    registry = Depends(get_agent_registry)
) -> Dict[str, Any]:
    """
    Get metrics for a specific agent.
    
    Args:
        agent_id: Agent identifier
        
    Returns:
        Agent metrics
    """
    agent = registry.get(agent_id)
    
    if agent is None:
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")
    
    try:
        stats = agent.get_stats()
        
        return {
            "success": True,
            "data": stats,
        }
        
    except Exception as e:
        logger.error(f"Error getting metrics for {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{agent_id}/health")
async def get_agent_health(
    agent_id: str,
    health_checker = Depends(get_health_checker_instance)
) -> Dict[str, Any]:
    """
    Get health status for a specific agent.
    
    Args:
        agent_id: Agent identifier
        
    Returns:
        Agent health status
    """
    try:
        health = health_checker.check_agent(agent_id)
        
        if health.get("status") == "not_found":
            raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")
        
        return {
            "success": True,
            "data": health,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking health for {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{agent_id}/invoke")
async def invoke_agent(
    agent_id: str,
    request: InvokeAgentRequest,
    registry = Depends(get_agent_registry)
) -> Dict[str, Any]:
    """
    Invoke an agent with input state.
    
    **DEPRECATED**: Use POST /api/v2/agents/{agent_id}/run instead.
    This endpoint is maintained for backward compatibility and internally
    uses the new ClaudeOrchestrator for all agent types.
    
    Args:
        agent_id: Agent identifier
        request: Invocation request
        
    Returns:
        Agent response
    """
    # Try new orchestrator first
    try:
        from src.api.claude_agent_endpoints import run_agent_sync
        
        result = await run_agent_sync(
            agent_id=agent_id,
            messages=request.messages,
            context=request.config,
            timeout=300.0
        )
        
        return {
            "success": True,
            "data": result,
            "deprecated": True,
            "message": "This endpoint is deprecated. Use POST /api/v2/agents/{agent_id}/run",
        }
    except ValueError:
        # Agent not found in new system, fall back to legacy
        pass
    except Exception as e:
        logger.warning(f"New orchestrator failed for {agent_id}, falling back to legacy: {e}")
    
    # Legacy fallback (for agents still using LangGraph)
    agent = registry.get(agent_id)
    
    if agent is None:
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")
    
    try:
        # Pass messages as a dict with 'messages' key - CompiledAgent expects a state dict
        input_state = {"messages": request.messages}
        result = agent.invoke(input_state, request.config)
        
        return {
            "success": True,
            "data": result,
            "deprecated": True,
            "message": "This endpoint is deprecated. Use POST /api/v2/agents/{agent_id}/run",
        }
        
    except Exception as e:
        logger.error(f"Error invoking agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{agent_id}/config")
async def update_agent_config(
    agent_id: str,
    request: UpdateAgentConfigRequest,
    registry = Depends(get_agent_registry)
) -> Dict[str, Any]:
    """
    Update agent configuration.
    
    Note: This only updates runtime config. For full config reload,
    use the config hot reload feature.
    
    Args:
        agent_id: Agent identifier
        request: Config update request
        
    Returns:
        Success message
    """
    agent = registry.get(agent_id)
    
    if agent is None:
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")
    
    # Update runtime config values
    if request.llm_model is not None:
        agent.config.llm_model = request.llm_model
    if request.temperature is not None:
        agent.config.temperature = request.temperature
    if request.max_tokens is not None:
        agent.config.max_tokens = request.max_tokens
    if request.custom is not None:
        agent.config.custom.update(request.custom)
    
    return {
        "success": True,
        "message": f"Agent {agent_id} config updated",
    }


@router.delete("/{agent_id}")
async def delete_agent(
    agent_id: str,
    registry = Depends(get_agent_registry)
) -> Dict[str, Any]:
    """
    Delete/unregister an agent.
    
    Args:
        agent_id: Agent identifier
        
    Returns:
        Success message
    """
    try:
        success = registry.unregister(agent_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")
        
        return {
            "success": True,
            "message": f"Agent {agent_id} deleted",
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

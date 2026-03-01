from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional, List

from src.agents.llm_provider import (
    ProviderType,
    AGENT_MODELS,
    PROVIDER_BASE_URLS,
    has_api_key,
)

router = APIRouter(prefix="/api/agents", tags=["agents"])

# Available models per provider
PROVIDER_MODELS: Dict[str, List[dict]] = {
    ProviderType.ANTHROPIC.value: [
        {"id": "claude-opus-4-20250514", "name": "Claude Opus 4", "tier": "opus"},
        {"id": "claude-sonnet-4-20250514", "name": "Claude Sonnet 4", "tier": "sonnet"},
        {"id": "claude-haiku-3-20240307", "name": "Claude Haiku 3.5", "tier": "haiku"},
    ],
    ProviderType.MINIMAX.value: [
        {"id": "MiniMax-M2.5", "name": "MiniMax M2.5", "tier": "sonnet"},
        {"id": "MiniMax-M2.1", "name": "MiniMax M2.1", "tier": "sonnet"},
        {"id": "MiniMax-M2", "name": "MiniMax M2", "tier": "haiku"},
    ],
    ProviderType.ZHIPU.value: [
        {"id": "glm-4-plus", "name": "GLM-4 Plus", "tier": "sonnet"},
        {"id": "glm-4-flash", "name": "GLM-4 Flash", "tier": "haiku"},
    ],
    ProviderType.OPENAI.value: [
        {"id": "gpt-4o", "name": "GPT-4o", "tier": "opus"},
        {"id": "gpt-4o-mini", "name": "GPT-4o Mini", "tier": "haiku"},
    ],
}


class ModelUpdate(BaseModel):
    model: str
    provider: Optional[str] = None

# In-memory model config
_agent_models: Dict[str, dict] = {
    "copilot": {"model": "opus", "provider": "anthropic"},
    "floor_manager": {"model": "opus", "provider": "anthropic"},
    "analysis": {"model": "sonnet", "provider": "anthropic"},
    "research": {"model": "sonnet", "provider": "anthropic"},
    "risk": {"model": "sonnet", "provider": "anthropic"},
    "execution": {"model": "sonnet", "provider": "anthropic"},
    "portfolio": {"model": "sonnet", "provider": "anthropic"},
}

@router.get("/models")
async def list_agent_models():
    """Get model configuration for all agents."""
    return _agent_models


@router.get("/available-models")
async def get_available_models():
    """Get all available models based on configured API keys."""
    available = {}
    for provider in ProviderType:
        is_available = has_api_key(provider)
        models = PROVIDER_MODELS.get(provider.value, [])
        available[provider.value] = {
            "available": is_available,
            "models": models if is_available else []
        }
    return {"providers": available}

@router.get("/{agent_id}/model")
async def get_agent_model(agent_id: str):
    """Get model for specific agent."""
    if agent_id not in _agent_models:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    return _agent_models[agent_id]

@router.patch("/{agent_id}/model")
async def update_agent_model(agent_id: str, config: ModelUpdate):
    """Update model for specific agent."""
    if agent_id not in _agent_models:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

    _agent_models[agent_id] = {"model": config.model, "provider": config.provider or "anthropic"}
    return {"status": "success", "agent": agent_id, "model": config.model}

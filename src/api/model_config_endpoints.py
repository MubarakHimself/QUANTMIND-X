from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional, List
import json
import os
from pathlib import Path

from src.agents.llm_provider import ProviderType, has_api_key
from src.agents.providers.router import get_router

router = APIRouter(prefix="/api/agent-config", tags=["agent-config"])

# Config file for persistence
CONFIG_DIR = Path(".quantmind")
CONFIG_FILE = CONFIG_DIR / "agent_models.json"
CONFIG_FILE.parent.mkdir(exist_ok=True)

# Available models per provider
PROVIDER_MODELS: Dict[str, List[dict]] = {
    ProviderType.ANTHROPIC.value: [
        {"id": "claude-opus-4-20250514", "name": "Claude Opus 4", "tier": "opus"},
        {"id": "claude-sonnet-4-20250514", "name": "Claude Sonnet 4", "tier": "sonnet"},
        {"id": "claude-haiku-3-20240307", "name": "Claude Haiku 3.5", "tier": "haiku"},
    ],
    ProviderType.MINIMAX.value: [
        {"id": "MiniMax-M2.7", "name": "MiniMax M2.7", "tier": "opus"},
        {"id": "MiniMax-M2.5", "name": "MiniMax M2.5", "tier": "sonnet"},
        {"id": "MiniMax-M2.1", "name": "MiniMax M2.1", "tier": "sonnet"},
        {"id": "MiniMax-M2", "name": "MiniMax M2", "tier": "haiku"},
    ],
    ProviderType.ZHIPU.value: [
        {"id": "glm-4-plus", "name": "GLM-4 Plus (Flagship)", "tier": "opus"},
        {"id": "glm-4", "name": "GLM-4", "tier": "sonnet"},
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

# Canonical department-era agent ids.
# Legacy aliases are accepted for backward compatibility but are not persisted.
_DEFAULT_MODEL_TIERS: Dict[str, str] = {
    "floor_manager": "opus",
    "research": "sonnet",
    "development": "sonnet",
    "trading": "sonnet",
    "risk": "sonnet",
    "portfolio": "sonnet",
}

_DEFAULT_MODELS: Dict[str, dict] = {
    agent_id: {"model": "", "provider": ""}
    for agent_id in _DEFAULT_MODEL_TIERS
}

_LEGACY_AGENT_ALIASES: Dict[str, str] = {
    "copilot": "floor_manager",
    "analyst": "research",
    "quantcode": "development",
}

_LEGACY_MODEL_MAP: Dict[str, str] = {
    "opus": "opus",
    "sonnet": "sonnet",
    "haiku": "haiku",
    "glm-4-6": "opus",
    "glm-4-plus": "opus",
    "glm-4": "sonnet",
    "glm-4-flash": "haiku",
    "claude-opus-4-20250514": "opus",
    "claude-sonnet-4-20250514": "sonnet",
    "claude-haiku-3-20240307": "haiku",
}


def _resolve_default_model_entry(agent_id: str) -> dict:
    tier = _DEFAULT_MODEL_TIERS.get(agent_id, "sonnet")
    runtime = get_router().resolve_runtime_config(tier=tier)
    if runtime and (runtime.provider_type or runtime.model):
        return {
            "model": str(runtime.model or "").strip(),
            "provider": str(runtime.provider_type or "").strip().lower(),
        }
    return {"model": "", "provider": ""}


def _normalize_model_entry(agent_id: str, entry: dict) -> dict:
    default_entry = _resolve_default_model_entry(agent_id)
    provider = str(entry.get("provider") or default_entry.get("provider") or "").strip().lower()
    model = str(entry.get("model") or "").strip()

    # Preserve old shorthand tier aliases, but resolve them through the current runtime contract.
    mapped_tier = _LEGACY_MODEL_MAP.get(model.lower()) if model else None
    if mapped_tier:
        runtime = get_router().resolve_runtime_config(tier=mapped_tier)
        if runtime and (runtime.provider_type or runtime.model):
            provider = str(runtime.provider_type or provider).strip().lower()
            model = str(runtime.model or "").strip()

    if not model:
        model = str(default_entry.get("model") or "").strip()

    valid_models = {m["id"] for m in PROVIDER_MODELS.get(provider, [])}
    if valid_models and model and model not in valid_models:
        # Coerce unknown models to the provider's first supported model, without switching vendors.
        model = next(iter(valid_models))

    return {"model": model, "provider": provider}


def _canonical_agent_id(agent_id: str) -> str:
    normalized = str(agent_id or "").strip().lower()
    return _LEGACY_AGENT_ALIASES.get(normalized, normalized)


def _normalize_models(models: Dict[str, dict]) -> Dict[str, dict]:
    normalized: Dict[str, dict] = {}
    for agent_id, defaults in _DEFAULT_MODELS.items():
        normalized[agent_id] = _normalize_model_entry(agent_id, models.get(agent_id, defaults))
    return normalized


def _load_models() -> Dict[str, dict]:
    """Load model config from file or return defaults."""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    migrated: Dict[str, dict] = {}
                    for canonical in _DEFAULT_MODELS:
                        # Prefer canonical key from disk if present
                        value = loaded.get(canonical)
                        if value is None:
                            # Backward-compatibility: pull from known legacy ids
                            for legacy, mapped in _LEGACY_AGENT_ALIASES.items():
                                if mapped == canonical and legacy in loaded:
                                    value = loaded.get(legacy)
                                    break
                        if isinstance(value, dict):
                            migrated[canonical] = value

                    normalized = _normalize_models(migrated)
                    if normalized != loaded:
                        _save_models(normalized)
                    return normalized
        except (json.JSONDecodeError, IOError):
            pass
    return _normalize_models(_DEFAULT_MODELS.copy())


def _save_models(models: Dict[str, dict]) -> None:
    """Persist model config to file."""
    with open(CONFIG_FILE, "w") as f:
        json.dump(models, f, indent=2)


# In-memory model config (loaded from file or defaults)
_agent_models: Dict[str, dict] = _load_models()

@router.get("/models")
async def list_agent_models():
    """Get model configuration for all agents."""
    return _normalize_models(_agent_models)


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
    canonical = _canonical_agent_id(agent_id)
    if canonical not in _agent_models:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    return _agent_models[canonical]

@router.patch("/{agent_id}/model")
async def update_agent_model(agent_id: str, config: ModelUpdate):
    """Update model for specific agent."""
    canonical = _canonical_agent_id(agent_id)
    if canonical not in _agent_models:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

    updated = _normalize_model_entry(
        canonical,
        {"model": config.model, "provider": config.provider or ""},
    )
    _agent_models[canonical] = updated
    _save_models(_agent_models)  # Persist to file
    return {
        "status": "success",
        "agent": canonical,
        "model": updated["model"],
        "provider": updated["provider"],
    }

"""
Provider Configuration API Endpoints

Manages provider configurations (API keys, base URLs, enabled status)
for LLM providers like Anthropic, OpenAI, DeepSeek, etc.

Phase 5: Provider Configuration in Settings UI
"""

import logging
import uuid
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.database.models import ProviderConfig, get_db_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/providers", tags=["providers"])


# =============================================================================
# Request/Response Models
# =============================================================================

class ProviderConfigRequest(BaseModel):
    """Request model for creating/updating a provider configuration."""
    name: str = Field(..., description="Provider name (e.g., 'anthropic', 'openai', 'deepseek')")
    api_key: Optional[str] = Field(None, description="API key for the provider")
    base_url: Optional[str] = Field(None, description="Custom base URL for API endpoint")
    enabled: bool = Field(True, description="Whether the provider is enabled")


class ProviderConfigResponse(BaseModel):
    """Response model for provider configuration (API key masked)."""
    id: str
    name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    enabled: bool
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True


class ProviderConfigDetail(BaseModel):
    """Detailed provider config for dropdowns (includes whether it's configured)."""
    id: str
    name: str
    display_name: str
    has_api_key: bool
    enabled: bool


# Provider display names mapping
PROVIDER_DISPLAY_NAMES = {
    "anthropic": "Anthropic Claude",
    "openai": "OpenAI",
    "openrouter": "OpenRouter",
    "deepseek": "DeepSeek",
    "glm": "GLM (Zhipu)",
    "minimax": "MiniMax",
    "google": "Google Gemini",
    "azure": "Azure OpenAI",
    "cohere": "Cohere",
    "mistral": "Mistral AI",
}

# Provider to available models mapping
PROVIDER_MODELS = {
    "anthropic": [
        {"id": "claude-opus-4-6-20250514", "name": "Claude Opus 4.6"},
        {"id": "claude-sonnet-4-6-20250514", "name": "Claude Sonnet 4.6"},
        {"id": "claude-haiku-4-5-20251001", "name": "Claude Haiku 4.5"},
        {"id": "claude-3-5-sonnet-20241022", "name": "Claude 3.5 Sonnet"},
        {"id": "claude-3-opus-20240229", "name": "Claude 3 Opus"},
        {"id": "claude-3-haiku-20240307", "name": "Claude 3 Haiku"},
    ],
    "openai": [
        {"id": "gpt-4o", "name": "GPT-4o"},
        {"id": "gpt-4o-mini", "name": "GPT-4o Mini"},
        {"id": "gpt-4-turbo", "name": "GPT-4 Turbo"},
        {"id": "gpt-4", "name": "GPT-4"},
        {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo"},
    ],
    "openrouter": [
        {"id": "anthropic/claude-3.5-sonnet", "name": "Claude 3.5 Sonnet (OR)"},
        {"id": "anthropic/claude-3-opus", "name": "Claude 3 Opus (OR)"},
        {"id": "google/gemini-pro-1.5", "name": "Gemini Pro 1.5 (OR)"},
        {"id": "meta-llama/llama-3.1-70b-instruct", "name": "Llama 3.1 70B (OR)"},
        {"id": "mistralai/mistral-7b-instruct", "name": "Mistral 7B (OR)"},
    ],
    "deepseek": [
        {"id": "deepseek-chat", "name": "DeepSeek Chat"},
        {"id": "deepseek-coder", "name": "DeepSeek Coder"},
    ],
    "glm": [
        {"id": "glm-4", "name": "GLM-4"},
        {"id": "glm-4-flash", "name": "GLM-4 Flash"},
        {"id": "glm-4-plus", "name": "GLM-4 Plus"},
        {"id": "glm-3-turbo", "name": "GLM-3 Turbo"},
    ],
    "minimax": [
        {"id": "MiniMax-M2.5", "name": "MiniMax M2.5"},
        {"id": "MiniMax-M2.1", "name": "MiniMax M2.1"},
        {"id": "MiniMax-M2", "name": "MiniMax M2"},
    ],
    "google": [
        {"id": "gemini-2.0-flash-exp", "name": "Gemini 2.0 Flash Exp"},
        {"id": "gemini-1.5-pro", "name": "Gemini 1.5 Pro"},
        {"id": "gemini-1.5-flash", "name": "Gemini 1.5 Flash"},
    ],
    "azure": [
        {"id": "gpt-4o", "name": "GPT-4o (Azure)"},
        {"id": "gpt-4", "name": "GPT-4 (Azure)"},
        {"id": "gpt-35-turbo", "name": "GPT-3.5 Turbo (Azure)"},
    ],
    "cohere": [
        {"id": "command-r-plus", "name": "Command R+"},
        {"id": "command-r", "name": "Command R"},
        {"id": "command", "name": "Command"},
    ],
    "mistral": [
        {"id": "mistral-large-latest", "name": "Mistral Large"},
        {"id": "mistral-small-latest", "name": "Mistral Small"},
        {"id": "mistral-medium-latest", "name": "Mistral Medium"},
    ],
}


def mask_api_key(api_key: Optional[str]) -> Optional[str]:
    """Mask API key for display - returns None if no key, otherwise masks it."""
    if not api_key or api_key.strip() == "":
        return None
    return "***"


# =============================================================================
# API Endpoints
# =============================================================================

@router.get("", response_model=dict)
async def list_providers():
    """
    List all providers with config status (no API keys exposed).

    Returns providers with masked API keys and config status.
    """
    try:
        with get_db_session() as db:
            providers = db.query(ProviderConfig).all()

            result = []
            for p in providers:
                result.append({
                    "id": p.id,
                    "name": p.name,
                    "api_key": mask_api_key(p.api_key),
                    "base_url": p.base_url,
                    "enabled": p.enabled,
                    "created_at": p.created_at.isoformat() if p.created_at else None,
                    "updated_at": p.updated_at.isoformat() if p.updated_at else None,
                })

            return {"providers": result}
    except Exception as e:
        logger.error(f"Error listing providers: {e}")
        # Return empty list on error for graceful degradation
        return {"providers": []}


@router.post("", response_model=dict)
async def save_provider(config: ProviderConfigRequest):
    """
    Save or update provider configuration.

    If a provider with the same name exists, it will be updated.
    Otherwise, a new provider will be created.
    """
    try:
        with get_db_session() as db:
            # Check if provider with this name exists
            existing = db.query(ProviderConfig).filter(ProviderConfig.name == config.name).first()

            if existing:
                # Update existing provider
                existing.api_key = config.api_key
                existing.base_url = config.base_url
                existing.enabled = config.enabled
                db.commit()

                return {
                    "success": True,
                    "message": f"Provider '{config.name}' updated",
                    "provider": {
                        "id": existing.id,
                        "name": existing.name,
                        "api_key": mask_api_key(existing.api_key),
                        "base_url": existing.base_url,
                        "enabled": existing.enabled,
                    }
                }
            else:
                # Create new provider
                new_provider = ProviderConfig(
                    id=str(uuid.uuid4()),
                    name=config.name,
                    api_key=config.api_key,
                    base_url=config.base_url,
                    enabled=config.enabled,
                )
                db.add(new_provider)
                db.commit()
                db.refresh(new_provider)

                return {
                    "success": True,
                    "message": f"Provider '{config.name}' created",
                    "provider": {
                        "id": new_provider.id,
                        "name": new_provider.name,
                        "api_key": mask_api_key(new_provider.api_key),
                        "base_url": new_provider.base_url,
                        "enabled": new_provider.enabled,
                    }
                }
    except Exception as e:
        logger.error(f"Error saving provider: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save provider: {str(e)}")


@router.delete("/{provider_id}", response_model=dict)
async def delete_provider(provider_id: str):
    """
    Delete provider configuration by ID.
    """
    try:
        with get_db_session() as db:
            provider = db.query(ProviderConfig).filter(ProviderConfig.id == provider_id).first()

            if not provider:
                raise HTTPException(status_code=404, detail=f"Provider with ID '{provider_id}' not found")

            provider_name = provider.name
            db.delete(provider)
            db.commit()

            return {
                "success": True,
                "message": f"Provider '{provider_name}' deleted"
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting provider: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete provider: {str(e)}")


@router.get("/available", response_model=dict)
async def get_available_providers():
    """
    Get list of providers with API keys configured (for dropdowns).

    Returns all known providers with a flag indicating whether they
    have an API key configured and are enabled, plus available models.
    """
    try:
        with get_db_session() as db:
            configured_providers = db.query(ProviderConfig).all()

            # Create a lookup dict by name
            configured_by_name = {p.name: p for p in configured_providers}

            # Build list of all known providers with their status and models
            result = []
            for provider_id, display_name in PROVIDER_DISPLAY_NAMES.items():
                # Get models for this provider
                models = PROVIDER_MODELS.get(provider_id, [])

                if provider_id in configured_by_name:
                    p = configured_by_name[provider_id]
                    has_key = bool(p.api_key and p.api_key.strip())
                    result.append({
                        "id": p.id,
                        "name": p.name,
                        "display_name": display_name,
                        "has_api_key": has_key,
                        "enabled": p.enabled,
                        "available": has_key and p.enabled,
                        "models": models if has_key and p.enabled else [],
                    })
                else:
                    # Provider exists in our known list but not configured
                    result.append({
                        "id": provider_id,
                        "name": provider_id,
                        "display_name": display_name,
                        "has_api_key": False,
                        "enabled": False,
                        "available": False,
                        "models": [],
                    })

            return {"providers": result}
    except Exception as e:
        logger.error(f"Error getting available providers: {e}")
        # Return known providers on error
        return {
            "providers": [
                {
                    "id": pid,
                    "name": pid,
                    "display_name": name,
                    "has_api_key": False,
                    "enabled": False,
                    "available": False,
                    "models": [],
                }
                for pid, name in PROVIDER_DISPLAY_NAMES.items()
            ]
        }


@router.get("/{provider_id}", response_model=dict)
async def get_provider(provider_id: str):
    """
    Get a specific provider by ID.
    """
    try:
        with get_db_session() as db:
            provider = db.query(ProviderConfig).filter(ProviderConfig.id == provider_id).first()

            if not provider:
                raise HTTPException(status_code=404, detail=f"Provider with ID '{provider_id}' not found")

            return {
                "id": provider.id,
                "name": provider.name,
                "api_key": mask_api_key(provider.api_key),
                "base_url": provider.base_url,
                "enabled": provider.enabled,
                "created_at": provider.created_at.isoformat() if provider.created_at else None,
                "updated_at": provider.updated_at.isoformat() if provider.updated_at else None,
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting provider: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get provider: {str(e)}")

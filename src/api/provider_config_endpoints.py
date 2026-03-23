"""
Provider Configuration API Endpoints

Manages provider configurations (API keys, base URLs, enabled status)
for LLM providers like Anthropic, OpenAI, DeepSeek, etc.

Features:
- Encrypted API key storage at rest
- Tier assignment for model routing
- Provider availability testing

Phase 5: Provider Configuration in Settings UI
"""

import logging
import uuid
import time
from typing import List, Optional, Dict, Any

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
    provider_type: Optional[str] = Field(None, description="Provider type (e.g., 'anthropic', 'openai', 'deepseek')")
    display_name: Optional[str] = Field(None, description="Human-readable display name")
    api_key: Optional[str] = Field(None, description="API key for the provider (will be encrypted)")
    base_url: Optional[str] = Field(None, description="Custom base URL for API endpoint")
    model_list: Optional[List[Dict[str, str]]] = Field(None, description="List of available models")
    tier_assignment: Optional[Dict[str, str]] = Field(None, description="Tier to model mapping")
    is_active: bool = Field(True, description="Whether the provider is active")

    # Legacy field aliases
    name: Optional[str] = None
    enabled: Optional[bool] = None

    def __init__(self, **data):
        # Handle legacy field names
        if 'name' in data and data['name']:
            if 'provider_type' not in data or not data['provider_type']:
                data['provider_type'] = data['name']
        if 'enabled' in data and data['enabled'] is not None:
            if 'is_active' not in data:
                data['is_active'] = data['enabled']
        super().__init__(**data)


class ProviderTestRequest(BaseModel):
    """Request model for testing a provider configuration."""
    provider_type: str = Field(..., description="Provider type to test")
    api_key: Optional[str] = Field(None, description="API key to test (optional if already configured)")
    base_url: Optional[str] = Field(None, description="Base URL to test")


class ProviderConfigResponse(BaseModel):
    """Response model for provider configuration (API key masked)."""
    id: str
    provider_type: str
    display_name: Optional[str] = None
    base_url: Optional[str] = None
    is_active: bool
    tier_assignment: Optional[Dict[str, str]] = None
    model_count: int = 0
    created_at_utc: Optional[str] = None
    updated_at: Optional[str] = None

    # Legacy aliases
    @property
    def name(self) -> str:
        return self.provider_type

    @property
    def enabled(self) -> bool:
        return self.is_active

    class Config:
        from_attributes = True


class ProviderConfigDetail(BaseModel):
    """Detailed provider config for dropdowns (includes whether it's configured)."""
    id: str
    provider_type: str
    display_name: str
    has_api_key: bool
    is_active: bool
    available: bool
    tier_assignment: Optional[Dict[str, str]] = None
    model_count: int = 0


class ProviderTestResult(BaseModel):
    """Result of provider test."""
    success: bool
    latency_ms: Optional[int] = None
    model_count: Optional[int] = None
    error: Optional[str] = None


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

# Provider base URLs
PROVIDER_BASE_URLS = {
    "anthropic": "https://api.anthropic.com/v1",
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "glm": "https://open.bigmodel.cn/api/paas/v4",
    "minimax": "https://api.minimax.chat/v1",
    "google": "https://generativelanguage.googleapis.com/v1",
    "azure": "",  # Custom per deployment
    "cohere": "https://api.cohere.ai/v1",
    "mistral": "https://api.mistral.ai/v1",
}


def mask_api_key(api_key: Optional[str]) -> Optional[str]:
    """Mask API key for display - shows last 4 chars, masks the rest with bullets."""
    if not api_key or api_key.strip() == "":
        return None
    if len(api_key) <= 4:
        return "•" * len(api_key)
    return ("•" * (len(api_key) - 4)) + api_key[-4:]


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
                    "provider_type": p.provider_type,
                    "display_name": p.display_name,
                    "base_url": p.base_url,
                    "is_active": p.is_active,
                    "tier_assignment": p.tier_assignment_dict,
                    "model_count": len(p.model_list_json),
                    "created_at_utc": p.created_at_utc.isoformat() if p.created_at_utc else None,
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

    If a provider with the same type exists, it will be updated.
    Otherwise, a new provider will be created.
    API keys are encrypted at rest.
    """
    # provider_type is required for POST (create new)
    if not config.provider_type:
        raise HTTPException(status_code=400, detail="provider_type is required")
    try:
        with get_db_session() as db:
            # Check if provider with this type exists
            existing = db.query(ProviderConfig).filter(
                ProviderConfig.provider_type == config.provider_type
            ).first()

            if existing:
                # Update existing provider
                if config.api_key:
                    existing.set_api_key(config.api_key)
                if config.base_url is not None:
                    existing.base_url = config.base_url
                if config.display_name is not None:
                    existing.display_name = config.display_name
                if config.tier_assignment is not None:
                    existing.tier_assignment_dict = config.tier_assignment
                if config.model_list is not None:
                    existing.model_list_json = config.model_list
                if config.is_active is not None:
                    existing.is_active = config.is_active
                db.commit()

                return {
                    "success": True,
                    "message": f"Provider '{config.provider_type}' updated",
                    "provider": {
                        "id": existing.id,
                        "provider_type": existing.provider_type,
                        "display_name": existing.display_name,
                        "is_active": existing.is_active,
                    }
                }
            else:
                # Create new provider
                display_name = config.display_name or PROVIDER_DISPLAY_NAMES.get(
                    config.provider_type, config.provider_type
                )

                new_provider = ProviderConfig(
                    id=str(uuid.uuid4()),
                    provider_type=config.provider_type,
                    display_name=display_name,
                    base_url=config.base_url or PROVIDER_BASE_URLS.get(config.provider_type, ""),
                    is_active=config.is_active if config.is_active is not None else True,
                )

                # Encrypt and set API key
                if config.api_key:
                    new_provider.set_api_key(config.api_key)

                # Set tier assignment
                if config.tier_assignment:
                    new_provider.tier_assignment_dict = config.tier_assignment

                # Set model list
                if config.model_list:
                    new_provider.model_list_json = config.model_list
                else:
                    # Default to known models
                    new_provider.model_list_json = PROVIDER_MODELS.get(config.provider_type, [])

                db.add(new_provider)
                db.commit()
                db.refresh(new_provider)

                return {
                    "success": True,
                    "message": f"Provider '{config.provider_type}' created",
                    "provider": {
                        "id": new_provider.id,
                        "provider_type": new_provider.provider_type,
                        "display_name": new_provider.display_name,
                        "is_active": new_provider.is_active,
                    }
                }
    except Exception as e:
        logger.error(f"Error saving provider: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save provider: {str(e)}")


@router.put("/{provider_id}", response_model=dict)
async def update_provider(provider_id: str, config: ProviderConfigRequest):
    """
    Update provider configuration by ID.

    Only provided fields are updated. If api_key is absent, existing key is preserved.
    """
    try:
        with get_db_session() as db:
            provider = db.query(ProviderConfig).filter(ProviderConfig.id == provider_id).first()

            if not provider:
                raise HTTPException(status_code=404, detail=f"Provider with ID '{provider_id}' not found")

            # Update only provided fields
            if config.api_key:
                provider.set_api_key(config.api_key)
            if config.base_url is not None:
                provider.base_url = config.base_url
            if config.display_name is not None:
                provider.display_name = config.display_name
            if config.tier_assignment is not None:
                provider.tier_assignment_dict = config.tier_assignment
            if config.model_list is not None:
                provider.model_list_json = config.model_list
            if config.is_active is not None:
                provider.is_active = config.is_active

            db.commit()
            db.refresh(provider)

            return {
                "success": True,
                "message": f"Provider '{provider.provider_type}' updated",
                "provider": {
                    "id": provider.id,
                    "provider_type": provider.provider_type,
                    "display_name": provider.display_name,
                    "is_active": provider.is_active,
                }
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating provider: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update provider: {str(e)}")


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

            # Check if provider is in use (has active tier assignment)
            if provider.is_active:
                raise HTTPException(
                    status_code=409,
                    detail=f"Provider '{provider.provider_type}' is in use. Set is_active=false first."
                )

            provider_type = provider.provider_type
            db.delete(provider)
            db.commit()

            return {
                "success": True,
                "message": f"Provider '{provider_type}' deleted"
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting provider: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete provider: {str(e)}")


@router.post("/test", response_model=dict)
async def test_provider(test_req: ProviderTestRequest):
    """
    Test a provider configuration by making a minimal API call.

    Returns success status, latency, and model count.
    """
    result = await _test_provider_internal(test_req)
    return result


async def _test_provider_internal(test_req: ProviderTestRequest) -> Dict[str, Any]:
    """Internal function to test provider configuration."""
    import httpx

    api_key = test_req.api_key
    base_url = test_req.base_url or PROVIDER_BASE_URLS.get(test_req.provider_type, "")

    if not api_key:
        return {"success": False, "error": "API key required for testing"}

    if not base_url:
        return {"success": False, "error": "Base URL not configured"}

    start_time = time.time()

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Make a minimal API call based on provider type
            headers = {}

            if test_req.provider_type == "anthropic":
                headers["x-api-key"] = api_key
                headers["anthropic-version"] = "2023-06-01"
                response = await client.get(
                    f"{base_url}/models",
                    headers=headers,
                )
            elif test_req.provider_type == "openai":
                headers["Authorization"] = f"Bearer {api_key}"
                response = await client.get(
                    f"{base_url}/models",
                    headers=headers,
                )
            elif test_req.provider_type == "openrouter":
                headers["Authorization"] = f"Bearer {api_key}"
                headers["HTTP-Referer"] = "https://quantmind.app"
                headers["X-Title"] = "QuantMind"
                response = await client.get(
                    f"{base_url}/models",
                    headers=headers,
                )
            elif test_req.provider_type == "deepseek":
                headers["Authorization"] = f"Bearer {api_key}"
                response = await client.get(
                    f"{base_url}/models",
                    headers=headers,
                )
            else:
                # Generic test - just try to call the base URL
                headers["Authorization"] = f"Bearer {api_key}"
                response = await client.get(base_url, headers=headers, follow_redirects=True)

            latency_ms = int((time.time() - start_time) * 1000)

            if response.status_code == 200:
                # Try to count models
                model_count = 0
                try:
                    data = response.json()
                    if "data" in data:
                        model_count = len(data["data"])
                    elif "models" in data:
                        model_count = len(data["models"])
                except Exception:
                    pass

                return {
                    "success": True,
                    "latency_ms": latency_ms,
                    "model_count": model_count,
                }
            else:
                return {
                    "success": False,
                    "latency_ms": latency_ms,
                    "error": f"HTTP {response.status_code}: {response.text[:100]}",
                }

    except httpx.TimeoutException:
        return {
            "success": False,
            "error": "Connection timeout",
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)[:100],
        }


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

            # Create a lookup dict by type
            configured_by_type = {p.provider_type: p for p in configured_providers}

            # Build list of all known providers with their status and models
            result = []
            for provider_id, display_name in PROVIDER_DISPLAY_NAMES.items():
                # Get models for this provider
                models = PROVIDER_MODELS.get(provider_id, [])

                if provider_id in configured_by_type:
                    p = configured_by_type[provider_id]
                    decrypted_key = p.get_api_key()
                    has_key = bool(decrypted_key and decrypted_key.strip())
                    result.append({
                        "id": p.id,
                        "provider_type": p.provider_type,
                        "display_name": p.display_name or display_name,
                        "has_api_key": has_key,
                        "is_active": p.is_active,
                        "available": has_key and p.is_active,
                        "models": models if has_key and p.is_active else [],
                        "tier_assignment": p.tier_assignment_dict,
                        "model_count": len(p.model_list_json) if has_key and p.is_active else 0,
                    })
                else:
                    # Provider exists in our known list but not configured
                    result.append({
                        "id": provider_id,
                        "provider_type": provider_id,
                        "display_name": display_name,
                        "has_api_key": False,
                        "is_active": False,
                        "available": False,
                        "models": [],
                        "tier_assignment": {},
                        "model_count": 0,
                    })

            return {"providers": result}
    except Exception as e:
        logger.error(f"Error getting available providers: {e}")
        # Return known providers on error
        return {
            "providers": [
                {
                    "id": pid,
                    "provider_type": pid,
                    "display_name": name,
                    "has_api_key": False,
                    "is_active": False,
                    "available": False,
                    "models": [],
                    "tier_assignment": {},
                    "model_count": 0,
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

            # Don't expose API key in response
            return {
                "id": provider.id,
                "provider_type": provider.provider_type,
                "display_name": provider.display_name,
                "base_url": provider.base_url,
                "is_active": provider.is_active,
                "tier_assignment": provider.tier_assignment_dict,
                "model_list": provider.model_list_json,
                "model_count": len(provider.model_list_json),
                "created_at_utc": provider.created_at_utc.isoformat() if provider.created_at_utc else None,
                "updated_at": provider.updated_at.isoformat() if provider.updated_at else None,
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting provider: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get provider: {str(e)}")


@router.post("/refresh", response_model=dict)
async def refresh_providers():
    """
    Force refresh of provider configuration cache.

    This endpoint allows hot-swapping providers without restart.
    It clears the cached provider configuration and reloads from database.
    """
    try:
        from src.agents.providers import refresh_router, get_router

        router = get_router()
        last_refresh = router.last_refresh_timestamp

        refresh_router()

        return {
            "success": True,
            "message": "Provider configuration refreshed",
            "last_refresh": last_refresh,
        }
    except Exception as e:
        logger.error(f"Error refreshing providers: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to refresh providers: {str(e)}")

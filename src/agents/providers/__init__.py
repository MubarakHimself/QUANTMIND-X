"""
Provider routing package.

Provides AI provider routing for QUANTMINDX with support for:
- Multiple providers (Anthropic, OpenAI, DeepSeek, GLM, etc.)
- Automatic failover on failure
- Tier-based model selection
- Custom base URLs per provider

Story 2.3: Claude Agent SDK Provider Routing
"""

from .router import ProviderRouter, ProviderInfo, get_router, refresh_router
from .client import ProviderClient, get_client

__all__ = [
    "ProviderRouter",
    "ProviderInfo",
    "ProviderClient",
    "get_router",
    "get_client",
    "refresh_router",
]
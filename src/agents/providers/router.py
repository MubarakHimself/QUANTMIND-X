"""
Provider Router

Routes AI requests to different LLM providers based on configuration.
Supports primary/fallback selection and automatic retry on failure.

Story 2.3: Claude Agent SDK Provider Routing
"""

import logging
import time
import os
from typing import Optional, Dict, Any, List, TypeVar, Callable
from dataclasses import dataclass

from src.database.models import ProviderConfig, get_db_session

logger = logging.getLogger(__name__)

# Default cache TTL in seconds (5 minutes)
DEFAULT_CACHE_TTL = int(os.environ.get("PROVIDER_CACHE_TTL", 300))


@dataclass
class ProviderInfo:
    """Information about a configured provider."""
    id: str
    provider_type: str
    display_name: str
    base_url: str
    api_key: str
    is_active: bool
    model_list: List[Dict[str, str]]
    tier_assignment: Dict[str, str]


class ProviderRouter:
    """
    Routes AI requests to configured providers.

    Supports:
    - Primary provider selection
    - Fallback on failure
    - Custom base URLs per provider
    """

    # Default base URLs for each provider type
    DEFAULT_BASE_URLS = {
        "anthropic": "https://api.anthropic.com/v1",
        "openai": "https://api.openai.com/v1",
        "openrouter": "https://openrouter.ai/api/v1",
        "deepseek": "https://api.deepseek.com/v1",
        "glm": "https://open.bigmodel.cn/api/paas/v4",
        "minimax": "https://api.minimax.chat/v1",
        "google": "https://generativelanguage.googleapis.com/v1",
        "cohere": "https://api.cohere.ai/v1",
        "mistral": "https://api.mistral.ai/v1",
        # Azure requires custom endpoint
    }

    def __init__(self, cache_ttl: int = DEFAULT_CACHE_TTL):
        self._primary_provider: Optional[ProviderInfo] = None
        self._fallback_provider: Optional[ProviderInfo] = None
        self._all_providers: Dict[str, ProviderInfo] = {}
        self._initialized = False
        self._last_refresh: float = 0
        self._cache_ttl: int = cache_ttl

    def _load_providers(self) -> None:
        """Load all provider configurations from database."""
        try:
            with get_db_session() as db:
                providers = db.query(ProviderConfig).all()

                self._all_providers = {}
                primary_candidate = None
                fallback_candidate = None

                for p in providers:
                    if not p.is_active:
                        continue

                    api_key = p.get_api_key()
                    if not api_key or not api_key.strip():
                        logger.warning(f"Provider {p.provider_type} is active but has no API key")
                        continue

                    provider_info = ProviderInfo(
                        id=p.id,
                        provider_type=p.provider_type,
                        display_name=p.display_name or p.provider_type,
                        base_url=p.base_url or self.DEFAULT_BASE_URLS.get(p.provider_type, ""),
                        api_key=api_key,
                        is_active=p.is_active,
                        model_list=p.model_list_json or [],
                        tier_assignment=p.tier_assignment_dict or {},
                    )

                    self._all_providers[p.provider_type] = provider_info

                    # Set primary candidate (first by creation)
                    if primary_candidate is None:
                        primary_candidate = provider_info
                    # Set fallback candidate (second by creation)
                    elif fallback_candidate is None:
                        fallback_candidate = provider_info

                self._primary_provider = primary_candidate
                self._fallback_provider = fallback_candidate
                self._initialized = True
                self._last_refresh = time.time()

                logger.info(f"Loaded {len(self._all_providers)} providers: {list(self._all_providers.keys())}")
                if self._primary_provider:
                    logger.info(f"Primary provider: {self._primary_provider.provider_type}")
                if self._fallback_provider:
                    logger.info(f"Fallback provider: {self._fallback_provider.provider_type}")

        except Exception as e:
            logger.error(f"Error loading providers: {e}")
            # Keep existing configuration on error
            if not self._initialized:
                self._all_providers = {}
                self._primary_provider = None
                self._fallback_provider = None

    def refresh(self) -> None:
        """Force refresh of provider configuration."""
        self._initialized = False
        self._load_providers()

    def _should_refresh(self) -> bool:
        """Check if config should be refreshed based on TTL."""
        if not self._initialized:
            return True
        return (time.time() - self._last_refresh) > self._cache_ttl

    @property
    def last_refresh_timestamp(self) -> Optional[float]:
        """Get the last refresh timestamp."""
        return self._last_refresh if self._initialized else None

    @property
    def primary(self) -> Optional[ProviderInfo]:
        """Get the primary (default) provider."""
        if not self._initialized or self._should_refresh():
            self._load_providers()
        return self._primary_provider

    @property
    def fallback(self) -> Optional[ProviderInfo]:
        """Get the fallback provider."""
        if not self._initialized or self._should_refresh():
            self._load_providers()
        return self._fallback_provider

    def get_provider(self, provider_type: Optional[str] = None) -> Optional[ProviderInfo]:
        """
        Get a specific provider by type, or primary if not specified.

        Args:
            provider_type: Specific provider type to use (e.g., 'anthropic', 'openai')
                          If None, returns primary provider.

        Returns:
            ProviderInfo for the selected provider, or None if not available.
        """
        if not self._initialized or self._should_refresh():
            self._load_providers()

        if provider_type:
            return self._all_providers.get(provider_type)

        return self._primary_provider

    def get_all_providers(self) -> List[ProviderInfo]:
        """Get all available providers."""
        if not self._initialized or self._should_refresh():
            self._load_providers()
        return list(self._all_providers.values())

    def get_provider_for_tier(self, tier: str) -> Optional[ProviderInfo]:
        """
        Get provider configured for a specific tier.

        Args:
            tier: Tier name (e.g., 'fast', 'balanced', 'smart')

        Returns:
            ProviderInfo for the tier, or primary if not configured.
        """
        if not self._initialized or self._should_refresh():
            self._load_providers()

        # Check each provider's tier assignment
        for provider in self._all_providers.values():
            if provider.tier_assignment.get(tier):
                return provider

        # Fall back to primary
        return self._primary_provider

    def execute_with_fallback(
        self,
        func: Callable[[ProviderInfo], Any],
        preferred_provider: Optional[str] = None,
    ) -> Any:
        """
        Execute a function with automatic fallback on failure.

        Args:
            func: Function that takes ProviderInfo and returns a result.
            preferred_provider: Specific provider type to try first.

        Returns:
            Result from the function, or raises last exception.

        Raises:
            Exception: If all providers fail.
        """
        if not self._initialized or self._should_refresh():
            self._load_providers()

        # Determine try order
        providers_to_try = []

        if preferred_provider and preferred_provider in self._all_providers:
            providers_to_try.append(self._all_providers[preferred_provider])

        if self._primary_provider and self._primary_provider.provider_type != preferred_provider:
            providers_to_try.append(self._primary_provider)

        if self._fallback_provider and self._fallback_provider.provider_type != preferred_provider:
            if self._fallback_provider not in providers_to_try:
                providers_to_try.append(self._fallback_provider)

        # Try each provider
        last_error = None
        for provider in providers_to_try:
            try:
                logger.info(f"Trying provider: {provider.provider_type}")
                return func(provider)
            except Exception as e:
                logger.warning(f"Provider {provider.provider_type} failed: {e}")
                last_error = e
                continue

        if last_error:
            raise last_error

        raise RuntimeError("No providers available")


# Global router instance
_router: Optional[ProviderRouter] = None


def get_router() -> ProviderRouter:
    """Get the global provider router instance."""
    global _router
    if _router is None:
        _router = ProviderRouter(cache_ttl=DEFAULT_CACHE_TTL)
    return _router


def refresh_router() -> None:
    """Force refresh the provider router."""
    global _router
    if _router is not None:
        _router.refresh()
    else:
        _router = ProviderRouter(cache_ttl=DEFAULT_CACHE_TTL)
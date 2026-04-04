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

from src.database.models import ProviderConfig, db_session_scope

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


@dataclass
class RuntimeLLMConfig:
    """Resolved runtime configuration for an anthropic-compatible client."""
    provider_type: str
    api_key: str
    base_url: str
    model: str
    source: str
    display_name: str = ""


class ProviderRouter:
    """
    Routes AI requests to configured providers.

    Supports:
    - Primary provider selection
    - Fallback on failure
    - Custom base URLs per provider
    """

    # Default base URLs for each provider type
    # MiniMax & GLM use Anthropic-compatible API (Claude SDK format)
    # Anthropic, OpenAI, DeepSeek, etc. use standard OpenAI-compatible format
    DEFAULT_BASE_URLS = {
        "anthropic": "https://api.anthropic.com/v1",
        "openai": "https://api.openai.com/v1",
        "openrouter": "https://openrouter.ai/api/v1",
        "deepseek": "https://api.deepseek.com/v1",
        "glm": "https://api.z.ai/api/anthropic",
        "minimax": "https://api.minimax.io/anthropic",
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
            with db_session_scope() as db:
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

    @staticmethod
    def _first_model_id(model_list: Optional[List[Dict[str, str]]]) -> Optional[str]:
        """Return the first usable model id from a provider model list."""
        if not model_list:
            return None
        for entry in model_list:
            if not isinstance(entry, dict):
                continue
            model_id = entry.get("id") or entry.get("model_id") or entry.get("name")
            if model_id:
                return str(model_id)
        return None

    @staticmethod
    def _provider_env_prefix(provider_type: str) -> str:
        return str(provider_type or "").strip().upper().replace("-", "_")

    def _resolve_env_runtime_config(
        self,
        *,
        preferred_provider: Optional[str] = None,
        preferred_model: Optional[str] = None,
        tier: Optional[str] = None,
        default_model: Optional[str] = None,
    ) -> Optional[RuntimeLLMConfig]:
        """
        Resolve generic environment-driven runtime config.

        This keeps active node/session logic provider-neutral while preserving
        legacy env compatibility for existing deployments.
        """
        provider_type = (
            preferred_provider
            or os.getenv("QMX_LLM_PROVIDER")
            or os.getenv("MODEL_PROVIDER")
            or ""
        ).strip().lower()

        if not provider_type:
            if os.getenv("MINIMAX_API_KEY"):
                provider_type = "minimax"
            elif os.getenv("ANTHROPIC_API_KEY"):
                provider_type = "anthropic"
            elif os.getenv("OPENAI_API_KEY"):
                provider_type = "openai"

        provider_prefix = self._provider_env_prefix(provider_type) if provider_type else ""

        api_key = (
            os.getenv("QMX_LLM_API_KEY")
            or os.getenv("MODEL_API_KEY")
            or (os.getenv(f"{provider_prefix}_API_KEY") if provider_prefix else None)
            or os.getenv("MINIMAX_API_KEY")
            or os.getenv("ANTHROPIC_API_KEY")
        )
        if not api_key:
            return None

        base_url = (
            os.getenv("QMX_LLM_BASE_URL")
            or os.getenv("MODEL_BASE_URL")
            or (os.getenv(f"{provider_prefix}_BASE_URL") if provider_prefix else None)
            or self.DEFAULT_BASE_URLS.get(provider_type or "", "")
            or os.getenv("MINIMAX_BASE_URL")
            or os.getenv("ANTHROPIC_BASE_URL")
        )

        tier_model = None
        if tier:
            normalized_tier = str(tier).strip().lower()
            if normalized_tier == "opus":
                tier_model = os.getenv("QMX_LLM_MODEL_OPUS") or os.getenv("MODEL_OPUS")
            elif normalized_tier == "haiku":
                tier_model = os.getenv("QMX_LLM_MODEL_HAIKU") or os.getenv("MODEL_HAIKU")

        model = (
            preferred_model
            or tier_model
            or os.getenv("QMX_LLM_MODEL")
            or os.getenv("MODEL_ID")
            or (os.getenv(f"{provider_prefix}_MODEL") if provider_prefix else None)
            or os.getenv("MINIMAX_MODEL")
            or os.getenv("ANTHROPIC_MODEL")
            or default_model
            or ""
        )

        return RuntimeLLMConfig(
            provider_type=provider_type or "env",
            api_key=api_key,
            base_url=base_url,
            model=model,
            source="env",
            display_name=provider_type or "env",
        )

    def resolve_runtime_config(
        self,
        *,
        preferred_provider: Optional[str] = None,
        preferred_model: Optional[str] = None,
        tier: Optional[str] = None,
        default_model: Optional[str] = None,
    ) -> Optional[RuntimeLLMConfig]:
        """
        Resolve the configured runtime LLM without hardcoding a vendor path.

        Order:
        1. Explicit configured provider
        2. Primary configured provider
        3. Generic env contract (`QMX_LLM_*` / `MODEL_*`)
        4. Legacy provider-specific env compatibility
        """
        if not self._initialized or self._should_refresh():
            self._load_providers()

        provider = self.get_provider(preferred_provider) if preferred_provider else self.primary
        if provider is None and preferred_provider:
            provider = self.primary

        if provider:
            model = (
                preferred_model
                or provider.tier_assignment.get(tier or "")
                or self._first_model_id(provider.model_list)
                or default_model
                or ""
            )
            base_url = provider.base_url or self.DEFAULT_BASE_URLS.get(provider.provider_type, "")
            return RuntimeLLMConfig(
                provider_type=provider.provider_type,
                api_key=provider.api_key,
                base_url=base_url,
                model=model,
                source="provider_config",
                display_name=provider.display_name,
            )

        return self._resolve_env_runtime_config(
            preferred_provider=preferred_provider,
            preferred_model=preferred_model,
            tier=tier,
            default_model=default_model,
        )

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

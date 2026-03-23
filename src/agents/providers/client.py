"""
Provider Client

Wrapper around Claude Agent SDK and other LLM clients with provider routing.

Story 2.3: Claude Agent SDK Provider Routing
"""

import logging
from typing import Optional, Dict, Any, List
import asyncio

from .router import ProviderRouter, ProviderInfo, get_router

logger = logging.getLogger(__name__)


class ProviderClient:
    """
    Client for making LLM requests with provider routing.

    Supports:
    - Automatic provider selection (primary/fallback)
    - Claude Agent SDK integration with custom base_url
    - Retry on failure with fallback provider
    """

    def __init__(
        self,
        router: Optional[ProviderRouter] = None,
        timeout: float = 60.0,
        max_retries: int = 2,
    ):
        """
        Initialize provider client.

        Args:
            router: Provider router instance (uses global if not provided)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        self._router = router or get_router()
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = None

    async def complete(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        provider_type: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Make a completion request with automatic provider routing.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model ID (e.g., 'claude-opus-4-6-20250514')
            provider_type: Specific provider to use (overrides default selection)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature

        Returns:
            Response dict with 'content', 'model', 'provider_type', etc.
        """
        # Get provider info
        provider = self._router.get_provider(provider_type)

        if not provider:
            if provider_type:
                raise ValueError(f"Provider '{provider_type}' not configured or no API key")
            raise RuntimeError("No providers available. Please configure a provider first.")

        # Make the request with retries
        last_error = None
        providers_to_try = self._get_provider_try_order(provider_type)

        for try_provider in providers_to_try:
            try:
                return await self._make_request(
                    provider=try_provider,
                    messages=messages,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs,
                )
            except Exception as e:
                logger.warning(f"Request to {try_provider.provider_type} failed: {e}")
                last_error = e
                continue

        raise RuntimeError(f"All providers failed. Last error: {last_error}")

    def _get_provider_try_order(self, preferred: Optional[str]) -> List[ProviderInfo]:
        """Get ordered list of providers to try."""
        order = []

        if preferred:
            provider = self._router.get_provider(preferred)
            if provider:
                order.append(provider)

        if self._router.primary and self._router.primary not in order:
            order.append(self._router.primary)

        if self._router.fallback and self._router.fallback not in order:
            order.append(self._router.fallback)

        return order

    async def _make_request(
        self,
        provider: ProviderInfo,
        messages: List[Dict[str, str]],
        model: Optional[str],
        max_tokens: int,
        temperature: float,
        **kwargs,
    ) -> Dict[str, Any]:
        """Make a request to a specific provider."""

        if provider.provider_type == "anthropic":
            return await self._anthropic_request(
                provider=provider,
                messages=messages,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )
        elif provider.provider_type == "openai" or provider.provider_type == "openrouter":
            return await self._openai_request(
                provider=provider,
                messages=messages,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )
        elif provider.provider_type == "deepseek":
            return await self._deepseek_request(
                provider=provider,
                messages=messages,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )
        elif provider.provider_type == "glm":
            return await self._glm_request(
                provider=provider,
                messages=messages,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )
        else:
            raise ValueError(f"Unsupported provider type: {provider.provider_type}")

    async def _anthropic_request(
        self,
        provider: ProviderInfo,
        messages: List[Dict[str, str]],
        model: Optional[str],
        max_tokens: int,
        temperature: float,
        **kwargs,
    ) -> Dict[str, Any]:
        """Make request to Anthropic API."""
        import httpx

        headers = {
            "x-api-key": provider.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        # Convert messages to Anthropic format
        anthropic_messages = []
        system_content = None

        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                anthropic_messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                })

        body = {
            "model": model or "claude-sonnet-4-6-20250514",
            "messages": anthropic_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if system_content:
            body["system"] = system_content

        url = f"{provider.base_url}/messages"

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, headers=headers, json=body)

        if response.status_code != 200:
            raise RuntimeError(f"Anthropic API error: {response.status_code} - {response.text[:200]}")

        data = response.json()

        return {
            "content": data["content"][0]["text"],
            "model": data["model"],
            "provider_type": "anthropic",
            "input_tokens": data.get("usage", {}).get("input_tokens", 0),
            "output_tokens": data.get("usage", {}).get("output_tokens", 0),
        }

    async def _openai_request(
        self,
        provider: ProviderInfo,
        messages: List[Dict[str, str]],
        model: Optional[str],
        max_tokens: int,
        temperature: float,
        **kwargs,
    ) -> Dict[str, Any]:
        """Make request to OpenAI-compatible API."""
        import httpx

        headers = {
            "Authorization": f"Bearer {provider.api_key}",
            "Content-Type": "application/json",
        }

        # Add OpenRouter specific headers
        extra_headers = kwargs.get("extra_headers", {})
        headers.update(extra_headers)

        body = {
            "model": model or "gpt-4o",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        url = f"{provider.base_url}/chat/completions"

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, headers=headers, json=body)

        if response.status_code != 200:
            raise RuntimeError(f"OpenAI API error: {response.status_code} - {response.text[:200]}")

        data = response.json()

        return {
            "content": data["choices"][0]["message"]["content"],
            "model": data["model"],
            "provider_type": provider.provider_type,
            "input_tokens": data.get("usage", {}).get("prompt_tokens", 0),
            "output_tokens": data.get("usage", {}).get("completion_tokens", 0),
        }

    async def _deepseek_request(
        self,
        provider: ProviderInfo,
        messages: List[Dict[str, str]],
        model: Optional[str],
        max_tokens: int,
        temperature: float,
        **kwargs,
    ) -> Dict[str, Any]:
        """Make request to DeepSeek API."""
        # DeepSeek is OpenAI-compatible
        return await self._openai_request(
            provider=provider,
            messages=messages,
            model=model or "deepseek-chat",
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

    async def _glm_request(
        self,
        provider: ProviderInfo,
        messages: List[Dict[str, str]],
        model: Optional[str],
        max_tokens: int,
        temperature: float,
        **kwargs,
    ) -> Dict[str, Any]:
        """Make request to GLM (Zhipu) API."""
        import httpx

        # GLM uses a slightly different format
        glm_messages = []
        for msg in messages:
            glm_messages.append({
                "role": msg["role"],
                "content": msg["content"],
            })

        body = {
            "model": model or "glm-4-flash",
            "messages": glm_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        headers = {
            "Authorization": f"Bearer {provider.api_key}",
            "Content-Type": "application/json",
        }

        url = f"{provider.base_url}/chat/completions"

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, headers=headers, json=body)

        if response.status_code != 200:
            raise RuntimeError(f"GLM API error: {response.status_code} - {response.text[:200]}")

        data = response.json()

        return {
            "content": data["choices"][0]["message"]["content"],
            "model": data["model"],
            "provider_type": "glm",
            "input_tokens": data.get("usage", {}).get("prompt_tokens", 0),
            "output_tokens": data.get("usage", {}).get("completion_tokens", 0),
        }

    async def list_models(self, provider_type: Optional[str] = None) -> List[Dict[str, str]]:
        """
        List available models for a provider.

        Args:
            provider_type: Specific provider type (uses primary if not specified)

        Returns:
            List of model dicts with 'id' and 'name'
        """
        provider = self._router.get_provider(provider_type)

        if not provider:
            return []

        return provider.model_list


def get_client(timeout: float = 60.0, max_retries: int = 2) -> ProviderClient:
    """Get a provider client instance."""
    return ProviderClient(
        router=get_router(),
        timeout=timeout,
        max_retries=max_retries,
    )
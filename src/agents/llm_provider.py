"""
LLM Provider Configuration for QuantMind Agents

This module provides centralized LLM provider configuration following AGENTS.md specs.
Supports OpenRouter (primary), Z.ai (Zhipu), and direct Anthropic access.

Provider Priority:
1. OpenRouter (most flexible, best routing)
2. Z.ai (cost-effective, fast)
3. Anthropic direct (fallback for Claude models)

DEPRECATED: LangChain imports removed. Use ClaudeOrchestrator instead.
For backward compatibility, this module provides simple stubs.
"""

import os
import logging
from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class ProviderType(str, Enum):
    OPENROUTER = "openrouter"
    ZHIPU = "zhipu"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"  # Fallback for local development


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    model_id: str
    provider: ProviderType
    fallback_model: Optional[str] = None
    fallback_provider: Optional[ProviderType] = None
    temperature: float = 0.0
    max_tokens: int = 4096


# Agent-specific model configurations per AGENTS.md
AGENT_MODELS: Dict[str, ModelConfig] = {
    "copilot": ModelConfig(
        model_id="anthropic/claude-sonnet-4",
        provider=ProviderType.OPENROUTER,
        fallback_model="glm-4-plus",
        fallback_provider=ProviderType.ZHIPU,
        temperature=0.0,
    ),
    "quantcode": ModelConfig(
        model_id="deepseek/deepseek-coder",
        provider=ProviderType.OPENROUTER,
        fallback_model="glm-4-flash",
        fallback_provider=ProviderType.ZHIPU,
        temperature=0.0,
    ),
    "analyst": ModelConfig(
        model_id="anthropic/claude-sonnet-4",
        provider=ProviderType.OPENROUTER,
        fallback_model="glm-4-plus",
        fallback_provider=ProviderType.ZHIPU,
        temperature=0.0,
    ),
}

# Provider base URLs
PROVIDER_BASE_URLS: Dict[ProviderType, str] = {
    ProviderType.OPENROUTER: "https://openrouter.ai/api/v1",
    ProviderType.ZHIPU: "https://open.bigmodel.cn/api/paas/v4",
    ProviderType.ANTHROPIC: "https://api.anthropic.com/v1",
    ProviderType.OPENAI: "https://api.openai.com/v1",
}

# Environment variable names for API keys
API_KEY_ENV_VARS: Dict[ProviderType, str] = {
    ProviderType.OPENROUTER: "OPENROUTER_API_KEY",
    ProviderType.ZHIPU: "ZHIPU_API_KEY",
    ProviderType.ANTHROPIC: "ANTHROPIC_API_KEY",
    ProviderType.OPENAI: "OPENAI_API_KEY",
}


def get_api_key(provider: ProviderType) -> Optional[str]:
    """Get API key for a provider from environment."""
    env_var = API_KEY_ENV_VARS.get(provider)
    if env_var:
        return os.getenv(env_var)
    return None


def has_api_key(provider: ProviderType) -> bool:
    """Check if API key is available for a provider."""
    return get_api_key(provider) is not None


def get_available_providers() -> List[ProviderType]:
    """Get list of providers with available API keys."""
    return [p for p in ProviderType if has_api_key(p)]


class SimpleLLM:
    """
    Simple LLM wrapper that provides LangChain-like interface.

    This is a stub for backward compatibility. Use ClaudeOrchestrator
    for actual LLM operations.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
    ):
        """Initialize the LLM wrapper."""
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url
        self.api_key = api_key
        self.default_headers = default_headers or {}
        self._tools = []

    def bind_tools(self, tools: List[Any]) -> "SimpleLLM":
        """Bind tools to the LLM."""
        self._tools = tools
        return self

    def invoke(self, messages: List[Any]) -> Any:
        """
        Invoke the LLM with messages.

        Note: This is a stub. Use ClaudeOrchestrator for actual LLM calls.
        """
        logger.warning(
            "SimpleLLM.invoke() is a stub. Use ClaudeOrchestrator for actual LLM operations."
        )

        # Return a mock response
        class MockResponse:
            content = "LLM stub response - use ClaudeOrchestrator for actual operations"
            tool_calls = []

        return MockResponse()

    async def ainvoke(self, messages: List[Any]) -> Any:
        """Async invoke the LLM."""
        return self.invoke(messages)


def get_llm_for_agent(
    agent_type: str,
    tools: List[Any] = None,
    use_fallback: bool = False
) -> Any:
    """
    Get configured LLM for a specific agent type.

    DEPRECATED: This returns a SimpleLLM stub. Use ClaudeOrchestrator instead.

    Args:
        agent_type: Agent type (copilot, analyst, quantcode)
        tools: List of tools to bind to the LLM
        use_fallback: Force use of fallback model

    Returns:
        SimpleLLM instance (stub for backward compatibility)
    """
    config = AGENT_MODELS.get(agent_type)
    if not config:
        logger.warning(f"Unknown agent type: {agent_type}, using default config")
        config = ModelConfig(
            model_id="gpt-4",
            provider=ProviderType.OPENAI,
            temperature=0.0,
        )

    # Determine which model/provider to use
    if use_fallback and config.fallback_model:
        model_id = config.fallback_model
        provider = config.fallback_provider
    else:
        model_id = config.model_id
        provider = config.provider

    # Check if primary provider is available, fall back if not
    if not has_api_key(provider):
        if config.fallback_provider and has_api_key(config.fallback_provider):
            logger.info(
                f"Primary provider {provider.value} not available for {agent_type}, "
                f"using fallback {config.fallback_provider.value}"
            )
            model_id = config.fallback_model
            provider = config.fallback_provider
        elif has_api_key(ProviderType.OPENAI):
            # Last resort fallback to OpenAI
            logger.warning(
                f"No configured providers available for {agent_type}, "
                f"falling back to OpenAI gpt-4"
            )
            model_id = "gpt-4"
            provider = ProviderType.OPENAI
        else:
            logger.warning(
                f"No API keys available for agent {agent_type}. "
                f"Using stub LLM. Set one of: OPENROUTER_API_KEY, ZHIPU_API_KEY, "
                f"ANTHROPIC_API_KEY, or OPENAI_API_KEY"
            )

    # Get base URL and API key
    base_url = PROVIDER_BASE_URLS.get(provider)
    api_key = get_api_key(provider)

    # For OpenRouter, set extra headers
    extra_headers = {}
    if provider == ProviderType.OPENROUTER:
        extra_headers = {
            "HTTP-Referer": "https://quantmind.app",
            "X-Title": "QuantMind Trading System",
        }

    # Create SimpleLLM stub
    llm = SimpleLLM(
        model=model_id,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        base_url=base_url,
        api_key=api_key,
        default_headers=extra_headers,
    )

    logger.info(f"Created SimpleLLM stub for {agent_type}: {model_id} via {provider.value}")

    # Bind tools if provided
    if tools:
        llm = llm.bind_tools(tools)
        logger.debug(f"Bound {len(tools)} tools to LLM stub")

    return llm


def get_llm_with_fallback(
    agent_type: str,
    tools: List[Any] = None,
) -> Any:
    """
    Get LLM with automatic fallback on failure.

    This tries the primary model first, then falls back to the
    configured fallback model if the primary fails.

    Args:
        agent_type: Agent type (copilot, analyst, quantcode)
        tools: List of tools to bind to the LLM

    Returns:
        Configured LLM instance
    """
    try:
        return get_llm_for_agent(agent_type, tools, use_fallback=False)
    except Exception as e:
        logger.warning(f"Primary LLM failed for {agent_type}: {e}, trying fallback")
        return get_llm_for_agent(agent_type, tools, use_fallback=True)


# Convenience functions for each agent
def get_copilot_llm(tools: List[Any] = None) -> Any:
    """Get LLM configured for Copilot agent."""
    return get_llm_with_fallback("copilot", tools)


def get_analyst_llm(tools: List[Any] = None) -> Any:
    """Get LLM configured for Analyst agent."""
    return get_llm_with_fallback("analyst", tools)


def get_quantcode_llm(tools: List[Any] = None) -> Any:
    """Get LLM configured for QuantCode agent."""
    return get_llm_with_fallback("quantcode", tools)

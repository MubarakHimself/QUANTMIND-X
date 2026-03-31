# src/agents/departments/subagents/llm_utils.py
"""
Shared LLM client initialization for all sub-agents.

Resolves the configured provider (base_url + api_key + model) via the
ProviderRouter first, then falls back to ANTHROPIC_* env vars.

All sub-agents MUST use get_subagent_client() instead of bare Anthropic().
"""
import logging
import os
from typing import Tuple

logger = logging.getLogger(__name__)

# Default model for all sub-agents — Sonnet for quality
DEFAULT_SUBAGENT_MODEL = "claude-sonnet-4-6"


def get_subagent_client() -> Tuple["Anthropic", str]:
    """
    Create an Anthropic client using the configured provider.

    Resolution order:
    1. ProviderRouter (database-configured provider with base_url)
    2. Environment variables (ANTHROPIC_BASE_URL + ANTHROPIC_API_KEY)
    3. Bare Anthropic() with default API key

    Returns:
        Tuple of (Anthropic client, model_id string)
    """
    from anthropic import Anthropic

    # --- 1. Try ProviderRouter ---
    try:
        from src.agents.providers.router import get_router
        provider = get_router().primary
        if provider and provider.api_key:
            client = Anthropic(
                api_key=provider.api_key,
                base_url=provider.base_url,
            )
            # Resolve model from provider config
            ml = provider.model_list
            if ml:
                model = (
                    ml[0].get("id")
                    or ml[0].get("model_id")
                    or DEFAULT_SUBAGENT_MODEL
                )
            else:
                model = DEFAULT_SUBAGENT_MODEL
            logger.info(f"SubAgent LLM: provider={provider.name}, model={model}")
            return client, model
    except Exception as e:
        logger.debug(f"ProviderRouter unavailable: {e}")

    # --- 2. Try env vars ---
    base_url = os.getenv("ANTHROPIC_BASE_URL")
    api_key = os.getenv("ANTHROPIC_API_KEY")
    model = os.getenv("ANTHROPIC_MODEL_SONNET", DEFAULT_SUBAGENT_MODEL)

    if api_key:
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        client = Anthropic(**kwargs)
        logger.info(f"SubAgent LLM: env-based, model={model}")
        return client, model

    # --- 3. Bare fallback ---
    logger.warning("SubAgent LLM: no provider configured, using bare Anthropic()")
    return Anthropic(), DEFAULT_SUBAGENT_MODEL

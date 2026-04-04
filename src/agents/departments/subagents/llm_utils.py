# src/agents/departments/subagents/llm_utils.py
"""
Shared LLM client initialization for all sub-agents.

Resolves runtime configuration through the configured provider router first,
then through the generic env contract (`QMX_LLM_*` / `MODEL_*`), while
preserving legacy env compatibility for existing deployments.

All sub-agents MUST use get_subagent_client() instead of bare Anthropic().
"""
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

DEFAULT_SUBAGENT_MODEL = ""


def get_subagent_client() -> Tuple["Anthropic", str]:
    """
    Create an Anthropic client using the configured provider.

    Resolution order:
    1. ProviderRouter (database-configured provider with base_url)
    2. Generic environment variables (`QMX_LLM_*` / `MODEL_*`)
    3. Legacy environment compatibility fallback
    4. Bare Anthropic() with ambient auth

    Returns:
        Tuple of (Anthropic client, model_id string)
    """
    from anthropic import Anthropic

    try:
        from src.agents.providers.router import get_router
        runtime_config = get_router().resolve_runtime_config(default_model=DEFAULT_SUBAGENT_MODEL)
        if runtime_config and runtime_config.api_key:
            client = Anthropic(
                api_key=runtime_config.api_key,
                base_url=runtime_config.base_url,
            )
            model = runtime_config.model or DEFAULT_SUBAGENT_MODEL
            logger.info(
                "SubAgent LLM resolved from %s: provider=%s model=%s",
                runtime_config.source,
                runtime_config.provider_type,
                model or "<auto>",
            )
            return client, model
    except Exception as e:
        logger.debug(f"ProviderRouter unavailable: {e}")

    # --- 3. Bare fallback ---
    logger.warning("SubAgent LLM: no provider configured, using bare Anthropic()")
    return Anthropic(), DEFAULT_SUBAGENT_MODEL

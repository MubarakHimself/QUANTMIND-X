"""
SDK Agent Configuration

Configuration module for Claude Agent SDK-based agents.
Loads MCP configs and defines agent settings.
Supports multiple LLM providers (Anthropic, Z.AI).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import logging
import os

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

# Agent definitions
AGENT_TYPES = ["analyst", "quantcode", "copilot", "pinescript", "router", "executor"]

# Model mapping for different providers
# Z.AI models (Anthropic API-compatible): https://docs.z.ai/devpack/tool/claude
MODEL_MAPPING = {
    "anthropic": {
        "opus": "claude-opus-4-20250514",
        "sonnet": "claude-sonnet-4-20250514",
        "haiku": "claude-haiku-3-5-20241022",
    },
    "zai": {
        "opus": "glm-4-plus",
        "sonnet": "glm-4",
        "haiku": "glm-4-flash",
    }
}

DEFAULT_MODEL = "claude-sonnet-4-20250514"
DEFAULT_TIER = "sonnet"  # opus, sonnet, haiku


@dataclass
class SDKAgentConfig:
    """Configuration for Claude Agent SDK-based agent."""
    agent_id: str
    model: str = DEFAULT_MODEL
    system_prompt: str = ""
    mcp_config_path: Optional[Path] = None
    system_prompt_path: Optional[Path] = None
    workspace: Optional[Path] = None
    timeout_seconds: int = 300
    permission_mode: str = "auto"  # auto, ask, accept-edits

    def __post_init__(self):
        """Set default paths based on agent_id."""
        # Set default paths based on agent_id
        base_path = Path(__file__).parent.parent.parent / "workspaces" / self.agent_id
        config_base = Path(__file__).parent.parent.parent / "config" / "mcp"

        if not self.mcp_config_path:
            self.mcp_config_path = config_base / f"{self.agent_id}-mcp.json"
        if not self.system_prompt_path:
            self.system_prompt_path = base_path / "context" / "CLAUDE.md"
        if not self.workspace:
            self.workspace = base_path


def get_sdk_agent_config(agent_id: str) -> SDKAgentConfig:
    """
    Get SDK configuration for an agent type.

    Args:
        agent_id: The agent identifier (analyst, quantcode, copilot, etc.)

    Returns:
        SDKAgentConfig instance with default settings

    Raises:
        ValueError: If agent_id is not recognized
    """
    if agent_id not in AGENT_TYPES:
        raise ValueError(f"Unknown agent: {agent_id}")
    return SDKAgentConfig(agent_id=agent_id)


def get_provider_config() -> Dict[str, str]:
    """
    Get the current LLM provider configuration.

    Returns:
        Dictionary with provider, api_key, and base_url
    """
    provider = os.getenv("LLM_PROVIDER", "zai").lower()

    if provider == "zai":
        # Support both ZAI_API_KEY and ZHIPU_API_KEY for GLM
        api_key = os.getenv("ZAI_API_KEY") or os.getenv("ZHIPU_API_KEY", "")
        return {
            "provider": "zai",
            "api_key": api_key,
            "base_url": os.getenv("ZAI_BASE_URL", "https://api.z.ai/api/anthropic"),
        }
    elif provider == "zhipu":
        return {
            "provider": "zai",  # Zhipu uses Z.AI endpoint
            "api_key": os.getenv("ZHIPU_API_KEY", ""),
            "base_url": os.getenv("ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
        }
    else:
        return {
            "provider": "anthropic",
            "api_key": os.getenv("ANTHROPIC_API_KEY", ""),
            "base_url": None,  # Use default Anthropic URL
        }


def get_thinking_config() -> Dict[str, Any]:
    """
    Get GLM thinking mode configuration from environment.

    Returns:
        Dictionary with thinking configuration
    """
    thinking_type = os.getenv("GLM_THINKING_TYPE", "enabled")
    thinking_mode = os.getenv("GLM_THINKING_MODE", "interleaved")

    return {
        "type": thinking_type,
        "mode": thinking_mode,
        "clear_thinking": os.getenv("GLM_CLEAR_THINKING", "false").lower() == "true",
    }


def get_model_for_tier(tier: str = "sonnet") -> str:
    """
    Get the model name for a given tier based on current provider.

    Args:
        tier: Model tier (opus, sonnet, haiku)

    Returns:
        Model name string for the current provider
    """
    provider_config = get_provider_config()
    provider = provider_config["provider"]

    if provider not in MODEL_MAPPING:
        logger.warning(f"Unknown provider {provider}, falling back to anthropic")
        provider = "anthropic"

    return MODEL_MAPPING[provider].get(tier, MODEL_MAPPING[provider]["sonnet"])


def load_system_prompt(config: SDKAgentConfig) -> str:
    """
    Load system prompt from file.

    Args:
        config: SDKAgentConfig instance with system_prompt_path

    Returns:
        System prompt string, or empty string if file not found
    """
    if config.system_prompt_path and config.system_prompt_path.exists():
        try:
            with open(config.system_prompt_path, "r", encoding="utf-8") as f:
                return f.read()
        except IOError as e:
            logger.warning(f"Failed to load system prompt from {config.system_prompt_path}: {e}")
    return ""


def load_mcp_config(config: SDKAgentConfig) -> Dict[str, Any]:
    """
    Load MCP configuration from JSON file.

    Args:
        config: SDKAgentConfig instance with mcp_config_path

    Returns:
        MCP configuration dictionary, or empty config if file not found
    """
    if config.mcp_config_path and config.mcp_config_path.exists():
        try:
            with open(config.mcp_config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to load MCP config from {config.mcp_config_path}: {e}")
    return {"mcpServers": {}}

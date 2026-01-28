"""
Configuration management for the analyst agent.

This module handles loading configuration from JSON files and environment variables,
with support for LLM provider configuration and knowledge base paths.
"""

import json
import os
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, field


@dataclass
class LLMConfig:
    """Configuration for LLM provider."""

    provider: str = "openrouter"
    model: str = "qwen/qwen3-vl-30b-a3b-thinking"
    api_key: Optional[str] = None
    base_url: str = "https://openrouter.ai/api/v1"
    temperature: float = 0.7
    max_tokens: int = 4096


@dataclass
class KnowledgeBaseConfig:
    """Configuration for knowledge base."""

    chroma_path: str = "./data/chromadb"
    collection_name: str = "quantmindx_kb"
    embedding_model: str = "all-MiniLM-L6-v2"


@dataclass
class PathsConfig:
    """Configuration for file paths."""

    nprd_output_dir: str = "./outputs/nprd"
    trd_output_dir: str = "./outputs/trd"
    log_dir: str = "./logs"


@dataclass
class Config:
    """Main configuration class for the analyst agent."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    kb: KnowledgeBaseConfig = field(default_factory=KnowledgeBaseConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        """Create Config from dictionary."""
        config = cls()

        # Load LLM config
        if "llm" in data:
            llm_data = data["llm"]
            config.llm = LLMConfig(
                provider=llm_data.get("provider", config.llm.provider),
                model=llm_data.get("model", config.llm.model),
                api_key=llm_data.get("api_key"),
                base_url=llm_data.get("base_url", config.llm.base_url),
                temperature=llm_data.get("temperature", config.llm.temperature),
                max_tokens=llm_data.get("max_tokens", config.llm.max_tokens),
            )

        # Load KB config
        if "kb" in data:
            kb_data = data["kb"]
            config.kb = KnowledgeBaseConfig(
                chroma_path=kb_data.get("chroma_path", config.kb.chroma_path),
                collection_name=kb_data.get("collection_name", config.kb.collection_name),
                embedding_model=kb_data.get("embedding_model", config.kb.embedding_model),
            )

        # Load paths config
        if "paths" in data:
            paths_data = data["paths"]
            config.paths = PathsConfig(
                nprd_output_dir=paths_data.get("nprd_output_dir", config.paths.nprd_output_dir),
                trd_output_dir=paths_data.get("trd_output_dir", config.paths.trd_output_dir),
                log_dir=paths_data.get("log_dir", config.paths.log_dir),
            )

        return config

    def to_dict(self) -> dict[str, Any]:
        """Convert Config to dictionary."""
        return {
            "llm": {
                "provider": self.llm.provider,
                "model": self.llm.model,
                "api_key": self.llm.api_key,
                "base_url": self.llm.base_url,
                "temperature": self.llm.temperature,
                "max_tokens": self.llm.max_tokens,
            },
            "kb": {
                "chroma_path": self.kb.chroma_path,
                "collection_name": self.kb.collection_name,
                "embedding_model": self.kb.embedding_model,
            },
            "paths": {
                "nprd_output_dir": self.paths.nprd_output_dir,
                "trd_output_dir": self.paths.trd_output_dir,
                "log_dir": self.paths.log_dir,
            },
        }


def load_config_file(config_path: Optional[Path] = None) -> Optional[dict[str, Any]]:
    """
    Load configuration from JSON file.

    Args:
        config_path: Path to config file. If None, searches default locations.

    Returns:
        Configuration dictionary, or None if file not found.
    """
    if config_path is None:
        # Search default locations
        default_locations = [
            Path.cwd() / ".analyst_config.json",
            Path.home() / ".analyst_config.json",
            Path(__file__).parent.parent.parent.parent / ".analyst_config.json",
        ]
        for location in default_locations:
            if location.exists():
                config_path = location
                break

    if config_path is None or not config_path.exists():
        return None

    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file {config_path}: {e}")
    except Exception as e:
        raise RuntimeError(f"Error loading config file {config_path}: {e}")


def get_config(config_path: Optional[Path] = None) -> Config:
    """
    Get configuration, loading from file and environment variables.

    Configuration priority (highest to lowest):
    1. Environment variables
    2. Config file
    3. Defaults

    Args:
        config_path: Optional path to config file.

    Returns:
        Config instance with all settings loaded.
    """
    # Start with defaults
    config = Config()

    # Load from file if available
    file_config = load_config_file(config_path)
    if file_config:
        config = Config.from_dict(file_config)

    # Override with environment variables
    if os.getenv("OPENROUTER_API_KEY"):
        config.llm.api_key = os.getenv("OPENROUTER_API_KEY")

    if os.getenv("CHROMA_PATH"):
        config.kb.chroma_path = os.getenv("CHROMA_PATH")

    if os.getenv("LLM_MODEL"):
        config.llm.model = os.getenv("LLM_MODEL")

    if os.getenv("LLM_PROVIDER"):
        config.llm.provider = os.getenv("LLM_PROVIDER")

    if os.getenv("LLM_BASE_URL"):
        config.llm.base_url = os.getenv("LLM_BASE_URL")

    if os.getenv("NPRD_OUTPUT_DIR"):
        config.paths.nprd_output_dir = os.getenv("NPRD_OUTPUT_DIR")

    if os.getenv("TRD_OUTPUT_DIR"):
        config.paths.trd_output_dir = os.getenv("TRD_OUTPUT_DIR")

    return config

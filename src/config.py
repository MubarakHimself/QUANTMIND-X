"""
QuantMind Configuration

Centralized configuration management using Pydantic Settings.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False

    # TradingView Settings
    tradingview_webhook_secret: str = ""
    tradingview_api_key: str = ""

    # Redis Settings
    redis_url: str = "redis://localhost:6379"

    # Database Settings
    database_url: str = "sqlite:///data/quantmind.db"

    # MT5 Settings
    mt5_login: Optional[int] = None
    mt5_password: str = ""
    mt5_server: str = ""

    # Security
    secret_key: Optional[str] = None

    @property
    def effective_secret_key(self) -> str:
        """Get the effective secret key, raising error if not set in production."""
        if self.secret_key is None:
            raise ValueError(
                "SECRET_KEY environment variable is not set. "
                "Please set SECRET_KEY for production deployments."
            )
        # Warn if using default/weak key in production
        if self.secret_key in ("your-secret-key-change-in-production", "change-me", "secret"):
            import warnings
            warnings.warn(
                "Using default/weak SECRET_KEY. This is insecure for production. "
                "Please set a strong, unique SECRET_KEY."
            )
        return self.secret_key

    # External APIs
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None

    # ZMQ Settings
    zmq_endpoint: str = "tcp://localhost:5555"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields from .env


# Singleton instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get application settings (singleton)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def get_database_url() -> str:
    """Get database URL from environment or default."""
    return get_settings().database_url


def get_redis_url() -> str:
    """Get Redis URL from environment or default."""
    return get_settings().redis_url


def get_zmq_endpoint() -> str:
    """Get ZMQ endpoint from environment or default."""
    return get_settings().zmq_endpoint


def get_secret_key() -> str:
    """Get the application secret key.

    Raises:
        ValueError: If SECRET_KEY environment variable is not set.
    """
    return get_settings().effective_secret_key


# Module-level constant for direct import (backward compatibility)
ZMQ_ENDPOINT = os.environ.get('ZMQ_ENDPOINT', 'tcp://localhost:5555')

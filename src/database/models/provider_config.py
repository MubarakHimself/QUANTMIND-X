"""
Provider configuration models.

Contains models for storing API keys and base URLs for model providers
(Anthropic, GLM, MiniMax, DeepSeek, OpenAI, OpenRouter).
"""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, DateTime, Boolean
from ..models.base import Base


class ProviderConfig(Base):
    """
    Provider configuration for storing API keys and base URLs.

    This table stores API credentials and endpoint configurations for
    various LLM providers used by agents.

    Attributes:
        id: Primary key (UUID)
        name: Provider name (e.g., 'anthropic', 'glm', 'minimax', 'deepseek', 'openai', 'openrouter')
        api_key_encrypted: Encrypted API key for the provider
        base_url: Custom base URL for API endpoint (optional)
        enabled: Whether the provider is enabled
        created_at: Creation timestamp
        updated_at: Last update timestamp
    """
    __tablename__ = 'provider_configs'

    id = Column(String(36), primary_key=True)
    name = Column(String(50), nullable=False, unique=True, index=True)
    api_key_encrypted = Column(String(500), nullable=True)
    base_url = Column(String(500), nullable=True)
    enabled = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)

    def __repr__(self):
        return f"<ProviderConfig(id={self.id}, name={self.name}, enabled={self.enabled})>"

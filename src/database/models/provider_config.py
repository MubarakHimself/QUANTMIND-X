"""
Provider configuration models.

Contains models for storing API keys and base URLs for model providers
(Anthropic, GLM, MiniMax, DeepSeek, OpenAI, OpenRouter).

Supports encrypted storage of API keys using Fernet with machine-local keys.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Column, String, DateTime, Boolean, Text
from ..models.base import Base

logger = logging.getLogger(__name__)


class ProviderConfig(Base):
    """
    Provider configuration for storing API keys and base URLs.

    This table stores API credentials and endpoint configurations for
    various LLM providers used by agents. API keys are encrypted at rest.

    Attributes:
        id: Primary key (UUID)
        provider_type: Provider type identifier (e.g., 'anthropic', 'glm', 'minimax')
        display_name: Human-readable display name
        api_key_encrypted: Encrypted API key (Fernet encrypted)
        base_url: Custom base URL for API endpoint
        model_list: JSON list of available models for this provider
        tier_assignment: JSON dict mapping tier names to model names
            e.g., {"floor_manager": "claude-opus-4", "dept_heads": "claude-sonnet-4", "sub_agents": "claude-haiku-4"}
        is_active: Whether this provider is active for use
        created_at_utc: Creation timestamp in UTC
        updated_at: Last update timestamp
    """
    __tablename__ = 'provider_configs'

    id = Column(String(36), primary_key=True)
    provider_type = Column(String(50), nullable=False, unique=True, index=True)
    display_name = Column(String(100), nullable=False)
    api_key_encrypted = Column(Text, nullable=True)  # Encrypted API key
    base_url = Column(String(500), nullable=True)
    model_list = Column(Text, nullable=True)  # JSON list
    tier_assignment = Column(Text, nullable=True)  # JSON dict
    is_active = Column(Boolean, default=True, nullable=False)
    created_at_utc = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)

    # Legacy column aliases for backward compatibility
    @property
    def name(self) -> str:
        """Legacy alias for provider_type."""
        return self.provider_type

    @name.setter
    def name(self, value: str) -> None:
        """Legacy alias for provider_type."""
        self.provider_type = value

    @property
    def enabled(self) -> bool:
        """Legacy alias for is_active."""
        return self.is_active

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Legacy alias for is_active."""
        self.is_active = value

    @property
    def api_key(self) -> Optional[str]:
        """
        Get decrypted API key.

        Note: Returns None for security - use get_api_key() for decryption.
        """
        return None

    @api_key.setter
    def api_key(self, value: Optional[str]) -> None:
        """Set API key (will be encrypted on save)."""
        if value:
            from src.database.encryption import encrypt_api_key
            self.api_key_encrypted = encrypt_api_key(value)

    def get_api_key(self) -> Optional[str]:
        """Get decrypted API key."""
        if not self.api_key_encrypted:
            return None
        from src.database.encryption import decrypt_api_key
        return decrypt_api_key(self.api_key_encrypted)

    def set_api_key(self, api_key: str) -> None:
        """Set and encrypt API key."""
        from src.database.encryption import encrypt_api_key
        self.api_key_encrypted = encrypt_api_key(api_key)

    @property
    def model_list_json(self) -> list:
        """Get model list as Python list."""
        if not self.model_list:
            return []
        try:
            return json.loads(self.model_list)
        except json.JSONDecodeError:
            return []

    @model_list_json.setter
    def model_list_json(self, value: list) -> None:
        """Set model list from Python list."""
        self.model_list = json.dumps(value) if value else None

    @property
    def tier_assignment_dict(self) -> dict:
        """Get tier assignment as Python dict."""
        if not self.tier_assignment:
            return {}
        try:
            return json.loads(self.tier_assignment)
        except json.JSONDecodeError:
            return {}

    @tier_assignment_dict.setter
    def tier_assignment_dict(self, value: dict) -> None:
        """Set tier assignment from Python dict."""
        self.tier_assignment = json.dumps(value) if value else None

    @property
    def created_at(self) -> datetime:
        """Legacy alias for created_at_utc."""
        return self.created_at_utc

    def __repr__(self):
        return f"<ProviderConfig(id={self.id}, provider_type={self.provider_type}, is_active={self.is_active})>"

"""
Server configuration models.

Contains models for storing server connection configurations
for Cloudzy, Contabo, MT5, etc.
"""

import json
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from sqlalchemy import Column, String, DateTime, Boolean, Integer, Text
from ..models.base import Base

logger = logging.getLogger(__name__)


class ServerType(str, Enum):
    """Server type enumeration."""
    CLOUDZY = "cloudzy"
    CONTABO = "contabo"
    MT5 = "mt5"
    # Generic aliases (preferred)
    NODE_TRADING = "node_trading"  # alias for CLOUDZY
    NODE_BACKEND = "node_backend"  # alias for CONTABO


class ServerConfig(Base):
    """
    Server configuration for storing connection details.

    This table stores connection configurations for various servers
    used by QUANTMINDX (Cloudzy, Contabo, MT5).

    Attributes:
        id: Primary key (UUID)
        name: Display name for the server
        server_type: Type of server (cloudzy, contabo, mt5)
        host: Hostname or IP address
        port: Port number
        username: Username for authentication (encrypted)
        password: Password for authentication (encrypted)
        ssh_key_path: Path to SSH key file (optional)
        api_key: API key for the server (encrypted)
        is_active: Whether this server is active
        is_primary: Whether this is the primary server of this type
        metadata: JSON dict for extra configuration
        created_at_utc: Creation timestamp in UTC
        updated_at: Last update timestamp
    """
    __tablename__ = 'server_configs'

    id = Column(String(36), primary_key=True)
    name = Column(String(100), nullable=False)
    server_type = Column(String(20), nullable=False, index=True)
    host = Column(String(255), nullable=False)
    port = Column(Integer, nullable=False, default=22)
    username_encrypted = Column(Text, nullable=True)
    password_encrypted = Column(Text, nullable=True)
    ssh_key_path = Column(String(500), nullable=True)
    api_key_encrypted = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    is_primary = Column(Boolean, default=False, nullable=False)
    server_metadata = Column(Text, nullable=True)  # JSON dict
    created_at_utc = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)

    @property
    def username(self) -> Optional[str]:
        """Get decrypted username."""
        if not self.username_encrypted:
            return None
        from src.database.encryption import decrypt_api_key
        return decrypt_api_key(self.username_encrypted)

    @username.setter
    def username(self, value: Optional[str]) -> None:
        """Set and encrypt username."""
        if value:
            from src.database.encryption import encrypt_api_key
            self.username_encrypted = encrypt_api_key(value)
        else:
            self.username_encrypted = None

    @property
    def password(self) -> Optional[str]:
        """Get decrypted password."""
        if not self.password_encrypted:
            return None
        from src.database.encryption import decrypt_api_key
        return decrypt_api_key(self.password_encrypted)

    @password.setter
    def password(self, value: Optional[str]) -> None:
        """Set and encrypt password."""
        if value:
            from src.database.encryption import encrypt_api_key
            self.password_encrypted = encrypt_api_key(value)
        else:
            self.password_encrypted = None

    @property
    def api_key(self) -> Optional[str]:
        """Get decrypted API key."""
        if not self.api_key_encrypted:
            return None
        from src.database.encryption import decrypt_api_key
        return decrypt_api_key(self.api_key_encrypted)

    @api_key.setter
    def api_key(self, value: Optional[str]) -> None:
        """Set and encrypt API key."""
        if value:
            from src.database.encryption import encrypt_api_key
            self.api_key_encrypted = encrypt_api_key(value)
        else:
            self.api_key_encrypted = None

    @property
    def metadata_dict(self) -> dict:
        """Get metadata as Python dict."""
        if not self.server_metadata:
            return {}
        try:
            return json.loads(self.server_metadata)
        except json.JSONDecodeError:
            return {}

    @metadata_dict.setter
    def metadata_dict(self, value: dict) -> None:
        """Set metadata from Python dict."""
        self.server_metadata = json.dumps(value) if value else None

    def __repr__(self):
        return f"<ServerConfig(id={self.id}, name={self.name}, server_type={self.server_type}, is_active={self.is_active})>"

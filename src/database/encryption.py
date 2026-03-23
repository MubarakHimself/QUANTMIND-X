"""
Encryption utilities for secure storage of sensitive data.

Provides Fernet encryption with machine-local key derivation.
"""

import base64
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Machine UUID file location
_MACHINE_KEY_FILE = Path.home() / ".quantmind" / "machine.key"


def _ensure_machine_key_dir() -> None:
    """Ensure the machine key directory exists."""
    _MACHINE_KEY_FILE.parent.mkdir(parents=True, exist_ok=True)


def _get_or_create_machine_key() -> str:
    """
    Get or create a machine-local encryption key.

    The key is derived from the machine's unique identifier (UUID).
    If no machine UUID exists, generates one and stores it locally.

    Returns:
        Base64-encoded 32-byte key suitable for Fernet
    """
    _ensure_machine_key_dir()

    if _MACHINE_KEY_FILE.exists():
        try:
            with open(_MACHINE_KEY_FILE, 'r') as f:
                key = f.read().strip()
                if key:
                    logger.debug("Loaded existing machine key")
                    return key
        except Exception as e:
            logger.warning(f"Failed to read machine key: {e}")

    # Generate new machine UUID and derive key
    # Use multiple machine identifiers for better uniqueness
    machine_ids = []

    # Try to get machine UUID from various sources
    try:
        import platform
        machine_ids.append(platform.node())
    except Exception:
        pass

    try:
        machine_ids.append(str(uuid.getnode()))
    except Exception:
        pass

    try:
        machine_ids.append(str(uuid.uuid4()))
    except Exception:
        pass

    # Create a deterministic key from machine identifiers
    combined = "|".join(machine_ids) if machine_ids else str(uuid.uuid4())

    # Derive a 32-byte key using SHA256
    import hashlib
    key_bytes = hashlib.sha256(combined.encode()).digest()
    key = base64.urlsafe_b64encode(key_bytes).decode()

    try:
        with open(_MACHINE_KEY_FILE, 'w') as f:
            f.write(key)
        # Restrict permissions to owner only
        os.chmod(_MACHINE_KEY_FILE, 0o600)
        logger.info("Generated new machine-local encryption key")
    except Exception as e:
        logger.error(f"Failed to save machine key: {e}")

    return key


class SecureStorage:
    """
    Secure storage wrapper using Fernet encryption.

    Encrypts sensitive data at rest using a machine-local key.
    """

    _instance: Optional['SecureStorage'] = None
    _fernet: Optional[object] = None

    def __new__(cls) -> 'SecureStorage':
        """Singleton pattern for shared encryption state."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        """Initialize Fernet with machine key."""
        try:
            from cryptography.fernet import Fernet
            key = _get_or_create_machine_key()
            self._fernet = Fernet(key.encode())
            logger.info("SecureStorage initialized successfully")
        except ImportError:
            logger.warning("cryptography package not installed. Encryption disabled.")
            self._fernet = None
        except Exception as e:
            logger.error(f"Failed to initialize SecureStorage: {e}")
            self._fernet = None

    def encrypt(self, plaintext: str) -> str:
        """
        Encrypt plaintext string.

        Args:
            plaintext: String to encrypt

        Returns:
            Base64-encoded encrypted data, or original string if encryption unavailable
        """
        if not self._fernet:
            return plaintext

        try:
            encrypted = self._fernet.encrypt(plaintext.encode())
            return encrypted.decode()
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            return plaintext

    def decrypt(self, ciphertext: str) -> str:
        """
        Decrypt encrypted string.

        Args:
            ciphertext: Base64-encoded encrypted string

        Returns:
            Decrypted plaintext, or original string if decryption fails
        """
        if not self._fernet:
            return ciphertext

        try:
            decrypted = self._fernet.decrypt(ciphertext.encode())
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return ciphertext

    def is_available(self) -> bool:
        """Check if encryption is available."""
        return self._fernet is not None


# Convenience functions
def encrypt_api_key(api_key: Optional[str]) -> Optional[str]:
    """Encrypt an API key for storage."""
    if not api_key:
        return None
    storage = SecureStorage()
    return storage.encrypt(api_key)


def decrypt_api_key(encrypted_key: Optional[str]) -> Optional[str]:
    """Decrypt an API key from storage."""
    if not encrypted_key:
        return None
    storage = SecureStorage()
    return storage.decrypt(encrypted_key)

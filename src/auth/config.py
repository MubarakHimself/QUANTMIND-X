"""
Auth0 OAuth 2.1 Configuration

Loads OAuth provider settings from environment variables.
Client secret is NEVER committed — always loaded from env.
"""

import os
from dataclasses import dataclass
from typing import Optional

from src.config import get_internal_api_base_url


@dataclass
class Auth0Config:
    """Auth0 OAuth configuration."""

    domain: str
    client_id: str
    client_secret: Optional[str]  # Never log this
    audience: str
    callback_url: str
    logout_url: str
    # OAuth 2.1 PKCE params
    authorization_url: str
    token_url: str
    userinfo_url: str
    logout_callback_url: str
    # Security settings
    pkce_code_verifier_length: int = 64
    access_token_ttl_seconds: int = 900  # 15 minutes
    refresh_token_ttl_seconds: int = 604800  # 7 days


def get_auth0_config() -> Auth0Config:
    """
    Build Auth0Config from environment variables.

    Raises:
        ValueError: If required AUTH0_DOMAIN or AUTH0_CLIENT_ID are missing.
    """
    domain = os.getenv("AUTH0_DOMAIN", "").strip()
    client_id = os.getenv("AUTH0_CLIENT_ID", "").strip()
    client_secret = os.getenv("AUTH0_CLIENT_SECRET")
    audience = os.getenv("AUTH0_AUDIENCE", f"https://{domain}/api/v2/").strip()
    callback_path = os.getenv("AUTH0_CALLBACK_PATH", "/api/auth/callback").strip()
    logout_path = os.getenv("AUTH0_LOGOUT_PATH", "/api/auth/logout").strip()

    if not domain:
        raise ValueError("AUTH0_DOMAIN environment variable is required")
    if not client_id:
        raise ValueError("AUTH0_CLIENT_ID environment variable is required")

    # Build full callback/logout URLs based on API_BASE_URL env var
    api_base = (
        os.getenv("API_BASE_URL")
        or os.getenv("QUANTMIND_API_URL")
        or get_internal_api_base_url()
    ).rstrip("/")
    callback_url = f"{api_base}{callback_path}"
    logout_callback_url = f"{api_base}{logout_path}"

    return Auth0Config(
        domain=domain,
        client_id=client_id,
        client_secret=client_secret,
        audience=audience,
        callback_url=callback_url,
        logout_url=f"https://{domain}/v2/logout",
        authorization_url=f"https://{domain}/authorize",
        token_url=f"https://{domain}/oauth/token",
        userinfo_url=f"https://{domain}/userinfo",
        logout_callback_url=logout_callback_url,
        pkce_code_verifier_length=int(os.getenv("OAUTH_PKCE_CODE_VERIFIER_LENGTH", "64")),
        access_token_ttl_seconds=int(os.getenv("AUTH0_ACCESS_TOKEN_TTL", "900")),
        refresh_token_ttl_seconds=int(os.getenv("AUTH0_REFRESH_TOKEN_TTL", "604800")),
    )


# Cached config instance
_auth0_config: Optional[Auth0Config] = None


def get_auth0_config_cached() -> Auth0Config:
    """Get cached Auth0Config (lazy-loaded)."""
    global _auth0_config
    if _auth0_config is None:
        _auth0_config = get_auth0_config()
    return _auth0_config

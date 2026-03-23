"""
QuantMindX Authentication Module

Implements OAuth 2.1 authorization code flow with PKCE for secure authentication.
"""

from .config import get_auth0_config, Auth0Config
from .models import User, OAuthTokens, AuthSession
from .oauth import Auth0Client, get_auth0_client
from .session import SessionManager, get_session_manager
from .middleware import OAuthMiddleware
from .dependencies import get_current_user, require_role

__all__ = [
    "get_auth0_config",
    "Auth0Config",
    "User",
    "OAuthTokens",
    "AuthSession",
    "Auth0Client",
    "get_auth0_client",
    "SessionManager",
    "get_session_manager",
    "OAuthMiddleware",
    "get_current_user",
    "require_role",
]

"""
Auth0 OAuth 2.1 client with PKCE support.

Implements the authorization code flow with PKCE for secure authentication.
"""

import base64
import hashlib
import logging
import secrets
import urllib.parse
from typing import Optional, Tuple

import httpx

from .config import get_auth0_config, Auth0Config
from .models import OAuthTokens, User

logger = logging.getLogger("quantmind.auth")


class Auth0Client:
    """
    Auth0 OAuth 2.1 client.

    Implements:
    - Authorization code flow with PKCE
    - Token exchange and refresh
    - User info retrieval
    - Logout
    """

    def __init__(self, config: Optional[Auth0Config] = None):
        self.config = config or get_auth0_config()
        self._http = httpx.AsyncClient(timeout=30.0)

    async def close(self):
        """Close the HTTP client."""
        await self._http.aclose()

    # -------------------------------------------------------------------------
    # PKCE Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def generate_pkce_pair() -> Tuple[str, str]:
        """
        Generate a PKCE code verifier and challenge.

        Returns:
            Tuple of (code_verifier, code_challenge)
        """
        code_verifier = secrets.token_urlsafe(64)[:64]
        # SHA256 hash, base64url encoded (no padding)
        digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
        code_challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
        return code_verifier, code_challenge

    @staticmethod
    def generate_state() -> str:
        """Generate a cryptographically random state parameter."""
        return secrets.token_urlsafe(32)

    @staticmethod
    def generate_nonce() -> str:
        """Generate a cryptographically random nonce for ID token validation."""
        return secrets.token_urlsafe(32)

    # -------------------------------------------------------------------------
    # Authorization URL
    # -------------------------------------------------------------------------

    def get_authorization_url(
        self,
        state: str,
        nonce: str,
        code_challenge: str,
        redirect_uri: Optional[str] = None,
    ) -> str:
        """
        Build the Auth0 authorization URL with PKCE.

        Args:
            state: CSRF protection state
            nonce: Nonce for ID token validation
            code_challenge: PKCE code challenge (S256 method)
            redirect_uri: Override callback URL

        Returns:
            Full authorization URL to redirect the user to
        """
        redirect_uri = redirect_uri or self.config.callback_url

        params = {
            "response_type": "code",
            "client_id": self.config.client_id,
            "redirect_uri": redirect_uri,
            "scope": "openid profile email",
            "state": state,
            "nonce": nonce,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }

        return f"{self.config.authorization_url}?{urllib.parse.urlencode(params)}"

    # -------------------------------------------------------------------------
    # Token Exchange
    # -------------------------------------------------------------------------

    async def exchange_code(
        self,
        code: str,
        code_verifier: str,
        redirect_uri: Optional[str] = None,
    ) -> OAuthTokens:
        """
        Exchange authorization code for tokens.

        Args:
            code: Authorization code from Auth0 callback
            code_verifier: PKCE code verifier (random string)
            redirect_uri: Must match the redirect_uri used in authorization URL

        Returns:
            OAuthTokens with access_token, refresh_token, id_token, etc.

        Raises:
            httpx.HTTPStatusError: If token exchange fails
        """
        redirect_uri = redirect_uri or self.config.callback_url

        data = {
            "grant_type": "authorization_code",
            "client_id": self.config.client_id,
            "client_secret": self.config.client_secret,
            "code": code,
            "redirect_uri": redirect_uri,
            "code_verifier": code_verifier,
        }

        response = await self._http.post(self.config.token_url, data=data)
        response.raise_for_status()

        token_data = response.json()
        return OAuthTokens(**token_data)

    # -------------------------------------------------------------------------
    # Token Refresh
    # -------------------------------------------------------------------------

    async def refresh_access_token(self, refresh_token: str) -> OAuthTokens:
        """
        Use refresh token to obtain a new access token.

        Args:
            refresh_token: OAuth refresh token

        Returns:
            New OAuthTokens with fresh access_token (and possibly new refresh_token)

        Raises:
            httpx.HTTPStatusError: If refresh fails
        """
        data = {
            "grant_type": "refresh_token",
            "client_id": self.config.client_id,
            "client_secret": self.config.client_secret,
            "refresh_token": refresh_token,
        }

        response = await self._http.post(self.config.token_url, data=data)
        response.raise_for_status()

        token_data = response.json()
        return OAuthTokens(**token_data)

    # -------------------------------------------------------------------------
    # User Info
    # -------------------------------------------------------------------------

    async def get_user_info(self, access_token: str) -> User:
        """
        Retrieve user info from Auth0 /userinfo endpoint.

        Args:
            access_token: Valid OAuth access token

        Returns:
            User object with claims

        Raises:
            httpx.HTTPStatusError: If userinfo request fails
        """
        headers = {"Authorization": f"Bearer {access_token}"}
        response = await self._http.get(self.config.userinfo_url, headers=headers)
        response.raise_for_status()

        claims = response.json()

        # Extract roles from custom claim if present (set in Auth0 Action/Rules)
        roles = claims.pop("roles", [])

        return User(
            **claims,
            roles=roles,
        )

    # -------------------------------------------------------------------------
    # Logout
    # -------------------------------------------------------------------------

    def get_logout_url(self, logout_token: Optional[str] = None) -> str:
        """
        Build the Auth0 logout URL.

        Args:
            logout_token: Optional RP-initiated logout token (requires confirm=true)

        Returns:
            Full logout URL
        """
        params = {
            "client_id": self.config.client_id,
            "returnTo": self.config.logout_callback_url,
        }
        if logout_token:
            params["logout_token"] = logout_token

        return f"{self.config.logout_url}?{urllib.parse.urlencode(params)}"


# Singleton client instance
_auth0_client: Optional[Auth0Client] = None


def get_auth0_client() -> Auth0Client:
    """Get or create the singleton Auth0Client instance."""
    global _auth0_client
    if _auth0_client is None:
        _auth0_client = Auth0Client()
    return _auth0_client

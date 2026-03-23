"""
Auth data models for OAuth 2.1 flow.
"""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel


class User(BaseModel):
    """
    User extracted from OAuth ID token claims.

    Corresponds to the /userinfo endpoint response.
    """

    sub: str  # Auth0 user ID (e.g., "auth0|123456")
    email: str
    email_verified: bool = False
    name: Optional[str] = None
    nickname: Optional[str] = None
    picture: Optional[str] = None
    roles: List[str] = []  # Application-level roles from custom claim

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class OAuthTokens(BaseModel):
    """
    OAuth token pair received from Auth0 token endpoint.
    """

    access_token: str
    token_type: str = "Bearer"
    expires_in: int = 900  # seconds
    refresh_token: Optional[str] = None
    id_token: Optional[str] = None  # JWT ID token from Auth0
    scope: str = "openid profile email"

    @property
    def expires_at(self) -> datetime:
        """Calculate absolute expiry datetime."""
        return datetime.utcnow() + datetime.timedelta(seconds=self.expires_in)

    @property
    def is_expired(self) -> bool:
        """Check if access token has expired."""
        return datetime.utcnow() >= self.expires_at


class AuthSession(BaseModel):
    """
    Server-side session stored in Redis.

    Maps a session ID cookie to full OAuth state.
    """

    session_id: str
    user_sub: str  # Auth0 sub
    email: str
    email_verified: bool = False
    name: Optional[str] = None
    nickname: Optional[str] = None
    picture: Optional[str] = None
    roles: List[str] = []
    # Token state
    access_token: str  # Encrypted at rest
    refresh_token: Optional[str] = None  # Encrypted at rest
    expires_at: str  # ISO datetime string
    created_at: str  # ISO datetime string
    last_refreshed_at: Optional[str] = None  # ISO datetime string
    # Auth0 metadata
    auth0_domain: str
    auth0_client_id: str

    @property
    def is_expired(self) -> bool:
        """Check if the session's access token has expired."""
        return datetime.utcnow() >= datetime.fromisoformat(self.expires_at)

    def to_user(self) -> User:
        """Convert session to User model."""
        return User(
            sub=self.user_sub,
            email=self.email,
            email_verified=self.email_verified,
            name=self.name,
            nickname=self.nickname,
            picture=self.picture,
            roles=self.roles,
        )


class PKCEState(BaseModel):
    """
    PKCE authorization state — stored temporarily in Redis during OAuth flow.

    Keys: authorization_code -> { code_verifier, state, nonce, redirect_uri, created_at }
    """

    code_verifier: str
    state: str
    nonce: str
    redirect_uri: str
    created_at: str  # ISO datetime string

    @property
    def is_expired(self) -> bool:
        """PKCE state expires after 10 minutes."""
        from datetime import timedelta

        return datetime.utcnow() >= datetime.fromisoformat(self.created_at) + timedelta(minutes=10)


# --- Request/Response models for auth endpoints ---


class AuthCallbackRequest(BaseModel):
    """Query params received at /api/auth/callback."""

    code: Optional[str] = None
    state: Optional[str] = None
    error: Optional[str] = None
    error_description: Optional[str] = None


class AuthRefreshRequest(BaseModel):
    """Request body for /api/auth/refresh."""

    session_id: Optional[str] = None  # If not using cookie


class AuthMigrateRequest(BaseModel):
    """Request body for /api/auth/migrate."""

    legacy_token: str


class AuthMeResponse(BaseModel):
    """Response from /api/auth/me."""

    user: User
    session_id: str
    expires_at: str


class AuthMigrateResponse(BaseModel):
    """Response from /api/auth/migrate."""

    success: bool
    message: str
    user_id: Optional[int] = None


class AuthLogoutResponse(BaseModel):
    """Response from /api/auth/logout."""

    success: bool
    message: str

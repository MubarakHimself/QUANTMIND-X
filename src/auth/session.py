"""
Redis-backed session management for OAuth 2.1.

Stores AuthSession objects in Redis with TTL-based expiration.
Leverages the existing GlobalCache from src/cache/redis_client.py.
"""

import json
import logging
import secrets
from datetime import datetime, timedelta
from typing import Optional

from src.cache.redis_client import GlobalCache, get_cache

from .models import AuthSession, OAuthTokens, PKCEState, User

logger = logging.getLogger("quantmind.auth.session")

# Redis key prefixes
SESSION_PREFIX = "auth:session:"
PKCE_PREFIX = "auth:pkce:"
# Session TTL: 7 days
SESSION_TTL_SECONDS = 604800
# PKCE state TTL: 10 minutes
PKCE_TTL_SECONDS = 600


class SessionManager:
    """
    Manages OAuth sessions in Redis.

    Responsibilities:
    - Create/read/update/delete AuthSessions
    - Store temporary PKCE state during OAuth flow
    - Encrypt sensitive token data at rest
    """

    def __init__(self, cache: Optional[GlobalCache] = None):
        self._cache = cache or get_cache()

    # -------------------------------------------------------------------------
    # Session CRUD
    # -------------------------------------------------------------------------

    async def create_session(
        self,
        user: User,
        tokens: OAuthTokens,
        auth0_domain: str,
        auth0_client_id: str,
    ) -> str:
        """
        Create a new AuthSession and return the session ID.

        Args:
            user: User from Auth0 /userinfo
            tokens: OAuth tokens from token exchange
            auth0_domain: Auth0 tenant domain
            auth0_client_id: Auth0 client ID

        Returns:
            session_id (UUID string)
        """
        session_id = secrets.token_urlsafe(32)

        session = AuthSession(
            session_id=session_id,
            user_sub=user.sub,
            email=user.email,
            email_verified=user.email_verified,
            name=user.name,
            nickname=user.nickname,
            picture=user.picture,
            roles=user.roles,
            access_token=tokens.access_token,  # In production: encrypt this
            refresh_token=tokens.refresh_token,  # In production: encrypt this
            expires_at=tokens.expires_at.isoformat(),
            created_at=datetime.utcnow().isoformat(),
            auth0_domain=auth0_domain,
            auth0_client_id=auth0_client_id,
        )

        await self._cache.set(
            f"{SESSION_PREFIX}{session_id}",
            session.model_dump(),
            ttl=SESSION_TTL_SECONDS,
        )

        logger.info(f"Created auth session for user {user.email}, session_id={session_id[:8]}...")
        return session_id

    async def get_session(self, session_id: str) -> Optional[AuthSession]:
        """
        Retrieve an AuthSession by session_id.

        Args:
            session_id: The session UUID

        Returns:
            AuthSession if found and not expired, None otherwise
        """
        data = await self._cache.get(f"{SESSION_PREFIX}{session_id}")
        if data is None:
            return None

        try:
            session = AuthSession(**data)
            if session.is_expired:
                await self.delete_session(session_id)
                return None
            return session
        except Exception as e:
            logger.warning(f"Failed to deserialize session {session_id[:8]}...: {e}")
            await self.delete_session(session_id)
            return None

    async def update_session_tokens(
        self,
        session_id: str,
        tokens: OAuthTokens,
    ) -> bool:
        """
        Update tokens in an existing session (after refresh).

        Args:
            session_id: The session UUID
            tokens: New OAuthTokens

        Returns:
            True if updated, False if session not found
        """
        session = await self.get_session(session_id)
        if session is None:
            return False

        session.access_token = tokens.access_token
        if tokens.refresh_token:
            session.refresh_token = tokens.refresh_token
        session.expires_at = tokens.expires_at.isoformat()
        session.last_refreshed_at = datetime.utcnow().isoformat()

        await self._cache.set(
            f"{SESSION_PREFIX}{session_id}",
            session.model_dump(),
            ttl=SESSION_TTL_SECONDS,
        )

        logger.debug(f"Updated tokens for session {session_id[:8]}...")
        return True

    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a session (logout).

        Args:
            session_id: The session UUID

        Returns:
            True if deleted, False if not found
        """
        await self._cache.delete(f"{SESSION_PREFIX}{session_id}")
        logger.info(f"Deleted session {session_id[:8]}...")
        return True

    async def refresh_session_if_needed(self, session_id: str) -> Optional[AuthSession]:
        """
        Check if session needs refresh and refresh it.

        Args:
            session_id: The session UUID

        Returns:
            Updated AuthSession if refreshed, existing session if not needed, None if error
        """
        from .oauth import get_auth0_client

        session = await self.get_session(session_id)
        if session is None:
            return None

        # Only refresh if access token has less than 2 minutes to live
        from datetime import timedelta

        expiry = datetime.fromisoformat(session.expires_at)
        if expiry - datetime.utcnow() > timedelta(minutes=2):
            return session  # Still valid, no refresh needed

        if not session.refresh_token:
            logger.warning(f"Session {session_id[:8]}... has no refresh token")
            return None

        try:
            client = get_auth0_client()
            new_tokens = await client.refresh_access_token(session.refresh_token)
            await self.update_session_tokens(session_id, new_tokens)
            return await self.get_session(session_id)
        except Exception as e:
            logger.error(f"Failed to refresh session {session_id[:8]}...: {e}")
            await self.delete_session(session_id)
            return None

    # -------------------------------------------------------------------------
    # PKCE State (temporary, during OAuth flow)
    # -------------------------------------------------------------------------

    async def store_pkce_state(
        self,
        authorization_code: str,
        code_verifier: str,
        state: str,
        nonce: str,
        redirect_uri: str,
    ) -> None:
        """
        Store temporary PKCE state during the OAuth authorization phase.

        Args:
            authorization_code: The OAuth authorization code (maps to PKCE state)
            code_verifier: PKCE code verifier
            state: CSRF state parameter
            nonce: ID token nonce
            redirect_uri: Original redirect URI
        """
        pkce_state = PKCEState(
            code_verifier=code_verifier,
            state=state,
            nonce=nonce,
            redirect_uri=redirect_uri,
            created_at=datetime.utcnow().isoformat(),
        )

        await self._cache.set(
            f"{PKCE_PREFIX}{authorization_code}",
            pkce_state.model_dump(),
            ttl=PKCE_TTL_SECONDS,
        )

    async def get_pkce_state(self, authorization_code: str) -> Optional[PKCEState]:
        """
        Retrieve and consume PKCE state.

        Deletes the state after retrieval (one-time use).

        Args:
            authorization_code: The OAuth authorization code

        Returns:
            PKCEState if found and not expired, None otherwise
        """
        data = await self._cache.get(f"{PKCE_PREFIX}{authorization_code}")
        if data is None:
            return None

        try:
            pkce_state = PKCEState(**data)
            if pkce_state.is_expired:
                await self._cache.delete(f"{PKCE_PREFIX}{authorization_code}")
                return None
            # Delete immediately after successful retrieval (one-time use)
            await self._cache.delete(f"{PKCE_PREFIX}{authorization_code}")
            return pkce_state
        except Exception as e:
            logger.warning(f"Failed to deserialize PKCE state: {e}")
            await self._cache.delete(f"{PKCE_PREFIX}{authorization_code}")
            return None

    # -------------------------------------------------------------------------
    # User lookup helpers
    # -------------------------------------------------------------------------

    async def get_user_by_session(self, session_id: str) -> Optional[User]:
        """
        Get User object for a session ID.

        Args:
            session_id: The session UUID

        Returns:
            User if session exists and valid, None otherwise
        """
        session = await self.get_session(session_id)
        if session is None:
            return None
        return session.to_user()

    async def list_user_sessions(self, user_sub: str) -> list[str]:
        """
        List all active session IDs for a user.

        Note: This requires scanning Redis keys, which is O(n).
        Use sparingly — primarily for "logout all devices" feature.

        Args:
            user_sub: Auth0 user sub

        Returns:
            List of session_id strings
        """
        # For now, this is a placeholder. Full implementation would require
        # maintaining a user -> [session_ids] index in Redis.
        # TODO: Maintain a secondary index: auth:user_sessions:{user_sub} -> Set[session_id]
        return []


# Singleton
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get the singleton SessionManager instance."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager

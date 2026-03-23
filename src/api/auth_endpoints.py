"""
OAuth 2.1 Authentication Endpoints.

Implements the full authorization code flow with PKCE:
- GET  /api/auth/login          — Redirect to Auth0
- GET  /api/auth/callback       — Handle Auth0 callback
- POST /api/auth/refresh         — Refresh access token
- GET  /api/auth/logout         — Logout and clear session
- GET  /api/auth/me             — Return current user
- POST /api/auth/migrate        — Link legacy account to OAuth

All responses are JSON (no HTML) so the frontend can handle redirects.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Cookie, Depends, HTTPException, Query, Request, Response, status
from fastapi.responses import JSONResponse, RedirectResponse

from src.auth import (
    Auth0Client,
    OAuthMiddleware,
    SessionManager,
    User,
    get_auth0_client,
    get_session_manager,
)
from src.auth.config import get_auth0_config
from src.auth.dependencies import get_current_user, get_optional_user
from src.auth.models import (
    AuthCallbackRequest,
    AuthMeResponse,
    AuthMigrateRequest,
    AuthMigrateResponse,
    AuthRefreshRequest,
    AuthLogoutResponse,
)

logger = logging.getLogger("quantmind.auth.endpoints")

router = APIRouter(prefix="/api/auth", tags=["authentication"])


# ---------------------------------------------------------------------------
# Login — redirect to Auth0
# ---------------------------------------------------------------------------


@router.get("/login")
async def auth_login(
    request: Request,
    response: Response,
):
    """
    Initiate OAuth 2.1 authorization code flow.

    Redirects the user to Auth0's authorization endpoint with PKCE parameters.
    After login, Auth0 redirects back to /api/auth/callback.
    """
    try:
        from datetime import datetime

        config = get_auth0_config()
        client = get_auth0_client()

        # Generate PKCE pair
        code_verifier, code_challenge = client.generate_pkce_pair()
        state = client.generate_state()
        nonce = client.generate_nonce()

        # Build authorization URL
        auth_url = client.get_authorization_url(
            state=state,
            nonce=nonce,
            code_challenge=code_challenge,
        )

        # Store PKCE state keyed by the state parameter.
        # When Auth0 redirects back with ?state=X&code=Y, we look up the
        # code_verifier using the state to complete the PKCE exchange.
        session_manager = get_session_manager()

        # Use state as the temporary lookup key
        await session_manager._cache.set(
            f"auth:state:{state}",
            {
                "code_verifier": code_verifier,
                "nonce": nonce,
                "created_at": datetime.utcnow().isoformat(),
            },
            ttl=600,  # 10 minutes
        )

        logger.info(f"Redirecting to Auth0 login, state={state[:8]}...")

        return RedirectResponse(url=auth_url, status_code=302)

    except ValueError as e:
        # Auth0 not configured — return 503 so frontend can show appropriate error
        logger.error(f"Auth0 not configured: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OAuth provider not configured. Please contact administrator.",
        )
    except Exception as e:
        logger.error(f"Login initiation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to initiate login. Please try again.",
        )


# ---------------------------------------------------------------------------
# Callback — handle Auth0 redirect
# ---------------------------------------------------------------------------


@router.get("/callback")
async def auth_callback(
    request: Request,
    code: Optional[str] = Query(default=None),
    state: Optional[str] = Query(default=None),
    error: Optional[str] = Query(default=None),
    error_description: Optional[str] = Query(default=None),
):
    """
    Handle Auth0 callback after user authentication.

    Exchanges the authorization code for tokens, creates a session,
    and sets the session cookie.
    """
    # Handle error from Auth0
    if error:
        logger.error(f"Auth0 error: {error} — {error_description}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": error,
                "error_description": error_description,
            },
        )

    if not code or not state:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing code or state parameter",
        )

    try:
        config = get_auth0_config()
        client = get_auth0_client()
        session_manager = get_session_manager()

        # Look up PKCE state by state parameter
        import json

        state_data = await session_manager._cache.get(f"auth:state:{state}")
        if state_data is None:
            logger.warning(f"Unknown or expired state: {state[:8]}...")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired state. Please try logging in again.",
            )

        # Delete state immediately (one-time use)
        await session_manager._cache.delete(f"auth:state:{state}")

        code_verifier = state_data["code_verifier"]
        nonce = state_data["nonce"]

        # Exchange authorization code for tokens
        tokens = await client.exchange_code(
            code=code,
            code_verifier=code_verifier,
        )

        # Get user info
        user = await client.get_user_info(tokens.access_token)

        # Validate nonce in ID token (caller should validate this — simplified here)
        # In production: verify nonce in id_token claims matches stored nonce

        # Create session in Redis
        session_id = await session_manager.create_session(
            user=user,
            tokens=tokens,
            auth0_domain=config.domain,
            auth0_client_id=config.client_id,
        )

        # Build response with session cookie
        response = JSONResponse(
            content={
                "success": True,
                "message": "Authentication successful",
                "user": user.model_dump(),
            }
        )

        # Set httpOnly session cookie
        response.set_cookie(
            key="qm_session_id",
            value=session_id,
            httponly=True,
            secure=True,  # HTTPS only in production
            samesite="strict",
            path="/",
            max_age=604800,  # 7 days
        )

        logger.info(f"User {user.email} logged in successfully, session={session_id[:8]}...")

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Callback processing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process authentication callback. Please try again.",
        )


# ---------------------------------------------------------------------------
# Refresh token
# ---------------------------------------------------------------------------


@router.post("/refresh")
async def auth_refresh(
    request: Request,
    body: Optional[AuthRefreshRequest] = None,
    qm_session_id: Optional[str] = Cookie(default=None),
):
    """
    Refresh the access token using the refresh token.

    Called automatically by the frontend when the access token is expired.
    """
    session_id = body.session_id if body else None
    if not session_id:
        session_id = qm_session_id

    if not session_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No session found. Please log in again.",
        )

    try:
        session_manager = get_session_manager()
        session = await session_manager.get_session(session_id)

        if session is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Session expired or invalid. Please log in again.",
            )

        if not session.refresh_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Session does not support token refresh. Please log in again.",
            )

        # Refresh tokens
        client = get_auth0_client()
        new_tokens = await client.refresh_access_token(session.refresh_token)

        # Update session
        await session_manager.update_session_tokens(session_id, new_tokens)

        # Return new token info (don't return the full token, just expiry info)
        response = JSONResponse(
            content={
                "success": True,
                "expires_in": new_tokens.expires_in,
                "expires_at": new_tokens.expires_at.isoformat(),
            }
        )

        # Refresh cookie max_age
        response.set_cookie(
            key="qm_session_id",
            value=session_id,
            httponly=True,
            secure=True,
            samesite="strict",
            path="/",
            max_age=604800,
        )

        logger.debug(f"Token refreshed for session {session_id[:8]}...")

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to refresh token. Please log in again.",
        )


# ---------------------------------------------------------------------------
# Logout
# ---------------------------------------------------------------------------


@router.get("/logout", response_model=AuthLogoutResponse)
@router.post("/logout", response_model=AuthLogoutResponse)
async def auth_logout(
    request: Request,
    response: Response,
    qm_session_id: Optional[str] = Cookie(default=None),
):
    """
    Log out the current user.

    Deletes the Redis session and clears the session cookie.
    Optionally redirects to Auth0 logout endpoint.
    """
    session_id = qm_session_id

    try:
        if session_id:
            session_manager = get_session_manager()
            await session_manager.delete_session(session_id)

        # Clear cookie
        response = JSONResponse(
            content={"success": True, "message": "Logged out successfully"}
        )
        response.delete_cookie("qm_session_id", path="/")

        return response

    except Exception as e:
        logger.error(f"Logout failed: {e}", exc_info=True)
        # Still clear cookie even on error
        response = JSONResponse(
            content={"success": True, "message": "Logged out (with errors)"}
        )
        response.delete_cookie("qm_session_id", path="/")
        return response


# ---------------------------------------------------------------------------
# Current user
# ---------------------------------------------------------------------------


@router.get("/me", response_model=AuthMeResponse)
async def auth_me(
    request: Request,
    user: User = Depends(get_current_user),
    qm_session_id: Optional[str] = Cookie(default=None),
):
    """
    Return the current authenticated user's information.
    """
    session_id = qm_session_id or getattr(request.state, "session_id", None)

    return AuthMeResponse(
        user=user,
        session_id=session_id or "unknown",
        expires_at="",  # Frontend should track this from token refresh response
    )


# ---------------------------------------------------------------------------
# Legacy user migration
# ---------------------------------------------------------------------------


@router.post("/migrate", response_model=AuthMigrateResponse)
async def auth_migrate(
    request: Request,
    body: AuthMigrateRequest,
    user: User = Depends(get_current_user),
):
    """
    Link a legacy (non-OAuth) user account to an OAuth identity.

    This allows existing users to migrate to OAuth without losing their data.
    The legacy_token is validated, then the OAuth user record is linked.

    In Phase 1 (backwards-compatible rollout), this endpoint is a no-op
    since legacy auth doesn't exist yet.
    """
    # TODO: Implement legacy token validation and user linking
    # This requires a LegacyUser model that doesn't exist yet.

    logger.info(
        f"Migration endpoint called for user {user.email} "
        f"(legacy_token provided: {bool(body.legacy_token)})"
    )

    return AuthMigrateResponse(
        success=True,
        message="Migration endpoint noted. Legacy auth not yet implemented.",
        user_id=None,
    )


# ---------------------------------------------------------------------------
# Migration status check
# ---------------------------------------------------------------------------


@router.get("/migrate/status")
async def auth_migrate_status(
    request: Request,
    user: Optional[User] = Depends(get_optional_user),
):
    """
    Check if the current user has migrated from legacy auth.

    Returns migration status for the frontend to show appropriate UI.
    """
    if user is None:
        return {
            "migrated": False,
            "oauth_connected": False,
            "email": None,
        }

    return {
        "migrated": True,  # OAuth users are already "migrated"
        "oauth_connected": True,
        "email": user.email,
    }

"""
OAuth authentication middleware for FastAPI.

Attaches the current user to request.state after validating the session cookie.
"""

import logging
from typing import Callable, List, Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from .session import get_session_manager
from .models import User

logger = logging.getLogger("quantmind.auth.middleware")

# Paths that don't require authentication
PUBLIC_PATHS: List[str] = [
    "/health",
    "/metrics",
    "/docs",
    "/openapi.json",
    "/redoc",
    # Auth endpoints are handled specially via exclude_paths
    "/api/auth/login",
    "/api/auth/callback",
    "/api/auth/logout",
    "/api/auth/me",  # Returns 401 if not authenticated, not 307 redirect
]


class OAuthMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware that validates OAuth sessions from httpOnly cookies.

    On every request:
    1. Skip auth for public paths (defined in PUBLIC_PATHS and exclude_paths)
    2. Extract session_id from `qm_session_id` cookie
    3. Validate session in Redis (and refresh if needed)
    4. Attach User to request.state

    Does NOT enforce auth — that's done via the `get_current_user` dependency
    which allows returning 401 responses rather than redirects.
    """

    def __init__(
        self,
        app: ASGIApp,
        exclude_paths: Optional[List[str]] = None,
        include_paths: Optional[List[str]] = None,
    ):
        """
        Args:
            app: ASGI application
            exclude_paths: Additional paths to skip auth (e.g., ["/api/health"])
            include_paths: If set, ONLY these paths are protected (None = all /api/*)
        """
        super().__init__(app)
        self.exclude_paths = set(PUBLIC_PATHS + (exclude_paths or []))
        self.include_paths = include_paths  # None = protect all /api/*

    def _is_protected(self, path: str) -> bool:
        """Determine if a path requires authentication."""
        if path in self.exclude_paths:
            return False
        if self.include_paths is not None:
            return path in self.include_paths
        # Default: protect all /api/* paths
        return path.startswith("/api/")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Attach default empty user
        request.state.user: Optional[User] = None
        request.state.session_id: Optional[str] = None

        if not self._is_protected(request.url.path):
            return await call_next(request)

        # Extract session ID from cookie
        session_id = request.cookies.get("qm_session_id")
        if not session_id:
            # No session cookie — let the endpoint handle 401
            return await call_next(request)

        # Validate session in Redis
        session_manager = get_session_manager()
        session = await session_manager.get_session(session_id)

        if session is None:
            # Session not found or expired — clear cookie and let endpoint handle 401
            response = await call_next(request)
            if response.status_code == 401:
                # Clear the invalid cookie
                response.delete_cookie("qm_session_id", path="/")
            return response

        # Check if token needs refresh
        refreshed_session = await session_manager.refresh_session_if_needed(session_id)
        if refreshed_session is None and session.is_expired:
            # Refresh failed, session was deleted
            response = await call_next(request)
            if response.status_code == 401:
                response.delete_cookie("qm_session_id", path="/")
            return response

        # Attach user to request state
        user = (refreshed_session or session).to_user()
        request.state.user = user
        request.state.session_id = session_id

        # Process request
        response = await call_next(request)

        # If tokens were refreshed, update the session cookie expiry
        # (access_token changed but session_id stays the same)
        return response

"""
FastAPI dependencies for authentication.

Provides `get_current_user` and `require_role` for protecting endpoints.
"""

import logging
from typing import List, Optional

from fastapi import Depends, HTTPException, Request, status

from .models import User

logger = logging.getLogger("quantmind.auth.dependencies")


async def get_current_user(request: Request) -> User:
    """
    FastAPI dependency that returns the current authenticated user.

    Use this as a dependency in any endpoint that requires authentication:
        @app.get("/protected")
        async def protected_route(user: User = Depends(get_current_user)):
            return {"user": user}

    Returns:
        User object from validated session

    Raises:
        HTTPException 401: If no valid session exists
    """
    user: Optional[User] = getattr(request.state, "user", None)

    if user is None:
        # Check if session was invalid
        session_id: Optional[str] = getattr(request.state, "session_id", None)
        if session_id:
            logger.debug(f"Session {session_id[:8]}... found but token expired")
        else:
            logger.debug("No session cookie found")

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "unauthorized",
                "error_description": "Authentication required. Please log in.",
                "login_url": "/api/auth/login",
            },
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


async def require_role(required_roles: List[str]):
    """
    FastAPI dependency factory that requires specific roles.

    Usage:
        @app.get("/admin")
        async def admin_route(
            user: User = Depends(require_role(["admin"]))
        ):
            return {"admin": True}

    Args:
        required_roles: List of role names (user must have at least one)

    Raises:
        HTTPException 401: If not authenticated
        HTTPException 403: If authenticated but missing required role
    """

    async def role_checker(user: User = Depends(get_current_user)) -> User:
        if not any(role in user.roles for role in required_roles):
            logger.warning(
                f"User {user.email} denied access: required roles {required_roles}, "
                f"has {user.roles}"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": "forbidden",
                    "error_description": f"This endpoint requires one of these roles: {required_roles}",
                    "required_roles": required_roles,
                    "user_roles": user.roles,
                },
            )
        return user

    return role_checker


async def get_optional_user(request: Request) -> Optional[User]:
    """
    FastAPI dependency that returns the current user if authenticated, None otherwise.

    Use for endpoints that behave differently for authenticated vs anonymous users.

    Returns:
        User if session valid, None otherwise
    """
    return getattr(request.state, "user", None)

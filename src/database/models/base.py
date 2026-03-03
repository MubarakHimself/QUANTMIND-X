"""
Base models and utilities.

Contains the SQLAlchemy Base, enums, and session utilities.
"""

from datetime import datetime, timezone
from enum import Enum as PyEnum

from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Enum
from sqlalchemy.orm import declarative_base, Session
from typing import Generator

# Import session from engine for FastAPI dependency injection
from ..engine import get_session

Base = declarative_base()


def get_db_session() -> Generator[Session, None, None]:
    """
    FastAPI dependency for database session management.

    Yields a database session and ensures cleanup after request.

    Usage:
        @router.get("/items")
        async def get_items(db: Session = Depends(get_db_session)):
            items = db.query(Item).all()
            return items
    """
    session = get_session()
    try:
        yield session
    finally:
        session.close()


class TradingMode(PyEnum):
    """Trading mode enum for demo/live distinction."""
    DEMO = "demo"
    LIVE = "live"

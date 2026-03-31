"""
Base models and utilities.

Contains the SQLAlchemy Base, enums, and session utilities.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
from enum import Enum as PyEnum

from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Enum, JSON
from sqlalchemy.orm import declarative_base, Session
from typing import Generator

# Import session from engine for FastAPI dependency injection
from ..engine import get_session

Base = declarative_base()


def get_db_session():
    """
    FastAPI dependency for database session management.

    Compatible with FastAPI Depends() and also works as a context manager.

    Usage (FastAPI dependency — RECOMMENDED):
        @router.get("/items")
        async def get_items(db: Session = Depends(get_db_session)):
            items = db.query(Item).all()
            return items

    Usage (context manager):
        with get_db_session() as db:
            items = db.query(Item).all()
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


class AccountType(PyEnum):
    """Account type enum for personal vs prop firm accounts."""
    PERSONAL = "personal"
    PROP_FIRM = "prop_firm"

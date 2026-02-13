"""
Database Manager for SQLAlchemy ORM sessions.
"""

import os
from contextlib import contextmanager
from typing import Optional
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError

from .models import Base


class DBManager:
    def __init__(self, db_url: Optional[str] = None):
        if db_url is None:
            db_url = "sqlite:///data/quantmind.db"
        self.engine = create_engine(db_url, echo=False)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

    @property
    def session(self) -> Session:
        """Direct session access for simple queries."""
        if not hasattr(self, '_session') or self._session is None:
            self._session = self.SessionLocal()
        return self._session

    @contextmanager
    def get_session(self):
        """Context manager for session with commit/rollback."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except SQLAlchemyError:
            session.rollback()
            raise
        finally:
            session.close()

    def commit(self):
        """Manual commit for direct session usage."""
        if hasattr(self, '_session') and self._session:
            self._session.commit()

    def close(self):
        """Close session."""
        if hasattr(self, '_session') and self._session:
            self._session.close()
            self._session = None
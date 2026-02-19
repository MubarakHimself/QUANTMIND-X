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


# Default database paths
DEFAULT_SQLITE_PATH = "sqlite:///data/quantmind.db"


def get_hot_db_url() -> Optional[str]:
    """
    Get HOT tier PostgreSQL URL from environment variable.
    
    Returns:
        HOT database URL if HOT_DB_URL environment variable is set, None otherwise
    """
    return os.environ.get('HOT_DB_URL')


class DBManager:
    def __init__(self, db_url: Optional[str] = None, is_hot: bool = False):
        """
        Initialize DBManager.
        
        Args:
            db_url: Database URL (defaults to SQLite if not provided)
            is_hot: If True, try to use HOT tier PostgreSQL from HOT_DB_URL env var
        """
        if db_url is None:
            # Try to get HOT_DB_URL if is_hot is True
            if is_hot:
                hot_url = get_hot_db_url()
                if hot_url:
                    db_url = hot_url
                else:
                    # HOT requested but no URL - fall back to default
                    db_url = DEFAULT_SQLITE_PATH
                    import logging
                    logging.getLogger(__name__).warning(
                        "HOT tier requested but HOT_DB_URL not set, falling back to default SQLite"
                    )
            else:
                db_url = DEFAULT_SQLITE_PATH
        
        self.engine = create_engine(db_url, echo=False)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self._db_url = db_url

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


class HOTDBManager(DBManager):
    """
    HOT tier DBManager for PostgreSQL tick_cache queries.
    
    This class specifically targets the HOT tier PostgreSQL database
    that contains real-time tick data.
    """
    
    def __init__(self, db_url: Optional[str] = None):
        """
        Initialize HOT DBManager.
        
        Args:
            db_url: Optional explicit database URL. If not provided,
                   uses HOT_DB_URL from environment.
        """
        if db_url is None:
            db_url = get_hot_db_url()
            if db_url is None:
                import logging
                logging.getLogger(__name__).error(
                    "HOTDBManager initialized without HOT_DB_URL environment variable set. "
                    "Tick cache queries will fail."
                )
        
        super().__init__(db_url=db_url, is_hot=False)
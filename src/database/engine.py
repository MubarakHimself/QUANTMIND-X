"""
Database Engine Configuration

Configures SQLAlchemy engine with SQLite backend and session management.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.pool import StaticPool

# Database path
DB_PATH = os.path.join(os.path.dirname(__file__), '../../data/db/quantmind.db')
DB_ABSOLUTE_PATH = os.path.abspath(DB_PATH)

# Ensure data directory exists
os.makedirs(os.path.dirname(DB_ABSOLUTE_PATH), exist_ok=True)

# Create engine with SQLite-specific optimizations
engine = create_engine(
    f'sqlite:///{DB_ABSOLUTE_PATH}',
    connect_args={
        'check_same_thread': False,  # Allow multi-threaded access
    },
    poolclass=StaticPool,  # Simple pool for SQLite
    echo=False,  # Set to True for SQL query logging
)

# Enable foreign key constraints (SQLite requires this)
from sqlalchemy import event
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_conn, connection_record):
    """Enable foreign keys in SQLite."""
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()

# Create session factory
session_factory = sessionmaker(bind=engine)
Session = scoped_session(session_factory)


def get_engine():
    """Get the database engine."""
    return engine


def get_session():
    """Get a thread-local session."""
    return Session()


def close_session():
    """Close the thread-local session."""
    Session.remove()


def init_database():
    """
    Initialize all database tables.

    This function is idempotent - safe to call multiple times.
    It will create tables if they don't exist, but won't drop existing data.
    """
    from .models import Base
    Base.metadata.create_all(bind=engine)
    print(f"Database initialized at: {DB_ABSOLUTE_PATH}")

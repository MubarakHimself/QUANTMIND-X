"""
Database Engine Configuration

Configures SQLAlchemy engine with SQLite backend and session management.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.pool import StaticPool
from src.config import get_database_url

# Get database URL from config (supports environment variable override)
database_url = get_database_url()

# If using SQLite, handle file path
if database_url.startswith('sqlite'):
    # Extract path from sqlite:/// URI
    db_path = database_url.replace('sqlite:///', '')
    if not db_path.startswith('/'):
        # Relative path - make it relative to project root
        DB_ABSOLUTE_PATH = os.path.abspath(db_path)
    else:
        DB_ABSOLUTE_PATH = db_path
    # Ensure data directory exists
    os.makedirs(os.path.dirname(DB_ABSOLUTE_PATH), exist_ok=True)
    engine_url = f'sqlite:///{DB_ABSOLUTE_PATH}'
else:
    # Use as-is for PostgreSQL, MySQL, etc.
    engine_url = database_url

# Create engine with SQLite-specific optimizations
engine = create_engine(
    engine_url,
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

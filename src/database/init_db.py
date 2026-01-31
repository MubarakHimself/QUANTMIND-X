"""
Database Initialization Module

Provides manual `create_all()` initialization for SQLite database tables.
This module implements idempotent table creation without Alembic migrations.
"""

import logging
from pathlib import Path

from .engine import engine
from .models import Base

# Configure logging
logger = logging.getLogger(__name__)


def init_database(drop_all: bool = False) -> bool:
    """
    Initialize all database tables.

    This function is idempotent - safe to call multiple times.
    It will create tables if they don't exist, but won't drop existing data
    unless drop_all=True.

    Args:
        drop_all: If True, drop all existing tables before creating.
                  WARNING: This will delete all existing data.

    Returns:
        True if initialization was successful, False otherwise.
    """
    try:
        db_path = engine.url.database
        logger.info(f"Database path: {db_path}")

        # Ensure parent directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        if drop_all:
            logger.warning("Dropping all existing tables...")
            Base.metadata.drop_all(bind=engine)

        # Create all tables
        logger.info("Creating database tables...")
        Base.metadata.create_all(bind=engine)

        # List created tables
        created_tables = list(Base.metadata.tables.keys())
        logger.info(f"Successfully created {len(created_tables)} tables: {created_tables}")

        return True

    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        return False


def is_initialized() -> bool:
    """
    Check if database tables have been created.

    Returns:
        True if all expected tables exist, False otherwise.
    """
    try:
        from sqlalchemy import inspect

        inspector = inspect(engine)
        existing_tables = set(inspector.get_table_names())
        expected_tables = {
            'prop_firm_accounts',
            'daily_snapshots',
            'trade_proposals',
            'agent_tasks',
            'strategy_performance'
        }

        return expected_tables.issubset(existing_tables)

    except Exception as e:
        logger.error(f"Failed to check database initialization status: {e}")
        return False


def get_table_info() -> dict:
    """
    Get information about database tables.

    Returns:
        Dictionary with table names as keys and column info as values.
    """
    try:
        from sqlalchemy import inspect

        inspector = inspect(engine)
        tables = {}

        for table_name in inspector.get_table_names():
            columns = []
            for column in inspector.get_columns(table_name):
                columns.append({
                    'name': column['name'],
                    'type': str(column['type']),
                    'nullable': column['nullable'],
                    'primary_key': column.get('primary_key', False),
                })

            tables[table_name] = {
                'columns': columns,
                'row_count': _get_row_count(table_name)
            }

        return tables

    except Exception as e:
        logger.error(f"Failed to get table info: {e}")
        return {}


def _get_row_count(table_name: str) -> int:
    """Get the row count for a specific table."""
    try:
        with engine.connect() as conn:
            result = conn.execute(f"SELECT COUNT(*) FROM {table_name}")
            return result.scalar()
    except Exception:
        return 0


if __name__ == "__main__":
    # Run initialization when executed directly
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if init_database():
        print("\nDatabase initialized successfully!")
        print("\nTable Information:")
        for table_name, info in get_table_info().items():
            print(f"  {table_name}: {info['row_count']} rows")
    else:
        print("\nFailed to initialize database!")

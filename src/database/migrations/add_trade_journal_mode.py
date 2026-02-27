"""
Migration: Add mode columns to trading tables
==============================================

This migration adds the missing `mode` column to multiple tables that have
the column defined in models but missing from the database schema.

Tables affected:
- trade_journal: Add mode column (VARCHAR(10), default 'live')
- daily_snapshots: Add mode column (VARCHAR(10), default 'live')
- strategy_performance: Add mode column (VARCHAR(10), default 'live')
- bot_circuit_breaker: Add mode column (VARCHAR(10), default 'live')
- crypto_trades: Add mode column (VARCHAR(10), default 'live')

This script is idempotent - safe to run multiple times.

Version: 009
"""

import logging
from sqlalchemy import text, inspect

logger = logging.getLogger(__name__)


def _get_engine():
    """Get the database engine (lazy import to avoid circular dependencies)."""
    from src.database.engine import engine
    return engine


def check_column_exists(table_name: str, column_name: str) -> bool:
    """Check if a column exists in a table."""
    engine = _get_engine()
    inspector = inspect(engine)
    try:
        columns = [col['name'] for col in inspector.get_columns(table_name)]
        return column_name in columns
    except Exception:
        return False


# Define all tables that need the mode column
MODE_COLUMN_TABLES = [
    ('trade_journal', 'live'),
    ('daily_snapshots', 'live'),
    ('strategy_performance', 'live'),
    ('bot_circuit_breaker', 'live'),
    ('crypto_trades', 'live'),
]


def run_migration() -> bool:
    """
    Add the mode column to all tables that are missing it.

    Returns:
        True if migration succeeded or was already applied.
    """
    try:
        engine = _get_engine()
        success = True

        for table_name, default_value in MODE_COLUMN_TABLES:
            # Check if column already exists
            if check_column_exists(table_name, 'mode'):
                logger.info(f"Migration skipped: mode column already exists in {table_name}")
                continue

            try:
                # Add the mode column with default
                with engine.begin() as conn:
                    conn.execute(text(f"""
                        ALTER TABLE {table_name}
                        ADD COLUMN mode VARCHAR(10) NOT NULL DEFAULT '{default_value}'
                    """))

                    # Create index for the mode column
                    conn.execute(text(f"""
                        CREATE INDEX IF NOT EXISTS idx_{table_name}_mode
                        ON {table_name}(mode)
                    """))

                logger.info(f"Migration completed: added mode column to {table_name}")
            except Exception as e:
                logger.error(f"Failed to add mode column to {table_name}: {e}")
                success = False

        return success

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return False


def rollback_migration() -> bool:
    """
    Rollback the migration by dropping the indexes.
    Note: SQLite doesn't support DROP COLUMN directly in older versions,
    so we only remove the indexes.

    Returns:
        True if rollback succeeded.
    """
    try:
        engine = _get_engine()
        with engine.begin() as conn:
            for table_name, _ in MODE_COLUMN_TABLES:
                conn.execute(text(f"DROP INDEX IF EXISTS idx_{table_name}_mode"))

        logger.info("Rollback completed: removed mode indexes")
        return True

    except Exception as e:
        logger.error(f"Rollback failed: {e}")
        return False


if __name__ == "__main__":
    # Run migration directly
    logging.basicConfig(level=logging.INFO)
    success = run_migration()
    print(f"Migration {'succeeded' if success else 'failed'}")

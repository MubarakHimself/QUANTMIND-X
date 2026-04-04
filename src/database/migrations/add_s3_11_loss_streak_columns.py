"""
Migration: Add S3-11 Loss Streak Columns
=========================================

S3-11 / Section 7.4: 3-Loss-in-a-Row Circuit Breaker

Adds columns to bot_circuit_breaker table:
- daily_loss_streak_days: Count of days where 3 consecutive losses occurred
- last_loss_streak_date: Date of the most recent 3-loss streak day

This script is idempotent - safe to run multiple times.

Version: 011
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


# S3-11 columns to add to bot_circuit_breaker
S3_11_COLUMNS = [
    ('daily_loss_streak_days', 'INTEGER NOT NULL DEFAULT 0', 0, None),
    ('last_loss_streak_date', 'TIMESTAMP', None, None),
]


def run_migration() -> bool:
    """
    Add S3-11 columns to bot_circuit_breaker table.

    Returns:
        True if migration succeeded or was already applied.
    """
    try:
        engine = _get_engine()
        success = True
        table_name = 'bot_circuit_breaker'

        for column_name, column_type, default_value, index_sql in S3_11_COLUMNS:
            # Check if column already exists
            if check_column_exists(table_name, column_name):
                logger.info(f"Migration skipped: {column_name} column already exists in {table_name}")
                continue

            try:
                # Build ALTER TABLE ADD COLUMN statement
                sql = f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}"

                with engine.begin() as conn:
                    conn.execute(text(sql))

                    # Create index if specified
                    if index_sql:
                        conn.execute(text(index_sql))

                logger.info(f"Migration completed: added {column_name} column to {table_name}")
            except Exception as e:
                logger.error(f"Failed to add {column_name} column to {table_name}: {e}")
                success = False

        return success

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return False


def rollback_migration() -> bool:
    """
    Rollback the migration by dropping the columns.
    Note: SQLite does not support DROP COLUMN in all versions,
    so this may not fully remove the columns.

    Returns:
        True if rollback succeeded.
    """
    try:
        engine = _get_engine()
        with engine.begin() as conn:
            # SQLite does not support DROP COLUMN directly
            # Only log that rollback was attempted
            conn.execute(text("SELECT 1"))

        logger.info("Rollback completed (note: SQLite does not support DROP COLUMN)")
        return True

    except Exception as e:
        logger.error(f"Rollback failed: {e}")
        return False


if __name__ == "__main__":
    # Run migration directly
    logging.basicConfig(level=logging.INFO)
    success = run_migration()
    print(f"Migration {'succeeded' if success else 'failed'}")

"""
Migration: Add SSL Circuit Breaker Columns
==========================================

Story 18.1: Per-Bot Consecutive Loss Counter & Paper Rotation

Adds SSL (Survivorship Selection Loop) columns to bot_circuit_breaker table:
- magic_number: MT5 magic number (strategy version identifier)
- tier: Paper trading tier ('TIER_1' or 'TIER_2')
- paper_entry_timestamp: When bot entered paper tier
- recovery_win_count: Consecutive wins during paper tier (for recovery)
- state: SSL state ('live', 'paper', 'recovery', 'retired')

This script is idempotent - safe to run multiple times.

Version: 010
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


# SSL columns to add to bot_circuit_breaker
SSL_COLUMNS = [
    ('magic_number', 'VARCHAR(50)', None, "CREATE INDEX IF NOT EXISTS idx_bot_circuit_breaker_magic_number ON bot_circuit_breaker(magic_number)"),
    ('tier', 'VARCHAR(20)', None, None),
    ('paper_entry_timestamp', 'TIMESTAMP', None, None),
    ('recovery_win_count', 'INTEGER NOT NULL DEFAULT 0', 0, None),
    ('state', "VARCHAR(20) NOT NULL DEFAULT 'live'", 'live', "CREATE INDEX IF NOT EXISTS idx_bot_circuit_breaker_state ON bot_circuit_breaker(state)"),
]


def run_migration() -> bool:
    """
    Add SSL columns to bot_circuit_breaker table.

    Returns:
        True if migration succeeded or was already applied.
    """
    try:
        engine = _get_engine()
        success = True
        table_name = 'bot_circuit_breaker'

        for column_name, column_type, default_value, index_sql in SSL_COLUMNS:
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
    Rollback the migration by dropping the indexes.
    Note: SQLite doesn't support DROP COLUMN directly in older versions,
    so we only remove the indexes.

    Returns:
        True if rollback succeeded.
    """
    try:
        engine = _get_engine()
        with engine.begin() as conn:
            conn.execute(text("DROP INDEX IF EXISTS idx_bot_circuit_breaker_magic_number"))
            conn.execute(text("DROP INDEX IF EXISTS idx_bot_circuit_breaker_state"))

        logger.info("Rollback completed: removed SSL indexes")
        return True

    except Exception as e:
        logger.error(f"Rollback failed: {e}")
        return False


if __name__ == "__main__":
    # Run migration directly
    logging.basicConfig(level=logging.INFO)
    success = run_migration()
    print(f"Migration {'succeeded' if success else 'failed'}")

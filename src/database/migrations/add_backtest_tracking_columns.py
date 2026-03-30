"""
Migration: Add Backtest Tracking Columns
=========================================

Adds backtest-specific columns to strategy_performance table:
- variant: Backtest variant (vanilla, spiced, vanilla_full, spiced_full)
- symbol: Trading symbol (EURUSD, GBPUSD, etc.)
- timeframe: Timeframe (M1, M5, M15, M30, H1, H4, D1)
- parent_id: Parent strategy performance ID for genealogy tracking

This script is idempotent - safe to run multiple times.
"""

import logging
from sqlalchemy import text, inspect

logger = logging.getLogger(__name__)


def _get_engine():
    """Get the database engine (lazy import to avoid circular dependencies)."""
    from src.database.engine import engine
    return engine


def upgrade():
    """Add backtest tracking columns to strategy_performance."""
    engine = _get_engine()
    inspector = inspect(engine)

    with engine.connect() as conn:
        # Get existing columns
        existing_columns = [col['name'] for col in inspector.get_columns('strategy_performance')]

        if 'variant' not in existing_columns:
            conn.execute(text(
                "ALTER TABLE strategy_performance ADD COLUMN variant VARCHAR(50)"
            ))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS ix_strategy_performance_variant "
                "ON strategy_performance(variant)"
            ))
            logger.info("Added 'variant' column to strategy_performance")

        if 'symbol' not in existing_columns:
            conn.execute(text(
                "ALTER TABLE strategy_performance ADD COLUMN symbol VARCHAR(20)"
            ))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS ix_strategy_performance_symbol "
                "ON strategy_performance(symbol)"
            ))
            logger.info("Added 'symbol' column to strategy_performance")

        if 'timeframe' not in existing_columns:
            conn.execute(text(
                "ALTER TABLE strategy_performance ADD COLUMN timeframe VARCHAR(10)"
            ))
            logger.info("Added 'timeframe' column to strategy_performance")

        if 'parent_id' not in existing_columns:
            conn.execute(text(
                "ALTER TABLE strategy_performance ADD COLUMN parent_id INTEGER"
            ))
            logger.info("Added 'parent_id' column to strategy_performance")

        conn.commit()

    logger.info("Migration add_backtest_tracking_columns completed")


def downgrade():
    """Remove backtest tracking columns (not supported in SQLite)."""
    logger.warning("Downgrade not supported for SQLite ALTER TABLE ADD COLUMN")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    upgrade()

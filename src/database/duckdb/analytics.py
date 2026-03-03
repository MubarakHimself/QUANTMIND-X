"""
DuckDB Analytics module.

Provides analytics tables initialization for backtesting results, trade journal,
and market data cache.
"""

import logging
from typing import Optional

from src.database.duckdb.connection import DuckDBConnection

# Configure logging
logger = logging.getLogger(__name__)


def initialize_analytics_tables(db_path: Optional[str] = None) -> bool:
    """
    Initialize all DuckDB analytics tables.

    Creates:
    - backtest_results: Store backtest metrics and equity curves
    - trade_journal: Enhanced trade logging with full context
    - market_data_cache: Metadata for cached Parquet files

    Args:
        db_path: Optional custom database path.

    Returns:
        True if initialization succeeded, False otherwise.
    """
    try:
        with DuckDBConnection(db_path=db_path) as conn:
            # Create backtest_results table
            conn.execute_query("""
                CREATE TABLE IF NOT EXISTS backtest_results (
                    id INTEGER PRIMARY KEY,
                    strategy_name VARCHAR NOT NULL,
                    symbol VARCHAR NOT NULL,
                    timeframe VARCHAR NOT NULL,
                    variant VARCHAR NOT NULL,
                    start_date DATE NOT NULL,
                    end_date DATE NOT NULL,
                    initial_cash DOUBLE NOT NULL,
                    final_equity DOUBLE NOT NULL,
                    total_return DOUBLE NOT NULL,
                    sharpe_ratio DOUBLE,
                    sortino_ratio DOUBLE,
                    max_drawdown DOUBLE NOT NULL,
                    max_drawdown_pct DOUBLE NOT NULL,
                    win_rate DOUBLE,
                    profit_factor DOUBLE,
                    total_trades INTEGER,
                    winning_trades INTEGER,
                    losing_trades INTEGER,
                    avg_trade DOUBLE,
                    largest_win DOUBLE,
                    largest_loss DOUBLE,
                    equity_curve JSON,
                    regime_distribution JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes for backtest_results
            conn.execute_query(
                "CREATE INDEX IF NOT EXISTS idx_backtest_strategy ON backtest_results(strategy_name)"
            )
            conn.execute_query(
                "CREATE INDEX IF NOT EXISTS idx_backtest_symbol ON backtest_results(symbol)"
            )
            conn.execute_query(
                "CREATE INDEX IF NOT EXISTS idx_backtest_variant ON backtest_results(variant)"
            )
            conn.execute_query(
                "CREATE INDEX IF NOT EXISTS idx_backtest_created ON backtest_results(created_at DESC)"
            )

            # Create trade_journal table
            conn.execute_query("""
                CREATE TABLE IF NOT EXISTS trade_journal (
                    id INTEGER PRIMARY KEY,
                    backtest_id INTEGER REFERENCES backtest_results(id),
                    timestamp TIMESTAMP NOT NULL,
                    symbol VARCHAR NOT NULL,
                    direction VARCHAR NOT NULL,
                    entry_price DOUBLE NOT NULL,
                    exit_price DOUBLE,
                    stop_loss DOUBLE,
                    take_profit DOUBLE,
                    lot_size DOUBLE NOT NULL,
                    profit DOUBLE,
                    profit_pips DOUBLE,
                    regime VARCHAR,
                    chaos_score DOUBLE,
                    regime_quality DOUBLE,
                    rejection_reason VARCHAR,
                    trade_confidence DOUBLE,
                    hold_time_minutes INTEGER,
                    is_winner BOOLEAN,
                    bot_id VARCHAR,
                    account_id VARCHAR,
                    governor_values JSON,
                    exit_reason VARCHAR,
                    post_trade_analysis JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes for trade_journal
            conn.execute_query(
                "CREATE INDEX IF NOT EXISTS idx_journal_backtest ON trade_journal(backtest_id)"
            )
            conn.execute_query(
                "CREATE INDEX IF NOT EXISTS idx_journal_symbol ON trade_journal(symbol)"
            )
            conn.execute_query(
                "CREATE INDEX IF NOT EXISTS idx_journal_timestamp ON trade_journal(timestamp DESC)"
            )
            conn.execute_query(
                "CREATE INDEX IF NOT EXISTS idx_journal_regime ON trade_journal(regime)"
            )
            conn.execute_query(
                "CREATE INDEX IF NOT EXISTS idx_journal_bot ON trade_journal(bot_id)"
            )

            # Create market_data_cache table
            conn.execute_query("""
                CREATE TABLE IF NOT EXISTS market_data_cache (
                    id INTEGER PRIMARY KEY,
                    symbol VARCHAR NOT NULL,
                    timeframe VARCHAR NOT NULL,
                    file_path VARCHAR NOT NULL,
                    start_date DATE NOT NULL,
                    end_date DATE NOT NULL,
                    record_count INTEGER NOT NULL,
                    data_quality DOUBLE,
                    source VARCHAR,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata JSON
                )
            """)

            # Create indexes for market_data_cache
            conn.execute_query(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_cache_unique ON market_data_cache(symbol, timeframe)"
            )
            conn.execute_query(
                "CREATE INDEX IF NOT EXISTS idx_cache_symbol ON market_data_cache(symbol)"
            )
            conn.execute_query(
                "CREATE INDEX IF NOT EXISTS idx_cache_updated ON market_data_cache(last_updated DESC)"
            )

            logger.info("DuckDB analytics tables initialized successfully")
            return True

    except Exception as e:
        logger.error(f"Failed to initialize DuckDB tables: {e}")
        return False

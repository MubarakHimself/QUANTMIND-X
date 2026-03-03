"""
DuckDB Market Data module.

WARM TIER - Market Data Table (DuckDB) - 30-day retention
"""

import logging
from typing import List, Optional

import pandas as pd

from src.database.duckdb.connection import DuckDBConnection

# Configure logging
logger = logging.getLogger(__name__)

# Default database path
WARM_DB_PATH = "data/market_data.duckdb"


def create_market_data_table(db_path: str = WARM_DB_PATH) -> None:
    """
    Create the WARM tier market_data table.

    Schema:
    - symbol: Trading symbol
    - timeframe: Timeframe (M1, M5, M15, H1, H4, D1)
    - timestamp: Bar timestamp
    - open, high, low, close: OHLC prices
    - volume: Trading volume
    - tick_volume: Tick volume
    - spread: Spread
    - is_synthetic: Whether data was synthesized (gap-filled)
    """
    with DuckDBConnection(db_path) as conn:
        conn.execute_query("""
            CREATE TABLE IF NOT EXISTS market_data (
                symbol VARCHAR(20) NOT NULL,
                timeframe VARCHAR(5) NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                open DOUBLE NOT NULL,
                high DOUBLE NOT NULL,
                low DOUBLE NOT NULL,
                close DOUBLE NOT NULL,
                volume BIGINT,
                tick_volume INTEGER,
                spread INTEGER,
                is_synthetic BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indexes
        conn.execute_query("""
            CREATE INDEX IF NOT EXISTS idx_market_data_lookup
            ON market_data(symbol, timeframe, timestamp)
        """)

        conn.execute_query("""
            CREATE INDEX IF NOT EXISTS idx_market_data_symbol
            ON market_data(symbol)
        """)

        logger.info("Created market_data table in DuckDB")


def insert_market_data(
    data: List[dict],
    db_path: str = WARM_DB_PATH
) -> int:
    """
    Insert market data into WARM tier.

    Args:
        data: List of dictionaries with OHLC data
        db_path: Path to DuckDB file

    Returns:
        Number of rows inserted
    """
    if not data:
        return 0

    with DuckDBConnection(db_path) as conn:
        # Ensure table exists
        if not conn.table_exists('market_data'):
            create_market_data_table(db_path)

        # Prepare insert statement
        query = """
            INSERT INTO market_data
            (symbol, timeframe, timestamp, open, high, low, close, volume, tick_volume, spread, is_synthetic)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        count = 0
        for bar in data:
            try:
                conn.execute_query(query, (
                    bar.get('symbol'),
                    bar.get('timeframe'),
                    bar.get('timestamp'),
                    bar.get('open'),
                    bar.get('high'),
                    bar.get('low'),
                    bar.get('close'),
                    bar.get('volume', 0),
                    bar.get('tick_volume', 0),
                    bar.get('spread', 0),
                    bar.get('is_synthetic', False)
                ))
                count += 1
            except Exception as e:
                logger.error(f"Failed to insert market data: {e}")

        return count


def query_market_data(
    symbol: str,
    timeframe: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db_path: str = WARM_DB_PATH
) -> pd.DataFrame:
    """
    Query market data from WARM tier.

    Args:
        symbol: Trading symbol
        timeframe: Timeframe
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        db_path: Path to DuckDB file

    Returns:
        DataFrame with market data
    """
    with DuckDBConnection(db_path) as conn:
        query = "SELECT * FROM market_data WHERE symbol = ? AND timeframe = ?"
        params = [symbol, timeframe]

        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)

        query += " ORDER BY timestamp ASC"

        result = conn.execute_query(query, tuple(params))
        return result.df()


def cleanup_old_market_data(
    days: int = 30,
    db_path: str = WARM_DB_PATH
) -> int:
    """
    Clean up market data older than specified days.

    Args:
        days: Number of days to retain
        db_path: Path to DuckDB file

    Returns:
        Number of rows deleted
    """
    with DuckDBConnection(db_path) as conn:
        query = f"""
            DELETE FROM market_data
            WHERE timestamp < CURRENT_DATE - INTERVAL '{days} days'
        """
        result = conn.execute_query(query)
        return result.rowcount


def query_historical_data(
    symbol: str,
    timeframe: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    parquet_base_path: str = "data/historical"
) -> pd.DataFrame:
    """
    Query historical market data from Parquet files.

    Args:
        symbol: Trading symbol (e.g., 'EURUSD', 'XAUUSD').
        timeframe: Timeframe (e.g., 'H1', 'D1').
        start_date: Optional start date (YYYY-MM-DD format).
        end_date: Optional end date (YYYY-MM-DD format).
        parquet_base_path: Base path for Parquet files.

    Returns:
        Pandas DataFrame with historical data.

    Example:
        ```python
        df = query_historical_data('EURUSD', 'H1', '2026-01-01', '2026-01-31')
        ```
    """
    import os

    # Build Parquet path
    parquet_path = os.path.join(
        parquet_base_path,
        symbol,
        timeframe,
        "*.parquet"
    )

    # Build query
    query = f"SELECT * FROM '{parquet_path}'"

    # Add date filters if provided
    if start_date or end_date:
        conditions = []
        if start_date:
            conditions.append(f"timestamp >= '{start_date}'")
        if end_date:
            conditions.append(f"timestamp <= '{end_date}'")
        query += " WHERE " + " AND ".join(conditions)

    query += " ORDER BY timestamp ASC"

    # Execute query
    from src.database.duckdb.connection import get_analytics_connection
    with get_analytics_connection() as conn:
        return conn.query_parquet(query)

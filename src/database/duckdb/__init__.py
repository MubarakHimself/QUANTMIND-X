"""
DuckDB Package - Modular DuckDB connection and utilities.

This package provides modular components for DuckDB operations:
- exceptions: Custom exceptions
- connection: Connection management and query execution
- market_data: Market data storage and retrieval (WARM tier)
- analytics: Analytics tables initialization

Usage:
    # Direct imports from submodules
    from src.database.duckdb.connection import DuckDBConnection
    from src.database.duckdb.market_data import query_market_data

    # Or use package-level exports
    from src.database.duckdb import DuckDBConnection, query_market_data
"""

# Re-export public APIs from submodules for convenience
from src.database.duckdb.exceptions import DuckDBConnectionError
from src.database.duckdb.connection import (
    DuckDBConnection,
    get_analytics_connection
)
from src.database.duckdb.market_data import (
    create_market_data_table,
    insert_market_data,
    query_market_data,
    cleanup_old_market_data,
    query_historical_data,
    WARM_DB_PATH
)
from src.database.duckdb.analytics import (
    initialize_analytics_tables
)

__all__ = [
    # Exceptions
    "DuckDBConnectionError",
    # Connection
    "DuckDBConnection",
    "get_analytics_connection",
    # Market Data
    "create_market_data_table",
    "insert_market_data",
    "query_market_data",
    "cleanup_old_market_data",
    "query_historical_data",
    "WARM_DB_PATH",
    # Analytics
    "initialize_analytics_tables",
]

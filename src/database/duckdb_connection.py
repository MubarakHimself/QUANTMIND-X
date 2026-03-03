"""
DuckDB Connection Manager for QuantMindX

Provides DuckDB connection with analytics query support, direct Parquet querying,
connection pooling, retry logic, and backup/restore functionality.

Reference: specs/2026-02-07-quantmindx-trading-system/spec.md
Task Group 2: Hybrid Database Architecture

NOTE: This module is maintained for backward compatibility.
The implementation has been moved to src.database.duckdb package.
Please update imports to use the new modular structure:

    from src.database.duckdb import DuckDBConnection
    from src.database.duckdb.market_data import query_market_data
    from src.database.duckdb.analytics import initialize_analytics_tables
"""

# Re-export all public APIs from the new modular package for backward compatibility
from src.database.duckdb import (
    DuckDBConnection,
    DuckDBConnectionError,
    get_analytics_connection,
    create_market_data_table,
    insert_market_data,
    query_market_data,
    cleanup_old_market_data,
    query_historical_data,
    initialize_analytics_tables,
    WARM_DB_PATH
)

__all__ = [
    "DuckDBConnection",
    "DuckDBConnectionError",
    "get_analytics_connection",
    "create_market_data_table",
    "insert_market_data",
    "query_market_data",
    "cleanup_old_market_data",
    "query_historical_data",
    "initialize_analytics_tables",
    "WARM_DB_PATH",
]

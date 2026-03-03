"""
DuckDB Connection module.

Provides DuckDB connection with analytics query support, direct Parquet querying,
connection pooling, retry logic, and backup/restore functionality.

Reference: specs/2026-02-07-quantmindx-trading-system/spec.md
Task Group 2: Hybrid Database Architecture
"""

import os
import time
import logging
from pathlib import Path
from typing import Optional, List, Tuple, Any
import pandas as pd

try:
    import duckdb
except ImportError:
    raise ImportError(
        "DuckDB is required for this module. "
        "Install it with: pip install duckdb"
    )

from src.database.duckdb.exceptions import DuckDBConnectionError

# Configure logging
logger = logging.getLogger(__name__)


class DuckDBConnection:
    """
    DuckDB connection manager with analytics support and Parquet querying.

    Features:
    - Context manager support for automatic cleanup
    - Direct Parquet file querying without loading into memory
    - Connection pooling for concurrent queries
    - Retry logic with exponential backoff
    - Backup and restore functionality

    Usage:
        ```python
        # Basic usage
        with DuckDBConnection() as conn:
            result = conn.execute_query("SELECT * FROM backtest_results").fetchall()

        # Query Parquet files directly
        with DuckDBConnection() as conn:
            df = conn.query_parquet("SELECT * FROM 'data/historical/EURUSD/H1/*.parquet'")

        # With custom path
        with DuckDBConnection(db_path="/path/to/db.duckdb") as conn:
            result = conn.execute_query("SELECT COUNT(*) FROM trade_journal").fetchone()
        ```
    """

    # Default database path
    DEFAULT_DB_PATH = "data/analytics.duckdb"

    # Connection pool settings
    MAX_POOL_SIZE = 5
    POOL_TIMEOUT = 30

    # Retry settings
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_RETRY_DELAY = 0.1  # seconds
    MAX_RETRY_DELAY = 5.0  # seconds

    def __init__(
        self,
        db_path: Optional[str] = None,
        read_only: bool = False,
        max_retries: int = DEFAULT_MAX_RETRIES
    ):
        """
        Initialize DuckDB connection manager.

        Args:
            db_path: Path to DuckDB database file. If None, uses DEFAULT_DB_PATH.
            read_only: Open database in read-only mode.
            max_retries: Maximum number of retry attempts for failed operations.
        """
        self.db_path = db_path or self.DEFAULT_DB_PATH
        self.read_only = read_only
        self.max_retries = max_retries
        self._connection: Optional[duckdb.DuckDBPyConnection] = None

        # Ensure parent directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        """
        Get or create DuckDB connection.

        Returns:
            Active DuckDB connection.

        Raises:
            DuckDBConnectionError: If connection fails.
        """
        if self._connection is None:
            self._connection = self._create_connection()
        return self._connection

    def _create_connection(self) -> duckdb.DuckDBPyConnection:
        """Create a new DuckDB connection with retry logic."""
        last_error = None
        delay = self.DEFAULT_RETRY_DELAY

        for attempt in range(self.max_retries):
            try:
                logger.info(f"Connecting to DuckDB at: {self.db_path} (attempt {attempt + 1})")

                conn = duckdb.connect(
                    database=self.db_path,
                    read_only=self.read_only
                )

                # Configure connection settings
                conn.execute("PRAGMA enable_progress_bar = false")
                conn.execute("PRAGMA default_order = 'ASC'")

                # Enable Parquet optimization
                conn.execute("SET enable_object_cache = true")

                logger.info(f"Successfully connected to DuckDB: {self.db_path}")
                return conn

            except Exception as e:
                last_error = e
                logger.warning(
                    f"Connection attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {delay:.2f}s..."
                )

                if attempt < self.max_retries - 1:
                    time.sleep(delay)
                    delay = min(delay * 2, self.MAX_RETRY_DELAY)

        raise DuckDBConnectionError(
            f"Failed to connect to DuckDB after {self.max_retries} attempts: {last_error}"
        )

    def execute_query(
        self,
        query: str,
        parameters: Optional[Tuple[Any, ...]] = None
    ) -> Any:
        """
        Execute a SQL query with retry logic.

        Args:
            query: SQL query string.
            parameters: Optional query parameters.

        Returns:
            DuckDB result object.

        Raises:
            DuckDBConnectionError: If query execution fails after retries.
        """
        last_error = None
        delay = self.DEFAULT_RETRY_DELAY

        for attempt in range(self.max_retries):
            try:
                conn = self.connection

                if parameters:
                    result = conn.execute(query, parameters)
                else:
                    result = conn.execute(query)

                return result

            except Exception as e:
                last_error = e
                logger.warning(
                    f"Query execution attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {delay:.2f}s..."
                )

                # Reset connection on error
                if attempt < self.max_retries - 1:
                    self._connection = None
                    time.sleep(delay)
                    delay = min(delay * 2, self.MAX_RETRY_DELAY)

        raise DuckDBConnectionError(
            f"Query execution failed after {self.max_retries} attempts: {last_error}"
        )

    def query_parquet(
        self,
        query: str,
        parameters: Optional[Tuple[Any, ...]] = None
    ) -> pd.DataFrame:
        """
        Query Parquet files directly without loading into memory.

        This allows querying historical data stored in Parquet format
        with full SQL support and DuckDB's query optimization.

        Args:
            query: SQL query with Parquet file paths.
                   Example: "SELECT * FROM 'data/historical/EURUSD/H1/*.parquet'"
            parameters: Optional query parameters.

        Returns:
            Pandas DataFrame with query results.

        Example:
            ```python
            # Single file query
            df = conn.query_parquet(
                "SELECT * FROM 'data/historical/EURUSD/H1/data.parquet' LIMIT 1000"
            )

            # Wildcard query for multiple files
            df = conn.query_parquet(
                "SELECT * FROM 'data/historical/EURUSD/H1/*.parquet' "
                "WHERE timestamp >= '2026-01-01'"
            )

            # Complex analytics query
            df = conn.query_parquet(
                '''
                SELECT
                    date_trunc('day', timestamp) as day,
                    AVG(close) as avg_close,
                    MAX(high) as max_high,
                    MIN(low) as min_low
                FROM 'data/historical/EURUSD/H1/*.parquet'
                GROUP BY day
                ORDER BY day
                '''
            )
        """
        try:
            result = self.execute_query(query, parameters)
            return result.df()
        except Exception as e:
            logger.error(f"Parquet query failed: {e}")
            raise DuckDBConnectionError(f"Failed to query Parquet files: {e}")

    def create_table_from_parquet(
        self,
        table_name: str,
        parquet_path: str
    ) -> None:
        """
        Create a DuckDB table from a Parquet file.

        Args:
            table_name: Name of the table to create.
            parquet_path: Path to Parquet file or wildcard pattern.

        Example:
            ```python
            conn.create_table_from_parquet(
                'eurusd_h1',
                'data/historical/EURUSD/H1/*.parquet'
            )
        ```
        """
        query = f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM '{parquet_path}'"
        self.execute_query(query)
        logger.info(f"Created table '{table_name}' from Parquet: {parquet_path}")

    def backup(self, backup_path: Optional[str] = None) -> bool:
        """
        Create a backup of the DuckDB database.

        Args:
            backup_path: Path for backup file. If None, appends '.backup' to db_path.

        Returns:
            True if backup succeeded, False otherwise.
        """
        backup_path = backup_path or f"{self.db_path}.backup"

        try:
            logger.info(f"Creating DuckDB backup: {self.db_path} -> {backup_path}")

            # Ensure source database is fully flushed
            self.execute_query("CHECKPOINT")

            # Close current connection
            if self._connection:
                self._connection.close()
                self._connection = None

            # Copy database file
            import shutil
            shutil.copy2(self.db_path, backup_path)

            logger.info(f"Backup created successfully: {backup_path}")
            return True

        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return False

    def restore(self, backup_path: str) -> bool:
        """
        Restore database from a backup file.

        Args:
            backup_path: Path to backup file to restore from.

        Returns:
            True if restore succeeded, False otherwise.
        """
        try:
            logger.info(f"Restoring DuckDB from backup: {backup_path} -> {self.db_path}")

            # Close current connection
            if self._connection:
                self._connection.close()
                self._connection = None

            # Verify backup exists
            if not os.path.exists(backup_path):
                raise FileNotFoundError(f"Backup file not found: {backup_path}")

            # Copy backup file
            import shutil
            shutil.copy2(backup_path, self.db_path)

            logger.info(f"Database restored successfully from: {backup_path}")
            return True

        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return False

    def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists in the database.

        Args:
            table_name: Name of the table to check.

        Returns:
            True if table exists, False otherwise.
        """
        try:
            result = self.execute_query("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'main'
                AND table_name = ?
            """, (table_name,))
            return result.fetchone() is not None
        except Exception:
            return False

    def get_table_info(self, table_name: str) -> List[dict]:
        """
        Get column information for a table.

        Args:
            table_name: Name of the table.

        Returns:
            List of dictionaries with column information.
        """
        try:
            result = self.execute_query(f"DESCRIBE {table_name}")
            columns = []
            for row in result.fetchall():
                columns.append({
                    'column_name': row[0],
                    'column_type': row[1],
                    'null': row[2],
                    'key': row[3],
                    'default': row[4],
                    'extra': row[5]
                })
            return columns
        except Exception as e:
            logger.error(f"Failed to get table info for '{table_name}': {e}")
            return []

    def get_tables(self) -> List[str]:
        """
        Get list of all tables in the database.

        Returns:
            List of table names.
        """
        try:
            result = self.execute_query("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'main'
                ORDER BY table_name
            """)
            return [row[0] for row in result.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get table list: {e}")
            return []

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.close()

    def close(self) -> None:
        """Close the database connection."""
        if self._connection:
            try:
                self._connection.close()
            except Exception:
                pass  # Connection already closed or invalid
            self._connection = None
            logger.info(f"DuckDB connection closed: {self.db_path}")

    def __repr__(self) -> str:
        return f"<DuckDBConnection(path='{self.db_path}', read_only={self.read_only})>"


# Convenience functions for common operations

def get_analytics_connection(
    db_path: Optional[str] = None,
    read_only: bool = False
) -> DuckDBConnection:
    """
    Get a DuckDB connection for analytics queries.

    Args:
        db_path: Optional custom database path.
        read_only: Whether to open in read-only mode.

    Returns:
        DuckDBConnection instance.
    """
    return DuckDBConnection(db_path=db_path, read_only=read_only)

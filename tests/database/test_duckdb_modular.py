"""
Tests for modular DuckDB structure.

Verifies that:
1. Modular structure imports correctly
2. Backward compatibility with original duckdb_connection.py
3. All submodules are accessible
"""

import os
import pytest
import tempfile
from pathlib import Path


class TestDuckDBModularStructure:
    """Test the modular DuckDB package structure."""

    def test_exceptions_module_importable(self):
        """Test that exceptions module can be imported."""
        from src.database.duckdb.exceptions import DuckDBConnectionError
        assert DuckDBConnectionError is not None
        assert issubclass(DuckDBConnectionError, Exception)

    def test_connection_module_importable(self):
        """Test that connection module can be imported."""
        from src.database.duckdb.connection import DuckDBConnection
        assert DuckDBConnection is not None

    def test_market_data_module_importable(self):
        """Test that market_data module can be imported."""
        from src.database.duckdb.market_data import (
            create_market_data_table,
            insert_market_data,
            query_market_data,
            cleanup_old_market_data
        )
        assert create_market_data_table is not None
        assert insert_market_data is not None
        assert query_market_data is not None
        assert cleanup_old_market_data is not None

    def test_analytics_module_importable(self):
        """Test that analytics module can be imported."""
        from src.database.duckdb.analytics import initialize_analytics_tables
        assert initialize_analytics_tables is not None

    def test_package_init_exports(self):
        """Test that __init__.py exports all public APIs."""
        from src.database.duckdb import (
            DuckDBConnection,
            DuckDBConnectionError,
            get_analytics_connection,
            create_market_data_table,
            insert_market_data,
            query_market_data,
            cleanup_old_market_data,
            query_historical_data,
            initialize_analytics_tables
        )
        # Verify all expected exports exist
        assert DuckDBConnection is not None
        assert DuckDBConnectionError is not None
        assert get_analytics_connection is not None
        assert create_market_data_table is not None
        assert insert_market_data is not None
        assert query_market_data is not None
        assert cleanup_old_market_data is not None
        assert query_historical_data is not None
        assert initialize_analytics_tables is not None


class TestBackwardCompatibility:
    """Test backward compatibility with original imports."""

    def test_original_module_still_importable(self):
        """Test that original duckdb_connection.py still works."""
        from src.database.duckdb_connection import DuckDBConnection
        assert DuckDBConnection is not None

    def test_original_convenience_function(self):
        """Test that get_analytics_connection still works."""
        from src.database.duckdb_connection import get_analytics_connection
        assert get_analytics_connection is not None

    def test_original_exceptions(self):
        """Test that original exceptions still work."""
        from src.database.duckdb_connection import DuckDBConnectionError
        assert DuckDBConnectionError is not None

    def test_original_market_data_functions(self):
        """Test that original market data functions still work."""
        from src.database.duckdb_connection import (
            create_market_data_table,
            insert_market_data,
            query_market_data,
            cleanup_old_market_data
        )
        assert create_market_data_table is not None
        assert insert_market_data is not None
        assert query_market_data is not None
        assert cleanup_old_market_data is not None

    def test_original_historical_data_function(self):
        """Test that original query_historical_data still works."""
        from src.database.duckdb_connection import query_historical_data
        assert query_historical_data is not None

    def test_original_initialize_function(self):
        """Test that original initialize_analytics_tables still works."""
        from src.database.duckdb_connection import initialize_analytics_tables
        assert initialize_analytics_tables is not None


class TestDuckDBConnection:
    """Test DuckDBConnection functionality."""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield os.path.join(tmpdir, "test.duckdb")

    def test_connection_creation(self, temp_db_path):
        """Test that DuckDBConnection can be created."""
        from src.database.duckdb import DuckDBConnection
        conn = DuckDBConnection(db_path=temp_db_path)
        assert conn.db_path == temp_db_path
        conn.close()

    def test_connection_context_manager(self, temp_db_path):
        """Test DuckDBConnection as context manager."""
        from src.database.duckdb import DuckDBConnection
        with DuckDBConnection(db_path=temp_db_path) as conn:
            assert conn is not None
            result = conn.execute_query("SELECT 1 as test")
            assert result.fetchone()[0] == 1

    def test_table_operations(self, temp_db_path):
        """Test table existence and listing."""
        from src.database.duckdb import DuckDBConnection
        with DuckDBConnection(db_path=temp_db_path) as conn:
            # Create a test table
            conn.execute_query("CREATE TABLE test_table (id INTEGER, name VARCHAR)")

            # Test table_exists
            assert conn.table_exists("test_table") is True
            assert conn.table_exists("nonexistent_table") is False

            # Test get_tables
            tables = conn.get_tables()
            assert "test_table" in tables

    def test_get_table_info(self, temp_db_path):
        """Test get_table_info returns column information."""
        from src.database.duckdb import DuckDBConnection
        with DuckDBConnection(db_path=temp_db_path) as conn:
            conn.execute_query("CREATE TABLE test_info (id INTEGER, name VARCHAR)")
            info = conn.get_table_info("test_info")
            assert len(info) == 2
            column_names = [col['column_name'] for col in info]
            assert 'id' in column_names
            assert 'name' in column_names


class TestMarketDataFunctions:
    """Test market data functions."""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield os.path.join(tmpdir, "market_data.duckdb")

    def test_create_market_data_table(self, temp_db_path):
        """Test market data table creation."""
        from src.database.duckdb.market_data import create_market_data_table
        create_market_data_table(db_path=temp_db_path)

        from src.database.duckdb import DuckDBConnection
        with DuckDBConnection(db_path=temp_db_path) as conn:
            assert conn.table_exists("market_data")
            tables = conn.get_tables()
            assert "market_data" in tables

    def test_insert_and_query_market_data(self, temp_db_path):
        """Test inserting and querying market data."""
        from src.database.duckdb.market_data import (
            create_market_data_table,
            insert_market_data,
            query_market_data
        )

        create_market_data_table(db_path=temp_db_path)

        test_data = [
            {
                'symbol': 'EURUSD',
                'timeframe': 'H1',
                'timestamp': '2026-01-01 10:00:00',
                'open': 1.1000,
                'high': 1.1010,
                'low': 1.0990,
                'close': 1.1005,
                'volume': 1000,
                'tick_volume': 500,
                'spread': 2,
                'is_synthetic': False
            }
        ]

        count = insert_market_data(test_data, db_path=temp_db_path)
        assert count == 1

        df = query_market_data('EURUSD', 'H1', db_path=temp_db_path)
        assert len(df) == 1
        assert df.iloc[0]['symbol'] == 'EURUSD'


class TestAnalyticsFunctions:
    """Test analytics initialization functions."""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield os.path.join(tmpdir, "analytics.duckdb")

    def test_initialize_analytics_tables(self, temp_db_path):
        """Test analytics tables initialization."""
        from src.database.duckdb.analytics import initialize_analytics_tables
        from src.database.duckdb import DuckDBConnection

        result = initialize_analytics_tables(db_path=temp_db_path)
        assert result is True

        with DuckDBConnection(db_path=temp_db_path) as conn:
            tables = conn.get_tables()
            assert "backtest_results" in tables
            assert "trade_journal" in tables
            assert "market_data_cache" in tables

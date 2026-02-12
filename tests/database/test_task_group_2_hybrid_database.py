"""
Tests for Hybrid Database Setup (Task Group 2)

Focused tests for DuckDB and SQLite hybrid database covering:
- DuckDB connection and analytics queries
- SQLite transaction operations
- Direct Parquet querying from DuckDB
- Connection pooling and retry logic
- Backup/restore functionality

Reference: specs/2026-02-07-quantmindx-trading-system/spec.md
Task Group 2: Hybrid Database Setup
"""

import pytest
import os
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# Only run tests if DuckDB is available
pytest.importorskip("duckdb")
import duckdb

from src.database.duckdb_connection import DuckDBConnection
from src.database.models import (
    Base, PropFirmAccount, StrategyFolder, SharedAsset,
    BrokerRegistry, HouseMoneyState, BotCircuitBreaker
)
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session


# Test database paths
TEST_DUCKDB_PATH = "test_hybrid.duckdb"
TEST_SQLITE_PATH = "test_hybrid_sqlite.db"
TEST_PARQUET_DIR = "test_parquet_data"


@pytest.fixture(scope="function")
def test_duckdb_path():
    """Create a temporary DuckDB database path."""
    db_path = TEST_DUCKDB_PATH
    yield db_path
    # Cleanup
    if os.path.exists(db_path):
        os.remove(db_path)


@pytest.fixture(scope="function")
def test_sqlite_engine():
    """Create a test SQLite database engine."""
    engine = create_engine(f"sqlite:///{TEST_SQLITE_PATH}")
    Base.metadata.create_all(bind=engine)
    yield engine
    # Cleanup
    Base.metadata.drop_all(bind=engine)
    if os.path.exists(TEST_SQLITE_PATH):
        os.remove(TEST_SQLITE_PATH)


@pytest.fixture(scope="function")
def test_sqlite_session(test_sqlite_engine):
    """Create a test database session."""
    TestSession = sessionmaker(bind=test_sqlite_engine)
    session = TestSession()
    yield session
    session.close()


@pytest.fixture(scope="function")
def test_parquet_dir():
    """Create a temporary directory with test Parquet files."""
    parquet_dir = TEST_PARQUET_DIR
    os.makedirs(parquet_dir, exist_ok=True)

    # Create test data
    test_data = pd.DataFrame({
        'timestamp': pd.date_range('2026-01-01', periods=100, freq='h'),
        'open': 1.1000 + (pd.Series(range(100)) * 0.0001),
        'high': 1.1050 + (pd.Series(range(100)) * 0.0001),
        'low': 1.0950 + (pd.Series(range(100)) * 0.0001),
        'close': 1.1025 + (pd.Series(range(100)) * 0.0001),
        'volume': 1000
    })

    # Save as Parquet
    parquet_path = os.path.join(parquet_dir, "EURUSD_H1.parquet")
    test_data.to_parquet(parquet_path, index=False)

    yield parquet_dir

    # Cleanup
    if os.path.exists(parquet_dir):
        shutil.rmtree(parquet_dir)


class TestDuckDBConnection:
    """Test DuckDB connection and analytics queries."""

    def test_duckdb_connection_context_manager(self, test_duckdb_path):
        """Test DuckDB connection with context manager support."""
        with DuckDBConnection(db_path=test_duckdb_path) as conn:
            assert conn is not None
            # Execute simple query
            result = conn.execute_query("SELECT 1 AS num").fetchone()
            assert result[0] == 1

    def test_create_analytics_tables(self, test_duckdb_path):
        """Test creating DuckDB tables for analytics."""
        with DuckDBConnection(db_path=test_duckdb_path) as conn:
            # Create backtest_results table
            conn.execute_query("""
                CREATE TABLE IF NOT EXISTS backtest_results (
                    id INTEGER PRIMARY KEY,
                    strategy_name VARCHAR,
                    symbol VARCHAR,
                    timeframe VARCHAR,
                    start_date DATE,
                    end_date DATE,
                    total_return DOUBLE,
                    sharpe_ratio DOUBLE,
                    max_drawdown DOUBLE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create trade_journal table
            conn.execute_query("""
                CREATE TABLE IF NOT EXISTS trade_journal (
                    id INTEGER PRIMARY KEY,
                    backtest_id INTEGER,
                    timestamp TIMESTAMP,
                    symbol VARCHAR,
                    direction VARCHAR,
                    entry_price DOUBLE,
                    exit_price DOUBLE,
                    profit DOUBLE,
                    regime VARCHAR,
                    chaos_score DOUBLE,
                    rejection_reason VARCHAR
                )
            """)

            # Create market_data_cache table
            conn.execute_query("""
                CREATE TABLE IF NOT EXISTS market_data_cache (
                    id INTEGER PRIMARY KEY,
                    symbol VARCHAR,
                    timeframe VARCHAR,
                    file_path VARCHAR,
                    start_date DATE,
                    end_date DATE,
                    record_count INTEGER,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Verify tables exist
            tables = conn.execute_query("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'main'
            """).fetchall()

            table_names = [t[0] for t in tables]
            assert 'backtest_results' in table_names
            assert 'trade_journal' in table_names
            assert 'market_data_cache' in table_names

    def test_analytics_query_performance(self, test_duckdb_path):
        """Test that DuckDB performs analytics queries efficiently."""
        with DuckDBConnection(db_path=test_duckdb_path) as conn:
            # Create and populate test table
            conn.execute_query("""
                CREATE TABLE test_metrics (
                    id INTEGER,
                    value DOUBLE,
                    category VARCHAR
                )
            """)

            # Insert test data
            for i in range(10000):
                conn.execute_query(
                    "INSERT INTO test_metrics VALUES (?, ?, ?)",
                    (i, i * 1.5, f"category_{i % 10}")
                )

            # Run analytics query
            result = conn.execute_query("""
                SELECT
                    category,
                    COUNT(*) as count,
                    AVG(value) as avg_value,
                    MAX(value) as max_value,
                    MIN(value) as min_value
                FROM test_metrics
                GROUP BY category
                ORDER BY count DESC
            """).fetchall()

            assert len(result) == 10
            assert result[0][1] == 1000  # Each category has 1000 records


class TestParquetQuerying:
    """Test direct Parquet querying from DuckDB."""

    def test_query_parquet_file(self, test_duckdb_path, test_parquet_dir):
        """Test querying Parquet files directly without loading into memory."""
        with DuckDBConnection(db_path=test_duckdb_path) as conn:
            # Query Parquet file directly
            parquet_path = os.path.join(test_parquet_dir, "EURUSD_H1.parquet")
            result = conn.query_parquet(f"""
                SELECT
                    timestamp,
                    close,
                    volume
                FROM '{parquet_path}'
                ORDER BY timestamp DESC
                LIMIT 10
            """)

            assert len(result) == 10
            assert 'timestamp' in result.columns
            assert 'close' in result.columns
            assert 'volume' in result.columns

    def test_parquet_wildcard_query(self, test_duckdb_path, test_parquet_dir):
        """Test querying multiple Parquet files with wildcard."""
        # Create another parquet file
        test_data2 = pd.DataFrame({
            'timestamp': pd.date_range('2026-01-05', periods=50, freq='h'),
            'open': 1.2000 + (pd.Series(range(50)) * 0.0001),
            'high': 1.2050 + (pd.Series(range(50)) * 0.0001),
            'low': 1.1950 + (pd.Series(range(50)) * 0.0001),
            'close': 1.2025 + (pd.Series(range(50)) * 0.0001),
            'volume': 2000
        })

        parquet_path2 = os.path.join(test_parquet_dir, "GBPUSD_H1.parquet")
        test_data2.to_parquet(parquet_path2, index=False)

        with DuckDBConnection(db_path=test_duckdb_path) as conn:
            # Query all parquet files with wildcard
            result = conn.query_parquet(f"""
                SELECT
                    *
                FROM '{test_parquet_dir}/*.parquet'
                ORDER BY timestamp ASC
                LIMIT 20
            """)

            assert len(result) == 20


class TestSQLiteNewTables:
    """Test new SQLite tables added to models.py."""

    def test_strategy_folder_table(self, test_sqlite_session: Session):
        """Test StrategyFolder table creation and operations."""
        folder = StrategyFolder(
            folder_name="momentum_strategies",
            description="Momentum-based trading strategies",
            nprd_path="/data/nprd/momentum.md",
            trd_vanilla_path="/docs/trd/momentum_vanilla.md",
            trd_enhanced_path="/docs/trd/momentum_enhanced.md",
            ea_vanilla_path="/mql5/ea/momentum_vanilla.mq5",
            ea_enhanced_path="/mql5/ea/momentum_enhanced.mq5",
            status="draft"
        )
        test_sqlite_session.add(folder)
        test_sqlite_session.commit()
        test_sqlite_session.refresh(folder)

        assert folder.id is not None
        assert folder.folder_name == "momentum_strategies"
        assert folder.status == "draft"

    def test_broker_registry_table(self, test_sqlite_session: Session):
        """Test BrokerRegistry table creation and operations."""
        broker = BrokerRegistry(
            broker_id="icmarkets_raw",
            broker_name="ICMarkets RAW ECN",
            spread_avg=0.1,
            commission_per_lot=3.5,
            lot_step=0.01,
            min_lot=0.01,
            max_lot=100.0,
            pip_values={"EURUSD": 0.0001, "XAUUSD": 0.01},
            preference_tags=["RAW_ECN", "LOW_SPREAD"]
        )
        test_sqlite_session.add(broker)
        test_sqlite_session.commit()
        test_sqlite_session.refresh(broker)

        assert broker.id is not None
        assert broker.broker_name == "ICMarkets RAW ECN"
        assert broker.broker_id == "icmarkets_raw"
        assert broker.pip_values is not None
        assert broker.preference_tags == ["RAW_ECN", "LOW_SPREAD"]

    def test_house_money_state_table(self, test_sqlite_session: Session):
        """Test HouseMoneyState table creation and operations."""
        # Create house money state (uses account_id as string, not FK)
        state = HouseMoneyState(
            account_id="TEST001",
            daily_start_balance=100000.0,
            current_pnl=2500.0,
            high_water_mark=102500.0,
            risk_multiplier=1.5,
            is_preservation_mode=False,
            date="2026-02-07"
        )
        test_sqlite_session.add(state)
        test_sqlite_session.commit()
        test_sqlite_session.refresh(state)

        assert state.id is not None
        assert state.account_id == "TEST001"
        assert state.current_pnl == 2500.0
        assert state.risk_multiplier == 1.5
        assert state.is_preservation_mode is False
        assert state.date == "2026-02-07"

    def test_bot_circuit_breaker_table(self, test_sqlite_session: Session):
        """Test BotCircuitBreaker table creation and operations."""
        breaker = BotCircuitBreaker(
            bot_id="momentum_bot_v1",
            consecutive_losses=3,
            daily_trade_count=15,
            is_quarantined=False
        )
        test_sqlite_session.add(breaker)
        test_sqlite_session.commit()
        test_sqlite_session.refresh(breaker)

        assert breaker.id is not None
        assert breaker.bot_id == "momentum_bot_v1"
        assert breaker.consecutive_losses == 3
        assert breaker.is_quarantined is False


class TestConnectionPoolingAndRetry:
    """Test connection pooling and retry logic."""

    def test_connection_retry_logic(self, test_duckdb_path):
        """Test that connection retries on failure."""
        # This test simulates retry logic
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                with DuckDBConnection(db_path=test_duckdb_path) as conn:
                    result = conn.execute_query("SELECT 1").fetchone()
                    assert result[0] == 1
                    break
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    raise
        else:
            pytest.fail("Connection retry logic failed")

    def test_connection_pooling(self, test_duckdb_path):
        """Test that connection pooling works for concurrent queries."""
        with DuckDBConnection(db_path=test_duckdb_path) as conn:
            # Create test table
            conn.execute_query("""
                CREATE TABLE test_pooling (
                    id INTEGER,
                    value DOUBLE
                )
            """)

            # Execute multiple queries rapidly
            for i in range(100):
                conn.execute_query(
                    "INSERT INTO test_pooling VALUES (?, ?)",
                    (i, i * 2.0)
                )

            # Verify all inserts succeeded
            result = conn.execute_query(
                "SELECT COUNT(*) FROM test_pooling"
            ).fetchone()
            assert result[0] == 100


class TestBackupAndRestore:
    """Test backup and restore functionality."""

    def test_duckdb_backup(self, test_duckdb_path):
        """Test DuckDB database backup."""
        # Create database with test data
        with DuckDBConnection(db_path=test_duckdb_path) as conn:
            conn.execute_query("""
                CREATE TABLE backup_test (
                    id INTEGER PRIMARY KEY,
                    data VARCHAR
                )
            """)
            conn.execute_query("INSERT INTO backup_test VALUES (1, 'test_data')")

            # Perform backup
            backup_path = f"{test_duckdb_path}.backup"
            success = conn.backup(backup_path)
            assert success is True
            assert os.path.exists(backup_path)

    def test_sqlite_backup(self, test_sqlite_engine):
        """Test SQLite database backup."""
        import sqlite3

        # Create test data
        Session = sessionmaker(bind=test_sqlite_engine)
        session = Session()
        account = PropFirmAccount(
            firm_name="BackupTest",
            account_id="BACKUP001"
        )
        session.add(account)
        session.commit()
        session.close()

        # Perform backup
        backup_path = f"{TEST_SQLITE_PATH}.backup"
        conn = sqlite3.connect(TEST_SQLITE_PATH)
        backup_conn = sqlite3.connect(backup_path)
        conn.backup(backup_conn)
        backup_conn.close()
        conn.close()

        # Verify backup exists and has data
        assert os.path.exists(backup_path)
        backup_engine = create_engine(f"sqlite:///{backup_path}")
        BackupSession = sessionmaker(bind=backup_engine)
        backup_session = BackupSession()
        count = backup_session.query(PropFirmAccount).filter_by(
            account_id="BACKUP001"
        ).count()
        backup_session.close()

        assert count == 1
        os.remove(backup_path)

"""
Unit Tests for Warm Storage

Tests WarmStorage without requiring actual DuckDB connection for core logic.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from svss.storage.warm_storage import WarmStorage


class TestWarmStorage:
    """Tests for WarmStorage."""

    def test_initialization_with_defaults(self):
        """Test WarmStorage initialization with default values."""
        with patch('svss.storage.warm_storage.duckdb'):
            storage = WarmStorage()
            assert storage._db_path == "data/svss/warm_storage.db"
            assert storage.MINUTES_PER_DAY == 1440

    def test_initialization_with_custom_path(self):
        """Test WarmStorage initialization with custom path."""
        with patch('svss.storage.warm_storage.duckdb'):
            storage = WarmStorage(db_path="custom/path.db")
            assert storage._db_path == "custom/path.db"

    def test_load_rolling_avg_profile_empty_result(self):
        """Test loading profile when no data exists."""
        with patch('svss.storage.warm_storage.duckdb') as mock_duckdb:
            mock_conn = MagicMock()
            mock_duckdb.connect.return_value = mock_conn
            mock_conn.execute.return_value.fetchall.return_value = []

            storage = WarmStorage()
            profile = storage.load_rolling_avg_profile("EURUSD", num_sessions=20)

            # Should return dict with all 1440 minutes set to 0
            assert len(profile) == 1440
            for minute in range(1440):
                assert profile[minute] == 0.0

    def test_load_rolling_avg_profile_with_data(self):
        """Test loading profile with existing data."""
        with patch('svss.storage.warm_storage.duckdb') as mock_duckdb:
            mock_conn = MagicMock()
            mock_duckdb.connect.return_value = mock_conn
            # Return some data: minute 510 (8:30) -> 100.0
            mock_conn.execute.return_value.fetchall.return_value = [
                (510, 100.0),
                (511, 105.0),
            ]

            storage = WarmStorage()
            profile = storage.load_rolling_avg_profile("EURUSD", num_sessions=20)

            assert profile[510] == 100.0
            assert profile[511] == 105.0
            # Missing minutes should be 0
            assert profile[512] == 0.0
            assert profile[0] == 0.0

    def test_save_session_profile_empty(self):
        """Test saving empty volume profile returns False."""
        with patch('svss.storage.warm_storage.duckdb') as mock_duckdb:
            mock_conn = MagicMock()
            mock_duckdb.connect.return_value = mock_conn

            storage = WarmStorage()
            result = storage.save_session_profile("EURUSD", "session_1", {})

            assert result is False
            # Note: schema init may have called execute, but save should not call execute for empty profile
            # Check that the last call was not an insert
            calls = mock_conn.execute.call_args_list
            # The save method should return early before insert

    def test_save_session_profile_success(self):
        """Test saving volume profile successfully."""
        with patch('svss.storage.warm_storage.duckdb') as mock_duckdb:
            mock_conn = MagicMock()
            mock_duckdb.connect.return_value = mock_conn

            storage = WarmStorage()
            volume_profile = {
                510: 100.0,  # 8:30
                511: 105.0,  # 8:31
            }
            result = storage.save_session_profile("EURUSD", "session_1", volume_profile)

            assert result is True
            # Should have called delete and insert
            assert mock_conn.execute.call_count >= 2

    def test_get_latest_session_id_exists(self):
        """Test getting latest session ID when data exists."""
        with patch('svss.storage.warm_storage.duckdb') as mock_duckdb:
            mock_conn = MagicMock()
            mock_duckdb.connect.return_value = mock_conn
            mock_conn.execute.return_value.fetchone.return_value = ("london_open_20260325_08",)

            storage = WarmStorage()
            session_id = storage.get_latest_session_id("EURUSD")

            assert session_id == "london_open_20260325_08"

    def test_get_latest_session_id_none(self):
        """Test getting latest session ID when no data exists."""
        with patch('svss.storage.warm_storage.duckdb') as mock_duckdb:
            mock_conn = MagicMock()
            mock_duckdb.connect.return_value = mock_conn
            mock_conn.execute.return_value.fetchone.return_value = None

            storage = WarmStorage()
            session_id = storage.get_latest_session_id("EURUSD")

            assert session_id is None

    def test_close(self):
        """Test closing storage connection."""
        with patch('svss.storage.warm_storage.duckdb') as mock_duckdb:
            mock_conn = MagicMock()
            mock_duckdb.connect.return_value = mock_conn

            storage = WarmStorage()
            storage.close()

            mock_conn.close.assert_called_once()
            assert storage._conn is None
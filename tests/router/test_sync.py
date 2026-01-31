"""
Tests for DiskSyncer in Hybrid Disk Synchronization Layer.

Tests atomic file writes, file locking, retry logic, JSON validation,
and Wine path handling for MT5 integration.
Following test-writing standards: focused tests for core user flows only.
"""

import pytest
import json
import tempfile
import os
import time
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from dataclasses import asdict

from src.router.sync import DiskSyncer
from src.router.governor import RiskMandate


class TestDiskSyncerInitialization:
    """Test DiskSyncer class initialization and setup."""

    def test_disksyncer_initialization_with_defaults(self):
        """Test DiskSyncer initializes with default MT5 Wine path."""
        syncer = DiskSyncer()

        # Should have default Wine path
        assert syncer.mt5_path is not None
        assert "MQL5" in syncer.mt5_path
        assert "Files" in syncer.mt5_path

        # Should have retry configuration
        assert syncer.max_retries == 5
        assert syncer.initial_backoff == 1.0

    def test_disksyncer_initialization_with_custom_path(self):
        """Test DiskSyncer initializes with custom MT5 path."""
        custom_path = "/custom/path/to/mt5"
        syncer = DiskSyncer(mt5_path=custom_path)

        assert syncer.mt5_path == custom_path

    def test_disksyncer_custom_retry_config(self):
        """Test DiskSyncer with custom retry configuration."""
        syncer = DiskSyncer(max_retries=3, initial_backoff=0.5)

        assert syncer.max_retries == 3
        assert syncer.initial_backoff == 0.5


class TestAtomicFileWrite:
    """Test atomic file write operations to prevent half-written reads."""

    def test_atomic_write_creates_complete_file(self, tmp_path):
        """Test atomic write creates complete file, never half-written."""
        syncer = DiskSyncer()
        test_file = tmp_path / "test_risk.json"
        test_data = {
            "EURUSD": {"multiplier": 1.0, "timestamp": 1234567890}
        }

        # Perform atomic write
        syncer._atomic_write(str(test_file), test_data)

        # Verify file exists and is complete
        assert test_file.exists()

        # Verify content is complete JSON
        with open(test_file, 'r') as f:
            loaded_data = json.load(f)

        assert loaded_data == test_data

    def test_atomic_write_replaces_existing_file(self, tmp_path):
        """Test atomic write replaces existing file atomically."""
        syncer = DiskSyncer()
        test_file = tmp_path / "test_risk.json"

        # Write initial data
        initial_data = {"EURUSD": {"multiplier": 0.5, "timestamp": 1234567890}}
        syncer._atomic_write(str(test_file), initial_data)

        # Verify initial data
        with open(test_file, 'r') as f:
            assert json.load(f) == initial_data

        # Atomic write with new data
        new_data = {"GBPUSD": {"multiplier": 1.5, "timestamp": 1234567891}}
        syncer._atomic_write(str(test_file), new_data)

        # Verify file has new data (atomically replaced)
        with open(test_file, 'r') as f:
            loaded_data = json.load(f)

        assert loaded_data == new_data
        assert "EURUSD" not in loaded_data  # Old data gone

    def test_atomic_write_uses_temp_file(self, tmp_path):
        """Test atomic write uses temp file then renames."""
        syncer = DiskSyncer()
        test_file = tmp_path / "test_risk.json"
        test_data = {"AUDUSD": {"multiplier": 1.2, "timestamp": 1234567890}}

        # Track temp files created
        temp_files_before = list(tmp_path.glob("*.tmp"))

        # Perform atomic write
        syncer._atomic_write(str(test_file), test_data)

        # Verify target file exists
        assert test_file.exists()

        # Verify no temp files left behind (cleanup successful)
        temp_files_after = list(tmp_path.glob("*.tmp"))
        assert len(temp_files_after) == len(temp_files_before)


class TestFileLocking:
    """Test file locking mechanism for concurrent access prevention."""

    @pytest.mark.skipif(
        os.name != 'posix',
        reason="fcntl locking only available on Unix/Linux"
    )
    def test_file_locking_unix_fcntl(self, tmp_path):
        """Test file locking with fcntl on Unix/Linux systems."""
        syncer = DiskSyncer()
        test_file = tmp_path / "test_lock.json"
        test_data = {"USDJPY": {"multiplier": 0.8, "timestamp": 1234567890}}

        # Write with locking should succeed
        syncer._atomic_write(str(test_file), test_data)

        assert test_file.exists()

    def test_file_locking_handles_lock_failures(self, tmp_path):
        """Test file locking handles lock acquisition failures gracefully."""
        syncer = DiskSyncer()
        test_file = tmp_path / "test_lock_fail.json"

        # Mock locking to simulate failure
        with patch.object(syncer, '_acquire_lock', side_effect=IOError("Lock failed")):
            test_data = {"NZDUSD": {"multiplier": 1.0, "timestamp": 1234567890}}

            # Should raise error or handle gracefully
            with pytest.raises((IOError, OSError)):
                syncer._atomic_write(str(test_file), test_data)


class TestRetryLogic:
    """Test retry logic with exponential backoff for transient failures."""

    def test_retry_on_transient_failure(self, tmp_path):
        """Test retry logic retries on transient file write failures."""
        syncer = DiskSyncer(max_retries=3, initial_backoff=0.01)
        test_file = tmp_path / "test_retry.json"
        test_data = {"CADJPY": {"multiplier": 1.1, "timestamp": 1234567890}}

        attempt_count = [0]

        # Mock write to fail twice then succeed
        def mock_write_with_retry(path, data):
            attempt_count[0] += 1
            if attempt_count[0] < 3:
                raise IOError("Transient write failure")
            # Success on 3rd attempt
            with open(path, 'w') as f:
                json.dump(data, f)

        with patch.object(syncer, '_atomic_write', side_effect=mock_write_with_retry):
            # Should retry and eventually succeed
            syncer.sync_risk_matrix(test_data, str(test_file))

            # Verify retries occurred
            assert attempt_count[0] == 3

    def test_retry_exponential_backoff_timing(self, tmp_path):
        """Test retry uses exponential backoff timing."""
        syncer = DiskSyncer(max_retries=3, initial_backoff=0.05)
        test_file = tmp_path / "test_backoff.json"
        test_data = {"CHFJPY": {"multiplier": 0.9, "timestamp": 1234567890}}

        attempt_times = []

        # Mock write to record attempt times
        def mock_write_with_timing(path, data):
            attempt_times.append(time.time())
            if len(attempt_times) < 3:
                raise IOError("Transient failure")
            with open(path, 'w') as f:
                json.dump(data, f)

        with patch.object(syncer, '_atomic_write', side_effect=mock_write_with_timing):
            syncer.sync_risk_matrix(test_data, str(test_file))

        # Verify exponential backoff (each wait should be longer)
        if len(attempt_times) >= 3:
            first_gap = attempt_times[1] - attempt_times[0]
            second_gap = attempt_times[2] - attempt_times[1]
            # Second gap should be roughly double the first (exponential)
            assert second_gap > first_gap * 1.5  # Allow some timing variance

    def test_retry_max_attempts_respected(self, tmp_path):
        """Test retry respects max_retries limit and gives up."""
        syncer = DiskSyncer(max_retries=2, initial_backoff=0.01)
        test_file = tmp_path / "test_max_retries.json"

        # Mock write to always fail
        def mock_write_always_fails(path, data):
            raise IOError("Persistent failure")

        with patch.object(syncer, '_atomic_write', side_effect=mock_write_always_fails):
            # Should try max_retries times and then raise
            with pytest.raises(IOError):
                syncer.sync_risk_matrix({"XAUUSD": {"multiplier": 1.0, "timestamp": 1234567890}}, str(test_file))


class TestJSONSchemaValidation:
    """Test JSON schema validation before writes to prevent corruption."""

    def test_validates_correct_risk_matrix_structure(self):
        """Test validation accepts correct risk matrix structure."""
        syncer = DiskSyncer()

        # Valid structure matching RiskMandate pattern
        valid_data = {
            "EURUSD": {"multiplier": 1.0, "timestamp": 1234567890},
            "GBPUSD": {"multiplier": 0.8, "timestamp": 1234567891}
        }

        # Should not raise
        assert syncer._validate_risk_matrix(valid_data) is True

    def test_rejects_missing_multiplier_field(self):
        """Test validation rejects missing multiplier field."""
        syncer = DiskSyncer()

        # Invalid structure - missing multiplier
        invalid_data = {
            "EURUSD": {"timestamp": 1234567890}
        }

        with pytest.raises(ValueError, match="multiplier"):
            syncer._validate_risk_matrix(invalid_data)

    def test_rejects_missing_timestamp_field(self):
        """Test validation rejects missing timestamp field."""
        syncer = DiskSyncer()

        # Invalid structure - missing timestamp
        invalid_data = {
            "EURUSD": {"multiplier": 1.0}
        }

        with pytest.raises(ValueError, match="timestamp"):
            syncer._validate_risk_matrix(invalid_data)

    def test_rejects_invalid_multiplier_type(self):
        """Test validation rejects non-numeric multiplier."""
        syncer = DiskSyncer()

        # Invalid structure - multiplier is string
        invalid_data = {
            "EURUSD": {"multiplier": "invalid", "timestamp": 1234567890}
        }

        with pytest.raises(ValueError, match="multiplier"):
            syncer._validate_risk_matrix(invalid_data)

    def test_rejects_empty_risk_matrix(self):
        """Test validation rejects empty risk matrix."""
        syncer = DiskSyncer()

        # Empty data
        invalid_data = {}

        with pytest.raises(ValueError, match="empty"):
            syncer._validate_risk_matrix(invalid_data)


class TestWinePathHandling:
    """Test Wine path detection and handling for MT5 on Linux."""

    def test_detects_default_wine_path(self):
        """Test Wine path detection returns default ~/.wine path."""
        syncer = DiskSyncer()

        # Should detect Wine path with MQL5/Files
        path = syncer._get_wine_mt5_path()

        assert path is not None
        assert ".wine" in path or "Program Files" in path
        assert "MQL5" in path
        assert "Files" in path

    def test_creates_missing_directory(self, tmp_path):
        """Test creates missing directory for Wine MT5 path."""
        custom_path = tmp_path / "MT5" / "MQL5" / "Files"
        syncer = DiskSyncer(mt5_path=str(custom_path))

        test_data = {"EURAUD": {"multiplier": 1.3, "timestamp": 1234567890}}

        # Should create directory if it doesn't exist
        syncer.sync_risk_matrix(test_data)

        # Verify directory was created
        assert custom_path.exists()
        assert (custom_path / "risk_matrix.json").exists()

    def test_handles_custom_mt5_path(self):
        """Test DiskSyncer handles custom MT5 path configuration."""
        custom_path = "/custom/mt5/files"
        syncer = DiskSyncer(mt5_path=custom_path)

        assert syncer.mt5_path == custom_path


class TestDiskSyncerIntegration:
    """Integration tests for DiskSyncer with full workflow."""

    def test_full_sync_workflow(self, tmp_path):
        """Test complete sync workflow: validate -> atomic write -> lock."""
        syncer = DiskSyncer(mt5_path=str(tmp_path))
        test_data = {
            "EURUSD": {"multiplier": 1.0, "timestamp": 1234567890},
            "GBPUSD": {"multiplier": 0.8, "timestamp": 1234567891},
            "USDJPY": {"multiplier": 1.2, "timestamp": 1234567892}
        }

        # Full sync workflow
        syncer.sync_risk_matrix(test_data)

        # Verify file exists at correct location
        expected_file = tmp_path / "risk_matrix.json"
        assert expected_file.exists()

        # Verify content is correct
        with open(expected_file, 'r') as f:
            loaded_data = json.load(f)

        assert loaded_data == test_data

    def test_sync_risk_matrix_from_mandate(self, tmp_path):
        """Test syncing from RiskMandate objects."""
        syncer = DiskSyncer(mt5_path=str(tmp_path))

        # Create RiskMandate objects
        mandates = {
            "EURUSD": RiskMandate(allocation_scalar=1.0, risk_mode="STANDARD"),
            "GBPUSD": RiskMandate(allocation_scalar=0.5, risk_mode="CLAMPED")
        }

        # Convert to risk matrix format
        risk_matrix = {
            symbol: {
                "multiplier": mandate.allocation_scalar,
                "timestamp": int(time.time())
            }
            for symbol, mandate in mandates.items()
        }

        syncer.sync_risk_matrix(risk_matrix)

        # Verify file synced
        expected_file = tmp_path / "risk_matrix.json"
        assert expected_file.exists()

        with open(expected_file, 'r') as f:
            loaded_data = json.load(f)

        assert loaded_data["EURUSD"]["multiplier"] == 1.0
        assert loaded_data["GBPUSD"]["multiplier"] == 0.5

    def test_default_sync_to_wine_path(self):
        """Test default sync writes to Wine MT5 path."""
        syncer = DiskSyncer()

        # This will attempt to write to Wine path
        # We mock the actual write to avoid creating real directories
        test_data = {"EURUSD": {"multiplier": 1.0, "timestamp": 1234567890}}

        with patch.object(syncer, '_atomic_write') as mock_write:
            syncer.sync_risk_matrix(test_data)

            # Verify write was called with Wine path
            mock_write.assert_called_once()
            call_args = mock_write.call_args
            file_path = call_args[0][0]

            # Should contain Wine path components
            assert "risk_matrix.json" in file_path

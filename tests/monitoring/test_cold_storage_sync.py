# tests/monitoring/test_cold_storage_sync.py
"""
Tests for Cold Storage Sync Service

Story 10.3: Notification Configuration API & Cold Storage Sync
Tests for log retention and cold storage sync with integrity verification.

AC4: Nightly log sync with integrity verification (checksums)
"""
import pytest
import os
import tempfile
import hashlib
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock
import sys


class TestChecksumCalculation:
    """Test file checksum calculation."""

    def test_calculate_sha256_checksum(self):
        """Should calculate SHA256 checksum correctly."""
        # Add src to path for import
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from src.monitoring.cold_storage_sync import calculate_file_checksum

        # Create a temp file with known content
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("test content for checksum")
            temp_path = f.name

        try:
            checksum = calculate_file_checksum(temp_path, "sha256")

            # Verify against known hash
            expected = hashlib.sha256(b"test content for checksum").hexdigest()
            assert checksum == expected
        finally:
            os.unlink(temp_path)

    def test_calculate_md5_checksum(self):
        """Should calculate MD5 checksum correctly."""
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from src.monitoring.cold_storage_sync import calculate_file_checksum

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("test content")
            temp_path = f.name

        try:
            checksum = calculate_file_checksum(temp_path, "md5")
            expected = hashlib.md5(b"test content").hexdigest()
            assert checksum == expected
        finally:
            os.unlink(temp_path)


class TestFileIntegrityVerification:
    """Test file integrity verification."""

    def test_verify_file_integrity_success(self):
        """Should verify file with correct checksum."""
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from src.monitoring.cold_storage_sync import verify_file_integrity

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("test content")
            temp_path = f.name

        try:
            # Calculate correct checksum
            correct_checksum = hashlib.sha256(b"test content").hexdigest()

            # Should return True
            result = verify_file_integrity(temp_path, correct_checksum, "sha256")
            assert result is True
        finally:
            os.unlink(temp_path)

    def test_verify_file_integrity_failure(self):
        """Should fail verification with wrong checksum."""
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from src.monitoring.cold_storage_sync import verify_file_integrity

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("test content")
            temp_path = f.name

        try:
            # Wrong checksum
            wrong_checksum = "wrongchecksum123"

            result = verify_file_integrity(temp_path, wrong_checksum, "sha256")
            assert result is False
        finally:
            os.unlink(temp_path)


class TestColdStorageSyncClass:
    """Test ColdStorageSync class."""

    def test_init_default_values(self):
        """Should initialize with default values."""
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from src.monitoring.cold_storage_sync import ColdStorageSync

        sync = ColdStorageSync()

        assert sync.hot_retention_days == 90
        assert sync.checksum_algorithm == "sha256"

    def test_init_custom_values(self):
        """Should accept custom initialization values."""
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from src.monitoring.cold_storage_sync import ColdStorageSync

        sync = ColdStorageSync(
            hot_retention_days=30,
            cold_storage_path="/custom/path",
            checksum_algorithm="md5"
        )

        assert sync.hot_retention_days == 30
        assert sync.cold_storage_path == "/custom/path"
        assert sync.checksum_algorithm == "md5"

    def test_get_logs_to_sync_nonexistent_path(self):
        """Should handle non-existent source path."""
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from src.monitoring.cold_storage_sync import ColdStorageSync

        sync = ColdStorageSync(log_source_path="/nonexistent/path")

        logs = sync.get_logs_to_sync()

        assert logs == []

    def test_get_sync_status(self):
        """Should return sync status information."""
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from src.monitoring.cold_storage_sync import ColdStorageSync

        sync = ColdStorageSync()

        status = sync.get_sync_status()

        assert "logs_pending_sync" in status
        assert "cold_storage_size_bytes" in status
        assert "cold_storage_path" in status
        assert status["hot_retention_days"] == 90
        assert status["checksum_algorithm"] == "sha256"


class TestModuleLevelFunctions:
    """Test module-level convenience functions."""

    @patch('src.monitoring.cold_storage_sync.ColdStorageSync.run_nightly_sync')
    def test_run_cold_storage_sync(self, mock_run):
        """Should call sync via module function."""
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from src.monitoring.cold_storage_sync import run_cold_storage_sync

        mock_run.return_value = {"success": True, "synced_count": 0, "failed_count": 0}

        result = run_cold_storage_sync(hot_retention_days=90)

        assert result["success"] is True

    @patch('src.monitoring.cold_storage_sync.ColdStorageSync.get_sync_status')
    def test_get_cold_storage_status(self, mock_status):
        """Should get status via module function."""
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from src.monitoring.cold_storage_sync import get_cold_storage_status

        mock_status.return_value = {"logs_pending_sync": 0}

        result = get_cold_storage_status()

        assert "logs_pending_sync" in result
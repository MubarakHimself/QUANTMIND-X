"""
Tests for Compliance Files API

Tests the FilesAPI class for storing and retrieving compliance documents.
"""

import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.agents.files.compliance_files import (
    FileStatus,
    FileType,
    FileMetadata,
    FilesAPI,
)


@pytest.fixture
def temp_storage():
    """Create a temporary storage directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def files_api(temp_storage):
    """Create a FilesAPI instance with temporary storage."""
    return FilesAPI(storage_path=temp_storage, max_file_size_mb=10)


class TestFilesAPI:
    """Test suite for FilesAPI."""

    def test_init_creates_directories(self, temp_storage):
        """Test that initialization creates required directories."""
        api = FilesAPI(storage_path=temp_storage)
        for file_type in FileType:
            assert (Path(temp_storage) / file_type.value).exists()

    def test_store_compliance_document(self, files_api):
        """Test storing a compliance document."""
        content = b"Compliance document content"
        metadata = files_api.store_file(
            content=content,
            filename="test_compliance.pdf",
            file_type=FileType.COMPLIANCE_DOCUMENT,
            created_by="test_user",
            tags=["regulatory", "2024"],
        )

        assert metadata is not None
        assert metadata.filename == "test_compliance.pdf"
        assert metadata.file_type == FileType.COMPLIANCE_DOCUMENT
        assert metadata.created_by == "test_user"
        assert metadata.status == FileStatus.ACTIVE
        assert "regulatory" in metadata.tags
        assert "2024" in metadata.tags

    def test_store_trade_log(self, files_api):
        """Test storing a trade log."""
        content = b"Trade: BUY 1.0 BTCUSDT at 50000"
        metadata = files_api.store_file(
            content=content,
            filename="trades_2024_01.csv",
            file_type=FileType.TRADE_LOG,
            created_by="trading_bot",
            tags=["btc", "daily"],
        )

        assert metadata.file_type == FileType.TRADE_LOG
        assert metadata.size_bytes == len(content)

    def test_store_report(self, files_api):
        """Test storing a report."""
        content = b"Monthly performance report data"
        metadata = files_api.store_file(
            content=content,
            filename="monthly_report.xlsx",
            file_type=FileType.REPORT,
            created_by="analyst",
        )

        assert metadata.file_type == FileType.REPORT
        assert metadata.content_hash is not None

    def test_retrieve_file(self, files_api):
        """Test retrieving stored file content."""
        original_content = b"Test file content for retrieval"
        metadata = files_api.store_file(
            content=original_content,
            filename="retrieve_test.txt",
            file_type=FileType.AUDIT_LOG,
            created_by="test",
        )

        retrieved = files_api.retrieve_file(metadata.file_id)
        assert retrieved == original_content

    def test_retrieve_nonexistent_file(self, files_api):
        """Test retrieving a non-existent file returns None."""
        result = files_api.retrieve_file("nonexistent-id")
        assert result is None

    def test_get_metadata(self, files_api):
        """Test getting file metadata."""
        metadata = files_api.store_file(
            content=b"Test content",
            filename="metadata_test.txt",
            file_type=FileType.CONFIG,
            created_by="test",
            tags=["test"],
        )

        retrieved = files_api.get_metadata(metadata.file_id)
        assert retrieved is not None
        assert retrieved.file_id == metadata.file_id
        assert retrieved.filename == "metadata_test.txt"

    def test_search_by_file_type(self, files_api):
        """Test searching files by file type."""
        # Store files of different types
        files_api.store_file(
            content=b"Compliance",
            filename="doc1.txt",
            file_type=FileType.COMPLIANCE_DOCUMENT,
            created_by="test",
        )
        files_api.store_file(
            content=b"Trade",
            filename="trades.csv",
            file_type=FileType.TRADE_LOG,
            created_by="test",
        )
        files_api.store_file(
            content=b"Report",
            filename="report.pdf",
            file_type=FileType.REPORT,
            created_by="test",
        )

        # Search for compliance documents
        results = files_api.search_files(file_type=FileType.COMPLIANCE_DOCUMENT)
        assert len(results) == 1
        assert results[0].file_type == FileType.COMPLIANCE_DOCUMENT

        # Search for trade logs
        results = files_api.search_files(file_type=FileType.TRADE_LOG)
        assert len(results) == 1

    def test_search_by_tags(self, files_api):
        """Test searching files by tags."""
        files_api.store_file(
            content=b"Doc 1",
            filename="doc1.txt",
            file_type=FileType.COMPLIANCE_DOCUMENT,
            created_by="test",
            tags=["important", "q1"],
        )
        files_api.store_file(
            content=b"Doc 2",
            filename="doc2.txt",
            file_type=FileType.COMPLIANCE_DOCUMENT,
            created_by="test",
            tags=["important", "q2"],
        )
        files_api.store_file(
            content=b"Doc 3",
            filename="doc3.txt",
            file_type=FileType.REPORT,
            created_by="test",
            tags=["q3"],
        )

        # Search for important documents
        results = files_api.search_files(tags=["important"])
        assert len(results) == 2

        # Search for q1 documents
        results = files_api.search_files(tags=["q1"])
        assert len(results) == 1
        assert results[0].filename == "doc1.txt"

    def test_search_by_creator(self, files_api):
        """Test searching files by creator."""
        files_api.store_file(
            content=b"User 1 file",
            filename="user1.txt",
            file_type=FileType.TRADE_LOG,
            created_by="user1",
        )
        files_api.store_file(
            content=b"User 2 file",
            filename="user2.txt",
            file_type=FileType.TRADE_LOG,
            created_by="user2",
        )

        results = files_api.search_files(created_by="user1")
        assert len(results) == 1
        assert results[0].created_by == "user1"

    def test_update_file_status(self, files_api):
        """Test updating file status."""
        metadata = files_api.store_file(
            content=b"Test",
            filename="status_test.txt",
            file_type=FileType.COMPLIANCE_DOCUMENT,
            created_by="test",
        )

        # Archive the file
        updated = files_api.update_file_status(metadata.file_id, FileStatus.ARCHIVED)
        assert updated.status == FileStatus.ARCHIVED

        # Verify it's still in search results when filtering
        results = files_api.search_files(status=FileStatus.ARCHIVED)
        assert len(results) == 1

    def test_soft_delete(self, files_api):
        """Test soft deleting a file."""
        metadata = files_api.store_file(
            content=b"Test",
            filename="delete_test.txt",
            file_type=FileType.COMPLIANCE_DOCUMENT,
            created_by="test",
        )

        # Soft delete
        result = files_api.delete_file(metadata.file_id, hard_delete=False)
        assert result is True

        # Verify it's not in active search
        results = files_api.search_files()
        assert len(results) == 0

        # But still exists in deleted status
        results = files_api.search_files(status=FileStatus.DELETED)
        assert len(results) == 1

    def test_hard_delete(self, files_api):
        """Test hard deleting a file."""
        metadata = files_api.store_file(
            content=b"Test",
            filename="hard_delete_test.txt",
            file_type=FileType.COMPLIANCE_DOCUMENT,
            created_by="test",
        )
        file_id = metadata.file_id

        # Hard delete
        result = files_api.delete_file(file_id, hard_delete=True)
        assert result is True

        # Verify it's gone completely
        assert files_api.get_metadata(file_id) is None

    def test_add_tags(self, files_api):
        """Test adding tags to a file."""
        metadata = files_api.store_file(
            content=b"Test",
            filename="tags_test.txt",
            file_type=FileType.REPORT,
            created_by="test",
            tags=["initial"],
        )

        updated = files_api.add_tags(metadata.file_id, ["new_tag", "another"])
        assert "new_tag" in updated.tags
        assert "another" in updated.tags
        assert "initial" in updated.tags

    def test_remove_tags(self, files_api):
        """Test removing tags from a file."""
        metadata = files_api.store_file(
            content=b"Test",
            filename="remove_tags_test.txt",
            file_type=FileType.REPORT,
            created_by="test",
            tags=["tag1", "tag2", "tag3"],
        )

        updated = files_api.remove_tags(metadata.file_id, ["tag2"])
        assert "tag1" in updated.tags
        assert "tag2" not in updated.tags
        assert "tag3" in updated.tags

    def test_file_size_limit(self, temp_storage):
        """Test that file size limit is enforced."""
        api = FilesAPI(storage_path=temp_storage, max_file_size_mb=1)

        # 1 MB limit
        large_content = b"x" * (2 * 1024 * 1024)  # 2 MB

        with pytest.raises(ValueError, match="exceeds maximum"):
            api.store_file(
                content=large_content,
                filename="large.txt",
                file_type=FileType.COMPLIANCE_DOCUMENT,
                created_by="test",
            )

    def test_list_file_types(self, files_api):
        """Test listing available file types."""
        types = files_api.list_file_types()
        assert FileType.COMPLIANCE_DOCUMENT.value in types
        assert FileType.TRADE_LOG.value in types
        assert FileType.REPORT.value in types

    def test_get_stats(self, files_api):
        """Test getting file statistics."""
        # Store some files
        files_api.store_file(
            content=b"Doc 1",
            filename="doc1.txt",
            file_type=FileType.COMPLIANCE_DOCUMENT,
            created_by="test",
        )
        files_api.store_file(
            content=b"Trade 1",
            filename="trade1.csv",
            file_type=FileType.TRADE_LOG,
            created_by="test",
        )

        stats = files_api.get_stats()
        assert stats["total_files"] == 2
        assert stats["active_files"] == 2
        assert stats["by_type"][FileType.COMPLIANCE_DOCUMENT.value]["count"] == 1
        assert stats["by_type"][FileType.TRADE_LOG.value]["count"] == 1

    def test_content_hash_integrity(self, files_api):
        """Test that content hashing works for integrity verification."""
        content = b"Integrity test content"
        metadata = files_api.store_file(
            content=content,
            filename="integrity.txt",
            file_type=FileType.AUDIT_LOG,
            created_by="test",
        )

        # Retrieve and verify hash
        retrieved = files_api.retrieve_file(metadata.file_id)
        import hashlib
        expected_hash = hashlib.sha256(retrieved).hexdigest()
        assert expected_hash == metadata.content_hash


class TestFileMetadata:
    """Test suite for FileMetadata dataclass."""

    def test_to_dict(self):
        """Test FileMetadata serialization to dict."""
        metadata = FileMetadata(
            file_id="test-id",
            filename="test.txt",
            file_type=FileType.COMPLIANCE_DOCUMENT,
            content_hash="abc123",
            size_bytes=100,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            updated_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            created_by="test_user",
            tags=["tag1", "tag2"],
            status=FileStatus.ACTIVE,
            metadata={"key": "value"},
        )

        result = metadata.to_dict()
        assert result["file_id"] == "test-id"
        assert result["filename"] == "test.txt"
        assert result["file_type"] == "compliance_document"
        assert result["content_hash"] == "abc123"
        assert result["tags"] == ["tag1", "tag2"]
        assert result["status"] == "active"

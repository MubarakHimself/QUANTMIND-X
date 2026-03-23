"""
P3 Tests: Shared Assets Rare Cases

Epic 6 - Knowledge & Research Engine
Priority: P3 (Low)
Coverage: Rare edge cases for shared assets

Covers:
- Concurrent upload conflicts
- Storage quota handling
- Corrupted metadata recovery
"""

import json
import os
import shutil
import tempfile
import threading
from pathlib import Path

import pytest

from src.agents.tools.shared_assets_tool import (
    AssetCategory,
    SharedAssetsTool,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_storage():
    """Create a temporary storage directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_file():
    """Create a temporary test file."""
    fd, path = tempfile.mkstemp(suffix=".txt", prefix="test_")
    os.write(fd, b"Test content for shared assets")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def tools(temp_storage):
    """Create SharedAssetsTool instance with temp storage."""
    return SharedAssetsTool(storage_path=temp_storage)


# =============================================================================
# P3 Tests: Shared Assets Rare Cases
# =============================================================================

class TestSharedAssetsRareCases:
    """P3 rare scenario tests for shared assets."""

    def test_concurrent_uploads_handled_safely(self, tools, temp_file):
        """
        P3: Simultaneous uploads of same file should not corrupt metadata.

        Thread safety test for metadata writes.
        """
        results = []

        def upload_task():
            result = tools.upload(
                file_path=temp_file,
                name="concurrent_test",
                category=AssetCategory.SCREENSHOT,
            )
            results.append(result)

        threads = [threading.Thread(target=upload_task) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should succeed without corruption
        assert len(results) == 3
        assert all(r["success"] for r in results)

        # Metadata should still be readable
        list_result = tools.list()
        assert list_result["success"] is True
        assert list_result["count"] == 3

    def test_storage_quota_exceeded_returns_error(self, tools, temp_file, monkeypatch):
        """
        P3: Storage quota exceeded should return appropriate error.

        Users should see clear message when quota full.
        """
        # Set limit to 10 bytes - temp_file is ~30 bytes so this will exceed quota
        monkeypatch.setattr(tools, "MAX_STORAGE_BYTES", 10)

        result = tools.upload(
            file_path=temp_file,
            name="quota_test",
            category=AssetCategory.PROP_FIRM_REPORT,
        )

        assert result["success"] is False
        assert "quota" in result["error"].lower() or \
               "storage" in result["error"].lower() or \
               "size" in result["error"].lower()

    def test_corrupted_metadata_file_recovery(self, tools, temp_storage):
        """
        P3: Corrupted metadata JSON should be recovered gracefully.

        Should not crash, should rebuild metadata.
        """
        metadata_file = Path(temp_storage) / "metadata.json"
        metadata_file.write_text("not valid json{{{")  # Corrupt

        # Should not crash, should create new metadata
        result = tools.list()
        assert result["success"] is True
        # Should still work with empty or rebuilt metadata

    def test_upload_with_unicode_filename(self, tools, temp_file):
        """
        P3: Files with unicode characters in name should be handled.

        International users may have unicode filenames.
        """
        result = tools.upload(
            file_path=temp_file,
            name="\u0422\u0435\u0441\u0442 Report",  # "Test" in Cyrillic
            category=AssetCategory.PROP_FIRM_REPORT,
        )

        assert result["success"] is True
        assert "\u0422\u0435\u0441\u0442" in result["asset"]["name"]

    def test_delete_nonexistent_asset_returns_false(self, tools):
        """
        P3: Deleting non-existent asset should return success=False.

        Idempotent delete - should not raise error.
        """
        result = tools.delete("definitely-does-not-exist-12345")
        assert result["success"] is False
        assert "not found" in result["error"].lower()


# =============================================================================
# Test Summary
# =============================================================================

"""
P3 Shared Assets Test Coverage:
| Test | Scenario | Notes |
|------|----------|-------|
| test_concurrent_uploads_handled_safely | Thread safety | Rare race condition |
| test_storage_quota_exceeded_returns_error | Quota | Storage limit |
| test_corrupted_metadata_file_recovery | Recovery | Corruption handling |
| test_upload_with_unicode_filename | Unicode | International |
| test_delete_nonexistent_asset_returns_false | Idempotent | API design |

Total: 5 P3 tests added
"""

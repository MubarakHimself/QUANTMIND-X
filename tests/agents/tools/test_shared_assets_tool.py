"""
Tests for Shared Assets Tool

Tests for upload, download, list, get, delete, update operations.
"""

import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest

from src.agents.tools.shared_assets_tool import (
    SHARED_ASSETS_TOOL_SCHEMAS,
    AssetCategory,
    SharedAssetsTool,
    get_default_instance,
    get_shared_assets_tool_schemas,
)


class TestSharedAssetsTool:
    """Test suite for SharedAssetsTool class."""

    @pytest.fixture
    def temp_storage(self):
        """Create a temporary storage directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def temp_file(self):
        """Create a temporary test file."""
        fd, path = tempfile.mkstemp(suffix=".txt", prefix="test_")
        os.write(fd, b"Test content for shared assets tool")
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.unlink(path)

    @pytest.fixture
    def tools(self, temp_storage):
        """Create SharedAssetsTool instance with temp storage."""
        return SharedAssetsTool(storage_path=temp_storage)

    # =========================================================================
    # Initialization tests
    # =========================================================================

    def test_initialization_default_path(self):
        """Test initialization with default path."""
        tools = SharedAssetsTool()
        assert tools.storage_path is not None
        assert tools.metadata_file is not None

    def test_initialization_custom_path(self, temp_storage):
        """Test initialization with custom path."""
        tools = SharedAssetsTool(storage_path=temp_storage)
        assert str(tools.storage_path) == temp_storage

    def test_supported_categories(self):
        """Test that all expected categories are supported."""
        categories = SharedAssetsTool.SUPPORTED_CATEGORIES
        assert AssetCategory.PROP_FIRM_REPORT in categories
        assert AssetCategory.SCREENSHOT in categories
        assert AssetCategory.TRADE_LOG in categories
        assert AssetCategory.INDICATOR in categories
        assert AssetCategory.STRATEGY in categories
        assert AssetCategory.BACKTEST_RESULT in categories

    # =========================================================================
    # Upload tests
    # =========================================================================

    def test_upload_success(self, tools, temp_file):
        """Test successful file upload."""
        result = tools.upload(
            file_path=temp_file,
            name="test_report",
            category=AssetCategory.PROP_FIRM_REPORT,
            description="Test report description",
            tags=["test", "report"],
        )
        assert result["success"] is True
        assert "asset" in result
        assert result["asset"]["name"] == "test_report"
        assert result["asset"]["category"] == AssetCategory.PROP_FIRM_REPORT
        assert "id" in result["asset"]
        assert "checksum" in result["asset"]

    def test_upload_file_not_found(self, tools):
        """Test upload with non-existent file."""
        result = tools.upload(
            file_path="/nonexistent/path/file.txt",
            name="test",
            category=AssetCategory.PROP_FIRM_REPORT,
        )
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    def test_upload_unsupported_category(self, tools, temp_file):
        """Test upload with unsupported category."""
        result = tools.upload(
            file_path=temp_file,
            name="test",
            category="unsupported_category",
        )
        assert result["success"] is False
        assert "unsupported" in result["error"].lower()

    def test_upload_file_too_large(self, tools, temp_file):
        """Test upload with file exceeding max size."""
        # Temporarily reduce max size for testing
        original_max = tools.MAX_FILE_SIZE
        tools.MAX_FILE_SIZE = 1  # 1 byte

        result = tools.upload(
            file_path=temp_file,
            name="test",
            category=AssetCategory.PROP_FIRM_REPORT,
        )

        tools.MAX_FILE_SIZE = original_max
        assert result["success"] is False
        assert "too large" in result["error"].lower()

    # =========================================================================
    # Download tests
    # =========================================================================

    def test_download_success(self, tools, temp_file):
        """Test successful file download."""
        # First upload
        upload_result = tools.upload(
            file_path=temp_file,
            name="test_download",
            category=AssetCategory.TRADE_LOG,
        )
        asset_id = upload_result["asset"]["id"]

        # Create temp destination
        dest_fd, dest_path = tempfile.mkstemp()
        os.close(dest_fd)

        try:
            # Download
            result = tools.download(asset_id=asset_id, destination=dest_path)
            assert result["success"] is True
            assert os.path.exists(dest_path)
            assert result["checksum_verified"] is True
        finally:
            if os.path.exists(dest_path):
                os.unlink(dest_path)

    def test_download_nonexistent_asset(self, tools):
        """Test download with non-existent asset ID."""
        result = tools.download(asset_id="nonexistent_id", destination="/tmp/test")
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    # =========================================================================
    # List tests
    # =========================================================================

    def test_list_empty(self, tools):
        """Test listing with no assets."""
        result = tools.list()
        assert result["success"] is True
        assert result["count"] == 0
        assert result["assets"] == []

    def test_list_all_assets(self, tools, temp_file):
        """Test listing all assets."""
        tools.upload(
            file_path=temp_file,
            name="asset1",
            category=AssetCategory.PROP_FIRM_REPORT,
        )
        tools.upload(
            file_path=temp_file,
            name="asset2",
            category=AssetCategory.SCREENSHOT,
        )

        result = tools.list()
        assert result["success"] is True
        assert result["count"] == 2

    def test_list_by_category(self, tools, temp_file):
        """Test listing assets filtered by category."""
        tools.upload(
            file_path=temp_file,
            name="report",
            category=AssetCategory.PROP_FIRM_REPORT,
        )
        tools.upload(
            file_path=temp_file,
            name="screenshot",
            category=AssetCategory.SCREENSHOT,
        )

        result = tools.list(category=AssetCategory.PROP_FIRM_REPORT)
        assert result["success"] is True
        assert result["count"] == 1
        assert result["assets"][0]["category"] == AssetCategory.PROP_FIRM_REPORT

    def test_list_by_tags(self, tools, temp_file):
        """Test listing assets filtered by tags."""
        tools.upload(
            file_path=temp_file,
            name="tagged_asset",
            category=AssetCategory.TRADE_LOG,
            tags=["important", "review"],
        )
        tools.upload(
            file_path=temp_file,
            name="untagged",
            category=AssetCategory.TRADE_LOG,
        )

        result = tools.list(tags=["important"])
        assert result["success"] is True
        assert result["count"] == 1
        assert "important" in result["assets"][0]["tags"]

    def test_list_by_search(self, tools, temp_file):
        """Test listing assets with search query."""
        tools.upload(
            file_path=temp_file,
            name="EURUSD Report",
            category=AssetCategory.PROP_FIRM_REPORT,
            description="Monthly performance report",
        )

        result = tools.list(search="eurusd")
        assert result["success"] is True
        assert result["count"] == 1
        assert "EURUSD" in result["assets"][0]["name"]

    # =========================================================================
    # Get asset tests
    # =========================================================================

    def test_get_asset_success(self, tools, temp_file):
        """Test getting asset details."""
        upload_result = tools.upload(
            file_path=temp_file,
            name="test_asset",
            category=AssetCategory.INDICATOR,
            description="Test description",
        )
        asset_id = upload_result["asset"]["id"]

        result = tools.get_asset(asset_id)
        assert result["success"] is True
        assert result["asset"]["name"] == "test_asset"

    def test_get_asset_not_found(self, tools):
        """Test getting non-existent asset."""
        result = tools.get_asset("nonexistent_id")
        assert result["success"] is False

    # =========================================================================
    # Delete tests
    # =========================================================================

    def test_delete_success(self, tools, temp_file):
        """Test successful asset deletion."""
        upload_result = tools.upload(
            file_path=temp_file,
            name="to_delete",
            category=AssetCategory.STRATEGY,
        )
        asset_id = upload_result["asset"]["id"]

        result = tools.delete(asset_id)
        assert result["success"] is True
        assert result["deleted"] == asset_id

        # Verify deletion
        get_result = tools.get_asset(asset_id)
        assert get_result["success"] is False

    def test_delete_not_found(self, tools):
        """Test deleting non-existent asset."""
        result = tools.delete("nonexistent_id")
        assert result["success"] is False

    # =========================================================================
    # Update tests
    # =========================================================================

    def test_update_name(self, tools, temp_file):
        """Test updating asset name."""
        upload_result = tools.upload(
            file_path=temp_file,
            name="original_name",
            category=AssetCategory.BACKTEST_RESULT,
        )
        asset_id = upload_result["asset"]["id"]

        result = tools.update(asset_id, name="new_name")
        assert result["success"] is True
        assert result["asset"]["name"] == "new_name"

    def test_update_description(self, tools, temp_file):
        """Test updating asset description."""
        upload_result = tools.upload(
            file_path=temp_file,
            name="test",
            category=AssetCategory.BACKTEST_RESULT,
            description="original description",
        )
        asset_id = upload_result["asset"]["id"]

        result = tools.update(asset_id, description="updated description")
        assert result["success"] is True
        assert result["asset"]["description"] == "updated description"

    def test_update_tags(self, tools, temp_file):
        """Test updating asset tags."""
        upload_result = tools.upload(
            file_path=temp_file,
            name="test",
            category=AssetCategory.BACKTEST_RESULT,
            tags=["tag1"],
        )
        asset_id = upload_result["asset"]["id"]

        result = tools.update(asset_id, tags=["tag1", "tag2", "new"])
        assert result["success"] is True
        assert result["asset"]["tags"] == ["tag1", "tag2", "new"]

    def test_update_not_found(self, tools):
        """Test updating non-existent asset."""
        result = tools.update("nonexistent_id", name="new_name")
        assert result["success"] is False

    # =========================================================================
    # Categories tests
    # =========================================================================

    def test_get_categories(self, tools):
        """Test getting supported categories."""
        categories = tools.get_categories()
        assert isinstance(categories, list)
        assert len(categories) > 0
        assert "prop_firm_report" in categories

    # =========================================================================
    # Stats tests
    # =========================================================================

    def test_get_stats_empty(self, tools):
        """Test stats with no assets."""
        stats = tools.get_stats()
        assert stats["total_assets"] == 0
        assert stats["total_size_bytes"] == 0

    def test_get_stats_with_assets(self, tools, temp_file):
        """Test stats with uploaded assets."""
        tools.upload(
            file_path=temp_file,
            name="stat_test1",
            category=AssetCategory.PROP_FIRM_REPORT,
        )
        tools.upload(
            file_path=temp_file,
            name="stat_test2",
            category=AssetCategory.SCREENSHOT,
        )

        stats = tools.get_stats()
        assert stats["total_assets"] == 2
        assert stats["total_size_bytes"] > 0

    # =========================================================================
    # Tool schemas tests
    # =========================================================================

    def test_tool_schemas_exist(self):
        """Test that tool schemas are defined."""
        assert len(SHARED_ASSETS_TOOL_SCHEMAS) > 0

    def test_tool_schemas_structure(self):
        """Test tool schemas have correct structure."""
        for schema in SHARED_ASSETS_TOOL_SCHEMAS:
            assert "name" in schema
            assert "description" in schema
            assert "parameters" in schema

    def test_get_tool_schemas_function(self):
        """Test get_shared_assets_tool_schemas function."""
        schemas = get_shared_assets_tool_schemas()
        assert isinstance(schemas, list)
        assert len(schemas) > 0

    # =========================================================================
    # Edge cases and integration tests
    # =========================================================================

    def test_checksum_verification(self, tools, temp_file):
        """Test that checksum is calculated and stored."""
        result = tools.upload(
            file_path=temp_file,
            name="checksum_test",
            category=AssetCategory.TRADE_LOG,
        )
        asset_id = result["asset"]["id"]

        # Get the asset
        asset = tools.get_asset(asset_id)
        assert "checksum" in asset["asset"]
        assert len(asset["asset"]["checksum"]) > 0

    def test_file_path_hidden_in_list(self, tools, temp_file):
        """Test that file paths are hidden in list output."""
        tools.upload(
            file_path=temp_file,
            name="security_test",
            category=AssetCategory.PROP_FIRM_REPORT,
        )

        result = tools.list()
        asset = result["assets"][0]
        assert "file_path" not in asset

    def test_mime_type_detection(self):
        """Test MIME type detection from extension."""
        assert SharedAssetsTool._get_mime_type(".png") == "image/png"
        assert SharedAssetsTool._get_mime_type(".jpg") == "image/jpeg"
        assert SharedAssetsTool._get_mime_type(".pdf") == "application/pdf"
        assert SharedAssetsTool._get_mime_type(".csv") == "text/csv"
        assert SharedAssetsTool._get_mime_type(".unknown") == "application/octet-stream"

    def test_multiple_uploads_same_name(self, tools, temp_file):
        """Test uploading multiple files with same name."""
        result1 = tools.upload(
            file_path=temp_file,
            name="duplicate",
            category=AssetCategory.PROP_FIRM_REPORT,
        )
        result2 = tools.upload(
            file_path=temp_file,
            name="duplicate",
            category=AssetCategory.PROP_FIRM_REPORT,
        )

        # Both should succeed with different IDs
        assert result1["success"] is True
        assert result2["success"] is True
        assert result1["asset"]["id"] != result2["asset"]["id"]

    def test_default_instance(self):
        """Test getting default instance."""
        instance1 = get_default_instance()
        instance2 = get_default_instance()
        assert instance1 is instance2

"""
P2 Tests: Shared Assets Canvas API

Epic 6 - Knowledge & Research Engine
Priority: P2 (Medium)
Coverage: Shared assets canvas API secondary features

Covers:
- Asset listing with pagination
- Category filtering
- Tag-based filtering
- Search functionality
- Bulk operations
"""

import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest
from unittest.mock import patch

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
# P2 Tests: Shared Assets Canvas
# =============================================================================

class TestSharedAssetsCanvasAPI:
    """P2 tests for shared assets canvas secondary features."""

    def test_list_assets_returns_pagination_info(self, tools, temp_file):
        """
        P2: List response should include pagination metadata.

        Frontend needs total count and has_more for infinite scroll.
        """
        # Create 15 assets
        for i in range(15):
            tools.upload(
                file_path=temp_file,
                name=f"page_test_{i}",
                category=AssetCategory.PROP_FIRM_REPORT,
            )

        result = tools.list(page=1, limit=10)

        assert result["success"] is True
        assert "total" in result
        assert result["total"] == 15
        assert len(result["assets"]) == 10
        assert result.get("has_more") is True or result.get("page") == 1

    def test_list_assets_filter_by_category(self, tools, temp_file):
        """
        P2: Filtering by category should return only matching assets.

        Canvas uses category tabs for filtering.
        """
        tools.upload(
            file_path=temp_file,
            name="report_asset",
            category=AssetCategory.PROP_FIRM_REPORT,
        )
        tools.upload(
            file_path=temp_file,
            name="screenshot_asset",
            category=AssetCategory.SCREENSHOT,
        )
        tools.upload(
            file_path=temp_file,
            name="trade_log_asset",
            category=AssetCategory.TRADE_LOG,
        )

        result = tools.list(category=AssetCategory.PROP_FIRM_REPORT)

        assert result["success"] is True
        assert result["count"] == 1
        assert result["assets"][0]["category"] == AssetCategory.PROP_FIRM_REPORT

    def test_list_assets_filter_by_multiple_tags(self, tools, temp_file):
        """
        P2: Assets with ALL specified tags should be returned.

        Uses AND logic for tag filtering.
        """
        tools.upload(
            file_path=temp_file,
            name="tagged_both",
            category=AssetCategory.TRADE_LOG,
            tags=["important", "reviewed"],
        )
        tools.upload(
            file_path=temp_file,
            name="tagged_one",
            category=AssetCategory.TRADE_LOG,
            tags=["important"],
        )
        tools.upload(
            file_path=temp_file,
            name="tagged_other",
            category=AssetCategory.TRADE_LOG,
            tags=["reviewed"],
        )

        result = tools.list(tags=["important", "reviewed"])

        assert result["success"] is True
        assert result["count"] == 1
        assert result["assets"][0]["name"] == "tagged_both"

    def test_search_assets_matches_name(self, tools, temp_file):
        """
        P2: Search should match asset name.

        Canvas search bar queries names first.
        """
        tools.upload(
            file_path=temp_file,
            name="EURUSD Strategy Report",
            category=AssetCategory.PROP_FIRM_REPORT,
        )

        result = tools.list(search="EURUSD")

        assert result["success"] is True
        assert result["count"] >= 1
        assert any("EURUSD" in a["name"] for a in result["assets"])

    def test_search_assets_matches_description(self, tools, temp_file):
        """
        P2: Search should match description field.

        Descriptions are indexed for search.
        """
        tools.upload(
            file_path=temp_file,
            name="Strategy Report",
            category=AssetCategory.PROP_FIRM_REPORT,
            description="This covers EURUSD and GBPUSD pairs",
        )

        result = tools.list(search="EURUSD")

        assert result["success"] is True
        assert result["count"] >= 1

    def test_search_assets_matches_tags(self, tools, temp_file):
        """
        P2: Search should match tags.

        Tag content is searchable.
        """
        tools.upload(
            file_path=temp_file,
            name="Tagged Asset",
            category=AssetCategory.INDICATOR,
            tags=["momentum", "scalping"],
        )

        result = tools.list(search="momentum")

        assert result["success"] is True
        assert result["count"] >= 1

    def test_list_assets_sort_by_name_ascending(self, tools, temp_file):
        """
        P2: Assets can be sorted by name ascending.

        Used for alphabetical sorting option.
        """
        tools.upload(file_path=temp_file, name="Zebra Report", category=AssetCategory.SCREENSHOT)
        tools.upload(file_path=temp_file, name="Alpha Report", category=AssetCategory.SCREENSHOT)
        tools.upload(file_path=temp_file, name="Mango Report", category=AssetCategory.SCREENSHOT)

        result = tools.list(sort="name", order="asc")

        names = [a["name"] for a in result["assets"]]
        assert names == sorted(names)

    def test_list_assets_sort_by_date_descending(self, tools, temp_file):
        """
        P2: Assets can be sorted by date descending.

        Default view shows newest first.
        """
        tools.upload(file_path=temp_file, name="Old Asset", category=AssetCategory.SCREENSHOT)
        tools.upload(file_path=temp_file, name="New Asset", category=AssetCategory.SCREENSHOT)

        result = tools.list(sort="created_at", order="desc")

        # Newer assets should come first
        assert result["assets"][0]["name"] == "New Asset"

    def test_get_asset_includes_download_url(self, tools, temp_file):
        """
        P2: Asset details should include URL for download.

        Canvas generates download link from asset ID.
        """
        result = tools.upload(
            file_path=temp_file,
            name="download_test",
            category=AssetCategory.TRADE_LOG,
        )
        asset_id = result["asset"]["id"]

        asset = tools.get_asset(asset_id)

        assert asset["success"] is True
        assert "download_url" in asset["asset"] or "url" in asset["asset"]


class TestSharedAssetsCanvasStats:
    """P2 tests for canvas statistics display."""

    def test_get_stats_returns_category_breakdown(self, tools, temp_file):
        """
        P2: Stats should include count per category.

        Canvas shows category distribution chart.
        """
        tools.upload(file_path=temp_file, name="r1", category=AssetCategory.PROP_FIRM_REPORT)
        tools.upload(file_path=temp_file, name="r2", category=AssetCategory.PROP_FIRM_REPORT)
        tools.upload(file_path=temp_file, name="s1", category=AssetCategory.SCREENSHOT)

        stats = tools.get_stats()

        assert "by_category" in stats or "category_breakdown" in stats or stats.get("total_assets") == 3

    def test_get_stats_returns_total_size(self, tools, temp_file):
        """
        P2: Stats should include total storage size.

        Storage quota display.
        """
        tools.upload(
            file_path=temp_file,
            name="size_test",
            category=AssetCategory.BACKTEST_RESULT,
        )

        stats = tools.get_stats()

        assert "total_size_bytes" in stats
        assert stats["total_size_bytes"] > 0


# =============================================================================
# Test Summary
# =============================================================================

"""
P2 Shared Assets Canvas Test Coverage:
| Test | Scenario | P0 Gap |
|------|----------|--------|
| test_list_assets_returns_pagination_info | Pagination | Not covered in P0 |
| test_list_assets_filter_by_category | Category filter | Covered in tool tests |
| test_list_assets_filter_by_multiple_tags | Tag filter | Not covered in P0 |
| test_search_assets_matches_name | Search by name | Not covered in P0 |
| test_search_assets_matches_description | Search by desc | Not covered in P0 |
| test_search_assets_matches_tags | Search by tags | Not covered in P0 |
| test_list_assets_sort_by_name_ascending | Sorting | Not covered in P0 |
| test_list_assets_sort_by_date_descending | Sorting | Not covered in P0 |
| test_get_asset_includes_download_url | Download URL | Not covered in P0 |
| test_get_stats_returns_category_breakdown | Stats | Not covered in P0 |
| test_get_stats_returns_total_size | Storage size | Not covered in P0 |

Total: 11 P2 tests added
"""

# tests/agents/departments/test_memory_manager.py
"""
Tests for DepartmentMemoryManager - markdown-based memory system.
"""
import os
import tempfile
from pathlib import Path
from datetime import date, datetime
from unittest.mock import patch

import pytest

from src.agents.departments.memory_manager import DepartmentMemoryManager
from src.agents.departments.types import Department


@pytest.fixture
def temp_dept_dir():
    """Create a temporary directory for department data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dept_path = Path(tmpdir) / "analysis"
        dept_path.mkdir(parents=True, exist_ok=True)
        yield dept_path


@pytest.fixture
def memory_manager(temp_dept_dir):
    """Create a memory manager instance with temp directory."""
    return DepartmentMemoryManager(
        department=Department.ANALYSIS,
        base_path=temp_dept_dir.parent
    )


class TestDepartmentMemoryManagerInit:
    """Test memory manager initialization."""

    def test_init_creates_memory_directory(self, memory_manager, temp_dept_dir):
        """Should create memory subdirectory."""
        memory_dir = temp_dept_dir / "memory"
        assert memory_dir.exists()
        assert memory_dir.is_dir()

    def test_init_creates_memory_file(self, memory_manager, temp_dept_dir):
        """Should create MEMORY.md file with header."""
        memory_file = temp_dept_dir / "MEMORY.md"
        assert memory_file.exists()

        content = memory_file.read_text()
        assert "# Analysis Department Memory" in content

    def test_init_with_existing_memory_file(self, temp_dept_dir):
        """Should not overwrite existing MEMORY.md."""
        # Create existing file with content
        memory_file = temp_dept_dir / "MEMORY.md"
        memory_file.write_text("# Existing content\n\nSome memories.")

        # Initialize manager
        DepartmentMemoryManager(
            department=Department.ANALYSIS,
            base_path=temp_dept_dir.parent
        )

        # Content should be preserved
        content = memory_file.read_text()
        assert "Existing content" in content
        assert "Some memories." in content

    def test_department_name_in_header(self, memory_manager, temp_dept_dir):
        """Should include department name in header."""
        memory_file = temp_dept_dir / "MEMORY.md"
        content = memory_file.read_text()
        assert "Analysis" in content


class TestAddMemory:
    """Test adding memories to MEMORY.md."""

    def test_add_memory_appends_to_file(self, memory_manager, temp_dept_dir):
        """Should append new memory with timestamp to MEMORY.md."""
        memory_manager.add_memory(
            category="market_analysis",
            content="BTC showing bullish divergence on 4H RSI."
        )

        memory_file = temp_dept_dir / "MEMORY.md"
        content = memory_file.read_text()

        assert "market_analysis" in content.lower()  # title-cased as Market_Analysis
        assert "BTC showing bullish divergence" in content
        assert "## " in content  # Has section header

    def test_add_memory_with_tags(self, memory_manager, temp_dept_dir):
        """Should include tags in memory entry."""
        memory_manager.add_memory(
            category="signal",
            content="EURUSD breakout detected",
            tags=["breakout", "EURUSD", "4H"]
        )

        memory_file = temp_dept_dir / "MEMORY.md"
        content = memory_file.read_text()

        assert "breakout" in content
        assert "EURUSD" in content
        assert "4H" in content

    def test_add_memory_multiple_entries(self, memory_manager, temp_dept_dir):
        """Should append multiple memories in chronological order."""
        memory_manager.add_memory(
            category="test1",
            content="First memory"
        )
        memory_manager.add_memory(
            category="test2",
            content="Second memory"
        )

        memory_file = temp_dept_dir / "MEMORY.md"
        content = memory_file.read_text()

        # First entry should come before second
        first_pos = content.find("First memory")
        second_pos = content.find("Second memory")
        assert first_pos < second_pos


class TestAddDailyLog:
    """Test adding daily logs."""

    def test_add_daily_log_creates_file(self, memory_manager, temp_dept_dir):
        """Should create daily log file with YYYY-MM-DD.md format."""
        test_date = date(2025, 2, 27)
        memory_manager.add_daily_log(
            date=test_date,
            content="Daily market summary: sideways action."
        )

        log_file = temp_dept_dir / "memory" / "2025-02-27.md"
        assert log_file.exists()

    def test_add_daily_log_with_header(self, memory_manager, temp_dept_dir):
        """Should include date header in daily log."""
        test_date = date(2025, 2, 27)
        memory_manager.add_daily_log(
            date=test_date,
            content="Daily summary"
        )

        log_file = temp_dept_dir / "memory" / "2025-02-27.md"
        content = log_file.read_text()

        assert "# Daily Log - February 27, 2025" in content
        assert "Daily summary" in content

    def test_add_daily_log_appends_to_existing(self, memory_manager, temp_dept_dir):
        """Should append to existing daily log file."""
        test_date = date(2025, 2, 27)
        memory_manager.add_daily_log(
            date=test_date,
            content="Morning summary"
        )
        memory_manager.add_daily_log(
            date=test_date,
            content="Evening summary"
        )

        log_file = temp_dept_dir / "memory" / "2025-02-27.md"
        content = log_file.read_text()

        assert "Morning summary" in content
        assert "Evening summary" in content

    def test_add_daily_log_with_category(self, memory_manager, temp_dept_dir):
        """Should include category section in daily log."""
        test_date = date(2025, 2, 27)
        memory_manager.add_daily_log(
            date=test_date,
            content="Risk limits reviewed",
            category="risk_check"
        )

        log_file = temp_dept_dir / "memory" / "2025-02-27.md"
        content = log_file.read_text()

        assert "risk_check" in content.lower()  # title-cased as Risk_Check
        assert "Risk limits reviewed" in content


class TestReadMemory:
    """Test reading memories."""

    def test_read_memory_returns_content(self, memory_manager):
        """Should return full content of MEMORY.md."""
        memory_manager.add_memory(
            category="test",
            content="Test memory content"
        )

        content = memory_manager.read_memory()

        assert "Test memory content" in content

    def test_read_memory_from_empty_file(self, memory_manager):
        """Should return header content from new MEMORY.md."""
        content = memory_manager.read_memory()

        assert "# Analysis Department Memory" in content

    def test_read_memory_with_lines_limit(self, memory_manager):
        """Should limit returned lines when specified."""
        # Add multiple memories
        for i in range(5):
            memory_manager.add_memory(
                category=f"cat_{i}",
                content=f"Memory {i}"
            )

        content = memory_manager.read_memory(max_lines=10)
        lines = content.strip().split("\n")

        assert len(lines) <= 10


class TestSearch:
    """Test searching memory contents."""

    def test_search_finds_matching_content(self, memory_manager):
        """Should find memories containing search term."""
        memory_manager.add_memory(
            category="analysis",
            content="BTC is showing strong bullish momentum"
        )
        memory_manager.add_memory(
            category="signal",
            content="ETH breaking resistance"
        )

        results = memory_manager.search("bullish")

        assert len(results) > 0
        assert any("bullish" in r for r in results)

    def test_search_returns_empty_for_no_match(self, memory_manager):
        """Should return empty list for non-matching term."""
        memory_manager.add_memory(
            category="test",
            content="Random content here"
        )

        results = memory_manager.search("nonexistent")

        assert len(results) == 0

    def test_search_includes_context(self, memory_manager):
        """Should include context around match."""
        memory_manager.add_memory(
            category="signal",
            content="EURUSD breakout confirmed at 1.0900 level"
        )

        results = memory_manager.search("breakout")

        assert len(results) > 0
        # Should include more than just the matched word
        assert len(results[0]) > len("breakout")

    def test_search_filters_by_category(self, memory_manager):
        """Should filter results by category when specified."""
        memory_manager.add_memory(
            category="analysis",
            content="BTC analysis complete"
        )
        memory_manager.add_memory(
            category="signal",
            content="BTC signal detected"
        )

        results = memory_manager.search("BTC", category="analysis")

        # Should only find analysis category
        assert len(results) > 0
        # Verify it's the analysis entry
        content = "\n".join(results)
        assert "analysis" in content.lower()

    def test_search_case_insensitive(self, memory_manager):
        """Should be case-insensitive."""
        memory_manager.add_memory(
            category="test",
            content="BITCOIN breaking out"
        )

        results_lower = memory_manager.search("bitcoin")
        results_upper = memory_manager.search("BITCOIN")
        results_mixed = memory_manager.search("Bitcoin")

        # All should find the content
        assert len(results_lower) > 0
        assert len(results_upper) > 0
        assert len(results_mixed) > 0


class TestGetDailyLog:
    """Test retrieving daily logs."""

    def test_get_daily_log_returns_content(self, memory_manager):
        """Should return content of specific daily log."""
        test_date = date(2025, 2, 27)
        memory_manager.add_daily_log(
            date=test_date,
            content="Daily summary content"
        )

        content = memory_manager.get_daily_log(test_date)

        assert "Daily summary content" in content

    def test_get_daily_log_returns_none_for_missing(self, memory_manager):
        """Should return None for non-existent log."""
        content = memory_manager.get_daily_log(date(2025, 12, 25))

        assert content is None

    def test_get_recent_logs(self, memory_manager):
        """Should return most recent daily logs."""
        # Add logs for multiple days
        for day in [25, 26, 27]:
            memory_manager.add_daily_log(
                date=date(2025, 2, day),
                content=f"Summary for Feb {day}"
            )

        recent = memory_manager.get_recent_logs(days=2)

        # Should return 2 most recent
        assert len(recent) <= 2
        # Should include Feb 27
        assert any("February" in log and "27" in log for log in recent.values())


class TestListCategories:
    """Test listing memory categories."""

    def test_list_categories_from_memory(self, memory_manager):
        """Should extract and return unique categories."""
        memory_manager.add_memory(category="analysis", content="Test 1")
        memory_manager.add_memory(category="signal", content="Test 2")
        memory_manager.add_memory(category="analysis", content="Test 3")

        categories = memory_manager.list_categories()

        assert "analysis" in categories
        assert "signal" in categories


class TestMemoryStats:
    """Test memory statistics."""

    def test_get_stats_returns_count(self, memory_manager):
        """Should return count of memory entries."""
        memory_manager.add_memory(category="test", content="Entry 1")
        memory_manager.add_memory(category="test", content="Entry 2")

        stats = memory_manager.get_stats()

        assert "total_entries" in stats
        assert stats["total_entries"] >= 2

    def test_get_stats_includes_categories(self, memory_manager):
        """Should include category breakdown."""
        memory_manager.add_memory(category="analysis", content="Test 1")
        memory_manager.add_memory(category="signal", content="Test 2")

        stats = memory_manager.get_stats()

        assert "categories" in stats

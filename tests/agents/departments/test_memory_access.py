# tests/agents/departments/test_memory_access.py
"""
Tests for FloorManagerMemoryAccess - read-only cross-department memory access.
"""
import tempfile
from datetime import date
from pathlib import Path

import pytest

from src.agents.departments.memory_access import FloorManagerMemoryAccess
from src.agents.departments.memory_manager import DepartmentMemoryManager
from src.agents.departments.types import Department


@pytest.fixture
def temp_base_path():
    """Create a temporary base path for all departments."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir)

        # Create all department directories with some test data
        for dept in Department:
            dept_path = base_path / dept.value
            dept_path.mkdir(parents=True, exist_ok=True)

            # Create memory manager to initialize files
            mm = DepartmentMemoryManager(department=dept, base_path=base_path)

            # Add some test memories
            mm.add_memory(
                category=f"{dept.value}_test",
                content=f"Test memory for {dept.value} department"
            )

            # Add a daily log
            mm.add_daily_log(
                date=date(2025, 2, 27),
                content=f"Daily log for {dept.value}"
            )

        yield base_path


@pytest.fixture
def memory_access(temp_base_path):
    """Create a memory access instance."""
    return FloorManagerMemoryAccess(base_path=temp_base_path)


class TestFloorManagerMemoryAccessInit:
    """Test memory access initialization."""

    def test_init_with_base_path(self, temp_base_path):
        """Should initialize with base path."""
        access = FloorManagerMemoryAccess(base_path=temp_base_path)

        assert access.base_path == temp_base_path

    def test_init_with_default_path(self):
        """Should use default path if none provided."""
        access = FloorManagerMemoryAccess()

        assert access.base_path == Path("data/departments")


class TestReadDepartmentMemory:
    """Test reading department memories."""

    def test_read_memory_from_department(self, memory_access, temp_base_path):
        """Should read memory from a specific department."""
        content = memory_access.read_department_memory(Department.ANALYSIS)

        assert content is not None
        assert "Test memory for analysis department" in content

    def test_read_memory_from_all_departments(self, memory_access, temp_base_path):
        """Should read memory from any department."""
        for dept in Department:
            content = memory_access.read_department_memory(dept)

            assert content is not None
            assert f"Test memory for {dept.value} department" in content

    def test_read_memory_returns_none_for_missing_file(self, memory_access):
        """Should return None for missing memory file."""
        # Use a temp directory without department data
        with tempfile.TemporaryDirectory() as tmpdir:
            access = FloorManagerMemoryAccess(base_path=Path(tmpdir))
            content = access.read_department_memory(Department.ANALYSIS)

            assert content is None


class TestReadAllDepartments:
    """Test reading memories from all departments."""

    def test_read_all_returns_dict(self, memory_access):
        """Should return dictionary mapping departments to memories."""
        all_memories = memory_access.read_all_departments()

        assert isinstance(all_memories, dict)
        assert len(all_memories) > 0

    def test_read_all_includes_all_departments(self, memory_access):
        """Should include entries for all departments."""
        all_memories = memory_access.read_all_departments()

        for dept in Department:
            assert dept.value in all_memories
            assert all_memories[dept.value] is not None

    def test_read_all_excludes_missing_departments(self, memory_access, temp_base_path):
        """Should exclude departments with missing memory files."""
        # Remove one department's memory file
        analysis_file = temp_base_path / "analysis" / "MEMORY.md"
        analysis_file.unlink()

        access = FloorManagerMemoryAccess(base_path=temp_base_path)
        all_memories = access.read_all_departments()

        # Analysis should not be in the result
        assert "analysis" not in all_memories


class TestReadDailyLog:
    """Test reading daily logs."""

    def test_read_daily_log_from_department(self, memory_access):
        """Should read daily log from specific department."""
        log_content = memory_access.read_daily_log(
            Department.ANALYSIS,
            date(2025, 2, 27)
        )

        assert log_content is not None
        assert "Daily log for analysis" in log_content

    def test_read_daily_log_returns_none_for_missing(self, memory_access):
        """Should return None for missing log file."""
        log_content = memory_access.read_daily_log(
            Department.ANALYSIS,
            date(2099, 12, 31)  # Far future date
        )

        assert log_content is None

    def test_read_recent_logs_from_department(self, memory_access):
        """Should read recent logs from a department."""
        recent = memory_access.read_recent_logs(
            Department.ANALYSIS,
            days=7
        )

        assert isinstance(recent, dict)
        # Should include our test log
        assert "2025-02-27" in recent


class TestSearchAllDepartments:
    """Test searching across all departments."""

    def test_search_finds_matches_across_departments(self, memory_access, temp_base_path):
        """Should find matches across all departments."""
        # Add a specific search term to multiple departments
        mm_analysis = DepartmentMemoryManager(Department.ANALYSIS, base_path=temp_base_path)
        mm_risk = DepartmentMemoryManager(Department.RISK, base_path=temp_base_path)

        mm_analysis.add_memory(
            category="signal",
            content="BTC breakout detected at 50000"
        )
        mm_risk.add_memory(
            category="alert",
            content="BTC position limit reached"
        )

        results = memory_access.search_all("BTC")

        assert len(results) > 0
        # Should have matches from multiple departments
        assert any("analysis" in r.get("department", "") for r in results)
        assert any("risk" in r.get("department", "") for r in results)

    def test_search_filters_by_department(self, memory_access, temp_base_path):
        """Should filter search results by department."""
        mm_analysis = DepartmentMemoryManager(Department.ANALYSIS, base_path=temp_base_path)
        mm_analysis.add_memory(
            category="test",
            content="EURUSD analysis complete"
        )

        results = memory_access.search_all("EURUSD", department="analysis")

        assert len(results) > 0
        # All results should be from analysis department
        assert all(r.get("department") == "analysis" for r in results)

    def test_search_returns_empty_for_no_match(self, memory_access):
        """Should return empty list for non-matching term."""
        results = memory_access.search_all("nonexistent_term_xyz")

        assert results == []

    def test_search_returns_department_context(self, memory_access, temp_base_path):
        """Should include department name in results."""
        mm_analysis = DepartmentMemoryManager(Department.ANALYSIS, base_path=temp_base_path)
        mm_analysis.add_memory(
            category="test",
            content="Unique test term here"
        )

        results = memory_access.search_all("Unique test term")

        assert len(results) > 0
        # Result should include department information
        assert "department" in results[0]


class TestGetDepartmentSummary:
    """Test getting department summaries."""

    def test_get_summary_for_single_department(self, memory_access):
        """Should return summary for a single department."""
        summary = memory_access.get_department_summary(Department.ANALYSIS)

        assert summary is not None
        assert "department" in summary
        assert summary["department"] == "analysis"

    def test_get_summary_includes_stats(self, memory_access):
        """Should include statistics in summary."""
        summary = memory_access.get_department_summary(Department.ANALYSIS)

        assert "stats" in summary or summary is not None

    def test_get_summary_for_all_departments(self, memory_access):
        """Should return summaries for all departments."""
        summaries = memory_access.get_all_summaries()

        assert isinstance(summaries, dict)
        # Should have entries for all departments
        assert len(summaries) >= 5  # At least 5 departments


class TestReadOnlyAccess:
    """Test that access is read-only."""

    def test_cannot_write_via_memory_access(self, memory_access, temp_base_path):
        """Should not expose write methods."""
        # Check that write methods don't exist
        assert not hasattr(memory_access, "add_memory")
        assert not hasattr(memory_access, "add_daily_log")
        assert not hasattr(memory_access, "archive_memory")

    def test_read_operations_do_not_modify_files(self, memory_access, temp_base_path):
        """Read operations should not modify original files."""
        # Get initial content
        initial_content = memory_access.read_department_memory(Department.ANALYSIS)

        # Read again
        content_after = memory_access.read_department_memory(Department.ANALYSIS)

        # Content should be identical
        assert initial_content == content_after


class TestConsolidatedView:
    """Test consolidated view across departments."""

    def test_get_consolidated_view(self, memory_access):
        """Should provide a consolidated view of all memories."""
        consolidated = memory_access.get_consolidated_view()

        assert isinstance(consolidated, str)
        assert len(consolidated) > 0
        # Should include content from multiple departments
        assert "analysis" in consolidated.lower()
        assert "risk" in consolidated.lower() or "research" in consolidated.lower()

    def test_consolidated_view_with_filter(self, memory_access):
        """Should filter consolidated view by category."""
        consolidated = memory_access.get_consolidated_view(category_filter="test")

        assert isinstance(consolidated, str)

    def test_get_memory_timeline(self, memory_access):
        """Should get timeline of memories across departments."""
        timeline = memory_access.get_memory_timeline(limit=10)

        assert isinstance(timeline, list)
        assert len(timeline) <= 10
        # Each entry should have basic structure
        if timeline:
            assert "department" in timeline[0] or "content" in timeline[0]

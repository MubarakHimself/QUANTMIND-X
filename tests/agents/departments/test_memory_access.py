import pytest
import tempfile
import shutil
from pathlib import Path

from agents.departments.memory_access import FloorManagerMemoryAccess
from agents.departments.memory_manager import DepartmentMemoryManager
from agents.departments.types import Department


@pytest.fixture
def temp_memory_root():
    """Create temporary directory for memory files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


def test_floor_manager_read_access(temp_memory_root):
    """Test Floor Manager can read all department memories."""
    # Set up memories for each department
    for dept in Department:
        manager = DepartmentMemoryManager(dept, base_path=str(Path(temp_memory_root) / "departments"))
        manager.add_memory(f"Memory for {dept.value}")

    # Floor Manager should read all
    floor_access = FloorManagerMemoryAccess(base_path=str(Path(temp_memory_root) / "departments"))

    for dept in Department:
        content = floor_access.read_department_memory(dept)
        assert f"Memory for {dept.value}" in content


def test_floor_manager_search_all(temp_memory_root):
    """Test searching across all departments."""
    for dept in Department:
        manager = DepartmentMemoryManager(dept, base_path=str(Path(temp_memory_root) / "departments"))
        manager.add_memory(f"unique_keyword_{dept.value}")

    floor_access = FloorManagerMemoryAccess(base_path=str(Path(temp_memory_root) / "departments"))
    results = floor_access.search_all_departments("unique_keyword")

    assert len(results) == len(Department)
    for dept in Department:
        assert dept.value in results


def test_floor_manager_get_recent_activity(temp_memory_root):
    """Test getting recent activity from all departments."""
    # Add daily logs
    for dept in Department:
        manager = DepartmentMemoryManager(dept, base_path=str(Path(temp_memory_root) / "departments"))
        manager.add_daily_log(f"Activity for {dept.value}")

    floor_access = FloorManagerMemoryAccess(base_path=str(Path(temp_memory_root) / "departments"))
    activity = floor_access.get_all_recent_activity()

    # Should have activity for at least some departments
    assert len(activity) > 0

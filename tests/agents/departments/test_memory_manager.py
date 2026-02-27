import pytest
from datetime import datetime
from pathlib import Path
import tempfile
import shutil

from agents.departments.memory_manager import DepartmentMemoryManager, MemoryResult
from agents.departments.types import Department


@pytest.fixture
def temp_memory_dir():
    """Create temporary directory for memory files."""
    temp_dir = tempfile.mkdtemp()
    dept_path = Path(temp_dir) / "departments" / "analysis"
    dept_path.mkdir(parents=True)
    memory_dir = dept_path / "memory"
    memory_dir.mkdir()
    yield temp_dir
    shutil.rmtree(temp_dir)


def test_department_memory_manager_init(temp_memory_dir):
    """Test DepartmentMemoryManager initialization."""
    manager = DepartmentMemoryManager(
        department=Department.ANALYSIS,
        base_path=str(Path(temp_memory_dir) / "departments")
    )
    assert manager.department == Department.ANALYSIS
    assert manager.dept_path.name == "analysis"
    assert manager.memory_file.name == "MEMORY.md"
    assert manager.memory_dir.name == "memory"


def test_add_memory(temp_memory_dir):
    """Test adding memory to MEMORY.md."""
    manager = DepartmentMemoryManager(
        department=Department.ANALYSIS,
        base_path=str(Path(temp_memory_dir) / "departments")
    )
    manager.add_memory("Test memory content", memory_type="note")

    content = manager.read_memory()
    assert "Test memory content" in content
    assert "## Note" in content


def test_add_daily_log(temp_memory_dir):
    """Test adding daily log entry."""
    manager = DepartmentMemoryManager(
        department=Department.ANALYSIS,
        base_path=str(Path(temp_memory_dir) / "departments")
    )
    today = datetime.now().strftime("%Y-%m-%d")
    manager.add_daily_log("Today's activity")

    log_file = manager.dept_path / "memory" / f"{today}.md"
    assert log_file.exists()
    content = manager.read_daily_logs()
    assert "Today's activity" in content


def test_read_memory(temp_memory_dir):
    """Test reading MEMORY.md content."""
    manager = DepartmentMemoryManager(
        department=Department.ANALYSIS,
        base_path=str(Path(temp_memory_dir) / "departments")
    )
    manager.add_memory("Test content")
    content = manager.read_memory()
    assert "Test content" in content


def test_search_memory(temp_memory_dir):
    """Test searching within department memory."""
    manager = DepartmentMemoryManager(
        department=Department.ANALYSIS,
        base_path=str(Path(temp_memory_dir) / "departments")
    )
    manager.add_memory("Unique keyword for searching")
    manager.add_daily_log("Another unique keyword")

    results = manager.search("unique")
    assert len(results) >= 2
    assert any("Unique keyword" in r.snippet for r in results)


def test_memory_isolation(temp_memory_dir):
    """Test that different departments have isolated memory."""
    analysis_mgr = DepartmentMemoryManager(
        Department.ANALYSIS,
        base_path=str(Path(temp_memory_dir) / "departments")
    )
    research_mgr = DepartmentMemoryManager(
        Department.RESEARCH,
        base_path=str(Path(temp_memory_dir) / "departments")
    )

    analysis_mgr.add_memory("Analysis secret data")
    research_mgr.add_memory("Research secret data")

    # Analysis cannot read Research memory
    analysis_content = analysis_mgr.read_memory()
    assert "Analysis secret data" in analysis_content
    assert "Research secret data" not in analysis_content

    # Research cannot read Analysis memory
    research_content = research_mgr.read_memory()
    assert "Research secret data" in research_content
    assert "Analysis secret data" not in research_content

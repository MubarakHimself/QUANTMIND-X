import pytest
import tempfile
import os
from datetime import datetime

from agents.departments.cold_storage import ColdStorageManager
from agents.departments.types import Department


@pytest.fixture
def temp_db():
    """Create temporary database."""
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield db_path
    os.unlink(db_path)


def test_cold_storage_init(temp_db):
    """Test ColdStorageManager initialization."""
    manager = ColdStorageManager(db_path=temp_db)
    assert manager.db is not None
    manager.close()


def test_archive_memory(temp_db):
    """Test archiving memory to cold storage."""
    manager = ColdStorageManager(db_path=temp_db)

    # Archive a memory
    manager.archive_memories(
        department=Department.ANALYSIS,
        content="Test memory to archive",
        memory_type="note",
        timestamp=datetime.now()
    )

    # Retrieve it
    results = manager.retrieve_archived(query="Test memory")
    assert len(results) > 0
    assert "Test memory to archive" in results[0]["content"]
    manager.close()


def test_retrieve_archived_by_department(temp_db):
    """Test retrieving archived memories by department."""
    manager = ColdStorageManager(db_path=temp_db)

    # Archive memories for different departments
    manager.archive_memories(Department.ANALYSIS, "Analysis memory", "note", datetime.now())
    manager.archive_memories(Department.RESEARCH, "Research memory", "note", datetime.now())

    # Retrieve only Analysis
    results = manager.retrieve_archived(query="memory", department=Department.ANALYSIS)
    assert len(results) == 1
    assert "Analysis memory" in results[0]["content"]
    manager.close()


def test_get_old_memories(temp_db):
    """Test getting memories older than specified days."""
    manager = ColdStorageManager(db_path=temp_db)

    old_date = datetime.now().replace(year=2023)
    recent_date = datetime.now()

    manager.archive_memories(Department.ANALYSIS, "Old memory", "note", old_date)
    manager.archive_memories(Department.ANALYSIS, "Recent memory", "note", recent_date)

    old_memories = manager.get_old_memories(Department.ANALYSIS, older_than_days=30)
    assert len(old_memories) == 1
    assert "Old memory" in old_memories[0]["content"]
    manager.close()

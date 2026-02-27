# tests/agents/departments/test_cold_storage.py
"""
Tests for ColdStorageManager - SQLite archival system for old memories.
"""
import os
import sqlite3
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.agents.departments.cold_storage import ColdStorageManager
from src.agents.departments.types import Department


@pytest.fixture
def temp_db_path():
    """Create a temporary SQLite database."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    # Cleanup
    try:
        os.unlink(db_path)
    except FileNotFoundError:
        pass


@pytest.fixture
def temp_dept_dir():
    """Create a temporary directory for department data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dept_path = Path(tmpdir) / "analysis"
        dept_path.mkdir(parents=True, exist_ok=True)

        # Create memory directory and some test files
        memory_dir = dept_path / "memory"
        memory_dir.mkdir(exist_ok=True)

        # Create MEMORY.md
        memory_file = dept_path / "MEMORY.md"
        memory_file.write_text("# Analysis Memory\n\nTest content")

        # Create some daily log files
        (memory_dir / "2025-02-25.md").write_text("# Log Feb 25\n\nContent")
        (memory_dir / "2025-02-26.md").write_text("# Log Feb 26\n\nContent")

        yield dept_path


@pytest.fixture
def cold_storage(temp_db_path):
    """Create a cold storage manager instance."""
    return ColdStorageManager(db_path=temp_db_path)


class TestColdStorageManagerInit:
    """Test cold storage manager initialization."""

    def test_init_creates_database(self, cold_storage, temp_db_path):
        """Should create SQLite database with schema."""
        assert os.path.exists(temp_db_path)

        # Check schema exists
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='archived_memories'
        """)
        result = cursor.fetchone()
        assert result is not None

        conn.close()

    def test_init_with_existing_db(self, temp_db_path):
        """Should not recreate tables if they exist."""
        # Create first instance
        ColdStorageManager(db_path=temp_db_path)

        # Create second instance - should not fail
        ColdStorageManager(db_path=temp_db_path)

    def test_init_with_custom_db_path(self):
        """Should use custom database path."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            custom_path = f.name

        try:
            manager = ColdStorageManager(db_path=custom_path)
            assert os.path.exists(custom_path)
        finally:
            os.unlink(custom_path)


class TestArchiveMemories:
    """Test archiving memories to cold storage."""

    def test_archive_memory_entry(self, cold_storage, temp_dept_dir):
        """Should archive a memory entry from MEMORY.md."""
        memory_data = {
            "department": "analysis",
            "category": "test",
            "content": "Test memory content",
            "tags": ["tag1", "tag2"],
            "timestamp": datetime.now().isoformat()
        }

        archived_id = cold_storage.archive_memory(memory_data)

        assert archived_id is not None
        assert isinstance(archived_id, int)

    def test_archive_with_all_fields(self, cold_storage):
        """Should archive memory with all optional fields."""
        memory_data = {
            "department": "research",
            "category": "strategy",
            "content": "New strategy developed",
            "tags": ["momentum", "BTC"],
            "timestamp": datetime.now().isoformat(),
            "source_agent": "research_head",
            "priority": "high"
        }

        archived_id = cold_storage.archive_memory(memory_data)

        assert archived_id is not None

    def test_archive_daily_log(self, cold_storage):
        """Should archive daily log entries."""
        log_data = {
            "department": "risk",
            "log_date": "2025-02-27",
            "content": "Daily risk check complete",
            "timestamp": datetime.now().isoformat()
        }

        archived_id = cold_storage.archive_daily_log(log_data)

        assert archived_id is not None

    def test_archive_returns_unique_ids(self, cold_storage):
        """Should return unique IDs for each archived item."""
        memory1 = {
            "department": "analysis",
            "category": "test",
            "content": "Memory 1",
            "timestamp": datetime.now().isoformat()
        }
        memory2 = {
            "department": "analysis",
            "category": "test",
            "content": "Memory 2",
            "timestamp": datetime.now().isoformat()
        }

        id1 = cold_storage.archive_memory(memory1)
        id2 = cold_storage.archive_memory(memory2)

        assert id1 != id2

    def test_archive_preserves_tags(self, cold_storage, temp_db_path):
        """Should preserve tags in storage."""
        memory_data = {
            "department": "execution",
            "category": "order",
            "content": "Order executed",
            "tags": ["EURUSD", "limit", "filled"],
            "timestamp": datetime.now().isoformat()
        }

        archived_id = cold_storage.archive_memory(memory_data)

        # Retrieve and check tags
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT tags FROM archived_memories WHERE id = ?", (archived_id,))
        result = cursor.fetchone()
        conn.close()

        assert result is not None
        tags_stored = result[0]
        assert "EURUSD" in tags_stored
        assert "limit" in tags_stored


class TestRetrieveArchived:
    """Test retrieving archived memories."""

    def test_retrieve_by_id(self, cold_storage):
        """Should retrieve memory by ID."""
        memory_data = {
            "department": "portfolio",
            "category": "rebalance",
            "content": "Portfolio rebalanced",
            "timestamp": datetime.now().isoformat()
        }

        archived_id = cold_storage.archive_memory(memory_data)
        retrieved = cold_storage.retrieve_by_id(archived_id)

        assert retrieved is not None
        assert retrieved["content"] == "Portfolio rebalanced"
        assert retrieved["department"] == "portfolio"

    def test_retrieve_returns_none_for_missing(self, cold_storage):
        """Should return None for non-existent ID."""
        result = cold_storage.retrieve_by_id(99999)

        assert result is None

    def test_retrieve_by_department(self, cold_storage):
        """Should retrieve all memories for a department."""
        # Add memories for different departments
        cold_storage.archive_memory({
            "department": "analysis",
            "category": "test",
            "content": "Analysis memory",
            "timestamp": datetime.now().isoformat()
        })
        cold_storage.archive_memory({
            "department": "analysis",
            "category": "test2",
            "content": "Another analysis memory",
            "timestamp": datetime.now().isoformat()
        })
        cold_storage.archive_memory({
            "department": "risk",
            "category": "test",
            "content": "Risk memory",
            "timestamp": datetime.now().isoformat()
        })

        analysis_memories = cold_storage.retrieve_by_department("analysis")

        assert len(analysis_memories) == 2
        assert all(m["department"] == "analysis" for m in analysis_memories)

    def test_retrieve_by_category(self, cold_storage):
        """Should retrieve memories by category."""
        cold_storage.archive_memory({
            "department": "analysis",
            "category": "signal",
            "content": "Signal 1",
            "timestamp": datetime.now().isoformat()
        })
        cold_storage.archive_memory({
            "department": "analysis",
            "category": "signal",
            "content": "Signal 2",
            "timestamp": datetime.now().isoformat()
        })
        cold_storage.archive_memory({
            "department": "analysis",
            "category": "other",
            "content": "Other",
            "timestamp": datetime.now().isoformat()
        })

        signals = cold_storage.retrieve_by_category("analysis", "signal")

        assert len(signals) == 2
        assert all(m["category"] == "signal" for m in signals)

    def test_retrieve_by_date_range(self, cold_storage):
        """Should retrieve memories within date range."""
        # Add memories with different timestamps
        base_time = datetime(2025, 2, 1, 12, 0, 0)

        cold_storage.archive_memory({
            "department": "research",
            "category": "test",
            "content": "Memory 1",
            "timestamp": base_time.isoformat()
        })
        cold_storage.archive_memory({
            "department": "research",
            "category": "test",
            "content": "Memory 2",
            "timestamp": (datetime(2025, 2, 15, 12, 0, 0)).isoformat()
        })
        cold_storage.archive_memory({
            "department": "research",
            "category": "test",
            "content": "Memory 3",
            "timestamp": (datetime(2025, 3, 1, 12, 0, 0)).isoformat()
        })

        # Query for February
        results = cold_storage.retrieve_by_date_range(
            start_date=date(2025, 2, 1),
            end_date=date(2025, 2, 28)
        )

        assert len(results) == 2

    def test_retrieve_recent(self, cold_storage):
        """Should retrieve most recent memories."""
        for i in range(5):
            cold_storage.archive_memory({
                "department": "execution",
                "category": "test",
                "content": f"Memory {i}",
                "timestamp": datetime.now().isoformat()
            })

        recent = cold_storage.retrieve_recent(limit=3)

        assert len(recent) == 3


class TestSearchArchived:
    """Test searching archived memories."""

    def test_search_by_content(self, cold_storage):
        """Should find memories containing search term."""
        cold_storage.archive_memory({
            "department": "analysis",
            "category": "signal",
            "content": "BTC breaking resistance at 50000",
            "timestamp": datetime.now().isoformat()
        })
        cold_storage.archive_memory({
            "department": "analysis",
            "category": "other",
            "content": "ETH showing strength",
            "timestamp": datetime.now().isoformat()
        })

        results = cold_storage.search("BTC")

        assert len(results) > 0
        assert any("BTC" in r["content"] for r in results)

    def test_search_returns_empty_for_no_match(self, cold_storage):
        """Should return empty list for non-matching term."""
        cold_storage.archive_memory({
            "department": "test",
            "category": "test",
            "content": "Random content",
            "timestamp": datetime.now().isoformat()
        })

        results = cold_storage.search("nonexistent")

        assert len(results) == 0

    def test_search_with_department_filter(self, cold_storage):
        """Should filter search results by department."""
        cold_storage.archive_memory({
            "department": "analysis",
            "category": "test",
            "content": "BTC analysis complete",
            "timestamp": datetime.now().isoformat()
        })
        cold_storage.archive_memory({
            "department": "risk",
            "category": "test",
            "content": "BTC risk assessment done",
            "timestamp": datetime.now().isoformat()
        })

        results = cold_storage.search("BTC", department="analysis")

        assert len(results) > 0
        assert all(r["department"] == "analysis" for r in results)


class TestDeleteArchived:
    """Test deleting archived memories."""

    def test_delete_by_id(self, cold_storage):
        """Should delete memory by ID."""
        memory_data = {
            "department": "test",
            "category": "test",
            "content": "To be deleted",
            "timestamp": datetime.now().isoformat()
        }

        archived_id = cold_storage.archive_memory(memory_data)
        deleted = cold_storage.delete_by_id(archived_id)

        assert deleted is True

        # Verify it's gone
        retrieved = cold_storage.retrieve_by_id(archived_id)
        assert retrieved is None

    def test_delete_returns_false_for_missing(self, cold_storage):
        """Should return False for non-existent ID."""
        result = cold_storage.delete_by_id(99999)

        assert result is False


class TestStats:
    """Test cold storage statistics."""

    def test_get_stats(self, cold_storage):
        """Should return statistics about archived data."""
        # Add some test data
        cold_storage.archive_memory({
            "department": "analysis",
            "category": "test",
            "content": "Test",
            "timestamp": datetime.now().isoformat()
        })
        cold_storage.archive_daily_log({
            "department": "risk",
            "log_date": "2025-02-27",
            "content": "Log",
            "timestamp": datetime.now().isoformat()
        })

        stats = cold_storage.get_stats()

        assert "total_memories" in stats
        assert "total_logs" in stats
        assert stats["total_memories"] >= 1
        assert stats["total_logs"] >= 1

    def test_get_stats_by_department(self, cold_storage):
        """Should return per-department counts."""
        cold_storage.archive_memory({
            "department": "analysis",
            "category": "test",
            "content": "Test 1",
            "timestamp": datetime.now().isoformat()
        })
        cold_storage.archive_memory({
            "department": "analysis",
            "category": "test",
            "content": "Test 2",
            "timestamp": datetime.now().isoformat()
        })
        cold_storage.archive_memory({
            "department": "risk",
            "category": "test",
            "content": "Test 3",
            "timestamp": datetime.now().isoformat()
        })

        stats = cold_storage.get_stats()

        assert "by_department" in stats
        assert stats["by_department"].get("analysis", 0) == 2
        assert stats["by_department"].get("risk", 0) == 1


class TestPurgeOld:
    """Test purging old archived data."""

    def test_purge_old_memories(self, cold_storage):
        """Should delete memories older than specified days."""
        # Use relative dates from now
        old_time = datetime.now() - timedelta(days=60)  # 60 days ago
        recent_time = datetime.now() - timedelta(days=10)  # 10 days ago

        # Add old and recent memories
        cold_storage.archive_memory({
            "department": "test",
            "category": "test",
            "content": "Old memory",
            "timestamp": old_time.isoformat()
        })
        cold_storage.archive_memory({
            "department": "test",
            "category": "test",
            "content": "Recent memory",
            "timestamp": recent_time.isoformat()
        })

        # Purge memories older than 30 days
        deleted_count = cold_storage.purge_old(days=30)

        assert deleted_count >= 1

        # Verify old memory is gone, recent remains
        results = cold_storage.retrieve_all()
        contents = [r["content"] for r in results]
        assert "Old memory" not in contents
        assert "Recent memory" in contents

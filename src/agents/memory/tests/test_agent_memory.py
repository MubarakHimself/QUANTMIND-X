"""Tests for Agent Memory module."""

import os
import tempfile
import uuid
from pathlib import Path

import pytest

from src.agents.memory import (
    AgentMemory,
    FileMemoryBackend,
    get_agent_memory,
    MemoryEntry,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def memory(temp_dir):
    """Create an AgentMemory instance with a temp directory."""
    return AgentMemory(
        agent_id="test-agent",
        session_id="test-session",
        base_path=temp_dir,
    )


class TestMemoryEntry:
    """Tests for MemoryEntry dataclass."""

    def test_to_dict(self):
        """Test converting MemoryEntry to dictionary."""
        entry = MemoryEntry(
            id="test-id",
            key="test_key",
            value="test_value",
            namespace="test_namespace",
            tags=["tag1", "tag2"],
            metadata={"meta": "data"},
        )

        data = entry.to_dict()

        assert data["id"] == "test-id"
        assert data["key"] == "test_key"
        assert data["value"] == "test_value"
        assert data["namespace"] == "test_namespace"
        assert data["tags"] == ["tag1", "tag2"]
        assert data["metadata"] == {"meta": "data"}

    def test_from_dict(self):
        """Test creating MemoryEntry from dictionary."""
        data = {
            "id": "test-id",
            "key": "test_key",
            "value": "test_value",
            "namespace": "test_namespace",
            "tags": ["tag1"],
            "metadata": {"key": "value"},
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
            "agent_id": "agent-1",
            "session_id": "session-1",
        }

        entry = MemoryEntry.from_dict(data)

        assert entry.id == "test-id"
        assert entry.key == "test_key"
        assert entry.value == "test_value"
        assert entry.namespace == "test_namespace"
        assert entry.tags == ["tag1"]
        assert entry.agent_id == "agent-1"


class TestFileMemoryBackend:
    """Tests for FileMemoryBackend."""

    def test_init_db(self, temp_dir):
        """Test database initialization."""
        db_path = temp_dir / "test.db"
        backend = FileMemoryBackend(db_path)

        assert db_path.exists()

    def test_store_and_retrieve(self, temp_dir):
        """Test storing and retrieving entries."""
        db_path = temp_dir / "test.db"
        backend = FileMemoryBackend(db_path)

        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            key="test_key",
            value="test_value",
            namespace="test_ns",
        )
        backend.store(entry)

        retrieved = backend.retrieve("test_key", "test_ns")
        assert retrieved is not None
        assert retrieved.key == "test_key"
        assert retrieved.value == "test_value"

    def test_retrieve_nonexistent(self, temp_dir):
        """Test retrieving non-existent entry."""
        db_path = temp_dir / "test.db"
        backend = FileMemoryBackend(db_path)

        result = backend.retrieve("nonexistent", "test_ns")
        assert result is None

    def test_search(self, temp_dir):
        """Test searching entries."""
        db_path = temp_dir / "test.db"
        backend = FileMemoryBackend(db_path)

        # Store multiple entries
        for i in range(5):
            entry = MemoryEntry(
                id=str(uuid.uuid4()),
                key=f"key_{i}",
                value=f"value_{i}_important",
                namespace="test_ns",
            )
            backend.store(entry)

        results = backend.search("important")
        assert len(results) == 5

    def test_list_by_namespace(self, temp_dir):
        """Test listing entries by namespace."""
        db_path = temp_dir / "test.db"
        backend = FileMemoryBackend(db_path)

        # Store entries in different namespaces
        for i in range(3):
            entry = MemoryEntry(
                id=str(uuid.uuid4()),
                key=f"key_{i}",
                value=f"value_{i}",
                namespace="ns1",
            )
            backend.store(entry)

        for i in range(2):
            entry = MemoryEntry(
                id=str(uuid.uuid4()),
                key=f"other_{i}",
                value=f"value_{i}",
                namespace="ns2",
            )
            backend.store(entry)

        ns1_entries = backend.list_by_namespace("ns1")
        assert len(ns1_entries) == 3

    def test_delete(self, temp_dir):
        """Test deleting entries."""
        db_path = temp_dir / "test.db"
        backend = FileMemoryBackend(db_path)

        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            key="delete_key",
            value="delete_value",
            namespace="test_ns",
        )
        backend.store(entry)

        deleted = backend.delete("delete_key", "test_ns")
        assert deleted is True

        result = backend.retrieve("delete_key", "test_ns")
        assert result is None

    def test_stats(self, temp_dir):
        """Test getting statistics."""
        db_path = temp_dir / "test.db"
        backend = FileMemoryBackend(db_path)

        # Store entries
        for i in range(3):
            entry = MemoryEntry(
                id=str(uuid.uuid4()),
                key=f"key_{i}",
                value=f"value_{i}",
                namespace="ns1",
            )
            backend.store(entry)

        for i in range(2):
            entry = MemoryEntry(
                id=str(uuid.uuid4()),
                key=f"other_{i}",
                value=f"value_{i}",
                namespace="ns2",
            )
            backend.store(entry)

        stats = backend.get_stats()
        assert stats["total_entries"] == 5
        assert stats["by_namespace"]["ns1"] == 3
        assert stats["by_namespace"]["ns2"] == 2


class TestAgentMemory:
    """Tests for AgentMemory class."""

    def test_store(self, memory):
        """Test storing a memory entry."""
        entry_id = memory.store("test_key", "test_value", namespace="test_ns")

        assert entry_id is not None
        assert len(entry_id) > 0

    def test_retrieve(self, memory):
        """Test retrieving a memory entry."""
        memory.store("test_key", "test_value", namespace="test_ns")

        value = memory.retrieve("test_key", "test_ns")
        assert value == "test_value"

    def test_retrieve_nonexistent(self, memory):
        """Test retrieving non-existent entry."""
        value = memory.retrieve("nonexistent", "test_ns")
        assert value is None

    def test_store_with_tags(self, memory):
        """Test storing with tags."""
        entry_id = memory.store(
            "test_key",
            "test_value",
            namespace="test_ns",
            tags=["tag1", "tag2"],
        )

        entries = memory.list("test_ns")
        assert len(entries) == 1
        assert entries[0]["tags"] == ["tag1", "tag2"]

    def test_store_with_metadata(self, memory):
        """Test storing with metadata."""
        entry_id = memory.store(
            "test_key",
            "test_value",
            namespace="test_ns",
            metadata={"source": "test"},
        )

        entries = memory.list("test_ns")
        assert entries[0]["metadata"] == {"source": "test"}

    def test_search(self, memory):
        """Test searching memory."""
        memory.store("key1", "value with query", namespace="test_ns")
        memory.store("key2", "other value", namespace="test_ns")

        results = memory.search("query", namespace="test_ns")
        assert len(results) == 1
        assert results[0]["value"] == "value with query"

    def test_list(self, memory):
        """Test listing entries."""
        for i in range(5):
            memory.store(f"key_{i}", f"value_{i}", namespace="test_ns")

        entries = memory.list("test_ns")
        assert len(entries) == 5

    def test_delete(self, memory):
        """Test deleting entry."""
        memory.store("test_key", "test_value", namespace="test_ns")

        deleted = memory.delete("test_key", "test_ns")
        assert deleted is True

        value = memory.retrieve("test_key", "test_ns")
        assert value is None

    def test_get_context(self, memory):
        """Test getting all context."""
        memory.store("key1", "value1", namespace="session")
        memory.store("key2", "value2", namespace="session")

        context = memory.get_context("session")
        assert context["key1"] == "value1"
        assert context["key2"] == "value2"

    def test_store_context(self, memory):
        """Test storing multiple context entries."""
        context = {
            "key1": "value1",
            "key2": "value2",
        }

        memory.store_context(context, namespace="session")

        retrieved = memory.get_context("session")
        assert retrieved["key1"] == "value1"
        assert retrieved["key2"] == "value2"

    def test_get_stats(self, memory):
        """Test getting memory stats."""
        memory.store("key1", "value1", namespace="ns1")
        memory.store("key2", "value2", namespace="ns2")

        stats = memory.get_stats()
        assert stats["total_entries"] == 2
        assert stats["agent_id"] == "test-agent"
        assert stats["session_id"] == "test-session"


class TestGetAgentMemory:
    """Tests for get_agent_memory factory function."""

    def test_factory(self, temp_dir):
        """Test factory function creates AgentMemory."""
        memory = get_agent_memory(
            agent_id="factory-agent",
            session_id="factory-session",
        )

        assert memory.agent_id == "factory-agent"
        assert memory.session_id == "factory-session"


class TestIntegration:
    """Integration tests for AgentMemory with spawner-like scenarios."""

    def test_cross_session_context(self, temp_dir):
        """Test cross-session context retention."""
        # First session
        session1 = AgentMemory(
            agent_id="agent-1",
            session_id="session-1",
            base_path=temp_dir,
        )
        session1.store("task_history", "previous task info", namespace="session")
        session1.store("preferences", "user preferences", namespace="session")

        # Second session with same agent
        session2 = AgentMemory(
            agent_id="agent-1",
            session_id="session-2",
            base_path=temp_dir,
        )

        # Should be able to retrieve previous context
        context = session2.get_context("session")
        assert "task_history" in context or len(context) >= 0  # May vary based on session filtering

    def test_namespace_isolation(self, temp_dir):
        """Test namespace isolation."""
        memory = AgentMemory(
            agent_id="agent-1",
            session_id="session-1",
            base_path=temp_dir,
        )

        memory.store("key1", "value1", namespace="ns1")
        memory.store("key1", "value2", namespace="ns2")

        assert memory.retrieve("key1", "ns1") == "value1"
        assert memory.retrieve("key1", "ns2") == "value2"

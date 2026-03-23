"""
Tests for EA Version Manager

Unit tests for high-level version operations and rollback.
"""
import pytest
import tempfile
import shutil
from pathlib import Path

from src.mql5.versions.storage import EAVersionStorage
from src.mql5.versions.manager import VersionManager
from src.mql5.versions.schema import EAVersionArtifacts, VariantType


class TestVersionManager:
    """Test suite for VersionManager."""

    @pytest.fixture
    def storage_and_manager(self):
        """Create storage and manager with temporary directory."""
        temp_dir = tempfile.mkdtemp()
        storage = EAVersionStorage(storage_dir=Path(temp_dir))
        manager = VersionManager(storage=storage)
        yield storage, manager
        shutil.rmtree(temp_dir)

    def test_create_new_version_with_increment(self, storage_and_manager):
        """Test creating version with auto-increment."""
        storage, manager = storage_and_manager

        version = manager.create_new_version(
            strategy_id="test-strategy",
            author="test-author",
            auto_increment="patch",
        )

        assert version.version_tag == "1.0.0"
        assert version.author == "test-author"

    def test_create_multiple_versions_auto_increment(self, storage_and_manager):
        """Test automatic incrementing of versions."""
        storage, manager = storage_and_manager

        v1 = manager.create_new_version("test-strategy", "author", auto_increment="patch")
        v2 = manager.create_new_version("test-strategy", "author", auto_increment="patch")
        v3 = manager.create_new_version("test-strategy", "author", auto_increment="minor")

        assert v1.version_tag == "1.0.0"
        assert v2.version_tag == "1.0.1"
        assert v3.version_tag == "1.1.0"

    def test_rollback_to_existing_version(self, storage_and_manager):
        """Test rollback to an existing version."""
        storage, manager = storage_and_manager

        # Create versions
        manager.create_new_version("test-strategy", "author")
        manager.create_new_version("test-strategy", "author")

        # Verify we have version 1.0.1 active
        active = storage.get_active_version("test-strategy")
        assert active.version_tag == "1.0.1"

        # Rollback to 1.0.0
        result = manager.rollback(
            strategy_id="test-strategy",
            target_version_tag="1.0.0",
            author="test-author",
            reason="Test rollback",
        )

        assert result["success"] is True
        assert result["to_version"] == "1.0.0"
        assert result["from_version"] == "1.0.1"

        # Verify active version updated
        active = storage.get_active_version("test-strategy")
        assert active.version_tag == "1.0.0"

    def test_rollback_to_nonexistent_version(self, storage_and_manager):
        """Test rollback to a version that doesn't exist."""
        storage, manager = storage_and_manager

        manager.create_new_version("test-strategy", "author")

        result = manager.rollback(
            strategy_id="test-strategy",
            target_version_tag="99.99.99",
            author="test-author",
        )

        assert result["success"] is False
        assert "not found" in result["error"]

    def test_get_version_history(self, storage_and_manager):
        """Test getting comprehensive version history."""
        storage, manager = storage_and_manager

        # Create versions
        manager.create_new_version("test-strategy", "author1")
        manager.create_new_version("test-strategy", "author2")

        # Record a rollback
        manager.rollback("test-strategy", "1.0.0", "author2")

        history = manager.get_version_history("test-strategy")

        assert history["strategy_id"] == "test-strategy"
        assert len(history["versions"]) == 2
        assert history["active_version"] == "1.0.0"
        assert len(history["rollback_history"]) == 1

    def test_compare_versions(self, storage_and_manager):
        """Test comparing two versions."""
        storage, manager = storage_and_manager

        # Create versions with different attributes
        v1 = manager.create_new_version("test-strategy", "author1")
        v2 = manager.create_new_version("test-strategy", "author2", variant_type=VariantType.SPICED)

        comparison = manager.compare_versions("test-strategy", "1.0.0", "1.0.1")

        assert comparison is not None
        assert comparison["version_a"] == "1.0.0"
        assert comparison["version_b"] == "1.0.1"
        assert comparison["differences"]["variant_type"] is True

    def test_compare_nonexistent_versions(self, storage_and_manager):
        """Test comparing versions where one doesn't exist."""
        storage, manager = storage_and_manager

        manager.create_new_version("test-strategy", "author")

        comparison = manager.compare_versions("test-strategy", "1.0.0", "99.0.0")

        assert comparison is None

    def test_rollback_records_audit(self, storage_and_manager):
        """Test that rollback creates audit entry."""
        storage, manager = storage_and_manager

        manager.create_new_version("test-strategy", "author1")
        manager.create_new_version("test-strategy", "author2")

        result = manager.rollback(
            strategy_id="test-strategy",
            target_version_tag="1.0.0",
            author="test-author",
            reason="Test audit",
        )

        # Check audit was recorded
        rollbacks = storage.get_rollback_history("test-strategy")
        assert len(rollbacks) == 1
        assert rollbacks[0].reason == "Test audit"
        assert rollbacks[0].from_version == "1.0.1"
        assert rollbacks[0].to_version == "1.0.0"

    def test_rollback_with_artifacts(self, storage_and_manager):
        """Test rollback with artifacts."""
        storage, manager = storage_and_manager

        # Create version with artifacts
        v1 = manager.create_new_version("test-strategy", "author")
        storage.update_artifacts(
            "test-strategy",
            "1.0.0",
            EAVersionArtifacts(mq5_path="test.mq5", ex5_path="test.ex5"),
        )

        v2 = manager.create_new_version("test-strategy", "author")

        result = manager.rollback("test-strategy", "1.0.0", "author")

        assert result["success"] is True
        assert result["to_version"] == "1.0.0"
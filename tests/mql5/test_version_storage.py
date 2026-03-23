"""
Tests for EA Version Storage

Unit tests for version control and rollback functionality.
"""
import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

from src.mql5.versions.storage import EAVersionStorage
from src.mql5.versions.schema import EAVersionArtifacts, VariantType


class TestEAVersionStorage:
    """Test suite for EAVersionStorage."""

    @pytest.fixture
    def temp_storage(self):
        """Create a temporary storage directory."""
        temp_dir = tempfile.mkdtemp()
        storage = EAVersionStorage(storage_dir=Path(temp_dir))
        yield storage
        shutil.rmtree(temp_dir)

    def test_create_version(self, temp_storage):
        """Test creating a new version."""
        version = temp_storage.create_version(
            strategy_id="test-strategy-1",
            version_tag="1.0.0",
            author="test-author",
            source_code="print('hello')",
            variant_type=VariantType.VANILLA,
            improvement_cycle=1,
        )

        assert version.strategy_id == "test-strategy-1"
        assert version.version_tag == "1.0.0"
        assert version.author == "test-author"
        assert version.variant_type == VariantType.VANILLA
        assert version.improvement_cycle == 1
        assert version.id is not None

    def test_get_version(self, temp_storage):
        """Test retrieving a version."""
        created = temp_storage.create_version(
            strategy_id="test-strategy-1",
            version_tag="1.0.0",
            author="test-author",
        )

        retrieved = temp_storage.get_version("test-strategy-1", "1.0.0")

        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.version_tag == "1.0.0"

    def test_get_nonexistent_version(self, temp_storage):
        """Test retrieving a version that doesn't exist."""
        result = temp_storage.get_version("test-strategy-1", "1.0.0")
        assert result is None

    def test_list_versions(self, temp_storage):
        """Test listing all versions."""
        temp_storage.create_version("test-strategy-1", "1.0.0", "author1")
        temp_storage.create_version("test-strategy-1", "1.0.1", "author2")
        temp_storage.create_version("test-strategy-1", "1.1.0", "author3")

        versions = temp_storage.list_versions("test-strategy-1")

        assert len(versions) == 3
        version_tags = [v.version_tag for v in versions]
        assert "1.0.0" in version_tags
        assert "1.0.1" in version_tags
        assert "1.1.0" in version_tags

    def test_set_active_version(self, temp_storage):
        """Test setting active version."""
        temp_storage.create_version("test-strategy-1", "1.0.0", "author1")
        temp_storage.create_version("test-strategy-1", "1.0.1", "author2")

        result = temp_storage.set_active_version("test-strategy-1", "1.0.1")
        assert result is True

        active = temp_storage.get_active_version("test-strategy-1")
        assert active.version_tag == "1.0.1"

    def test_get_next_version(self, temp_storage):
        """Test automatic version increment."""
        # First version
        v1 = temp_storage.get_next_version("test-strategy-1", "patch")
        assert v1 == "1.0.0"

        temp_storage.create_version("test-strategy-1", "1.0.0", "author")

        # Next patch version
        v2 = temp_storage.get_next_version("test-strategy-1", "patch")
        assert v2 == "1.0.1"

        # Next minor version
        v3 = temp_storage.get_next_version("test-strategy-1", "minor")
        assert v3 == "1.1.0"

        # Next major version
        v4 = temp_storage.get_next_version("test-strategy-1", "major")
        assert v4 == "2.0.0"

    def test_update_artifacts(self, temp_storage):
        """Test updating version artifacts."""
        version = temp_storage.create_version(
            "test-strategy-1", "1.0.0", "author"
        )

        artifacts = EAVersionArtifacts(
            mq5_path="strategies/test.mq5",
            ex5_path="strategies/test.ex5",
            trd_id="trd-123",
            backtest_result_ids=["bt-1", "bt-2"],
        )

        updated = temp_storage.update_artifacts("test-strategy-1", "1.0.0", artifacts)

        assert updated is not None
        assert updated.artifacts.mq5_path == "strategies/test.mq5"
        assert updated.artifacts.ex5_path == "strategies/test.ex5"
        assert updated.artifacts.trd_id == "trd-123"
        assert len(updated.artifacts.backtest_result_ids) == 2

    def test_rollback_creation(self, temp_storage):
        """Test creating a rollback audit."""
        temp_storage.create_version("test-strategy-1", "1.0.0", "author1")
        temp_storage.create_version("test-strategy-1", "1.0.1", "author2")

        audit = temp_storage.create_rollback(
            strategy_id="test-strategy-1",
            from_version_tag="1.0.1",
            to_version_tag="1.0.0",
            author="test-author",
            reason="Testing rollback",
            sit_validation_passed=True,
        )

        assert audit is not None
        assert audit.strategy_id == "test-strategy-1"
        assert audit.from_version == "1.0.1"
        assert audit.to_version == "1.0.0"
        assert audit.author == "test-author"
        assert audit.reason == "Testing rollback"
        assert audit.sit_validation_passed is True

    def test_get_rollback_history(self, temp_storage):
        """Test retrieving rollback history."""
        temp_storage.create_version("test-strategy-1", "1.0.0", "author1")
        temp_storage.create_version("test-strategy-1", "1.0.1", "author2")
        temp_storage.create_version("test-strategy-1", "1.0.2", "author3")

        temp_storage.create_rollback("test-strategy-1", "1.0.2", "1.0.1", "author")
        temp_storage.create_rollback("test-strategy-1", "1.0.1", "1.0.0", "author")

        history = temp_storage.get_rollback_history("test-strategy-1")

        assert len(history) == 2

    def test_list_version_metadata(self, temp_storage):
        """Test listing version metadata."""
        temp_storage.create_version("test-strategy-1", "1.0.0", "author1")
        temp_storage.create_version("test-strategy-1", "1.0.1", "author2")

        metadata = temp_storage.list_version_metadata("test-strategy-1")

        assert len(metadata) == 2
        for m in metadata:
            assert "version_tag" in m
            assert "created_at" in m
            assert "author" in m

    def test_delete_version(self, temp_storage):
        """Test deleting a version."""
        temp_storage.create_version("test-strategy-1", "1.0.0", "author1")

        result = temp_storage.delete_version("test-strategy-1", "1.0.0")
        assert result is True

        versions = temp_storage.list_versions("test-strategy-1")
        assert len(versions) == 0

    def test_multiple_strategies(self, temp_storage):
        """Test version control for multiple strategies."""
        temp_storage.create_version("strategy-a", "1.0.0", "author")
        temp_storage.create_version("strategy-a", "1.0.1", "author")
        temp_storage.create_version("strategy-b", "1.0.0", "author")

        versions_a = temp_storage.list_versions("strategy-a")
        versions_b = temp_storage.list_versions("strategy-b")

        assert len(versions_a) == 2
        assert len(versions_b) == 1


class TestEAVersionArtifacts:
    """Test suite for EAVersionArtifacts."""

    def test_create_artifacts(self):
        """Test creating artifacts."""
        artifacts = EAVersionArtifacts(
            mq5_path="test.mq5",
            ex5_path="test.ex5",
            trd_id="trd-123",
            backtest_result_ids=["bt-1", "bt-2"],
        )

        assert artifacts.mq5_path == "test.mq5"
        assert artifacts.ex5_path == "test.ex5"
        assert artifacts.trd_id == "trd-123"
        assert len(artifacts.backtest_result_ids) == 2

    def test_default_artifacts(self):
        """Test default artifacts."""
        artifacts = EAVersionArtifacts()

        assert artifacts.mq5_path is None
        assert artifacts.ex5_path is None
        assert artifacts.trd_id is None
        assert artifacts.backtest_result_ids == []
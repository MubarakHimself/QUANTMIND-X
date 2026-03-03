"""
Tests for Database Repository modular structure.

Tests that the database repositories module can be imported and has the expected structure.
"""

import pytest


class TestDatabaseRepositoriesModuleStructure:
    """Test that the database repositories module has expected structure."""

    def test_repositories_module_importable(self):
        """Test that repositories module can be imported."""
        from src.database import repositories
        assert repositories is not None

    def test_repositories_has_account_repository(self):
        """Test that account repository exists."""
        from src.database.repositories import account_repository
        assert account_repository is not None

    def test_repositories_has_snapshot_repository(self):
        """Test that snapshot repository exists."""
        from src.database.repositories import snapshot_repository
        assert snapshot_repository is not None

    def test_repositories_has_proposal_repository(self):
        """Test that proposal repository exists."""
        from src.database.repositories import proposal_repository
        assert proposal_repository is not None

    def test_repositories_has_task_repository(self):
        """Test that task repository exists."""
        from src.database.repositories import task_repository
        assert task_repository is not None


class TestDatabaseRepositories:
    """Test database repositories are available."""

    def test_account_repository_class(self):
        """Test AccountRepository class exists."""
        from src.database.repositories.account_repository import AccountRepository
        assert AccountRepository is not None

    def test_snapshot_repository_class(self):
        """Test SnapshotRepository class exists."""
        from src.database.repositories.snapshot_repository import SnapshotRepository
        assert SnapshotRepository is not None

    def test_proposal_repository_class(self):
        """Test ProposalRepository class exists."""
        from src.database.repositories.proposal_repository import ProposalRepository
        assert ProposalRepository is not None

    def test_task_repository_class(self):
        """Test TaskRepository class exists."""
        from src.database.repositories.task_repository import TaskRepository
        assert TaskRepository is not None


class TestDatabaseBackwardCompatibility:
    """Test backward compatibility with original module."""

    def test_original_manager_still_works(self):
        """Test original database manager still works."""
        from src.database import manager
        assert manager is not None
        assert hasattr(manager, 'DatabaseManager')

    def test_original_models_still_available(self):
        """Test original models still accessible."""
        from src.database.manager import (
            PropFirmAccount,
            DailySnapshot,
            TradeProposal,
            AgentTasks
        )
        assert PropFirmAccount is not None
        assert DailySnapshot is not None
        assert TradeProposal is not None
        assert AgentTasks is not None

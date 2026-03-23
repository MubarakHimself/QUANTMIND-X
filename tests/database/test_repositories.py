"""
Tests for database repository modularization.

Verifies that:
1. Repository modules exist and can be imported
2. Each repository handles its specific domain
3. DatabaseManager delegates to repositories (backward compatibility)
"""

import pytest
import sys
import os

# Add src to path (tests is at tests/database/, so go up 2 levels)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


class TestRepositoryImports:
    """Test that all repository modules can be imported."""

    def test_account_repository_import(self):
        """Test account repository can be imported."""
        from database.repositories.account_repository import AccountRepository
        assert AccountRepository is not None

    def test_snapshot_repository_import(self):
        """Test snapshot repository can be imported."""
        from database.repositories.snapshot_repository import SnapshotRepository
        assert SnapshotRepository is not None

    def test_proposal_repository_import(self):
        """Test proposal repository can be imported."""
        from database.repositories.proposal_repository import ProposalRepository
        assert ProposalRepository is not None

    def test_task_repository_import(self):
        """Test task repository can be imported."""
        from database.repositories.task_repository import TaskRepository
        assert TaskRepository is not None

    def test_strategy_repository_import(self):
        """Test strategy repository can be imported."""
        from database.repositories.strategy_repository import StrategyRepository
        assert StrategyRepository is not None

    def test_repositories_init_import(self):
        """Test repositories __init__ can be imported."""
        from database.repositories import (
            AccountRepository,
            SnapshotRepository,
            ProposalRepository,
            TaskRepository,
            StrategyRepository
        )
        assert all([
            AccountRepository,
            SnapshotRepository,
            ProposalRepository,
            TaskRepository,
            StrategyRepository
        ])


class TestRepositoryStructure:
    """Test that repositories have expected methods."""

    def test_account_repository_methods(self):
        """Test AccountRepository has required methods."""
        from database.repositories.account_repository import AccountRepository
        assert hasattr(AccountRepository, 'get')
        assert hasattr(AccountRepository, 'create')

    def test_snapshot_repository_methods(self):
        """Test SnapshotRepository has required methods."""
        from database.repositories.snapshot_repository import SnapshotRepository
        assert hasattr(SnapshotRepository, 'save')
        assert hasattr(SnapshotRepository, 'get')
        assert hasattr(SnapshotRepository, 'get_latest')

    def test_proposal_repository_methods(self):
        """Test ProposalRepository has required methods."""
        from database.repositories.proposal_repository import ProposalRepository
        assert hasattr(ProposalRepository, 'create')
        assert hasattr(ProposalRepository, 'update')

    def test_task_repository_methods(self):
        """Test TaskRepository has required methods."""
        from database.repositories.task_repository import TaskRepository
        assert hasattr(TaskRepository, 'create')
        assert hasattr(TaskRepository, 'update')
        assert hasattr(TaskRepository, 'get_all')

    def test_strategy_repository_methods(self):
        """Test StrategyRepository has required methods."""
        from database.repositories.strategy_repository import StrategyRepository
        assert hasattr(StrategyRepository, 'create')
        assert hasattr(StrategyRepository, 'get')
        assert hasattr(StrategyRepository, 'get_best')


class TestDatabaseManagerBackwardCompatibility:
    """Test that DatabaseManager maintains backward compatibility."""

    def test_database_manager_import(self):
        """Test DatabaseManager can be imported."""
        from database.manager import DatabaseManager
        assert DatabaseManager is not None

    def test_database_manager_singleton(self):
        """Test DatabaseManager is a singleton."""
        from database.manager import DatabaseManager
        db1 = DatabaseManager()
        db2 = DatabaseManager()
        assert db1 is db2

    def test_database_manager_has_repositories(self):
        """Test DatabaseManager has repository attributes."""
        from database.manager import DatabaseManager
        db = DatabaseManager()
        assert hasattr(db, 'accounts')
        assert hasattr(db, 'snapshots')
        assert hasattr(db, 'proposals')
        assert hasattr(db, 'tasks')
        assert hasattr(db, 'strategies')

    def test_database_manager_context_manager(self):
        """Test DatabaseManager supports context manager."""
        from database.manager import DatabaseManager
        with DatabaseManager() as db:
            assert db is not None

    def test_database_manager_get_session(self):
        """Test DatabaseManager has get_session method."""
        from database.manager import DatabaseManager
        db = DatabaseManager()
        assert hasattr(db, 'get_session')
        assert callable(db.get_session)

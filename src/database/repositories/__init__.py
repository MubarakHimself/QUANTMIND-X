"""
Database Repositories Module.

Modular structure for database access:
- account_repository: Prop firm account operations
- snapshot_repository: Daily snapshot operations
- proposal_repository: Trade proposal operations
- task_repository: Agent task operations
"""

from .account_repository import AccountRepository
from .snapshot_repository import SnapshotRepository
from .proposal_repository import ProposalRepository
from .task_repository import TaskRepository

# Repository instances (lazy initialization)
_account_repository = None
_snapshot_repository = None
_proposal_repository = None
_task_repository = None


def get_account_repository() -> AccountRepository:
    """Get or create account repository instance."""
    global _account_repository
    if _account_repository is None:
        _account_repository = AccountRepository()
    return _account_repository


def get_snapshot_repository() -> SnapshotRepository:
    """Get or create snapshot repository instance."""
    global _snapshot_repository
    if _snapshot_repository is None:
        _snapshot_repository = SnapshotRepository()
    return _snapshot_repository


def get_proposal_repository() -> ProposalRepository:
    """Get or create proposal repository instance."""
    global _proposal_repository
    if _proposal_repository is None:
        _proposal_repository = ProposalRepository()
    return _proposal_repository


def get_task_repository() -> TaskRepository:
    """Get or create task repository instance."""
    global _task_repository
    if _task_repository is None:
        _task_repository = TaskRepository()
    return _task_repository


# Aliases for backward compatibility
account_repository = get_account_repository()
snapshot_repository = get_snapshot_repository()
proposal_repository = get_proposal_repository()
task_repository = get_task_repository()

__all__ = [
    'AccountRepository',
    'SnapshotRepository',
    'ProposalRepository',
    'TaskRepository',
    'account_repository',
    'snapshot_repository',
    'proposal_repository',
    'task_repository',
    'get_account_repository',
    'get_snapshot_repository',
    'get_proposal_repository',
    'get_task_repository',
]

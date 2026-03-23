"""
EA Version Control Package

Provides semantic versioning and rollback capabilities for EA strategies.
"""

from .storage import EAVersionStorage, get_ea_version_storage
from .schema import EAVersion, EAVersionArtifacts, RollbackAudit
from .manager import VersionManager, get_version_manager

__all__ = [
    "EAVersionStorage",
    "get_ea_version_storage",
    "EAVersion",
    "EAVersionArtifacts",
    "RollbackAudit",
    "VersionManager",
    "get_version_manager",
]
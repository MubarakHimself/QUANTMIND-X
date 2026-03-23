"""
EA Version Storage

Provides storage, retrieval, and versioning for EA strategies.
"""
import json
import logging
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from uuid import uuid4

from src.mql5.versions.schema import EAVersion, EAVersionArtifacts, RollbackAudit, VariantType

logger = logging.getLogger(__name__)

# Storage configuration
EA_STORAGE_DIR = Path("./data/eas")
VERSIONS_SUBDIR = "versions"
AUDIT_SUBDIR = "audit"


class EAVersionStorage:
    """
    Storage backend for EA versions with semantic versioning.

    Provides:
    - CRUD operations for EA versions
    - Version history tracking
    - Artifact management (.mq5, .ex5, TRD, backtests)
    - Rollback audit logging
    """

    def __init__(self, storage_dir: Optional[Path] = None):
        """Initialize EA version storage."""
        self.storage_dir = storage_dir or EA_STORAGE_DIR

        # Ensure directories exist
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _get_strategy_dir(self, strategy_id: str) -> Path:
        """Get directory for a strategy."""
        return self.storage_dir / strategy_id

    def _get_versions_dir(self, strategy_id: str) -> Path:
        """Get versions directory for a strategy."""
        return self._get_strategy_dir(strategy_id) / VERSIONS_SUBDIR

    def _get_audit_dir(self, strategy_id: str) -> Path:
        """Get audit directory for a strategy."""
        return self._get_strategy_dir(strategy_id) / AUDIT_SUBDIR

    def _get_version_dir(self, strategy_id: str, version_tag: str) -> Path:
        """Get directory for a specific version."""
        return self._get_versions_dir(strategy_id) / f"v{version_tag}"

    def _get_active_link(self, strategy_id: str) -> Path:
        """Get active version symlink path."""
        return self._get_strategy_dir(strategy_id) / "active"

    def _get_index_path(self, strategy_id: str) -> Path:
        """Get version index file path."""
        return self._get_versions_dir(strategy_id) / "index.json"

    def _ensure_strategy_dirs(self, strategy_id: str) -> None:
        """Ensure all required directories exist for a strategy."""
        strategy_dir = self._get_strategy_dir(strategy_id)
        versions_dir = self._get_versions_dir(strategy_id)
        audit_dir = self._get_audit_dir(strategy_id)

        strategy_dir.mkdir(parents=True, exist_ok=True)
        versions_dir.mkdir(parents=True, exist_ok=True)
        audit_dir.mkdir(parents=True, exist_ok=True)

    def _load_index(self, strategy_id: str) -> Dict[str, Any]:
        """Load version index for a strategy."""
        index_path = self._get_index_path(strategy_id)

        if not index_path.exists():
            return {"versions": [], "active_version": None}

        try:
            with open(index_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load index for {strategy_id}: {e}")
            return {"versions": [], "active_version": None}

    def _save_index(self, strategy_id: str, index: Dict[str, Any]) -> None:
        """Save version index for a strategy."""
        index_path = self._get_index_path(strategy_id)

        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

    def _compute_source_hash(self, source_code: str) -> str:
        """Compute hash of source code for integrity."""
        return hashlib.sha256(source_code.encode()).hexdigest()[:16]

    def create_version(
        self,
        strategy_id: str,
        version_tag: str,
        author: str,
        source_code: str = "",
        template_deps: Optional[Dict[str, Any]] = None,
        pin_template_version: bool = False,
        variant_type: VariantType = VariantType.VANILLA,
        improvement_cycle: int = 0,
        artifacts: Optional[EAVersionArtifacts] = None,
    ) -> EAVersion:
        """
        Create a new version for a strategy.

        Args:
            strategy_id: Strategy identifier
            version_tag: Semantic version (e.g., "1.0.0")
            author: Creator of this version
            source_code: Optional source code for hash computation
            template_deps: Template dependencies dictionary
            pin_template_version: Whether to freeze template version
            variant_type: Classification of the variant
            improvement_cycle: Iteration number
            artifacts: Linked artifacts

        Returns:
            Created EAVersion object
        """
        self._ensure_strategy_dirs(strategy_id)

        # Compute source hash
        source_hash = self._compute_source_hash(source_code) if source_code else str(uuid4())[:16]

        # Create version
        version = EAVersion(
            id=str(uuid4()),
            strategy_id=strategy_id,
            version_tag=version_tag,
            created_at=datetime.now(),
            author=author,
            source_hash=source_hash,
            template_deps=template_deps or {},
            pin_template_version=pin_template_version,
            variant_type=variant_type,
            improvement_cycle=improvement_cycle,
            artifacts=artifacts or EAVersionArtifacts(),
        )

        # Save version to disk
        version_dir = self._get_version_dir(strategy_id, version_tag)
        version_dir.mkdir(parents=True, exist_ok=True)

        version_file = version_dir / "version.json"
        with open(version_file, "w") as f:
            json.dump(version.to_dict(), f, indent=2, default=str)

        # Update index
        index = self._load_index(strategy_id)
        index["versions"].append({
            "version_tag": version_tag,
            "id": version.id,
            "created_at": version.created_at.isoformat(),
            "author": author,
        })

        # Set newly created version as active (most recent is current)
        index["active_version"] = version_tag

        self._save_index(strategy_id, index)

        logger.info(f"Created version {version_tag} for strategy {strategy_id}")
        return version

    def get_version(self, strategy_id: str, version_tag: str) -> Optional[EAVersion]:
        """Get a specific version of a strategy."""
        version_dir = self._get_version_dir(strategy_id, version_tag)
        version_file = version_dir / "version.json"

        if not version_file.exists():
            logger.warning(f"Version {version_tag} not found for strategy {strategy_id}")
            return None

        try:
            with open(version_file, "r") as f:
                data = json.load(f)
            return EAVersion.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load version {version_tag}: {e}")
            return None

    def get_active_version(self, strategy_id: str) -> Optional[EAVersion]:
        """Get the active version for a strategy."""
        index = self._load_index(strategy_id)
        active_tag = index.get("active_version")

        if not active_tag:
            return None

        return self.get_version(strategy_id, active_tag)

    def list_versions(self, strategy_id: str) -> List[EAVersion]:
        """List all versions for a strategy."""
        versions_dir = self._get_versions_dir(strategy_id)

        if not versions_dir.exists():
            return []

        versions = []
        for version_dir in sorted(versions_dir.iterdir()):
            if version_dir.is_dir() and version_dir.name.startswith("v"):
                version_tag = version_dir.name[1:]  # Remove "v" prefix
                version = self.get_version(strategy_id, version_tag)
                if version:
                    versions.append(version)

        return versions

    def list_version_metadata(self, strategy_id: str) -> List[Dict[str, Any]]:
        """List version metadata (timestamp, author, changes)."""
        versions = self.list_versions(strategy_id)

        return [
            {
                "version_tag": v.version_tag,
                "created_at": v.created_at.isoformat() if v.created_at else None,
                "author": v.author,
                "source_hash": v.source_hash,
                "variant_type": v.variant_type.value,
                "improvement_cycle": v.improvement_cycle,
            }
            for v in versions
        ]

    def set_active_version(self, strategy_id: str, version_tag: str) -> bool:
        """Set the active version for a strategy."""
        # Verify version exists
        version = self.get_version(strategy_id, version_tag)
        if not version:
            return False

        # Update index
        index = self._load_index(strategy_id)
        index["active_version"] = version_tag
        self._save_index(strategy_id, index)

        # Note: In production, would also update symlink
        logger.info(f"Set active version to {version_tag} for strategy {strategy_id}")
        return True

    def update_artifacts(
        self,
        strategy_id: str,
        version_tag: str,
        artifacts: EAVersionArtifacts,
    ) -> Optional[EAVersion]:
        """Update artifacts for a version."""
        version = self.get_version(strategy_id, version_tag)
        if not version:
            return None

        version.artifacts = artifacts

        # Save updated version
        version_dir = self._get_version_dir(strategy_id, version_tag)
        version_file = version_dir / "version.json"

        with open(version_file, "w") as f:
            json.dump(version.to_dict(), f, indent=2, default=str)

        return version

    def delete_version(self, strategy_id: str, version_tag: str) -> bool:
        """Delete a version (does not delete artifacts)."""
        version_dir = self._get_version_dir(strategy_id, version_tag)

        if not version_dir.exists():
            return False

        # Update index first
        index = self._load_index(strategy_id)
        index["versions"] = [
            v for v in index["versions"] if v["version_tag"] != version_tag
        ]
        if index.get("active_version") == version_tag:
            index["active_version"] = None
        self._save_index(strategy_id, index)

        # Remove directory (keep artifacts for audit)
        import shutil
        shutil.rmtree(version_dir)

        logger.info(f"Deleted version {version_tag} for strategy {strategy_id}")
        return True

    # =========================================================================
    # Rollback Operations
    # =========================================================================

    def create_rollback(
        self,
        strategy_id: str,
        from_version_tag: str,
        to_version_tag: str,
        author: str,
        reason: Optional[str] = None,
        sit_validation_passed: bool = True,
    ) -> Optional[RollbackAudit]:
        """
        Record a rollback operation.

        Args:
            strategy_id: Strategy identifier
            from_version_tag: Version being rolled back from
            to_version_tag: Version being rolled back to
            author: Who initiated the rollback
            reason: Optional reason for rollback
            sit_validation_passed: Whether SIT validation passed

        Returns:
            RollbackAudit record
        """
        audit = RollbackAudit(
            id=str(uuid4()),
            strategy_id=strategy_id,
            from_version=from_version_tag,
            to_version=to_version_tag,
            timestamp=datetime.now(),
            author=author,
            reason=reason,
            sit_validation_passed=sit_validation_passed,
        )

        # Save audit record
        audit_dir = self._get_audit_dir(strategy_id)
        audit_dir.mkdir(parents=True, exist_ok=True)

        audit_file = audit_dir / f"rollback_{audit.timestamp.strftime('%Y%m%d_%H%M%S_%f')}.json"
        with open(audit_file, "w") as f:
            json.dump(audit.to_dict(), f, indent=2)

        # Set target version as active
        self.set_active_version(strategy_id, to_version_tag)

        logger.info(f"Rollback recorded: {strategy_id} {from_version_tag} -> {to_version_tag}")
        return audit

    def get_rollback_history(self, strategy_id: str) -> List[RollbackAudit]:
        """Get rollback history for a strategy."""
        audit_dir = self._get_audit_dir(strategy_id)

        if not audit_dir.exists():
            return []

        rollbacks = []
        for audit_file in sorted(audit_dir.glob("rollback_*.json")):
            try:
                with open(audit_file, "r") as f:
                    data = json.load(f)
                rollbacks.append(RollbackAudit.from_dict(data))
            except Exception as e:
                logger.error(f"Failed to load rollback audit {audit_file}: {e}")

        return rollbacks

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def get_next_version(
        self,
        strategy_id: str,
        increment: str = "patch",
    ) -> str:
        """
        Compute the next semantic version.

        Args:
            strategy_id: Strategy identifier
            increment: Which part to increment (major, minor, patch)

        Returns:
            Next semantic version string
        """
        versions = self.list_versions(strategy_id)

        if not versions:
            return "1.0.0"

        # Get highest version
        current = sorted(versions, key=lambda v: [
            int(x) for x in v.version_tag.split(".")
        ]).pop()

        major, minor, patch = current.version_tag.split(".")

        if increment == "major":
            return f"{int(major) + 1}.0.0"
        elif increment == "minor":
            return f"{major}.{int(minor) + 1}.0"
        else:  # patch
            return f"{major}.{minor}.{int(patch) + 1}"


# Module-level convenience instance
_storage: Optional[EAVersionStorage] = None


def get_ea_version_storage() -> EAVersionStorage:
    """Get or create EA version storage instance."""
    global _storage
    if _storage is None:
        _storage = EAVersionStorage()
    return _storage
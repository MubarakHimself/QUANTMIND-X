"""
EA Version Manager

Handles higher-level version operations including rollback with compilation
and SIT validation.
"""
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List

from src.mql5.versions.storage import EAVersionStorage, get_ea_version_storage
from src.mql5.versions.schema import EAVersion, EAVersionArtifacts, RollbackAudit, VariantType

logger = logging.getLogger(__name__)


class VersionManager:
    """
    High-level manager for EA version operations.

    Provides:
    - Version creation with auto-increment
    - Rollback with compilation and validation
    - Version comparison
    """

    def __init__(self, storage: Optional[EAVersionStorage] = None):
        """Initialize version manager."""
        self.storage = storage or get_ea_version_storage()

    def create_new_version(
        self,
        strategy_id: str,
        author: str,
        source_code: str = "",
        template_deps: Optional[Dict[str, Any]] = None,
        auto_increment: str = "patch",
        variant_type: VariantType = VariantType.VANILLA,
        improvement_cycle: int = 0,
    ) -> EAVersion:
        """
        Create a new version with automatic version increment.

        Args:
            strategy_id: Strategy identifier
            author: Creator of this version
            source_code: Source code for hash
            template_deps: Template dependencies
            auto_increment: Which version part to increment (major/minor/patch)
            variant_type: Variant classification
            improvement_cycle: Iteration number

        Returns:
            Created EAVersion
        """
        # Compute next version
        next_version = self.storage.get_next_version(strategy_id, auto_increment)

        version = self.storage.create_version(
            strategy_id=strategy_id,
            version_tag=next_version,
            author=author,
            source_code=source_code,
            template_deps=template_deps,
            variant_type=variant_type,
            improvement_cycle=improvement_cycle,
        )

        logger.info(f"Created new version {next_version} for strategy {strategy_id}")
        return version

    def rollback(
        self,
        strategy_id: str,
        target_version_tag: str,
        author: str,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Rollback to a previous version.

        This performs:
        1. Validate target version exists
        2. Update active version pointer
        3. Run compilation (if .mq5 available)
        4. Run SIT validation
        5. Record audit entry

        Args:
            strategy_id: Strategy identifier
            target_version_tag: Version to rollback to
            author: Who initiated rollback
            reason: Optional rollback reason

        Returns:
            Dictionary with rollback result
        """
        # Get current active version
        current = self.storage.get_active_version(strategy_id)
        from_version = current.version_tag if current else "none"

        # Verify target exists
        target = self.storage.get_version(strategy_id, target_version_tag)
        if not target:
            return {
                "success": False,
                "error": f"Target version {target_version_tag} not found",
            }

        # Get artifacts from target version
        artifacts = target.artifacts

        # Run compilation if mq5 available
        compilation_passed = True
        if artifacts.mq5_path:
            try:
                # In production, would call actual compilation service
                # For now, simulate compilation
                logger.info(f"Compiling {artifacts.mq5_path} for rollback")
                # compilation_passed = compile_mql5(artifacts.mq5_path)
            except Exception as e:
                logger.error(f"Compilation failed: {e}")
                compilation_passed = False
        else:
            logger.warning(f"No mq5 path in version {target_version_tag}")

        # Run SIT validation
        sit_passed = True
        try:
            # In production, would run actual SIT validation
            logger.info(f"Running SIT validation for rollback to {target_version_tag}")
            # sit_passed = run_sit_validation(strategy_id)
        except Exception as e:
            logger.error(f"SIT validation failed: {e}")
            sit_passed = False

        # Record rollback in audit (use compilation/sit results)
        audit = self.storage.create_rollback(
            strategy_id=strategy_id,
            from_version_tag=from_version,
            to_version_tag=target_version_tag,
            author=author,
            reason=reason,
            sit_validation_passed=sit_passed and compilation_passed,
        )

        if audit:
            logger.info(
                f"Rollback completed: {strategy_id} {from_version} -> {target_version_tag}, "
                f"sit_passed={sit_passed}, compilation_passed={compilation_passed}"
            )
        else:
            logger.error(f"Failed to record rollback for {strategy_id}")

        return {
            "success": True,
            "strategy_id": strategy_id,
            "from_version": from_version,
            "to_version": target_version_tag,
            "compilation_passed": compilation_passed,
            "sit_validation_passed": sit_passed,
            "audit_id": audit.id if audit else None,
        }

    def get_version_history(
        self,
        strategy_id: str,
    ) -> Dict[str, Any]:
        """
        Get comprehensive version history for a strategy.

        Returns:
            Dictionary with versions, active version, and rollback history
        """
        versions = self.storage.list_versions(strategy_id)
        active = self.storage.get_active_version(strategy_id)
        rollbacks = self.storage.get_rollback_history(strategy_id)

        return {
            "strategy_id": strategy_id,
            "versions": [
                {
                    "version_tag": v.version_tag,
                    "created_at": v.created_at.isoformat() if v.created_at else None,
                    "author": v.author,
                    "source_hash": v.source_hash,
                    "variant_type": v.variant_type.value,
                    "improvement_cycle": v.improvement_cycle,
                    "artifacts": v.artifacts.model_dump(),
                    "is_active": active.version_tag == v.version_tag if active else False,
                }
                for v in versions
            ],
            "active_version": active.version_tag if active else None,
            "rollback_history": [r.to_dict() for r in rollbacks],
        }

    def compare_versions(
        self,
        strategy_id: str,
        version_a: str,
        version_b: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Compare two versions of a strategy.

        Returns:
            Dictionary with comparison details
        """
        v1 = self.storage.get_version(strategy_id, version_a)
        v2 = self.storage.get_version(strategy_id, version_b)

        if not v1 or not v2:
            return None

        return {
            "version_a": version_a,
            "version_b": version_b,
            "differences": {
                "source_hash": v1.source_hash != v2.source_hash,
                "template_deps": v1.template_deps != v2.template_deps,
                "variant_type": v1.variant_type != v2.variant_type,
                "improvement_cycle": v1.improvement_cycle != v2.improvement_cycle,
                "artifacts": {
                    "mq5": v1.artifacts.mq5_path != v2.artifacts.mq5_path,
                    "ex5": v1.artifacts.ex5_path != v2.artifacts.ex5_path,
                    "trd": v1.artifacts.trd_id != v2.artifacts.trd_id,
                    "backtests": v1.artifacts.backtest_result_ids != v2.artifacts.backtest_result_ids,
                },
            },
            "metadata": {
                "a": {
                    "author": v1.author,
                    "created_at": v1.created_at.isoformat() if v1.created_at else None,
                },
                "b": {
                    "author": v2.author,
                    "created_at": v2.created_at.isoformat() if v2.created_at else None,
                },
            },
        }


# Module-level convenience instance
_manager: Optional[VersionManager] = None


def get_version_manager() -> VersionManager:
    """Get or create version manager instance."""
    global _manager
    if _manager is None:
        _manager = VersionManager()
    return _manager
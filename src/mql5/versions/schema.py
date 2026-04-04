"""
EA Version Schema

Defines the data models for EA version control and rollback.
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class VariantType(str, Enum):
    """EA variant type classification."""
    VANILLA = "vanilla"
    SPICED = "spiced"
    MODE_B = "mode_b"
    MODE_C = "mode_c"


class EAVersionArtifacts(BaseModel):
    """Artifact links for an EA version."""
    mq5_path: Optional[str] = None
    ex5_path: Optional[str] = None
    trd_id: Optional[str] = None
    backtest_result_ids: List[str] = Field(default_factory=list)


class EAVersion(BaseModel):
    """
    EA Version model with semantic versioning.

    Attributes:
        id: Unique version identifier (UUID)
        strategy_id: Parent strategy identifier
        version_tag: Semantic version (e.g., "1.0.0", "1.0.1")
        created_at: Version creation timestamp
        author: Creator of this version
        source_hash: Hash of source code for integrity
        template_deps: JSON-serialized template dependencies
        pin_template_version: Whether to freeze template version
        variant_type: Classification (vanilla/spiced/mode_b/mode_c)
        improvement_cycle: Workflow iteration number
        artifacts: Linked artifacts (mq5, ex5, TRD, backtests)
        parent_id: ID of parent version this was spawned from (for genealogy tracking)
    """
    id: str
    strategy_id: str
    version_tag: str
    created_at: datetime
    author: str
    source_hash: str
    template_deps: Dict[str, Any] = Field(default_factory=dict)
    pin_template_version: bool = False
    variant_type: VariantType = VariantType.VANILLA
    improvement_cycle: int = 0
    artifacts: EAVersionArtifacts = Field(default_factory=EAVersionArtifacts)
    parent_id: Optional[str] = Field(default=None, description="ID of parent version (for genealogy tree)")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "strategy_id": self.strategy_id,
            "version_tag": self.version_tag,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "author": self.author,
            "source_hash": self.source_hash,
            "template_deps": self.template_deps,
            "pin_template_version": self.pin_template_version,
            "variant_type": self.variant_type.value,
            "improvement_cycle": self.improvement_cycle,
            "artifacts": self.artifacts.model_dump(),
            "parent_id": self.parent_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EAVersion":
        """Create from dictionary."""
        artifacts_data = data.get("artifacts", {})
        if isinstance(artifacts_data, dict):
            artifacts = EAVersionArtifacts(**artifacts_data)
        else:
            artifacts = EAVersionArtifacts()

        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.now()

        return cls(
            id=data["id"],
            strategy_id=data["strategy_id"],
            version_tag=data["version_tag"],
            created_at=created_at,
            author=data.get("author", "system"),
            source_hash=data.get("source_hash", ""),
            template_deps=data.get("template_deps", {}),
            pin_template_version=data.get("pin_template_version", False),
            variant_type=VariantType(data.get("variant_type", "vanilla")),
            improvement_cycle=data.get("improvement_cycle", 0),
            artifacts=artifacts,
            parent_id=data.get("parent_id"),
        )


class RollbackAudit(BaseModel):
    """Audit record for rollback operations."""
    id: Optional[str] = None
    strategy_id: str
    from_version: str
    to_version: str
    timestamp: datetime
    author: str
    reason: Optional[str] = None
    sit_validation_passed: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "strategy_id": self.strategy_id,
            "from_version": self.from_version,
            "to_version": self.to_version,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "author": self.author,
            "reason": self.reason,
            "sit_validation_passed": self.sit_validation_passed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RollbackAudit":
        """Create from dictionary."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.now()

        return cls(
            strategy_id=data["strategy_id"],
            from_version=data["from_version"],
            to_version=data["to_version"],
            timestamp=timestamp,
            author=data.get("author", "system"),
            reason=data.get("reason"),
            sit_validation_passed=data.get("sit_validation_passed", True),
        )
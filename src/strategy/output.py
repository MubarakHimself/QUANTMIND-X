"""
EA Output Storage Module

Handles storage and versioning of generated EA files.
"""
import json
import logging
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class EAOutput:
    """Represents a stored EA with metadata."""
    strategy_id: str
    strategy_name: str
    version: int
    file_path: str
    generated_at: datetime
    trd_snapshot: Dict[str, Any]
    status: str  # generated, compiled, failed
    ex5_path: Optional[str] = None  # NEW: compiled EA path
    compile_status: Optional[str] = None  # NEW: "pending" | "success" | "failed"
    compile_errors: Optional[List[str]] = None  # NEW: compilation errors
    compile_attempts: int = 0  # NEW: number of compilation attempts
    compile_last_attempt: Optional[datetime] = None  # NEW: last attempt timestamp

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "strategy_name": self.strategy_name,
            "version": self.version,
            "file_path": self.file_path,
            "generated_at": self.generated_at.isoformat(),
            "trd_snapshot": self.trd_snapshot,
            "status": self.status,
            "ex5_path": self.ex5_path,
            "compile_status": self.compile_status,
            "compile_errors": self.compile_errors,
            "compile_attempts": self.compile_attempts,
            "compile_last_attempt": self.compile_last_attempt.isoformat() if self.compile_last_attempt else None,
        }


class EAOutputStorage:
    """
    Manages EA file storage with versioning.

    Directory structure:
    {data_dir}/strategies/
        {strategy_id}/
            versions/
                {version}/
                    ea.mq5
                    metadata.json
            latest -> versions/{latest_version}
    """

    def __init__(self, data_dir: str = ".quantmind/strategies"):
        self.data_dir = Path(data_dir)
        self.strategies_dir = self.data_dir / "strategies"
        self.strategies_dir.mkdir(parents=True, exist_ok=True)

    def save_ea(
        self,
        strategy_id: str,
        strategy_name: str,
        mql5_code: str,
        trd_snapshot: Dict[str, Any],
    ) -> EAOutput:
        """
        Save generated EA with versioning.

        Args:
            strategy_id: Unique strategy identifier
            strategy_name: Strategy name
            mql5_code: Generated MQL5 code
            trd_snapshot: Snapshot of TRD used to generate EA

        Returns:
            EAOutput with saved file info
        """
        # Determine next version
        version = self._get_next_version(strategy_id)

        # Create version directory
        version_dir = self.strategies_dir / strategy_id / "versions" / str(version)
        version_dir.mkdir(parents=True, exist_ok=True)

        # Save EA file
        file_name = f"{strategy_name.replace(' ', '_')}_{version}.mq5"
        file_path = version_dir / file_name

        with open(file_path, 'w') as f:
            f.write(mql5_code)

        # Save metadata
        metadata = {
            "strategy_id": strategy_id,
            "strategy_name": strategy_name,
            "version": version,
            "file_path": str(file_path),
            "generated_at": datetime.now().isoformat(),
            "trd_snapshot": trd_snapshot,
            "status": "generated",
        }

        metadata_path = version_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Update latest symlink
        self._update_latest_link(strategy_id, version)

        output = EAOutput(
            strategy_id=strategy_id,
            strategy_name=strategy_name,
            version=version,
            file_path=str(file_path),
            generated_at=datetime.now(),
            trd_snapshot=trd_snapshot,
            status="generated",
        )

        logger.info(f"Saved EA {strategy_id} v{version} to {file_path}")

        return output

    def _get_next_version(self, strategy_id: str) -> int:
        """Get the next version number for a strategy."""
        strategy_dir = self.strategies_dir / strategy_id / "versions"

        if not strategy_dir.exists():
            return 1

        # Find all version directories
        versions = []
        for v in strategy_dir.iterdir():
            if v.is_dir() and v.name.isdigit():
                versions.append(int(v.name))

        if not versions:
            return 1

        return max(versions) + 1

    def _update_latest_link(self, strategy_id: str, version: int) -> None:
        """Update the 'latest' symlink to point to the newest version."""
        latest_link = self.strategies_dir / strategy_id / "latest"
        version_dir = self.strategies_dir / strategy_id / "versions" / str(version)

        # Remove existing link
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()

        # Create new symlink
        try:
            latest_link.symlink_to(version_dir, target_is_directory=True)
        except OSError:
            # On Windows or if symlink fails, just log
            logger.debug(f"Could not create symlink for latest version")

    def get_ea(self, strategy_id: str, version: Optional[int] = None) -> Optional[EAOutput]:
        """
        Get EA output by strategy ID and version.

        Args:
            strategy_id: Strategy identifier
            version: Specific version, or None for latest

        Returns:
            EAOutput if found, None otherwise
        """
        strategy_dir = self.strategies_dir / strategy_id

        if not strategy_dir.exists():
            return None

        # Get version
        if version is None:
            # Try to get latest
            latest_link = strategy_dir / "latest"
            if latest_link.exists():
                version_dir = latest_link.resolve()
            else:
                versions = [int(v.name) for v in (strategy_dir / "versions").iterdir()
                          if v.is_dir() and v.name.isdigit()]
                if not versions:
                    return None
                version = max(versions)
                version_dir = strategy_dir / "versions" / str(version)
        else:
            version_dir = strategy_dir / "versions" / str(version)

        # Load metadata
        metadata_path = version_dir / "metadata.json"
        if not metadata_path.exists():
            return None

        with open(metadata_path) as f:
            metadata = json.load(f)

        return EAOutput(
            strategy_id=metadata["strategy_id"],
            strategy_name=metadata["strategy_name"],
            version=metadata["version"],
            file_path=metadata["file_path"],
            generated_at=datetime.fromisoformat(metadata["generated_at"]),
            trd_snapshot=metadata.get("trd_snapshot", {}),
            status=metadata.get("status", "unknown"),
            ex5_path=metadata.get("ex5_path"),
            compile_status=metadata.get("compile_status"),
            compile_errors=metadata.get("compile_errors"),
            compile_attempts=metadata.get("compile_attempts", 0),
            compile_last_attempt=datetime.fromisoformat(metadata["compile_last_attempt"]) if metadata.get("compile_last_attempt") else None,
        )

    def get_all_versions(self, strategy_id: str) -> List[EAOutput]:
        """Get all versions of a strategy."""
        strategy_dir = self.strategies_dir / strategy_id / "versions"

        if not strategy_dir.exists():
            return []

        versions = []
        for v in strategy_dir.iterdir():
            if v.is_dir() and v.name.isdigit():
                ea = self.get_ea(strategy_id, int(v.name))
                if ea:
                    versions.append(ea)

        return sorted(versions, key=lambda x: x.version)

    def get_output_path(self, strategy_id: str) -> Path:
        """Get the output directory path for a strategy."""
        return self.strategies_dir / strategy_id

    def store_ex5(
        self,
        strategy_id: str,
        version: int,
        ex5_content: bytes,
    ) -> bool:
        """
        Store compiled .ex5 file alongside .mq5.

        Args:
            strategy_id: Strategy identifier
            version: Version number
            ex5_content: Binary content of .ex5 file

        Returns:
            True if stored successfully
        """
        try:
            ea = self.get_ea(strategy_id, version)
            if not ea:
                logger.error(f"EA not found: {strategy_id} v{version}")
                return False

            # Determine output path
            mq5_path = Path(ea.file_path)
            ex5_path = mq5_path.with_suffix(".ex5")

            # Write .ex5 file
            with open(ex5_path, "wb") as f:
                f.write(ex5_content)

            # Update metadata
            metadata_path = mq5_path.parent / "metadata.json"
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            metadata["ex5_path"] = str(ex5_path)
            metadata["status"] = "compiled"

            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Stored .ex5 for {strategy_id} v{version} at {ex5_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to store .ex5: {e}")
            return False

    def update_compile_status(
        self,
        strategy_id: str,
        version: int,
        compile_status: str,
        compile_errors: Optional[List[str]] = None,
        compile_attempts: int = 1,
    ) -> bool:
        """
        Update compile status in metadata.

        Args:
            strategy_id: Strategy identifier
            version: Version number
            compile_status: "pending", "success", or "failed"
            compile_errors: List of error messages
            compile_attempts: Number of compilation attempts

        Returns:
            True if updated successfully
        """
        try:
            ea = self.get_ea(strategy_id, version)
            if not ea:
                logger.error(f"EA not found: {strategy_id} v{version}")
                return False

            # Update metadata
            metadata_path = Path(ea.file_path).parent / "metadata.json"
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            metadata["compile_status"] = compile_status
            metadata["compile_errors"] = compile_errors
            metadata["compile_attempts"] = compile_attempts
            metadata["compile_last_attempt"] = datetime.now().isoformat()

            if compile_status == "success":
                metadata["status"] = "compiled"

            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Updated compile status for {strategy_id} v{version}: {compile_status}")
            return True

        except Exception as e:
            logger.error(f"Failed to update compile status: {e}")
            return False

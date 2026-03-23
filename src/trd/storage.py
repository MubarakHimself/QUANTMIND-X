"""
TRD Storage - Strategy Version Control for TRD Documents

Provides storage, retrieval, and versioning for TRD documents
with history tracking.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.trd.schema import TRDDocument

logger = logging.getLogger(__name__)

# Storage configuration
TRD_STORAGE_DIR = Path("./data/trd")
TRD_HISTORY_DIR = TRD_STORAGE_DIR / "history"


class TRDStorage:
    """
    Storage backend for TRD documents with version control.

    Provides:
    - CRUD operations for TRD documents
    - Version history tracking
    - Retrieval by strategy_id
    """

    def __init__(self, storage_dir: Optional[Path] = None):
        """
        Initialize TRD storage.

        Args:
            storage_dir: Optional custom storage directory
        """
        self.storage_dir = storage_dir or TRD_STORAGE_DIR
        self.history_dir = TRD_HISTORY_DIR

        # Ensure directories exist
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(parents=True, exist_ok=True)

    def _get_trd_path(self, strategy_id: str) -> Path:
        """Get file path for TRD document."""
        return self.storage_dir / f"{strategy_id}.json"

    def _get_history_path(self, strategy_id: str) -> Path:
        """Get directory path for TRD history."""
        return self.history_dir / strategy_id

    def save(self, trd: TRDDocument) -> str:
        """
        Save TRD document with versioning.

        Args:
            trd: TRD document to save

        Returns:
            strategy_id of saved document
        """
        strategy_id = trd.strategy_id

        # Save current version
        trd_path = self._get_trd_path(strategy_id)
        with open(trd_path, 'w') as f:
            json.dump(trd.to_dict(), f, indent=2, default=str)

        # Archive to history
        self._archive_to_history(trd)

        logger.info(f"TRD saved: {strategy_id}")
        return strategy_id

    def _archive_to_history(self, trd: TRDDocument) -> None:
        """
        Archive current version to history.

        Args:
            trd: TRD document to archive
        """
        history_path = self._get_history_path(trd.strategy_id)
        history_path.mkdir(parents=True, exist_ok=True)

        # Create versioned filename: v{version}_{timestamp}.json
        version = trd.version
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_file = history_path / f"v{version}_{timestamp}.json"

        with open(version_file, 'w') as f:
            json.dump(trd.to_dict(), f, indent=2, default=str)

    def load(self, strategy_id: str) -> Optional[TRDDocument]:
        """
        Load TRD document by strategy_id.

        Args:
            strategy_id: Strategy identifier

        Returns:
            TRD document or None if not found
        """
        trd_path = self._get_trd_path(strategy_id)

        if not trd_path.exists():
            logger.warning(f"TRD not found: {strategy_id}")
            return None

        try:
            with open(trd_path, 'r') as f:
                data = json.load(f)

            return TRDDocument.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load TRD {strategy_id}: {e}")
            return None

    def delete(self, strategy_id: str) -> bool:
        """
        Delete TRD document.

        Args:
            strategy_id: Strategy identifier

        Returns:
            True if deleted, False if not found
        """
        trd_path = self._get_trd_path(strategy_id)

        if not trd_path.exists():
            return False

        trd_path.unlink()

        # Note: We keep history even after deletion for audit purposes

        logger.info(f"TRD deleted: {strategy_id}")
        return True

    def list_all(self) -> List[Dict[str, Any]]:
        """
        List all TRD documents (summary only).

        Returns:
            List of TRD summaries
        """
        trds = []

        for trd_path in self.storage_dir.glob("*.json"):
            try:
                with open(trd_path, 'r') as f:
                    data = json.load(f)

                trds.append({
                    "strategy_id": data.get("strategy_id"),
                    "strategy_name": data.get("strategy_name"),
                    "version": data.get("version"),
                    "symbol": data.get("symbol"),
                    "timeframe": data.get("timeframe"),
                    "status": data.get("status", "active"),
                    "created_at": data.get("created_at"),
                    "updated_at": data.get("updated_at"),
                })
            except Exception as e:
                logger.error(f"Failed to read TRD {trd_path}: {e}")

        return trds

    def get_history(self, strategy_id: str) -> List[Dict[str, Any]]:
        """
        Get version history for a strategy.

        Args:
            strategy_id: Strategy identifier

        Returns:
            List of version metadata
        """
        history_path = self._get_history_path(strategy_id)

        if not history_path.exists():
            return []

        versions = []

        for version_file in sorted(history_path.glob("v*.json")):
            try:
                with open(version_file, 'r') as f:
                    data = json.load(f)

                versions.append({
                    "version": data.get("version"),
                    "strategy_id": data.get("strategy_id"),
                    "created_at": data.get("created_at"),
                    "author": data.get("author"),
                })
            except Exception as e:
                logger.error(f"Failed to read version {version_file}: {e}")

        return versions

    def get_latest_version(self, strategy_id: str) -> Optional[TRDDocument]:
        """
        Get the latest version of a TRD.

        Args:
            strategy_id: Strategy identifier

        Returns:
            Latest TRD document or None
        """
        return self.load(strategy_id)

    def create_new_version(
        self,
        strategy_id: str,
        updates: Dict[str, Any]
    ) -> Optional[TRDDocument]:
        """
        Create a new version of an existing TRD with updates.

        Args:
            strategy_id: Strategy identifier
            updates: Dictionary of fields to update

        Returns:
            New TRD document or None if original not found
        """
        # Load current
        current = self.load(strategy_id)
        if not current:
            return None

        # Apply updates
        for key, value in updates.items():
            if hasattr(current, key):
                setattr(current, key, value)

        # Increment version
        current.version += 1
        current.updated_at = datetime.now()

        # Save new version
        self.save(current)

        logger.info(f"Created new version {current.version} for {strategy_id}")
        return current


# Module-level convenience instance
_storage: Optional[TRDStorage] = None


def get_trd_storage() -> TRDStorage:
    """Get or create TRD storage instance."""
    global _storage
    if _storage is None:
        _storage = TRDStorage()
    return _storage


def save_trd(trd: TRDDocument) -> str:
    """Convenience function to save TRD."""
    return get_trd_storage().save(trd)


def load_trd(strategy_id: str) -> Optional[TRDDocument]:
    """Convenience function to load TRD."""
    return get_trd_storage().load(strategy_id)


def list_trds() -> List[Dict[str, Any]]:
    """Convenience function to list all TRDs."""
    return get_trd_storage().list_all()


def get_trd_history(strategy_id: str) -> List[Dict[str, Any]]:
    """Convenience function to get TRD history."""
    return get_trd_storage().get_history(strategy_id)
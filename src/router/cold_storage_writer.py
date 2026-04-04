"""
Cold Storage Writer
===================

Writes workflow output files (EOD reports, fortnight data) to cold storage.
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ColdStorageWriter:
    """
    Writes data files to cold storage.

    Cold storage path is configured via COLD_STORAGE_PATH environment variable.
    """

    def __init__(self, cold_storage_path: Optional[str] = None):
        self._cold_storage_path = cold_storage_path or os.environ.get(
            "COLD_STORAGE_PATH", "/data/cold_storage"
        )
        logger.info(f"ColdStorageWriter initialized with path: {self._cold_storage_path}")

    async def write(self, file_key: str, data: Dict[str, Any]) -> str:
        """
        Write data to cold storage.

        Args:
            file_key: Path relative to cold storage root (e.g., "eod_reports/2026-03-24_eod_report.json")
            data: Dictionary data to write

        Returns:
            Full path where data was written
        """
        import aiofiles
        import aiofiles.os

        # Ensure directory exists
        full_path = os.path.join(self._cold_storage_path, file_key)
        dir_path = os.path.dirname(full_path)

        try:
            await aiofiles.os.makedirs(dir_path, exist_ok=True)
        except Exception as e:
            logger.warning(f"Could not create directory {dir_path}: {e}")

        # Write data
        try:
            async with aiofiles.open(full_path, 'w') as f:
                await f.write(json.dumps(data, indent=2, default=str))

            logger.info(f"Written to cold storage: {full_path}")
            return full_path

        except Exception as e:
            logger.error(f"Failed to write to cold storage: {e}")
            raise

    async def read(self, file_key: str) -> Optional[Dict[str, Any]]:
        """
        Read data from cold storage.

        Args:
            file_key: Path relative to cold storage root

        Returns:
            Dictionary data or None if not found
        """
        import aiofiles

        full_path = os.path.join(self._cold_storage_path, file_key)

        try:
            async with aiofiles.open(full_path, 'r') as f:
                content = await f.read()
                return json.loads(content)

        except FileNotFoundError:
            logger.warning(f"File not found in cold storage: {full_path}")
            return None
        except Exception as e:
            logger.error(f"Failed to read from cold storage: {e}")
            return None

    def exists(self, file_key: str) -> bool:
        """Check if file exists in cold storage."""
        full_path = os.path.join(self._cold_storage_path, file_key)
        return os.path.exists(full_path)

    async def list_files(self, prefix: str = "") -> list[str]:
        """
        List all files in cold storage under a prefix.

        Args:
            prefix: Path prefix to filter files (e.g., "eod_reports/")

        Returns:
            List of file keys (relative paths)
        """
        import aiofiles.os

        search_path = os.path.join(self._cold_storage_path, prefix) if prefix else self._cold_storage_path

        try:
            files = []
            async for entry in aiofiles.os.scandir(search_path):
                if entry.is_file():
                    # Return relative path
                    rel_path = os.path.relpath(entry.path, self._cold_storage_path)
                    files.append(rel_path)
                elif entry.is_dir():
                    # Recursively list subdirectories
                    sub_prefix = os.path.join(prefix, os.path.basename(entry.path))
                    sub_files = await self.list_files(sub_prefix)
                    files.extend(sub_files)
            return sorted(files)
        except FileNotFoundError:
            logger.warning(f"Directory not found in cold storage: {search_path}")
            return []
        except Exception as e:
            logger.error(f"Failed to list files in cold storage: {e}")
            return []


# ============= Singleton Factory =============
_cold_storage_writer: Optional[ColdStorageWriter] = None


def get_cold_storage_writer() -> ColdStorageWriter:
    """Get singleton instance of ColdStorageWriter."""
    global _cold_storage_writer
    if _cold_storage_writer is None:
        _cold_storage_writer = ColdStorageWriter()
    return _cold_storage_writer

"""
Hybrid Disk Synchronization Layer for QuantMindX

Responsible for atomic file writes to MT5 risk_matrix.json with:
- Atomic file operations (no half-written reads)
- File locking (fcntl/msvcrt for cross-platform)
- Retry logic with exponential backoff
- JSON schema validation
- Wine path handling for MT5 on Linux
"""

import os
import json
import tempfile
import logging
import time
import fcntl
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class DiskSyncer:
    """
    Hybrid Disk Synchronization Layer for MT5 Risk Matrix

    Provides atomic, thread-safe file writes to risk_matrix.json
    for consumption by MQL5 Expert Advisors.

    Features:
    - Atomic writes using temp file + os.replace
    - File locking (fcntl on Unix, msvcrt on Windows)
    - Retry logic with exponential backoff
    - JSON schema validation
    - Wine path detection for Linux MT5 installations
    """

    def __init__(
        self,
        mt5_path: Optional[str] = None,
        max_retries: int = 5,
        initial_backoff: float = 1.0
    ):
        """
        Initialize DiskSyncer with MT5 path configuration.

        Args:
            mt5_path: Custom path to MT5 MQL5/Files directory.
                     If None, auto-detects Wine path.
            max_retries: Maximum retry attempts for file writes (default: 5)
            initial_backoff: Initial backoff in seconds (default: 1.0)
        """
        self.mt5_path = mt5_path or self._get_wine_mt5_path()
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff

        logger.info(f"DiskSyncer initialized with MT5 path: {self.mt5_path}")

    def _get_wine_mt5_path(self) -> str:
        """
        Detect and return Wine MT5 path for Linux installations.

        Returns:
            Path to MT5 MQL5/Files directory (e.g., ~/.wine/drive_c/...)
        """
        home = Path.home()
        wine_paths = [
            home / ".wine" / "drive_c" / "Program Files" / "MetaTrader 5" / "MQL5" / "Files",
            home / ".wine" / "drive_c" / "Program Files (x86)" / "MetaTrader 5" / "MQL5" / "Files",
            # Fallback for custom Wine installations
            home / ".wine" / "dosdevices" / "c:" / "Program Files" / "MetaTrader 5" / "MQL5" / "Files",
        ]

        for path in wine_paths:
            if path.exists():
                logger.debug(f"Detected Wine MT5 path: {path}")
                return str(path)

        # Default to first path if none exist (will create on demand)
        default_path = wine_paths[0]
        logger.debug(f"Using default Wine MT5 path: {default_path}")
        return str(default_path)

    def _validate_risk_matrix(self, risk_matrix: Dict[str, Any]) -> bool:
        """
        Validate risk_matrix JSON structure before write.

        Schema: {"symbol": {"multiplier": float, "timestamp": int}}

        Args:
            risk_matrix: Dictionary to validate

        Returns:
            True if valid

        Raises:
            ValueError: If validation fails
        """
        if not risk_matrix:
            raise ValueError("Risk matrix cannot be empty")

        for symbol, data in risk_matrix.items():
            if not isinstance(data, dict):
                raise ValueError(f"Invalid data for symbol {symbol}: must be dict")

            if "multiplier" not in data:
                raise ValueError(f"Missing 'multiplier' field for symbol {symbol}")

            if "timestamp" not in data:
                raise ValueError(f"Missing 'timestamp' field for symbol {symbol}")

            # Validate multiplier is numeric
            try:
                float(data["multiplier"])
            except (ValueError, TypeError):
                raise ValueError(f"Invalid 'multiplier' value for symbol {symbol}: must be numeric")

            # Validate timestamp is integer
            if not isinstance(data["timestamp"], int):
                raise ValueError(f"Invalid 'timestamp' value for symbol {symbol}: must be integer")

        logger.debug("Risk matrix validation passed")
        return True

    def _acquire_lock(self, file_handle):
        """
        Acquire exclusive lock on file handle.

        Uses fcntl on Unix/Linux, msvcrt on Windows.

        Args:
            file_handle: Open file handle to lock

        Raises:
            IOError: If lock cannot be acquired
        """
        try:
            if os.name == 'posix':  # Unix/Linux
                fcntl.flock(file_handle.fileno(), fcntl.LOCK_EX)
                logger.debug("Acquired fcntl lock (Unix/Linux)")
            elif os.name == 'nt':  # Windows
                import msvcrt
                msvcrt.locking(file_handle.fileno(), msvcrt.LK_LOCK, 1)
                logger.debug("Acquired msvcrt lock (Windows)")
            else:
                logger.warning(f"Unsupported platform {os.name}, no locking applied")
        except Exception as e:
            logger.error(f"Failed to acquire file lock: {e}")
            raise IOError(f"Could not acquire file lock: {e}")

    def _release_lock(self, file_handle):
        """
        Release exclusive lock on file handle.

        Args:
            file_handle: Open file handle to unlock
        """
        try:
            if os.name == 'posix':  # Unix/Linux
                fcntl.flock(file_handle.fileno(), fcntl.LOCK_UN)
                logger.debug("Released fcntl lock")
            elif os.name == 'nt':  # Windows
                import msvcrt
                msvcrt.locking(file_handle.fileno(), msvcrt.LK_UNLCK, 1)
                logger.debug("Released msvcrt lock")
        except Exception as e:
            logger.warning(f"Failed to release file lock: {e}")

    def _atomic_write(self, file_path: str, data: Dict[str, Any]):
        """
        Perform atomic file write using temp file + os.replace.

        This ensures MT5 EAs never see half-written files.

        Args:
            file_path: Target file path
            data: Data to write (will be JSON serialized)

        Raises:
            IOError: If write fails after retries
        """
        # Create parent directory if it doesn't exist
        parent_dir = Path(file_path).parent
        parent_dir.mkdir(parents=True, exist_ok=True)

        # Write to temp file in same directory (ensures same filesystem)
        with tempfile.NamedTemporaryFile(
            mode='w',
            dir=parent_dir,
            suffix='.tmp',
            delete=False
        ) as tmp_file:
            tmp_path = tmp_file.name

            try:
                # Acquire lock on temp file
                self._acquire_lock(tmp_file)

                # Write JSON data
                json.dump(data, tmp_file, indent=2)
                tmp_file.flush()
                os.fsync(tmp_file.fileno())  # Force write to disk

                # Release lock
                self._release_lock(tmp_file)

            except Exception as e:
                # Clean up temp file on error
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise IOError(f"Failed to write temp file: {e}")

        # Atomic rename (os.replace is atomic on POSIX)
        try:
            os.replace(tmp_path, file_path)
            logger.debug(f"Atomic write successful: {file_path}")
        except Exception as e:
            # Clean up temp file if rename fails
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise IOError(f"Failed to rename temp file: {e}")

    def sync_risk_matrix(
        self,
        risk_matrix: Dict[str, Any],
        target_file: Optional[str] = None
    ):
        """
        Sync risk matrix to MT5 with retry logic.

        Performs validation, atomic write, and retry with exponential backoff.

        Args:
            risk_matrix: Risk data to sync
                         Format: {"symbol": {"multiplier": float, "timestamp": int}}
            target_file: Optional custom target file path.
                        If None, uses mt5_path/risk_matrix.json

        Raises:
            ValueError: If JSON validation fails
            IOError: If write fails after all retries
        """
        # Validate JSON structure
        self._validate_risk_matrix(risk_matrix)

        # Determine target file path
        if target_file is None:
            target_file = os.path.join(self.mt5_path, "risk_matrix.json")

        # Retry logic with exponential backoff
        backoff = self.initial_backoff
        last_error = None

        for attempt in range(self.max_retries):
            try:
                self._atomic_write(target_file, risk_matrix)
                logger.info(f"Successfully synced risk matrix to {target_file}")
                return

            except IOError as e:
                last_error = e
                logger.warning(
                    f"Write attempt {attempt + 1}/{self.max_retries} failed: {e}. "
                    f"Retrying in {backoff}s..."
                )

                # Exponential backoff
                if attempt < self.max_retries - 1:
                    time.sleep(backoff)
                    backoff *= 2

        # All retries exhausted
        logger.error(
            f"Failed to sync risk matrix after {self.max_retries} attempts. "
            f"Last error: {last_error}"
        )
        raise IOError(
            f"Failed to sync risk matrix after {self.max_retries} attempts: {last_error}"
        )

    def read_risk_matrix(self, source_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Read risk matrix from MT5 file.

        Args:
            source_file: Optional custom source file path.
                        If None, uses mt5_path/risk_matrix.json

        Returns:
            Risk matrix dictionary

        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If file contains invalid JSON
        """
        if source_file is None:
            source_file = os.path.join(self.mt5_path, "risk_matrix.json")

        if not os.path.exists(source_file):
            raise FileNotFoundError(f"Risk matrix file not found: {source_file}")

        with open(source_file, 'r') as f:
            self._acquire_lock(f)
            try:
                data = json.load(f)
                return data
            finally:
                self._release_lock(f)

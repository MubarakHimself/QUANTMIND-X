"""
Cold storage sync service for log retention.

Handles nightly synchronization of logs from hot storage to cold storage
on Contabo with integrity verification (checksums).
"""

import hashlib
import logging
import os
import shutil
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)


def calculate_file_checksum(file_path: str, algorithm: str = "sha256") -> str:
    """
    Calculate checksum of a file using specified algorithm.

    Args:
        file_path: Path to the file
        algorithm: Hash algorithm (sha256, md5)

    Returns:
        Hexadecimal checksum string
    """
    hash_obj = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()


def verify_file_integrity(file_path: str, expected_checksum: str, algorithm: str = "sha256") -> bool:
    """
    Verify file integrity against expected checksum.

    Args:
        file_path: Path to the file
        expected_checksum: Expected checksum value
        algorithm: Hash algorithm used

    Returns:
        True if checksum matches, False otherwise
    """
    actual_checksum = calculate_file_checksum(file_path, algorithm)
    return actual_checksum == expected_checksum


class ColdStorageSync:
    """
    Cold storage synchronization service.

    Handles:
    - Identifying logs older than hot retention threshold
    - Syncing logs to cold storage (Contabo)
    - Generating and storing checksums for verification
    - Cleaning up old logs after successful sync
    """

    def __init__(
        self,
        hot_retention_days: int = 90,
        cold_storage_path: Optional[str] = None,
        checksum_algorithm: str = "sha256",
        log_source_path: Optional[str] = None
    ):
        """
        Initialize the cold storage sync service.

        Args:
            hot_retention_days: Days to keep in hot storage (default: 90)
            cold_storage_path: Path to cold storage (Contabo)
            checksum_algorithm: Algorithm for integrity verification
            log_source_path: Path to source logs (default: from env or "/data/logs")
        """
        self.hot_retention_days = hot_retention_days
        self.cold_storage_path = cold_storage_path or os.getenv("COLD_STORAGE_PATH", "/mnt/cold-storage/logs")
        self.checksum_algorithm = checksum_algorithm
        self.log_source_path = log_source_path or os.getenv("LOG_SOURCE_PATH", "/data/logs")

    def get_logs_to_sync(self) -> list:
        """
        Get list of log files older than hot retention period.

        Returns:
            List of file paths to sync
        """
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.hot_retention_days)
        logs_to_sync = []

        if not os.path.exists(self.log_source_path):
            logger.warning(f"Log source path does not exist: {self.log_source_path}")
            return []

        for root, dirs, files in os.walk(self.log_source_path):
            for filename in files:
                if filename.endswith(".log") or filename.endswith(".json"):
                    file_path = os.path.join(root, filename)
                    try:
                        file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path), tz=timezone.utc)
                        if file_mtime < cutoff_date:
                            logs_to_sync.append({
                                "path": file_path,
                                "size": os.path.getsize(file_path),
                                "modified": file_mtime.isoformat()
                            })
                    except OSError as e:
                        logger.error(f"Error checking file {file_path}: {e}")

        return logs_to_sync

    def sync_logs(self, logs: list) -> Dict[str, Any]:
        """
        Sync logs to cold storage with integrity verification.

        Args:
            logs: List of log files to sync

        Returns:
            Sync result with statistics
        """
        if not logs:
            return {
                "success": True,
                "message": "No logs to sync",
                "synced_count": 0,
                "failed_count": 0
            }

        # Ensure cold storage path exists
        os.makedirs(self.cold_storage_path, exist_ok=True)

        synced_count = 0
        failed_count = 0
        results = []

        for log in logs:
            source_path = log["path"]
            try:
                # Create relative path structure in cold storage
                rel_path = os.path.relpath(source_path, self.log_source_path)
                dest_path = os.path.join(self.cold_storage_path, rel_path)
                checksum_path = dest_path + f".{self.checksum_algorithm}"

                # Check if file already exists in cold storage with a checksum
                if os.path.exists(dest_path) and os.path.exists(checksum_path):
                    # Read stored checksum
                    with open(checksum_path, "r") as f:
                        stored_checksum = f.read().strip()

                    # Verify existing file against stored checksum
                    if not verify_file_integrity(dest_path, stored_checksum, self.checksum_algorithm):
                        # File was tampered with - flag as corrupted and remove
                        os.remove(dest_path)
                        os.remove(checksum_path)
                        failed_count += 1
                        results.append({
                            "file": source_path,
                            "status": "failed",
                            "error": "Tamper detected: cold storage file checksum mismatch"
                        })
                        logger.error(f"Tamper detected for {dest_path}")
                        continue

                # Calculate checksum of source file
                original_checksum = calculate_file_checksum(source_path, self.checksum_algorithm)

                # Create destination directory
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)

                # Copy file to cold storage
                shutil.copy2(source_path, dest_path)

                # Verify integrity
                if verify_file_integrity(dest_path, original_checksum, self.checksum_algorithm):
                    # Write checksum file
                    with open(checksum_path, "w") as f:
                        f.write(original_checksum)

                    synced_count += 1
                    results.append({
                        "file": source_path,
                        "status": "synced",
                        "checksum": original_checksum
                    })
                else:
                    # Integrity check failed - remove corrupted file
                    os.remove(dest_path)
                    failed_count += 1
                    results.append({
                        "file": source_path,
                        "status": "failed",
                        "error": "Integrity verification failed"
                    })
                    logger.error(f"Integrity check failed for {source_path}")

            except Exception as e:
                failed_count += 1
                results.append({
                    "file": source_path,
                    "status": "failed",
                    "error": str(e)
                })
                logger.error(f"Error syncing {source_path}: {e}")

        return {
            "success": failed_count == 0,
            "synced_count": synced_count,
            "failed_count": failed_count,
            "results": results,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    def run_nightly_sync(self) -> Dict[str, Any]:
        """
        Run the nightly log sync job.

        Returns:
            Sync result with statistics
        """
        logger.info("Starting nightly cold storage sync")

        # Get logs to sync
        logs = self.get_logs_to_sync()
        logger.info(f"Found {len(logs)} logs to sync")

        # Sync logs
        result = self.sync_logs(logs)

        logger.info(f"Nightly sync completed: {result['synced_count']} synced, {result['failed_count']} failed")
        return result

    def get_sync_status(self) -> Dict[str, Any]:
        """
        Get current sync status and statistics.

        Returns:
            Status information
        """
        logs = self.get_logs_to_sync()

        # Calculate storage usage
        cold_storage_size = 0
        if os.path.exists(self.cold_storage_path):
            for root, dirs, files in os.walk(self.cold_storage_path):
                for f in files:
                    try:
                        cold_storage_size += os.path.getsize(os.path.join(root, f))
                    except OSError:
                        pass

        return {
            "logs_pending_sync": len(logs),
            "cold_storage_size_bytes": cold_storage_size,
            "cold_storage_path": self.cold_storage_path,
            "hot_retention_days": self.hot_retention_days,
            "checksum_algorithm": self.checksum_algorithm
        }


# Module-level function for use in API
def run_cold_storage_sync(
    hot_retention_days: int = 90,
    cold_storage_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run cold storage sync job.

    Args:
        hot_retention_days: Days to keep in hot storage
        cold_storage_path: Path to cold storage

    Returns:
        Sync result
    """
    sync = ColdStorageSync(
        hot_retention_days=hot_retention_days,
        cold_storage_path=cold_storage_path
    )
    return sync.run_nightly_sync()


def get_cold_storage_status() -> Dict[str, Any]:
    """
    Get cold storage sync status.

    Returns:
        Status information
    """
    sync = ColdStorageSync()
    return sync.get_sync_status()


class ColdStorageScheduler:
    """
    Scheduler for automatic cold storage sync.

    Runs nightly sync based on configured cron schedule.
    """

    def __init__(self, sync_cron: str = "0 2 * * *"):
        """
        Initialize scheduler.

        Args:
            sync_cron: Cron expression for sync schedule (default: 2 AM daily)
        """
        self.sync_cron = sync_cron
        self.scheduler = None

    def start(self) -> bool:
        """Start the scheduler. Returns True if successful."""
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            from apscheduler.triggers.cron import CronTrigger

            self.scheduler = BackgroundScheduler()
            # Parse cron and create trigger
            parts = self.sync_cron.split()
            if len(parts) == 5:
                trigger = CronTrigger(
                    minute=parts[0],
                    hour=parts[1],
                    day=parts[2],
                    month=parts[3],
                    day_of_week=parts[4]
                )
                self.scheduler.add_job(
                    run_cold_storage_sync,
                    trigger,
                    id='cold_storage_sync',
                    replace_existing=True
                )
                self.scheduler.start()
                logger.info(f"Cold storage scheduler started with cron: {self.sync_cron}")
                return True
            else:
                logger.error(f"Invalid cron expression: {self.sync_cron}")
                return False
        except ImportError:
            logger.warning("APScheduler not installed - automatic sync disabled")
            return False
        except Exception as e:
            logger.error(f"Failed to start cold storage scheduler: {e}")
            return False

    def stop(self):
        """Stop the scheduler."""
        if self.scheduler:
            self.scheduler.shutdown()
            logger.info("Cold storage scheduler stopped")


# Global scheduler instance
_scheduler: Optional[ColdStorageScheduler] = None


def start_cold_storage_scheduler(sync_cron: str = "0 2 * * *") -> bool:
    """
    Start the global cold storage scheduler.

    Args:
        sync_cron: Cron expression for sync schedule

    Returns:
        True if started successfully
    """
    global _scheduler
    if _scheduler is None:
        _scheduler = ColdStorageScheduler(sync_cron)
    return _scheduler.start()


def stop_cold_storage_scheduler():
    """Stop the global cold storage scheduler."""
    global _scheduler
    if _scheduler:
        _scheduler.stop()
        _scheduler = None
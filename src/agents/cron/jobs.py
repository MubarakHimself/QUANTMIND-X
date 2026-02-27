"""
Built-in Cron Jobs for QuantMindX Agents

This module provides commonly-used cron jobs that can be registered
with the scheduler for automated maintenance tasks.

Available jobs:
- MemoryConsolidationJob: Periodically consolidate and prune memories
- SessionCleanupJob: Clean up old session files
- EmbeddingSyncJob: Sync embeddings for new/modified files
- HealthCheckJob: Periodic system health checks
"""

import asyncio
import logging
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MemoryConsolidationJob:
    """
    Job for consolidating and pruning agent memories.

    Runs periodically to:
    - Remove old/irrelevant memories
    - Consolidate similar memories
    - Update memory statistics
    """

    def __init__(
        self,
        memory_path: str = "./data/memories",
        retention_days: int = 30,
        consolidation_interval_hours: int = 24,
    ):
        """
        Initialize the memory consolidation job.

        Args:
            memory_path: Path to memories directory
            retention_days: Days to keep memories before pruning
            consolidation_interval_hours: Hours between consolidations
        """
        self.memory_path = Path(memory_path)
        self.retention_days = retention_days
        self.consolidation_interval_hours = consolidation_interval_hours

    async def execute(self) -> Dict[str, Any]:
        """Execute memory consolidation."""
        logger.info("Starting memory consolidation")

        stats = {
            "memories_pruned": 0,
            "memories_consolidated": 0,
            "errors": [],
        }

        try:
            # Ensure memory directory exists
            if not self.memory_path.exists():
                logger.warning(f"Memory path {self.memory_path} does not exist")
                return stats

            # Get cutoff date
            cutoff = datetime.now(timezone.utc) - timedelta(days=self.retention_days)

            # Process memory files
            for file_path in self.memory_path.rglob("*.json"):
                try:
                    # Check file modification time
                    mtime = datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc)

                    if mtime < cutoff:
                        # Old file - check if should be pruned
                        await self._process_old_memory(file_path, stats)

                except Exception as e:
                    stats["errors"].append(str(e))
                    logger.error(f"Error processing {file_path}: {e}")

            logger.info(f"Memory consolidation complete: {stats}")

        except Exception as e:
            logger.error(f"Memory consolidation failed: {e}")
            stats["errors"].append(str(e))

        return stats

    async def _process_old_memory(self, file_path: Path, stats: Dict[str, Any]):
        """Process an old memory file."""
        try:
            import json

            with open(file_path, "r") as f:
                data = json.load(f)

            # Check if memory is marked important
            if data.get("metadata", {}).get("important", False):
                # Important - consolidate
                await self._consolidate_memory(file_path, data)
                stats["memories_consolidated"] += 1
            else:
                # Not important - prune
                file_path.unlink()
                stats["memories_pruned"] += 1

        except Exception as e:
            stats["errors"].append(f"{file_path}: {e}")

    async def _consolidate_memory(self, file_path: Path, data: Dict[str, Any]):
        """Consolidate a memory file."""
        # Update consolidated timestamp
        data["metadata"]["consolidated_at"] = datetime.now(timezone.utc).isoformat()

        # Write back
        with open(file_path, "w") as f:
            import json
            json.dump(data, f, indent=2)


class SessionCleanupJob:
    """
    Job for cleaning up old session files.

    Runs periodically to:
    - Remove old session files
    - Archive important sessions
    - Clean up temporary files
    """

    def __init__(
        self,
        session_path: str = "./data/sessions",
        retention_days: int = 7,
        archive_path: Optional[str] = None,
    ):
        """
        Initialize the session cleanup job.

        Args:
            session_path: Path to sessions directory
            retention_days: Days to keep sessions before cleanup
            archive_path: Path for archived sessions (optional)
        """
        self.session_path = Path(session_path)
        self.retention_days = retention_days
        self.archive_path = Path(archive_path) if archive_path else None

    async def execute(self) -> Dict[str, Any]:
        """Execute session cleanup."""
        logger.info("Starting session cleanup")

        stats = {
            "sessions_cleaned": 0,
            "sessions_archived": 0,
            "errors": [],
        }

        try:
            if not self.session_path.exists():
                logger.warning(f"Session path {self.session_path} does not exist")
                return stats

            cutoff = datetime.now(timezone.utc) - timedelta(days=self.retention_days)

            for file_path in self.session_path.rglob("*.json"):
                try:
                    mtime = datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc)

                    if mtime < cutoff:
                        await self._process_old_session(file_path, stats)

                except Exception as e:
                    stats["errors"].append(str(e))
                    logger.error(f"Error processing {file_path}: {e}")

            logger.info(f"Session cleanup complete: {stats}")

        except Exception as e:
            logger.error(f"Session cleanup failed: {e}")
            stats["errors"].append(str(e))

        return stats

    async def _process_old_session(self, file_path: Path, stats: Dict[str, Any]):
        """Process an old session file."""
        try:
            import json

            with open(file_path, "r") as f:
                data = json.load(f)

            # Check if session is marked important
            if data.get("metadata", {}).get("archive", False):
                # Archive if archive path is set
                if self.archive_path:
                    self.archive_path.mkdir(parents=True, exist_ok=True)
                    archive_file = self.archive_path / file_path.name
                    file_path.rename(archive_file)
                    stats["sessions_archived"] += 1
                else:
                    file_path.unlink()
                    stats["sessions_cleaned"] += 1
            else:
                # Delete
                file_path.unlink()
                stats["sessions_cleaned"] += 1

        except Exception as e:
            stats["errors"].append(f"{file_path}: {e}")


class EmbeddingSyncJob:
    """
    Job for syncing embeddings for new/modified files.

    Runs periodically to:
    - Generate embeddings for new files
    - Update embeddings for modified files
    - Remove embeddings for deleted files
    """

    def __init__(
        self,
        source_path: str = "./src",
        embedding_path: str = "./data/embeddings",
        file_patterns: List[str] = None,
    ):
        """
        Initialize the embedding sync job.

        Args:
            source_path: Path to source files
            embedding_path: Path to embedding cache
            file_patterns: File patterns to include (default: .py, .md, .txt)
        """
        self.source_path = Path(source_path)
        self.embedding_path = Path(embedding_path)
        self.file_patterns = file_patterns or ["*.py", "*.md", "*.txt"]

    async def execute(self) -> Dict[str, Any]:
        """Execute embedding sync."""
        logger.info("Starting embedding sync")

        stats = {
            "files_processed": 0,
            "embeddings_created": 0,
            "embeddings_updated": 0,
            "errors": [],
        }

        try:
            if not self.source_path.exists():
                logger.warning(f"Source path {self.source_path} does not exist")
                return stats

            # Find all matching files
            files = []
            for pattern in self.file_patterns:
                files.extend(self.source_path.rglob(pattern))

            for file_path in files:
                try:
                    await self._process_file(file_path, stats)

                except Exception as e:
                    stats["errors"].append(str(e))
                    logger.error(f"Error processing {file_path}: {e}")

            logger.info(f"Embedding sync complete: {stats}")

        except Exception as e:
            logger.error(f"Embedding sync failed: {e}")
            stats["errors"].append(str(e))

        return stats

    async def _process_file(self, file_path: Path, stats: Dict[str, Any]):
        """Process a single file for embedding."""
        # Get file hash
        file_hash = self._get_file_hash(file_path)

        # Check if embedding exists and is up to date
        embedding_file = self.embedding_path / f"{file_path.name}.json"

        if embedding_file.exists():
            # Check if needs update
            try:
                import json
                with open(embedding_file, "r") as f:
                    data = json.load(f)

                if data.get("hash") == file_hash:
                    return  # Up to date

                # Update needed
                await self._create_embedding(file_path, file_hash)
                stats["embeddings_updated"] += 1

            except Exception:
                # Corrupted - recreate
                await self._create_embedding(file_path, file_hash)
                stats["embeddings_updated"] += 1
        else:
            # Create new embedding
            await self._create_embedding(file_path, file_hash)
            stats["embeddings_created"] += 1

        stats["files_processed"] += 1

    async def _create_embedding(self, file_path: Path, file_hash: str):
        """Create embedding for a file."""
        self.embedding_path.mkdir(parents=True, exist_ok=True)

        # Read file content
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Create embedding data
        embedding_data = {
            "path": str(file_path),
            "hash": file_hash,
            "size": len(content),
            "created_at": datetime.now(timezone.utc).isoformat(),
            # Actual embedding would be generated by embedding service
            "embedding": None,
        }

        # Save
        embedding_file = self.embedding_path / f"{file_path.name}.json"
        import json
        with open(embedding_file, "w") as f:
            json.dump(embedding_data, f, indent=2)

    def _get_file_hash(self, file_path: Path) -> str:
        """Get hash of file contents."""
        import hashlib

        with open(file_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]


class HealthCheckJob:
    """
    Job for periodic system health checks.

    Runs periodically to:
    - Check disk space
    - Check memory usage
    - Check agent status
    - Check connectivity
    """

    def __init__(
        self,
        checks: List[str] = None,
        alert_threshold_percent: float = 90.0,
    ):
        """
        Initialize the health check job.

        Args:
            checks: List of checks to run (default: all)
            alert_threshold_percent: Threshold for alerts
        """
        self.checks = checks or ["disk", "memory", "agents", "connectivity"]
        self.alert_threshold_percent = alert_threshold_percent

    async def execute(self) -> Dict[str, Any]:
        """Execute health checks."""
        logger.info("Starting health checks")

        results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": {},
            "alerts": [],
            "status": "healthy",
        }

        for check in self.checks:
            try:
                result = await self._run_check(check)
                results["checks"][check] = result

                if result.get("status") == "warning":
                    results["status"] = "degraded"
                    results["alerts"].append(f"{check}: {result.get('message')}")
                elif result.get("status") == "critical":
                    results["status"] = "critical"
                    results["alerts"].append(f"{check}: {result.get('message')}")

            except Exception as e:
                results["checks"][check] = {"status": "error", "error": str(e)}
                logger.error(f"Health check {check} failed: {e}")

        logger.info(f"Health checks complete: {results['status']}")
        return results

    async def _run_check(self, check: str) -> Dict[str, Any]:
        """Run a single health check."""
        if check == "disk":
            return await self._check_disk()
        elif check == "memory":
            return await self._check_memory()
        elif check == "agents":
            return await self._check_agents()
        elif check == "connectivity":
            return await self._check_connectivity()
        else:
            return {"status": "unknown", "message": f"Unknown check: {check}"}

    async def _check_disk(self) -> Dict[str, Any]:
        """Check disk space."""
        import shutil

        usage = shutil.disk_usage("/")
        percent_used = (usage.used / usage.total) * 100

        status = "ok"
        message = f"Disk usage: {percent_used:.1f}%"

        if percent_used > self.alert_threshold_percent:
            status = "critical"
        elif percent_used > self.alert_threshold_percent - 10:
            status = "warning"

        return {
            "status": status,
            "message": message,
            "total_gb": usage.total / (1024**3),
            "used_gb": usage.used / (1024**3),
            "free_gb": usage.free / (1024**3),
            "percent_used": percent_used,
        }

    async def _check_memory(self) -> Dict[str, Any]:
        """Check memory usage."""
        import psutil

        mem = psutil.virtual_memory()
        percent_used = mem.percent

        status = "ok"
        message = f"Memory usage: {percent_used:.1f}%"

        if percent_used > self.alert_threshold_percent:
            status = "critical"
        elif percent_used > self.alert_threshold_percent - 10:
            status = "warning"

        return {
            "status": status,
            "message": message,
            "total_gb": mem.total / (1024**3),
            "available_gb": mem.available / (1024**3),
            "percent_used": percent_used,
        }

    async def _check_agents(self) -> Dict[str, Any]:
        """Check agent status."""
        # This would integrate with the actual agent system
        return {
            "status": "ok",
            "message": "Agents operational",
            "agents_running": 0,
        }

    async def _check_connectivity(self) -> Dict[str, Any]:
        """Check network connectivity."""
        import socket

        try:
            # Try to connect to a reliable host
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return {
                "status": "ok",
                "message": "Network connectivity OK",
            }
        except Exception:
            return {
                "status": "warning",
                "message": "Network connectivity issues",
            }


# Helper function to register all built-in jobs
def register_builtin_jobs(scheduler, config: Dict[str, Any] = None) -> List[str]:
    """
    Register all built-in jobs with the scheduler.

    Args:
        scheduler: The scheduler instance
        config: Configuration dict for job parameters

    Returns:
        List of registered job IDs
    """
    config = config or {}

    job_ids = []

    # Memory consolidation (daily at 3 AM)
    memory_job = MemoryConsolidationJob(
        memory_path=config.get("memory_path", "./data/memories"),
        retention_days=config.get("memory_retention_days", 30),
    )

    scheduler.add_job(
        job_id="memory-consolidation",
        handler=memory_job.execute,
        name="Memory Consolidation",
        cron="0 3 * * *",
        description="Consolidate and prune old memories",
    )
    job_ids.append("memory-consolidation")

    # Session cleanup (daily at 4 AM)
    session_job = SessionCleanupJob(
        session_path=config.get("session_path", "./data/sessions"),
        retention_days=config.get("session_retention_days", 7),
        archive_path=config.get("session_archive_path"),
    )

    scheduler.add_job(
        job_id="session-cleanup",
        handler=session_job.execute,
        name="Session Cleanup",
        cron="0 4 * * *",
        description="Clean up old session files",
    )
    job_ids.append("session-cleanup")

    # Embedding sync (every 6 hours)
    embedding_job = EmbeddingSyncJob(
        source_path=config.get("source_path", "./src"),
        embedding_path=config.get("embedding_path", "./data/embeddings"),
    )

    scheduler.add_job(
        job_id="embedding-sync",
        handler=embedding_job.execute,
        name="Embedding Sync",
        cron="0 */6 * * *",
        description="Sync embeddings for source files",
    )
    job_ids.append("embedding-sync")

    # Health check (hourly)
    health_job = HealthCheckJob(
        checks=["disk", "memory"],
        alert_threshold_percent=config.get("health_alert_threshold", 90.0),
    )

    scheduler.add_job(
        job_id="health-check",
        handler=health_job.execute,
        name="Health Check",
        cron="0 * * * *",
        description="Periodic system health checks",
    )
    job_ids.append("health-check")

    logger.info(f"Registered {len(job_ids)} built-in jobs")

    return job_ids

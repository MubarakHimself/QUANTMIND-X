"""
GitHub EA Sync Scheduler

Provides scheduled synchronization of Expert Advisors from GitHub repositories.
Uses APScheduler to run sync jobs at configurable intervals.

**Validates: Property 19: Scheduled EA Sync**
"""

import logging
import os
from datetime import datetime
from typing import Optional, Dict, Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger

from src.integrations.github_ea_sync import GitHubEASync, get_github_ea_sync
from src.database.models import get_db_session

logger = logging.getLogger(__name__)


class GitHubEAScheduler:
    """
    Scheduler for automated GitHub EA synchronization.
    
    Provides:
    - Scheduled sync at configurable intervals
    - Manual sync trigger
    - Sync status monitoring
    """
    
    def __init__(
        self,
        repo_url: str,
        library_path: str = "/data/ea-library",
        branch: str = "main",
        sync_interval_hours: int = 24
    ):
        """
        Initialize the GitHub EA Scheduler.
        
        Args:
            repo_url: GitHub repository URL
            library_path: Local path for cloned repository
            branch: Branch to sync
            sync_interval_hours: Hours between scheduled syncs (default: 24)
        """
        self.repo_url = repo_url
        self.library_path = library_path
        self.branch = branch
        self.sync_interval_hours = sync_interval_hours
        
        self.sync_service = GitHubEASync(repo_url, library_path, branch)
        self.scheduler = AsyncIOScheduler()
        
        self._is_running = False
        self._last_sync_result: Optional[Dict[str, Any]] = None
        self._sync_count = 0
        self._error_count = 0

    @property
    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._is_running

    @property
    def last_sync_result(self) -> Optional[Dict[str, Any]]:
        """Get last sync result."""
        return self._last_sync_result

    @property
    def sync_count(self) -> int:
        """Get total number of syncs performed."""
        return self._sync_count

    def start(self) -> bool:
        """
        Start the scheduler.
        
        Returns:
            True if scheduler started successfully
        """
        try:
            if self._is_running:
                logger.warning("Scheduler is already running")
                return True
            
            # Add scheduled job
            self.scheduler.add_job(
                self._scheduled_sync,
                IntervalTrigger(hours=self.sync_interval_hours),
                id='github_ea_sync',
                name='GitHub EA Sync',
                replace_existing=True
            )
            
            # Start scheduler
            self.scheduler.start()
            self._is_running = True
            
            logger.info(
                f"GitHub EA Scheduler started with {self.sync_interval_hours}h interval "
                f"for {self.repo_url}"
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to start scheduler: {e}")
            return False

    def stop(self) -> bool:
        """
        Stop the scheduler.
        
        Returns:
            True if scheduler stopped successfully
        """
        try:
            if not self._is_running:
                return True
            
            self.scheduler.shutdown(wait=False)
            self._is_running = False
            
            logger.info("GitHub EA Scheduler stopped")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop scheduler: {e}")
            return False

    async def _scheduled_sync(self) -> Dict[str, Any]:
        """
        Perform scheduled sync job.
        
        Returns:
            Sync result dictionary
        """
        logger.info("Starting scheduled GitHub EA sync...")
        
        try:
            with get_db_session() as db:
                result = self.sync_service.full_sync(db)
            
            self._last_sync_result = result
            self._sync_count += 1
            
            if result.get('sync_status') == 'success':
                logger.info(
                    f"Scheduled sync completed: {result.get('eas_found', 0)} EAs found, "
                    f"{result.get('eas_new', 0)} new, {result.get('eas_updated', 0)} updated"
                )
            else:
                self._error_count += 1
                logger.error(f"Scheduled sync failed: {result}")
            
            return result
            
        except Exception as e:
            self._error_count += 1
            error_msg = f"Scheduled sync error: {str(e)}"
            logger.error(error_msg)
            return {'sync_status': 'error', 'error': error_msg}

    async def manual_sync(self) -> Dict[str, Any]:
        """
        Trigger a manual sync.
        
        Returns:
            Sync result dictionary
        """
        logger.info("Triggering manual GitHub EA sync...")
        
        try:
            with get_db_session() as db:
                result = self.sync_service.full_sync(db)
            
            self._last_sync_result = result
            self._sync_count += 1
            
            return result
            
        except Exception as e:
            error_msg = f"Manual sync error: {str(e)}"
            logger.error(error_msg)
            return {'sync_status': 'error', 'error': error_msg}

    def get_status(self) -> Dict[str, Any]:
        """
        Get scheduler status.
        
        Returns:
            Dictionary with scheduler status information
        """
        jobs = self.scheduler.get_jobs() if self._is_running else []
        
        next_run = None
        for job in jobs:
            if job.id == 'github_ea_sync':
                next_run = job.next_run_time.isoformat() if job.next_run_time else None
                break
        
        return {
            'is_running': self._is_running,
            'repo_url': self.repo_url,
            'branch': self.branch,
            'sync_interval_hours': self.sync_interval_hours,
            'sync_count': self._sync_count,
            'error_count': self._error_count,
            'last_sync_time': (
                self.sync_service.last_sync_time.isoformat() 
                if self.sync_service.last_sync_time else None
            ),
            'last_commit_hash': self.sync_service.last_commit_hash,
            'next_scheduled_run': next_run,
            'last_sync_result': self._last_sync_result
        }

    def update_interval(self, hours: int) -> bool:
        """
        Update the sync interval.
        
        Args:
            hours: New interval in hours
            
        Returns:
            True if interval updated successfully
        """
        try:
            self.sync_interval_hours = hours
            
            if self._is_running:
                # Remove existing job and add new one
                self.scheduler.remove_job('github_ea_sync')
                self.scheduler.add_job(
                    self._scheduled_sync,
                    IntervalTrigger(hours=hours),
                    id='github_ea_sync',
                    name='GitHub EA Sync',
                    replace_existing=True
                )
            
            logger.info(f"Sync interval updated to {hours} hours")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update interval: {e}")
            return False


# Global scheduler instance
_scheduler: Optional[GitHubEAScheduler] = None


def get_scheduler() -> Optional[GitHubEAScheduler]:
    """
    Get the global scheduler instance.
    
    Returns:
        GitHubEAScheduler instance or None if not initialized
    """
    return _scheduler


def initialize_scheduler() -> GitHubEAScheduler:
    """
    Initialize the global scheduler from environment variables.
    
    Returns:
        Initialized GitHubEAScheduler instance
    """
    global _scheduler
    
    repo_url = os.getenv('GITHUB_EA_REPO_URL', '')
    library_path = os.getenv('EA_LIBRARY_PATH', '/data/ea-library')
    branch = os.getenv('GITHUB_EA_BRANCH', 'main')
    sync_interval_hours = int(os.getenv('GITHUB_EA_SYNC_INTERVAL_HOURS', '24'))
    
    if not repo_url:
        raise ValueError("GITHUB_EA_REPO_URL environment variable not set")
    
    _scheduler = GitHubEAScheduler(
        repo_url=repo_url,
        library_path=library_path,
        branch=branch,
        sync_interval_hours=sync_interval_hours
    )
    
    return _scheduler


def start_scheduler() -> bool:
    """
    Initialize and start the global scheduler.
    
    Returns:
        True if scheduler started successfully
    """
    try:
        scheduler = initialize_scheduler()
        return scheduler.start()
    except Exception as e:
        logger.error(f"Failed to start scheduler: {e}")
        return False


def stop_scheduler() -> bool:
    """
    Stop the global scheduler.
    
    Returns:
        True if scheduler stopped successfully
    """
    global _scheduler
    
    if _scheduler:
        result = _scheduler.stop()
        _scheduler = None
        return result
    
    return True


if __name__ == '__main__':
    import asyncio
    
    async def test_scheduler():
        """Test the scheduler."""
        # Get configuration from environment
        repo_url = os.getenv('GITHUB_EA_REPO_URL')
        if not repo_url:
            print("Set GITHUB_EA_REPO_URL environment variable")
            return
        
        scheduler = GitHubEAScheduler(
            repo_url=repo_url,
            sync_interval_hours=1  # 1 hour for testing
        )
        
        # Test manual sync
        print("Testing manual sync...")
        result = await scheduler.manual_sync()
        print(f"Sync result: {result}")
        
        # Get status
        status = scheduler.get_status()
        print(f"\nScheduler status: {status}")
        
        # Start scheduler
        print("\nStarting scheduler...")
        scheduler.start()
        
        # Let it run briefly
        await asyncio.sleep(2)
        
        # Check status again
        status = scheduler.get_status()
        print(f"After start status: {status}")
        
        # Stop
        scheduler.stop()
        print("Scheduler stopped")
    
    asyncio.run(test_scheduler())
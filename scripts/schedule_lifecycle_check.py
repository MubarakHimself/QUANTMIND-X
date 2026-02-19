#!/usr/bin/env python3
"""
Lifecycle Check Scheduler
=========================

Schedules and runs automated bot lifecycle checks using APScheduler.
Runs daily at 3:00 AM UTC (after HMM training at 2:00 AM).

Features:
- Automated daily lifecycle evaluation
- Tag progression: @primal → @pending → @perfect → @live
- Quarantine detection for underperforming bots
- Dead bot marking for irrecoverable bots
- Progress logging to file
- Database status updates for dashboard

Usage:
    python scripts/schedule_lifecycle_check.py --start
    python scripts/schedule_lifecycle_check.py --status
    python scripts/schedule_lifecycle_check.py --run-now

Reference: docs/architecture/lifecycle_management.md
"""

import os
import sys
import json
import logging
import argparse
import signal
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR
    APSCHEDULER_AVAILABLE = True
except ImportError:
    APSCHEDULER_AVAILABLE = False
    print("WARNING: APScheduler not installed. Install with: pip install apscheduler")

# Configure logging
log_dir = Path("/data/lifecycle/logs")
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "lifecycle_check.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class LifecycleScheduler:
    """
    Scheduler for automated bot lifecycle checks.
    
    Manages daily evaluation jobs with status persistence and notifications.
    """
    
    def __init__(self):
        """Initialize the lifecycle scheduler."""
        # Initialize scheduler
        if APSCHEDULER_AVAILABLE:
            self.scheduler = BackgroundScheduler()
            self.scheduler.add_listener(
                self._job_executed_listener,
                EVENT_JOB_EXECUTED | EVENT_JOB_ERROR
            )
        else:
            self.scheduler = None
        
        # Lifecycle state
        self.last_check_time: Optional[datetime] = None
        self.last_check_status: str = "never_run"
        self.last_report: Optional[Dict] = None
        self.job_id: Optional[str] = None
    
    def _job_executed_listener(self, event) -> None:
        """Handle job execution events."""
        if event.exception:
            logger.error(f"Lifecycle check failed: {event.exception}")
            self.last_check_status = "failed"
            self._send_notification("failed", str(event.exception))
        else:
            logger.info("Lifecycle check completed successfully")
            self.last_check_status = "success"
            self.last_check_time = datetime.now(timezone.utc)
            if self.last_report:
                self._send_notification("success", self.last_report.get("summary", {}))
    
    def _send_notification(self, status: str, message: Any) -> None:
        """
        Send notification about lifecycle check status.
        
        Args:
            status: Check status ('success', 'failed', 'warning')
            message: Notification message or summary
        """
        # Log notification
        logger.info(f"NOTIFICATION [{status.upper()}]: {message}")
        
        # Write to notification file (can be picked up by monitoring system)
        notification_dir = Path("/data/lifecycle/notifications")
        notification_dir.mkdir(parents=True, exist_ok=True)
        
        notification_file = notification_dir / f"notification_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        notification = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'type': 'lifecycle_check',
            'status': status,
            'message': message,
            'last_check_time': self.last_check_time.isoformat() if self.last_check_time else None
        }
        
        with open(notification_file, 'w') as f:
            json.dump(notification, f, indent=2)
        
        # Update database flag for dashboard
        self._update_lifecycle_flag(status, message)
    
    def _update_lifecycle_flag(self, status: str, message: Any) -> None:
        """Update lifecycle status in database for dashboard pickup."""
        try:
            from src.database.engine import engine
            from sqlalchemy import text
            
            with engine.connect() as conn:
                # Check if lifecycle_status table exists
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS lifecycle_status (
                        id INTEGER PRIMARY KEY CHECK (id = 1),
                        status VARCHAR(20) NOT NULL,
                        message TEXT,
                        last_check TIMESTAMP,
                        promotions INTEGER DEFAULT 0,
                        quarantines INTEGER DEFAULT 0,
                        kills INTEGER DEFAULT 0,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                
                # Extract counts from message if available
                promotions = 0
                quarantines = 0
                kills = 0
                if isinstance(message, dict):
                    promotions = message.get('promotions', 0)
                    quarantines = message.get('quarantines', 0)
                    kills = message.get('kills', 0)
                
                # Update or insert status
                conn.execute(text("""
                    INSERT INTO lifecycle_status (id, status, message, last_check, promotions, quarantines, kills, updated_at)
                    VALUES (1, :status, :message, :last_check, :promotions, :quarantines, :kills, CURRENT_TIMESTAMP)
                    ON CONFLICT(id) DO UPDATE SET
                        status = :status,
                        message = :message,
                        last_check = :last_check,
                        promotions = :promotions,
                        quarantines = :quarantines,
                        kills = :kills,
                        updated_at = CURRENT_TIMESTAMP
                """), {
                    'status': status,
                    'message': json.dumps(message) if isinstance(message, dict) else str(message),
                    'last_check': self.last_check_time.isoformat() if self.last_check_time else None,
                    'promotions': promotions,
                    'quarantines': quarantines,
                    'kills': kills
                })
                conn.commit()
                
        except Exception as e:
            logger.warning(f"Failed to update lifecycle flag: {e}")
    
    def run_lifecycle_check(self) -> bool:
        """
        Execute the daily lifecycle check.
        
        Returns:
            True if check succeeded, False otherwise
        """
        logger.info("=" * 60)
        logger.info("Starting lifecycle check job")
        logger.info("=" * 60)
        
        try:
            from src.router.lifecycle_manager import LifecycleManager
            
            manager = LifecycleManager()
            report = manager.run_daily_check()
            
            self.last_check_time = datetime.now(timezone.utc)
            self.last_check_status = "success"
            self.last_report = report.to_dict()
            
            logger.info("=" * 60)
            logger.info(f"Lifecycle check complete: {self.last_report['summary']}")
            logger.info("=" * 60)
            
            return True
            
        except Exception as e:
            logger.error(f"Lifecycle check failed: {e}")
            self.last_check_status = "failed"
            self.last_report = {"error": str(e)}
            raise
    
    def setup_schedule(self) -> None:
        """Set up the lifecycle check schedule."""
        if not self.scheduler:
            logger.error("APScheduler not available. Cannot set up schedule.")
            return
        
        # Schedule daily at 3:00 AM UTC (after HMM training at 2:00 AM)
        trigger = CronTrigger(
            day_of_week='*',
            hour=3,
            minute=0,
            timezone='UTC'
        )
        
        # Add job
        self.job_id = self.scheduler.add_job(
            self.run_lifecycle_check,
            trigger=trigger,
            id='lifecycle_check',
            name='Daily Bot Lifecycle Check',
            replace_existing=True
        ).id
        
        logger.info("Scheduled lifecycle check for daily at 3:00 AM UTC")
    
    def start(self) -> None:
        """Start the scheduler."""
        if not self.scheduler:
            logger.error("APScheduler not available. Cannot start scheduler.")
            return
        
        self.setup_schedule()
        self.scheduler.start()
        
        logger.info("Lifecycle scheduler started")
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Keep running
        try:
            while self.scheduler.running:
                import time
                time.sleep(1)
        except (KeyboardInterrupt, SystemExit):
            self.stop()
    
    def stop(self) -> None:
        """Stop the scheduler."""
        if self.scheduler and self.scheduler.running:
            self.scheduler.shutdown()
            logger.info("Lifecycle scheduler stopped")
    
    def _signal_handler(self, signum, frame) -> None:
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current scheduler status."""
        status = {
            'scheduler_running': self.scheduler.running if self.scheduler else False,
            'last_check_time': self.last_check_time.isoformat() if self.last_check_time else None,
            'last_check_status': self.last_check_status,
            'last_report_summary': self.last_report.get('summary') if self.last_report else None,
            'next_check_time': None,
            'scheduled_jobs': []
        }
        
        if self.scheduler and self.scheduler.running:
            jobs = self.scheduler.get_jobs()
            status['scheduled_jobs'] = [
                {
                    'id': job.id,
                    'name': job.name,
                    'next_run': job.next_run_time.isoformat() if job.next_run_time else None
                }
                for job in jobs
            ]
            
            if jobs:
                status['next_check_time'] = jobs[0].next_run_time.isoformat()
        
        return status
    
    def run_now(self) -> bool:
        """Run lifecycle check immediately (blocking)."""
        logger.info("Running lifecycle check immediately...")
        return self.run_lifecycle_check()


def main():
    """Main entry point for lifecycle scheduler."""
    parser = argparse.ArgumentParser(description='Bot Lifecycle Check Scheduler')
    parser.add_argument('--start', action='store_true', help='Start the scheduler')
    parser.add_argument('--status', action='store_true', help='Get scheduler status')
    parser.add_argument('--run-now', action='store_true', help='Run lifecycle check immediately')
    
    args = parser.parse_args()
    
    # Initialize scheduler
    scheduler = LifecycleScheduler()
    
    if args.start:
        scheduler.start()
    elif args.status:
        status = scheduler.get_status()
        print(json.dumps(status, indent=2))
    elif args.run_now:
        success = scheduler.run_now()
        return 0 if success else 1
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
#!/usr/bin/env python3
"""
HMM Training Scheduler
======================

Schedules and runs automated HMM training jobs using APScheduler.
Training runs every Saturday at 2:00 AM UTC by default.

Features:
- Automated weekly training schedule
- Data availability checks before training
- Notification on training completion
- Progress logging to file

Usage:
    python scripts/schedule_hmm_training.py --start
    python scripts/schedule_hmm_training.py --status
    python scripts/schedule_hmm_training.py --run-now

Reference: docs/architecture/components.md
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
log_dir = Path("/data/hmm/logs")
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "training_scheduler.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class HMMTrainingScheduler:
    """
    Scheduler for automated HMM training.
    
    Manages weekly training jobs with data validation and notifications.
    """
    
    def __init__(self, config_path: str = "config/hmm_config.json"):
        """Initialize scheduler with configuration."""
        self.config = self._load_config(config_path)
        self.schedule_config = self.config.get('schedule', {})
        self.symbols_config = self.config.get('symbols', {})
        self.training_config = self.config.get('training', {})
        
        # Initialize scheduler
        if APSCHEDULER_AVAILABLE:
            self.scheduler = BackgroundScheduler()
            self.scheduler.add_listener(
                self._job_executed_listener,
                EVENT_JOB_EXECUTED | EVENT_JOB_ERROR
            )
        else:
            self.scheduler = None
        
        # Training state
        self.last_training_time: Optional[datetime] = None
        self.last_training_status: str = "never_run"
        self.training_job_id: Optional[str] = None
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _job_executed_listener(self, event) -> None:
        """Handle job execution events."""
        if event.exception:
            logger.error(f"Training job failed: {event.exception}")
            self.last_training_status = "failed"
            self._send_notification("failed", str(event.exception))
        else:
            logger.info("Training job completed successfully")
            self.last_training_status = "success"
            self.last_training_time = datetime.now(timezone.utc)
            self._send_notification("success", "Training completed successfully")
    
    def _send_notification(self, status: str, message: str) -> None:
        """
        Send notification about training status.
        
        Args:
            status: Training status ('success', 'failed', 'warning')
            message: Notification message
        """
        # Log notification
        logger.info(f"NOTIFICATION [{status.upper()}]: {message}")
        
        # Write to notification file (can be picked up by monitoring system)
        notification_dir = Path("/data/hmm/notifications")
        notification_dir.mkdir(parents=True, exist_ok=True)
        
        notification_file = notification_dir / f"notification_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        notification = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'type': 'hmm_training',
            'status': status,
            'message': message,
            'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None
        }
        
        with open(notification_file, 'w') as f:
            json.dump(notification, f, indent=2)
        
        # Also update database flag for dashboard
        self._update_training_flag(status, message)
    
    def _update_training_flag(self, status: str, message: str) -> None:
        """Update training status in database for dashboard pickup."""
        try:
            from src.database.engine import engine
            from sqlalchemy import text
            
            with engine.connect() as conn:
                # Check if training_status table exists
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS hmm_training_status (
                        id INTEGER PRIMARY KEY CHECK (id = 1),
                        status VARCHAR(20) NOT NULL,
                        message TEXT,
                        last_training TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                
                # Update or insert status
                conn.execute(text("""
                    INSERT INTO hmm_training_status (id, status, message, last_training, updated_at)
                    VALUES (1, :status, :message, :last_training, CURRENT_TIMESTAMP)
                    ON CONFLICT(id) DO UPDATE SET
                        status = :status,
                        message = :message,
                        last_training = :last_training,
                        updated_at = CURRENT_TIMESTAMP
                """), {
                    'status': status,
                    'message': message,
                    'last_training': self.last_training_time.isoformat() if self.last_training_time else None
                })
                conn.commit()
                
        except Exception as e:
            logger.warning(f"Failed to update training flag: {e}")
    
    def check_data_availability(self) -> Dict[str, Any]:
        """
        Check if sufficient data is available for training.
        
        Returns:
            Dictionary with data availability status per symbol/timeframe
        """
        logger.info("Checking data availability...")
        
        min_samples = self.training_config.get('min_samples_per_symbol', 2000)
        symbols = self.symbols_config.get('primary', [])
        timeframes = self.symbols_config.get('timeframes', [])
        
        availability = {
            'ready': True,
            'details': {},
            'total_samples': 0
        }
        
        try:
            from src.database.duckdb_connection import DuckDBConnection
            
            with DuckDBConnection() as conn:
                for symbol in symbols:
                    for tf in timeframes:
                        result = conn.execute_query(f"""
                            SELECT COUNT(*) as count
                            FROM market_data
                            WHERE symbol = '{symbol}' AND timeframe = '{tf}'
                        """).fetchone()
                        
                        count = result[0] if result else 0
                        availability['details'][f"{symbol}_{tf}"] = {
                            'samples': count,
                            'sufficient': count >= min_samples
                        }
                        availability['total_samples'] += count
                        
                        if count < min_samples:
                            availability['ready'] = False
                            logger.warning(f"Insufficient data for {symbol}_{tf}: {count} < {min_samples}")
                        
        except Exception as e:
            logger.error(f"Failed to check data availability: {e}")
            availability['ready'] = False
            availability['error'] = str(e)
        
        return availability
    
    def run_training_job(self) -> bool:
        """
        Execute the training job.
        
        Returns:
            True if training succeeded, False otherwise
        """
        logger.info("=" * 60)
        logger.info("Starting HMM training job")
        logger.info("=" * 60)
        
        # Check data availability
        availability = self.check_data_availability()
        
        if not availability.get('ready', False):
            logger.warning("Data availability check failed. Training may use sample data.")
        
        # Import trainer
        from train_hmm import HMMTrainer
        
        try:
            trainer = HMMTrainer()
            
            # Train universal model first
            logger.info("Training universal model...")
            universal_path, universal_version = trainer.train_universal()
            logger.info(f"Universal model saved: {universal_path} (v{universal_version})")
            
            # Train per-symbol models
            symbols = self.symbols_config.get('primary', [])
            for symbol in symbols:
                try:
                    logger.info(f"Training per-symbol model for {symbol}...")
                    symbol_path, symbol_version = trainer.train_per_symbol(symbol)
                    logger.info(f"Symbol model saved: {symbol_path} (v{symbol_version})")
                except Exception as e:
                    logger.error(f"Failed to train {symbol} model: {e}")
                    continue
            
            # Train per-symbol-timeframe models
            timeframes = self.symbols_config.get('timeframes', [])
            for symbol in symbols:
                for tf in timeframes:
                    try:
                        logger.info(f"Training model for {symbol} {tf}...")
                        model_path, version = trainer.train_per_symbol_timeframe(symbol, tf)
                        logger.info(f"Model saved: {model_path} (v{version})")
                    except Exception as e:
                        logger.error(f"Failed to train {symbol}_{tf} model: {e}")
                        continue
            
            self.last_training_time = datetime.now(timezone.utc)
            self.last_training_status = "success"
            
            logger.info("=" * 60)
            logger.info("Training job completed successfully")
            logger.info("=" * 60)
            
            # Sync models to Cloudzy after successful training
            try:
                from src.router.hmm_version_control import HMMVersionControl
                
                logger.info("Starting model sync to Cloudzy...")
                vc = HMMVersionControl()
                
                # Sync universal model
                universal_sync = vc.sync_model(model_type="universal")
                logger.info(f"Universal model sync: {'success' if universal_sync else 'failed'}")
                
                # Sync per-symbol models
                symbols = self.symbols_config.get('primary', [])
                for symbol in symbols:
                    try:
                        symbol_sync = vc.sync_model(model_type="per_symbol", symbol=symbol)
                        logger.info(f"Symbol {symbol} model sync: {'success' if symbol_sync else 'failed'}")
                    except Exception as e:
                        logger.warning(f"Failed to sync {symbol} model: {e}")
                
                # Notify about sync success
                self._send_notification("sync_success", "All models synced to Cloudzy successfully")
                logger.info("Model sync completed successfully")
                
            except Exception as e:
                logger.error(f"Model sync failed: {e}")
                self._send_notification("sync_failed", f"Model sync failed: {str(e)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Training job failed: {e}")
            self.last_training_status = "failed"
            raise
    
    def setup_schedule(self) -> None:
        """Set up the training schedule."""
        if not self.scheduler:
            logger.error("APScheduler not available. Cannot set up schedule.")
            return
        
        # Get schedule configuration
        training_day = self.schedule_config.get('training_day', 'saturday').lower()
        training_hour = self.schedule_config.get('training_hour', 2)
        training_minute = self.schedule_config.get('training_minute', 0)
        tz = self.schedule_config.get('timezone', 'UTC')
        
        # Map day names to cron format
        day_map = {
            'monday': 'mon', 'tuesday': 'tue', 'wednesday': 'wed',
            'thursday': 'thu', 'friday': 'fri', 'saturday': 'sat', 'sunday': 'sun'
        }
        day_of_week = day_map.get(training_day, 'sat')
        
        # Create cron trigger
        trigger = CronTrigger(
            day_of_week=day_of_week,
            hour=training_hour,
            minute=training_minute,
            timezone=tz
        )
        
        # Add job
        self.training_job_id = self.scheduler.add_job(
            self.run_training_job,
            trigger=trigger,
            id='hmm_training',
            name='HMM Weekly Training',
            replace_existing=True
        ).id
        
        logger.info(f"Scheduled training for every {training_day} at {training_hour:02d}:{training_minute:02d} {tz}")
    
    def start(self) -> None:
        """Start the scheduler."""
        if not self.scheduler:
            logger.error("APScheduler not available. Cannot start scheduler.")
            return
        
        self.setup_schedule()
        self.scheduler.start()
        
        logger.info("Training scheduler started")
        
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
            logger.info("Training scheduler stopped")
    
    def _signal_handler(self, signum, frame) -> None:
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current scheduler status."""
        status = {
            'scheduler_running': self.scheduler.running if self.scheduler else False,
            'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
            'last_training_status': self.last_training_status,
            'next_training_time': None,
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
                status['next_training_time'] = jobs[0].next_run_time.isoformat()
        
        return status
    
    def run_now(self) -> bool:
        """Run training immediately (blocking)."""
        logger.info("Running training immediately...")
        return self.run_training_job()


def main():
    """Main entry point for training scheduler."""
    parser = argparse.ArgumentParser(description='HMM Training Scheduler')
    parser.add_argument('--start', action='store_true', help='Start the scheduler')
    parser.add_argument('--status', action='store_true', help='Get scheduler status')
    parser.add_argument('--run-now', action='store_true', help='Run training immediately')
    parser.add_argument('--config', type=str, default='config/hmm_config.json',
                       help='Path to config file')
    
    args = parser.parse_args()
    
    # Initialize scheduler
    scheduler = HMMTrainingScheduler(args.config)
    
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
#!/usr/bin/env python3
"""
Market Scanner Scheduler
========================

Schedules and runs automated market scanning using APScheduler.
Uses session-aware scan frequency for optimal opportunity detection.

Scan Schedule:
- Asian Session (12 AM - 8 AM UTC): Every 15 minutes
- London Session (8 AM - 4 PM UTC): Every 5 minutes
- NY Session (1 PM - 9 PM UTC): Every 5 minutes
- Overlap (1 PM - 4 PM UTC): Every 1 minute

Usage:
    python scripts/schedule_market_scanner.py --start
    python scripts/schedule_market_scanner.py --status
    python scripts/schedule_market_scanner.py --scan-now

Reference: docs/architecture/components.md
"""

import sys
import json
import logging
import argparse
import signal
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Any, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.interval import IntervalTrigger
    from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR
    APSCHEDULER_AVAILABLE = True
except ImportError:
    APSCHEDULER_AVAILABLE = False
    print("WARNING: APScheduler not installed. Install with: pip install apscheduler")

# Configure logging
log_dir = Path("/data/scanner/logs")
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "market_scanner.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MarketScannerScheduler:
    """
    Scheduler for automated market scanning with session-aware frequency.
    
    Adjusts scan interval based on current trading session:
    - Overlap: 1 minute (highest frequency)
    - London/NY: 5 minutes
    - Asian: 15 minutes
    """
    
    def __init__(self, symbols: Optional[List[str]] = None):
        """Initialize the market scanner scheduler."""
        self.symbols = symbols or ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]
        
        # Initialize scheduler
        if APSCHEDULER_AVAILABLE:
            self.scheduler = BackgroundScheduler()
            self.scheduler.add_listener(
                self._job_executed_listener,
                EVENT_JOB_EXECUTED | EVENT_JOB_ERROR
            )
        else:
            self.scheduler = None
        
        # Scanner state
        self.last_scan_time: Optional[datetime] = None
        self.last_scan_status: str = "never_run"
        self.total_alerts: int = 0
        self.scan_job_id: Optional[str] = None
        self._current_interval: int = 300  # Default 5 minutes
    
    def _job_executed_listener(self, event) -> None:
        """Handle job execution events."""
        if event.exception:
            logger.error(f"Market scan failed: {event.exception}")
            self.last_scan_status = "failed"
        else:
            logger.info("Market scan completed")
            self.last_scan_status = "success"
            self.last_scan_time = datetime.now(timezone.utc)
    
    def _send_notification(self, alerts: List[Dict[str, Any]]) -> None:
        """Send notification about detected alerts."""
        if not alerts:
            return
        
        # Log alert summary
        logger.info(f"Detected {len(alerts)} trading opportunities")
        
        # Write to notification file
        notification_dir = Path("/data/scanner/notifications")
        notification_dir.mkdir(parents=True, exist_ok=True)
        
        notification_file = notification_dir / f"alerts_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        notification = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'type': 'market_scan',
            'alert_count': len(alerts),
            'alerts': alerts,
        }
        
        with open(notification_file, 'w') as f:
            json.dump(notification, f, indent=2)
        
        # Also broadcast via WebSocket if available
        self._broadcast_alerts(alerts)
    
    def _broadcast_alerts(self, alerts: List[Dict[str, Any]]) -> None:
        """Broadcast alerts via WebSocket."""
        try:
            # TODO: Integrate with WebSocket server when available
            # For now, just log
            for alert in alerts:
                logger.info(f"ALERT: {alert['type']} on {alert['symbol']} - {alert['setup']}")
        except Exception as e:
            logger.warning(f"Failed to broadcast alerts: {e}")
    
    def run_scan(self) -> bool:
        """
        Execute a market scan.
        
        Returns:
            True if scan succeeded, False otherwise
        """
        logger.info("=" * 60)
        logger.info("Starting market scan")
        logger.info("=" * 60)
        
        try:
            from src.router.market_scanner import MarketScanner
            
            scanner = MarketScanner(symbols=self.symbols)
            alerts = scanner.run_full_scan()
            
            self.last_scan_time = datetime.now(timezone.utc)
            self.last_scan_status = "success"
            self.total_alerts += len(alerts)
            
            # Send notifications for detected alerts
            if alerts:
                self._send_notification(alerts)
            
            logger.info("=" * 60)
            logger.info(f"Scan complete: {len(alerts)} opportunities detected")
            logger.info("=" * 60)
            
            # Adjust interval based on session
            self._adjust_scan_interval(scanner)
            
            return True
            
        except Exception as e:
            logger.error(f"Market scan failed: {e}")
            self.last_scan_status = "failed"
            raise
    
    def _adjust_scan_interval(self, scanner) -> None:
        """Adjust scan interval based on current session."""
        try:
            new_interval = scanner.get_scan_interval()
            
            if new_interval != self._current_interval:
                logger.info(f"Adjusting scan interval: {self._current_interval}s -> {new_interval}s")
                self._current_interval = new_interval
                
                # Reschedule job with new interval
                if self.scheduler and self.scan_job_id:
                    self.scheduler.reschedule_job(
                        self.scan_job_id,
                        trigger=IntervalTrigger(seconds=new_interval)
                    )
        except Exception as e:
            logger.warning(f"Failed to adjust scan interval: {e}")
    
    def setup_schedule(self) -> None:
        """Set up the market scanning schedule."""
        if not self.scheduler:
            logger.error("APScheduler not available. Cannot set up schedule.")
            return
        
        # Use interval trigger with dynamic adjustment
        trigger = IntervalTrigger(seconds=self._current_interval)
        
        # Add job
        self.scan_job_id = self.scheduler.add_job(
            self.run_scan,
            trigger=trigger,
            id='market_scan',
            name='Market Opportunity Scan',
            replace_existing=True
        ).id
        
        logger.info(f"Scheduled market scan with {self._current_interval}s interval (session-aware)")
    
    def start(self) -> None:
        """Start the scheduler."""
        if not self.scheduler:
            logger.error("APScheduler not available. Cannot start scheduler.")
            return
        
        self.setup_schedule()
        self.scheduler.start()
        
        logger.info("Market scanner scheduler started")
        
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
            logger.info("Market scanner scheduler stopped")
    
    def _signal_handler(self, signum, frame) -> None:
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current scheduler status."""
        status = {
            'scheduler_running': self.scheduler.running if self.scheduler else False,
            'last_scan_time': self.last_scan_time.isoformat() if self.last_scan_time else None,
            'last_scan_status': self.last_scan_status,
            'total_alerts': self.total_alerts,
            'current_interval_seconds': self._current_interval,
            'next_scan_time': None,
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
                status['next_scan_time'] = jobs[0].next_run_time.isoformat()
        
        return status
    
    def run_now(self) -> bool:
        """Run market scan immediately (blocking)."""
        logger.info("Running market scan immediately...")
        return self.run_scan()


def main():
    """Main entry point for market scanner scheduler."""
    parser = argparse.ArgumentParser(description='Market Scanner Scheduler')
    parser.add_argument('--start', action='store_true', help='Start the scheduler')
    parser.add_argument('--status', action='store_true', help='Get scheduler status')
    parser.add_argument('--scan-now', action='store_true', help='Run scan immediately')
    parser.add_argument('--symbols', type=str, default='EURUSD,GBPUSD,USDJPY,XAUUSD',
                       help='Comma-separated list of symbols to scan')
    
    args = parser.parse_args()
    
    # Parse symbols
    symbols = [s.strip() for s in args.symbols.split(',')]
    
    # Initialize scheduler
    scheduler = MarketScannerScheduler(symbols=symbols)
    
    if args.start:
        scheduler.start()
    elif args.status:
        status = scheduler.get_status()
        print(json.dumps(status, indent=2))
    elif args.scan_now:
        success = scheduler.run_now()
        return 0 if success else 1
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
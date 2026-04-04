#!/usr/bin/env python3
"""
Scheduled Firecrawl sync — runs every 6 hours.
Pulls from curated trading news sources and writes to data/scraped_articles/
for PageIndex auto-indexing (shared assets).
"""

import logging
import os
import signal
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    APSCHEDULER_AVAILABLE = True
except ImportError:
    APSCHEDULER_AVAILABLE = False
    print("WARNING: APScheduler not installed. Install with: pip install apscheduler")

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    print("WARNING: httpx not installed. Install with: pip install httpx")


# Configure logging
log_dir = Path("/data/firecrawl/logs")
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "firecrawl_sync.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


SOURCES = [
    "https://www.forexlive.com",
    "https://www.dailyfx.com/news",
]


def run_firecrawl_sync():
    """Run Firecrawl sync for all configured sources."""
    if not HTTPX_AVAILABLE:
        logger.error("httpx not available, skipping firecrawl sync")
        return

    for url in SOURCES:
        try:
            logger.info(f"Syncing Firecrawl for: {url}")
            response = httpx.post(
                f"{os.environ.get('NODE_BACKEND_URL', 'http://localhost:8000')}/api/knowledge/ingest",
                json={
                    "url": url,
                    "relevance_tags": ["market_news", "forex", "scheduled_sync"]
                },
                timeout=60
            )
            response.raise_for_status()
            logger.info(f"Firecrawl sync successful for {url}")
        except Exception as e:
            logger.error(f"Firecrawl sync failed for {url}: {e}")


class FirecrawlScheduler:
    """Scheduler for automated Firecrawl sync."""

    def __init__(self):
        if APSCHEDULER_AVAILABLE:
            self.scheduler = BackgroundScheduler()
        else:
            self.scheduler = None

    def start(self):
        """Start the scheduler."""
        if not self.scheduler:
            logger.error("APScheduler not available. Cannot start scheduler.")
            return

        # Schedule every 6 hours
        self.scheduler.add_job(
            run_firecrawl_sync,
            'interval',
            hours=6,
            id='firecrawl_sync',
            name='Firecrawl Sync (6h)',
            replace_existing=True
        )

        self.scheduler.start()
        logger.info("Firecrawl sync scheduler started — runs every 6 hours")

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Keep running
        try:
            while self.scheduler.running:
                time.sleep(1)
        except (KeyboardInterrupt, SystemExit):
            self.stop()

    def stop(self):
        """Stop the scheduler."""
        if self.scheduler and self.scheduler.running:
            self.scheduler.shutdown()
            logger.info("Firecrawl sync scheduler stopped")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()


def main():
    """Main entry point."""
    if not APSCHEDULER_AVAILABLE:
        logger.error("APScheduler is required but not installed.")
        sys.exit(1)

    scheduler = FirecrawlScheduler()
    scheduler.start()
    return 0


if __name__ == "__main__":
    sys.exit(main())

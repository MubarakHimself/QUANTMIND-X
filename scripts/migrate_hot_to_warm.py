#!/usr/bin/env python3
"""
HOT to WARM Migration Script
=============================

Migrates tick data from HOT tier (PostgreSQL) to WARM tier (DuckDB).
Runs every hour via cron.

Usage:
    python scripts/migrate_hot_to_warm.py
    python scripts/migrate_hot_to_warm.py --dry-run

Environment:
    HOT_DB_URL: PostgreSQL connection string for HOT tier
    WARM_DB_PATH: Path to DuckDB file for WARM tier
"""

import os
import sys
import json
import logging
import argparse
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
from decimal import Decimal

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("migrate_hot_to_warm")

# Prometheus metrics
try:
    from prometheus_client import Counter, Gauge, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not available - metrics disabled")

# Metrics
if PROMETHEUS_AVAILABLE:
    MIGRATION_ROWS_TOTAL = Counter(
        'quantmind_migration_rows_total',
        'Total rows migrated',
        ['direction']
    )
    MIGRATION_DURATION = Gauge(
        'quantmind_migration_duration_seconds',
        'Migration duration in seconds',
        ['direction']
    )


def get_hot_connection():
    """Create connection to HOT tier PostgreSQL."""
    from sqlalchemy import create_engine

    # Prefer CLOUDZY_HOT_DB_URL for Cloudzy deployments, fall back to HOT_DB_URL
    hot_db_url = os.environ.get("CLOUDZY_HOT_DB_URL") or os.environ.get("HOT_DB_URL")
    if not hot_db_url:
        raise ValueError("CLOUDZY_HOT_DB_URL or HOT_DB_URL not set")

    return create_engine(hot_db_url)


def fetch_old_ticks(engine, batch_size: int = 10000) -> List[Dict]:
    """
    Fetch ticks older than 1 hour from HOT tier.
    
    Args:
        engine: SQLAlchemy engine
        batch_size: Number of rows per batch
        
    Returns:
        List of tick dictionaries
    """
    from sqlalchemy import text
    
    cutoff_time = datetime.now(timezone.utc) - timedelta(hours=1)
    
    query = text("""
        SELECT 
            symbol,
            timestamp,
            bid,
            ask,
            volume
        FROM tick_cache
        WHERE timestamp < :cutoff_time
        ORDER BY timestamp ASC
        LIMIT :batch_size
    """)
    
    with engine.connect() as conn:
        result = conn.execute(query, {
            "cutoff_time": cutoff_time,
            "batch_size": batch_size
        })
        
        rows = []
        for row in result:
            rows.append({
                'symbol': row[0],
                'timestamp': row[1],
                'bid': float(row[2]) if row[2] else 0.0,
                'ask': float(row[3]) if row[3] else 0.0,
                'volume': int(row[4]) if row[4] else 0,
                # Add derived fields
                'open': (float(row[2]) + float(row[3])) / 2 if row[2] and row[3] else 0.0,
                'high': max(float(row[2]), float(row[3])) if row[2] and row[3] else 0.0,
                'low': min(float(row[2]), float(row[3])) if row[2] and row[3] else 0.0,
                'close': (float(row[2]) + float(row[3])) / 2 if row[2] and row[3] else 0.0,
                'timeframe': 'M1',
                'tick_volume': int(row[4]) if row[4] else 0,
                'spread': int((float(row[3]) - float(row[2])) * 10000) if row[2] and row[3] else 0,
                'is_synthetic': False
            })
    
    return rows


def delete_migrated_ticks(engine, ticks: List[Dict]) -> int:
    """
    Delete migrated ticks from HOT tier.
    
    Args:
        engine: SQLAlchemy engine
        ticks: List of tick dictionaries with timestamps
        
    Returns:
        Number of rows deleted
    """
    from sqlalchemy import text
    
    if not ticks:
        return 0
    
    # Get min and max timestamps for batch deletion
    timestamps = [t['timestamp'] for t in ticks]
    min_ts = min(timestamps)
    max_ts = max(timestamps)
    
    query = text("""
        DELETE FROM tick_cache
        WHERE timestamp >= :min_ts AND timestamp < :max_ts
    """)
    
    with engine.connect() as conn:
        result = conn.execute(query, {"min_ts": min_ts, "max_ts": max_ts})
        conn.commit()
        return result.rowcount


def migrate_batch(dry_run: bool = False) -> Dict[str, Any]:
    """
    Migrate a batch of data from HOT to WARM tier.
    
    Args:
        dry_run: If True, don't actually migrate
        
    Returns:
        Migration statistics
    """
    start_time = time.time()
    
    stats = {
        'rows_fetched': 0,
        'rows_inserted': 0,
        'rows_deleted': 0,
        'batches': 0,
        'errors': []
    }
    
    try:
        # Connect to HOT tier
        hot_engine = get_hot_connection()
        
        # Fetch batch
        batch_size = 10000
        ticks = fetch_old_ticks(hot_engine, batch_size)
        
        stats['rows_fetched'] = len(ticks)
        stats['batches'] = 1
        
        if not ticks:
            logger.info("No ticks to migrate")
            return stats
        
        logger.info(f"Fetched {len(ticks)} ticks from HOT tier")
        
        if dry_run:
            logger.info("Dry run - skipping insertion and deletion")
            return stats
        
        # Insert into WARM tier
        from src.database.duckdb_connection import insert_market_data
        
        warm_db_path = os.environ.get("WARM_DB_PATH", "/data/market_data.duckdb")
        
        inserted = insert_market_data(ticks, db_path=warm_db_path)
        stats['rows_inserted'] = inserted
        
        logger.info(f"Inserted {inserted} rows into WARM tier")
        
        # Delete from HOT tier
        deleted = delete_migrated_ticks(hot_engine, ticks)
        stats['rows_deleted'] = deleted
        
        logger.info(f"Deleted {deleted} rows from HOT tier")
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        stats['errors'].append(str(e))
    
    # Update metrics
    duration = time.time() - start_time
    
    if PROMETHEUS_AVAILABLE:
        MIGRATION_ROWS_TOTAL.labels(direction="hot_to_warm").inc(stats['rows_inserted'])
        MIGRATION_DURATION.labels(direction="hot_to_warm").set(duration)
    
    return stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Migrate HOT to WARM tier")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually migrate")
    parser.add_argument("--batch-size", type=int, default=10000, help="Batch size")
    parser.add_argument("--metrics-port", type=int, default=9092, help="Prometheus metrics port")
    args = parser.parse_args()
    
    # Start Prometheus metrics server
    if PROMETHEUS_AVAILABLE:
        try:
            start_http_server(args.metrics_port)
            logger.info(f"Prometheus metrics server started on port {args.metrics_port}")
        except Exception as e:
            logger.warning(f"Failed to start metrics server: {e}")
    
    logger.info("Starting HOT to WARM migration...")
    
    # Run migration
    stats = migrate_batch(dry_run=args.dry_run)
    
    # Log results
    logger.info("=" * 60)
    logger.info("Migration complete")
    logger.info(f"  Rows fetched: {stats['rows_fetched']}")
    logger.info(f"  Rows inserted: {stats['rows_inserted']}")
    logger.info(f"  Rows deleted: {stats['rows_deleted']}")
    logger.info(f"  Batches: {stats['batches']}")
    if stats['errors']:
        logger.warning(f"  Errors: {len(stats['errors'])}")
        for err in stats['errors']:
            logger.warning(f"    - {err}")
    logger.info("=" * 60)
    
    # Exit with error if there were errors
    return 1 if stats['errors'] else 0


if __name__ == "__main__":
    sys.exit(main())

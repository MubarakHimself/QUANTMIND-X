#!/usr/bin/env python3
"""
WARM to COLD Archival Script
=============================

Archives market data from WARM tier (DuckDB) to COLD tier (Parquet files).
Runs daily at 03:00 UTC via cron.

Usage:
    python scripts/archive_warm_to_cold.py
    python scripts/archive_warm_to_cold.py --dry-run

Environment:
    WARM_DB_PATH: Path to DuckDB file for WARM tier
    COLD_STORAGE_PATH: Path to cold storage directory
"""

import os
import sys
import json
import logging
import argparse
import time
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("archive_warm_to_cold")

# Prometheus metrics
try:
    from prometheus_client import Counter, Gauge, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not available - metrics disabled")

# Metrics
if PROMETHEUS_AVAILABLE:
    ARCHIVE_ROWS_TOTAL = Counter(
        'quantmind_migration_rows_total',
        'Total rows archived',
        ['direction']
    )
    ARCHIVE_DURATION = Gauge(
        'quantmind_migration_duration_seconds',
        'Archive duration in seconds',
        ['direction']
    )


def get_warm_connection():
    """Get DuckDB connection for WARM tier."""
    from src.database.duckdb_connection import DuckDBConnection
    
    warm_db_path = os.environ.get("WARM_DB_PATH", "/data/market_data.duckdb")
    return DuckDBConnection(db_path=warm_db_path)


def fetch_old_data(days: int = 7) -> pd.DataFrame:
    """
    Fetch market data older than specified days from WARM tier.
    
    Args:
        days: Number of days to retain
        
    Returns:
        DataFrame with historical data
    """
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
    
    with get_warm_connection() as conn:
        query = """
            SELECT * FROM market_data
            WHERE timestamp < :cutoff_date
            ORDER BY symbol, timeframe, timestamp
        """
        
        result = conn.execute_query(query, {"cutoff_date": cutoff_date})
        return result.df()


def group_by_partition(df: pd.DataFrame) -> Dict[tuple, pd.DataFrame]:
    """
    Group data by symbol and date for partitioning.
    
    Args:
        df: DataFrame with market data
        
    Returns:
        Dictionary mapping (symbol, year, month, day) to DataFrame
    """
    if df.empty:
        return {}
    
    # Ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Add partition columns
    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month
    df['day'] = df['timestamp'].dt.day
    
    # Group by partition
    groups = {}
    for (symbol, year, month, day), group_df in df.groupby(['symbol', 'year', 'month', 'day']):
        # Remove partition columns from data
        group_df = group_df.drop(columns=['year', 'month', 'day'])
        groups[(symbol, year, month, day)] = group_df
    
    return groups


def write_parquet(data: pd.DataFrame, symbol: str, year: int, month: int, day: int) -> str:
    """
    Write data to Parquet file in cold storage.
    
    Args:
        data: DataFrame to write
        symbol: Trading symbol
        year: Year
        month: Month
        day: Day
        
    Returns:
        Path to written file
    """
    cold_storage_path = os.environ.get("COLD_STORAGE_PATH", "/data/cold_storage")
    
    # Build path
    date_path = Path(cold_storage_path) / symbol / str(year) / f"{month:02d}" / f"{day:02d}"
    date_path.mkdir(parents=True, exist_ok=True)
    
    # Write Parquet file
    parquet_path = date_path / "data.parquet"
    data.to_parquet(parquet_path, index=False, compression='snappy')
    
    return str(parquet_path)


def update_manifest(symbol: str, year: int, month: int, day: int, row_count: int, file_path: str):
    """
    Update manifest JSON with partition metadata.
    
    Args:
        symbol: Trading symbol
        year: Year
        month: Month
        day: Day
        row_count: Number of rows in partition
        file_path: Path to Parquet file
    """
    cold_storage_path = os.environ.get("COLD_STORAGE_PATH", "/data/cold_storage")
    manifest_path = Path(cold_storage_path) / "manifest.json"
    
    # Load existing manifest
    manifest = {}
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    
    # Add partition entry
    partition_key = f"{symbol}/{year}/{month:02d}/{day:02d}"
    manifest[partition_key] = {
        "symbol": symbol,
        "year": year,
        "month": month,
        "day": day,
        "row_count": row_count,
        "file_path": file_path,
        "archived_at": datetime.now(timezone.utc).isoformat()
    }
    
    # Write manifest
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"Updated manifest for {partition_key}: {row_count} rows")


def sync_to_contabo(dry_run: bool = False) -> Dict[str, Any]:
    """
    Sync cold storage to Contabo VPS using rsync.

    Args:
        dry_run: If True, don't actually sync

    Returns:
        Sync statistics
    """
    sync_stats = {
        'synced': False,
        'error': None
    }

    contabo_cold_storage = os.environ.get("CONTABO_COLD_STORAGE")
    cold_storage_path = os.environ.get("COLD_STORAGE_PATH", "/data/cold_storage")

    if not contabo_cold_storage:
        logger.info("CONTABO_COLD_STORAGE not set, skipping remote sync")
        return sync_stats

    try:
        rsync_cmd = [
            "rsync", "-avz", "--delete",
            cold_storage_path + "/",
            contabo_cold_storage + "/"
        ]

        if dry_run:
            rsync_cmd.append("--dry-run")

        logger.info(f"Syncing to Contabo: {' '.join(rsync_cmd)}")
        result = subprocess.run(rsync_cmd, capture_output=True, text=True)

        if result.returncode == 0:
            sync_stats['synced'] = True
            logger.info("Successfully synced to Contabo")
            if result.stdout:
                logger.debug(f"rsync output: {result.stdout}")
        else:
            sync_stats['error'] = result.stderr
            logger.error(f"rsync failed: {result.stderr}")

    except Exception as e:
        sync_stats['error'] = str(e)
        logger.error(f"Failed to sync to Contabo: {e}")

    return sync_stats


def archive_data(dry_run: bool = False, days: int = 7) -> Dict[str, Any]:
    """
    Archive data from WARM to COLD tier.
    
    Args:
        dry_run: If True, don't actually archive
        days: Number of days to retain in WARM
        
    Returns:
        Archive statistics
    """
    start_time = time.time()
    
    stats = {
        'partitions_archived': 0,
        'rows_archived': 0,
        'files_written': 0,
        'rows_deleted': 0,
        'errors': []
    }
    
    try:
        # Fetch old data
        logger.info(f"Fetching data older than {days} days from WARM tier...")
        df = fetch_old_data(days)
        
        if df.empty:
            logger.info("No data to archive")
            return stats
        
        logger.info(f"Fetched {len(df)} rows from WARM tier")
        
        # Group by partition
        partitions = group_by_partition(df)
        stats['partitions_archived'] = len(partitions)
        
        logger.info(f"Grouped into {len(partitions)} partitions")
        
        if dry_run:
            logger.info("Dry run - skipping archival")
            return stats
        
        # Write each partition to Parquet
        for (symbol, year, month, day), partition_df in partitions.items():
            try:
                file_path = write_parquet(partition_df, symbol, year, month, day)
                stats['files_written'] += 1
                stats['rows_archived'] += len(partition_df)
                
                update_manifest(symbol, year, month, day, len(partition_df), file_path)
                
                logger.info(f"Archived {len(partition_df)} rows to {file_path}")
                
            except Exception as e:
                logger.error(f"Failed to archive partition {symbol}/{year}/{month}/{day}: {e}")
                stats['errors'].append(str(e))
        
        # Clean up old data from WARM tier
        if stats['files_written'] > 0:
            from src.database.duckdb_connection import cleanup_old_market_data
            
            warm_db_path = os.environ.get("WARM_DB_PATH", "/data/market_data.duckdb")
            deleted = cleanup_old_market_data(days=days, db_path=warm_db_path)
            stats['rows_deleted'] = deleted
            
            logger.info(f"Deleted {deleted} rows from WARM tier")
        
    except Exception as e:
        logger.error(f"Archival failed: {e}")
        stats['errors'].append(str(e))
    
    # Update metrics
    duration = time.time() - start_time
    
    if PROMETHEUS_AVAILABLE:
        ARCHIVE_ROWS_TOTAL.labels(direction="warm_to_cold").inc(stats['rows_archived'])
        ARCHIVE_DURATION.labels(direction="warm_to_cold").set(duration)
    
    return stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Archive WARM to COLD tier")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually archive")
    parser.add_argument("--days", type=int, default=7, help="Days to retain in WARM tier")
    parser.add_argument("--metrics-port", type=int, default=9093, help="Prometheus metrics port")
    args = parser.parse_args()
    
    # Start Prometheus metrics server
    if PROMETHEUS_AVAILABLE:
        try:
            start_http_server(args.metrics_port)
            logger.info(f"Prometheus metrics server started on port {args.metrics_port}")
        except Exception as e:
            logger.warning(f"Failed to start metrics server: {e}")
    
    logger.info("Starting WARM to COLD archival...")
    
    # Run archival
    stats = archive_data(dry_run=args.dry_run, days=args.days)
    
    # Log results
    logger.info("=" * 60)
    logger.info("Archival complete")
    logger.info(f"  Partitions archived: {stats['partitions_archived']}")
    logger.info(f"  Files written: {stats['files_written']}")
    logger.info(f"  Rows archived: {stats['rows_archived']}")
    logger.info(f"  Rows deleted from WARM: {stats['rows_deleted']}")
    if stats['errors']:
        logger.warning(f"  Errors: {len(stats['errors'])}")
        for err in stats['errors']:
            logger.warning(f"    - {err}")
    logger.info("=" * 60)
    
    # Exit with error if there were errors
    return 1 if stats['errors'] else 0


if __name__ == "__main__":
    sys.exit(main())

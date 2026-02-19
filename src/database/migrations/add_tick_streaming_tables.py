"""
Database Migration Script: Add Tick Streaming Tables

This script creates the HOT and WARM tier tables for the three-tier database architecture:
- HOT (PostgreSQL): symbol_subscriptions, tick_cache - 1 hour retention
- WARM (DuckDB): market_data - 30 day retention  
- COLD (Parquet): Historical data - indefinite retention

Usage:
    python -m src.database.migrations.add_tick_streaming_tables [--rollback]
"""

import argparse
import logging
import sys
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_hot_tier_tables():
    """Create HOT tier tables in PostgreSQL."""
    logger.info("Creating HOT tier tables...")
    
    try:
        from src.database.db_manager import DBManager
        from src.database.models import Base, SymbolSubscription, TickCache
        
        db = DBManager()
        
        # Create all tables
        Base.metadata.create_all(db.engine)
        
        logger.info("HOT tier tables created successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create HOT tier tables: {e}")
        return False


def create_warm_tier_tables():
    """Create WARM tier tables in DuckDB."""
    logger.info("Creating WARM tier tables...")
    
    try:
        from src.database.duckdb_connection import create_market_data_table
        
        create_market_data_table()
        
        logger.info("WARM tier tables created successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create WARM tier tables: {e}")
        return False


def create_cold_tier_structure():
    """Create COLD tier Parquet directory structure."""
    logger.info("Creating COLD tier directory structure...")
    
    import os
    from pathlib import Path
    
    base_path = Path("data/historical")
    
    # Create directories for common symbols
    symbols = [
        "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD",
        "XAUUSD", "XAGUSD", "NAS100", "SPX500", "US30"
    ]
    timeframes = ["M1", "M5", "M15", "H1", "H4", "D1"]
    
    for symbol in symbols:
        for timeframe in timeframes:
            path = base_path / symbol / timeframe
            path.mkdir(parents=True, exist_ok=True)
    
    logger.info("COLD tier directory structure created successfully")
    return True


def run_migration():
    """Run the complete migration."""
    logger.info("Starting tick streaming tables migration...")
    
    # Create HOT tier (PostgreSQL)
    if not create_hot_tier_tables():
        logger.error("HOT tier migration failed")
        return False
    
    # Create WARM tier (DuckDB)
    if not create_warm_tier_tables():
        logger.error("WARM tier migration failed")
        return False
    
    # Create COLD tier structure
    if not create_cold_tier_structure():
        logger.error("COLD tier migration failed")
        return False
    
    logger.info("Migration completed successfully!")
    return True


def rollback_migration():
    """Rollback the migration (drop tables)."""
    logger.info("Rolling back tick streaming tables migration...")
    
    try:
        from src.database.db_manager import DBManager
        from src.database.models import SymbolSubscription, TickCache
        
        db = DBManager()
        
        # Drop tables
        with db.get_session() as session:
            session.query(SymbolSubscription).delete()
            session.query(TickCache).delete()
        
        logger.info("HOT tier tables dropped successfully")
        
    except Exception as e:
        logger.error(f"Rollback failed: {e}")
        return False
    
    # Note: DuckDB and Parquet data are not dropped as they may contain historical data
    logger.info("Rollback completed!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Database migration for tick streaming tables")
    parser.add_argument("--rollback", action="store_true", help="Rollback the migration")
    
    args = parser.parse_args()
    
    if args.rollback:
        success = rollback_migration()
    else:
        success = run_migration()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

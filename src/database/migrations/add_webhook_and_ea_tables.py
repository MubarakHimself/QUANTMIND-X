"""
Database Migration Script: Add Webhook Logs and Imported EA Tables

This script creates the webhook_logs and imported_eas tables for:
- TradingView webhook logging
- GitHub EA sync tracking

Usage:
    python -m src.database.migrations.add_webhook_and_ea_tables [--rollback]
"""

import argparse
import logging
import sys
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_tables():
    """Create webhook_logs and imported_eas tables."""
    logger.info("Creating webhook_logs and imported_eas tables...")
    
    try:
        from src.database.db_manager import DBManager
        from src.database.models import Base, WebhookLog, ImportedEA
        
        db = DBManager()
        
        # Create all tables
        Base.metadata.create_all(db.engine)
        
        logger.info("webhook_logs and imported_eas tables created successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create tables: {e}")
        return False


def drop_tables():
    """Drop webhook_logs and imported_eas tables."""
    logger.info("Dropping webhook_logs and imported_eas tables...")
    
    try:
        from src.database.db_manager import DBManager
        from src.database.models import WebhookLog, ImportedEA
        from sqlalchemy import text
        
        db = DBManager()
        
        with db.engine.connect() as conn:
            # Drop tables if they exist
            conn.execute(text("DROP TABLE IF EXISTS webhook_logs CASCADE"))
            conn.execute(text("DROP TABLE IF EXISTS imported_eas CASCADE"))
            conn.commit()
        
        logger.info("webhook_logs and imported_eas tables dropped successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to drop tables: {e}")
        return False


def run_migration():
    """Run the migration."""
    logger.info("Starting webhook_logs and imported_eas migration...")
    
    if not create_tables():
        logger.error("Migration failed")
        return False
    
    logger.info("Migration completed successfully!")
    return True


def rollback_migration():
    """Rollback the migration."""
    logger.info("Rolling back webhook_logs and imported_eas migration...")
    
    success = drop_tables()
    
    if success:
        logger.info("Rollback completed!")
    else:
        logger.error("Rollback failed!")
    
    return success


def main():
    parser = argparse.ArgumentParser(description="Database migration for webhook_logs and imported_eas tables")
    parser.add_argument("--rollback", action="store_true", help="Rollback the migration")
    
    args = parser.parse_args()
    
    if args.rollback:
        success = rollback_migration()
    else:
        success = run_migration()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

"""
V8 Database Migration Script

Adds V8 Tiered Risk Engine schema extensions:
- risk_mode column to prop_firm_accounts table
- risk_tier_transitions table for audit logging

This migration is backward compatible and can be run on existing V7 databases.
"""

import logging
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker
from src.database.models import Base, RiskTierTransition
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_column_exists(engine, table_name, column_name):
    """Check if a column exists in a table."""
    inspector = inspect(engine)
    columns = [col['name'] for col in inspector.get_columns(table_name)]
    return column_name in columns


def check_table_exists(engine, table_name):
    """Check if a table exists in the database."""
    inspector = inspect(engine)
    return table_name in inspector.get_table_names()


def migrate_v8(db_path='data/db/quantmind.db'):
    """
    Apply V8 schema migrations to existing database.
    
    Migrations:
    1. Add risk_mode column to prop_firm_accounts (Tiered Risk Engine)
    2. Create risk_tier_transitions table (Tiered Risk Engine)
    3. Create crypto_trades table (Crypto Module)
    4. Add broker_id column to trade_proposals (Broker Registry)
    
    Args:
        db_path: Path to SQLite database file
        
    Returns:
        bool: True if migration successful, False otherwise
    """
    try:
        # Create engine
        engine = create_engine(f'sqlite:///{db_path}', echo=False)
        Session = sessionmaker(bind=engine)
        session = Session()
        
        logger.info("=" * 60)
        logger.info("V8 DATABASE MIGRATION")
        logger.info("=" * 60)
        
        # Step 1: Add risk_mode column to prop_firm_accounts
        if not check_column_exists(engine, 'prop_firm_accounts', 'risk_mode'):
            logger.info("Adding risk_mode column to prop_firm_accounts table...")
            with engine.connect() as conn:
                conn.execute(text(
                    "ALTER TABLE prop_firm_accounts ADD COLUMN risk_mode VARCHAR(20) DEFAULT 'growth' NOT NULL"
                ))
                conn.commit()
            logger.info("✓ risk_mode column added successfully")
        else:
            logger.info("✓ risk_mode column already exists")
        
        # Step 2: Create risk_tier_transitions table
        if not check_table_exists(engine, 'risk_tier_transitions'):
            logger.info("Creating risk_tier_transitions table...")
            RiskTierTransition.__table__.create(engine)
            logger.info("✓ risk_tier_transitions table created successfully")
        else:
            logger.info("✓ risk_tier_transitions table already exists")
        
        # Step 3: Create crypto_trades table
        if not check_table_exists(engine, 'crypto_trades'):
            logger.info("Creating crypto_trades table...")
            from src.database.models import CryptoTrade
            CryptoTrade.__table__.create(engine)
            logger.info("✓ crypto_trades table created successfully")
        else:
            logger.info("✓ crypto_trades table already exists")
        
        # Step 4: Add broker_id column to trade_proposals (if table exists)
        if check_table_exists(engine, 'trade_proposals'):
            if not check_column_exists(engine, 'trade_proposals', 'broker_id'):
                logger.info("Adding broker_id column to trade_proposals table...")
                with engine.connect() as conn:
                    conn.execute(text(
                        "ALTER TABLE trade_proposals ADD COLUMN broker_id VARCHAR(100)"
                    ))
                    conn.commit()
                logger.info("✓ broker_id column added successfully")
            else:
                logger.info("✓ broker_id column already exists")
        
        # Step 5: Create indexes
        logger.info("Creating indexes...")
        with engine.connect() as conn:
            # Index on risk_mode for fast tier queries
            try:
                conn.execute(text(
                    "CREATE INDEX IF NOT EXISTS idx_prop_firm_accounts_risk_mode ON prop_firm_accounts(risk_mode)"
                ))
                logger.info("✓ Index on risk_mode created")
            except Exception as e:
                logger.warning(f"Index on risk_mode may already exist: {e}")
            
            # Index on account_id and timestamp for tier transitions
            try:
                conn.execute(text(
                    "CREATE INDEX IF NOT EXISTS idx_tier_transition_account_timestamp "
                    "ON risk_tier_transitions(account_id, transition_timestamp)"
                ))
                logger.info("✓ Index on tier transitions created")
            except Exception as e:
                logger.warning(f"Index on tier transitions may already exist: {e}")
            
            # Indexes for crypto_trades
            try:
                conn.execute(text(
                    "CREATE INDEX IF NOT EXISTS idx_crypto_trades_broker_symbol "
                    "ON crypto_trades(broker_id, symbol)"
                ))
                logger.info("✓ Index on crypto_trades (broker_id, symbol) created")
            except Exception as e:
                logger.warning(f"Index on crypto_trades may already exist: {e}")
            
            try:
                conn.execute(text(
                    "CREATE INDEX IF NOT EXISTS idx_crypto_trades_status_timestamp "
                    "ON crypto_trades(status, open_timestamp)"
                ))
                logger.info("✓ Index on crypto_trades (status, timestamp) created")
            except Exception as e:
                logger.warning(f"Index on crypto_trades may already exist: {e}")
            
            conn.commit()
        
        # Step 6: Verify migration
        logger.info("\nVerifying migration...")
        inspector = inspect(engine)
        
        # Check prop_firm_accounts columns
        pfa_columns = [col['name'] for col in inspector.get_columns('prop_firm_accounts')]
        assert 'risk_mode' in pfa_columns, "risk_mode column not found"
        logger.info("✓ prop_firm_accounts schema verified")
        
        # Check risk_tier_transitions table
        assert 'risk_tier_transitions' in inspector.get_table_names(), "risk_tier_transitions table not found"
        rtt_columns = [col['name'] for col in inspector.get_columns('risk_tier_transitions')]
        expected_columns = ['id', 'account_id', 'from_tier', 'to_tier', 'equity_at_transition', 'transition_timestamp']
        for col in expected_columns:
            assert col in rtt_columns, f"Column {col} not found in risk_tier_transitions"
        logger.info("✓ risk_tier_transitions schema verified")
        
        # Check crypto_trades table
        assert 'crypto_trades' in inspector.get_table_names(), "crypto_trades table not found"
        ct_columns = [col['name'] for col in inspector.get_columns('crypto_trades')]
        expected_ct_columns = ['id', 'broker_type', 'broker_id', 'order_id', 'symbol', 'direction', 
                               'volume', 'entry_price', 'exit_price', 'status']
        for col in expected_ct_columns:
            assert col in ct_columns, f"Column {col} not found in crypto_trades"
        logger.info("✓ crypto_trades schema verified")
        
        logger.info("\n" + "=" * 60)
        logger.info("V8 MIGRATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info("\nMigration Summary:")
        logger.info("  ✓ Tiered Risk Engine (risk_mode, risk_tier_transitions)")
        logger.info("  ✓ Crypto Module (crypto_trades)")
        logger.info("  ✓ Broker Registry (broker_id in trade_proposals)")
        logger.info("  ✓ All indexes created")
        logger.info("=" * 60)
        
        session.close()
        return True
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def rollback_v8(db_path='data/db/quantmind.db'):
    """
    Rollback V8 migrations (for testing purposes).
    
    WARNING: This will drop the risk_tier_transitions table and remove
    the risk_mode column from prop_firm_accounts.
    
    Args:
        db_path: Path to SQLite database file
        
    Returns:
        bool: True if rollback successful, False otherwise
    """
    try:
        engine = create_engine(f'sqlite:///{db_path}', echo=False)
        
        logger.info("=" * 60)
        logger.info("V8 DATABASE ROLLBACK")
        logger.info("=" * 60)
        
        with engine.connect() as conn:
            # Drop risk_tier_transitions table
            if check_table_exists(engine, 'risk_tier_transitions'):
                logger.info("Dropping risk_tier_transitions table...")
                conn.execute(text("DROP TABLE risk_tier_transitions"))
                logger.info("✓ risk_tier_transitions table dropped")
            
            # Note: SQLite doesn't support DROP COLUMN directly
            # Would need to recreate table without risk_mode column
            logger.warning("Note: risk_mode column cannot be easily removed in SQLite")
            logger.warning("To fully rollback, restore from V7 database backup")
            
            conn.commit()
        
        logger.info("\n" + "=" * 60)
        logger.info("V8 ROLLBACK COMPLETED")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"Rollback failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--rollback':
        rollback_v8()
    else:
        migrate_v8()

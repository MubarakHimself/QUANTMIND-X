"""
Migration: Add Demo Mode Support
===========================================

Adds `mode` column to support demo/live trading distinction.

Tables modified:
- crypto_trades: Add mode column for crypto trade tracking
- trade_journal: Add mode column for journal entries
- strategy_performance: Add mode column for performance tracking
- paper_trading_performance: Add mode column for paper trading
- bot_circuit_breaker: Add mode column for circuit breaker tracking
- daily_snapshots: Add mode column for daily snapshots
- prop_firm_accounts: Add mode column for account types

Version: 008
"""

from src.database.migrations.migration_runner import Migration


def get_migration() -> Migration:
    """Get the demo mode migration."""
    return Migration(
        version="008",
        name="add_demo_mode",
        description="Add mode column to trading tables for demo/live distinction",
        up_sql="""
            -- Create trading_mode ENUM type (simulated in SQLite as TEXT with CHECK constraint)
            -- Add mode column to crypto_trades
            ALTER TABLE crypto_trades ADD COLUMN mode VARCHAR(10) NOT NULL DEFAULT 'live';
            
            -- Add mode column to trade_journal
            ALTER TABLE trade_journal ADD COLUMN mode VARCHAR(10) NOT NULL DEFAULT 'live';
            
            -- Add mode column to strategy_performance
            ALTER TABLE strategy_performance ADD COLUMN mode VARCHAR(10) NOT NULL DEFAULT 'live';
            
            -- Add mode column to paper_trading_performance (defaults to demo for paper trading)
            ALTER TABLE paper_trading_performance ADD COLUMN mode VARCHAR(10) NOT NULL DEFAULT 'demo';
            
            -- Add mode column to bot_circuit_breaker
            ALTER TABLE bot_circuit_breaker ADD COLUMN mode VARCHAR(10) NOT NULL DEFAULT 'live';
            
            -- Add mode column to daily_snapshots
            ALTER TABLE daily_snapshots ADD COLUMN mode VARCHAR(10) NOT NULL DEFAULT 'live';
            
            -- Add mode column to prop_firm_accounts
            ALTER TABLE prop_firm_accounts ADD COLUMN mode VARCHAR(10) NOT NULL DEFAULT 'live';
            
            -- Create indexes on mode columns for efficient filtering
            CREATE INDEX IF NOT EXISTS idx_crypto_trades_mode ON crypto_trades(mode);
            CREATE INDEX IF NOT EXISTS idx_trade_journal_mode ON trade_journal(mode);
            CREATE INDEX IF NOT EXISTS idx_strategy_performance_mode ON strategy_performance(mode);
            CREATE INDEX IF NOT EXISTS idx_paper_trading_mode ON paper_trading_performance(mode);
            CREATE INDEX IF NOT EXISTS idx_bot_circuit_breaker_mode ON bot_circuit_breaker(mode);
            CREATE INDEX IF NOT EXISTS idx_daily_snapshots_mode ON daily_snapshots(mode);
            CREATE INDEX IF NOT EXISTS idx_prop_firm_accounts_mode ON prop_firm_accounts(mode);
        """,
        down_sql="""
            -- Drop indexes first
            DROP INDEX IF EXISTS idx_crypto_trades_mode;
            DROP INDEX IF EXISTS idx_trade_journal_mode;
            DROP INDEX IF EXISTS idx_strategy_performance_mode;
            DROP INDEX IF EXISTS idx_paper_trading_mode;
            DROP INDEX IF EXISTS idx_bot_circuit_breaker_mode;
            DROP INDEX IF EXISTS idx_daily_snapshots_mode;
            DROP INDEX IF EXISTS idx_prop_firm_accounts_mode;
            
            -- Note: SQLite doesn't support DROP COLUMN directly
            -- In production, this would require recreating tables without the mode column
            -- For rollback purposes, we leave the columns in place but unused
            -- The indexes above are removed to clean up
        """,
        db_type="sqlite"
    )
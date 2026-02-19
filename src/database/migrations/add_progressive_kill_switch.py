"""
Migration: Add Progressive Kill Switch Tables
===============================================

Creates the following tables:
- strategy_family_states: Tracks strategy family quarantine state
- account_loss_states: Tracks account daily/weekly loss limits
- alert_history: Stores all raised alerts for audit

Version: 006
"""

from sqlalchemy import text
from src.database.migrations.migration_runner import Migration


def get_migration() -> Migration:
    """Get the progressive kill switch migration."""
    return Migration(
        version="006",
        name="add_progressive_kill_switch_tables",
        description="Create progressive kill switch tables for strategy family states, account loss states, and alert history",
        up_sql="""
            -- Strategy Family States table (Tier 2)
            CREATE TABLE IF NOT EXISTS strategy_family_states (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                family VARCHAR(20) NOT NULL UNIQUE,
                failed_bots JSON NOT NULL DEFAULT '[]',
                total_pnl REAL NOT NULL DEFAULT 0.0,
                initial_capital REAL NOT NULL DEFAULT 10000.0,
                is_quarantined BOOLEAN NOT NULL DEFAULT 0,
                quarantine_time TIMESTAMP,
                quarantine_reason VARCHAR(200),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE INDEX IF NOT EXISTS idx_strategy_family_quarantined 
            ON strategy_family_states(is_quarantined);
            
            CREATE INDEX IF NOT EXISTS idx_strategy_family_name 
            ON strategy_family_states(family);

            -- Account Loss States table (Tier 3)
            CREATE TABLE IF NOT EXISTS account_loss_states (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                account_id VARCHAR(50) NOT NULL UNIQUE,
                initial_balance REAL NOT NULL DEFAULT 10000.0,
                daily_pnl REAL NOT NULL DEFAULT 0.0,
                weekly_pnl REAL NOT NULL DEFAULT 0.0,
                last_reset_date VARCHAR(10),
                week_start VARCHAR(10),
                daily_stop_triggered BOOLEAN NOT NULL DEFAULT 0,
                weekly_stop_triggered BOOLEAN NOT NULL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE INDEX IF NOT EXISTS idx_account_loss_account_id 
            ON account_loss_states(account_id);
            
            CREATE INDEX IF NOT EXISTS idx_account_loss_stops 
            ON account_loss_states(daily_stop_triggered, weekly_stop_triggered);

            -- Alert History table
            CREATE TABLE IF NOT EXISTS alert_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                level VARCHAR(10) NOT NULL,
                tier INTEGER NOT NULL,
                message TEXT NOT NULL,
                threshold_pct REAL NOT NULL,
                triggered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                source VARCHAR(50) NOT NULL,
                metadata JSON,
                is_active BOOLEAN NOT NULL DEFAULT 1,
                cleared_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE INDEX IF NOT EXISTS idx_alert_history_level_tier 
            ON alert_history(level, tier);
            
            CREATE INDEX IF NOT EXISTS idx_alert_history_triggered 
            ON alert_history(triggered_at);
            
            CREATE INDEX IF NOT EXISTS idx_alert_history_source 
            ON alert_history(source);
            
            CREATE INDEX IF NOT EXISTS idx_alert_history_is_active 
            ON alert_history(is_active);
        """,
        down_sql="""
            DROP TABLE IF EXISTS alert_history;
            DROP TABLE IF EXISTS account_loss_states;
            DROP TABLE IF EXISTS strategy_family_states;
        """,
        db_type="sqlite"
    )

"""
Migration: Add Bot Lifecycle Log
=================================

Tracks bot tag transitions for the LifecycleManager system.

Schema (from spec lines 150-163):
- id: Primary key
- bot_id: Reference to bot (foreign key to bot_manifest)
- from_tag: Source tag (@primal, @pending, @perfect, @live, @quarantine, @dead)
- to_tag: Destination tag
- reason: Reason for transition (promotion criteria met, quarantine trigger, etc.)
- timestamp: When the transition occurred
- triggered_by: System (LifecycleManager) or manual
- performance_stats: JSON snapshot of bot performance at time of transition
- notes: Additional notes

Indexes:
- idx_lifecycle_bot_id on bot_id
- idx_lifecycle_timestamp on timestamp

Version: 009
"""

from src.database.migrations.migration_runner import Migration


def get_migration() -> Migration:
    """Get the bot lifecycle log migration."""
    return Migration(
        version="009",
        name="add_bot_lifecycle_log",
        description="Add bot_lifecycle_log table for tracking tag transitions",
        up_sql="""
            -- Create bot_lifecycle_log table
            CREATE TABLE IF NOT EXISTS bot_lifecycle_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bot_id VARCHAR(255) NOT NULL,
                from_tag VARCHAR(50) NOT NULL,
                to_tag VARCHAR(50) NOT NULL,
                reason TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                triggered_by VARCHAR(50) DEFAULT 'system',
                performance_stats TEXT,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- Create indexes for efficient querying
            CREATE INDEX IF NOT EXISTS idx_lifecycle_bot_id ON bot_lifecycle_log(bot_id);
            CREATE INDEX IF NOT EXISTS idx_lifecycle_timestamp ON bot_lifecycle_log(timestamp);
            CREATE INDEX IF NOT EXISTS idx_lifecycle_from_tag ON bot_lifecycle_log(from_tag);
            CREATE INDEX IF NOT EXISTS idx_lifecycle_to_tag ON bot_lifecycle_log(to_tag);
        """,
        down_sql="""
            -- Drop indexes
            DROP INDEX IF EXISTS idx_lifecycle_bot_id;
            DROP INDEX IF EXISTS idx_lifecycle_timestamp;
            DROP INDEX IF EXISTS idx_lifecycle_from_tag;
            DROP INDEX IF EXISTS idx_lifecycle_to_tag;
            
            -- Drop table
            DROP TABLE IF EXISTS bot_lifecycle_log;
        """,
        db_type="sqlite"
    )

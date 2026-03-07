"""
Migration: Add Activity Events Table
=====================================

Adds table for storing agent activity events persistently.

Tables created:
- activity_events: Store agent activity events for feed and analytics

Version: 013
"""

from src.database.migrations.migration_runner import Migration


def get_migration() -> Migration:
    """Get the activity events migration."""
    return Migration(
        version="013",
        name="add_activity_events",
        description="Add activity_events table for persistent agent activity feed storage",
        up_sql="""
            -- Create activity_events table
            CREATE TABLE IF NOT EXISTS activity_events (
                id VARCHAR(36) PRIMARY KEY,
                agent_id VARCHAR(100) NOT NULL,
                agent_type VARCHAR(50) NOT NULL,
                agent_name VARCHAR(100) NOT NULL,
                event_type VARCHAR(20) NOT NULL,
                action TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                details JSON,
                reasoning TEXT,
                tool_name VARCHAR(100),
                tool_result JSON,
                status VARCHAR(20) NOT NULL DEFAULT 'pending',
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            -- Create indexes for efficient querying
            CREATE INDEX IF NOT EXISTS idx_activity_events_agent_id ON activity_events(agent_id);
            CREATE INDEX IF NOT EXISTS idx_activity_events_agent_type ON activity_events(agent_type);
            CREATE INDEX IF NOT EXISTS idx_activity_events_event_type ON activity_events(event_type);
            CREATE INDEX IF NOT EXISTS idx_activity_events_timestamp ON activity_events(timestamp);
            CREATE INDEX IF NOT EXISTS idx_activity_events_status ON activity_events(status);
            CREATE INDEX IF NOT EXISTS idx_activity_events_created_at ON activity_events(created_at);
            CREATE INDEX IF NOT EXISTS idx_activity_events_agent_timestamp ON activity_events(agent_id, timestamp);
            CREATE INDEX IF NOT EXISTS idx_activity_events_type_timestamp ON activity_events(event_type, timestamp);
        """,
        down_sql="""
            -- Drop indexes
            DROP INDEX IF EXISTS idx_activity_events_agent_id;
            DROP INDEX IF EXISTS idx_activity_events_agent_type;
            DROP INDEX IF EXISTS idx_activity_events_event_type;
            DROP INDEX IF EXISTS idx_activity_events_timestamp;
            DROP INDEX IF EXISTS idx_activity_events_status;
            DROP INDEX IF EXISTS idx_activity_events_created_at;
            DROP INDEX IF EXISTS idx_activity_events_agent_timestamp;
            DROP INDEX IF EXISTS idx_activity_events_type_timestamp;

            -- Drop table
            DROP TABLE IF EXISTS activity_events;
        """,
        db_type="sqlite"
    )

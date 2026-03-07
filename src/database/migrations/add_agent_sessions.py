"""
Migration: Add Agent Sessions Table
=====================================

Adds table for persisting agent conversation sessions.

Tables created:
- agent_sessions: Store agent conversation sessions with full state

Version: 011
"""

from src.database.migrations.migration_runner import Migration


def get_migration() -> Migration:
    """Get the agent sessions migration."""
    return Migration(
        version="011",
        name="add_agent_sessions",
        description="Add agent_sessions table for conversation persistence",
        up_sql="""
            -- Create agent_sessions table
            CREATE TABLE IF NOT EXISTS agent_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id VARCHAR(100) NOT NULL UNIQUE,
                name VARCHAR(255) NOT NULL,
                agent_type VARCHAR(50) NOT NULL,
                status VARCHAR(20) NOT NULL DEFAULT 'active',
                conversation_history JSON NOT NULL DEFAULT('[]'),
                variables JSON NOT NULL DEFAULT('{}'),
                session_metadata JSON NOT NULL DEFAULT('{}'),
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                modified_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP
            );

            -- Create indexes for efficient querying
            CREATE INDEX IF NOT EXISTS idx_agent_sessions_session_id ON agent_sessions(session_id);
            CREATE INDEX IF NOT EXISTS idx_agent_sessions_agent_type ON agent_sessions(agent_type);
            CREATE INDEX IF NOT EXISTS idx_agent_sessions_status ON agent_sessions(status);
            CREATE INDEX IF NOT EXISTS idx_agent_sessions_created_at ON agent_sessions(created_at);
            CREATE INDEX IF NOT EXISTS idx_agent_sessions_agent_status ON agent_sessions(agent_type, status);
        """,
        down_sql="""
            -- Drop indexes
            DROP INDEX IF EXISTS idx_agent_sessions_session_id;
            DROP INDEX IF EXISTS idx_agent_sessions_agent_type;
            DROP INDEX IF EXISTS idx_agent_sessions_status;
            DROP INDEX IF EXISTS idx_agent_sessions_created_at;
            DROP INDEX IF EXISTS idx_agent_sessions_agent_status;

            -- Drop table
            DROP TABLE IF EXISTS agent_sessions;
        """,
        db_type="sqlite"
    )

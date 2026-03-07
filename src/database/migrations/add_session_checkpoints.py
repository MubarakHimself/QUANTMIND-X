"""
Migration: Add Session Checkpoints Table
=========================================

Adds table for storing session checkpoints.

Tables created:
- session_checkpoints: Store session state snapshots for resuming

Version: 012
"""

from src.database.migrations.migration_runner import Migration


def get_migration() -> Migration:
    """Get the session checkpoints migration."""
    return Migration(
        version="012",
        name="add_session_checkpoints",
        description="Add session_checkpoints table for session state snapshots",
        up_sql="""
            -- Create session_checkpoints table
            CREATE TABLE IF NOT EXISTS session_checkpoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id VARCHAR(100) NOT NULL,
                checkpoint_number INTEGER NOT NULL,
                checkpoint_type VARCHAR(20) NOT NULL DEFAULT 'auto',
                conversation_history JSON NOT NULL DEFAULT('[]'),
                variables JSON NOT NULL DEFAULT('{}'),
                progress_percent REAL NOT NULL DEFAULT 0.0,
                current_step VARCHAR(255),
                checkpoint_metadata JSON NOT NULL DEFAULT('{}'),
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                size_bytes INTEGER NOT NULL DEFAULT 0
            );

            -- Create indexes for efficient querying
            CREATE INDEX IF NOT EXISTS idx_session_checkpoints_session_id ON session_checkpoints(session_id);
            CREATE INDEX IF NOT EXISTS idx_session_checkpoints_created_at ON session_checkpoints(created_at);
            CREATE INDEX IF NOT EXISTS idx_session_checkpoints_session_num ON session_checkpoints(session_id, checkpoint_number);
            CREATE INDEX IF NOT EXISTS idx_session_checkpoints_type ON session_checkpoints(checkpoint_type);
        """,
        down_sql="""
            -- Drop indexes
            DROP INDEX IF EXISTS idx_session_checkpoints_session_id;
            DROP INDEX IF EXISTS idx_session_checkpoints_created_at;
            DROP INDEX IF EXISTS idx_session_checkpoints_session_num;
            DROP INDEX IF EXISTS idx_session_checkpoints_type;

            -- Drop table
            DROP TABLE IF EXISTS session_checkpoints;
        """,
        db_type="sqlite"
    )

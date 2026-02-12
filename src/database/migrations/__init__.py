"""
Migrations Package for QuantMindX Database

Provides schema version tracking and migration support for both
SQLite (transactional) and DuckDB (analytics) databases.
"""

from .migration_runner import (
    Migration,
    MigrationRunner,
    MigrationError,
    run_sqlite_migrations,
    run_duckdb_migrations,
    rollback_sqlite,
    rollback_duckdb,
    get_migration_status,
    create_task_group_2_migrations
)

__all__ = [
    'Migration',
    'MigrationRunner',
    'MigrationError',
    'run_sqlite_migrations',
    'run_duckdb_migrations',
    'rollback_sqlite',
    'rollback_duckdb',
    'get_migration_status',
    'create_task_group_2_migrations'
]

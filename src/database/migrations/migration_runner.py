"""
Migration System for QuantMindX Database

Provides schema version tracking, migration execution, and rollback support.
Supports both SQLite (transactional) and DuckDB (analytics) databases.

Reference: specs/2026-02-07-quantmindx-trading-system/spec.md
Task Group 2: Migration system with rollback support
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from .engine import engine
from .duckdb_connection import DuckDBConnection


# Configure logging
logger = logging.getLogger(__name__)


# Migration paths
SQLITE_MIGRATIONS_DIR = Path(__file__).parent / "migrations_sqlite"
DUCKDB_MIGRATIONS_DIR = Path(__file__).parent / "migrations_duckdb"

# Create directories if they don't exist
SQLITE_MIGRATIONS_DIR.mkdir(parents=True, exist_ok=True)
DUCKDB_MIGRATIONS_DIR.mkdir(parents=True, exist_ok=True)


class MigrationError(Exception):
    """Custom exception for migration errors."""
    pass


class Migration:
    """
    Represents a single database migration.

    Attributes:
        version: Migration version number (e.g., 001, 002)
        name: Human-readable migration name
        description: What this migration does
        up_sql: SQL to apply the migration
        down_sql: SQL to rollback the migration
    """

    def __init__(
        self,
        version: str,
        name: str,
        description: str,
        up_sql: str,
        down_sql: str,
        db_type: str = "sqlite"
    ):
        self.version = version
        self.name = name
        self.description = description
        self.up_sql = up_sql
        self.down_sql = down_sql
        self.db_type = db_type

    def to_dict(self) -> Dict[str, Any]:
        """Convert migration to dictionary for JSON serialization."""
        return {
            'version': self.version,
            'name': self.name,
            'description': self.description,
            'up_sql': self.up_sql,
            'down_sql': self.down_sql,
            'db_type': self.db_type
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Migration':
        """Create migration from dictionary."""
        return cls(
            version=data['version'],
            name=data['name'],
            description=data['description'],
            up_sql=data['up_sql'],
            down_sql=data['down_sql'],
            db_type=data.get('db_type', 'sqlite')
        )


class MigrationRunner:
    """
    Migration runner for managing database schema changes.

    Features:
    - Version tracking with schema_migrations table
    - Forward migration with SQL execution
    - Rollback support with down_sql
    - Transaction safety (for SQLite)
    - Migration history logging
    - Support for both SQLite and DuckDB

    Usage:
        ```python
        # Initialize migrations
        runner = MigrationRunner()

        # Create a new migration
        runner.create_migration(
            name="add_strategy_folders",
            description="Add strategy_folders table",
            up_sql="CREATE TABLE strategy_folders (...)",
            down_sql="DROP TABLE strategy_folders"
        )

        # Run pending migrations
        runner.migrate()

        # Rollback last migration
        runner.rollback()
        ```
    """

    # SQLite schema
    SQLITE_SCHEMA_TABLE = "schema_migrations"

    # DuckDB schema
    DUCKDB_SCHEMA_TABLE = "schema_migrations"

    def __init__(self, db_type: str = "sqlite"):
        """
        Initialize migration runner.

        Args:
            db_type: Type of database ('sqlite' or 'duckdb').
        """
        self.db_type = db_type
        self.migrations_dir = (
            SQLITE_MIGRATIONS_DIR if db_type == "sqlite"
            else DUCKDB_MIGRATIONS_DIR
        )

        # Initialize schema tracking table
        self._init_schema_table()

    def _init_schema_table(self) -> None:
        """Initialize the schema migrations tracking table."""
        if self.db_type == "sqlite":
            self._init_sqlite_schema_table()
        else:
            self._init_duckdb_schema_table()

    def _init_sqlite_schema_table(self) -> None:
        """Create schema_migrations table in SQLite."""
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version VARCHAR(20) PRIMARY KEY,
                    name VARCHAR(200) NOT NULL,
                    description TEXT,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    rollback_count INTEGER DEFAULT 0
                )
            """))
            conn.commit()

    def _init_duckdb_schema_table(self) -> None:
        """Create schema_migrations table in DuckDB."""
        try:
            with DuckDBConnection() as conn:
                conn.execute_query("""
                    CREATE TABLE IF NOT EXISTS schema_migrations (
                        version VARCHAR(20) PRIMARY KEY,
                        name VARCHAR NOT NULL,
                        description VARCHAR,
                        applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        rollback_count INTEGER DEFAULT 0
                    )
                """)
        except Exception as e:
            logger.error(f"Failed to initialize DuckDB schema table: {e}")

    def get_applied_versions(self) -> List[str]:
        """Get list of applied migration versions."""
        if self.db_type == "sqlite":
            with engine.connect() as conn:
                result = conn.execute(text(
                    "SELECT version FROM schema_migrations ORDER BY version"
                ))
                return [row[0] for row in result.fetchall()]
        else:
            try:
                with DuckDBConnection() as conn:
                    result = conn.execute_query(
                        "SELECT version FROM schema_migrations ORDER BY version"
                    )
                    return [row[0] for row in result.fetchall()]
            except Exception:
                return []

    def get_pending_migrations(self) -> List[Migration]:
        """Get list of pending (not yet applied) migrations."""
        applied = set(self.get_applied_versions())
        all_migrations = self._load_all_migrations()

        return [m for m in all_migrations if m.version not in applied]

    def _load_all_migrations(self) -> List[Migration]:
        """Load all migration files from migrations directory."""
        migrations = []

        for migration_file in sorted(self.migrations_dir.glob("*.json")):
            try:
                with open(migration_file, 'r') as f:
                    data = json.load(f)
                    migrations.append(Migration.from_dict(data))
            except Exception as e:
                logger.warning(f"Failed to load migration {migration_file}: {e}")

        return sorted(migrations, key=lambda m: m.version)

    def create_migration(
        self,
        version: str,
        name: str,
        description: str,
        up_sql: str,
        down_sql: str
    ) -> Migration:
        """
        Create a new migration file.

        Args:
            version: Migration version (e.g., "001", "002").
            name: Migration name (snake_case).
            description: Human-readable description.
            up_sql: SQL to apply the migration.
            down_sql: SQL to rollback the migration.

        Returns:
            Created Migration object.
        """
        migration = Migration(
            version=version,
            name=name,
            description=description,
            up_sql=up_sql,
            down_sql=down_sql,
            db_type=self.db_type
        )

        # Save migration to file
        migration_file = self.migrations_dir / f"{version}_{name}.json"
        with open(migration_file, 'w') as f:
            json.dump(migration.to_dict(), f, indent=2)

        logger.info(f"Created migration: {migration_file}")
        return migration

    def migrate(self, target_version: Optional[str] = None) -> bool:
        """
        Run pending migrations up to target version (or all if None).

        Args:
            target_version: Optional target version. If None, runs all pending.

        Returns:
            True if all migrations succeeded, False otherwise.
        """
        pending = self.get_pending_migrations()

        # Filter by target version if specified
        if target_version:
            pending = [m for m in pending if m.version <= target_version]

        if not pending:
            logger.info("No pending migrations to run")
            return True

        logger.info(f"Running {len(pending)} pending migrations...")

        for migration in pending:
            try:
                self._apply_migration(migration)
                logger.info(f"Applied migration {migration.version}: {migration.name}")
            except Exception as e:
                logger.error(f"Migration {migration.version} failed: {e}")
                return False

        return True

    def _apply_migration(self, migration: Migration) -> None:
        """Apply a single migration."""
        if self.db_type == "sqlite":
            self._apply_sqlite_migration(migration)
        else:
            self._apply_duckdb_migration(migration)

    def _apply_sqlite_migration(self, migration: Migration) -> None:
        """Apply migration to SQLite with transaction safety."""
        with engine.begin() as conn:  # begin() for transaction
            # Execute migration SQL
            conn.execute(text(migration.up_sql))

            # Record migration
            conn.execute(text("""
                INSERT INTO schema_migrations (version, name, description)
                VALUES (:version, :name, :description)
            """), {
                'version': migration.version,
                'name': migration.name,
                'description': migration.description
            })

    def _apply_duckdb_migration(self, migration: Migration) -> None:
        """Apply migration to DuckDB."""
        with DuckDBConnection() as conn:
            # Execute migration SQL
            conn.execute_query(migration.up_sql)

            # Record migration
            conn.execute_query("""
                INSERT INTO schema_migrations (version, name, description)
                VALUES (?, ?, ?)
            """, (migration.version, migration.name, migration.description))

    def rollback(self, steps: int = 1, target_version: Optional[str] = None) -> bool:
        """
        Rollback migrations.

        Args:
            steps: Number of migrations to rollback (default: 1).
            target_version: Rollback to this specific version (exclusive).

        Returns:
            True if rollback succeeded, False otherwise.
        """
        applied = self.get_applied_versions()

        if not applied:
            logger.info("No migrations to rollback")
            return True

        # Determine which migrations to rollback
        if target_version:
            to_rollback = [v for v in applied if v > target_version]
        else:
            to_rollback = applied[-steps:] if steps > 0 else []

        if not to_rollback:
            logger.info("No migrations to rollback")
            return True

        logger.info(f"Rolling back {len(to_rollback)} migrations...")

        # Rollback in reverse order
        for version in reversed(to_rollback):
            try:
                migration = self._get_migration(version)
                if migration:
                    self._rollback_migration(migration)
                    logger.info(f"Rolled back migration {version}: {migration.name}")
                else:
                    logger.warning(f"Migration file for version {version} not found")
            except Exception as e:
                logger.error(f"Rollback of {version} failed: {e}")
                return False

        return True

    def _rollback_migration(self, migration: Migration) -> None:
        """Rollback a single migration."""
        if self.db_type == "sqlite":
            self._rollback_sqlite_migration(migration)
        else:
            self._rollback_duckdb_migration(migration)

    def _rollback_sqlite_migration(self, migration: Migration) -> None:
        """Rollback migration in SQLite with transaction safety."""
        with engine.begin() as conn:  # begin() for transaction
            # Execute rollback SQL
            conn.execute(text(migration.down_sql))

            # Remove migration record
            conn.execute(text("""
                DELETE FROM schema_migrations WHERE version = :version
            """), {'version': migration.version})

    def _rollback_duckdb_migration(self, migration: Migration) -> None:
        """Rollback migration in DuckDB."""
        with DuckDBConnection() as conn:
            # Execute rollback SQL
            conn.execute_query(migration.down_sql)

            # Remove migration record
            conn.execute_query(
                "DELETE FROM schema_migrations WHERE version = ?",
                (migration.version,)
            )

    def _get_migration(self, version: str) -> Optional[Migration]:
        """Load a specific migration by version."""
        for migration_file in self.migrations_dir.glob(f"{version}_*.json"):
            try:
                with open(migration_file, 'r') as f:
                    data = json.load(f)
                    return Migration.from_dict(data)
            except Exception:
                continue
        return None

    def get_migration_status(self) -> Dict[str, Any]:
        """
        Get current migration status.

        Returns:
            Dictionary with migration status information.
        """
        applied = self.get_applied_versions()
        pending = self.get_pending_migrations()

        return {
            'db_type': self.db_type,
            'applied_count': len(applied),
            'pending_count': len(pending),
            'applied_versions': applied,
            'pending_versions': [m.version for m in pending],
            'latest_applied': applied[-1] if applied else None
        }

    def generate_migration_sql(
        self,
        table_name: str,
        operation: str = "create"
    ) -> tuple[str, str]:
        """
        Generate migration SQL for common operations.

        Args:
            table_name: Name of the table.
            operation: Type of operation ('create', 'drop', 'add_column').

        Returns:
            Tuple of (up_sql, down_sql).
        """
        if operation == "create":
            up_sql = f"CREATE TABLE {table_name} (\n    id INTEGER PRIMARY KEY\n)"
            down_sql = f"DROP TABLE IF EXISTS {table_name}"
        elif operation == "drop":
            up_sql = f"DROP TABLE IF EXISTS {table_name}"
            down_sql = f"-- Cannot rollback DROP TABLE operation"
        else:
            up_sql = f"-- Custom operation on {table_name}"
            down_sql = f"-- Rollback for custom operation"

        return up_sql, down_sql


# Convenience functions

def run_sqlite_migrations(target_version: Optional[str] = None) -> bool:
    """
    Run pending SQLite migrations.

    Args:
        target_version: Optional target version.

    Returns:
        True if all migrations succeeded.
    """
    runner = MigrationRunner(db_type="sqlite")
    return runner.migrate(target_version=target_version)


def run_duckdb_migrations(target_version: Optional[str] = None) -> bool:
    """
    Run pending DuckDB migrations.

    Args:
        target_version: Optional target version.

    Returns:
        True if all migrations succeeded.
    """
    runner = MigrationRunner(db_type="duckdb")
    return runner.migrate(target_version=target_version)


def rollback_sqlite(steps: int = 1, target_version: Optional[str] = None) -> bool:
    """
    Rollback SQLite migrations.

    Args:
        steps: Number of migrations to rollback.
        target_version: Rollback to this version (exclusive).

    Returns:
        True if rollback succeeded.
    """
    runner = MigrationRunner(db_type="sqlite")
    return runner.rollback(steps=steps, target_version=target_version)


def rollback_duckdb(steps: int = 1, target_version: Optional[str] = None) -> bool:
    """
    Rollback DuckDB migrations.

    Args:
        steps: Number of migrations to rollback.
        target_version: Rollback to this version (exclusive).

    Returns:
        True if rollback succeeded.
    """
    runner = MigrationRunner(db_type="duckdb")
    return runner.rollback(steps=steps, target_version=target_version)


def get_migration_status() -> Dict[str, Any]:
    """
    Get migration status for both databases.

    Returns:
        Dictionary with status for SQLite and DuckDB.
    """
    sqlite_runner = MigrationRunner(db_type="sqlite")
    duckdb_runner = MigrationRunner(db_type="duckdb")

    return {
        'sqlite': sqlite_runner.get_migration_status(),
        'duckdb': duckdb_runner.get_migration_status()
    }


# Initial migration for Task Group 2 tables
def create_task_group_2_migrations() -> None:
    """Create initial migrations for Task Group 2 tables."""
    logger.info("Creating Task Group 2 migrations...")

    # SQLite migrations
    sqlite_runner = MigrationRunner(db_type="sqlite")

    # Note: These migrations are already applied via models.py
    # This function is for reference when adding new migrations

    logger.info("Task Group 2 migrations ready")


if __name__ == "__main__":
    # Run migrations when executed directly
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "status":
            status = get_migration_status()
            print("\nMigration Status:")
            print(f"  SQLite: {status['sqlite']['applied_count']} applied, {status['sqlite']['pending_count']} pending")
            print(f"  DuckDB: {status['duckdb']['applied_count']} applied, {status['duckdb']['pending_count']} pending")

        elif command == "migrate":
            sqlite_success = run_sqlite_migrations()
            duckdb_success = run_duckdb_migrations()

            if sqlite_success and duckdb_success:
                print("\nAll migrations completed successfully!")
            else:
                print("\nSome migrations failed. Check logs for details.")
                sys.exit(1)

        elif command == "rollback":
            steps = int(sys.argv[2]) if len(sys.argv) > 2 else 1
            rollback_sqlite(steps=steps)
            rollback_duckdb(steps=steps)
            print(f"\nRolled back {steps} migration(s)")

        else:
            print(f"Unknown command: {command}")
            print("Usage: python migration_runner.py [status|migrate|rollback [steps]]")
            sys.exit(1)
    else:
        print("Usage: python migration_runner.py [status|migrate|rollback [steps]]")

"""
QuantMindLib V1 — Core Migrations Package

Provides lightweight SQLite migration management for the library's own
persistence layer (BotRegistry, etc.) using Python's sqlite3 standard library.

Design goals:
- No SQLAlchemy dependency in migration path (lightweight, standalone)
- Version tracking via schema_migrations table
- Migrations stored as .sql files for human readability
- Python MigrationManager class for programmatic control

Reference:
  docs/planning/quantmindlib/13_data_lifecycle_management.md
"""

from __future__ import annotations

import re
import sqlite3
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_MIGRATIONS_DIR = Path(__file__).parent
_SCHEMA_MIGRATIONS_TABLE = "schema_migrations"


# ---------------------------------------------------------------------------
# SQL snippets
# ---------------------------------------------------------------------------

_INIT_SCHEMA_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {_SCHEMA_MIGRATIONS_TABLE} (
    version     TEXT    NOT NULL PRIMARY KEY,
    applied_at_ms INTEGER NOT NULL
);
"""


def _split_migration(sql: str) -> tuple[str, str]:
    """
    Split a migration SQL file into UP and DOWN blocks.

    Looks for a separator line where the first token after '--' is 'DOWN'
    with no additional punctuation (e.g. '-- DOWN Migration' or '-- DOWN').
    This distinguishes the actual separator from header comments like
    '-- DOWN: Drops bot_registry table'.
    Everything before the separator is the UP block. Everything after is
    the DOWN block. Returns (up_sql, down_sql). Empty blocks return ''.
    """
    up_parts: list[str] = []
    down_parts: list[str] = []
    in_down = False

    for raw_line in sql.splitlines():
        stripped = raw_line.strip()
        # Match separator: '-- DOWN' followed by whitespace or end-of-line
        # Reject '-- DOWN:' or '-- DOWN: description' (header comments)
        # Accept: '-- DOWN Migration', '-- DOWN', '--  DOWN  '
        after_dash = stripped[2:].strip() if len(stripped) > 2 else ""
        is_separator = (
            after_dash.upper().startswith("DOWN")
            and (
                len(after_dash) == 4  # exactly "-- DOWN"
                or after_dash[4:5] == " "  # space after DOWN (e.g. "-- DOWN Migration")
                or len(after_dash) == 0  # edge case
            )
        )
        if is_separator:
            in_down = True
            continue

        if in_down:
            down_parts.append(raw_line)
        else:
            up_parts.append(raw_line)

    up_sql = "\n".join(up_parts).strip()
    down_sql = "\n".join(down_parts).strip()
    return up_sql, down_sql


# ---------------------------------------------------------------------------
# MigrationManager
# ---------------------------------------------------------------------------

class MigrationManager:
    """
    Lightweight SQLite migration manager for QuantMindLib core tables.

    Features:
    - Reads migrations from .sql files in this directory
    - UP / DOWN blocks separated by '-- DOWN' comment line
    - schema_migrations table tracks applied version
    - Duplicate migration detection (already applied → no-op)
    - Idempotent: running the same migration twice is safe

    Usage:
        conn = sqlite3.connect(":memory:")
        mgr = MigrationManager(conn)
        mgr.apply_migration()

        # Or use convenience
        apply_migration(conn)
        get_applied_version(conn)  # -> "001" or None
        get_migration_sql("001")    # -> SQL string
    """

    _VERSION_RE = re.compile(r"^\d{3}_(.+)\.sql$")

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    # --- Schema table management -------------------------------------------

    def _ensure_schema_table(self) -> None:
        """Create schema_migrations if it does not exist."""
        self.conn.execute(_INIT_SCHEMA_TABLE_SQL)
        self.conn.commit()

    # --- Version retrieval -------------------------------------------------

    def get_applied_version(self) -> Optional[str]:
        """
        Return the currently applied migration version, or None if no
        migrations have been applied yet.

        Only one migration version is stored per database (sequential model).
        The schema_migrations table may contain multiple rows after rollback
        scenarios, so we return the lexicographically greatest version.
        """
        self._ensure_schema_table()
        cur = self.conn.execute(
            f"SELECT MAX(version) FROM {_SCHEMA_MIGRATIONS_TABLE}"
        )
        row = cur.fetchone()
        return row[0] if row and row[0] else None

    # --- SQL retrieval -----------------------------------------------------

    @classmethod
    def get_migration_sql(cls, version: str) -> str:
        """
        Read and return the full SQL content of a migration file for the
        given version.

        Version must be a 3-digit string (e.g. "001"). The method looks for
        ``<version>_*.sql`` in the migrations directory.

        Raises:
            FileNotFoundError: if no matching migration file exists.
        """
        migrations_dir = _MIGRATIONS_DIR
        candidates = sorted(migrations_dir.glob(f"{version}_*.sql"))
        if not candidates:
            raise FileNotFoundError(
                f"No migration file found for version {version} in {migrations_dir}"
            )
        return candidates[0].read_text()

    # --- Migration discovery -----------------------------------------------

    @classmethod
    def _parse_version(cls, filename: str) -> Optional[str]:
        """Extract the 3-digit version prefix from a migration filename."""
        m = cls._VERSION_RE.match(filename)
        # Return the 3-digit prefix (e.g. "001" from "001_create_bot_registry.sql")
        return m.group(0)[:3] if m else None

    @classmethod
    def get_pending_version(cls) -> Optional[str]:
        """
        Return the version string of the next pending migration, or None if
        all migrations are already applied.

        Compares available .sql files against the current schema version
        using lexicographic string comparison (001 < 002 < ... < 010).
        """
        migrations_dir = _MIGRATIONS_DIR
        available = sorted(
            (cls._parse_version(f.name), f.name)
            for f in migrations_dir.glob("*.sql")
            if cls._parse_version(f.name) is not None
        )
        if not available:
            return None

        # Use an in-memory DB to let MigrationManager determine current state
        temp_conn = sqlite3.connect(":memory:")
        temp_conn.executescript(_INIT_SCHEMA_TABLE_SQL)
        temp_mgr = cls(temp_conn)
        current = temp_mgr.get_applied_version()

        for version, _ in available:
            if current is None or version > current:
                return version
        return None

    # --- Apply / Rollback --------------------------------------------------

    def apply_migration(self) -> bool:
        """
        Apply the next pending migration.

        Execution:
        1. Ensure schema_migrations table exists
        2. Determine current applied version
        3. Discover next pending migration from .sql files
        4. Parse UP / DOWN blocks from the file
        5. Execute UP SQL within a transaction
        6. Record version in schema_migrations
        7. Return True on success

        Returns:
            True if a migration was applied, False if nothing was pending.

        Raises:
            RuntimeError: if a migration file cannot be read.
        """
        self._ensure_schema_table()
        current = self.get_applied_version()

        migrations_dir = _MIGRATIONS_DIR
        available = sorted(
            (self._parse_version(f.name), f.name)
            for f in migrations_dir.glob("*.sql")
            if self._parse_version(f.name) is not None
        )

        next_version: Optional[str] = None
        for version, filename in available:
            if current is None or version > current:
                next_version = version
                break

        if next_version is None:
            return False  # No pending migration

        sql = (migrations_dir / filename).read_text()
        up_sql, _ = _split_migration(sql)

        now_ms = int(time.time() * 1000)
        with self.conn:
            self.conn.executescript(up_sql)
            self.conn.execute(
                f"INSERT INTO {_SCHEMA_MIGRATIONS_TABLE} (version, applied_at_ms) VALUES (?, ?)",
                (next_version, now_ms),
            )
            self.conn.commit()

        return True

    def rollback_migration(self) -> bool:
        """
        Roll back the last applied migration.

        Execution:
        1. Ensure schema_migrations table exists
        2. Determine current applied version
        3. Load the migration file for that version
        4. Parse DOWN SQL block
        5. Execute DOWN SQL within a transaction
        6. Delete the version record from schema_migrations
        7. Return True on success

        Returns:
            True if a migration was rolled back, False if nothing was applied.

        Raises:
            RuntimeError: if migration file cannot be read.
        """
        self._ensure_schema_table()
        current = self.get_applied_version()

        if current is None:
            return False  # Nothing to roll back

        sql = self.get_migration_sql(current)
        _, down_sql = _split_migration(sql)

        with self.conn:
            if down_sql:
                self.conn.executescript(down_sql)
            self.conn.execute(
                f"DELETE FROM {_SCHEMA_MIGRATIONS_TABLE} WHERE version = ?",
                (current,),
            )
            self.conn.commit()

        return True


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def get_migration_sql(version: str) -> str:
    """Read and return the full SQL content for a given version."""
    return MigrationManager.get_migration_sql(version)


def get_applied_version(conn: sqlite3.Connection) -> Optional[str]:
    """Return the currently applied migration version for the connection."""
    return MigrationManager(conn).get_applied_version()


def apply_migration(conn: sqlite3.Connection) -> bool:
    """Apply the next pending migration on the given connection."""
    return MigrationManager(conn).apply_migration()


def rollback_migration(conn: sqlite3.Connection) -> bool:
    """Roll back the last migration on the given connection."""
    return MigrationManager(conn).rollback_migration()
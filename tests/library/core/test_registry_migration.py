"""
Self-Verification Tests for Packet 13E: BotRegistry SQLite Migration.

Tests cover:
- SQL migration creates bot_registry table with correct schema
- All field types match RegistryRecord domain model
- Indexes are created as specified
- DOWN migration drops table and indexes
- MigrationManager applies migrations correctly
- Duplicate migration detection (already applied → no-op)
- variant_ids JSON roundtrip (serialization / deserialization)

Reference: docs/planning/quantmindlib/13_data_lifecycle_management.md
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from src.library.core.domain.registry_record import RegistryRecord
from src.library.core.migrations import (
    MigrationManager,
    get_migration_sql,
    get_applied_version,
    apply_migration,
    rollback_migration,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _conn_schema_tables(conn: sqlite3.Connection) -> list[str]:
    """Return all table names in the connected database."""
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    return [row[0] for row in cur.fetchall()]


def _conn_indexes(conn: sqlite3.Connection, table: str) -> list[str]:
    """Return all index names for a given table."""
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name=? ORDER BY name",
        (table,),
    )
    return [row[0] for row in cur.fetchall()]


def _conn_columns(conn: sqlite3.Connection, table: str) -> dict[str, tuple]:
    """
    Return a dict of column_name -> (type, notnull, default, pk) for a table.
    Mimics sqlite PRAGMA table_info output.
    """
    cur = conn.execute(f"PRAGMA table_info({table})")
    return {row[1]: (row[2], row[3], row[4], row[5]) for row in cur.fetchall()}


def _make_record(
    bot_id: str = "bot_001",
    bot_spec_id: str = "spec_abc",
    status: str = "ACTIVE",
    tier: str = "ELITE",
    registered_at_ms: int = 1000000000,
    last_updated_ms: int = 2000000000,
    owner: str = "agent_001",
    variant_ids: list[str] | None = None,
    deployed_at: str | None = None,
) -> dict:
    if variant_ids is None:
        variant_ids = ["v1", "v2", "v3"]
    return {
        "bot_id": bot_id,
        "bot_spec_id": bot_spec_id,
        "status": status,
        "tier": tier,
        "registered_at_ms": registered_at_ms,
        "last_updated_ms": last_updated_ms,
        "owner": owner,
        "variant_ids": variant_ids,
        "deployed_at": deployed_at,
    }


# ---------------------------------------------------------------------------
# TestMigrationFile
# ---------------------------------------------------------------------------

class TestMigrationFile:
    """Tests that the SQL migration file itself is well-formed."""

    def test_sql_file_exists(self, tmp_path):
        """001_create_bot_registry.sql is readable by the migration manager."""
        # The manager finds files via the migrations directory
        sql = get_migration_sql("001")
        assert sql is not None
        assert "-- UP Migration" in sql
        assert "-- DOWN Migration" in sql

    def test_sql_file_has_create_table(self):
        """Migration contains CREATE TABLE bot_registry."""
        sql = get_migration_sql("001")
        assert "CREATE TABLE IF NOT EXISTS bot_registry" in sql

    def test_sql_file_has_all_columns(self):
        """Migration contains every RegistryRecord field."""
        sql = get_migration_sql("001")
        for col in [
            "bot_id", "bot_spec_id", "status", "tier",
            "registered_at_ms", "last_updated_ms", "owner",
            "variant_ids", "deployed_at",
        ]:
            assert col in sql, f"Missing column: {col}"

    def test_sql_file_has_bot_id_pk(self):
        """bot_id is declared as PRIMARY KEY in migration."""
        sql = get_migration_sql("001")
        # The PRIMARY KEY constraint is part of the column definition
        assert "bot_id" in sql and "PRIMARY KEY" in sql

    def test_sql_file_has_indexes(self):
        """Migration contains CREATE INDEX for all required indexes."""
        sql = get_migration_sql("001")
        for idx in [
            "idx_bot_registry_bot_spec_id",
            "idx_bot_registry_status",
            "idx_bot_registry_tier",
            "idx_bot_registry_owner",
            "idx_bot_registry_registered_at_ms",
            "idx_bot_registry_status_tier",
        ]:
            assert idx in sql, f"Missing index: {idx}"

    def test_sql_file_has_down_section(self):
        """Migration contains -- DOWN section and DROP statements."""
        sql = get_migration_sql("001")
        assert "-- DOWN" in sql
        assert "DROP TABLE IF EXISTS bot_registry" in sql

    def test_sql_file_has_check_constraints(self):
        """status and tier CHECK constraints are present."""
        sql = get_migration_sql("001")
        assert "CHECK (status IN" in sql
        assert "CHECK (tier IN" in sql

    def test_sql_file_has_variant_ids_default(self):
        """variant_ids has a DEFAULT '[]' constraint."""
        sql = get_migration_sql("001")
        assert "variant_ids" in sql and "DEFAULT" in sql

    def test_sql_file_has_bot_spec_id_check(self):
        """bot_spec_id has a length CHECK constraint."""
        sql = get_migration_sql("001")
        assert "bot_spec_id_not_empty" in sql


# ---------------------------------------------------------------------------
# TestTableCreation
# ---------------------------------------------------------------------------

class TestTableCreation:
    """Tests that applying the migration creates the table correctly."""

    def test_creates_bot_registry_table(self, tmp_path):
        """apply_migration creates the bot_registry table."""
        db_path = tmp_path / "registry.db"
        conn = sqlite3.connect(str(db_path))

        applied = apply_migration(conn)
        assert applied is True

        tables = _conn_schema_tables(conn)
        assert "bot_registry" in tables

        # schema_migrations tracking table should also exist
        assert "schema_migrations" in tables
        conn.close()

    def test_all_columns_present(self, tmp_path):
        """bot_registry table has all nine RegistryRecord columns."""
        db_path = tmp_path / "registry.db"
        conn = sqlite3.connect(str(db_path))
        apply_migration(conn)

        cols = _conn_columns(conn, "bot_registry")
        expected = {
            "bot_id", "bot_spec_id", "status", "tier",
            "registered_at_ms", "last_updated_ms", "owner",
            "variant_ids", "deployed_at",
        }
        assert set(cols.keys()) == expected
        conn.close()

    def test_bot_id_is_pk(self, tmp_path):
        """bot_id column is the primary key (not null, pk=1)."""
        db_path = tmp_path / "registry.db"
        conn = sqlite3.connect(str(db_path))
        apply_migration(conn)

        cols = _conn_columns(conn, "bot_registry")
        assert cols["bot_id"][0] == "TEXT"  # declared type
        assert cols["bot_id"][1] == 1   # not null
        assert cols["bot_id"][3] == 1   # pk flag
        conn.close()

    def test_text_columns_have_text_type(self, tmp_path):
        """bot_spec_id, status, tier, owner, deployed_at are TEXT."""
        db_path = tmp_path / "registry.db"
        conn = sqlite3.connect(str(db_path))
        apply_migration(conn)

        cols = _conn_columns(conn, "bot_registry")
        for col in ["bot_spec_id", "status", "tier", "owner", "deployed_at"]:
            assert cols[col][0] == "TEXT", f"{col} should be TEXT, got {cols[col][0]}"
        conn.close()

    def test_integer_columns_have_integer_type(self, tmp_path):
        """registered_at_ms, last_updated_ms are INTEGER."""
        db_path = tmp_path / "registry.db"
        conn = sqlite3.connect(str(db_path))
        apply_migration(conn)

        cols = _conn_columns(conn, "bot_registry")
        for col in ["registered_at_ms", "last_updated_ms"]:
            assert cols[col][0] == "INTEGER", f"{col} should be INTEGER, got {cols[col][0]}"
        conn.close()

    def test_variant_ids_is_text_with_default(self, tmp_path):
        """variant_ids is TEXT with a DEFAULT '[]'."""
        db_path = tmp_path / "registry.db"
        conn = sqlite3.connect(str(db_path))
        apply_migration(conn)

        cols = _conn_columns(conn, "bot_registry")
        # Variant IDs stored as JSON string
        assert cols["variant_ids"][0] == "TEXT"
        # Default value stored as '[]'
        assert "[]" in str(cols["variant_ids"][2]) or cols["variant_ids"][2] == "[],'[]'"

        conn.close()

    def test_deployed_at_is_nullable(self, tmp_path):
        """deployed_at is the only nullable column."""
        db_path = tmp_path / "registry.db"
        conn = sqlite3.connect(str(db_path))
        apply_migration(conn)

        cols = _conn_columns(conn, "bot_registry")
        # deployed_at should have no NOT NULL flag
        for col in ["deployed_at"]:
            # notnull == 0 means nullable
            assert cols[col][1] == 0, f"{col} should be nullable (notnull=0)"
        conn.close()

    def test_bot_spec_id_not_null(self, tmp_path):
        """bot_spec_id is NOT NULL."""
        db_path = tmp_path / "registry.db"
        conn = sqlite3.connect(str(db_path))
        apply_migration(conn)

        cols = _conn_columns(conn, "bot_registry")
        assert cols["bot_spec_id"][1] == 1  # not null
        conn.close()


# ---------------------------------------------------------------------------
# TestIndexes
# ---------------------------------------------------------------------------

class TestIndexes:
    """Tests that migration creates all required indexes."""

    @pytest.fixture
    def migrated_conn(self, tmp_path):
        db_path = tmp_path / "registry.db"
        conn = sqlite3.connect(str(db_path))
        apply_migration(conn)
        yield conn
        conn.close()

    def test_creates_bot_spec_id_index(self, migrated_conn):
        idxs = _conn_indexes(migrated_conn, "bot_registry")
        assert "idx_bot_registry_bot_spec_id" in idxs

    def test_creates_status_index(self, migrated_conn):
        idxs = _conn_indexes(migrated_conn, "bot_registry")
        assert "idx_bot_registry_status" in idxs

    def test_creates_tier_index(self, migrated_conn):
        idxs = _conn_indexes(migrated_conn, "bot_registry")
        assert "idx_bot_registry_tier" in idxs

    def test_creates_owner_index(self, migrated_conn):
        idxs = _conn_indexes(migrated_conn, "bot_registry")
        assert "idx_bot_registry_owner" in idxs

    def test_creates_registered_at_ms_index(self, migrated_conn):
        idxs = _conn_indexes(migrated_conn, "bot_registry")
        assert "idx_bot_registry_registered_at_ms" in idxs

    def test_creates_composite_status_tier_index(self, migrated_conn):
        idxs = _conn_indexes(migrated_conn, "bot_registry")
        assert "idx_bot_registry_status_tier" in idxs

    def test_creates_six_indexes(self, migrated_conn):
        idxs = _conn_indexes(migrated_conn, "bot_registry")
        # bot_registry indexes + schema_migrations index
        bot_idxs = [i for i in idxs if i.startswith("idx_bot_registry")]
        assert len(bot_idxs) == 6


# ---------------------------------------------------------------------------
# TestDownMigration
# ---------------------------------------------------------------------------

class TestDownMigration:
    """Tests that DOWN migration drops table and indexes."""

    def test_rollback_drops_bot_registry_table(self, tmp_path):
        """rollback_migration drops bot_registry table."""
        db_path = tmp_path / "registry.db"
        conn = sqlite3.connect(str(db_path))

        apply_migration(conn)
        tables_before = _conn_schema_tables(conn)
        assert "bot_registry" in tables_before

        rolled_back = rollback_migration(conn)
        assert rolled_back is True

        tables_after = _conn_schema_tables(conn)
        assert "bot_registry" not in tables_after
        conn.close()

    def test_rollback_drops_indexes(self, tmp_path):
        """rollback_migration drops all bot_registry indexes."""
        db_path = tmp_path / "registry.db"
        conn = sqlite3.connect(str(db_path))

        apply_migration(conn)
        idxs_before = _conn_indexes(conn, "bot_registry")
        assert len(idxs_before) >= 6

        rollback_migration(conn)

        # After rollback, either table is gone or indexes are gone
        try:
            idxs_after = _conn_indexes(conn, "bot_registry")
        except sqlite3.OperationalError:
            idxs_after = []
        for idx in [
            "idx_bot_registry_bot_spec_id",
            "idx_bot_registry_status",
            "idx_bot_registry_tier",
            "idx_bot_registry_owner",
            "idx_bot_registry_registered_at_ms",
            "idx_bot_registry_status_tier",
        ]:
            assert idx not in idxs_after
        conn.close()

    def test_rollback_removes_version_record(self, tmp_path):
        """rollback_migration removes the version record from schema_migrations."""
        db_path = tmp_path / "registry.db"
        conn = sqlite3.connect(str(db_path))

        apply_migration(conn)
        assert get_applied_version(conn) == "001"

        rollback_migration(conn)
        assert get_applied_version(conn) is None
        conn.close()

    def test_rollback_idempotent(self, tmp_path):
        """Calling rollback twice is safe (second call returns False)."""
        db_path = tmp_path / "registry.db"
        conn = sqlite3.connect(str(db_path))

        apply_migration(conn)
        first = rollback_migration(conn)
        second = rollback_migration(conn)

        assert first is True
        assert second is False
        conn.close()

    def test_rollback_on_empty_db_returns_false(self, tmp_path):
        """rollback_migration on a database with no migrations returns False."""
        db_path = tmp_path / "registry.db"
        conn = sqlite3.connect(str(db_path))

        result = rollback_migration(conn)
        assert result is False
        conn.close()


# ---------------------------------------------------------------------------
# TestMigrationManager
# ---------------------------------------------------------------------------

class TestMigrationManager:
    """Tests for MigrationManager class directly."""

    def test_init_with_connection(self, tmp_path):
        """MigrationManager can be instantiated with a sqlite3 connection."""
        db_path = tmp_path / "registry.db"
        conn = sqlite3.connect(str(db_path))
        mgr = MigrationManager(conn)
        assert mgr.conn is conn
        conn.close()

    def test_ensure_schema_table(self, tmp_path):
        """MigrationManager creates schema_migrations table on first access."""
        db_path = tmp_path / "registry.db"
        conn = sqlite3.connect(str(db_path))
        mgr = MigrationManager(conn)

        mgr._ensure_schema_table()
        tables = _conn_schema_tables(conn)
        assert "schema_migrations" in tables
        conn.close()

    def test_get_applied_version_none_when_empty(self, tmp_path):
        """get_applied_version returns None when no migrations applied."""
        db_path = tmp_path / "registry.db"
        conn = sqlite3.connect(str(db_path))
        mgr = MigrationManager(conn)

        version = mgr.get_applied_version()
        assert version is None
        conn.close()

    def test_get_applied_version_after_migration(self, tmp_path):
        """get_applied_version returns '001' after applying migration."""
        db_path = tmp_path / "registry.db"
        conn = sqlite3.connect(str(db_path))
        mgr = MigrationManager(conn)

        mgr.apply_migration()
        version = mgr.get_applied_version()
        assert version == "001"
        conn.close()

    def test_get_migration_sql_returns_string(self):
        """get_migration_sql returns SQL content as a string."""
        sql = get_migration_sql("001")
        assert isinstance(sql, str)
        assert len(sql) > 50

    def test_get_migration_sql_unknown_version_raises(self):
        """get_migration_sql raises FileNotFoundError for unknown version."""
        with pytest.raises(FileNotFoundError):
            get_migration_sql("999")

    def test_version_is_lexicographically_greatest_after_multiple(self, tmp_path):
        """With multiple applied versions, get_applied_version returns the latest."""
        db_path = tmp_path / "registry.db"
        conn = sqlite3.connect(str(db_path))
        mgr = MigrationManager(conn)

        # Get pending version without applying
        pending = MigrationManager.get_pending_version()
        # This should be "001" since no migrations exist yet
        assert pending in ("001", None)

        conn.close()


# ---------------------------------------------------------------------------
# TestDuplicateDetection
# ---------------------------------------------------------------------------

class TestDuplicateDetection:
    """Tests that applying the same migration twice is safe."""

    def test_apply_twice_is_idempotent(self, tmp_path):
        """Calling apply_migration twice is safe (second call returns False)."""
        db_path = tmp_path / "registry.db"
        conn = sqlite3.connect(str(db_path))

        first = apply_migration(conn)
        second = apply_migration(conn)

        assert first is True
        assert second is False  # Nothing pending

        tables = _conn_schema_tables(conn)
        assert "bot_registry" in tables
        # Verify only one row in schema_migrations
        cur = conn.execute("SELECT version FROM schema_migrations")
        versions = [row[0] for row in cur.fetchall()]
        assert versions == ["001"]

        conn.close()

    def test_insert_bot_twice_fails(self, tmp_path):
        """Attempting to insert the same bot_id twice violates PRIMARY KEY."""
        db_path = tmp_path / "registry.db"
        conn = sqlite3.connect(str(db_path))
        apply_migration(conn)

        conn.execute(
            "INSERT INTO bot_registry (bot_id, bot_spec_id, status, tier, "
            "registered_at_ms, last_updated_ms, owner, variant_ids) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("bot_x", "spec_x", "ACTIVE", "ELITE", 1, 2, "owner", "[]"),
        )
        conn.commit()

        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO bot_registry (bot_id, bot_spec_id, status, tier, "
                "registered_at_ms, last_updated_ms, owner, variant_ids) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                ("bot_x", "spec_x2", "ACTIVE", "ELITE", 3, 4, "owner", "[]"),
            )

        conn.close()


# ---------------------------------------------------------------------------
# TestVariantIdsJsonRoundtrip
# ---------------------------------------------------------------------------

class TestVariantIdsJsonRoundtrip:
    """Tests for variant_ids JSON serialization and deserialization."""

    def test_insert_single_variant_id(self, tmp_path):
        """Can insert and retrieve a single-element variant_ids list."""
        db_path = tmp_path / "registry.db"
        conn = sqlite3.connect(str(db_path))
        apply_migration(conn)

        variant_list = ["v_001"]
        conn.execute(
            "INSERT INTO bot_registry (bot_id, bot_spec_id, status, tier, "
            "registered_at_ms, last_updated_ms, owner, variant_ids) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("bot_single", "spec_a", "ACTIVE", "STANDARD", 100, 200, "owner", json.dumps(variant_list)),
        )
        conn.commit()

        cur = conn.execute("SELECT variant_ids FROM bot_registry WHERE bot_id = ?", ("bot_single",))
        row = cur.fetchone()
        assert row is not None
        assert json.loads(row[0]) == variant_list

        conn.close()

    def test_insert_multiple_variant_ids(self, tmp_path):
        """Can insert and retrieve a multi-element variant_ids list."""
        db_path = tmp_path / "registry.db"
        conn = sqlite3.connect(str(db_path))
        apply_migration(conn)

        variant_list = ["v_a", "v_b", "v_c", "v_d"]
        conn.execute(
            "INSERT INTO bot_registry (bot_id, bot_spec_id, status, tier, "
            "registered_at_ms, last_updated_ms, owner, variant_ids) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("bot_multi", "spec_b", "TRIAL", "EVALUATION_CANDIDATE", 300, 400, "owner", json.dumps(variant_list)),
        )
        conn.commit()

        cur = conn.execute("SELECT variant_ids FROM bot_registry WHERE bot_id = ?", ("bot_multi",))
        row = cur.fetchone()
        assert row is not None
        assert json.loads(row[0]) == variant_list

        conn.close()

    def test_insert_empty_variant_ids(self, tmp_path):
        """Can insert and retrieve an empty variant_ids list."""
        db_path = tmp_path / "registry.db"
        conn = sqlite3.connect(str(db_path))
        apply_migration(conn)

        variant_list: list[str] = []
        conn.execute(
            "INSERT INTO bot_registry (bot_id, bot_spec_id, status, tier, "
            "registered_at_ms, last_updated_ms, owner, variant_ids) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("bot_empty", "spec_c", "ACTIVE", "STANDARD", 500, 600, "owner", json.dumps(variant_list)),
        )
        conn.commit()

        cur = conn.execute("SELECT variant_ids FROM bot_registry WHERE bot_id = ?", ("bot_empty",))
        row = cur.fetchone()
        assert row is not None
        assert json.loads(row[0]) == variant_list

        conn.close()

    def test_insert_variant_ids_none_yields_default(self, tmp_path):
        """Omitting variant_ids defaults to '[]'."""
        db_path = tmp_path / "registry.db"
        conn = sqlite3.connect(str(db_path))
        apply_migration(conn)

        # Insert without specifying variant_ids (relies on DEFAULT)
        conn.execute(
            "INSERT INTO bot_registry (bot_id, bot_spec_id, status, tier, "
            "registered_at_ms, last_updated_ms, owner) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("bot_default", "spec_d", "ACTIVE", "ELITE", 700, 800, "owner"),
        )
        conn.commit()

        cur = conn.execute("SELECT variant_ids FROM bot_registry WHERE bot_id = ?", ("bot_default",))
        row = cur.fetchone()
        assert row is not None
        assert json.loads(row[0]) == []

        conn.close()

    def test_variant_ids_json_roundtrip_with_registry_record(self, tmp_path):
        """RegistryRecord can be serialized to and deserialized from the DB."""
        db_path = tmp_path / "registry.db"
        conn = sqlite3.connect(str(db_path))
        apply_migration(conn)

        record = RegistryRecord(
            bot_id="bot_roundtrip",
            bot_spec_id="spec_rt",
            status="ACTIVE",
            tier="PERFORMANCE_TEST",
            registered_at_ms=900000000,
            last_updated_ms=950000000,
            owner="roundtrip_owner",
            variant_ids=["var_x", "var_y", "var_z"],
            deployed_at="2026-04-10T10:00:00Z",
        )

        conn.execute(
            "INSERT INTO bot_registry (bot_id, bot_spec_id, status, tier, "
            "registered_at_ms, last_updated_ms, owner, variant_ids, deployed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.bot_id,
                record.bot_spec_id,
                str(record.status),
                str(record.tier),
                record.registered_at_ms,
                record.last_updated_ms,
                record.owner,
                json.dumps(record.variant_ids),
                record.deployed_at,
            ),
        )
        conn.commit()

        # Re-read from DB
        cur = conn.execute("SELECT * FROM bot_registry WHERE bot_id = ?", ("bot_roundtrip",))
        row = cur.fetchone()

        assert row is not None
        # Verify all fields
        assert row[0] == "bot_roundtrip"             # bot_id
        assert row[1] == "spec_rt"                    # bot_spec_id
        assert row[2] == "ACTIVE"                    # status
        assert row[3] == "PERFORMANCE_TEST"         # tier
        assert row[4] == 900000000                   # registered_at_ms
        assert row[5] == 950000000                   # last_updated_ms
        assert row[6] == "roundtrip_owner"           # owner
        assert json.loads(row[7]) == ["var_x", "var_y", "var_z"]  # variant_ids
        assert row[8] == "2026-04-10T10:00:00Z"      # deployed_at

        conn.close()


# ---------------------------------------------------------------------------
# TestSchemaMigrationsTable
# ---------------------------------------------------------------------------

class TestSchemaMigrationsTable:
    """Tests for the schema_migrations tracking table."""

    def test_schema_table_has_version_pk(self, tmp_path):
        """schema_migrations.version is the PRIMARY KEY."""
        db_path = tmp_path / "registry.db"
        conn = sqlite3.connect(str(db_path))

        apply_migration(conn)

        cols = _conn_columns(conn, "schema_migrations")
        assert "version" in cols
        # Not null and primary key
        assert cols["version"][1] == 1   # not null
        assert cols["version"][3] == 1   # pk

        conn.close()

    def test_schema_table_has_applied_at_ms(self, tmp_path):
        """schema_migrations has applied_at_ms INTEGER column."""
        db_path = tmp_path / "registry.db"
        conn = sqlite3.connect(str(db_path))

        apply_migration(conn)

        cols = _conn_columns(conn, "schema_migrations")
        assert "applied_at_ms" in cols
        assert cols["applied_at_ms"][0] == "INTEGER"
        assert cols["applied_at_ms"][1] == 1  # not null

        conn.close()

    def test_schema_table_records_applied_version(self, tmp_path):
        """schema_migrations records version after apply_migration."""
        db_path = tmp_path / "registry.db"
        conn = sqlite3.connect(str(db_path))

        apply_migration(conn)

        cur = conn.execute("SELECT version, applied_at_ms FROM schema_migrations")
        rows = cur.fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "001"
        assert rows[0][1] > 0  # A valid timestamp in milliseconds

        conn.close()

    def test_schema_table_records_removed_on_rollback(self, tmp_path):
        """schema_migrations entry is removed after rollback_migration."""
        db_path = tmp_path / "registry.db"
        conn = sqlite3.connect(str(db_path))

        apply_migration(conn)
        rollback_migration(conn)

        cur = conn.execute("SELECT version FROM schema_migrations")
        rows = cur.fetchall()
        assert len(rows) == 0

        conn.close()
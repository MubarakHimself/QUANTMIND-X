# tests/api/test_epic10_p0_atdd.py
"""
Epic 10 P0 ATDD Tests — Audit, Monitoring & Notifications

TDD Red Phase: These tests define the expected behavior and should FAIL
until the implementation is completed.

P0 Criteria: Critical path + High risk (>=6) + No workaround

Risk Coverage:
- R-001: Audit log immutability at DB level
- R-002: Cold storage integrity (SHA256 verification + corruption detection)
- R-003: NL query time resolution and entity mapping
- R-004: Notification always-on events cannot be toggled
"""
import pytest
import os
import sys
import tempfile
import hashlib
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.database.models.base import Base


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def in_memory_engine():
    """Create an in-memory SQLite engine with all tables."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    return engine


@pytest.fixture
def db_session(in_memory_engine):
    """Provide a transactional DB session for tests."""
    Session = sessionmaker(bind=in_memory_engine)
    session = Session()
    yield session
    session.rollback()
    session.close()


class TestAuditLogImmutabilityDBLevel:
    """
    R-001: Audit log immutability not enforced at database level.

    P0 Test: Verify immutability is enforced at DATABASE LEVEL, not just API.
    This means DELETE and UPDATE operations should fail at the ORM/DB constraint level.
    """

    def test_delete_audit_entry_raises_error(self, db_session):
        """
        P0: DELETE on audit_log table should raise error at DB level.

        RED PHASE: This test should FAIL because DELETE currently succeeds.
        GREEN PHASE: After adding DB-level trigger, DELETE should be blocked.

        Verifies R-001 mitigation: DB-level constraints prevent DELETE.
        """
        from src.database.models import AuditLogEntry

        # Create a test audit entry
        test_entry = AuditLogEntry(
            id="test-delete-123",
            layer="trade",
            event_type="execution",
            action="Test delete",
            entity_id="EA_TEST",
            timestamp_utc=datetime.now(timezone.utc),
            created_at_utc=datetime.now(timezone.utc)
        )
        db_session.add(test_entry)
        db_session.commit()

        # ATTEMPT DELETE with commit - should raise error if immutability enforced
        db_session.delete(test_entry)

        # If we reach here without exception, immutability is NOT enforced
        # This will fail the test (RED phase of TDD)
        try:
            db_session.commit()
            # DELETE succeeded - immutability NOT enforced - TEST FAILS
            pytest.fail("DELETE on audit_log should raise error - DB-level immutability not enforced!")
        except Exception as e:
            # DELETE was blocked - immutability IS enforced - TEST PASSES
            error_msg = str(e).lower()
            # Verify error message indicates deletion was blocked
            assert any(word in error_msg for word in ["delete", "immutable", "not allowed", "permission denied"]), \
                f"Error message should indicate deletion is not allowed, got: {e}"

    def test_update_audit_entry_raises_error(self, db_session):
        """
        P0: UPDATE on audit_log table should raise error at DB level.

        RED PHASE: This test should FAIL because UPDATE currently succeeds.
        GREEN PHASE: After adding DB-level trigger, UPDATE should be blocked.

        Verifies R-001 mitigation: DB-level constraints prevent UPDATE.
        """
        from src.database.models import AuditLogEntry

        test_entry = AuditLogEntry(
            id="test-update-456",
            layer="trade",
            event_type="execution",
            action="Original action",
            entity_id="EA_TEST",
            timestamp_utc=datetime.now(timezone.utc),
            created_at_utc=datetime.now(timezone.utc)
        )
        db_session.add(test_entry)
        db_session.commit()

        # ATTEMPT UPDATE with commit - should raise error if immutability enforced
        test_entry.action = "Modified action"

        try:
            db_session.commit()
            # UPDATE succeeded - immutability NOT enforced - TEST FAILS
            pytest.fail("UPDATE on audit_log should raise error - DB-level immutability not enforced!")
        except Exception as e:
            # UPDATE was blocked - immutability IS enforced - TEST PASSES
            error_msg = str(e).lower()
            assert any(word in error_msg for word in ["update", "immutable", "not allowed", "permission denied"]), \
                f"Error message should indicate update is not allowed, got: {e}"

    def test_no_delete_method_on_model(self):
        """
        P0: AuditLogEntry model should not expose delete() method.

        Verifies R-001: No path exists to delete audit entries via ORM.
        """
        from src.database.models import AuditLogEntry

        # Model should not have a delete instance method
        entry = AuditLogEntry.__dict__.get("delete")
        assert entry is None, "AuditLogEntry should not have a delete method"


class TestColdStorageIntegrity:
    """
    R-002: Cold storage integrity - SHA256 verification untested.

    P0 Test: Verify checksum generation matches input, verification detects
    tampered files, and corruption triggers alert.
    """

    def test_sync_detects_tampered_file(self):
        """
        P0: Tampered file in cold storage should be detected and flagged.

        RED PHASE: This test should FAIL because sync doesn't detect tampering.
        GREEN PHASE: After implementing integrity verification, tampered files
        should be removed and flagged in failed_count.

        Verifies R-002 mitigation: Integrity verification fails on corrupted files.
        """
        from src.monitoring.cold_storage_sync import ColdStorageSync

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create source and cold storage directories
            source_dir = os.path.join(tmpdir, "source")
            cold_dir = os.path.join(tmpdir, "cold_storage")
            os.makedirs(source_dir, exist_ok=True)
            os.makedirs(cold_dir, exist_ok=True)

            # Create source log file
            source_path = os.path.join(source_dir, "test.log")
            with open(source_path, "w") as f:
                f.write("original log content")

            # Create ColdStorageSync with SHORT retention (0 days = sync immediately)
            sync = ColdStorageSync(
                hot_retention_days=0,
                cold_storage_path=cold_dir,
                log_source_path=source_dir
            )

            # Run first sync - should succeed
            result1 = sync.sync_logs([{"path": source_path}])
            assert result1["synced_count"] == 1, "First sync should succeed"

            # TAMPER with the file in cold storage (simulate corruption)
            dest_path = os.path.join(cold_dir, "test.log")
            with open(dest_path, "w") as f:
                f.write("tampered log content - someone modified this!")

            # Now re-sync - should detect corruption
            # The sync should calculate checksum of source and verify against stored checksum
            result2 = sync.sync_logs([{"path": source_path}])

            # RED PHASE: This assertion will FAIL because sync doesn't detect tampering
            # GREEN PHASE: After implementation, tampered file should be in failed_count
            assert result2["failed_count"] > 0, \
                f"Tampered file should be detected and moved to failed_count. Got: {result2}"

            # Verify the corrupted file was removed
            assert not os.path.exists(dest_path), \
                "Corrupted file should be removed from cold storage"

    def test_checksum_file_created_after_sync(self):
        """
        P0: SHA256 checksum file should be created alongside synced file.

        Verifies R-002: Checksums are persisted for future verification.
        """
        from src.monitoring.cold_storage_sync import ColdStorageSync

        with tempfile.TemporaryDirectory() as tmpdir:
            source_dir = os.path.join(tmpdir, "source")
            cold_dir = os.path.join(tmpdir, "cold_storage")
            os.makedirs(source_dir, exist_ok=True)

            source_path = os.path.join(source_dir, "test.log")
            with open(source_path, "w") as f:
                f.write("log content for checksum test")

            sync = ColdStorageSync(
                hot_retention_days=0,
                cold_storage_path=cold_dir,
                log_source_path=source_dir
            )

            result = sync.sync_logs([{"path": source_path}])

            checksum_path = os.path.join(cold_dir, "test.log.sha256")
            assert os.path.exists(checksum_path), \
                "SHA256 checksum file should be created after successful sync"

            with open(checksum_path, "r") as f:
                stored_checksum = f.read().strip()

            expected = hashlib.sha256(b"log content for checksum test").hexdigest()
            assert stored_checksum == expected, \
                f"Stored checksum should match file content, got {stored_checksum} expected {expected}"

    def test_verify_integrity_on_restored_file(self):
        """
        P0: verify_file_integrity should return True for matching checksum.

        Verifies R-002: Files can be verified against stored checksums.
        """
        from src.monitoring.cold_storage_sync import verify_file_integrity

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("integrity test content")
            temp_path = f.name

        try:
            expected = hashlib.sha256(b"integrity test content").hexdigest()

            result = verify_file_integrity(temp_path, expected, "sha256")
            assert result is True, \
                "verify_file_integrity should return True for matching checksum"
        finally:
            os.unlink(temp_path)


class TestNLQueryTimeResolution:
    """
    R-003: NL query degrades with large time ranges.

    P0 Test: Verify time references are parsed correctly for common query patterns.
    """

    def test_parse_last_week_time_reference(self):
        """
        P0: 'last week' should be parsed to 7-day range.

        Verifies R-003 mitigation: Time parsing handles common NL references.
        """
        from src.api.audit_endpoints import NLQueryParser

        parser = NLQueryParser()
        result = parser.parse("Show me events from last week")

        assert result["time_range"]["raw"] == "last week", \
            "Should recognize 'last week' reference"

        assert result["time_range"]["start"] is not None, \
            "Should parse start date for 'last week'"
        assert result["time_range"]["end"] is not None, \
            "Should parse end date for 'last week'"

        assert result["time_range"]["end"] > result["time_range"]["start"], \
            "End date should be after start date for 'last week'"

        delta = result["time_range"]["end"] - result["time_range"]["start"]
        assert timedelta(days=6) <= delta <= timedelta(days=8), \
            f"'last week' should be ~7 days, got {delta}"

    def test_parse_yesterday_specific_time(self):
        """
        P0: 'yesterday at HH:MM' should be parsed correctly.

        Verifies R-003: Time resolution works for specific yesterday times.
        """
        from src.api.audit_endpoints import NLQueryParser

        parser = NLQueryParser()
        result = parser.parse("Why was EA_X paused at 14:30 yesterday?")

        assert "yesterday" in result["time_range"]["raw"], \
            "Should recognize 'yesterday' in query"

        assert result["time_range"]["start"] is not None, \
            "Should parse time for 'yesterday at HH:MM'"

        start = result["time_range"]["start"]
        assert start.hour == 14, f"Expected hour 14, got {start.hour}"
        assert start.minute == 30, f"Expected minute 30, got {start.minute}"

    def test_parse_today_time_reference(self):
        """
        P0: 'today' should be parsed to current day range.
        """
        from src.api.audit_endpoints import NLQueryParser

        parser = NLQueryParser()
        result = parser.parse("What happened today with EA_001?")

        assert result["time_range"]["raw"] == "today", \
            "Should recognize 'today' reference"

        start = result["time_range"]["start"]
        now = datetime.now(timezone.utc)

        assert start.year == now.year, "Start should be today"
        assert start.month == now.month, "Start should be today"
        assert start.day == now.day, "Start should be today"


class TestNLQueryEntityMapping:
    """
    R-003: Entity references mapped correctly.

    P0 Test: Verify EA, strategy, and symbol entities are extracted.
    """

    def test_extract_single_ea_entity(self):
        """
        P0: Single EA_X entity should be extracted.
        """
        from src.api.audit_endpoints import NLQueryParser

        parser = NLQueryParser()
        result = parser.parse("Why was EA_X paused?")

        eas = result["entities"]["eas"]
        assert "EA_X" in eas, f"Should extract EA_X, got {eas}"

    def test_extract_multiple_ea_entities(self):
        """
        P0: Multiple EA entities should be extracted.
        """
        from src.api.audit_endpoints import NLQueryParser

        parser = NLQueryParser()
        result = parser.parse("Compare EA_001 and EA_002 performance")

        eas = result["entities"]["eas"]
        assert "EA_001" in eas, f"Should extract EA_001, got {eas}"
        assert "EA_002" in eas, f"Should extract EA_002, got {eas}"

    def test_extract_symbol_entity(self):
        """
        P0: Currency pair symbols should be extracted.
        """
        from src.api.audit_endpoints import NLQueryParser

        parser = NLQueryParser()
        result = parser.parse("Show GBPUSD trades for yesterday")

        symbols = result["entities"]["symbols"]
        assert "GBPUSD" in symbols, f"Should extract GBPUSD, got {symbols}"


class TestNotificationAlwaysOnEvents:
    """
    R-004: Notification toggle race with concurrent delivery.

    P0 Test: Verify always-on events cannot be disabled.
    """

    def test_notification_toggle_loss_cap_always_on(self):
        """
        P0: loss_cap_triggered_system should be always-on.

        Verifies R-004: kill_switch, loss_cap, system_critical cannot be toggled.
        """
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        from unittest.mock import patch, MagicMock

        from src.api.notification_config_endpoints import router

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        with patch('src.api.notification_config_endpoints.get_session') as mock_get_session:
            mock_session = MagicMock()
            mock_get_session.return_value = mock_session

            mock_config = MagicMock()
            mock_config.event_type = "loss_cap_triggered_system"
            mock_config.is_always_on = True
            mock_config.category = "system"
            mock_session.query.return_value.filter.return_value.first.return_value = mock_config

            response = client.put(
                "/api/notifications",
                json={"event_type": "loss_cap_triggered_system", "is_enabled": False}
            )

            assert response.status_code == 400, \
                f"Always-on event should return 400, got {response.status_code}"

            detail = response.json().get("detail", "").lower()
            assert "always-on" in detail or "cannot be disabled" in detail, \
                f"Error should explain why toggle failed, got: {detail}"

    def test_notification_toggle_system_critical_always_on(self):
        """
        P0: system_critical should be always-on.
        """
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        from unittest.mock import patch, MagicMock

        from src.api.notification_config_endpoints import router

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        with patch('src.api.notification_config_endpoints.get_session') as mock_get_session:
            mock_session = MagicMock()
            mock_get_session.return_value = mock_session

            mock_config = MagicMock()
            mock_config.event_type = "system_critical"
            mock_config.is_always_on = True
            mock_config.category = "system"
            mock_session.query.return_value.filter.return_value.first.return_value = mock_config

            response = client.put(
                "/api/notifications",
                json={"event_type": "system_critical", "is_enabled": False}
            )

            assert response.status_code == 400, \
                f"Always-on event should return 400, got {response.status_code}"


class TestReasoningAPIDepartmentQuery:
    """
    R-002/R-003: Reasoning API department filtering.

    P0 Test: Verify OPINION nodes can be queried by department/agent_role.
    """

    @patch('src.api.reasoning_log_endpoints._get_facade')
    def test_get_department_reasoning_research(self, mock_get_facade):
        """
        P0: GET /api/audit/reasoning/department/research should return OPINION nodes.

        Verifies AC2: hypothesis chain, evidence sources, confidence scores.
        """
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        from datetime import datetime

        from src.api.reasoning_log_endpoints import router

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        mock_facade = MagicMock()
        mock_store = MagicMock()

        mock_node = MagicMock()
        mock_node.id = "opinion-research-1"
        mock_node.session_id = "session-123"
        mock_node.content = "Analyzing GBPUSD for short opportunity"
        mock_node.action = "Recommend GBPUSD short"
        mock_node.reasoning = "Bearish signals from technical indicators"
        mock_node.confidence = 0.82
        mock_node.agent_role = "research"
        mock_node.created_at = datetime.utcnow()
        mock_node.metadata = {
            "evidence_sources": ["USD weakness", "technical charts"],
            "sub_agents": ["macro-analyst", "technical-analyst"]
        }
        mock_node.alternatives_considered = "long, hold"
        mock_node.constraints_applied = "risk limits, position size"

        mock_store.query_nodes.return_value = [mock_node]
        mock_facade.store = mock_store
        mock_get_facade.return_value = mock_facade

        response = client.get("/api/audit/reasoning/department/research")

        assert response.status_code == 200, \
            f"Department reasoning endpoint should return 200, got {response.status_code}"

        data = response.json()

        assert "department" in data, "Response should include department"
        assert data["department"] == "research", \
            f"Department should be 'research', got {data['department']}"

        assert "reasoning_chain" in data, "Response should include reasoning_chain"
        assert isinstance(data["reasoning_chain"], list), \
            "reasoning_chain should be a list"

        if data["reasoning_chain"]:
            chain_item = data["reasoning_chain"][0]
            assert "hypothesis" in chain_item or "action" in chain_item or "reasoning" in chain_item, \
                "Chain item should have hypothesis/action/reasoning"
            assert "confidence_score" in chain_item or "confidence" in chain_item, \
                "Chain item should include confidence"

    @patch('src.api.reasoning_log_endpoints._get_facade')
    def test_get_reasoning_by_decision_id(self, mock_get_facade):
        """
        P0: GET /api/audit/reasoning/{decision_id} should return OPINION nodes.

        Verifies AC1: decision_id lookup, context, model_used.
        """
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        from datetime import datetime

        from src.api.reasoning_log_endpoints import router

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        mock_facade = MagicMock()
        mock_store = MagicMock()

        mock_node = MagicMock()
        mock_node.id = "opinion-decision-1"
        mock_node.session_id = "decision-abc-123"
        mock_node.content = "Analysis content for decision"
        mock_node.action = "Take position"
        mock_node.reasoning = "Based on analysis"
        mock_node.confidence = 0.75
        mock_node.agent_role = "trading"
        mock_node.created_at = datetime.utcnow()
        mock_node.metadata = {
            "decision_id": "decision-abc-123",
            "model_used": "claude-3-opus",
            "context": {"symbol": "GBPUSD", "direction": "short"}
        }
        mock_node.alternatives_considered = "do nothing, long instead"
        mock_node.constraints_applied = "max position size"

        mock_store.query_nodes.return_value = [mock_node]
        mock_facade.store = mock_store
        mock_get_facade.return_value = mock_facade

        response = client.get("/api/audit/reasoning/decision-abc-123")

        assert response.status_code == 200, \
            f"Reasoning endpoint should return 200, got {response.status_code}"

        data = response.json()

        assert "decision_id" in data, "Response should include decision_id"
        assert "opinion_nodes" in data, "Response should include opinion_nodes"
        assert isinstance(data["opinion_nodes"], list), \
            "opinion_nodes should be a list"

        if data["opinion_nodes"]:
            node = data["opinion_nodes"][0]
            assert "node_id" in node, "Opinion node should have node_id"
            assert "confidence" in node or "reasoning" in node or "action" in node, \
                "Opinion node should have confidence/action/reasoning"

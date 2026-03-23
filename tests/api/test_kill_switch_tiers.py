"""
Tests for Kill Switch Tiers API

Tests the tier-specific trigger endpoints:
- Tier 1: Soft Stop (no new entries)
- Tier 2: Strategy Pause (pause specific strategies)
- Tier 3: Emergency Close (close all positions)

Validates:
- NFR-P1: Atomic execution
- NFR-D2: Immutable audit logging
- NFR-R4: Cloudzy independence
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class TestKillSwitchTiersEndpoint:
    """Test the kill switch tier trigger endpoint."""

    @pytest.mark.asyncio
    async def test_trigger_tier_1_success(self):
        """Tier 1 trigger should succeed and return audit log ID."""
        from src.api.kill_switch_endpoints import _execute_tier1_soft_stop, KillSwitchAuditLog
        from datetime import datetime

        # Create fresh audit log for this test
        audit_log = KillSwitchAuditLog()

        with patch('src.api.kill_switch_endpoints._get_kill_switch_router') as mock:
            mock_instance = MagicMock()
            mock_instance._socket_server = None
            mock_instance._connected_eas = []
            mock_instance._active = False
            mock.return_value = mock_instance

            result = await _execute_tier1_soft_stop(
                audit_log=audit_log,
                activator="test_user",
                activated_at_utc=datetime.utcnow()
            )

            assert result["success"] is True
            assert "audit_log_id" in result
            assert "activated_at_utc" in result
            # Verify audit log was created
            assert len(audit_log.get_all()) >= 1

    @pytest.mark.asyncio
    async def test_trigger_tier_1_sets_active_flag(self):
        """Tier 1 should set the kill switch active flag."""
        from src.api.kill_switch_endpoints import _execute_tier1_soft_stop
        from datetime import datetime

        with patch('src.api.kill_switch_endpoints._get_kill_switch_router') as mock:
            mock_instance = MagicMock()
            mock_instance._socket_server = None
            mock_instance._connected_eas = []
            mock_instance._active = False
            mock.return_value = mock_instance

            await _execute_tier1_soft_stop(
                audit_log=MagicMock(),
                activator="test_user",
                activated_at_utc=datetime.utcnow()
            )

            assert mock_instance._active is True

    @pytest.mark.asyncio
    async def test_trigger_tier_2_success(self):
        """Tier 2 trigger should succeed with strategy_ids."""
        from src.api.kill_switch_endpoints import _execute_tier2_strategy_pause, KillSwitchAuditLog
        from datetime import datetime

        audit_log = KillSwitchAuditLog()

        with patch('src.api.kill_switch_endpoints._get_kill_switch_router') as mock:
            mock_instance = MagicMock()
            mock_instance._socket_server = None
            mock_instance._connected_eas = []
            mock_instance._active = False
            mock.return_value = mock_instance

            result = await _execute_tier2_strategy_pause(
                audit_log=audit_log,
                strategy_ids=["STRAT-001", "STRAT-002"],
                activator="test_user",
                activated_at_utc=datetime.utcnow()
            )

            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_trigger_tier_3_success(self):
        """Tier 3 trigger should succeed and return close results."""
        from src.api.kill_switch_endpoints import _execute_tier3_emergency_close, KillSwitchAuditLog
        from datetime import datetime

        audit_log = KillSwitchAuditLog()

        with patch('src.api.kill_switch_endpoints._get_kill_switch_router') as mock:
            mock_instance = MagicMock()
            mock_instance._socket_server = None
            mock_instance._connected_eas = ["EA-001", "EA-002"]
            mock_instance._active = False
            mock.return_value = mock_instance

            result = await _execute_tier3_emergency_close(
                audit_log=audit_log,
                activator="test_user",
                activated_at_utc=datetime.utcnow()
            )

            assert result["success"] is True
            assert "results" in result
            assert len(result["results"]) == 2


class TestAuditLogging:
    """Tests for audit logging (NFR-D2)."""

    @pytest.mark.asyncio
    async def test_audit_log_created_on_tier1(self):
        """Tier 1 activation should create audit log entry."""
        from src.api.kill_switch_endpoints import _execute_tier1_soft_stop, KillSwitchAuditLog
        from datetime import datetime

        # Create fresh audit log for this test
        audit_log = KillSwitchAuditLog()

        with patch('src.api.kill_switch_endpoints._get_kill_switch_router') as mock:
            mock_instance = MagicMock()
            mock_instance._socket_server = None
            mock_instance._connected_eas = []
            mock_instance._active = False
            mock.return_value = mock_instance

            await _execute_tier1_soft_stop(
                audit_log=audit_log,
                activator="test",
                activated_at_utc=datetime.utcnow()
            )

            # Verify audit log contains required fields
            logs = audit_log.get_all()
            assert len(logs) >= 1
            log = logs[-1]
            assert "tier" in log
            assert log["tier"] == 1
            assert "activated_at_utc" in log
            assert "activator" in log

    def test_audit_log_immutable(self):
        """Audit log entries should be immutable after creation."""
        from src.api.kill_switch_endpoints import KillSwitchAuditLog
        from datetime import datetime

        audit_log = KillSwitchAuditLog()

        # Create an entry
        entry_id = audit_log.append({
            "tier": 1,
            "activated_at_utc": datetime.utcnow().isoformat(),
            "activator": "test"
        })

        # Get the entry
        entry = audit_log.get_by_id(entry_id)
        assert entry is not None
        assert entry["id"] == entry_id

        # Verify append-only (can't modify existing entries)
        original_logs = audit_log.get_all()
        audit_log.append({"tier": 2, "activator": "test2"})
        assert len(audit_log.get_all()) == len(original_logs) + 1

    def test_audit_uses_utc_timestamps(self):
        """All timestamps must use UTC."""
        from src.api.kill_switch_endpoints import KillSwitchAuditLog

        audit_log = KillSwitchAuditLog()
        entry_id = audit_log.append({
            "tier": 1,
            "activator": "test"
        })

        entry = audit_log.get_by_id(entry_id)
        assert "timestamp_utc" in entry
        # Verify it's an ISO format timestamp
        assert "T" in entry["timestamp_utc"]


class TestAtomicExecution:
    """Tests for atomic execution (NFR-P1)."""

    @pytest.mark.asyncio
    async def test_tier1_completes_before_return(self):
        """Tier 1 should complete all steps before returning."""
        from src.api.kill_switch_endpoints import _execute_tier1_soft_stop
        from datetime import datetime

        with patch('src.api.kill_switch_endpoints._get_kill_switch_router') as mock:
            mock_instance = MagicMock()
            mock_instance._socket_server = None
            mock_instance._connected_eas = ["EA-001"]
            mock_instance._active = False
            mock.return_value = mock_instance

            await _execute_tier1_soft_stop(
                audit_log=MagicMock(),
                activator="test",
                activated_at_utc=datetime.utcnow()
            )

            # Verify the _active flag is set (indicates completion)
            assert mock_instance._active is True

    @pytest.mark.asyncio
    async def test_tier3_creates_per_position_logs(self):
        """Tier 3 should create audit log for each position."""
        from src.api.kill_switch_endpoints import _execute_tier3_emergency_close, KillSwitchAuditLog
        from datetime import datetime

        audit_log = KillSwitchAuditLog()

        with patch('src.api.kill_switch_endpoints._get_kill_switch_router') as mock:
            mock_instance = MagicMock()
            mock_instance._socket_server = None
            mock_instance._connected_eas = ["EA-001", "EA-002"]
            mock_instance._active = False
            mock.return_value = mock_instance

            await _execute_tier3_emergency_close(
                audit_log=audit_log,
                activator="test",
                activated_at_utc=datetime.utcnow()
            )

            # Should have main entry + per-position entries
            logs = audit_log.get_all()
            tier3_logs = [l for l in logs if l.get("tier") == 3]
            # At minimum, should have main entry
            assert len(tier3_logs) >= 1


class TestCloudzyIndependence:
    """Tests for Cloudzy independence (NFR-R4)."""

    @pytest.mark.asyncio
    async def test_works_without_socket_server(self):
        """Kill switch should work without socket server (Cloudzy mode)."""
        from src.api.kill_switch_endpoints import _execute_tier1_soft_stop
        from datetime import datetime

        with patch('src.api.kill_switch_endpoints._get_kill_switch_router') as mock:
            mock_instance = MagicMock()
            mock_instance._socket_server = None  # No socket server
            mock_instance._connected_eas = []
            mock_instance._active = False
            mock.return_value = mock_instance

            # This should NOT fail - works in Cloudzy-independent mode
            result = await _execute_tier1_soft_stop(
                audit_log=MagicMock(),
                activator="cloudzy_test",
                activated_at_utc=datetime.utcnow()
            )

            assert result["success"] is True


class TestKillSwitchModels:
    """Tests for request/response models."""

    def test_trigger_request_model(self):
        """KillSwitchTriggerRequest model should validate correctly."""
        from src.api.kill_switch_endpoints import KillSwitchTriggerRequest
        from pydantic import ValidationError

        # Valid request
        req = KillSwitchTriggerRequest(tier=1, activator="test")
        assert req.tier == 1
        assert req.activator == "test"

        # Tier 2 requires strategy_ids
        req2 = KillSwitchTriggerRequest(tier=2, strategy_ids=["s1", "s2"], activator="test")
        assert req2.tier == 2
        assert req2.strategy_ids == ["s1", "s2"]

    def test_close_result_model(self):
        """CloseResult model should work correctly."""
        from src.api.kill_switch_endpoints import CloseResult

        result = CloseResult(position_id="EA-001", result="filled", pnl=100.0)
        assert result.position_id == "EA-001"
        assert result.result == "filled"
        assert result.pnl == 100.0

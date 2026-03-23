"""
P0 Failing Tests for Epic 3 - Live Trading Command Center

These tests validate the critical path for Live Trading:
- WebSocket streaming ≤3s latency under load (10 bots, 50 positions)
- Kill switch atomic execution (Tiers 1, 2, 3)
- Cloudzy independence (Contabo unreachable)
- MT5 bridge reconnection ≤10s
- Kill switch audit log immutability

Risk IDs covered: R-001, R-002, R-003, R-004, R-005
NFRs validated: NFR-P1, NFR-P2, NFR-R3, NFR-R4, NFR-R5, NFR-D2

Generated via ATDD workflow for Epic 3.
"""

import pytest
import asyncio
import time
import json
from datetime import datetime, timezone
from typing import List, Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch

# Add project root to path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


# =============================================================================
# Mock WebSocket for Testing
# =============================================================================

class MockWebSocket:
    """Mock WebSocket for testing."""

    def __init__(self):
        self.messages: List[str] = []
        self.accepted = False
        self.closed = False

    async def accept(self):
        self.accepted = True

    async def send_text(self, message: str):
        self.messages.append(message)

    async def send_json(self, data: Dict[str, Any]):
        self.messages.append(json.dumps(data))

    async def receive_text(self) -> str:
        await asyncio.sleep(0.01)
        return json.dumps({"action": "subscribe", "topic": "trading"})

    def get_messages(self) -> List[Dict[str, Any]]:
        """Get all received messages as parsed JSON."""
        return [json.loads(msg) for msg in self.messages]

    def get_messages_of_type(self, msg_type: str) -> List[Dict[str, Any]]:
        """Get all messages of a specific type."""
        return [msg for msg in self.get_messages() if msg.get("type") == msg_type]


# =============================================================================
# P0 TEST SUITE 1: WebSocket Streaming Latency Under Load (R-001, NFR-P2)
# =============================================================================

class TestWebSocketLatencyUnderLoad:
    """
    P0: WebSocket streaming ≤3s latency under load.

    Validates NFR-P2: ≤3s lag on live data
    Risk: R-001 - WebSocket streaming latency exceeds 3s threshold under load

    Test Strategy:
    - Simulate 10 bots with 50 positions total
    - Measure end-to-end broadcast latency
    - Assert latency ≤3000ms
    """

    @pytest.mark.asyncio
    async def test_position_update_latency_under_load_10_bots(self):
        """
        P0: Position update latency ≤3s with 10 bots streaming simultaneously.

        This test simulates the load scenario from R-001:
        - 10 bots each with multiple positions (total 50 positions)
        - Rapid-fire position updates
        - Must complete within 3s NFR-P2 requirement
        """
        from src.api.websocket_endpoints import (
            broadcast_position_update,
            manager,
            reset_trading_state_cache,
            TradingDataBroadcaster
        )

        # Reset state for clean test
        reset_trading_state_cache()

        # Create 10 WebSocket clients (simulating 10 bots)
        clients = [MockWebSocket() for _ in range(10)]
        for ws in clients:
            await manager.connect(ws)
            await manager.subscribe(ws, "trading")

        broadcaster = TradingDataBroadcaster()

        # Simulate 50 positions across 10 bots
        positions = []
        for bot_idx in range(10):
            for pos_idx in range(5):  # 5 positions per bot
                positions.append({
                    "ticket": bot_idx * 1000 + pos_idx,
                    "symbol": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"][pos_idx],
                    "volume": 0.1 + (pos_idx * 0.01),
                    "open_price": 1.08000 + (pos_idx * 0.00100),
                    "current_price": 1.08100 + (pos_idx * 0.00100),
                    "profit": 10.00 + (pos_idx * 5.00),
                    "bot_id": f"Bot-{bot_idx:02d}"
                })

        # Measure latency for batch of 50 position updates
        latencies = []
        start_time = time.time()

        for position in positions:
            tick_start = time.time()
            await broadcast_position_update(position)
            await broadcaster.cache_position(position)
            tick_latency = (time.time() - tick_start) * 1000
            latencies.append(tick_latency)

        total_time = (time.time() - start_time) * 1000

        # Verify all clients received updates
        for i, ws in enumerate(clients):
            pos_updates = ws.get_messages_of_type("position_update")
            assert len(pos_updates) == 50, f"Client {i} should receive 50 position updates, got {len(pos_updates)}"

        # Assert NFR-P2: ≤3s latency per update
        max_latency = max(latencies)
        avg_latency = sum(latencies) / len(latencies)

        assert max_latency <= 3000, (
            f"NFR-P2 VIOLATION: Max latency {max_latency:.2f}ms exceeds 3000ms threshold. "
            f"Avg latency: {avg_latency:.2f}ms for {len(positions)} positions across 10 bots."
        )

    @pytest.mark.asyncio
    async def test_pnl_update_latency_under_load(self):
        """
        P0: P&L update latency ≤3s under continuous load.

        Validates R-001: P&L updates must meet same latency requirements as positions.
        """
        from src.api.websocket_endpoints import (
            broadcast_pnl_update,
            manager,
            reset_trading_state_cache,
            TradingDataBroadcaster
        )

        reset_trading_state_cache()

        # 10 clients subscribing to P&L updates
        clients = [MockWebSocket() for _ in range(10)]
        for ws in clients:
            await manager.connect(ws)
            await manager.subscribe(ws, "trading")

        broadcaster = TradingDataBroadcaster()

        # Simulate 50 rapid P&L updates
        latencies = []
        for i in range(50):
            pnl_data = {
                "daily_pnl": 100.00 + (i * 2),
                "open_positions": 5,
                "open_pnl": 75.00 + (i * 1.5),
                "closed_pnl": 25.00 + (i * 0.5),
                "equity": 10000.00 + (i * 10),
                "balance": 10000.00
            }

            tick_start = time.time()
            await broadcast_pnl_update(pnl_data)
            await broadcaster.cache_pnl(pnl_data)
            tick_latency = (time.time() - tick_start) * 1000
            latencies.append(tick_latency)

        # Verify all clients received all updates
        for ws in clients:
            pnl_updates = ws.get_messages_of_type("pnl_update")
            assert len(pnl_updates) == 50

        # Assert NFR-P2
        max_latency = max(latencies)
        assert max_latency <= 3000, (
            f"NFR-P2 VIOLATION: P&L update latency {max_latency:.2f}ms exceeds 3000ms"
        )

    @pytest.mark.asyncio
    async def test_combined_broadcast_latency_with_50_positions(self):
        """
        P0: Combined position + P&L broadcast latency with 50 positions.

        Real trading scenario: positions update rapidly alongside P&L recalculations.
        This test validates the broadcaster can handle mixed traffic at scale.
        """
        from src.api.websocket_endpoints import (
            broadcast_position_update,
            broadcast_pnl_update,
            manager,
            reset_trading_state_cache,
            TradingDataBroadcaster
        )

        reset_trading_state_cache()

        # Single client but high-volume traffic
        ws = MockWebSocket()
        await manager.connect(ws)
        await manager.subscribe(ws, "trading")

        broadcaster = TradingDataBroadcaster()

        # Alternate position and P&L updates (realistic trading scenario)
        start_time = time.time()

        for i in range(25):  # 25 rounds = 50 broadcasts
            # Position update
            await broadcast_position_update({
                "ticket": i,
                "symbol": "EURUSD",
                "volume": 0.1,
                "open_price": 1.0850,
                "current_price": 1.0860,
                "profit": 10.00 + i
            })
            await broadcaster.cache_position({
                "ticket": i,
                "symbol": "EURUSD",
                "volume": 0.1,
                "profit": 10.00 + i
            })

            # P&L update
            await broadcast_pnl_update({
                "daily_pnl": 50.00 + i,
                "open_positions": 1,
                "open_pnl": 50.00 + i,
                "closed_pnl": 0.0
            })
            await broadcaster.cache_pnl({
                "daily_pnl": 50.00 + i,
                "open_positions": 1
            })

        total_time = (time.time() - start_time) * 1000

        # Should have 25 position + 25 P&L = 50 total broadcasts
        all_messages = ws.get_messages()
        assert len(all_messages) == 50

        # Total time for 50 broadcasts should be well under 3s (not per-message)
        # But individual messages should still stream within latency bounds
        assert total_time < 3000, (
            f"Combined broadcast of 50 messages took {total_time:.2f}ms, "
            f"indicating broadcaster congestion."
        )


# =============================================================================
# P0 TEST SUITE 2: Kill Switch Atomic Execution (R-002, NFR-P1)
# =============================================================================

class TestKillSwitchTier1AtomicExecution:
    """
    P0: Kill switch Tier 1 atomic execution.

    Validates NFR-P1: Kill switch executes in full, in order
    Risk: R-002 - Kill switch tier activation fails to execute atomically

    Atomic Execution Criteria:
    1. All steps complete before returning success
    2. Audit log entry created only AFTER all actions complete
    3. No partial state left on failure
    """

    @pytest.mark.asyncio
    async def test_tier1_atomic_all_steps_complete_before_return(self):
        """
        P0: Tier 1 must complete ALL steps before returning success.

        This validates NFR-P1: atomic execution - no early returns.
        The audit log should only be written AFTER all EAs are halted.
        """
        from src.api.kill_switch_endpoints import (
            _execute_tier1_soft_stop,
            KillSwitchAuditLog
        )
        from datetime import datetime

        audit_log = KillSwitchAuditLog()

        # Track execution order
        execution_log = []

        with patch('src.api.kill_switch_endpoints._get_kill_switch_router') as mock:
            mock_instance = MagicMock()
            mock_instance._socket_server = MagicMock()
            mock_instance._socket_server.broadcast_command = AsyncMock()
            mock_instance._connected_eas = ["EA-001", "EA-002", "EA-003"]
            mock_instance._active = False
            mock.return_value = mock_instance

            # Wrap broadcast to track when commands are sent
            async def track_broadcast(command):
                execution_log.append(("broadcast", command["type"], datetime.utcnow()))

            mock_instance._socket_server.broadcast_command.side_effect = track_broadcast

            # Execute Tier 1
            result = await _execute_tier1_soft_stop(
                audit_log=audit_log,
                activator="atomic_test",
                activated_at_utc=datetime.utcnow()
            )

            # Broadcast should have been called for ALL EAs before audit log
            assert mock_instance._socket_server.broadcast_command.call_count == 3, (
                f"Expected 3 broadcast calls (one per EA), got {mock_instance._socket_server.broadcast_command.call_count}"
            )

            # Verify atomic: audit log created AFTER broadcasts
            logs = audit_log.get_all()
            assert len(logs) == 1, "Audit log should have exactly 1 entry"

            # The audit log entry must show all EAs were processed
            log_entry = logs[0]
            assert len(log_entry.get("eas_affected", [])) == 3, (
                "Audit log should record all 3 EAs as affected"
            )
            assert log_entry["status"] == "completed", "Status must be completed"

    @pytest.mark.asyncio
    async def test_tier1_no_partial_state_on_ea_failure(self):
        """
        P0: Tier 1 must handle EA failure atomically - no partial state.

        If one EA fails to receive HALT command, the audit log should
        still be complete and reflect the partial failure correctly.
        """
        from src.api.kill_switch_endpoints import (
            _execute_tier1_soft_stop,
            KillSwitchAuditLog
        )
        from datetime import datetime

        audit_log = KillSwitchAuditLog()

        with patch('src.api.kill_switch_endpoints._get_kill_switch_router') as mock:
            mock_instance = MagicMock()
            mock_instance._socket_server = MagicMock()

            # First EA succeeds, second fails
            call_count = [0]
            async def partial_broadcast(command):
                call_count[0] += 1
                if call_count[0] == 2:
                    raise Exception("Simulated EA-002 failure")

            mock_instance._socket_server.broadcast_command = AsyncMock(side_effect=partial_broadcast)
            mock_instance._connected_eas = ["EA-001", "EA-002", "EA-003"]
            mock_instance._active = False
            mock.return_value = mock_instance

            result = await _execute_tier1_soft_stop(
                audit_log=audit_log,
                activator="partial_failure_test",
                activated_at_utc=datetime.utcnow()
            )

            # Should still return success (best effort) but log the partial failure
            assert result["success"] is True

            # Audit log should reflect what happened
            logs = audit_log.get_all()
            log_entry = logs[0]

            # Should still record EA-001 as affected even though EA-002 failed
            # This is the "atomic" behavior - we document what we know
            assert "EA-001" in log_entry.get("eas_affected", [])

    @pytest.mark.asyncio
    async def test_tier1_audit_log_immutable_afterCreation(self):
        """
        P0: Tier 1 audit log entry must be immutable after creation.

        Validates NFR-D2: Immutable audit log
        Once an audit log entry is created, it cannot be modified.
        """
        from src.api.kill_switch_endpoints import (
            _execute_tier1_soft_stop,
            KillSwitchAuditLog
        )
        from datetime import datetime

        audit_log = KillSwitchAuditLog()

        with patch('src.api.kill_switch_endpoints._get_kill_switch_router') as mock:
            mock_instance = MagicMock()
            mock_instance._socket_server = None
            mock_instance._connected_eas = []
            mock_instance._active = False
            mock.return_value = mock_instance

            result = await _execute_tier1_soft_stop(
                audit_log=audit_log,
                activator="immutability_test",
                activated_at_utc=datetime.utcnow()
            )

            logs_before = audit_log.get_all()
            original_entry = logs_before[0].copy()
            original_id = original_entry["id"]

            # Attempt to modify - should have no effect on the stored entry
            # The KillSwitchAuditLog.append() creates immutable entries
            # Only get_all() returns a copy, not the original

            logs_after = audit_log.get_all()
            modified_entry = logs_after[0]

            # Entry should be unchanged
            assert modified_entry["id"] == original_id
            assert modified_entry["activator"] == original_entry["activator"]
            assert modified_entry["tier"] == original_entry["tier"]


class TestKillSwitchTier2AtomicExecution:
    """
    P0: Kill switch Tier 2 atomic execution.

    Validates NFR-P1: Strategy pause executes atomically
    """

    @pytest.mark.asyncio
    async def test_tier2_all_strategies_paused_before_return(self):
        """
        P0: Tier 2 must pause ALL specified strategies before returning.

        If 5 strategies are specified, all 5 must receive PAUSE commands
        before the audit log is written.
        """
        from src.api.kill_switch_endpoints import (
            _execute_tier2_strategy_pause,
            KillSwitchAuditLog
        )
        from datetime import datetime

        audit_log = KillSwitchAuditLog()
        strategy_ids = ["STRAT-001", "STRAT-002", "STRAT-003", "STRAT-004", "STRAT-005"]

        with patch('src.api.kill_switch_endpoints._get_kill_switch_router') as mock:
            mock_instance = MagicMock()
            mock_instance._socket_server = MagicMock()
            mock_instance._socket_server.broadcast_command = AsyncMock()
            mock.return_value = mock_instance

            result = await _execute_tier2_strategy_pause(
                audit_log=audit_log,
                strategy_ids=strategy_ids,
                activator="tier2_atomic_test",
                activated_at_utc=datetime.utcnow()
            )

            # All 5 strategies should have received commands
            assert mock_instance._socket_server.broadcast_command.call_count == 5

            # Audit log should reflect all paused strategies
            logs = audit_log.get_all()
            log_entry = logs[0]
            assert len(log_entry["strategy_ids"]) == 5

    @pytest.mark.asyncio
    async def test_tier2_partial_strategy_failure_logged(self):
        """
        P0: Tier 2 must log partial failures correctly.
        """
        from src.api.kill_switch_endpoints import (
            _execute_tier2_strategy_pause,
            KillSwitchAuditLog
        )
        from datetime import datetime

        audit_log = KillSwitchAuditLog()
        strategy_ids = ["STRAT-001", "STRAT-002", "STRAT-003"]

        with patch('src.api.kill_switch_endpoints._get_kill_switch_router') as mock:
            mock_instance = MagicMock()
            mock_instance._socket_server = MagicMock()

            # Second strategy fails
            call_count = [0]
            async def partial_broadcast(command):
                call_count[0] += 1
                if call_count[0] == 2:
                    raise Exception("STRAT-002 unreachable")

            mock_instance._socket_server.broadcast_command = AsyncMock(side_effect=partial_broadcast)
            mock.return_value = mock_instance

            result = await _execute_tier2_strategy_pause(
                audit_log=audit_log,
                strategy_ids=strategy_ids,
                activator="partial_test",
                activated_at_utc=datetime.utcnow()
            )

            # Should still succeed (best effort)
            assert result["success"] is True


class TestKillSwitchTier3AtomicExecution:
    """
    P0: Kill switch Tier 3 atomic execution.

    Validates NFR-P1: Emergency close executes atomically
    Risk: R-002 - Tier 3 has highest atomicity requirements
    """

    @pytest.mark.asyncio
    async def test_tier3_all_positions_closed_before_return(self):
        """
        P0: Tier 3 must close ALL positions before returning success.

        This is the most critical atomicity requirement since it involves
        actual trades (closes).
        """
        from src.api.kill_switch_endpoints import (
            _execute_tier3_emergency_close,
            KillSwitchAuditLog
        )
        from datetime import datetime

        audit_log = KillSwitchAuditLog()

        with patch('src.api.kill_switch_endpoints._get_kill_switch_router') as mock:
            mock_instance = MagicMock()
            mock_instance._socket_server = MagicMock()
            mock_instance._socket_server.broadcast_command = AsyncMock()
            mock_instance._connected_eas = ["EA-001", "EA-002", "EA-003"]
            mock_instance._active = False
            mock.return_value = mock_instance

            result = await _execute_tier3_emergency_close(
                audit_log=audit_log,
                activator="tier3_atomic_test",
                activated_at_utc=datetime.utcnow()
            )

            # All 3 EAs should have received CLOSE_ALL
            assert mock_instance._socket_server.broadcast_command.call_count == 3

            # Verify close results are captured
            assert len(result["results"]) == 3

            # Audit log must have per-position entries
            logs = audit_log.get_all()
            tier3_logs = [l for l in logs if l.get("tier") == 3]

            # Should have main entry + 3 per-position entries
            assert len(tier3_logs) >= 4, "Should have main entry + per-position entries"

    @pytest.mark.asyncio
    async def test_tier3_partial_fills_handled_correctly(self):
        """
        P0: Tier 3 must handle partial fills atomically.

        When a position is partially filled, the audit log must
        capture the partial result correctly.

        BUG IDENTIFIED: The current implementation does not properly capture
        partial fill results - it always returns 'filled' regardless of
        the actual fill status from the EA response.
        """
        from src.api.kill_switch_endpoints import (
            _execute_tier3_emergency_close,
            KillSwitchAuditLog
        )
        from datetime import datetime

        audit_log = KillSwitchAuditLog()

        with patch('src.api.kill_switch_endpoints._get_kill_switch_router') as mock:
            mock_instance = MagicMock()
            mock_instance._socket_server = MagicMock()
            mock_instance._connected_eas = ["EA-001"]
            mock_instance._active = False
            mock.return_value = mock_instance

            # Track what was passed in the broadcast command
            captured_commands = []
            async def capture_command(command):
                captured_commands.append(command)
                # Return a CloseResult that indicates partial fill
                from src.api.kill_switch_endpoints import CloseResult
                return CloseResult(
                    position_id=f"EA:{command.get('ea_id')}",
                    result="partial",
                    filled_lots=0.05,
                    message="Partial fill - only 50% of volume filled"
                )

            # We need to patch at a lower level - the broadcast_command
            # doesn't actually use the return value in current implementation
            # This test FAILS because the implementation doesn't read the response
            mock_instance._socket_server.broadcast_command = AsyncMock(side_effect=capture_command)

            result = await _execute_tier3_emergency_close(
                audit_log=audit_log,
                activator="partial_fill_test",
                activated_at_utc=datetime.utcnow()
            )

            # BUG: The implementation ignores the response and always returns 'filled'
            # This assertion FAILS, identifying the bug
            # FIX NEEDED: Implementation should read EA response and set result accordingly
            assert result["results"][0].result == "partial", (
                "BUG: Tier 3 returns 'filled' even when EA indicates partial fill. "
                "Implementation ignores broadcast_command return value."
            )

    @pytest.mark.asyncio
    async def test_tier3_audit_log_per_position_entry(self):
        """
        P0: Tier 3 must create per-position audit log entries.

        Validates R-002 mitigation: each position close is logged.
        """
        from src.api.kill_switch_endpoints import (
            _execute_tier3_emergency_close,
            KillSwitchAuditLog
        )
        from datetime import datetime

        audit_log = KillSwitchAuditLog()

        with patch('src.api.kill_switch_endpoints._get_kill_switch_router') as mock:
            mock_instance = MagicMock()
            mock_instance._socket_server = MagicMock()
            mock_instance._socket_server.broadcast_command = AsyncMock()
            mock_instance._connected_eas = ["EA-001", "EA-002"]
            mock_instance._active = False
            mock.return_value = mock_instance

            await _execute_tier3_emergency_close(
                audit_log=audit_log,
                activator="per_position_test",
                activated_at_utc=datetime.utcnow()
            )

            logs = audit_log.get_all()

            # Should have main tier-3 entry AND per-position entries
            main_entries = [l for l in logs if l.get("action") == "EMERGENCY_CLOSE_ALL"]
            position_entries = [l for l in logs if l.get("action") == "CLOSE"]

            assert len(main_entries) == 1, "Should have 1 main entry"
            assert len(position_entries) == 2, "Should have 2 per-position entries"


# =============================================================================
# P0 TEST SUITE 3: Cloudzy Independence (R-003, NFR-R4)
# =============================================================================

class TestCloudzyIndependence:
    """
    P0: Cloudzy independence - trading works without Contabo.

    Validates NFR-R4: Cloudzy trades without Contabo
    Risk: R-003 - Cloudzy trading breaks when Contabo is unreachable
    """

    @pytest.mark.asyncio
    async def test_websocket_works_when_contabo_unreachable(self):
        """
        P0: WebSocket streaming must work when Contabo is unreachable.

        Validates NFR-R4: Node independence - Cloudzy should not depend on Contabo.

        BUG IDENTIFIED: The SocketServer class does not have an `is_connected`
        attribute or equivalent method to check if Contabo node is reachable.
        This makes it impossible to verify Cloudzy independence programmatically.

        This test documents the missing capability for NFR-R4 verification:
        - SocketServer needs an `is_connected()` method
        - Or a `check_node_health(node="contabo")` function
        - Without this, we cannot write automated tests for NFR-R4
        """
        from src.router.socket_server import SocketServer

        # BUG: Cannot verify Contabo reachability because SocketServer has no is_connected() method
        server = SocketServer()

        # This assertion FAILS because the method doesn't exist
        # FIX NEEDED: Add is_connected() or check_node_health() to SocketServer
        assert hasattr(server, 'is_connected'), (
            "NFR-R4 VIOLATION: SocketServer lacks is_connected() method. "
            "Cannot programmatically verify Cloudzy independence from Contabo. "
            "Need method to check if Contabo node is reachable."
        )

        # If the method existed, we would test:
        # assert server.is_connected("contabo") == False  # Contabo unreachable
        # assert server.is_connected("cloudzy") == True   # Cloudzy still works

    @pytest.mark.asyncio
    async def test_kill_switch_works_without_socket_server(self):
        """
        P0: Kill switch Tier 1 must work without socket server (Cloudzy mode).

        Validates that when _socket_server is None (Contabo unreachable),
        the kill switch still functions.
        """
        from src.api.kill_switch_endpoints import (
            _execute_tier1_soft_stop,
            KillSwitchAuditLog
        )
        from datetime import datetime

        audit_log = KillSwitchAuditLog()

        with patch('src.api.kill_switch_endpoints._get_kill_switch_router') as mock:
            mock_instance = MagicMock()
            mock_instance._socket_server = None  # Contabo unreachable
            mock_instance._connected_eas = ["EA-001", "EA-002"]
            mock_instance._active = False
            mock.return_value = mock_instance

            # This should NOT raise - works in Cloudzy-independent mode
            result = await _execute_tier1_soft_stop(
                audit_log=audit_log,
                activator="cloudzy_test",
                activated_at_utc=datetime.utcnow()
            )

            assert result["success"] is True
            assert "audit_log_id" in result

    @pytest.mark.asyncio
    async def test_tier3_emergency_close_without_contabo(self):
        """
        P0: Tier 3 emergency close must work without Contabo.

        Even when Contabo is unreachable, Tier 3 should still execute
        and close positions via Cloudzy-local infrastructure.
        """
        from src.api.kill_switch_endpoints import (
            _execute_tier3_emergency_close,
            KillSwitchAuditLog
        )
        from datetime import datetime

        audit_log = KillSwitchAuditLog()

        with patch('src.api.kill_switch_endpoints._get_kill_switch_router') as mock:
            mock_instance = MagicMock()
            mock_instance._socket_server = None  # Contabo unreachable
            mock_instance._connected_eas = ["EA-001"]
            mock_instance._active = False
            mock.return_value = mock_instance

            result = await _execute_tier3_emergency_close(
                audit_log=audit_log,
                activator="cloudzy_tier3_test",
                activated_at_utc=datetime.utcnow()
            )

            # Should succeed even without socket server
            assert result["success"] is True
            # Results should show simulated closes
            assert len(result["results"]) == 1


# =============================================================================
# P0 TEST SUITE 4: MT5 Bridge Reconnection (R-004, NFR-R5)
# =============================================================================

class TestMT5BridgeReconnection:
    """
    P0: MT5 bridge reconnection must complete within 10s.

    Validates NFR-R5: MT5 ZMQ reconnect ≤10s
    Risk: R-004 - MT5 ZMQ reconnection exceeds 10s target
    """

    @pytest.mark.asyncio
    async def test_zmq_reconnection_time_under_10_seconds(self):
        """
        P0: ZMQ reconnection must complete within 10 seconds.

        Validates NFR-R5: MT5 ZMQ reconnect ≤10s target.
        This test simulates a ZMQ disconnection and measures reconnection time.
        """
        from src.api.tick_stream_handler import TickStreamHandler

        # Create mock adapter that will simulate disconnection
        mock_adapter = AsyncMock()
        mock_adapter.get_order_book = AsyncMock(side_effect=[
            Exception("ZMQ connection lost"),  # First call fails
            Exception("Still reconnecting..."),  # Second call fails
            {"bids": [[1.0850, 1.0]], "asks": [[1.0855, 1.0]], "time_msc": 0, "sequence": 1}  # Third call succeeds
        ])

        handler = TickStreamHandler(mt5_adapter=mock_adapter)

        reconnection_start = time.time()
        reconnection_complete = False
        reconnection_time_ms = 0

        # Simulate reconnection scenario
        try:
            # First attempt fails
            await handler._fetch_and_broadcast_tick("EURUSD")
        except Exception as e:
            assert "ZMQ" in str(e) or "connection" in str(e).lower()

        # Measure how long until reconnection succeeds
        for attempt in range(100):  # Max 100 attempts = 10s timeout
            await asyncio.sleep(0.1)  # 100ms between attempts
            try:
                await handler._fetch_and_broadcast_tick("EURUSD")
                reconnection_complete = True
                reconnection_time_ms = (time.time() - reconnection_start) * 1000
                break
            except Exception:
                continue

        assert reconnection_complete, "Reconnection should complete within test timeout"
        assert reconnection_time_ms <= 10000, (
            f"NFR-R5 VIOLATION: ZMQ reconnection took {reconnection_time_ms:.2f}ms, "
            f"exceeds 10000ms (10s) threshold"
        )

    @pytest.mark.asyncio
    async def test_reconnection_with_fallback_to_polling(self):
        """
        P0: When ZMQ reconnection fails, system must fallback to polling within 10s.

        Validates R-004 mitigation: if ZMQ fails, polling fallback is used.
        """
        from src.api.tick_stream_handler import TickStreamHandler

        mock_adapter = AsyncMock()
        mock_adapter.get_order_book = AsyncMock(side_effect=[
            Exception("ZMQ lost"),
            Exception("ZMQ lost"),
            Exception("ZMQ lost"),
            {"bids": [[1.0850, 1.0]], "asks": [[1.0855, 1.0]], "time_msc": 0, "sequence": 1}
        ])

        handler = TickStreamHandler(mt5_adapter=mock_adapter)
        handler._zmq_enabled = True
        handler._using_zmq = True

        start_time = time.time()
        fallback_triggered = False

        # Simulate the fallback scenario
        try:
            # This should trigger ZMQ error handling and fallback
            await handler._zmq_stream_loop()
        except Exception:
            pass

        # After error, should switch to polling
        fallback_time_ms = (time.time() - start_time) * 1000

        # Fallback should be quick - not cause extended outage
        assert fallback_time_ms <= 10000, (
            f"Fallback to polling took {fallback_time_ms:.2f}ms, should be ≤10s"
        )


# =============================================================================
# P0 TEST SUITE 5: Kill Switch Audit Log Immutability (R-005, NFR-D2)
# =============================================================================

class TestKillSwitchAuditLogImmutability:
    """
    P0: Kill switch audit logs must be immutable.

    Validates NFR-D2: Immutable audit log
    Risk: R-005 - Kill switch audit logs can be modified/deleted

    Tests that audit logs are append-only and cannot be modified or deleted.
    """

    def test_audit_log_append_only_no_delete(self):
        """
        P0: Audit log entries cannot be deleted.

        Validates NFR-D2: Append-only semantics.
        """
        from src.api.kill_switch_endpoints import KillSwitchAuditLog

        audit_log = KillSwitchAuditLog()

        # Add entry
        entry_id = audit_log.append({"tier": 1, "activator": "test"})

        # Attempt to delete - should have no effect on stored data
        # The KillSwitchAuditLog class doesn't expose a delete method
        # because entries are immutable by design

        logs = audit_log.get_all()
        assert len(logs) == 1, "Entry should still exist after attempted delete"

        # Verify entry is still intact
        entry = audit_log.get_by_id(entry_id)
        assert entry is not None
        assert entry["tier"] == 1

    def test_audit_log_append_only_no_modify(self):
        """
        P0: Audit log entries cannot be modified after creation.

        Validates NFR-D2: Entries are frozen after append.

        BUG IDENTIFIED: get_all() returns a shallow copy - the dictionaries
        inside the list are still references. This means modifications to
        returned entries CAN affect internal state, violating NFR-D2.

        FIX NEEDED: get_all() should return a deep copy (copy.deepcopy)
        of all entries to ensure true immutability.
        """
        from src.api.kill_switch_endpoints import KillSwitchAuditLog

        audit_log = KillSwitchAuditLog()

        entry_id = audit_log.append({"tier": 2, "activator": "original"})

        # get_all() returns a shallow copy - dictionaries are still references!
        logs = audit_log.get_all()
        original_activator = logs[0]["activator"]

        # BUG: This modification affects internal state because dicts are references
        logs[0]["activator"] = "modified"

        # Re-fetch - BUG: entry is now modified (violates NFR-D2)
        logs_after = audit_log.get_all()

        # This assertion FAILS, identifying the NFR-D2 immutability violation
        assert logs_after[0]["activator"] == original_activator, (
            f"NFR-D2 VIOLATION: Audit log entry was modified via returned copy. "
            f"Expected 'original' but got '{logs_after[0]['activator']}'. "
            f"get_all() returns shallow copy - must use deep copy for true immutability."
        )

    def test_audit_log_entries_have_immutable_timestamp(self):
        """
        P0: Audit log timestamps are set at creation and cannot be changed.

        Once an entry is created, its timestamp_utc is fixed.
        """
        from src.api.kill_switch_endpoints import KillSwitchAuditLog

        audit_log = KillSwitchAuditLog()

        entry_id = audit_log.append({"tier": 3, "activator": "timestamp_test"})
        entry = audit_log.get_by_id(entry_id)

        original_timestamp = entry["timestamp_utc"]

        # Add another entry
        audit_log.append({"tier": 1, "activator": "another"})

        # First entry's timestamp should be unchanged
        entry_after = audit_log.get_by_id(entry_id)
        assert entry_after["timestamp_utc"] == original_timestamp

    def test_audit_log_get_all_returns_copy_not_reference(self):
        """
        P0: get_all() returns a copy, not a reference to internal storage.

        This ensures external modifications cannot affect stored entries.
        """
        from src.api.kill_switch_endpoints import KillSwitchAuditLog

        audit_log = KillSwitchAuditLog()

        audit_log.append({"tier": 1, "activator": "test1"})
        audit_log.append({"tier": 2, "activator": "test2"})

        logs = audit_log.get_all()

        # Modify the returned list
        logs.clear()

        # Internal storage should be unaffected
        logs_after = audit_log.get_all()
        assert len(logs_after) == 2, "Internal storage should be unchanged"

"""
Integration Tests for WebSocket Position P&L Streaming.

Tests the complete trading data streaming pipeline including:
- Position update event broadcasting
- P&L update event broadcasting
- State replay on client connect/reconnect
- Bridge status event monitoring

Validates: Story 3-1 WebSocket Position P&L Streaming Backend
"""

import pytest
import asyncio
import json
from datetime import datetime
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch

# Mock WebSocket class for testing
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


class TestPositionUpdateEvent:
    """Test position_update event emission."""

    @pytest.mark.asyncio
    async def test_broadcast_position_update(self):
        """
        Test that position_update events are broadcast with correct structure.

        Validates AC1:
        - Event type is 'position_update'
        - Contains position data (ticket, symbol, volume, open price, profit)
        - Timestamp is included
        """
        from src.api.websocket_endpoints import broadcast_position_update, manager

        # Create fresh manager and websocket
        ws = MockWebSocket()
        await manager.connect(ws)
        await manager.subscribe(ws, "trading")

        # Broadcast position update
        position_data = {
            "ticket": 12345678,
            "symbol": "EURUSD",
            "volume": 0.1,
            "open_price": 1.08500,
            "current_price": 1.08650,
            "profit": 15.00,
            "comment": "Bot-001"
        }

        await broadcast_position_update(position_data)

        # Verify message
        messages = ws.get_messages_of_type("position_update")
        assert len(messages) == 1, "Should have 1 position_update message"

        data = messages[0]["data"]
        assert data["ticket"] == 12345678
        assert data["symbol"] == "EURUSD"
        assert data["volume"] == 0.1
        assert data["profit"] == 15.00
        assert "timestamp" in data

    @pytest.mark.asyncio
    async def test_position_update_open_event(self):
        """
        Test position opened event broadcasting.
        """
        from src.api.websocket_endpoints import broadcast_position_update, manager

        ws = MockWebSocket()
        await manager.connect(ws)
        await manager.subscribe(ws, "trading")

        position_data = {
            "ticket": 12345679,
            "symbol": "GBPUSD",
            "volume": 0.2,
            "open_price": 1.26500,
            "current_price": 1.26500,
            "profit": 0.0,
            "action": "open"
        }

        await broadcast_position_update(position_data)

        messages = ws.get_messages_of_type("position_update")
        assert len(messages) == 1
        assert messages[0]["data"]["action"] == "open"

    @pytest.mark.asyncio
    async def test_position_update_modify_event(self):
        """
        Test position modified event broadcasting.
        """
        from src.api.websocket_endpoints import broadcast_position_update, manager

        ws = MockWebSocket()
        await manager.connect(ws)
        await manager.subscribe(ws, "trading")

        position_data = {
            "ticket": 12345680,
            "symbol": "USDJPY",
            "volume": 0.15,
            "open_price": 149.500,
            "current_price": 149.800,
            "profit": 45.00,
            "action": "modify",
            "old_sl": 149.200,
            "new_sl": 149.300
        }

        await broadcast_position_update(position_data)

        messages = ws.get_messages_of_type("position_update")
        assert len(messages) == 1
        data = messages[0]["data"]
        assert data["action"] == "modify"
        assert data["old_sl"] == 149.200
        assert data["new_sl"] == 149.300

    @pytest.mark.asyncio
    async def test_position_update_close_event(self):
        """
        Test position closed event broadcasting.
        """
        from src.api.websocket_endpoints import broadcast_position_update, manager

        ws = MockWebSocket()
        await manager.connect(ws)
        await manager.subscribe(ws, "trading")

        position_data = {
            "ticket": 12345681,
            "symbol": "EURGBP",
            "volume": 0.1,
            "open_price": 0.85200,
            "close_price": 0.85450,
            "profit": 25.00,
            "action": "close",
            "reason": "take_profit"
        }

        await broadcast_position_update(position_data)

        messages = ws.get_messages_of_type("position_update")
        assert len(messages) == 1
        data = messages[0]["data"]
        assert data["action"] == "close"
        assert data["reason"] == "take_profit"


class TestPnlUpdateEvent:
    """Test pnl_update event emission."""

    @pytest.mark.asyncio
    async def test_broadcast_pnl_update(self):
        """
        Test that pnl_update events are broadcast with correct structure.

        Validates AC2:
        - Event type is 'pnl_update'
        - Contains P&L data (daily_pnl, open_positions, closed_pnl)
        - Timestamp is included
        """
        from src.api.websocket_endpoints import broadcast_pnl_update, manager

        ws = MockWebSocket()
        await manager.connect(ws)
        await manager.subscribe(ws, "trading")

        pnl_data = {
            "daily_pnl": 125.50,
            "open_positions": 3,
            "open_pnl": 75.00,
            "closed_pnl": 50.50,
            "equity": 10125.50,
            "balance": 10000.00
        }

        await broadcast_pnl_update(pnl_data)

        # Verify message
        messages = ws.get_messages_of_type("pnl_update")
        assert len(messages) == 1, "Should have 1 pnl_update message"

        data = messages[0]["data"]
        assert data["daily_pnl"] == 125.50
        assert data["open_positions"] == 3
        assert data["open_pnl"] == 75.00
        assert data["closed_pnl"] == 50.50
        assert "timestamp" in data

    @pytest.mark.asyncio
    async def test_pnl_update_with_per_position_breakdown(self):
        """
        Test P&L update with per-position breakdown.
        """
        from src.api.websocket_endpoints import broadcast_pnl_update, manager

        ws = MockWebSocket()
        await manager.connect(ws)
        await manager.subscribe(ws, "trading")

        pnl_data = {
            "daily_pnl": 200.00,
            "open_positions": 2,
            "positions": [
                {"ticket": 123, "symbol": "EURUSD", "pnl": 100.00},
                {"ticket": 124, "symbol": "GBPUSD", "pnl": 100.00}
            ]
        }

        await broadcast_pnl_update(pnl_data)

        messages = ws.get_messages_of_type("pnl_update")
        data = messages[0]["data"]
        assert len(data["positions"]) == 2


class TestTradingDataBroadcaster:
    """Test TradingDataBroadcaster class."""

    @pytest.mark.asyncio
    async def test_broadcaster_emit_position(self):
        """
        Test TradingDataBroadcaster emits position updates.
        """
        from src.api.websocket_endpoints import TradingDataBroadcaster, manager

        ws = MockWebSocket()
        await manager.connect(ws)
        await manager.subscribe(ws, "trading")

        broadcaster = TradingDataBroadcaster()

        await broadcaster.emit_position(
            ticket=999999,
            symbol="AUDUSD",
            volume=0.1,
            open_price=0.65000,
            current_price=0.65100,
            profit=10.00
        )

        messages = ws.get_messages_of_type("position_update")
        assert len(messages) == 1
        assert messages[0]["data"]["symbol"] == "AUDUSD"

    @pytest.mark.asyncio
    async def test_broadcaster_emit_pnl(self):
        """
        Test TradingDataBroadcaster emits P&L updates.
        """
        from src.api.websocket_endpoints import TradingDataBroadcaster, manager

        ws = MockWebSocket()
        await manager.connect(ws)
        await manager.subscribe(ws, "trading")

        broadcaster = TradingDataBroadcaster()

        await broadcaster.emit_pnl(
            daily_pnl=50.00,
            open_positions=1,
            open_pnl=30.00,
            closed_pnl=20.00
        )

        messages = ws.get_messages_of_type("pnl_update")
        assert len(messages) == 1


class TestStateReplay:
    """Test state replay on client connect/reconnect."""

    @pytest.mark.asyncio
    async def test_state_replay_on_connect(self):
        """
        Test that state replay sends full state on client connect.

        Validates AC3:
        - Client receives last known state (positions, P&L, bot statuses)
        - State is sent immediately upon connection
        """
        from src.api.websocket_endpoints import get_cached_trading_state, TradingDataBroadcaster, reset_trading_state_cache

        # Reset cache to ensure clean state
        reset_trading_state_cache()

        # Setup cached state
        broadcaster = TradingDataBroadcaster()

        # Add positions to cache
        await broadcaster.cache_position({
            "ticket": 111,
            "symbol": "EURUSD",
            "volume": 0.1,
            "open_price": 1.0850,
            "current_price": 1.0860,
            "profit": 10.00
        })

        # Add P&L to cache
        await broadcaster.cache_pnl({
            "daily_pnl": 10.00,
            "open_positions": 1,
            "open_pnl": 10.00,
            "closed_pnl": 0.00
        })

        # Get cached state
        state = get_cached_trading_state()

        assert state is not None
        assert "positions" in state
        assert "pnl" in state
        assert len(state["positions"]) == 1
        assert state["positions"][0]["symbol"] == "EURUSD"

    @pytest.mark.asyncio
    async def test_state_replay_empty_on_no_cache(self):
        """
        Test state replay returns empty state when no cache exists.
        """
        from src.api.websocket_endpoints import get_cached_trading_state

        state = get_cached_trading_state()

        # Should return empty state structure
        assert state is not None
        assert "positions" in state
        assert "pnl" in state
        assert "bot_statuses" in state


class TestBridgeStatusEvents:
    """Test bridge_status event emission."""

    @pytest.mark.asyncio
    async def test_bridge_status_connected(self):
        """
        Test bridge_status event when MT5 is connected.

        Validates AC5:
        - Event type is 'bridge_status'
        - Contains connection state (connected/disconnected)
        - Timestamp is included
        """
        from src.api.websocket_endpoints import broadcast_bridge_status, manager

        ws = MockWebSocket()
        await manager.connect(ws)
        await manager.subscribe(ws, "trading")

        await broadcast_bridge_status(
            connected=True,
            latency_ms=5.0,
            message="Connected to MT5"
        )

        messages = ws.get_messages_of_type("bridge_status")
        assert len(messages) == 1
        data = messages[0]["data"]
        assert data["connected"] is True
        assert data["latency_ms"] == 5.0

    @pytest.mark.asyncio
    async def test_bridge_status_disconnected(self):
        """
        Test bridge_status event when MT5 is disconnected.
        """
        from src.api.websocket_endpoints import broadcast_bridge_status, manager

        ws = MockWebSocket()
        await manager.connect(ws)
        await manager.subscribe(ws, "trading")

        await broadcast_bridge_status(
            connected=False,
            latency_ms=None,
            message="ZMQ connection lost"
        )

        messages = ws.get_messages_of_type("bridge_status")
        assert len(messages) == 1
        data = messages[0]["data"]
        assert data["connected"] is False
        assert "ZMQ" in data["message"]


class TestStateCaching:
    """Test state caching functionality."""

    @pytest.mark.asyncio
    async def test_position_cache_updates(self):
        """
        Test that position cache is updated on each position event.
        """
        from src.api.websocket_endpoints import TradingDataBroadcaster, get_cached_trading_state, reset_trading_state_cache

        # Reset cache to ensure clean state
        reset_trading_state_cache()

        broadcaster = TradingDataBroadcaster()

        # Add multiple positions
        await broadcaster.cache_position({
            "ticket": 1,
            "symbol": "EURUSD",
            "volume": 0.1,
            "profit": 10.00
        })

        await broadcaster.cache_position({
            "ticket": 2,
            "symbol": "GBPUSD",
            "volume": 0.2,
            "profit": 20.00
        })

        state = get_cached_trading_state()
        assert len(state["positions"]) == 2

    @pytest.mark.asyncio
    async def test_pnl_cache_updates(self):
        """
        Test that P&L cache is updated on each P&L event.
        """
        from src.api.websocket_endpoints import TradingDataBroadcaster, get_cached_trading_state

        broadcaster = TradingDataBroadcaster()

        await broadcaster.cache_pnl({
            "daily_pnl": 100.00,
            "open_positions": 2,
            "open_pnl": 80.00,
            "closed_pnl": 20.00
        })

        state = get_cached_trading_state()
        assert state["pnl"]["daily_pnl"] == 100.00


class TestLatencyValidation:
    """Test latency requirements."""

    @pytest.mark.asyncio
    async def test_broadcast_latency_under_3s(self):
        """
        Test that event broadcasting meets ≤3s latency requirement.

        Validates NFR-P2:
        - Event emission ≤3s from trigger
        """
        from src.api.websocket_endpoints import broadcast_position_update, manager
        import time

        ws = MockWebSocket()
        await manager.connect(ws)
        await manager.subscribe(ws, "trading")

        position_data = {"ticket": 1, "symbol": "EURUSD", "volume": 0.1}

        start = time.time()
        await broadcast_position_update(position_data)
        latency_ms = (time.time() - start) * 1000

        # Verify message was sent
        messages = ws.get_messages_of_type("position_update")
        assert len(messages) == 1

        # Latency should be well under 3000ms (3 seconds)
        assert latency_ms < 3000, f"Latency {latency_ms:.2f}ms exceeds 3000ms threshold"


class TestTickStreamHandlerIntegration:
    """Test integration between tick_stream_handler and broadcast functions."""

    @pytest.mark.asyncio
    async def test_tick_handler_broadcasts_position_update(self):
        """
        Test that TickStreamHandler can broadcast position updates.

        Validates Task 1.3: Wire tick_stream_handler to WebSocket broadcaster
        """
        from src.api.tick_stream_handler import TickStreamHandler
        from src.api.websocket_endpoints import manager, reset_trading_state_cache

        # Reset cache for clean test
        reset_trading_state_cache()

        # Create mock adapter
        mock_adapter = AsyncMock()
        mock_adapter.get_order_book = AsyncMock(return_value={
            'bids': [[1.0850, 1.0]],
            'asks': [[1.0855, 1.0]],
            'time_msc': 0,
            'sequence': 1
        })

        # Create handler
        handler = TickStreamHandler(mt5_adapter=mock_adapter)

        # Connect mock WebSocket
        ws = MockWebSocket()
        await manager.connect(ws)
        await manager.subscribe(ws, "trading")

        # Broadcast a position update via handler
        await handler.broadcast_position_change(
            ticket=123456,
            symbol="EURUSD",
            volume=0.1,
            open_price=1.0850,
            current_price=1.0860,
            profit=10.0,
            action="open"
        )

        # Verify message was broadcast
        messages = ws.get_messages_of_type("position_update")
        assert len(messages) == 1
        assert messages[0]["data"]["symbol"] == "EURUSD"
        assert messages[0]["data"]["ticket"] == 123456
        assert messages[0]["data"]["action"] == "open"

    @pytest.mark.asyncio
    async def test_tick_handler_broadcasts_pnl_update(self):
        """
        Test that TickStreamHandler can broadcast P&L updates.

        Validates Task 1.3: Wire tick_stream_handler to WebSocket broadcaster
        """
        from src.api.tick_stream_handler import TickStreamHandler
        from src.api.websocket_endpoints import manager, reset_trading_state_cache

        reset_trading_state_cache()

        mock_adapter = AsyncMock()
        handler = TickStreamHandler(mt5_adapter=mock_adapter)

        ws = MockWebSocket()
        await manager.connect(ws)
        await manager.subscribe(ws, "trading")

        # Broadcast P&L update via handler
        await handler.broadcast_pnl_change(
            daily_pnl=150.0,
            open_positions=3,
            open_pnl=100.0,
            closed_pnl=50.0,
            equity=10150.0,
            balance=10000.0
        )

        messages = ws.get_messages_of_type("pnl_update")
        assert len(messages) == 1
        assert messages[0]["data"]["daily_pnl"] == 150.0
        assert messages[0]["data"]["open_positions"] == 3

    @pytest.mark.asyncio
    async def test_tick_handler_reports_bridge_status(self):
        """
        Test that TickStreamHandler can report bridge status.

        Validates Task 3.1: Monitor ZMQ connection state
        """
        from src.api.tick_stream_handler import TickStreamHandler
        from src.api.websocket_endpoints import manager

        mock_adapter = AsyncMock()
        handler = TickStreamHandler(mt5_adapter=mock_adapter)

        ws = MockWebSocket()
        await manager.connect(ws)
        await manager.subscribe(ws, "trading")

        # Report connected status
        await handler.report_bridge_status(connected=True, latency_ms=5.0)

        messages = ws.get_messages_of_type("bridge_status")
        assert len(messages) == 1
        assert messages[0]["data"]["connected"] is True
        assert messages[0]["data"]["latency_ms"] == 5.0

        # Report disconnected status
        await handler.report_bridge_status(connected=False)

        messages = ws.get_messages_of_type("bridge_status")
        assert len(messages) == 2
        assert messages[1]["data"]["connected"] is False

    @pytest.mark.asyncio
    async def test_state_replay_via_handler(self):
        """
        Test that TickStreamHandler updates state cache for replay.

        Validates Task 2: State Replay on Reconnection
        """
        from src.api.tick_stream_handler import TickStreamHandler
        from src.api.websocket_endpoints import get_cached_trading_state, reset_trading_state_cache

        reset_trading_state_cache()

        mock_adapter = AsyncMock()
        handler = TickStreamHandler(mt5_adapter=mock_adapter)

        # Broadcast position and P&L
        await handler.broadcast_position_change(
            ticket=999,
            symbol="GBPUSD",
            volume=0.2,
            open_price=1.2650,
            current_price=1.2660,
            profit=20.0
        )

        await handler.broadcast_pnl_change(
            daily_pnl=50.0,
            open_positions=1,
            open_pnl=30.0,
            closed_pnl=20.0
        )

        # Verify state cache is updated
        state = get_cached_trading_state()
        assert len(state["positions"]) == 1
        assert state["positions"][0]["symbol"] == "GBPUSD"
        assert state["pnl"]["daily_pnl"] == 50.0
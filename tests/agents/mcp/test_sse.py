"""
Unit Tests for MCP SSE Handler

Tests SSE event handling for MCP servers.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from src.agents.mcp.sse import (
    SSEEvent,
    SSEEventType,
    SSEEventBus,
    MCPSSEHandler,
    get_sse_handler,
)


class TestSSEEvent:
    """Tests for SSEEvent dataclass."""

    def test_create_event(self):
        """Test creating an SSE event."""
        event = SSEEvent(
            type=SSEEventType.CONNECTED,
            data={"server_id": "test"},
        )

        assert event.type == SSEEventType.CONNECTED
        assert event.data["server_id"] == "test"
        assert event.timestamp is not None

    def test_event_types(self):
        """Test all SSE event types."""
        assert SSEEventType.CONNECTED.value == "connected"
        assert SSEEventType.DISCONNECTED.value == "disconnected"
        assert SSEEventType.CONNECTING.value == "connecting"
        assert SSEEventType.ERROR.value == "error"
        assert SSEEventType.TOOL_DISCOVERED.value == "tool_discovered"
        assert SSEEventType.TOOL_CALLED.value == "tool_called"
        assert SSEEventType.TOOL_RESULT.value == "tool_result"
        assert SSEEventType.STATUS_CHANGE.value == "status_change"
        assert SSEEventType.HEARTBEAT.value == "heartbeat"


class TestSSEEventBus:
    """Tests for SSEEventBus class."""

    @pytest.fixture
    def event_bus(self):
        """Create an event bus."""
        return SSEEventBus()

    @pytest.mark.asyncio
    async def test_publish_and_subscribe(self, event_bus):
        """Test publishing and subscribing to events."""
        event = SSEEvent(
            type=SSEEventType.CONNECTED,
            data={"server_id": "test"},
        )

        # Publish event
        await event_bus.publish(event)

        # Subscribe and receive
        received = []
        async for received_event in event_bus.subscribe():
            received.append(received_event)
            break  # Only get one event

        assert len(received) == 1
        assert received[0].type == SSEEventType.CONNECTED

    @pytest.mark.asyncio
    async def test_multiple_subscribers(self, event_bus):
        """Test multiple subscribers receive events."""
        event = SSEEvent(
            type=SSEEventType.TOOL_DISCOVERED,
            data={"tool_name": "test_tool"},
        )

        await event_bus.publish(event)

        # Collect from two subscribers
        results = []

        async def collector():
            async for e in event_bus.subscribe():
                results.append(e)
                if len(results) >= 2:
                    break

        # Give time for subscribers to start
        await asyncio.sleep(0.01)

        # Publish another event
        event2 = SSEEvent(
            type=SSEEventType.TOOL_CALLED,
            data={"tool_name": "test_tool2"},
        )
        await event_bus.publish(event2)

        # Wait for collection
        await asyncio.sleep(0.1)


class TestMCPSSEHandler:
    """Tests for MCPSSEHandler class."""

    @pytest.fixture
    def handler(self):
        """Create an SSE handler."""
        return MCPSSEHandler()

    @pytest.mark.asyncio
    async def test_start_stop(self, handler):
        """Test handler start and stop."""
        await handler.start()
        # Should be running (heartbeat task created)

        await handler.stop()
        # Should be stopped

    @pytest.mark.asyncio
    async def test_publish_connection_event(self, handler):
        """Test publishing connection events."""
        await handler.publish_connection_event(
            server_id="test-server",
            status="connected",
        )

        # Should track active connections
        connections = await handler.get_active_connections()
        assert "test-server" in connections

    @pytest.mark.asyncio
    async def test_publish_disconnect_event(self, handler):
        """Test publishing disconnection."""
        await handler.publish_connection_event("test-server", "connected")
        await handler.publish_connection_event("test-server", "disconnected")

        connections = await handler.get_active_connections()
        assert "test-server" not in connections

    @pytest.mark.asyncio
    async def test_publish_tool_discovered(self, handler):
        """Test publishing tool discovery event."""
        await handler.publish_tool_discovered(
            server_id="test-server",
            tool_name="read_file",
            description="Read a file from disk",
            input_schema={"type": "object"},
        )

    @pytest.mark.asyncio
    async def test_publish_tool_called(self, handler):
        """Test publishing tool call event."""
        await handler.publish_tool_called(
            server_id="test-server",
            tool_name="read_file",
            call_id="call-123",
            arguments={"path": "/tmp/test.txt"},
        )

    @pytest.mark.asyncio
    async def test_publish_tool_result(self, handler):
        """Test publishing tool result event."""
        await handler.publish_tool_result(
            server_id="test-server",
            tool_name="read_file",
            call_id="call-123",
            result={"content": "test content"},
        )

    @pytest.mark.asyncio
    async def test_publish_tool_result_error(self, handler):
        """Test publishing tool error result."""
        await handler.publish_tool_result(
            server_id="test-server",
            tool_name="read_file",
            call_id="call-123",
            result=None,
            error="File not found",
        )

    @pytest.mark.asyncio
    async def test_stream_all_events(self, handler):
        """Test streaming all events."""
        await handler.publish_connection_event("server1", "connected")
        await handler.publish_tool_discovered(
            "server1", "tool1", "A tool", {}
        )

        events = []
        async for event in handler.stream_events():
            events.append(event)
            if len(events) >= 2:
                break

        assert len(events) == 2

    @pytest.mark.asyncio
    async def test_stream_filtered_events(self, handler):
        """Test streaming filtered events."""
        await handler.publish_connection_event("server1", "connected")
        await handler.publish_tool_discovered("server1", "tool1", "A tool", {})

        # Only get tool events
        events = []
        async for event in handler.stream_events(event_types=[SSEEventType.TOOL_DISCOVERED]):
            events.append(event)
            if len(events) >= 1:
                break

        assert len(events) == 1
        assert events[0].type == SSEEventType.TOOL_DISCOVERED

    @pytest.mark.asyncio
    async def test_stream_server_events(self, handler):
        """Test streaming server-specific events."""
        await handler.publish_connection_event("server1", "connected")
        await handler.publish_connection_event("server2", "connected")

        events = []
        async for event in handler.stream_server_events("server1"):
            events.append(event)
            if len(events) >= 1:
                break

        assert len(events) == 1
        assert events[0].server_id == "server1"

    @pytest.mark.asyncio
    async def test_stream_tool_events(self, handler):
        """Test streaming tool-specific events."""
        await handler.publish_tool_called(
            "server1", "tool1", "call-1", {}
        )
        await handler.publish_tool_discovered(
            "server1", "tool1", "A tool", {}
        )

        events = []
        async for event in handler.stream_tool_events():
            events.append(event)
            if len(events) >= 2:
                break

        tool_events = [SSEEventType.TOOL_CALLED, SSEEventType.TOOL_DISCOVERED]
        assert all(e.type in tool_events for e in events)

    def test_format_sse(self, handler):
        """Test SSE formatting."""
        event = SSEEvent(
            type=SSEEventType.CONNECTED,
            server_id="test-server",
            data={"status": "connected"},
        )

        formatted = handler.format_sse(event)

        assert "data:" in formatted
        assert "connected" in formatted
        assert "test-server" in formatted


class TestGetSseHandler:
    """Tests for get_sse_handler function."""

    def test_get_sse_handler_singleton(self):
        """Test that get_sse_handler returns singleton."""
        # Reset global
        import src.agents.mcp.sse as sse_module
        sse_module._sse_handler = None

        handler1 = get_sse_handler()
        handler2 = get_sse_handler()

        assert handler1 is handler2

        # Cleanup
        sse_module._sse_handler = None

    @pytest.mark.asyncio
    async def test_singleton_sharing(self):
        """Test that singleton handlers share state."""
        # Reset for test
        import src.agents.mcp.sse as sse_module
        sse_module._sse_handler = None

        handler = get_sse_handler()
        await handler.publish_connection_event("shared-server", "connected")

        # Get same handler
        handler2 = get_sse_handler()
        connections = await handler2.get_active_connections()

        assert "shared-server" in connections

        # Cleanup
        await handler.stop()
        sse_module._sse_handler = None

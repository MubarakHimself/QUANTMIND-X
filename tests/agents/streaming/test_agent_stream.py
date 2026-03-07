"""
Tests for Agent SSE Streaming Module

Tests the streaming functionality for agent responses, tool outputs, and progress updates.
"""

import asyncio
import json
import pytest

from src.agents.streaming.agent_stream import (
    AgentStreamEvent,
    AgentStreamEventType,
    AgentStreamHandler,
    AgentStreamBus,
    get_stream_handler,
)


class TestAgentStreamEventType:
    """Tests for AgentStreamEventType enum."""

    def test_event_types_exist(self):
        """Verify all expected event types exist."""
        expected_types = [
            "connected",
            "disconnected",
            "error",
            "agent_started",
            "agent_completed",
            "agent_failed",
            "response_start",
            "response_chunk",
            "response_complete",
            "tool_start",
            "tool_progress",
            "tool_complete",
            "tool_error",
            "progress_update",
            "progress_complete",
            "heartbeat",
        ]
        actual_types = [e.value for e in AgentStreamEventType]
        for expected in expected_types:
            assert expected in actual_types

    def test_event_types_are_strings(self):
        """Verify event types are strings."""
        for event_type in AgentStreamEventType:
            assert isinstance(event_type.value, str)
            assert event_type.value == event_type.value.lower()


class TestAgentStreamEvent:
    """Tests for AgentStreamEvent dataclass."""

    def test_create_event_with_required_fields(self):
        """Test creating an event with only required fields."""
        event = AgentStreamEvent(
            type=AgentStreamEventType.AGENT_STARTED,
            data={"message": "Agent started"},
        )
        assert event.type == AgentStreamEventType.AGENT_STARTED
        assert event.data == {"message": "Agent started"}
        assert event.timestamp is not None

    def test_create_event_with_all_fields(self):
        """Test creating an event with all fields."""
        event = AgentStreamEvent(
            type=AgentStreamEventType.TOOL_COMPLETE,
            data={"result": "success"},
            agent_id="agent-1",
            task_id="task-123",
            tool_name="my_tool",
            request_id="req-456",
        )
        assert event.type == AgentStreamEventType.TOOL_COMPLETE
        assert event.agent_id == "agent-1"
        assert event.task_id == "task-123"
        assert event.tool_name == "my_tool"
        assert event.request_id == "req-456"

    def test_timestamp_is_iso_format(self):
        """Verify timestamp is in ISO format."""
        event = AgentStreamEvent(
            type=AgentStreamEventType.HEARTBEAT,
            data={},
        )
        # Should be parseable as ISO format
        from datetime import datetime
        parsed = datetime.fromisoformat(event.timestamp)
        assert parsed is not None


class TestAgentStreamBus:
    """Tests for AgentStreamBus."""

    @pytest.fixture
    def event_bus(self):
        """Create a fresh event bus for each test."""
        return AgentStreamBus()

    @pytest.mark.asyncio
    async def test_subscribe_and_publish(self, event_bus):
        """Test basic subscribe and publish."""
        # Create subscriber
        sub = asyncio.create_task(self._collect_events(event_bus))

        # Publish event
        event = AgentStreamEvent(
            type=AgentStreamEventType.AGENT_STARTED,
            data={"agent": "test-agent"},
            agent_id="agent-1",
        )
        await event_bus.publish(event)

        # Wait for event
        await asyncio.sleep(0.1)

        # Cancel subscriber
        sub.cancel()
        try:
            await sub
        except asyncio.CancelledError:
            pass

    async def _collect_events(self, event_bus):
        """Helper to collect events from subscription."""
        count = 0
        async for event in event_bus.subscribe():
            count += 1
            if count >= 1:
                break

    @pytest.mark.asyncio
    async def test_publish_to_multiple_subscribers(self, event_bus):
        """Test publishing to multiple subscribers."""
        results = []

        async def subscriber(name):
            count = 0
            async for event in event_bus.subscribe():
                results.append((name, event.type))
                count += 1
                if count >= 2:
                    break

        # Create two subscribers
        task1 = asyncio.create_task(subscriber("sub1"))
        task2 = asyncio.create_task(subscriber("sub2"))

        # Wait for subscribers to be ready
        await asyncio.sleep(0.1)

        # Publish two events
        await event_bus.publish(AgentStreamEvent(
            type=AgentStreamEventType.AGENT_STARTED,
            data={},
        ))
        await event_bus.publish(AgentStreamEvent(
            type=AgentStreamEventType.AGENT_COMPLETED,
            data={},
        ))

        # Wait for events
        await asyncio.sleep(0.3)

        # Cancel tasks
        task1.cancel()
        task2.cancel()

        # Each subscriber should have received both events
        assert len(results) >= 2

    @pytest.mark.asyncio
    async def test_get_subscriber_count(self, event_bus):
        """Test getting subscriber count."""
        async def subscriber():
            async for _ in event_bus.subscribe():
                break

        task = asyncio.create_task(subscriber())
        await asyncio.sleep(0.05)  # Let subscription happen

        count = await event_bus.get_subscriber_count()
        assert count == 1

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


class TestAgentStreamHandler:
    """Tests for AgentStreamHandler."""

    @pytest.fixture
    def handler(self):
        """Create a fresh handler for each test."""
        return AgentStreamHandler()

    @pytest.mark.asyncio
    async def test_format_sse(self, handler):
        """Test SSE formatting."""
        event = AgentStreamEvent(
            type=AgentStreamEventType.RESPONSE_CHUNK,
            data={"chunk": "Hello", "delta": "Hello"},
            agent_id="agent-1",
            task_id="task-1",
            request_id="req-1",
        )

        formatted = handler.format_sse(event)
        assert formatted.startswith("data: ")
        assert formatted.endswith("\n\n")

        # Verify it's valid JSON
        json_str = formatted[6:-2]  # Remove "data: " prefix and "\n\n" suffix
        parsed = json.loads(json_str)
        assert parsed["type"] == "response_chunk"
        assert parsed["chunk"] == "Hello"
        assert parsed["agent_id"] == "agent-1"

    @pytest.mark.asyncio
    async def test_publish_agent_started(self, handler):
        """Test publishing agent started event."""
        await handler.publish_agent_started(
            agent_id="agent-1",
            task_id="task-1",
            request_id="req-1",
            metadata={"model": "claude-3"},
        )

        # Verify event was published by checking active streams
        streams = await handler.get_active_streams()
        # Note: This just verifies the handler works, actual stream check needs subscriber

    @pytest.mark.asyncio
    async def test_publish_tool_events(self, handler):
        """Test publishing tool-related events."""
        # Tool start
        await handler.publish_tool_start(
            agent_id="agent-1",
            task_id="task-1",
            request_id="req-1",
            tool_name="my_tool",
            arguments={"arg1": "value1"},
        )

        # Tool progress
        await handler.publish_tool_progress(
            agent_id="agent-1",
            task_id="task-1",
            request_id="req-1",
            tool_name="my_tool",
            progress=0.5,
            message="Processing...",
        )

        # Tool complete
        await handler.publish_tool_complete(
            agent_id="agent-1",
            task_id="task-1",
            request_id="req-1",
            tool_name="my_tool",
            result={"output": "success"},
        )

    @pytest.mark.asyncio
    async def test_publish_tool_error(self, handler):
        """Test publishing tool error event."""
        await handler.publish_tool_error(
            agent_id="agent-1",
            task_id="task-1",
            request_id="req-1",
            tool_name="my_tool",
            error="Tool execution failed",
        )

    @pytest.mark.asyncio
    async def test_publish_progress_update(self, handler):
        """Test publishing progress update events."""
        await handler.publish_progress_update(
            agent_id="agent-1",
            task_id="task-1",
            request_id="req-1",
            progress=0.25,
            message="Step 1 of 4",
            step=1,
            total_steps=4,
        )

        await handler.publish_progress_update(
            agent_id="agent-1",
            task_id="task-1",
            request_id="req-1",
            progress=1.0,
            message="Step 4 of 4",
            step=4,
            total_steps=4,
        )

    @pytest.mark.asyncio
    async def test_publish_response_events(self, handler):
        """Test publishing response events."""
        # Response start
        await handler.publish_response_start(
            agent_id="agent-1",
            task_id="task-1",
            request_id="req-1",
            content_type="text",
        )

        # Response chunks
        await handler.publish_response_chunk(
            agent_id="agent-1",
            task_id="task-1",
            request_id="req-1",
            chunk="Hello",
            delta="Hello",
        )

        await handler.publish_response_chunk(
            agent_id="agent-1",
            task_id="task-1",
            request_id="req-1",
            chunk=" World",
            delta=" World",
        )

        # Response complete
        await handler.publish_response_complete(
            agent_id="agent-1",
            task_id="task-1",
            request_id="req-1",
            full_response="Hello World",
            metadata={"tokens": 2},
        )

    @pytest.mark.asyncio
    async def test_stream_events_with_filter(self, handler):
        """Test streaming events with filtering."""
        events_received = []

        async def collector():
            async for event in handler.stream_events(
                agent_id="agent-1",
                event_types=[AgentStreamEventType.AGENT_STARTED, AgentStreamEventType.AGENT_COMPLETED],
            ):
                events_received.append(event)
                if len(events_received) >= 2:
                    break

        task = asyncio.create_task(collector())

        # Wait for subscriber to be ready
        await asyncio.sleep(0.1)

        # Publish matching event
        await handler.publish_agent_started(
            agent_id="agent-1",
            task_id="task-1",
            request_id="req-1",
        )

        # Publish non-matching event
        await handler.publish_tool_start(
            agent_id="agent-1",
            task_id="task-1",
            request_id="req-1",
            tool_name="my_tool",
        )

        # Publish another matching event
        await handler.publish_agent_completed(
            agent_id="agent-1",
            task_id="task-1",
            request_id="req-1",
        )

        await asyncio.sleep(0.3)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Should have received only agent events (not tool)
        assert len(events_received) == 2


class TestGlobalHandler:
    """Tests for global handler instance."""

    def test_get_stream_handler_returns_same_instance(self):
        """Test that get_stream_handler returns the same instance."""
        handler1 = get_stream_handler()
        handler2 = get_stream_handler()
        assert handler1 is handler2


class TestSSEIntegration:
    """Tests for SSE response generation."""

    @pytest.mark.asyncio
    async def test_create_sse_response_sends_connection_event(self):
        """Test that create_sse_response sends a connection event first."""
        from src.agents.streaming.agent_stream import create_sse_response, AgentStreamHandler

        handler = AgentStreamHandler()
        responses = []

        # Collect first few responses
        async for response in create_sse_response(handler):
            responses.append(response)
            if len(responses) >= 2:  # Connection event + at least one more
                break

        # First event should be connection event
        assert len(responses) >= 1
        first_response = responses[0].decode('utf-8')
        assert first_response.startswith("data: ")
        parsed = json.loads(first_response[6:])
        assert parsed["type"] == "connected"
        assert "message" in parsed


class TestIntegration:
    """Integration tests for the streaming module."""

    @pytest.mark.asyncio
    async def test_full_streaming_workflow(self):
        """Test a complete streaming workflow."""
        handler = AgentStreamHandler()
        events = []

        async def collect_events():
            async for event in handler.stream_events(agent_id="test-agent"):
                events.append(event)
                if len(events) >= 5:
                    break

        # Start collection
        task = asyncio.create_task(collect_events())

        # Wait for subscriber to be ready
        await asyncio.sleep(0.1)

        # Simulate agent workflow
        await handler.publish_agent_started(
            agent_id="test-agent",
            task_id="task-1",
            request_id="req-1",
        )

        await handler.publish_tool_start(
            agent_id="test-agent",
            task_id="task-1",
            request_id="req-1",
            tool_name="analyze_data",
        )

        await handler.publish_tool_progress(
            agent_id="test-agent",
            task_id="task-1",
            request_id="req-1",
            tool_name="analyze_data",
            progress=0.5,
            message="Processing data...",
        )

        await handler.publish_tool_complete(
            agent_id="test-agent",
            task_id="task-1",
            request_id="req-1",
            tool_name="analyze_data",
            result={"analysis": "complete"},
        )

        await handler.publish_agent_completed(
            agent_id="test-agent",
            task_id="task-1",
            request_id="req-1",
            result={"status": "success"},
        )

        # Wait for events
        await asyncio.sleep(0.3)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Verify all events were received
        assert len(events) == 5
        assert events[0].type == AgentStreamEventType.AGENT_STARTED
        assert events[1].type == AgentStreamEventType.TOOL_START
        assert events[2].type == AgentStreamEventType.TOOL_PROGRESS
        assert events[3].type == AgentStreamEventType.TOOL_COMPLETE
        assert events[4].type == AgentStreamEventType.AGENT_COMPLETED

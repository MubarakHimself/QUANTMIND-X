"""
MCP SSE (Server-Sent Events) Handler

Provides real-time event streaming for MCP server updates:
- Connection status changes
- Tool discovery events
- Tool execution events
- Error notifications
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class SSEEventType(str, Enum):
    """Types of SSE events for MCP."""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    ERROR = "error"
    TOOL_DISCOVERED = "tool_discovered"
    TOOL_CALLED = "tool_called"
    TOOL_RESULT = "tool_result"
    STATUS_CHANGE = "status_change"
    HEARTBEAT = "heartbeat"


@dataclass
class SSEEvent:
    """Represents an SSE event."""
    type: SSEEventType
    data: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    server_id: Optional[str] = None
    tool_name: Optional[str] = None


class SSEEventBus:
    """
    Simple in-memory event bus for SSE events.

    Allows publishing and subscribing to MCP events.
    """

    def __init__(self):
        self._subscribers: List[asyncio.Queue] = []
        self._lock = asyncio.Lock()

    async def subscribe(self) -> AsyncGenerator[SSEEvent, None]:
        """
        Subscribe to events.

        Yields:
            SSEEvent objects as they are published
        """
        queue: asyncio.Queue = asyncio.Queue()
        async with self._lock:
            self._subscribers.append(queue)

        try:
            while True:
                event = await queue.get()
                if event is None:  # Shutdown signal
                    break
                yield event
        finally:
            async with self._lock:
                if queue in self._subscribers:
                    self._subscribers.remove(queue)

    async def publish(self, event: SSEEvent) -> None:
        """Publish an event to all subscribers."""
        async with self._lock:
            subscribers = self._subscribers.copy()

        for queue in subscribers:
            try:
                await queue.put(event)
            except Exception as e:
                logger.warning(f"Failed to publish event to subscriber: {e}")


@dataclass
class MCPSSEHandler:
    """
    Handles SSE (Server-Sent Events) for MCP server updates.

    Provides real-time streaming of MCP server events including:
    - Server connection/disconnection
    - Tool discovery
    - Tool execution results
    - Error notifications

    Usage:
        handler = MCPSSEHandler()

        # Stream events
        async for event in handler.stream_events():
            print(f"Event: {event.type}, Data: {event.data}")

        # Or with server filtering
        async for event in handler.stream_server_events("filesystem"):
            print(f"Server update: {event.data}")
    """

    def __init__(self):
        self._event_bus = SSEEventBus()
        self._active_connections: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self._heartbeat_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the SSE handler (starts heartbeat)."""
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def stop(self) -> None:
        """Stop the SSE handler."""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeat events."""
        while True:
            try:
                await asyncio.sleep(30)
                await self._event_bus.publish(SSEEvent(
                    type=SSEEventType.HEARTBEAT,
                    data={"active_connections": len(self._active_connections)},
                ))
            except asyncio.CancelledError:
                break

    async def stream_events(
        self,
        event_types: Optional[List[SSEEventType]] = None,
    ) -> AsyncGenerator[SSEEvent, None]:
        """
        Stream all MCP events.

        Args:
            event_types: Optional filter for specific event types

        Yields:
            SSEEvent objects
        """
        async for event in self._event_bus.subscribe():
            if event_types is None or event.type in event_types:
                yield event

    async def stream_server_events(
        self,
        server_id: str,
    ) -> AsyncGenerator[SSEEvent, None]:
        """
        Stream events for a specific server.

        Args:
            server_id: Filter events to this server

        Yields:
            SSEEvent objects for the specified server
        """
        async for event in self._event_bus.subscribe():
            if event.server_id == server_id:
                yield event

    async def stream_tool_events(
        self,
        tool_name: Optional[str] = None,
    ) -> AsyncGenerator[SSEEvent, None]:
        """
        Stream tool-related events.

        Args:
            tool_name: Optional filter for specific tool

        Yields:
            SSEEvent objects related to tools
        """
        tool_events = [
            SSEEventType.TOOL_DISCOVERED,
            SSEEventType.TOOL_CALLED,
            SSEEventType.TOOL_RESULT,
        ]

        async for event in self._event_bus.subscribe():
            if event.type in tool_events:
                if tool_name is None or event.tool_name == tool_name:
                    yield event

    async def publish_connection_event(
        self,
        server_id: str,
        status: str,
        error: Optional[str] = None,
    ) -> None:
        """
        Publish a connection status event.

        Args:
            server_id: Server identifier
            status: Connection status (connected, disconnected, connecting, error)
            error: Optional error message
        """
        event_type = {
            "connected": SSEEventType.CONNECTED,
            "disconnected": SSEEventType.DISCONNECTED,
            "connecting": SSEEventType.CONNECTING,
            "error": SSEEventType.ERROR,
        }.get(status.lower(), SSEEventType.STATUS_CHANGE)

        async with self._lock:
            if status == "connected":
                self._active_connections[server_id] = {
                    "connected_at": datetime.now(timezone.utc).isoformat(),
                }
            elif status == "disconnected" and server_id in self._active_connections:
                del self._active_connections[server_id]

        await self._event_bus.publish(SSEEvent(
            type=event_type,
            server_id=server_id,
            data={
                "server_id": server_id,
                "status": status,
                "error": error,
            },
        ))

    async def publish_tool_discovered(
        self,
        server_id: str,
        tool_name: str,
        description: str,
        input_schema: Dict[str, Any],
    ) -> None:
        """
        Publish a tool discovery event.

        Args:
            server_id: Server identifier
            tool_name: Name of discovered tool
            description: Tool description
            input_schema: Tool input schema
        """
        await self._event_bus.publish(SSEEvent(
            type=SSEEventType.TOOL_DISCOVERED,
            server_id=server_id,
            tool_name=tool_name,
            data={
                "server_id": server_id,
                "tool_name": tool_name,
                "description": description,
                "input_schema": input_schema,
            },
        ))

    async def publish_tool_called(
        self,
        server_id: str,
        tool_name: str,
        call_id: str,
        arguments: Dict[str, Any],
    ) -> None:
        """
        Publish a tool call event.

        Args:
            server_id: Server identifier
            tool_name: Name of called tool
            call_id: Unique identifier for this call
            arguments: Tool arguments
        """
        await self._event_bus.publish(SSEEvent(
            type=SSEEventType.TOOL_CALLED,
            server_id=server_id,
            tool_name=tool_name,
            data={
                "server_id": server_id,
                "tool_name": tool_name,
                "call_id": call_id,
                "arguments": arguments,
            },
        ))

    async def publish_tool_result(
        self,
        server_id: str,
        tool_name: str,
        call_id: str,
        result: Any,
        error: Optional[str] = None,
    ) -> None:
        """
        Publish a tool result event.

        Args:
            server_id: Server identifier
            tool_name: Name of tool that was called
            call_id: Unique identifier for this call
            result: Tool result
            error: Optional error message
        """
        await self._event_bus.publish(SSEEvent(
            type=SSEEventType.TOOL_RESULT,
            server_id=server_id,
            tool_name=tool_name,
            data={
                "server_id": server_id,
                "tool_name": tool_name,
                "call_id": call_id,
                "result": result,
                "error": error,
            },
        ))

    async def get_active_connections(self) -> Dict[str, Dict[str, Any]]:
        """Get current active connections."""
        async with self._lock:
            return self._active_connections.copy()

    def format_sse(self, event: SSEEvent) -> str:
        """
        Format an event as SSE format.

        Args:
            event: Event to format

        Returns:
            SSE-formatted string
        """
        data = json.dumps({
            "type": event.type.value,
            "timestamp": event.timestamp,
            "server_id": event.server_id,
            "tool_name": event.tool_name,
            **event.data,
        })

        return f"data: {data}\n\n"


# Global SSE handler instance
_sse_handler: Optional[MCPSSEHandler] = None


def get_sse_handler() -> MCPSSEHandler:
    """Get the global SSE handler instance."""
    global _sse_handler
    if _sse_handler is None:
        _sse_handler = MCPSSEHandler()
    return _sse_handler

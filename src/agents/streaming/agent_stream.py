"""
Agent SSE Streaming Module

Provides real-time streaming for agent responses, tool outputs, and progress updates
using Server-Sent Events (SSE).

Reference: https://platform.claude.com/docs/en/agent-sdk/streaming-output
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional

logger = logging.getLogger(__name__)


class AgentStreamEventType(str, Enum):
    """Types of SSE events for agent streaming."""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    AGENT_STARTED = "agent_started"
    AGENT_COMPLETED = "agent_completed"
    AGENT_FAILED = "agent_failed"
    RESPONSE_START = "response_start"
    RESPONSE_CHUNK = "response_chunk"
    RESPONSE_COMPLETE = "response_complete"
    TOOL_START = "tool_start"
    TOOL_PROGRESS = "tool_progress"
    TOOL_COMPLETE = "tool_complete"
    TOOL_ERROR = "tool_error"
    PROGRESS_UPDATE = "progress_update"
    PROGRESS_COMPLETE = "progress_complete"
    HEARTBEAT = "heartbeat"


@dataclass
class AgentStreamEvent:
    """Represents an SSE event for agent streaming."""
    type: AgentStreamEventType
    data: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    agent_id: Optional[str] = None
    task_id: Optional[str] = None
    tool_name: Optional[str] = None
    request_id: Optional[str] = None


class AgentStreamBus:
    """In-memory event bus for agent SSE events."""

    def __init__(self, max_subscribers: int = 100):
        self._subscribers: List[asyncio.Queue] = []
        self._lock = asyncio.Lock()
        self._max_subscribers = max_subscribers

    async def subscribe(self) -> AsyncGenerator[AgentStreamEvent, None]:
        """Subscribe to events, yielding AgentStreamEvent objects."""
        if len(self._subscribers) >= self._max_subscribers:
            raise RuntimeError("Maximum subscribers reached")
        queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        async with self._lock:
            self._subscribers.append(queue)
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield event
        finally:
            async with self._lock:
                if queue in self._subscribers:
                    self._subscribers.remove(queue)

    async def publish(self, event: AgentStreamEvent) -> None:
        """Publish an event to all subscribers."""
        async with self._lock:
            subscribers = self._subscribers.copy()
        for queue in subscribers:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning("Subscriber queue full, dropping event")

    async def get_subscriber_count(self) -> int:
        async with self._lock:
            return len(self._subscribers)


class AgentStreamHandler:
    """Handles SSE for agent streaming - responses, tools, progress."""

    def __init__(self):
        self._event_bus = AgentStreamBus()
        self._active_streams: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self._heartbeat_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        if self._heartbeat_task is None:
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def stop(self) -> None:
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

    async def _heartbeat_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(30)
                await self._event_bus.publish(AgentStreamEvent(
                    type=AgentStreamEventType.HEARTBEAT,
                    data={"active_streams": len(self._active_streams)},
                ))
            except asyncio.CancelledError:
                break

    async def stream_events(self, agent_id: Optional[str] = None,
                           task_id: Optional[str] = None,
                           event_types: Optional[List[AgentStreamEventType]] = None
                           ) -> AsyncGenerator[AgentStreamEvent, None]:
        """Stream agent events with optional filtering."""
        async for event in self._event_bus.subscribe():
            if agent_id and event.agent_id != agent_id:
                continue
            if task_id and event.task_id != task_id:
                continue
            if event_types and event.type not in event_types:
                continue
            yield event

    async def stream_sse(self, agent_id: Optional[str] = None,
                          task_id: Optional[str] = None,
                          event_types: Optional[List[AgentStreamEventType]] = None
                          ) -> AsyncGenerator[str, None]:
        """Stream events as SSE-formatted strings."""
        async for event in self.stream_events(agent_id, task_id, event_types):
            yield self.format_sse(event)

    def format_sse(self, event: AgentStreamEvent) -> str:
        """Format an event as SSE format."""
        data = {"type": event.type.value, "timestamp": event.timestamp,
                "agent_id": event.agent_id, "task_id": event.task_id,
                "tool_name": event.tool_name, "request_id": event.request_id, **event.data}
        return f"data: {json.dumps(data)}\n\n"

    async def _publish(self, event: AgentStreamEvent) -> None:
        await self._event_bus.publish(event)

    async def publish_agent_started(self, agent_id: str, task_id: str, request_id: str,
                                    metadata: Optional[Dict[str, Any]] = None) -> None:
        await self._publish(AgentStreamEvent(AgentStreamEventType.AGENT_STARTED, metadata or {},
                                              agent_id=agent_id, task_id=task_id, request_id=request_id))

    async def publish_agent_completed(self, agent_id: str, task_id: str, request_id: str,
                                      result: Optional[Dict[str, Any]] = None) -> None:
        await self._publish(AgentStreamEvent(AgentStreamEventType.AGENT_COMPLETED, result or {},
                                              agent_id=agent_id, task_id=task_id, request_id=request_id))

    async def publish_agent_failed(self, agent_id: str, task_id: str, request_id: str,
                                  error: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        await self._publish(AgentStreamEvent(AgentStreamEventType.AGENT_FAILED,
                                              {"error": error, **(metadata or {})},
                                              agent_id=agent_id, task_id=task_id, request_id=request_id))

    async def publish_response_start(self, agent_id: str, task_id: str, request_id: str,
                                     content_type: str = "text") -> None:
        await self._publish(AgentStreamEvent(AgentStreamEventType.RESPONSE_START,
                                              {"content_type": content_type},
                                              agent_id=agent_id, task_id=task_id, request_id=request_id))

    async def publish_response_chunk(self, agent_id: str, task_id: str, request_id: str,
                                     chunk: str, delta: Optional[str] = None) -> None:
        await self._publish(AgentStreamEvent(AgentStreamEventType.RESPONSE_CHUNK,
                                              {"chunk": chunk, "delta": delta, "index": 0},
                                              agent_id=agent_id, task_id=task_id, request_id=request_id))

    async def publish_response_complete(self, agent_id: str, task_id: str, request_id: str,
                                        full_response: str,
                                        metadata: Optional[Dict[str, Any]] = None) -> None:
        await self._publish(AgentStreamEvent(AgentStreamEventType.RESPONSE_COMPLETE,
                                              {"full_response": full_response, **(metadata or {})},
                                              agent_id=agent_id, task_id=task_id, request_id=request_id))

    async def publish_tool_start(self, agent_id: str, task_id: str, request_id: str,
                                 tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> None:
        await self._publish(AgentStreamEvent(AgentStreamEventType.TOOL_START,
                                              {"tool_name": tool_name, "arguments": arguments or {}},
                                              agent_id=agent_id, task_id=task_id, request_id=request_id,
                                              tool_name=tool_name))

    async def publish_tool_progress(self, agent_id: str, task_id: str, request_id: str,
                                    tool_name: str, progress: float,
                                    message: Optional[str] = None) -> None:
        await self._publish(AgentStreamEvent(AgentStreamEventType.TOOL_PROGRESS,
                                              {"tool_name": tool_name, "progress": progress, "message": message},
                                              agent_id=agent_id, task_id=task_id, request_id=request_id,
                                              tool_name=tool_name))

    async def publish_tool_complete(self, agent_id: str, task_id: str, request_id: str,
                                    tool_name: str, result: Any,
                                    metadata: Optional[Dict[str, Any]] = None) -> None:
        await self._publish(AgentStreamEvent(AgentStreamEventType.TOOL_COMPLETE,
                                              {"tool_name": tool_name, "result": result, **(metadata or {})},
                                              agent_id=agent_id, task_id=task_id, request_id=request_id,
                                              tool_name=tool_name))

    async def publish_tool_error(self, agent_id: str, task_id: str, request_id: str,
                                 tool_name: str, error: str) -> None:
        await self._publish(AgentStreamEvent(AgentStreamEventType.TOOL_ERROR,
                                              {"tool_name": tool_name, "error": error},
                                              agent_id=agent_id, task_id=task_id, request_id=request_id,
                                              tool_name=tool_name))

    async def publish_progress_update(self, agent_id: str, task_id: str, request_id: str,
                                      progress: float, message: str,
                                      step: Optional[int] = None,
                                      total_steps: Optional[int] = None) -> None:
        await self._publish(AgentStreamEvent(AgentStreamEventType.PROGRESS_UPDATE,
                                              {"progress": progress, "message": message,
                                               "step": step, "total_steps": total_steps},
                                              agent_id=agent_id, task_id=task_id, request_id=request_id))

    async def publish_progress_complete(self, agent_id: str, task_id: str, request_id: str,
                                        message: str = "Task completed") -> None:
        await self._publish(AgentStreamEvent(AgentStreamEventType.PROGRESS_COMPLETE,
                                              {"message": message, "progress": 1.0},
                                              agent_id=agent_id, task_id=task_id, request_id=request_id))

    async def get_active_streams(self) -> Dict[str, Dict[str, Any]]:
        async with self._lock:
            return self._active_streams.copy()


# FastAPI Integration
async def create_sse_response(handler: AgentStreamHandler, agent_id: Optional[str] = None,
                               task_id: Optional[str] = None,
                               event_types: Optional[List[AgentStreamEventType]] = None
                               ) -> AsyncGenerator[bytes, None]:
    """Create an SSE response for FastAPI.

    Yields a connection event first, then streams all matching events.
    """
    # Send initial connection event
    from datetime import datetime, timezone
    connected_event = AgentStreamEvent(
        type=AgentStreamEventType.CONNECTED,
        data={"message": "Connected to agent stream"},
        timestamp=datetime.now(timezone.utc).isoformat()
    )
    yield handler.format_sse(connected_event).encode("utf-8")

    # Stream all events
    async for event in handler.stream_sse(agent_id, task_id, event_types):
        yield event.encode("utf-8")


# Global Handler Instance
_stream_handler: Optional[AgentStreamHandler] = None


def get_stream_handler() -> AgentStreamHandler:
    """Get the global stream handler instance."""
    global _stream_handler
    if _stream_handler is None:
        _stream_handler = AgentStreamHandler()
    return _stream_handler


async def init_stream_handler() -> AgentStreamHandler:
    """Initialize and start the global stream handler."""
    handler = get_stream_handler()
    await handler.start()
    return handler


async def close_stream_handler() -> None:
    """Stop the global stream handler."""
    global _stream_handler
    if _stream_handler is not None:
        await _stream_handler.stop()
        _stream_handler = None

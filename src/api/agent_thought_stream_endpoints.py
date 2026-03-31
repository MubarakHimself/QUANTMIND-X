"""
Agent Thought Streaming

SSE endpoint that streams agent reasoning thoughts to the frontend in real time.
Agents publish thoughts to a shared in-memory queue; SSE clients subscribe per session.
"""
import asyncio
import json
import uuid
import logging
from collections import deque
from typing import Dict, AsyncGenerator, Optional
from datetime import datetime
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/agent-thoughts", tags=["agent-thoughts"])

# In-memory pub/sub: session_id -> asyncio.Queue
_thought_queues: Dict[str, asyncio.Queue] = {}

# Ring buffer for recent thoughts (last 200)
_recent_thoughts: deque = deque(maxlen=200)


class ThoughtPublisher:
    """Singleton publisher. Agents call publish() to emit thoughts."""

    def publish(
        self,
        department: str,
        thought: str,
        thought_type: str = "reasoning",
        session_id: Optional[str] = None,
    ):
        """Publish a thought to all connected clients (or specific session).

        Args:
            department: Publishing department name.
            thought: The thought content string.
            thought_type: One of "reasoning" | "tool_call" | "decision" |
                          "memory_read" | "dispatch".
            session_id: If provided, publish only to that session's queue;
                        otherwise broadcast to all connected sessions.
        """
        event = {
            "id": str(uuid.uuid4()),
            "type": "thought",
            "department": department,
            "content": thought,
            "thought_type": thought_type,
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": session_id or "",
        }

        # Store in ring buffer regardless of connected clients
        _recent_thoughts.append(event)

        targets = (
            [session_id]
            if session_id and session_id in _thought_queues
            else list(_thought_queues.keys())
        )
        for sid in targets:
            try:
                _thought_queues[sid].put_nowait(event)
            except asyncio.QueueFull:
                pass  # Drop if queue full — client is too slow


_publisher = ThoughtPublisher()


def get_thought_publisher() -> ThoughtPublisher:
    """Return the singleton ThoughtPublisher instance."""
    return _publisher


@router.get("/stream")
async def stream_thoughts(request: Request, session_id: Optional[str] = None):
    """SSE endpoint — streams agent thoughts as they happen.

    Query Parameters:
        session_id: Optional. If provided, reuses an existing queue for that
                    session; otherwise a new UUID session is created.

    Returns:
        text/event-stream with JSON-encoded thought events.
    """
    sid = session_id or str(uuid.uuid4())
    _thought_queues[sid] = asyncio.Queue(maxsize=100)

    async def event_generator() -> AsyncGenerator[str, None]:
        # Send initial connection event so the client knows its session_id
        yield f"data: {json.dumps({'type': 'connected', 'session_id': sid})}\n\n"
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(
                        _thought_queues[sid].get(), timeout=30
                    )
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    # Heartbeat keeps the connection alive through proxies
                    yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
        finally:
            _thought_queues.pop(sid, None)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/history")
async def get_thought_history(limit: int = 50):
    """Returns recent published thoughts from the in-memory ring buffer (last 200).

    Query Parameters:
        limit: Number of recent thoughts to return (default 50, max 200).
    """
    return list(_recent_thoughts)[-limit:]

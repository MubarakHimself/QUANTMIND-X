"""
Copilot API Endpoints

Provides REST and SSE streaming endpoints for the QuantMind Copilot.
The Copilot is the Workshop assistant agent — distinct from the Floor Manager.

Endpoints:
- POST /api/copilot/chat - Chat (JSON) or stream (SSE when stream=true)
- GET /api/copilot/health - Health check
- WS  /api/copilot/ws/copilot - WebSocket chat
"""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import logging
import json

from src.api.services.workshop_copilot_service import (
    get_workshop_copilot_service,
    WorkshopCopilotService,
    WorkshopCopilotRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/copilot", tags=["copilot"])


# ========== Request/Response Models ==========


class ChatRequest(BaseModel):
    """Request to chat with the copilot."""
    message: str = Field(..., description="User message")
    history: Optional[List[Dict[str, Any]]] = Field(default=None, description="Conversation history")
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Canvas context (canvas_context, session_id, etc.)")
    stream: bool = Field(default=False, description="Enable SSE token streaming")


class ChatResponse(BaseModel):
    """Response from non-streaming chat endpoint."""
    reply: str
    action_taken: Optional[str] = None
    delegation: Optional[str] = None


class HealthResponse(BaseModel):
    """Response from health check."""
    status: str
    service: str = "copilot"


# ========== Helper ==========


def get_service() -> WorkshopCopilotService:
    return get_workshop_copilot_service()


# ========== API Endpoints ==========


@router.post("/chat")
async def chat(request: ChatRequest):
    """
    Chat with the Copilot.

    When stream=true returns SSE text/event-stream with events:
      {"type": "tool",      "tool": "thinking", "status": "started"}
      {"type": "thinking",  "content": "..."}
      {"type": "content",   "delta": "..."}
      {"type": "tool",      "tool": "thinking", "status": "completed"}
      {"type": "done"}

    When stream=false returns JSON ChatResponse.
    """
    service = get_service()

    if request.stream:
        async def event_generator():
            try:
                async for event in service.handle_message_stream(
                    message=request.message,
                    history=request.history,
                    context=request.context,
                    session_id=request.session_id,
                ):
                    yield f"data: {json.dumps(event)}\n\n"
            except Exception as e:
                logger.error(f"Copilot streaming failed: {e}")
                yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Non-streaming path
    try:
        copilot_request = WorkshopCopilotRequest(
            message=request.message,
            history=request.history,
            session_id=request.session_id,
        )
        result = await service.handle_message(copilot_request)
        return ChatResponse(
            reply=result.reply,
            action_taken=result.action_taken,
            delegation=result.delegation.get("type") if result.delegation else None,
        )
    except Exception as e:
        logger.error(f"Copilot chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="healthy", service="copilot")


# ========== WebSocket Endpoint ==========


@router.websocket("/ws/copilot")
async def copilot_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time chat with the Copilot.

    Send:  {"type": "chat", "message": "...", "history": [], "session_id": "..."}
    Recv:  {"type": "started"} | {"type": "content", "delta": "..."} |
           {"type": "thinking", "content": "..."} | {"type": "done"} |
           {"type": "error", "error": "..."}
    """
    await websocket.accept()
    service = get_service()

    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "error": "Invalid JSON"})
                continue

            msg_type = message.get("type")

            if msg_type == "chat":
                user_message = message.get("message", "")
                history = message.get("history")
                session_id = message.get("session_id")
                context = message.get("context")

                if not user_message:
                    await websocket.send_json({"type": "error", "error": "Message is required"})
                    continue

                await websocket.send_json({"type": "started"})

                try:
                    async for event in service.handle_message_stream(
                        message=user_message,
                        history=history,
                        context=context,
                        session_id=session_id,
                    ):
                        await websocket.send_json(event)
                except Exception as e:
                    logger.error(f"WS copilot stream failed: {e}")
                    await websocket.send_json({"type": "error", "error": str(e)})

            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})
            else:
                await websocket.send_json({"type": "error", "error": f"Unknown type: {msg_type}"})

    except WebSocketDisconnect:
        logger.info("Copilot WebSocket disconnected")
    except Exception as e:
        logger.error(f"Copilot WebSocket error: {e}")

"""
Workshop Copilot API Endpoints

Provides REST and WebSocket endpoints for the Workshop Copilot.
Handles message routing with intent classification.

Endpoints:
- POST /api/workshop/copilot/chat - Chat with workshop copilot
- GET /api/workshop/copilot/health - Health check
- WebSocket /api/workshop/copilot/ws - Real-time chat
"""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
import logging
import json

from src.api.services.workshop_copilot_service import (
    get_workshop_copilot_service,
    WorkshopCopilotService,
    WorkshopCopilotRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/workshop/copilot", tags=["workshop-copilot"])


# ========== Request/Response Models ==========


class ChatRequest(BaseModel):
    """Request to chat with the workshop copilot."""
    message: str = Field(..., description="User message")
    history: Optional[List[Dict[str, Any]]] = Field(default=None, description="Conversation history")
    session_id: Optional[str] = Field(default=None, description="Session identifier")


class ChatResponse(BaseModel):
    """Response from chat endpoint."""
    reply: str
    action_taken: Optional[str] = None
    delegation: Optional[str] = None


class HealthResponse(BaseModel):
    """Response from health check."""
    status: str
    service: str = "workshop-copilot"


# ========== Helper Functions ==========


def get_service() -> WorkshopCopilotService:
    """Get the Workshop Copilot service instance."""
    return get_workshop_copilot_service()


# ========== API Endpoints ==========


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat with the Workshop Copilot.

    The Workshop Copilot can:
    - Handle simple queries with direct responses
    - Route trading requests to the Floor Manager
    - Process workflow requests for department workflows

    Message is classified by intent:
    - simple: Direct responses
    - trading: Delegates to Floor Manager
    - workflow: Starts department workflow
    """
    service = get_service()

    try:
        # Create service request
        copilot_request = WorkshopCopilotRequest(
            message=request.message,
            history=request.history,
            session_id=request.session_id,
        )

        # Handle message
        result = await service.handle_message(copilot_request)

        return ChatResponse(
            reply=result.reply,
            action_taken=result.action_taken,
            delegation=result.delegation,
        )
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint for Workshop Copilot.

    Returns the service status.
    """
    return HealthResponse(
        status="healthy",
        service="workshop-copilot",
    )


# ========== WebSocket Endpoint for Streaming Chat ==========


@router.websocket("/ws")
async def workshop_copilot_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time chat with the Workshop Copilot.

    Message format (send):
    {
        "type": "chat",
        "message": "Your message here",
        "history": [],  // optional
        "session_id": "session-123"  // optional
    }

    Message format (receive):
    {
        "type": "started" | "text" | "completed" | "error",
        "delta": "...",  // for text events
        "reply": "...",  // for completed events
        "action_taken": "...",  // for completed events
        "delegation": "...",  // for completed events
        "error": "..."  // for error events
    }
    """
    await websocket.accept()
    service = get_service()

    try:
        while True:
            # Receive message
            data = await websocket.receive_text()

            try:
                message = json.loads(data)
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "error": "Invalid JSON message",
                })
                continue

            msg_type = message.get("type")

            if msg_type == "chat":
                # Handle chat message
                user_message = message.get("message", "")
                history = message.get("history")
                session_id = message.get("session_id")

                if not user_message:
                    await websocket.send_json({
                        "type": "error",
                        "error": "Message is required",
                    })
                    continue

                # Send started event
                await websocket.send_json({
                    "type": "started",
                })

                # Process message
                try:
                    request = WorkshopCopilotRequest(
                        message=user_message,
                        history=history,
                        session_id=session_id,
                    )
                    result = await service.handle_message(request)

                    # Send completed event with full response
                    await websocket.send_json({
                        "type": "completed",
                        "reply": result.reply,
                        "action_taken": result.action_taken,
                        "delegation": result.delegation,
                    })
                except Exception as e:
                    logger.error(f"WebSocket chat failed: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "error": str(e),
                    })

            elif msg_type == "ping":
                # Ping/pong for connection keepalive
                await websocket.send_json({
                    "type": "pong",
                })

            else:
                await websocket.send_json({
                    "type": "error",
                    "error": f"Unknown message type: {msg_type}",
                })

    except WebSocketDisconnect:
        logger.info("Workshop Copilot WebSocket disconnected")
    except Exception as e:
        logger.error(f"Workshop Copilot WebSocket error: {e}")

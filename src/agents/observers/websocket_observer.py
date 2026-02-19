"""
WebSocket Observer for Agent Factory

Provides WebSocket event emission for agent lifecycle and invocation events.

**Validates: Phase 5.2 - WebSocket Observer**
"""

import logging
import json
from typing import Any, Dict, Optional, List
from datetime import datetime

from src.agents.observer import AgentObserver
from src.agents.config import AgentConfig

logger = logging.getLogger(__name__)


# Global WebSocket manager reference
_websocket_manager = None


def set_websocket_manager(manager: Any) -> None:
    """
    Set the global WebSocket manager for event broadcasting.
    
    Args:
        manager: WebSocket manager with broadcast method
    """
    global _websocket_manager
    _websocket_manager = manager


def get_websocket_manager() -> Optional[Any]:
    """Get the global WebSocket manager."""
    return _websocket_manager


class WebSocketObserver(AgentObserver):
    """
    WebSocket observer for emitting agent events.
    
    Broadcasts agent lifecycle and invocation events to connected clients.
    """
    
    def __init__(self, event_prefix: str = "agent"):
        """
        Initialize the WebSocket observer.
        
        Args:
            event_prefix: Prefix for event types
        """
        self.event_prefix = event_prefix
        logger.info("WebSocketObserver initialized")
    
    def _broadcast(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Broadcast an event to connected clients.
        
        Args:
            event_type: Type of event
            data: Event data
        """
        global _websocket_manager
        
        if _websocket_manager is None:
            logger.debug(f"WebSocket manager not set, skipping broadcast: {event_type}")
            return
        
        try:
            message = {
                "type": f"{self.event_prefix}_{event_type}",
                "timestamp": datetime.utcnow().isoformat(),
                "data": data,
            }
            
            # Broadcast to all connected clients
            _websocket_manager.broadcast(json.dumps(message))
            
            logger.debug(f"Broadcasted event: {event_type}")
            
        except Exception as e:
            logger.error(f"Error broadcasting event: {e}")
    
    def on_agent_created(self, agent_id: str, config: AgentConfig) -> None:
        """Emit agent created event."""
        self._broadcast("created", {
            "agent_id": agent_id,
            "agent_type": config.agent_type,
            "name": config.name,
        })
    
    def on_agent_destroyed(self, agent_id: str) -> None:
        """Emit agent destroyed event."""
        self._broadcast("destroyed", {
            "agent_id": agent_id,
        })
    
    def on_invocation_start(
        self,
        agent_id: str,
        invocation_id: str
    ) -> None:
        """Emit invocation start event."""
        self._broadcast("invocation_start", {
            "agent_id": agent_id,
            "invocation_id": invocation_id,
        })
    
    def on_invocation_complete(
        self,
        agent_id: str,
        invocation_id: str,
        duration: float
    ) -> None:
        """Emit invocation complete event."""
        self._broadcast("invocation_complete", {
            "agent_id": agent_id,
            "invocation_id": invocation_id,
            "duration": duration,
        })
    
    def on_invocation_error(
        self,
        agent_id: str,
        invocation_id: str,
        error: Exception
    ) -> None:
        """Emit invocation error event."""
        self._broadcast("invocation_error", {
            "agent_id": agent_id,
            "invocation_id": invocation_id,
            "error_type": type(error).__name__,
            "error_message": str(error),
        })
    
    def on_tool_call(
        self,
        agent_id: str,
        tool_name: str,
        args: Dict[str, Any]
    ) -> None:
        """Emit tool call event."""
        self._broadcast("tool_call", {
            "agent_id": agent_id,
            "tool_name": tool_name,
            "args": args,
        })
    
    def on_agent_invoke(
        self,
        agent_id: str,
        state: Dict[str, Any]
    ) -> None:
        """Emit agent invoke event."""
        self._broadcast("invoke", {
            "agent_id": agent_id,
            "message_count": len(state.get("messages", [])),
        })
    
    def on_agent_response(
        self,
        agent_id: str,
        response: Any
    ) -> None:
        """Emit agent response event."""
        self._broadcast("response", {
            "agent_id": agent_id,
            "response_type": type(response).__name__,
            "has_tool_calls": hasattr(response, 'tool_calls') and bool(response.tool_calls),
        })
    
    def on_agent_error(
        self,
        agent_id: str,
        error: Exception
    ) -> None:
        """Emit agent error event."""
        self._broadcast("error", {
            "agent_id": agent_id,
            "error_type": type(error).__name__,
            "error_message": str(error),
        })

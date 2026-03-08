"""
Workshop Copilot Service

Handles message routing for the Workshop Copilot with intent classification.
Routes messages to appropriate handlers based on detected intent:
- Simple queries: Direct responses
- Trading requests: Delegate to floor manager
- Workflow requests: Start department workflow
"""

import logging
from typing import List, Dict, Any, Optional

import httpx

logger = logging.getLogger(__name__)

# =============================================================================
# Intent Classification Keywords
# =============================================================================

TRADING_KEYWORDS = [
    "deploy", "trade", "buy", "sell", "bot", "strategy",
    "backtest", "paper trade", "execute", "order"
]

WORKFLOW_KEYWORDS = [
    "create", "build", "generate", "develop",
    "analyze video", "from video", "youtube"
]


# =============================================================================
# Request/Response Models
# =============================================================================

class WorkshopCopilotRequest:
    """Request model for Workshop Copilot."""

    def __init__(
        self,
        message: str,
        history: Optional[List[Dict[str, Any]]] = None,
        session_id: Optional[str] = None
    ):
        self.message = message
        self.history = history or []
        self.session_id = session_id


class WorkshopCopilotResponse:
    """Response model for Workshop Copilot."""

    def __init__(
        self,
        reply: str,
        delegation: Optional[Dict[str, Any]] = None,
        action_taken: Optional[str] = None
    ):
        self.reply = reply
        self.delegation = delegation
        self.action_taken = action_taken

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reply": self.reply,
            "delegation": self.delegation,
            "action_taken": self.action_taken
        }


# =============================================================================
# Intent Classification
# =============================================================================

def _classify_intent(message: str) -> str:
    """
    Classify the intent of a message.

    Args:
        message: User message to classify

    Returns:
        Intent type: "simple", "trading", or "workflow"
    """
    msg_lower = message.lower()

    # Check for trading keywords first (higher priority)
    if any(keyword in msg_lower for keyword in TRADING_KEYWORDS):
        return "trading"

    # Check for workflow keywords
    if any(keyword in msg_lower for keyword in WORKFLOW_KEYWORDS):
        return "workflow"

    # Default to simple query
    return "simple"


# =============================================================================
# Workshop Copilot Service
# =============================================================================

class WorkshopCopilotService:
    """Service for handling Workshop Copilot messages."""

    def __init__(self, floor_manager_url: str = "http://localhost:8000"):
        self.floor_manager_url = floor_manager_url
        self.logger = logger

    async def handle_message(
        self,
        request: WorkshopCopilotRequest
    ) -> WorkshopCopilotResponse:
        """
        Main entry point for handling workshop copilot messages.

        Args:
            request: WorkshopCopilotRequest with message, history, session_id

        Returns:
            WorkshopCopilotResponse with reply, delegation, action_taken
        """
        # Classify intent
        intent = _classify_intent(request.message)
        self.logger.info(f"Classified intent: {intent} for message: {request.message[:50]}...")

        # Route to appropriate handler
        if intent == "simple":
            return await self._handle_simple_query(request)
        elif intent == "trading":
            return await self._handle_trading_request(request)
        elif intent == "workflow":
            return await self._handle_workflow_request(request)
        else:
            return WorkshopCopilotResponse(
                reply="I didn't understand that request. Try asking about trading, workflows, or general questions.",
                action_taken="unhandled"
            )

    async def _handle_simple_query(
        self,
        request: WorkshopCopilotRequest
    ) -> WorkshopCopilotResponse:
        """
        Handle simple queries with direct responses.

        Args:
            request: WorkshopCopilotRequest

        Returns:
            WorkshopCopilotResponse with direct reply
        """
        msg_lower = request.message.lower()

        # Greeting
        if any(word in msg_lower for word in ["hello", "hi", "hey", "help"]):
            return WorkshopCopilotResponse(
                reply="Hello! I'm the Workshop Copilot. I can help you with:\n"
                      "- Trading: deploy, backtest, paper trade\n"
                      "- Workflows: create, build, generate projects\n"
                      "- General questions\n\nWhat would you like to do?",
                action_taken="greeting"
            )

        # Status query
        if "status" in msg_lower:
            return WorkshopCopilotResponse(
                reply="Workshop Copilot is running. I can assist with trading and development tasks.",
                action_taken="status_check"
            )

        # Default simple response
        return WorkshopCopilotResponse(
            reply=f"I understand you're asking about: '{request.message}'. "
                 "This appears to be a simple query. How can I help you further?",
            action_taken="simple_query"
        )

    async def _handle_trading_request(
        self,
        request: WorkshopCopilotRequest
    ) -> WorkshopCopilotResponse:
        """
        Handle trading requests - delegate to floor manager.

        Args:
            request: WorkshopCopilotRequest

        Returns:
            WorkshopCopilotResponse delegating to floor manager
        """
        return await self._delegate_to_floor_manager(request.message)

    async def _delegate_to_floor_manager(self, message: str) -> WorkshopCopilotResponse:
        """Delegate a trading task to Floor Manager."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.floor_manager_url}/api/floor-manager/chat",
                    json={"message": message},
                    timeout=30.0
                )
                response.raise_for_status()
                data = response.json()
                return WorkshopCopilotResponse(
                    reply=data.get("reply", "Task delegated to Floor Manager"),
                    action_taken="delegated_to_floor_manager",
                    delegation={
                        "type": "floor_manager",
                        "department": data.get("delegated_department"),
                        "task_id": data.get("task_id")
                    }
                )
        except Exception as e:
            logger.error(f"Failed to delegate to Floor Manager: {e}")
            return WorkshopCopilotResponse(
                reply=f"I understand you want to: {message}. The Floor Manager is currently unavailable.",
                action_taken="error_delegating"
            )

    async def _handle_workflow_request(
        self,
        request: WorkshopCopilotRequest
    ) -> WorkshopCopilotResponse:
        """
        Handle workflow requests - start department workflow.

        Args:
            request: WorkshopCopilotRequest

        Returns:
            WorkshopCopilotResponse with workflow initiation
        """
        # Placeholder for department workflow
        # This will be implemented in future tasks
        self.logger.info(f"Processing workflow request: {request.message}")

        return WorkshopCopilotResponse(
            reply=f"I've received your workflow request: '{request.message}'. "
                 "Starting the appropriate department workflow...",
            delegation={"type": "department_workflow"},
            action_taken="workflow_request"
        )


# =============================================================================
# Singleton
# =============================================================================

_workshop_copilot_service: Optional[WorkshopCopilotService] = None


def get_workshop_copilot_service(floor_manager_url: str = "http://localhost:8000") -> WorkshopCopilotService:
    """Get the Workshop Copilot service singleton."""
    global _workshop_copilot_service
    if _workshop_copilot_service is None:
        _workshop_copilot_service = WorkshopCopilotService(floor_manager_url)
    return _workshop_copilot_service

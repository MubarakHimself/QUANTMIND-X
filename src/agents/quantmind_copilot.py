"""
QuantMind Copilot Agent

The QuantMind Copilot is the main entry point for user interactions.
It handles general chat and delegates trading-related tasks to the Floor Manager.

Model Tier: Sonnet (balanced reasoning)
"""
import logging
from typing import Dict, Any, Optional

from src.agents.departments.floor_manager import FloorManager
from src.agents.departments.department_mail import (
    DepartmentMailService,
    MessageType,
    Priority,
)

logger = logging.getLogger(__name__)


class QuantMindCopilot:
    """
    QuantMind Copilot Agent.

    The Copilot is the primary interface for user chat interactions.
    It determines whether a message requires trading floor processing
    or can be handled directly with a simple response.

    Attributes:
        floor_manager: FloorManager instance for delegating trading tasks
        mail_service: Mail service for cross-component communication
    """

    # Keywords that trigger delegation to Floor Manager
    TRADING_KEYWORDS = [
        "trade",
        "execute",
        "analyze",
        "research",
        "risk",
        "portfolio",
        "dispatch",
        "department",
    ]

    def __init__(
        self,
        mail_db_path: str = ".quantmind/department_mail.db",
    ):
        """
        Initialize the QuantMind Copilot.

        Args:
            mail_db_path: Path to SQLite mail database
        """
        self.floor_manager = FloorManager(mail_db_path=mail_db_path)
        self.mail_service = DepartmentMailService(db_path=mail_db_path)
        logger.info("QuantMindCopilot initialized")

    def chat(
        self,
        message: str,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a chat message.

        Determines whether the message requires trading floor processing
        or can be handled directly.

        Args:
            message: The user's message
            model: Optional model specification (for future use)

        Returns:
            Response dictionary with status and content
        """
        message_lower = message.lower()

        # Check if any trading keyword is present
        if any(keyword in message_lower for keyword in self.TRADING_KEYWORDS):
            return self._delegate_to_floor_manager(message)
        else:
            return self._general_chat(message)

    def _delegate_to_floor_manager(
        self,
        message: str,
    ) -> Dict[str, Any]:
        """
        Delegate a trading-related message to the Floor Manager.

        Sends a DISPATCH message to the floor_manager and processes the task.

        Args:
            message: The trading-related message

        Returns:
            Response with delegation status
        """
        logger.info(f"Delegating to Floor Manager: {message[:50]}...")

        # Send dispatch message to floor_manager
        dispatch_message = self.mail_service.send(
            from_dept="copilot",
            to_dept="floor_manager",
            type=MessageType.DISPATCH,
            subject=f"Task from Copilot: {message[:50]}...",
            body=message,
            priority=Priority.NORMAL,
        )

        # Process the task through Floor Manager
        result = self.floor_manager.process(task=message)

        return {
            "status": "delegated",
            "message_id": dispatch_message.id,
            "floor_manager_result": result,
            "response": f"Task delegated to Floor Manager for processing. Message ID: {dispatch_message.id}",
        }

    def _general_chat(
        self,
        message: str,
    ) -> Dict[str, Any]:
        """
        Handle general chat messages that don't require trading floor processing.

        For now, returns a simple response. Can be extended with LLM integration.

        Args:
            message: The user's general message

        Returns:
            Simple response dictionary
        """
        # Simple response for general chat
        responses = {
            "hello": "Hello! I'm QuantMind Copilot. How can I help you today?",
            "hi": "Hi there! I'm QuantMind Copilot. How can I assist you?",
            "help": "I can help with trading tasks like executing trades, analyzing markets, researching strategies, managing portfolio risk, and more. Just describe what you need!",
        }

        message_lower = message.lower().strip()

        # Check for known greetings
        for key, response in responses.items():
            if key in message_lower:
                return {
                    "status": "success",
                    "response": response,
                }

        # Default general response
        return {
            "status": "success",
            "response": f"I received your message: '{message}'. For trading-related tasks (analyze, research, trade, portfolio, risk), I'll delegate to the Floor Manager. How else can I help?",
        }

    def close(self):
        """Clean up resources."""
        if self.floor_manager:
            self.floor_manager.close()
        if self.mail_service:
            self.mail_service.close()
        logger.info("QuantMindCopilot closed")

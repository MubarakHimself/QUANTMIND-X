# tests/database/test_chat_models.py
import pytest
from src.database.models import ChatSession, ChatMessage


def test_chat_session_model_creation():
    """Test ChatSession model can be created."""
    session = ChatSession(
        id="test-session-123",
        agent_type="workshop",
        agent_id="copilot",
        title="Test Chat",
        user_id="user-1",
        context={},
        metadata={}
    )
    assert session.id == "test-session-123"
    assert session.agent_type == "workshop"


def test_chat_message_model_creation():
    """Test ChatMessage model can be created."""
    message = ChatMessage(
        id="msg-123",
        session_id="test-session-123",
        role="user",
        content="Hello"
    )
    assert message.id == "msg-123"
    assert message.role == "user"

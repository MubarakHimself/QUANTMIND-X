import pytest

@pytest.mark.asyncio
async def test_build_session_context():
    """Test building context from memory and history."""
    from src.api.services.chat_session_service import ChatSessionService

    service = ChatSessionService()

    # Create session with messages
    session = await service.create_session(
        agent_type="workshop",
        agent_id="copilot",
        user_id="user-1",
        title="Test"
    )

    await service.add_message(session.id, "user", "What is RSI?")
    await service.add_message(session.id, "assistant", "RSI is Relative Strength Index.")

    # Build context - should include memory + history
    context = await service.build_session_context(session.id)

    assert "history" in context
    assert "session_id" in context
    assert len(context["history"]) == 2

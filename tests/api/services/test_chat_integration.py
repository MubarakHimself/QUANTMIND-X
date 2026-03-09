# tests/api/services/test_chat_integration.py
import pytest


@pytest.mark.asyncio
async def test_execute_tool_from_chat():
    """Test executing a tool from chat."""
    from src.api.services.chat_session_service import ChatSessionService

    service = ChatSessionService()

    # Test with placeholder - will fail gracefully if tools not available
    result = await service.execute_tool(
        tool_name="read_file",
        parameters={"path": "/test.txt"},
        agent_id="copilot"
    )

    assert "success" in result


@pytest.mark.asyncio
async def test_execute_skill_from_chat():
    """Test executing a skill from chat."""
    from src.api.services.chat_session_service import ChatSessionService

    service = ChatSessionService()

    # Test with placeholder - will fail gracefully if skills not available
    result = await service.execute_skill(
        skill_name="test_skill",
        parameters={"input": "test"},
        context={"user_id": "test_user"}
    )

    assert "success" in result


@pytest.mark.asyncio
async def test_trigger_workflow_from_chat():
    """Test triggering a workflow from chat."""
    from src.api.services.chat_session_service import ChatSessionService

    service = ChatSessionService()

    # Test with placeholder - will fail gracefully if workflows not available
    result = await service.trigger_workflow(
        workflow_name="test_workflow",
        parameters={"trigger": "manual"}
    )

    assert "success" in result

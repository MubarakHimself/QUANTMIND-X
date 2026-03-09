import pytest
from datetime import datetime, timedelta

@pytest.mark.asyncio
async def test_archive_old_sessions():
    """Test archiving sessions older than specified days."""
    from src.api.services.chat_session_service import ChatSessionService
    from src.database.engine import Session
    from src.database.models import ChatSession

    service = ChatSessionService()

    # Create a session
    session = await service.create_session(
        agent_type="workshop",
        agent_id="copilot",
        user_id="user-1",
        title="Old Chat"
    )

    # Run archival with days=0 (archive all)
    archived = await service.archive_old_sessions(days=0)

    assert archived >= 1

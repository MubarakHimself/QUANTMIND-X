import pytest
from sqlalchemy import create_engine

from src.database.models import Base
from src.api.services.chat_session_service import ChatSessionService

# Test database configuration
TEST_DB_PATH = "test_chat_session_service.db"


@pytest.fixture(scope="function")
def test_engine():
    """Create a test database engine."""
    engine = create_engine(f"sqlite:///{TEST_DB_PATH}")
    Base.metadata.create_all(bind=engine)
    yield engine
    # Cleanup
    Base.metadata.drop_all(bind=engine)
    import os
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)


@pytest.fixture
def service(test_engine):
    """Create a ChatSessionService with test database."""
    # Override the engine for testing
    from src.database import engine as db_engine
    from src.database import engine
    original_engine = engine.engine
    engine.engine = test_engine

    # Also patch the Session in the service module
    from src.api.services import chat_session_service
    from sqlalchemy.orm import sessionmaker
    TestSession = sessionmaker(bind=test_engine)

    chat_session_service.Session = TestSession

    yield ChatSessionService()

    # Restore original engine
    engine.engine = original_engine


@pytest.mark.asyncio
async def test_create_session(service):
    """Test creating a new chat session."""
    session = await service.create_session(
        agent_type="workshop",
        agent_id="copilot",
        user_id="user-1",
        title="My Chat"
    )
    assert session.agent_type == "workshop"
    assert session.agent_id == "copilot"


@pytest.mark.asyncio
async def test_add_message(service):
    """Test adding a message to a session."""
    session = await service.create_session(
        agent_type="workshop",
        agent_id="copilot",
        user_id="user-1"
    )
    message = await service.add_message(
        session_id=session.id,
        role="user",
        content="Hello"
    )
    assert message.role == "user"
    assert message.content == "Hello"


@pytest.mark.asyncio
async def test_update_session_title(service):
    """Test updating a chat session title."""
    session = await service.create_session(
        agent_type="floor-manager",
        agent_id="floor-manager",
        user_id="user-1",
        title="Initial title",
    )

    updated = await service.update_session_title(session.id, "Renamed session")
    assert updated is not None
    assert updated.title == "Renamed session"


@pytest.mark.asyncio
async def test_first_user_message_auto_titles_default_session(service):
    session = await service.create_session(
        agent_type="department",
        agent_id="research",
        user_id="user-1",
    )

    await service.add_message(
        session_id=session.id,
        role="user",
        content="Investigate EURUSD breakout behavior during London open and summarize the edge.",
    )

    refreshed = await service.get_session(session.id)
    assert refreshed is not None
    assert refreshed.title.startswith("Investigate EURUSD breakout behavior")


@pytest.mark.asyncio
async def test_list_sessions_can_filter_agent_id_and_exclude_empty(service):
    empty_research = await service.create_session(
        agent_type="department",
        agent_id="research",
        user_id="user-1",
        title="Empty research",
    )
    populated_research = await service.create_session(
        agent_type="department",
        agent_id="research",
        user_id="user-1",
        title="Populated research",
    )
    trading = await service.create_session(
        agent_type="department",
        agent_id="trading",
        user_id="user-1",
        title="Trading session",
    )

    await service.add_message(populated_research.id, "user", "Hello research")
    await service.add_message(trading.id, "user", "Hello trading")

    sessions = await service.list_sessions(
        user_id="user-1",
        agent_type="department",
        agent_id="research",
        exclude_empty=True,
    )

    assert [session.id for session in sessions] == [populated_research.id]
    assert getattr(sessions[0], "message_count", 0) == 1
    assert all(session.id != empty_research.id for session in sessions)

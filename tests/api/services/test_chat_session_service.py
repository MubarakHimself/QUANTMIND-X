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

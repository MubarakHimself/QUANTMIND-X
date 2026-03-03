import pytest
from src.agents.departments.department_mail import MessageType
from src.agents.departments.floor_manager import FloorManager


def test_trd_dispatch_uses_correct_message_type():
    """TRD generation should dispatch with STRATEGY_DISPATCH message type."""
    fm = FloorManager(mail_db_path=":memory:")
    result = fm.dispatch(
        to_dept="research",
        task="Generate TRD from video",
        context={"video_url": "https://youtube.com/watch?v=123"}
    )
    assert result["status"] == "dispatched"
    messages = fm.mail_service.check_inbox("research", unread_only=False)
    assert len(messages) > 0
    assert messages[0].type == MessageType.STRATEGY_DISPATCH
    fm.close()

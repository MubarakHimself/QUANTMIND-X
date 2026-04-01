import asyncio
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock


def test_chat_returns_department_head_result_for_delegated_tasks():
    from src.agents.departments.floor_manager import FloorManager
    from src.agents.departments.types import Department

    class FakeHead:
        config = type("Config", (), {"model": "claude-sonnet-test"})()

        async def process_task(self, task: str, context: dict | None = None):
            return {
                "status": "success",
                "content": f"Processed: {task}",
                "tool_calls": [{"name": "research_summary"}],
            }

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_mail.db")
        manager = FloorManager(mail_db_path=db_path)
        manager.classify_task = lambda _: Department.RESEARCH
        manager._department_heads[Department.RESEARCH] = FakeHead()
        manager._invoke_llm = AsyncMock(return_value="LLM fallback should not run")

        result = asyncio.run(manager.chat("Research EURUSD momentum breakout"))

        assert result["status"] == "success"
        assert result["content"] == "Processed: Research EURUSD momentum breakout"
        assert result["delegation"]["department"] == "research"
        assert result["model"] == "claude-sonnet-test"
        manager._invoke_llm.assert_not_awaited()
        manager.close()


def test_chat_falls_back_to_llm_when_department_head_unavailable():
    from src.agents.departments.floor_manager import FloorManager
    from src.agents.departments.types import Department

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_mail.db")
        manager = FloorManager(mail_db_path=db_path)
        manager.classify_task = lambda _: Department.RESEARCH
        manager._department_heads.pop(Department.RESEARCH, None)
        manager._invoke_llm = AsyncMock(return_value="LLM fallback response")

        result = asyncio.run(manager.chat("Research EURUSD momentum breakout"))

        assert result["status"] == "success"
        assert result["content"] == "LLM fallback response"
        assert "delegation" not in result
        manager._invoke_llm.assert_awaited_once()
        manager.close()


def test_chat_uses_live_pending_approval_summary(monkeypatch):
    from src.agents.departments.floor_manager import FloorManager

    fake_pending = [
        {
            "id": "apr-1",
            "title": "Backtest Ready: EURUSD_v1",
            "department": "trading",
            "urgency": "high",
            "created_at": "2026-04-01T05:31:00Z",
        },
        {
            "id": "apr-2",
            "title": "Tool execution approval",
            "department": "risk",
            "urgency": "medium",
            "created_at": "2026-04-01T05:32:00Z",
        },
    ]

    monkeypatch.setattr(
        "src.agents.approval_manager.get_approval_manager",
        lambda: SimpleNamespace(get_pending=lambda: fake_pending),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_mail.db")
        manager = FloorManager(mail_db_path=db_path)
        manager._invoke_llm = AsyncMock(return_value="hallucinated response")

        result = asyncio.run(manager.chat("Show pending approvals"))

        assert result["status"] == "success"
        assert result["type"] == "approval_summary"
        assert result["approval_count"] == 2
        assert "Pending approvals: 2" in result["content"]
        assert "Backtest Ready: EURUSD_v1" in result["content"]
        manager._invoke_llm.assert_not_awaited()
        manager.close()

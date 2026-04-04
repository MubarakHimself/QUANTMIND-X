import asyncio
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock
import json


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


def test_invoke_llm_disables_httpx_env_inheritance(monkeypatch):
    from src.agents.departments.floor_manager import FloorManager
    from src.agents.providers.router import RuntimeLLMConfig

    captured: dict = {}

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"content": [{"type": "text", "text": "workshop"}]}

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            captured["trust_env"] = kwargs.get("trust_env")
            captured["timeout"] = kwargs.get("timeout")

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, headers=None, json=None):
            captured["url"] = url
            captured["headers"] = headers or {}
            captured["json"] = json or {}
            return FakeResponse()

    monkeypatch.setattr(
        "src.agents.providers.router.get_router",
        lambda: SimpleNamespace(
            resolve_runtime_config=lambda **kwargs: RuntimeLLMConfig(
                provider_type="minimax",
                api_key="test-key",
                base_url="https://api.minimax.io/anthropic",
                model="MiniMax-M2.7",
                source="test",
            )
        ),
    )
    monkeypatch.setattr("httpx.AsyncClient", FakeAsyncClient)

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = FloorManager(mail_db_path=os.path.join(tmpdir, "test_mail.db"))
        result = asyncio.run(
            manager._invoke_llm(
                "What canvas am I on right now?",
                history=[],
                model_name="MiniMax-M2.7",
                preferred_provider="minimax",
                system_prompt="You are concise.",
            )
        )

        assert result == "workshop"
        assert captured["trust_env"] is False
        assert captured["url"] == "https://api.minimax.io/anthropic/v1/messages"
        assert captured["headers"]["Authorization"] == "Bearer test-key"
        manager.close()


def test_invoke_llm_stream_disables_httpx_env_inheritance(monkeypatch):
    from src.agents.departments.floor_manager import FloorManager
    from src.agents.providers.router import RuntimeLLMConfig

    captured: dict = {}

    class FakeStreamResponse:
        status_code = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def aread(self):
            return b""

        async def aiter_lines(self):
            events = [
                {
                    "type": "content_block_start",
                    "content_block": {"type": "thinking"},
                },
                {
                    "type": "content_block_delta",
                    "delta": {"type": "thinking_delta", "thinking": "planning"},
                },
                {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": "workshop"},
                },
                {"type": "message_stop"},
            ]
            for event in events:
                yield f"data: {json.dumps(event)}"

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            captured["trust_env"] = kwargs.get("trust_env")
            captured["timeout"] = kwargs.get("timeout")

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def stream(self, method, url, headers=None, json=None):
            captured["method"] = method
            captured["url"] = url
            captured["headers"] = headers or {}
            captured["json"] = json or {}
            return FakeStreamResponse()

    monkeypatch.setattr(
        "src.agents.providers.router.get_router",
        lambda: SimpleNamespace(
            resolve_runtime_config=lambda **kwargs: RuntimeLLMConfig(
                provider_type="minimax",
                api_key="test-key",
                base_url="https://api.minimax.io/anthropic",
                model="MiniMax-M2.7",
                source="test",
            )
        ),
    )
    monkeypatch.setattr("httpx.AsyncClient", FakeAsyncClient)

    async def collect_events():
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = FloorManager(mail_db_path=os.path.join(tmpdir, "test_mail.db"))
            events = [
                event
                async for event in manager._invoke_llm_stream(
                    "What canvas am I on right now?",
                    history=[],
                    model_name="MiniMax-M2.7",
                    preferred_provider="minimax",
                    system_prompt="You are concise.",
                )
            ]
            manager.close()
            return events

    events = asyncio.run(collect_events())

    assert captured["trust_env"] is False
    assert captured["method"] == "POST"
    assert captured["url"] == "https://api.minimax.io/anthropic/v1/messages"
    assert any(event == {"type": "thinking_start"} for event in events)
    assert any(event == {"type": "content", "content": "workshop"} for event in events)
    assert events[-1] == {"type": "done"}

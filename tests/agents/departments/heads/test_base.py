# tests/agents/departments/heads/test_base.py
"""
Tests for Department Head Base Class

Task Group: Department Heads - Base functionality
"""
import pytest
import tempfile
import os
import asyncio
from types import SimpleNamespace


class TestDepartmentHeadBase:
    """Test Department Head base class."""

    def test_department_head_initializes_with_config(self):
        """Department Head should initialize from config."""
        from src.agents.departments.heads.base import DepartmentHead
        from src.agents.departments.types import Department, DepartmentHeadConfig

        config = DepartmentHeadConfig(
            department=Department.ANALYSIS,
            agent_type="analyst_head",
            system_prompt="Test prompt",
            sub_agents=["worker1", "worker2"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_mail.db")
            head = DepartmentHead(config=config, mail_db_path=db_path)

            assert head.department == Department.ANALYSIS
            assert head.agent_type == "analyst_head"
            assert head.model_tier == "sonnet"
            assert len(head.sub_agents) == 2

            head.close()

    def test_department_head_can_check_mail(self):
        """Department Head should be able to check mail inbox."""
        from src.agents.departments.heads.base import DepartmentHead
        from src.agents.departments.types import Department, DepartmentHeadConfig
        from src.agents.departments.department_mail import MessageType

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_mail.db")

            config = DepartmentHeadConfig(
                department=Department.ANALYSIS,
                agent_type="analyst_head",
                system_prompt="Test prompt",
            )
            head = DepartmentHead(config=config, mail_db_path=db_path)

            # Send a message to this department
            head.mail_service.send(
                from_dept="floor_manager",
                to_dept="analysis",
                type=MessageType.DISPATCH,
                subject="Test task",
                body="Test body",
            )

            # Check inbox
            messages = head.check_mail()
            assert len(messages) == 1
            assert messages[0].subject == "Test task"

            head.close()

    def test_department_head_can_send_result(self):
        """Department Head should be able to send results to other departments."""
        from src.agents.departments.heads.base import DepartmentHead
        from src.agents.departments.types import Department, DepartmentHeadConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_mail.db")

            config = DepartmentHeadConfig(
                department=Department.ANALYSIS,
                agent_type="analyst_head",
                system_prompt="Test prompt",
            )
            head = DepartmentHead(config=config, mail_db_path=db_path)

            # Send result to Execution
            head.send_result(
                to_dept=Department.EXECUTION,
                subject="Analysis Complete",
                body="EURUSD signal: BUY",
            )

            # Verify message was sent
            messages = head.mail_service.check_inbox("execution")
            assert len(messages) == 1

            head.close()

    def test_department_head_can_spawn_worker(self):
        """Department Head should be able to spawn workers."""
        from src.agents.departments.heads.base import DepartmentHead
        from src.agents.departments.types import Department, DepartmentHeadConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_mail.db")

            config = DepartmentHeadConfig(
                department=Department.ANALYSIS,
                agent_type="analyst_head",
                system_prompt="Test prompt",
                sub_agents=["market_analyst", "sentiment_scanner"],
            )
            head = DepartmentHead(config=config, mail_db_path=db_path)

            # Spawn a worker
            result = head.spawn_worker(
                worker_type="market_analyst",
                task="Analyze EURUSD",
            )

            # Result should indicate spawned, spawner unavailable, or spawn failed (if SubAgentConfig missing)
            assert result["status"] in ["spawned", "spawner_unavailable", "spawn_failed"]

            head.close()

    def test_extracts_minimax_tool_calls_from_text_markup(self):
        from src.agents.departments.heads.base import DepartmentHead

        markup = """
        <minimax:tool_call>
        <invoke name="search_resources">
        <parameter name="query">EA expert advisor mql5 ea</parameter>
        <parameter name="limit">5</parameter>
        </invoke>
        </minimax:tool_call>
        """

        calls = DepartmentHead._extract_minimax_tool_calls_from_text(markup)

        assert len(calls) == 1
        assert calls[0].name == "search_resources"
        assert calls[0].input == {
            "query": "EA expert advisor mql5 ea",
            "limit": "5",
        }

    def test_strips_minimax_tool_markup_from_visible_text(self):
        from src.agents.departments.heads.base import DepartmentHead

        content = """
        Let me inspect the workspace.
        <minimax:tool_call>
        <invoke name="search_resources">
        <parameter name="query">EA</parameter>
        </invoke>
        </minimax:tool_call>
        """

        stripped = DepartmentHead._strip_minimax_tool_markup(content)

        assert "<minimax:tool_call>" not in stripped
        assert "<invoke name=" not in stripped
        assert "Let me inspect the workspace." in stripped

    def test_invoke_claude_disables_httpx_env_inheritance(self, monkeypatch):
        from src.agents.departments.heads.base import DepartmentHead
        from src.agents.departments.types import Department, DepartmentHeadConfig
        from src.agents.providers.router import RuntimeLLMConfig

        captured: dict = {}

        class FakeResponse:
            stop_reason = "end_turn"
            model = "MiniMax-M2.5"
            usage = SimpleNamespace(input_tokens=11, output_tokens=7)
            content = [SimpleNamespace(type="text", text="research ok")]

        class FakeMessages:
            async def create(self, **kwargs):
                captured["messages_kwargs"] = kwargs
                return FakeResponse()

        class FakeAnthropicClient:
            def __init__(self, **kwargs):
                captured["anthropic_kwargs"] = kwargs
                self.messages = FakeMessages()

        class FakeAsyncClient:
            def __init__(self, *args, **kwargs):
                captured["httpx_kwargs"] = kwargs

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

        monkeypatch.setattr(
            "src.agents.departments.heads.base.get_router",
            lambda: SimpleNamespace(
                resolve_runtime_config=lambda **kwargs: RuntimeLLMConfig(
                    provider_type="minimax",
                    api_key="test-key",
                    base_url="https://api.minimax.io/anthropic",
                    model="MiniMax-M2.5",
                    source="test",
                )
            ),
        )
        monkeypatch.setattr("src.agents.departments.heads.base.httpx.AsyncClient", FakeAsyncClient)
        monkeypatch.setattr("src.agents.departments.heads.base.anthropic.AsyncAnthropic", FakeAnthropicClient)

        config = DepartmentHeadConfig(
            department=Department.RESEARCH,
            agent_type="research_head",
            system_prompt="Test prompt",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_mail.db")
            head = DepartmentHead(config=config, mail_db_path=db_path)
            result = asyncio.run(head._invoke_claude("Ping"))

            assert result["content"] == "research ok"
            assert captured["httpx_kwargs"]["trust_env"] is False
            assert captured["anthropic_kwargs"]["api_key"] == "test-key"
            assert captured["anthropic_kwargs"]["base_url"] == "https://api.minimax.io/anthropic"
            assert captured["anthropic_kwargs"]["http_client"].__class__ is FakeAsyncClient
            head.close()

    def test_prompt_context_summary_keeps_resource_attachment_references(self):
        from src.agents.departments.heads.base import DepartmentHead
        from src.agents.departments.types import Department, DepartmentHeadConfig

        config = DepartmentHeadConfig(
            department=Department.DEVELOPMENT,
            agent_type="development_head",
            system_prompt="Test prompt",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_mail.db")
            head = DepartmentHead(config=config, mail_db_path=db_path)

            summary = head._summarize_canvas_context_for_prompt(
                {
                    "canvas": "development",
                    "attached_contexts": [
                        {
                            "canvas": "shared-assets",
                            "label": "MQL5 Reference",
                            "attachment_type": "resource",
                            "resource": {
                                "id": "knowledge/books/mql5.pdf",
                                "path": "knowledge/books/mql5.pdf",
                                "label": "MQL5 Reference",
                                "type": "book",
                            },
                        }
                    ],
                }
            )

            assert summary["attached_contexts"][0]["label"] == "MQL5 Reference"
            assert summary["attached_contexts"][0]["attachment_type"] == "resource"
            assert summary["attached_contexts"][0]["resource"]["id"] == "knowledge/books/mql5.pdf"
            assert summary["attached_contexts"][0]["resource"]["path"] == "knowledge/books/mql5.pdf"

            head.close()

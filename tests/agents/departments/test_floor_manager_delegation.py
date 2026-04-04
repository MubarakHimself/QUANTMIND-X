# tests/agents/departments/test_floor_manager_delegation.py
"""
Tests for Floor Manager Delegation Functionality

Phase 3: Copilot -> Floor Manager Delegation UI
Tests for handling dispatch requests from Copilot and delegating to departments.
"""
import pytest
import tempfile
import os
import asyncio
from unittest.mock import AsyncMock


class TestHandleDispatch:
    """Test Floor Manager handle_dispatch method for processing Copilot messages."""

    def test_handle_dispatch_from_copilot(self):
        """Should handle dispatch request from Copilot and route to department."""
        from src.agents.departments.floor_manager import FloorManager

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_mail.db")
            manager = FloorManager(mail_db_path=db_path)

            # Simulate dispatch from Copilot
            result = manager.handle_dispatch(
                from_department="copilot",
                task="Analyze EURUSD for trading opportunities",
                suggested_department=None,  # Let Floor Manager classify
            )

            assert result["status"] == "dispatched"
            assert "message_id" in result
            assert result["from_department"] == "copilot"
            assert result["to_department"] == "analysis"  # Classified by keywords

            # Verify mail was sent
            messages = manager.mail_service.check_inbox("analysis")
            assert len(messages) == 1
            assert messages[0].from_dept == "copilot"

            manager.close()

    def test_handle_dispatch_with_suggested_department(self):
        """Should respect suggested department from Copilot."""
        from src.agents.departments.floor_manager import FloorManager

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_mail.db")
            manager = FloorManager(mail_db_path=db_path)

            # Copilot suggests specific department
            result = manager.handle_dispatch(
                from_department="copilot",
                task="Work on this trading strategy",
                suggested_department="research",
            )

            assert result["status"] == "dispatched"
            assert result["to_department"] == "research"

            messages = manager.mail_service.check_inbox("research")
            assert len(messages) == 1

            manager.close()

    def test_handle_dispatch_with_invalid_suggested_department(self):
        """Should fall back to classification if suggested department is invalid."""
        from src.agents.departments.floor_manager import FloorManager

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_mail.db")
            manager = FloorManager(mail_db_path=db_path)

            # Invalid suggested department
            result = manager.handle_dispatch(
                from_department="copilot",
                task="Execute buy order for EURUSD",
                suggested_department="invalid_dept",
            )

            assert result["status"] == "dispatched"
            # Should classify to execution based on keywords
            assert result["to_department"] == "execution"

            manager.close()

    def test_handle_dispatch_with_context(self):
        """Should include context in the dispatch message."""
        from src.agents.departments.floor_manager import FloorManager

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_mail.db")
            manager = FloorManager(mail_db_path=db_path)

            context = {
                "symbol": "EURUSD",
                "timeframe": "H1",
                "user_id": "test_user",
            }

            result = manager.handle_dispatch(
                from_department="copilot",
                task="Analyze this pair",
                suggested_department=None,
                context=context,
            )

            assert result["status"] == "dispatched"

            # Verify context is included in message body
            messages = manager.mail_service.check_inbox("analysis")
            assert len(messages) == 1
            import json
            body = json.loads(messages[0].body)
            assert body["context"]["symbol"] == "EURUSD"

            manager.close()


class TestDelegateToDepartment:
    """Test Floor Manager delegate_to_department method."""

    def test_delegate_to_department_sends_mail(self):
        """Should send mail message to target department."""
        from src.agents.departments.floor_manager import FloorManager
        from src.agents.departments.department_mail import MessageType, Priority

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_mail.db")
            manager = FloorManager(mail_db_path=db_path)

            message = manager.delegate_to_department(
                from_dept="copilot",
                to_dept="analysis",
                task="Analyze market conditions",
                priority="normal",
            )

            assert message.from_dept == "copilot"
            assert message.to_dept == "analysis"
            assert message.type == MessageType.DISPATCH
            assert message.priority == Priority.NORMAL
            assert "market conditions" in message.body

            manager.close()

    def test_delegate_to_department_with_high_priority(self):
        """Should set correct priority on mail message."""
        from src.agents.departments.floor_manager import FloorManager
        from src.agents.departments.department_mail import Priority

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_mail.db")
            manager = FloorManager(mail_db_path=db_path)

            message = manager.delegate_to_department(
                from_dept="copilot",
                to_dept="execution",
                task="Urgent order execution",
                priority="high",
            )

            assert message.priority == Priority.HIGH

            manager.close()

    def test_delegate_to_department_with_context(self):
        """Should include context in message body as JSON."""
        from src.agents.departments.floor_manager import FloorManager

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_mail.db")
            manager = FloorManager(mail_db_path=db_path)

            context = {"symbol": "GBPUSD", "lots": 0.1}

            message = manager.delegate_to_department(
                from_dept="copilot",
                to_dept="risk",
                task="Calculate position size",
                priority="normal",
                context=context,
            )

            import json
            body = json.loads(message.body)
            assert body["task"] == "Calculate position size"
            assert body["context"]["symbol"] == "GBPUSD"
            assert body["context"]["lots"] == 0.1

            manager.close()


class TestCopilotDelegationWorkflow:
    """Test end-to-end Copilot delegation workflow."""

    def test_copilot_delegation_to_floor_manager(self):
        """Complete workflow: Copilot delegates task to Floor Manager."""
        from src.agents.departments.floor_manager import FloorManager

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_mail.db")
            manager = FloorManager(mail_db_path=db_path)

            # Step 1: Copilot sends delegation request
            delegation_request = {
                "from_department": "copilot",
                "task": "Develop new RSI strategy",
                "suggested_department": "research",
            }

            # Step 2: Floor Manager handles dispatch
            result = manager.handle_dispatch(
                from_department=delegation_request["from_department"],
                task=delegation_request["task"],
                suggested_department=delegation_request["suggested_department"],
            )

            # Step 3: Verify result
            assert result["status"] == "dispatched"
            assert result["to_department"] == "research"

            # Step 4: Verify Research department received mail
            messages = manager.mail_service.check_inbox("research")
            assert len(messages) == 1
            assert messages[0].from_dept == "copilot"
            assert "RSI strategy" in messages[0].body

            manager.close()

    def test_floor_manager_auto_classification_when_no_suggestion(self):
        """Floor Manager should auto-classify when Copilot doesn't suggest department."""
        from src.agents.departments.floor_manager import FloorManager

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_mail.db")
            manager = FloorManager(mail_db_path=db_path)

            # Copilot doesn't suggest a department
            result = manager.handle_dispatch(
                from_department="copilot",
                task="Check portfolio drawdown",
                suggested_department=None,
            )

            # Floor Manager classifies based on keywords
            assert result["status"] == "dispatched"
            assert result["to_department"] == "risk"

            messages = manager.mail_service.check_inbox("risk")
            assert len(messages) == 1

            manager.close()


class TestFloorManagerChatDelegationGate:
    """Regression tests for chat delegation gating."""

    def test_chat_uses_llm_for_non_delegatable_prompt(self):
        """Conversational prompts should stay on Floor Manager LLM."""
        from src.agents.departments.floor_manager import FloorManager

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_mail.db")
            manager = FloorManager(mail_db_path=db_path)

            manager._invoke_llm = AsyncMock(return_value="direct floor manager reply")
            manager._delegate_to_department_head = AsyncMock(
                return_value={"status": "success", "content": "delegated"}
            )

            result = asyncio.run(manager.chat("stream check please"))

            assert result["status"] == "success"
            assert result["content"] == "direct floor manager reply"
            manager._invoke_llm.assert_called_once()
            manager._delegate_to_department_head.assert_not_called()

            manager.close()

    def test_chat_does_not_delegate_check_queries(self):
        """Read/query prompts should not be delegated by keyword side effects."""
        from src.agents.departments.floor_manager import FloorManager

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_mail.db")
            manager = FloorManager(mail_db_path=db_path)

            manager._invoke_llm = AsyncMock(return_value="workflow status summary")
            manager._delegate_to_department_head = AsyncMock(
                return_value={"status": "success", "content": "delegated"}
            )

            result = asyncio.run(manager.chat("check active workflows"))

            assert result["status"] == "success"
            assert result["content"] == "workflow status summary"
            manager._invoke_llm.assert_called_once()
            manager._delegate_to_department_head.assert_not_called()

            manager.close()

    def test_chat_delegates_on_explicit_dispatch(self):
        """Explicit delegation instructions should route to department head."""
        from src.agents.departments.floor_manager import FloorManager

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_mail.db")
            manager = FloorManager(mail_db_path=db_path)

            manager._invoke_llm = AsyncMock(return_value="direct floor manager reply")
            manager._delegate_to_department_head = AsyncMock(
                return_value={
                    "status": "success",
                    "content": "research task accepted",
                    "delegated_department": "research",
                }
            )

            result = asyncio.run(
                manager.chat("Delegate this to research and analyze EURUSD trend")
            )

            assert result["status"] == "success"
            assert result["content"] == "research task accepted"
            manager._delegate_to_department_head.assert_called_once()
            manager._invoke_llm.assert_not_called()

            manager.close()

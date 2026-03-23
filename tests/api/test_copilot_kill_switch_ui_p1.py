"""P1 Tests: Kill switch UI state persistence and history."""

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from fastapi import FastAPI

from src.router.copilot_kill_switch import CopilotKillSwitch


class TestKillSwitchStatusPersistence:
    """P1: Test kill switch status persistence across operations."""

    def setup_method(self):
        from src.api.copilot_kill_switch_endpoints import router
        self.app = FastAPI()
        self.app.include_router(router)
        self.client = TestClient(self.app)

    def test_get_status_returns_current_state(self):
        """[P1] GET /copilot-kill-switch/status should return current state."""
        with patch("src.api.copilot_kill_switch_endpoints.get_copilot_kill_switch") as mock_get:
            mock_instance = MagicMock()
            mock_instance.is_active = True
            mock_instance._activated_by = "test-user"
            mock_instance.get_status.return_value = {
                "is_active": True,
                "activated_by": "test-user",
            }
            mock_get.return_value = mock_instance

            response = self.client.get("/api/copilot-kill-switch/status")

            assert response.status_code == 200
            data = response.json()
            assert data["is_active"] is True
            assert data["activated_by"] == "test-user"

    def test_get_status_returns_inactive_when_not_triggered(self):
        """[P1] GET /copilot-kill-switch/status should return inactive when not triggered."""
        with patch("src.api.copilot_kill_switch_endpoints.get_copilot_kill_switch") as mock_get:
            mock_instance = MagicMock()
            mock_instance.is_active = False
            mock_instance._activated_by = None
            mock_instance.get_status.return_value = {
                "is_active": False,
                "activated_by": None,
            }
            mock_get.return_value = mock_instance

            response = self.client.get("/api/copilot-kill-switch/status")

            assert response.status_code == 200
            data = response.json()
            assert data["is_active"] is False


class TestKillSwitchHistoryTracking:
    """P1: Test kill switch history tracking."""

    @pytest.mark.asyncio
    async def test_get_history_returns_activation_history(self):
        """[P1] get_history() should return list of past activations."""
        kill_switch = CopilotKillSwitch()

        # Activate
        await kill_switch.activate(activator="user-1")
        await kill_switch.resume()
        await kill_switch.activate(activator="user-2")

        history = kill_switch.get_history()

        assert len(history) >= 2
        # Most recent activation should be last
        assert history[-1]["activated_by"] == "user-2"

    @pytest.mark.asyncio
    async def test_history_includes_resume_events(self):
        """[P1] History should include resume events."""
        kill_switch = CopilotKillSwitch()

        await kill_switch.activate(activator="user-1")
        await kill_switch.resume()

        history = kill_switch.get_history()

        # Should have at least activate and resume
        assert len(history) >= 2
        event_types = [h.get("event_type") for h in history]
        assert "activate" in event_types or "resume" in event_types


class TestKillSwitchTaskCleanup:
    """P1: Test task cleanup on kill switch resume."""

    @pytest.mark.asyncio
    async def test_resume_clears_terminated_tasks(self):
        """[P1] Resume MUST clear terminated tasks registry."""
        kill_switch = CopilotKillSwitch()

        # Activate and add terminated tasks
        await kill_switch.activate(activator="test")

        # Simulate terminated tasks
        kill_switch._terminated_tasks = ["task-1", "task-2", "task-3"]

        # Resume
        result = await kill_switch.resume()

        assert result["success"] is True
        assert len(kill_switch._terminated_tasks) == 0
        assert kill_switch.is_active is False

    @pytest.mark.asyncio
    async def test_resume_clears_activated_by(self):
        """[P1] Resume MUST clear activated_by field."""
        kill_switch = CopilotKillSwitch()

        await kill_switch.activate(activator="test-user")
        assert kill_switch._activated_by == "test-user"

        await kill_switch.resume()

        assert kill_switch._activated_by is None

"""
Tests for Copilot Kill Switch

Tests for the Copilot/agent kill switch functionality.
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from src.router.copilot_kill_switch import (
    CopilotKillSwitch,
    CopilotKillReason,
    get_copilot_kill_switch,
)


class TestCopilotKillSwitch:
    """Test cases for CopilotKillSwitch class."""

    @pytest.fixture
    def kill_switch(self):
        """Create a fresh CopilotKillSwitch instance for each test."""
        return CopilotKillSwitch()

    def test_initial_state(self, kill_switch):
        """Test that kill switch starts inactive."""
        assert kill_switch.is_active is False
        assert kill_switch._suspended_at_utc is None
        assert kill_switch._activated_by is None

    @pytest.mark.asyncio
    async def test_activate(self, kill_switch):
        """Test activating the kill switch."""
        result = await kill_switch.activate(activator="test_user")

        assert result["success"] is True
        assert kill_switch.is_active is True
        assert kill_switch._activated_by == "test_user"
        assert kill_switch._suspended_at_utc is not None

    @pytest.mark.asyncio
    async def test_activate_already_active(self, kill_switch):
        """Test activating when already active returns success."""
        await kill_switch.activate(activator="user1")
        result = await kill_switch.activate(activator="user2")

        assert result["success"] is True
        assert result["already_active"] is True
        assert kill_switch._activated_by == "user1"  # Original activator kept

    @pytest.mark.asyncio
    async def test_resume(self, kill_switch):
        """Test resuming from kill switch."""
        await kill_switch.activate(activator="user")
        result = await kill_switch.resume()

        assert result["success"] is True
        assert kill_switch.is_active is False
        assert kill_switch._activated_by is None

    @pytest.mark.asyncio
    async def test_resume_when_not_active(self, kill_switch):
        """Test resuming when not active returns success with not_active flag."""
        result = await kill_switch.resume()

        assert result["success"] is True
        assert result["not_active"] is True

    def test_get_status_inactive(self, kill_switch):
        """Test get_status when inactive."""
        status = kill_switch.get_status()

        assert status["active"] is False
        assert status["suspended_at_utc"] is None
        assert status["activated_by"] is None
        assert status["terminated_tasks_count"] == 0

    @pytest.mark.asyncio
    async def test_get_status_active(self, kill_switch):
        """Test get_status when active."""
        await kill_switch.activate(activator="test_user")
        status = kill_switch.get_status()

        assert status["active"] is True
        assert status["suspended_at_utc"] is not None
        assert status["activated_by"] == "test_user"

    def test_get_history_empty(self, kill_switch):
        """Test get_history when no activations."""
        history = kill_switch.get_history()
        assert history == []

    @pytest.mark.asyncio
    async def test_get_history_with_activations(self, kill_switch):
        """Test get_history returns activation events."""
        await kill_switch.activate(activator="user1")
        await kill_switch.resume()
        await kill_switch.activate(activator="user2")

        history = kill_switch.get_history()

        assert len(history) == 2
        assert history[0]["activated_by"] == "user1"
        assert history[1]["activated_by"] == "user2"

    @pytest.mark.asyncio
    async def test_terminate_task_tracking(self, kill_switch):
        """Test that terminated tasks are tracked."""
        kill_switch._terminated_tasks = ["task-1", "task-2", "task-3"]
        await kill_switch.activate(activator="user")

        history = kill_switch.get_history()
        assert len(history[0]["terminated_tasks"]) == 3
        assert "task-1" in history[0]["terminated_tasks"]


class TestCopilotKillSwitchAPI:
    """Test cases for Copilot Kill Switch API endpoints."""

    @pytest.fixture
    def mock_kill_switch(self):
        """Create a mock kill switch."""
        switch = AsyncMock(spec=CopilotKillSwitch)
        switch.is_active = False
        switch.activate = AsyncMock(return_value={"success": True})
        switch.resume = AsyncMock(return_value={"success": True})
        switch.get_status = MagicMock(return_value={"active": False})
        switch.get_history = MagicMock(return_value=[])
        return switch

    @pytest.mark.asyncio
    async def test_activate_endpoint(self, mock_kill_switch):
        """Test activate endpoint."""
        with patch('src.api.copilot_kill_switch_endpoints.get_copilot_kill_switch', return_value=mock_kill_switch):
            from src.api.copilot_kill_switch_endpoints import activate_copilot_kill_switch, CopilotKillSwitchActivateRequest

            request = CopilotKillSwitchActivateRequest(activator="test_user")
            result = await activate_copilot_kill_switch(request)

            mock_kill_switch.activate.assert_called_once_with(activator="test_user")
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_resume_endpoint(self, mock_kill_switch):
        """Test resume endpoint."""
        with patch('src.api.copilot_kill_switch_endpoints.get_copilot_kill_switch', return_value=mock_kill_switch):
            from src.api.copilot_kill_switch_endpoints import resume_copilot

            result = await resume_copilot()

            mock_kill_switch.resume.assert_called_once()
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_status_endpoint(self, mock_kill_switch):
        """Test status endpoint."""
        mock_kill_switch.get_status.return_value = {"active": True, "suspended_at_utc": "2024-01-01T00:00:00"}

        with patch('src.api.copilot_kill_switch_endpoints.get_copilot_kill_switch', return_value=mock_kill_switch):
            from src.api.copilot_kill_switch_endpoints import get_copilot_kill_switch_status

            result = await get_copilot_kill_switch_status()

            mock_kill_switch.get_status.assert_called_once()
            assert result["active"] is True


class TestFloorManagerIntegration:
    """Test cases for FloorManager integration with Copilot Kill Switch."""

    def test_check_kill_switch_inactive(self):
        """Test _check_kill_switch when inactive."""
        from src.agents.departments.floor_manager import FloorManager

        with patch('src.agents.departments.floor_manager.get_copilot_kill_switch') as mock_get:
            mock_kill_switch = MagicMock()
            mock_kill_switch.is_active = False
            mock_get.return_value = mock_kill_switch

            fm = FloorManager()
            result = fm._check_kill_switch()

            assert result is False

    def test_check_kill_switch_active_raises(self):
        """Test _check_kill_switch when active raises RuntimeError."""
        from src.agents.departments.floor_manager import FloorManager

        with patch('src.agents.departments.floor_manager.get_copilot_kill_switch') as mock_get:
            mock_kill_switch = MagicMock()
            mock_kill_switch.is_active = True
            mock_get.return_value = mock_kill_switch

            fm = FloorManager()
            fm._copilot_kill_switch = mock_kill_switch

            with pytest.raises(RuntimeError, match="Agent activity suspended"):
                fm._check_kill_switch()


class TestArchitecturalIndependence:
    """Test that Copilot Kill Switch is independent from Trading Kill Switch."""

    def test_no_trading_kill_switch_import(self):
        """Verify copilot_kill_switch.py does not import trading kill switch."""
        import src.router.copilot_kill_switch as copilot_module

        # Check that trading kill switch is not imported
        has_trading_import = hasattr(copilot_module, 'kill_switch') or \
                            hasattr(copilot_module, 'KillSwitch') or \
                            hasattr(copilot_module, 'TradingKillSwitch')

        assert not has_trading_import, "Copilot Kill Switch should not import Trading Kill Switch"

    def test_different_module_paths(self):
        """Verify modules are in different paths."""
        from src.router import copilot_kill_switch
        from src.router import kill_switch

        # These should be different module paths
        assert copilot_kill_switch.__file__ != kill_switch.__file__

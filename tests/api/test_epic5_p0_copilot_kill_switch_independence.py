"""P0 Tests: Copilot Kill Switch Architectural Independence (Epic 5 - R-004)

These tests verify that the Copilot kill switch is architecturally
independent from the Trading kill switch - they must not share code,
imports, or state.

Test Design Reference: _bmad-output/test-artifacts/test-design-epic-5.md
Risk: R-004 (Score 6) - Copilot kill switch activation inadvertently blocks trading
Priority: P0 - High risk (>=6), no workaround
"""

import pytest
import tempfile
import importlib
import inspect

from src.router.copilot_kill_switch import (
    CopilotKillSwitch,
    get_copilot_kill_switch,
)
from src.router.kill_switch import KillSwitch


class TestCopilotKillSwitchArchitecturalIndependence:
    """Test R-004: Copilot kill switch must be independent from trading kill switch."""

    def test_copilot_kill_switch_does_not_import_trading_kill_switch(self):
        """P0: copilot_kill_switch module MUST NOT import kill_switch module.

        This is the core architectural requirement - these two systems
        must be completely isolated at the code level.
        """
        # Get the copilot kill switch module
        copilot_module = inspect.getmodule(CopilotKillSwitch)

        # Get all imports in the copilot kill switch module
        copilot_source = inspect.getsource(copilot_module)

        # Check that trading kill switch is not imported
        # We check for direct import statements
        assert "from src.router.kill_switch import" not in copilot_source, (
            "ARCHITECTURAL VIOLATION: copilot_kill_switch.py imports from kill_switch.py! "
            "These must be completely independent modules."
        )

        assert "import src.router.kill_switch" not in copilot_source, (
            "ARCHITECTURAL VIOLATION: copilot_kill_switch.py imports src.router.kill_switch! "
            "These must be completely independent modules."
        )

        assert "from src.router import kill_switch" not in copilot_source, (
            "ARCHITECTURAL VIOLATION: copilot_kill_switch.py imports kill_switch from router! "
            "These must be completely independent modules."
        )

    def test_copilot_kill_switch_does_not_share_state_with_trading_kill_switch(self):
        """P0: Copilot and Trading kill switches MUST NOT share state.

        Activating one should not affect the state of the other.
        """
        # Create instances of both kill switches
        from src.router.kill_switch import KillSwitch

        trading_kill = KillSwitch()
        copilot_kill = CopilotKillSwitch()

        # Initially both should be inactive
        assert trading_kill._active is False, "Trading kill switch should start inactive"
        assert copilot_kill.is_active is False, "Copilot kill switch should start inactive"

        # Activate copilot kill switch
        import asyncio
        asyncio.get_event_loop().run_until_complete(
            copilot_kill.activate(activator="test")
        )

        # Copilot should be active, trading should still be inactive
        assert copilot_kill.is_active is True, "Copilot kill switch should be active"
        assert trading_kill._active is False, (
            "ARCHITECTURAL VIOLATION: Activating copilot kill switch somehow activated "
            "trading kill switch! These systems must not share state."
        )

    def test_copilot_kill_switch_activation_does_not_affect_trading_endpoints(self):
        """P0: Activating Copilot kill switch MUST NOT affect trading operations.

        This is an integration test that verifies the architectural isolation
        works at runtime - trading can continue even when Copilot is killed.
        """
        import asyncio

        copilot_kill = CopilotKillSwitch()

        # Verify copilot is inactive initially
        assert copilot_kill.is_active is False

        # Activate copilot kill switch
        result = asyncio.get_event_loop().run_until_complete(
            copilot_kill.activate(activator="test")
        )

        assert result["success"] is True
        assert copilot_kill.is_active is True

        # The trading kill switch (KillSwitch class) should be completely unaffected
        # It should still be in its own default state
        from src.router.kill_switch import KillSwitch

        trading_kill = KillSwitch()

        # Trading kill switch should remain inactive
        assert trading_kill._active is False, (
            "ARCHITECTURAL VIOLATION: Trading kill switch became active when "
            "copilot kill switch was activated! These are sharing state!"
        )

        # Trading kill switch should still be operational (can be activated independently)
        trading_kill.panic(reason="TEST")
        assert trading_kill._active is True, (
            "Trading kill switch should still be functional after "
            "copilot kill switch activation"
        )

    def test_copilot_kill_switch_methods_are_completely_different(self):
        """P0: Copilot and Trading kill switches MUST have different interfaces.

        They serve different purposes and should not share method names
        that could cause confusion or coupling.
        """
        copilot_methods = set(dir(CopilotKillSwitch))
        trading_methods = set(dir(KillSwitch))

        # These are the core methods that should be different
        copilot_core_methods = {"activate", "resume", "get_status", "get_history", "cleanup_draft_nodes"}
        trading_core_methods = {"panic", "resume", "get_status", "register_ea"}

        # At least the primary activation methods should be different
        # Trading uses "panic", Copilot uses "activate"
        assert "activate" in copilot_core_methods
        assert "panic" in trading_core_methods

        # Verify they don't share the exact same activation method
        assert "panic" not in copilot_methods, (
            "Copilot kill switch should not have 'panic' method - "
            "it should use 'activate' instead"
        )

        assert "activate" not in trading_methods, (
            "Trading kill switch should not have 'activate' method - "
            "it should use 'panic' instead"
        )

    def test_copilot_kill_switch_resume_clears_terminated_tasks(self):
        """P0: Resume operation MUST clear terminated tasks registry.

        When the Copilot kill switch is resumed, it must properly
        reset its internal state including the terminated tasks list.
        """
        import asyncio

        copilot_kill = CopilotKillSwitch()

        # Activate and check initial state
        asyncio.get_event_loop().run_until_complete(
            copilot_kill.activate(activator="test")
        )

        assert len(copilot_kill._terminated_tasks) == 0  # No tasks were registered

        # Manually add some terminated tasks to simulate cancellation
        copilot_kill._terminated_tasks = ["task-1", "task-2", "task-3"]

        # Resume
        result = asyncio.get_event_loop().run_until_complete(
            copilot_kill.resume()
        )

        assert result["success"] is True
        assert copilot_kill.is_active is False

        # CRITICAL: Terminated tasks should be cleared on resume
        assert len(copilot_kill._terminated_tasks) == 0, (
            f"Resume did not clear terminated tasks! Still have: {copilot_kill._terminated_tasks}"
        )
        assert copilot_kill._activated_by is None, (
            "Resume did not clear activated_by"
        )

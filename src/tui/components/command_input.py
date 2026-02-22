"""
CommandInput Component - Command Input with History

Provides a command input widget with history, tab completion,
and command execution for the TUI dashboard.
"""

from dataclasses import dataclass
from typing import List, Optional, Callable, Dict
from enum import Enum

from textual.widgets import Input
from textual.reactive import reactive
from textual import events
from textual.message import Message


class CommandMessageType(str, Enum):
    """Types of command messages."""
    EXECUTED = "executed"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass
class CommandEntry:
    """A command history entry."""
    command: str
    timestamp: str
    result: Optional[str] = None
    success: bool = True


class CommandInput(Input):
    """Command input widget with history and completion.

    Features:
    - Command history (up/down arrows)
    - Tab completion
    - Command execution callback
    - Visual feedback
    """

    DEFAULT_CSS = """
    CommandInput {
        dock: bottom;
        height: 3;
    }

    CommandInput.-focus {
        border: tall $primary;
    }

    CommandInput .prompt {
        text-style: bold;
        color: $success;
    }
    """

    # Reactive state
    history: reactive[List[CommandEntry]] = reactive(list)
    history_index: reactive[int] = reactive(-1)
    command_handlers: reactive[Dict[str, Callable]] = reactive(dict)

    def __init__(
        self,
        id: str | None = None,
        prompt: str = ">",
        placeholder: str = "Enter command...",
        max_history: int = 100
    ):
        """Initialize the CommandInput widget.

        Args:
            id: Widget ID
            prompt: Command prompt string
            placeholder: Placeholder text when empty
            max_history: Maximum number of commands to keep in history
        """
        super().__init__(
            id=id,
            placeholder=placeholder,
            value=""
        )
        self.prompt = prompt
        self.max_history = max_history
        self._current_input = ""

        # Register default command handlers
        self._register_default_handlers()

    def _register_default_handlers(self) -> None:
        """Register default command handlers."""
        self.command_handlers = {
            "help": self._cmd_help,
            "clear": self._cmd_clear,
            "refresh": self._cmd_refresh,
            "quit": self._cmd_quit,
            "exit": self._cmd_quit,
            "status": self._cmd_status,
            "strategies": self._cmd_strategies,
            "logs": self._cmd_logs,
        }

    def on_mount(self) -> None:
        """Initialize on mount."""
        self.border_title = "Command"
        self.placeholder = f"{self.prompt} Enter command..."

    def _add_to_history(
        self,
        command: str,
        result: Optional[str] = None,
        success: bool = True
    ) -> None:
        """Add a command to history.

        Args:
            command: The command string
            result: Optional result message
            success: Whether the command succeeded
        """
        from datetime import datetime

        entry = CommandEntry(
            command=command,
            timestamp=datetime.now().isoformat(),
            result=result,
            success=success
        )

        self.history = self.history[-self.max_history + 1:] + [entry]
        self.history_index = len(self.history)

    # Command handlers

    def _cmd_help(self, args: List[str]) -> str:
        """Show help message."""
        help_text = [
            "Available commands:",
            "  help      - Show this help message",
            "  clear     - Clear screen/logs",
            "  refresh   - Refresh data from API",
            "  status    - Show system status",
            "  strategies - List all strategies",
            "  logs      - Show log entries",
            "  quit/exit - Exit application",
        ]
        return "\n".join(help_text)

    def _cmd_clear(self, args: List[str]) -> str:
        """Clear screen/logs."""
        self.app.post_message(self.CommandClear(self))
        return "Screen cleared"

    def _cmd_refresh(self, args: List[str]) -> str:
        """Refresh data from API."""
        self.app.post_message(self.CommandRefresh(self))
        return "Refreshing data..."

    def _cmd_quit(self, args: List[str]) -> str:
        """Exit application."""
        self.app.exit()
        return "Exiting..."

    def _cmd_status(self, args: List[str]) -> str:
        """Show system status."""
        # Try to get status from parent app
        if hasattr(self.app, "get_system_status"):
            return self.app.get_system_status()
        return "Status: Running"

    def _cmd_strategies(self, args: List[str]) -> str:
        """List strategies."""
        if hasattr(self.app, "get_strategies_status"):
            return self.app.get_strategies_status()
        return "Use the Kanban board to view strategies"

    def _cmd_logs(self, args: List[str]) -> str:
        """Show log entries."""
        if hasattr(self.app, "get_logs_status"):
            return self.app.get_logs_status()
        return "Use the Log Viewer to view logs"

    # Command execution

    def execute_command(self, command: str) -> None:
        """Execute a command string.

        Args:
            command: The command to execute
        """
        command = command.strip()
        if not command:
            return

        parts = command.split()
        cmd_name = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []

        result = None
        success = True

        try:
            if cmd_name in self.command_handlers:
                # Execute registered handler
                result = self.command_handlers[cmd_name](args)
            else:
                # Unknown command
                result = f"Unknown command: {cmd_name}. Type 'help' for available commands."
                success = False

            self._add_to_history(command, result, success)
            self.post_message(
                self.CommandExecuted(
                    self,
                    command=command,
                    result=result,
                    success=success
                )
            )

        except Exception as e:
            error_msg = f"Error executing command: {e}"
            self._add_to_history(command, error_msg, False)
            self.post_message(
                self.CommandFailed(
                    self,
                    command=command,
                    error=str(e)
                )
            )

    # Event handlers

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission (Enter key)."""
        command = event.value
        self.value = ""  # Clear input

        if command:
            self.execute_command(command)

    async def on_key(self, event: events.Key) -> None:
        """Handle key events for navigation."""
        if event.key == "up":
            # Navigate back in history
            if self.history and self.history_index > 0:
                self.history_index -= 1
                self.value = self.history[self.history_index].command
                event.stop()

        elif event.key == "down":
            # Navigate forward in history
            if self.history_index < len(self.history) - 1:
                self.history_index += 1
                self.value = self.history[self.history_index].command
            else:
                self.history_index = len(self.history)
                self.value = ""
            event.stop()

        elif event.key == "tab":
            # Tab completion (basic)
            event.stop()
            await self._complete_command()

    async def _complete_command(self) -> None:
        """Perform tab completion on current input."""
        current = self.value.lower()
        if not current:
            return

        # Find matching commands
        matches = [cmd for cmd in self.command_handlers if cmd.startswith(current)]

        if len(matches) == 1:
            # Complete to single match
            self.value = matches[0] + " "
        elif len(matches) > 1:
            # Show possible completions
            longest_common = self._find_common_prefix(matches)
            if longest_common and longest_common > len(current):
                self.value = longest_common
            else:
                # Show matches in log
                self.post_message(
                    self.CommandCompletion(
                        self,
                        matches=matches
                    )
                )

    def _find_common_prefix(self, strings: List[str]) -> str:
        """Find the common prefix of a list of strings."""
        if not strings:
            return ""

        prefix = strings[0]
        for s in strings[1:]:
            while not s.startswith(prefix):
                prefix = prefix[:-1]
                if not prefix:
                    break
        return prefix

    # Messages

    class CommandExecuted(Message):
        """Message sent when a command is executed."""

        def __init__(self, sender: "CommandInput", command: str, result: Optional[str], success: bool):
            super().__init__(sender)
            self.command = command
            self.result = result
            self.success = success

    class CommandFailed(Message):
        """Message sent when a command fails."""

        def __init__(self, sender: "CommandInput", command: str, error: str):
            super().__init__(sender)
            self.command = command
            self.error = error

    class CommandClear(Message):
        """Message sent to request screen clear."""
        pass

    class CommandRefresh(Message):
        """Message sent to request data refresh."""
        pass

    class CommandCompletion(Message):
        """Message sent with tab completion results."""

        def __init__(self, sender: "CommandInput", matches: List[str]):
            super().__init__(sender)
            self.matches = matches

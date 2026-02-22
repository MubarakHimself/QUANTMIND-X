"""
YouTube-EA Pipeline TUI Server

Main TUI dashboard for monitoring the YouTube to Expert Advisor pipeline.
Integrates AuthBar, KanbanBoard, LogViewer, and CommandInput components.
"""

import os
import sys
from typing import Optional
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Static
from textual.containers import Horizontal, Vertical
from textual import log

from src.tui.components import (
    AuthBar,
    KanbanBoard,
    LogViewer,
    CommandInput,
)


class YouTubeEADashboard(App):
    """YouTube-EA Pipeline TUI Dashboard.

    Main application class that integrates all TUI components:
    - AuthBar: API authentication status
    - KanbanBoard: Strategy pipeline visualization
    - LogViewer: Real-time log monitoring
    - CommandInput: Command interface
    """

    # App configuration
    TITLE = "YouTube-EA Pipeline Dashboard"
    CSS_PATH = None  # No external CSS file
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh", "Refresh"),
        ("c", "focus_command", "Command"),
        ("l", "focus_logs", "Logs"),
        ("k", "focus_kanban", "Kanban"),
        ("?", "show_help", "Help"),
    ]

    def __init__(self, api_base_url: str = "http://localhost:8000"):
        """Initialize the dashboard.

        Args:
            api_base_url: Base URL for backend API
        """
        super().__init__()
        self._api_base_url = api_base_url
        self._log_viewer: Optional[LogViewer] = None
        self._kanban_board: Optional[KanbanBoard] = None

        # Log startup
        self._startup_time = datetime.now()

    def compose(self) -> ComposeResult:
        """Compose the UI layout."""
        yield Header(show_clock=True)

        # Auth bar at top
        yield AuthBar()

        # Main content area
        with Horizontal():
            # Left side: Kanban board (strategy pipeline)
            with Vertical(id="left-panel"):
                yield Static("Strategy Pipeline", id="kanban-title")
                yield KanbanBoard(id="kanban-board", api_base_url=self._api_base_url)

            # Right side: Log viewer
            with Vertical(id="right-panel"):
                yield Static("Log Viewer", id="logs-title")
                yield LogViewer(id="log-viewer", max_logs=500)

        # Command input at bottom
        yield CommandInput(id="command-input", prompt="EA>")

        yield Footer()

    def on_mount(self) -> None:
        """Initialize after mount."""
        # Get references to components
        self._log_viewer = self.query_one("#log-viewer", LogViewer)
        self._kanban_board = self.query_one("#kanban-board", KanbanBoard)

        # Log startup
        self._log_viewer.success(
            "YouTube-EA Pipeline Dashboard started",
            source="Dashboard"
        )
        self._log_viewer.info(
            f"Connected to API at {self._api_base_url}",
            source="API"
        )
        self._log_viewer.info(
            "Press '?' for help or 'q' to quit",
            source="Help"
        )

        # Set up auto-refresh interval for log status
        self.set_interval(5.0, self._update_status)

    def _update_status(self) -> None:
        """Periodic status update."""
        if self._log_viewer and self._kanban_board:
            pass  # Components handle their own refresh

    # Command handlers

    def action_quit(self) -> None:
        """Quit the application."""
        if self._log_viewer:
            self._log_viewer.info("Shutting down...", source="Dashboard")
        self.exit()

    def action_refresh(self) -> None:
        """Refresh all data."""
        if self._log_viewer:
            self._log_viewer.info("Refreshing data...", source="Dashboard")

        if self._kanban_board:
            self._kanban_board.fetch_strategies()

    def action_focus_command(self) -> None:
        """Focus the command input."""
        command_input = self.query_one("#command-input", CommandInput)
        command_input.focus()

    def action_focus_logs(self) -> None:
        """Focus the log viewer."""
        if self._log_viewer:
            self._log_viewer.focus()

    def action_focus_kanban(self) -> None:
        """Focus the kanban board."""
        if self._kanban_board:
            self._kanban_board.focus()

    def action_show_help(self) -> None:
        """Show help message."""
        help_text = """
[bold cyan]YouTube-EA Pipeline Dashboard[/bold cyan]

[yellow]Key Bindings:[/yellow]
  [cyan]q[/cyan]     - Quit application
  [cyan]r[/cyan]     - Refresh all data
  [cyan]c[/cyan]     - Focus command input
  [cyan]l[/cyan]     - Focus log viewer
  [cyan]k[/cyan]     - Focus kanban board
  [cyan]?[/cyan]     - Show this help

[yellow]Commands:[/yellow]
  [cyan]help[/cyan]      - Show available commands
  [cyan]clear[/cyan]     - Clear log viewer
  [cyan]refresh[/cyan]   - Refresh strategy data
  [cyan]status[/cyan]    - Show system status
  [cyan]strategies[/cyan]- List all strategies
  [cyan]quit[/cyan]      - Exit application

[yellow]Pipeline Stages:[/yellow]
  📥 [cyan]Inbox[/cyan]        - Pending strategies
  ⚙️ [cyan]Processing[/cyan]   - Being processed by TRD Agent
  📤 [cyan]Extracting[/cyan]   - Ready for EA extraction
  ✅ [cyan]Done[/cyan]         - Primal/complete strategies
        """

        if self._log_viewer:
            self._log_viewer.info(help_text.strip(), source="Help")

    # Command input message handlers

    def on_command_input_command_executed(self, message) -> None:
        """Handle command execution."""
        if self._log_viewer:
            self._log_viewer.info(
                f"Command executed: {message.command}",
                source="Command"
            )
            if message.result:
                self._log_viewer.info(message.result, source="Command")

    def on_command_input_command_failed(self, message) -> None:
        """Handle command failure."""
        if self._log_viewer:
            self._log_viewer.error(
                f"Command failed: {message.command} - {message.error}",
                source="Command"
            )

    def on_command_input_command_clear(self, message) -> None:
        """Handle clear command."""
        if self._log_viewer:
            self._log_viewer.clear_logs()

    def on_command_input_command_refresh(self, message) -> None:
        """Handle refresh command."""
        self.action_refresh()

    def on_command_input_command_completion(self, message) -> None:
        """Handle tab completion results."""
        if self._log_viewer and message.matches:
            self._log_viewer.debug(
                f"Completions: {', '.join(message.matches)}",
                source="Command"
            )

    # Public API for command handlers

    def get_system_status(self) -> str:
        """Get system status for status command."""
        uptime = datetime.now() - self._startup_time
        minutes, seconds = divmod(int(uptime.total_seconds()), 60)

        status = [
            "[bold]System Status[/bold]",
            f"  Uptime: {minutes}m {seconds}s",
            f"  API: {self._api_base_url}",
        ]

        if self._kanban_board:
            strategies = self._kanban_board.strategies
            status.append(f"  Strategies: {len(strategies)}")

        if self._log_viewer:
            logs = len(self._log_viewer.logs)
            status.append(f"  Log entries: {logs}")

        return "\n".join(status)

    def get_strategies_status(self) -> str:
        """Get strategies status for strategies command."""
        if not self._kanban_board:
            return "Kanban board not available"

        lines = ["[bold]Strategies:[/bold]"]
        for strategy in self._kanban_board.strategies:
            status_symbol = {
                "pending": "📥",
                "processing": "⚙️",
                "ready": "📤",
                "primal": "✅",
                "quarantined": "⚠️",
            }.get(strategy.status.value, "?")

            badges = []
            if strategy.has_nprd:
                badges.append("NPRD")
            if strategy.has_trd:
                badges.append("TRD")
            if strategy.has_ea:
                badges.append("EA")
            if strategy.has_backtest:
                badges.append("BT")

            badge_str = f" [{','.join(badges)}]" if badges else ""
            lines.append(f"  {status_symbol} {strategy.name}{badge_str}")

        return "\n".join(lines) if len(self._kanban_board.strategies) > 0 else "  No strategies found"

    def get_logs_status(self) -> str:
        """Get logs status for logs command."""
        if not self._log_viewer:
            return "Log viewer not available"

        counts = self._log_viewer.log_counts
        lines = ["[bold]Log Summary:[/bold]"]
        for level, count in counts.items():
            if count > 0:
                lines.append(f"  {level.value.upper()}: {count}")

        return "\n".join(lines) if any(counts.values()) else "  No log entries"


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point for the YouTube-EA TUI dashboard."""
    import argparse

    parser = argparse.ArgumentParser(
        description="YouTube-EA Pipeline TUI Dashboard"
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="Backend API URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )

    args = parser.parse_args()

    # Get API URL from environment if not specified
    api_url = args.api_url or os.getenv(
        "QUANTMIND_API_URL",
        "http://localhost:8000"
    )

    # Create and run the app
    app = YouTubeEADashboard(api_base_url=api_url)

    if args.debug:
        log("Debug mode enabled")

    app.run()


if __name__ == "__main__":
    main()

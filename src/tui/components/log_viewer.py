"""
LogViewer Component - Real-time Log Display

Displays logs with filtering, search, auto-scroll, and export functionality.
"""

import time
from datetime import datetime
from dataclasses import dataclass
from typing import List, Set, Optional
from enum import Enum

from textual.widget import Widget
from textual.reactive import reactive
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from rich.console import Group
from rich import box


class LogLevel(str, Enum):
    """Log levels for filtering and display."""
    DEBUG = "debug"
    INFO = "info"
    WARN = "warn"
    ERROR = "error"
    SUCCESS = "success"


@dataclass
class LogEntry:
    """A single log entry."""
    id: str
    timestamp: datetime
    level: LogLevel
    message: str
    source: Optional[str] = None

    def __post_init__(self):
        """Generate ID if not provided."""
        if not self.id:
            self.id = f"log-{int(time.time() * 1000)}-{id(self)}"


class LogViewer(Widget):
    """Log viewer widget with filtering and search capabilities.

    Features:
    - Log level filtering (debug, info, warn, error, success)
    - Search functionality
    - Auto-scroll to latest
    - Pause/resume scrolling
    - Export logs
    - Clear logs
    """

    DEFAULT_CSS = """
    LogViewer {
        height: 1fr;
    }

    LogViewer > Vertical {
        height: 100%;
    }

    LogViewer .log-toolbar {
        height: 3;
        dock: top;
    }

    LogViewer .log-container {
        height: 1fr;
        overflow-y: auto;
    }

    LogViewer .log-statusbar {
        height: 1;
        dock: bottom;
    }

    LogViewer .filter-btn {
        width: auto;
    }

    LogViewer .search-input {
        width: 1fr;
    }

    LogViewer .log-entry-debug {
        text-style: dim;
    }

    LogViewer .log-entry-info {
        color: $primary;
    }

    LogViewer .log-entry-warn {
        color: $warning;
        text-style: bold;
    }

    LogViewer .log-entry-error {
        color: $error;
        text-style: bold;
        background: $error 10%;
    }

    LogViewer .log-entry-success {
        color: $success;
        text-style: bold;
    }
    """

    # Reactive state
    logs: reactive[List[LogEntry]] = reactive(list)
    active_filters: reactive[Set[LogLevel]] = reactive(set)
    search_term: reactive[str] = reactive("")
    auto_scroll: reactive[bool] = reactive(True)
    paused: reactive[bool] = reactive(False)
    max_logs: reactive[int] = reactive(1000)

    def __init__(
        self,
        id: str | None = None,
        max_logs: int = 1000,
        auto_scroll: bool = True
    ):
        """Initialize the LogViewer widget.

        Args:
            id: Widget ID
            max_logs: Maximum number of logs to keep in memory
            auto_scroll: Whether to auto-scroll to new logs
        """
        super().__init__(id=id)
        self.max_logs = max_logs
        self._auto_scroll_enabled = auto_scroll
        self.active_filters = set(LogLevel)

    def on_mount(self) -> None:
        """Initialize on mount."""
        # Add initial welcome message
        self.add_log(LogLevel.INFO, "LogViewer initialized", "LogViewer")

    def add_log(self, level: LogLevel, message: str, source: Optional[str] = None) -> None:
        """Add a new log entry.

        Args:
            level: Log level
            message: Log message
            source: Optional source identifier
        """
        entry = LogEntry(
            id=f"log-{int(time.time() * 1000)}",
            timestamp=datetime.now(),
            level=level,
            message=message,
            source=source
        )

        # Keep only max_logs entries
        self.logs = self.logs[-self.max_logs + 1:] + [entry]

    # Convenience methods for adding logs
    def debug(self, message: str, source: Optional[str] = None) -> None:
        """Add a debug log entry."""
        self.add_log(LogLevel.DEBUG, message, source)

    def info(self, message: str, source: Optional[str] = None) -> None:
        """Add an info log entry."""
        self.add_log(LogLevel.INFO, message, source)

    def warn(self, message: str, source: Optional[str] = None) -> None:
        """Add a warning log entry."""
        self.add_log(LogLevel.WARN, message, source)

    def error(self, message: str, source: Optional[str] = None) -> None:
        """Add an error log entry."""
        self.add_log(LogLevel.ERROR, message, source)

    def success(self, message: str, source: Optional[str] = None) -> None:
        """Add a success log entry."""
        self.add_log(LogLevel.SUCCESS, message, source)

    def clear_logs(self) -> None:
        """Clear all log entries."""
        self.logs = []

    @property
    def filtered_logs(self) -> List[LogEntry]:
        """Get logs filtered by active filters and search term."""
        filtered = []

        for log_entry in self.logs:
            # Check level filter
            if log_entry.level not in self.active_filters:
                continue

            # Check search filter
            if self.search_term:
                search_lower = self.search_term.lower()
                if (search_lower not in log_entry.message.lower() and
                    (not log_entry.source or search_lower not in log_entry.source.lower())):
                    continue

            filtered.append(log_entry)

        return filtered

    @property
    def log_counts(self) -> dict[LogLevel, int]:
        """Get count of logs by level."""
        counts = {level: 0 for level in LogLevel}
        for log_entry in self.logs:
            counts[log_entry.level] += 1
        return counts

    def toggle_filter(self, level: LogLevel) -> None:
        """Toggle a log level filter on/off.

        Args:
            level: Log level to toggle
        """
        if level in self.active_filters:
            # Don't allow unchecking the last filter
            if len(self.active_filters) > 1:
                self.active_filters.remove(level)
        else:
            self.active_filters.add(level)

    def toggle_pause(self) -> None:
        """Toggle auto-scroll pause state."""
        self.paused = not self.paused

    def export_logs(self) -> str:
        """Export logs as text string.

        Returns:
            Log text in standard format
        """
        lines = []
        for log_entry in self.logs:
            timestamp = log_entry.timestamp.isoformat(timespec="milliseconds")
            source_part = f" [{log_entry.source}]" if log_entry.source else ""
            lines.append(f"[{timestamp}] [{log_entry.level.value.upper()}]{source_part} {log_entry.message}")
        return "\n".join(lines)

    def _format_timestamp(self, dt: datetime) -> str:
        """Format timestamp for display."""
        return dt.strftime("%H:%M:%S.%f")[:-3]

    def _get_level_color(self, level: LogLevel) -> str:
        """Get rich color for a log level."""
        colors = {
            LogLevel.DEBUG: "dim",
            LogLevel.INFO: "blue",
            LogLevel.WARN: "yellow",
            LogLevel.ERROR: "red bold",
            LogLevel.SUCCESS: "green bold"
        }
        return colors.get(level, "white")

    def _render_toolbar(self) -> Table:
        """Render the toolbar with filters and search."""
        toolbar = Table.grid(padding=(0, 1), expand=True)
        toolbar.add_column(justify="left")
        toolbar.add_column(justify="right")

        # Left side: Filter indicators
        counts = self.log_counts
        filter_parts = []
        for level in LogLevel:
            if level in self.active_filters:
                filter_parts.append(f"[{self._get_level_color(level)}]{level.value.upper()}:{counts[level]}[/]")

        filter_text = " | ".join(filter_parts) if filter_parts else "No filters"
        toolbar.add_row(f"Filter: {filter_text}", f"Logs: {len(self.logs)}")

        if self.search_term:
            toolbar.add_row(f"Search: '{self.search_term}'", f"Showing: {len(self.filtered_logs)}")

        return toolbar

    def _render_log_entry(self, entry: LogEntry) -> Text:
        """Render a single log entry."""
        text = Text()

        # Timestamp
        text.append(self._format_timestamp(entry.timestamp), style="dim")
        text.append(" ")

        # Level
        level_text = f"[{entry.level.value.upper()}]"
        text.append(level_text, style=self._get_level_color(entry.level))
        text.append(" ")

        # Source
        if entry.source:
            text.append(f"{entry.source}", style="cyan")
            text.append(" ")

        # Message
        text.append(entry.message)

        return text

    def render(self) -> Panel:
        """Render the log viewer."""
        # Build log entries
        log_lines = []
        for entry in self.filtered_logs:
            log_lines.append(self._render_log_entry(entry))

        if not log_lines:
            no_logs = Text("No logs to display", style="dim")
            if self.logs and not self.filtered_logs:
                no_logs = Text("No logs match current filters", style="dim italic")
            log_lines = [no_logs]

        content = Group(
            self._render_toolbar(),
            "",
            *log_lines
        )

        title = "📋 Log Viewer"
        if self.paused:
            title += " [dim](paused)[/]"
        if self.search_term:
            title += f" [dim]search: '{self.search_term}'[/]"

        return Panel(
            content,
            title=title,
            border_style="blue",
            padding=(0, 1),
            box=box.ROUNDED
        )

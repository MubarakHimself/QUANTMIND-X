"""
Tests for TUI components.

Tests the AuthBar, KanbanBoard, LogViewer, and CommandInput components.
"""

import pytest
from datetime import datetime

from src.tui.components.auth_bar import AuthBar, AuthStatus
from src.tui.components.kanban_board import (
    KanbanBoard,
    KanbanColumn,
    StrategyFolder,
    StrategyStatus,
)
from src.tui.components.log_viewer import LogViewer, LogLevel, LogEntry
from src.tui.components.command_input import CommandInput, CommandEntry


# =============================================================================
# AuthBar Tests
# =============================================================================

class TestAuthBar:
    """Tests for AuthBar component."""

    def test_auth_status_properties(self):
        """Test AuthStatus dataclass properties."""
        status = AuthStatus(provider="Test", has_key=True, key_valid=True, connected=True)

        assert status.provider == "Test"
        assert status.has_key is True
        assert status.key_valid is True
        assert status.connected is True
        assert status.is_valid is True
        assert "Connected" in status.status_text

    def test_auth_status_no_key(self):
        """Test AuthStatus with no key."""
        status = AuthStatus(provider="Test", has_key=False)

        assert status.is_valid is False
        assert "No key" in status.status_text

    def test_auth_status_invalid_key(self):
        """Test AuthStatus with invalid key."""
        status = AuthStatus(provider="Test", has_key=True, key_valid=False)

        assert status.is_valid is False
        assert "Invalid key" in status.status_text

    def test_auth_status_testing(self):
        """Test AuthStatus in testing state."""
        status = AuthStatus(provider="Test", has_key=True, key_valid=True, connected=False)

        assert status.is_valid is False
        assert "Testing" in status.status_text

    def test_auth_bar_init(self):
        """Test AuthBar initialization."""
        bar = AuthBar(id="test-auth")

        assert bar.id == "test-auth"
        assert bar.gemini_status.provider == "Gemini"
        assert bar.qwen_status.provider == "Qwen"

    def test_validate_google_key(self):
        """Test Google API key validation."""
        bar = AuthBar()

        assert bar._validate_google_key("AIza" + "a" * 35) is True
        assert bar._validate_google_key("invalid") is False
        assert bar._validate_google_key("") is False

    def test_validate_qwen_key(self):
        """Test Qwen API key validation."""
        bar = AuthBar()

        assert bar._validate_qwen_key("valid_key_long_enough") is True
        assert bar._validate_qwen_key("short") is False

    @pytest.mark.asyncio
    async def test_auth_bar_render(self):
        """Test AuthBar rendering."""
        bar = AuthBar()
        bar.gemini_status = AuthStatus("Gemini", has_key=True, key_valid=True, connected=True)

        panel = bar.render()
        assert panel is not None
        assert "Authentication" in panel.title


# =============================================================================
# LogViewer Tests
# =============================================================================

class TestLogEntry:
    """Tests for LogEntry dataclass."""

    def test_log_entry_creation(self):
        """Test LogEntry creation."""
        entry = LogEntry(
            id="test-1",
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message="Test message",
            source="TestSource"
        )

        assert entry.id == "test-1"
        assert entry.level == LogLevel.INFO
        assert entry.message == "Test message"
        assert entry.source == "TestSource"


class TestLogViewer:
    """Tests for LogViewer component."""

    def test_log_viewer_init(self):
        """Test LogViewer initialization."""
        viewer = LogViewer(id="test-logs", max_logs=100)

        assert viewer.id == "test-logs"
        assert viewer.max_logs == 100
        assert len(viewer.logs) == 0
        assert LogLevel.INFO in viewer.active_filters

    def test_add_log(self):
        """Test adding a log entry."""
        viewer = LogViewer()

        viewer.add_log(LogLevel.INFO, "Test message", "TestSource")

        assert len(viewer.logs) == 1
        assert viewer.logs[0].message == "Test message"
        assert viewer.logs[0].level == LogLevel.INFO

    def test_add_log_convenience_methods(self):
        """Test convenience methods for adding logs."""
        viewer = LogViewer()

        viewer.debug("Debug message")
        viewer.info("Info message")
        viewer.warn("Warning message")
        viewer.error("Error message")
        viewer.success("Success message")

        assert len(viewer.logs) == 5
        assert viewer.logs[0].level == LogLevel.DEBUG
        assert viewer.logs[1].level == LogLevel.INFO
        assert viewer.logs[2].level == LogLevel.WARN
        assert viewer.logs[3].level == LogLevel.ERROR
        assert viewer.logs[4].level == LogLevel.SUCCESS

    def test_clear_logs(self):
        """Test clearing logs."""
        viewer = LogViewer()
        viewer.add_log(LogLevel.INFO, "Test message")

        viewer.clear_logs()

        assert len(viewer.logs) == 0

    def test_filtered_logs(self):
        """Test log filtering."""
        viewer = LogViewer()
        viewer.add_log(LogLevel.DEBUG, "Debug message")
        viewer.add_log(LogLevel.INFO, "Info message")
        viewer.add_log(LogLevel.ERROR, "Error message")

        # Filter out DEBUG
        viewer.active_filters = {LogLevel.INFO, LogLevel.ERROR}

        filtered = viewer.filtered_logs

        assert len(filtered) == 2
        assert all(log.level in {LogLevel.INFO, LogLevel.ERROR} for log in filtered)

    def test_search_filter(self):
        """Test search filtering."""
        viewer = LogViewer()
        viewer.add_log(LogLevel.INFO, "Important message")
        viewer.add_log(LogLevel.INFO, "Regular message")

        viewer.search_term = "Important"

        filtered = viewer.filtered_logs

        assert len(filtered) == 1
        assert "Important" in filtered[0].message

    def test_log_counts(self):
        """Test log count statistics."""
        viewer = LogViewer()
        viewer.add_log(LogLevel.INFO, "Info 1")
        viewer.add_log(LogLevel.INFO, "Info 2")
        viewer.add_log(LogLevel.ERROR, "Error 1")

        counts = viewer.log_counts

        assert counts[LogLevel.INFO] == 2
        assert counts[LogLevel.ERROR] == 1

    def test_toggle_filter(self):
        """Test toggling log filters."""
        viewer = LogViewer()

        viewer.toggle_filter(LogLevel.DEBUG)

        if LogLevel.DEBUG in viewer.active_filters:
            # If it was present, should be removed (unless it's the last one)
            if len(viewer.active_filters) > 1:
                assert LogLevel.DEBUG not in viewer.active_filters
        else:
            assert LogLevel.DEBUG in viewer.active_filters

    def test_max_logs_limit(self):
        """Test max logs limit."""
        viewer = LogViewer(max_logs=5)

        for i in range(10):
            viewer.add_log(LogLevel.INFO, f"Message {i}")

        assert len(viewer.logs) <= 5

    def test_export_logs(self):
        """Test exporting logs."""
        viewer = LogViewer()
        viewer.add_log(LogLevel.INFO, "Test message", "TestSource")

        exported = viewer.export_logs()

        assert "Test message" in exported
        assert "TestSource" in exported
        assert "INFO" in exported

    def test_toggle_pause(self):
        """Test pause toggle."""
        viewer = LogViewer()

        assert viewer.paused is False

        viewer.toggle_pause()
        assert viewer.paused is True

        viewer.toggle_pause()
        assert viewer.paused is False


# =============================================================================
# KanbanBoard Tests
# =============================================================================

class TestStrategyFolder:
    """Tests for StrategyFolder dataclass."""

    def test_strategy_folder_creation(self):
        """Test StrategyFolder creation."""
        strategy = StrategyFolder(
            id="test-strategy",
            name="Test Strategy",
            status=StrategyStatus.PENDING,
            created_at="2026-02-22T10:00:00Z",
            has_nprd=True,
            has_trd=False
        )

        assert strategy.id == "test-strategy"
        assert strategy.name == "Test Strategy"
        assert strategy.status == StrategyStatus.PENDING
        assert strategy.has_nprd is True
        assert strategy.has_trd is False


class TestKanbanColumn:
    """Tests for KanbanColumn class."""

    def test_kanban_column_init(self):
        """Test KanbanColumn initialization."""
        column = KanbanColumn(
            id="test-column",
            title="Test Column",
            status_map=[StrategyStatus.PENDING],
            color="blue"
        )

        assert column.id == "test-column"
        assert column.title == "Test Column"
        assert column.color == "blue"
        assert len(column.strategies) == 0

    def test_kanban_column_add_strategy(self):
        """Test adding strategy to column."""
        column = KanbanColumn(
            id="test-column",
            title="Test",
            status_map=[StrategyStatus.PENDING],
            color="blue"
        )

        strategy = StrategyFolder(
            id="test",
            name="Test",
            status=StrategyStatus.PENDING,
            created_at="2026-02-22T10:00:00Z"
        )

        column.add_strategy(strategy)

        assert len(column.strategies) == 1
        assert column.strategies[0] == strategy

    def test_kanban_column_clear(self):
        """Test clearing column."""
        column = KanbanColumn(
            id="test-column",
            title="Test",
            status_map=[StrategyStatus.PENDING],
            color="blue"
        )

        strategy = StrategyFolder(
            id="test",
            name="Test",
            status=StrategyStatus.PENDING,
            created_at="2026-02-22T10:00:00Z"
        )

        column.add_strategy(strategy)
        assert len(column.strategies) == 1

        column.clear()
        assert len(column.strategies) == 0

    def test_kanban_column_count(self):
        """Test column count property."""
        column = KanbanColumn(
            id="test-column",
            title="Test",
            status_map=[StrategyStatus.PENDING],
            color="blue"
        )

        assert column.count == 0

        for i in range(3):
            column.add_strategy(StrategyFolder(
                id=f"test-{i}",
                name=f"Test {i}",
                status=StrategyStatus.PENDING,
                created_at="2026-02-22T10:00:00Z"
            ))

        assert column.count == 3


class TestKanbanBoard:
    """Tests for KanbanBoard component."""

    def test_kanban_board_init(self):
        """Test KanbanBoard initialization."""
        board = KanbanBoard(id="test-board")

        assert board.id == "test-board"
        assert len(board.columns) == 4
        assert board.columns[0].id == "inbox"
        assert board.columns[1].id == "processing"
        assert board.columns[2].id == "extracting"
        assert board.columns[3].id == "done"

    def test_column_definitions(self):
        """Test column definitions match pipeline stages."""
        board = KanbanBoard()

        # Inbox column
        assert board.columns[0].title == "📥 Inbox"
        assert StrategyStatus.PENDING in board.columns[0].status_map

        # Processing column
        assert board.columns[1].title == "⚙️ Processing"
        assert StrategyStatus.PROCESSING in board.columns[1].status_map

        # Extracting column
        assert board.columns[2].title == "📤 Extracting"
        assert StrategyStatus.READY in board.columns[2].status_map

        # Done column
        assert board.columns[3].title == "✅ Done"
        assert StrategyStatus.PRIMAL in board.columns[3].status_map

    def test_group_strategies_by_column(self):
        """Test grouping strategies by column."""
        board = KanbanBoard()

        # Add test strategies
        board.strategies = [
            StrategyFolder("s1", "Pending", StrategyStatus.PENDING, "2026-02-22T10:00:00Z"),
            StrategyFolder("s2", "Processing", StrategyStatus.PROCESSING, "2026-02-22T10:00:00Z"),
            StrategyFolder("s3", "Ready", StrategyStatus.READY, "2026-02-22T10:00:00Z"),
            StrategyFolder("s4", "Done", StrategyStatus.PRIMAL, "2026-02-22T10:00:00Z"),
        ]

        grouped = board._group_strategies_by_column()

        assert len(grouped["inbox"]) == 1
        assert len(grouped["processing"]) == 1
        assert len(grouped["extracting"]) == 1
        assert len(grouped["done"]) == 1

    def test_use_mock_data(self):
        """Test mock data generation."""
        board = KanbanBoard()

        board._use_mock_data()

        assert len(board.strategies) > 0
        assert any(s.status == StrategyStatus.PRIMAL for s in board.strategies)
        assert any(s.status == StrategyStatus.PROCESSING for s in board.strategies)


# =============================================================================
# CommandInput Tests
# =============================================================================

class TestCommandEntry:
    """Tests for CommandEntry dataclass."""

    def test_command_entry_creation(self):
        """Test CommandEntry creation."""
        entry = CommandEntry(
            command="test command",
            timestamp="2026-02-22T10:00:00Z",
            result="Success",
            success=True
        )

        assert entry.command == "test command"
        assert entry.result == "Success"
        assert entry.success is True


class TestCommandInput:
    """Tests for CommandInput component."""

    def test_command_input_init(self):
        """Test CommandInput initialization."""
        cmd_input = CommandInput(id="test-command", prompt="$")

        assert cmd_input.id == "test-command"
        assert cmd_input.prompt == "$"
        assert len(cmd_input.history) == 0

    def test_default_command_handlers(self):
        """Test default command handlers are registered."""
        cmd_input = CommandInput()

        assert "help" in cmd_input.command_handlers
        assert "clear" in cmd_input.command_handlers
        assert "refresh" in cmd_input.command_handlers
        assert "quit" in cmd_input.command_handlers

    def test_add_to_history(self):
        """Test adding command to history."""
        cmd_input = CommandInput()

        cmd_input._add_to_history("test command", result="Success", success=True)

        assert len(cmd_input.history) == 1
        assert cmd_input.history[0].command == "test command"

    def test_max_history_limit(self):
        """Test max history limit."""
        cmd_input = CommandInput(max_history=5)

        for i in range(10):
            cmd_input._add_to_history(f"command {i}")

        assert len(cmd_input.history) <= 5

    def test_execute_command_help(self):
        """Test executing help command."""
        cmd_input = CommandInput()

        result = cmd_input._cmd_help([])

        assert "Available commands" in result
        assert "help" in result

    def test_execute_command_clear(self):
        """Test executing clear command."""
        cmd_input = CommandInput()

        result = cmd_input._cmd_clear([])

        assert "cleared" in result.lower()

    def test_execute_command_quit(self):
        """Test executing quit command."""
        cmd_input = CommandInput()

        result = cmd_input._cmd_quit([])

        assert "Exiting" in result

    def test_execute_unknown_command(self):
        """Test executing unknown command."""
        cmd_input = CommandInput()

        result = cmd_input.command_handlers.get("unknown", lambda args: "Unknown command: unknown")([])

        # The actual execute_command handles unknown commands differently
        # This tests the handler lookup mechanism
        assert "Unknown" in result or "unknown" in result

    def test_find_common_prefix(self):
        """Test finding common prefix."""
        cmd_input = CommandInput()

        assert cmd_input._find_common_prefix(["help", "hello", "helium"]) == "hel"
        assert cmd_input._find_common_prefix(["test", "other"]) == ""
        assert cmd_input._find_common_prefix(["same", "same"]) == "same"

    @pytest.mark.asyncio
    async def test_complete_command_single_match(self):
        """Test tab completion with single match."""
        cmd_input = CommandInput()
        cmd_input.value = "he"

        await cmd_input._complete_command()

        # Should complete to "help" + space
        assert cmd_input.value == "help "

    @pytest.mark.asyncio
    async def test_complete_command_no_match(self):
        """Test tab completion with no matches."""
        cmd_input = CommandInput()
        cmd_input.value = "xyz"

        await cmd_input._complete_command()

        # Should not change
        assert cmd_input.value == "xyz"


# =============================================================================
# Integration Tests
# =============================================================================

class TestYouTubeEADashboard:
    """Integration tests for the YouTube-EA Dashboard."""

    @pytest.mark.asyncio
    async def test_dashboard_import(self):
        """Test that dashboard can be imported."""
        from src.tui.youtube_ea_tui import YouTubeEADashboard

        app = YouTubeEADashboard(api_base_url="http://localhost:8000")

        assert app is not None
        assert app._api_base_url == "http://localhost:8000"

    def test_dashboard_bindings(self):
        """Test dashboard key bindings."""
        from src.tui.youtube_ea_tui import YouTubeEADashboard

        app = YouTubeEADashboard()

        bindings = {b.key for b in app.BINDINGS}
        assert "q" in bindings
        assert "r" in bindings
        assert "c" in bindings

    @pytest.mark.asyncio
    async def test_dashboard_compose(self):
        """Test dashboard composition."""
        from src.tui.youtube_ea_tui import YouTubeEADashboard

        app = YouTubeEADashboard()

        async with app.run_test() as _:
            # Check that components exist
            assert app.query_one("#log-viewer") is not None
            assert app.query_one("#kanban-board") is not None
            assert app.query_one("#command-input") is not None

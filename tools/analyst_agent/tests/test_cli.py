"""
Test suite for Analyst Agent CLI commands.

Tests:
- generate command
- list command
- complete command
- CLI error handling
- CLI argument parsing
- CLI output formatting
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import typer.testing
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Import the CLI module
from tools.analyst_agent.cli.commands import app


@pytest.fixture
def runner():
    """CLI test runner."""
    return typer.testing.CliRunner()


@pytest.fixture
def mock_console():
    """Mock rich console for testing."""
    with patch('tools.analyst_agent.cli.commands.Console') as mock_console_class:
        mock_console_instance = Mock()
        mock_console_class.return_value = mock_console_instance
        yield mock_console_instance


def test_cli_generate_command_basic(runner, mock_console):
    """Test generate command basic functionality."""
    # Mock file existence check
    with patch('pathlib.Path.exists', return_value=True):

        result = runner.invoke(app, ["generate", "test_nprd.json"])

        assert result.exit_code == 0
        mock_console.print.assert_called()


def test_cli_generate_command_with_auto_mode(runner, mock_console):
    """Test generate command with auto mode."""
    with patch('pathlib.Path.exists', return_value=True):

        result = runner.invoke(app, ["generate", "test_nprd.json", "--auto"])

        assert result.exit_code == 0
        mock_console.print.assert_called()


def test_cli_generate_command_with_output_dir(runner, mock_console):
    """Test generate command with output directory."""
    with patch('pathlib.Path.exists', return_value=True):

        result = runner.invoke(app, ["generate", "test_nprd.json", "--output", "/tmp/output"])

        assert result.exit_code == 0
        mock_console.print.assert_called()


def test_cli_generate_command_file_not_found(runner, mock_console):
    """Test generate command with non-existent file."""
    with patch('pathlib.Path.exists', return_value=False):

        result = runner.invoke(app, ["generate", "nonexistent.json"])

        assert result.exit_code == 1
        mock_console.print.assert_called()


def test_cli_list_command_nprd_files(runner, mock_console):
    """Test list command for NPRD files."""
    result = runner.invoke(app, ["list", "--nprd"])

    assert result.exit_code == 0
    mock_console.print.assert_called()


def test_cli_list_command_trd_files(runner, mock_console):
    """Test list command for TRD files."""
    result = runner.invoke(app, ["list", "--trds"])

    assert result.exit_code == 0
    mock_console.print.assert_called()


def test_cli_list_command_no_option(runner, mock_console):
    """Test list command without specifying --nprd or --trds."""
    result = runner.invoke(app, ["list"])

    assert result.exit_code == 1
    mock_console.print.assert_called()


def test_cli_stats_command(runner, mock_console):
    """Test stats command."""
    result = runner.invoke(app, ["stats"])

    assert result.exit_code == 0
    mock_console.print.assert_called()


def test_cli_complete_command(runner, mock_console):
    """Test complete command."""
    with patch('pathlib.Path.exists', return_value=True):

        result = runner.invoke(app, ["complete", "test_trd.json"])

        assert result.exit_code == 0
        mock_console.print.assert_called()


def test_cli_complete_command_file_not_found(runner, mock_console):
    """Test complete command with non-existent file."""
    with patch('pathlib.Path.exists', return_value=False):

        result = runner.invoke(app, ["complete", "nonexistent.json"])

        assert result.exit_code == 1
        mock_console.print.assert_called()


def test_cli_config_command_list_all(runner, mock_console):
    """Test config command with --list."""
    result = runner.invoke(app, ["config", "--list"])

    assert result.exit_code == 0
    mock_console.print.assert_called()


def test_cli_config_command_get_key(runner, mock_console):
    """Test config command with --get."""
    result = runner.invoke(app, ["config", "--get", "auto_save"])

    assert result.exit_code == 0
    mock_console.print.assert_called()


def test_cli_config_command_set_key(runner, mock_console):
    """Test config command with --set."""
    result = runner.invoke(app, ["config", "--set", "auto_save=true"])

    assert result.exit_code == 0
    mock_console.print.assert_called()


def test_cli_config_command_no_option(runner, mock_console):
    """Test config command without specifying option."""
    result = runner.invoke(app, ["config"])

    assert result.exit_code == 1
    mock_console.print.assert_called()


def test_cli_argument_parsing(runner):
    """Test CLI argument parsing."""
    # Test required arguments
    result = runner.invoke(app, ["generate"])
    assert result.exit_code == 2  # Missing required argument

    # Test invalid arguments
    result = runner.invoke(app, ["generate", "test.json", "--invalid-option"])
    assert result.exit_code == 2  # Invalid option


def test_cli_output_formatting(runner, mock_console):
    """Test CLI output formatting."""
    with patch('pathlib.Path.exists', return_value=True):

        result = runner.invoke(app, ["generate", "test_nprd.json"])

        # Verify rich formatting was used
        mock_console.print.assert_called()
        call_args = mock_console.print.call_args
        assert isinstance(call_args[0][0], (Panel, Table))  # Should be rich Panel or Table


def test_cli_error_handling(runner, mock_console):
    """Test CLI error handling."""
    # Mock a function that raises an exception
    with patch('tools.analyst_agent.cli.commands.generate', side_effect=Exception("Test error")):

        result = runner.invoke(app, ["generate", "test_nprd.json"])

        assert result.exit_code == 1
        mock_console.print.assert_called()


def test_cli_help_output(runner):
    """Test CLI help output."""
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Analyst Agent CLI" in result.output
    assert "generate" in result.output
    assert "list" in result.output
    assert "complete" in result.output


def test_cli_command_completion(runner):
    """Test CLI command completion."""
    # Test subcommand completion
    result = runner.invoke(app, ["generate", "--help"])

    assert result.exit_code == 0
    assert "--auto" in result.output
    assert "--output" in result.output

    result = runner.invoke(app, ["list", "--help"])

    assert result.exit_code == 0
    assert "--nprd" in result.output
    assert "--trds" in result.output


def test_cli_file_path_validation(runner):
    """Test CLI file path validation."""
    # Test with invalid file paths
    result = runner.invoke(app, ["generate", "/nonexistent/path/test.json"])

    assert result.exit_code == 1

    # Test with directory instead of file
    result = runner.invoke(app, ["generate", "/tmp"])

    assert result.exit_code == 1


def test_cli_interactive_mode(runner, mock_console):
    """Test CLI interactive mode behavior."""
    with patch('pathlib.Path.exists', return_value=True):

        result = runner.invoke(app, ["generate", "test_nprd.json"])

        assert result.exit_code == 0
        mock_console.print.assert_called()


def test_cli_auto_mode_behavior(runner, mock_console):
    """Test CLI auto mode behavior."""
    with patch('pathlib.Path.exists', return_value=True):

        result = runner.invoke(app, ["generate", "test_nprd.json", "--auto"])

        assert result.exit_code == 0
        mock_console.print.assert_called()


def test_cli_progress_display(runner, mock_console):
    """Test CLI progress display."""
    with patch('pathlib.Path.exists', return_value=True):

        result = runner.invoke(app, ["generate", "test_nprd.json"])

        assert result.exit_code == 0
        # Verify progress was displayed
        mock_console.print.assert_any_call("[cyan]Converting NPRD to TRD...")


def test_cli_output_directory_creation(runner, mock_console):
    """Test CLI output directory creation."""
    with patch('pathlib.Path.exists', return_value=True), \
         patch('pathlib.Path.mkdir') as mock_mkdir:

        result = runner.invoke(app, ["generate", "test_nprd.json", "--output", "/tmp/output"])

        assert result.exit_code == 0
        mock_mkdir.assert_called()


def test_cli_config_management(runner, mock_console):
    """Test CLI configuration management."""
    # Test config list
    result = runner.invoke(app, ["config", "--list"])

    assert result.exit_code == 0
    mock_console.print.assert_called()

    # Test config get
    result = runner.invoke(app, ["config", "--get", "auto_save"])

    assert result.exit_code == 0
    mock_console.print.assert_called()

    # Test config set
    result = runner.invoke(app, ["config", "--set", "auto_save=true"])

    assert result.exit_code == 0
    mock_console.print.assert_called()


def test_cli_multiple_commands(runner, mock_console):
    """Test running multiple CLI commands sequentially."""
    # Test list command
    result1 = runner.invoke(app, ["list", "--nprd"])
    assert result1.exit_code == 0

    # Test stats command
    result2 = runner.invoke(app, ["stats"])
    assert result2.exit_code == 0

    # Test config command
    result3 = runner.invoke(app, ["config", "--list"])
    assert result3.exit_code == 0

    assert mock_console.print.call_count >= 3


def test_cli_error_messages(runner, mock_console):
    """Test CLI error messages."""
    # Test with non-existent file
    with patch('pathlib.Path.exists', return_value=False):

        result = runner.invoke(app, ["generate", "nonexistent.json"])

        assert result.exit_code == 1
        mock_console.print.assert_called()
        call_args = mock_console.print.call_args
        assert "[red]Error: NPRD file not found[/red]" in str(call_args[0][0])


def test_cli_success_messages(runner, mock_console):
    """Test CLI success messages."""
    with patch('pathlib.Path.exists', return_value=True):

        result = runner.invoke(app, ["generate", "test_nprd.json"])

        assert result.exit_code == 0
        mock_console.print.assert_called()
        call_args = mock_console.print.call_args
        assert "[green]✓" in str(call_args[0][0])  # Success indicator


def test_cli_warning_messages(runner, mock_console):
    """Test CLI warning messages."""
    # Test list command without options
    result = runner.invoke(app, ["list"])

    assert result.exit_code == 1
    mock_console.print.assert_called()
    call_args = mock_console.print.call_args
    assert "[yellow]Please specify either --nprd or --trds[/yellow]" in str(call_args[0][0])


def test_cli_info_messages(runner, mock_console):
    """Test CLI info messages."""
    with patch('pathlib.Path.exists', return_value=True):

        result = runner.invoke(app, ["generate", "test_nprd.json"])

        assert result.exit_code == 0
        mock_console.print.assert_called()
        call_args = mock_console.print.call_args
        assert "[blue]" in str(call_args[0][0])  # Info message color


def test_cli_table_formatting(runner, mock_console):
    """Test CLI table formatting."""
    result = runner.invoke(app, ["list", "--nprd"])

    assert result.exit_code == 0
    mock_console.print.assert_called()
    call_args = mock_console.print.call_args
    assert isinstance(call_args[0][0], Table)  # Should be a rich Table


def test_cli_panel_formatting(runner, mock_console):
    """Test CLI panel formatting."""
    result = runner.invoke(app, ["stats"])

    assert result.exit_code == 0
    mock_console.print.assert_called()
    call_args = mock_console.print.call_args
    assert isinstance(call_args[0][0], Panel)  # Should be a rich Panel


def test_cli_argument_validation(runner):
    """Test CLI argument validation."""
    # Test invalid n value
    result = runner.invoke(app, ["generate", "test.json", "--auto", "--output", "/tmp"])

    assert result.exit_code == 0  # Typer doesn't validate output dir existence by default

    # Test invalid auto flag combination (though not applicable here)
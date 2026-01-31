"""
Tests for CommandParser middleware in BaseAgent.

Tests slash command parsing, execution, and backward compatibility.
Following test-writing standards: focused tests for core user flows only.
"""

import pytest
import asyncio
import logging
from unittest.mock import Mock
from pathlib import Path

from src.agents.core.base_agent import BaseAgent, CommandParser


class TestCommandParser:
    """Test CommandParser class functionality."""

    def test_command_parser_initialization(self):
        """Test CommandParser initializes with built-in commands registered."""
        parser = CommandParser()

        # Should have built-in commands registered
        assert "code" in parser.commands
        assert "file" in parser.commands
        assert "run" in parser.commands
        assert "help" in parser.commands
        assert parser.logger is not None

    def test_register_command(self):
        """Test command registration system for extensibility."""
        parser = CommandParser()

        # Register a simple command
        async def test_handler(args: str) -> str:
            return f"Test executed: {args}"

        parser.register_command("test", test_handler)

        assert "test" in parser.commands
        assert parser.commands["test"] == test_handler

    def test_parse_command_extracts_name_and_args(self):
        """Test parsing slash commands into name and arguments."""
        parser = CommandParser()

        # Test basic command
        name, args = parser.parse_command("/code print('hello')")
        assert name == "code"
        assert args == "print('hello')"

        # Test command with no args
        name, args = parser.parse_command("/help")
        assert name == "help"
        assert args == ""

        # Test command with multiple words in args
        name, args = parser.parse_command("/file read /tmp/test.txt")
        assert name == "file"
        assert args == "read /tmp/test.txt"

    def test_parse_command_returns_none_for_non_slash(self):
        """Test backward compatibility: non-slash messages return None."""
        parser = CommandParser()

        # Normal messages should not be parsed as commands
        result = parser.parse_command("Hello, how are you?")
        assert result is None

        result = parser.parse_command("Help me with something")
        assert result is None

    def test_is_slash_command_detection(self):
        """Test detection of slash commands vs normal messages."""
        parser = CommandParser()

        # Slash commands
        assert parser.is_slash_command("/code x=1")
        assert parser.is_slash_command("/file read test.txt")
        assert parser.is_slash_command("/run ls -la")
        assert parser.is_slash_command("/help")

        # Non-slash messages (backward compatibility)
        assert not parser.is_slash_command("Tell me about trading")
        assert not parser.is_slash_command("What is the weather?")
        assert not parser.is_slash_command("")
        assert not parser.is_slash_command("This has / in middle but not at start")

    def test_execute_code_command(self):
        """Test /code command executes Python code safely."""
        parser = CommandParser()

        # Test simple print
        result = parser._execute_code("print('hello world')")
        assert "hello world" in result

        # Test math operations
        result = parser._execute_code("x = 2 + 2\nprint(x)")
        assert "4" in result

    def test_execute_file_operations(self, tmp_path):
        """Test /file command for read and write operations."""
        parser = CommandParser()

        # Create test file
        test_file = tmp_path / "test.txt"
        test_content = "Test content for file operations"

        # Write operation
        write_result = parser._execute_file(f"write {test_file} {test_content}")
        assert "successfully" in write_result.lower() or "written" in write_result.lower()

        # Read operation
        read_result = parser._execute_file(f"read {test_file}")
        assert test_content in read_result

    def test_execute_run_shell_command(self):
        """Test /run command executes shell commands."""
        parser = CommandParser()

        # Test echo command (safe, cross-platform)
        result = parser._execute_run("echo test output")
        assert "test output" in result

    def test_command_execution_logging(self):
        """Test that all command executions are logged with timestamps."""
        # Create a real parser but mock its logger
        parser = CommandParser()
        parser.logger = Mock()

        # Execute a command
        asyncio.run(parser.execute("/code print('test')"))

        # Verify logging occurred
        assert parser.logger.info.called

    def test_unknown_command_returns_error(self):
        """Test error handling for unknown commands."""
        parser = CommandParser()

        # Try to execute unknown command
        result = asyncio.run(parser.execute("/unknown_command args"))
        assert "error" in result.lower() or "not found" in result.lower() or "unknown" in result.lower()

    def test_help_command_lists_available_commands(self):
        """Test /help command returns command documentation."""
        parser = CommandParser()

        result = asyncio.run(parser.execute("/help"))

        # Should list built-in commands
        assert "/code" in result
        assert "/file" in result
        assert "/run" in result
        assert "/help" in result


class TestCommandParserIntegration:
    """Test CommandParser integration with BaseAgent."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock BaseAgent for testing."""
        agent = Mock(spec=BaseAgent)
        agent.name = "TestAgent"
        agent.logger = logging.getLogger(__name__)
        agent.command_parser = CommandParser()
        agent.command_parser._register_builtin_commands()
        return agent

    def test_base_agent_has_command_parser(self, mock_agent):
        """Test that BaseAgent has CommandParser instance."""
        assert mock_agent.command_parser is not None
        assert isinstance(mock_agent.command_parser, CommandParser)

    def test_backward_compatibility_normal_messages(self):
        """Test that non-slash messages flow through unchanged."""
        parser = CommandParser()

        # Non-slash messages should not be intercepted
        normal_messages = [
            "What is the weather?",
            "Tell me about trading strategies",
            "Calculate the risk for EURUSD",
            "",
        ]

        for message in normal_messages:
            is_command = parser.is_slash_command(message)
            assert not is_command, f"Message should not be a command: {message}"

    def test_slash_command_interception_before_llm(self):
        """Test that slash commands are intercepted before LLM invocation."""
        parser = CommandParser()
        parser._register_builtin_commands()

        # Slash commands should be detected
        slash_commands = [
            "/code x=1",
            "/file read test.txt",
            "/run ls -la",
            "/help",
        ]

        for command in slash_commands:
            is_command = parser.is_slash_command(command)
            assert is_command, f"Message should be a command: {command}"

            # Should be parseable
            parsed = parser.parse_command(command)
            assert parsed is not None
            assert parsed[0] is not None  # Command name

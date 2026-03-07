"""Tests for Gemini CLI Tool."""

import pytest
from unittest.mock import patch, MagicMock
from src.agents.tools.gemini_cli_tool import GeminiCLITool, research, is_gemini_available


class TestGeminiCLITool:
    """Test suite for GeminiCLITool class."""

    def test_initialization_defaults(self):
        """Tool should initialize with correct defaults."""
        tool = GeminiCLITool()
        assert tool.model == "gemini-2.0-flash"
        assert tool.timeout == 300

    def test_initialization_custom_params(self):
        """Tool should accept custom parameters."""
        tool = GeminiCLITool(model="gemini-pro", timeout=120)
        assert tool.model == "gemini-pro"
        assert tool.timeout == 120

    def test_build_prompt_simple(self):
        """Should build simple prompt correctly."""
        tool = GeminiCLITool()
        prompt = tool._build_prompt("What is RSI?", None)
        assert prompt == "What is RSI?"

    def test_build_prompt_with_context(self):
        """Should build prompt with context."""
        tool = GeminiCLITool()
        prompt = tool._build_prompt("Explain this indicator", "It is a momentum oscillator")
        assert "Context:" in prompt
        assert "Explain this indicator" in prompt

    @patch("subprocess.run")
    def test_research_success(self, mock_run):
        """Should return successful result."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="RSI stands for Relative Strength Index",
            stderr="",
        )

        tool = GeminiCLITool()
        result = tool.research("What is RSI?")

        assert result["success"] is True
        assert result["query"] == "What is RSI?"
        assert result["model"] == "gemini-2.0-flash"

    @patch("subprocess.run")
    def test_research_with_context(self, mock_run):
        """Should pass context to Gemini."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Research result",
            stderr="",
        )

        tool = GeminiCLITool()
        result = tool.research("Explain", context="Trading strategy")

        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        # Prompt is the last element in the command list
        prompt = call_args[-1]
        assert "Context:" in prompt

    @patch("subprocess.run")
    def test_research_timeout(self, mock_run):
        """Should handle timeout errors."""
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired("cmd", 300)

        tool = GeminiCLITool()
        result = tool.research("Complex query")

        assert result["success"] is False
        assert "timed out" in result["error"].lower()

    @patch("subprocess.run")
    def test_research_cli_not_found(self, mock_run):
        """Should handle CLI not found error gracefully."""
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="Command not found: gemini",
        )

        tool = GeminiCLITool()
        result = tool.research("Query")

        # Should still return success with fallback message
        assert result["success"] is True

    @patch("subprocess.run")
    def test_research_general_error(self, mock_run):
        """Should handle general errors."""
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="API Error",
        )

        tool = GeminiCLITool()
        result = tool.research("Query")

        assert result["success"] is False
        assert "error" in result

    @patch("subprocess.run")
    def test_is_available_true(self, mock_run):
        """Should return True when CLI is available."""
        mock_run.return_value = MagicMock(returncode=0, stdout="1.0.0")

        tool = GeminiCLITool()
        assert tool.is_available() is True

    @patch("subprocess.run")
    def test_is_available_false(self, mock_run):
        """Should return False when CLI is not available."""
        mock_run.side_effect = FileNotFoundError()

        tool = GeminiCLITool()
        assert tool.is_available() is False

    @patch("subprocess.run")
    def test_get_version_success(self, mock_run):
        """Should return version info."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="gemini-cli v2.0.1",
        )

        tool = GeminiCLITool()
        result = tool.get_version()

        assert result["success"] is True
        assert "version" in result

    @patch("subprocess.run")
    def test_get_version_not_found(self, mock_run):
        """Should handle version check failure."""
        mock_run.side_effect = FileNotFoundError()

        tool = GeminiCLITool()
        result = tool.get_version()

        assert result["success"] is False
        assert "not found" in result["error"].lower()


class TestModuleFunctions:
    """Test module-level convenience functions."""

    @patch("src.agents.tools.gemini_cli_tool.GeminiCLITool")
    def test_research_function(self, mock_tool_class):
        """research function should create tool and call research."""
        mock_instance = MagicMock()
        mock_instance.research.return_value = {"success": True}
        mock_tool_class.return_value = mock_instance

        result = research("test query")

        mock_tool_class.assert_called_once()
        mock_instance.research.assert_called_once_with("test query", None)
        assert result["success"] is True

    @patch("src.agents.tools.gemini_cli_tool.GeminiCLITool")
    def test_is_gemini_available_function(self, mock_tool_class):
        """is_gemini_available should check availability."""
        mock_instance = MagicMock()
        mock_instance.is_available.return_value = True
        mock_tool_class.return_value = mock_instance

        result = is_gemini_available()

        assert result is True
        mock_instance.is_available.assert_called_once()


class TestEdgeCases:
    """Test edge cases and error handling."""

    @patch("subprocess.run")
    def test_empty_query(self, mock_run):
        """Should handle empty query."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Result",
            stderr="",
        )

        tool = GeminiCLITool()
        result = tool.research("")

        assert result["success"] is True

    @patch("subprocess.run")
    def test_model_parameter(self, mock_run):
        """Should pass model parameter to CLI."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Result",
            stderr="",
        )

        tool = GeminiCLITool(model="gemini-pro")
        tool.research("Query")

        call_args = mock_run.call_args[0][0]
        assert "-m" in call_args
        assert "gemini-pro" in call_args

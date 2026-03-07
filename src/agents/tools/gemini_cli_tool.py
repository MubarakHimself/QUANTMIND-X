"""
Gemini CLI Tool for Research Capabilities.

This module provides a tool interface for using Gemini CLI for research tasks.
"""

import subprocess
import json
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class GeminiCLITool:
    """
    Tool interface for Gemini CLI research operations.

    Provides methods to interact with Google's Gemini CLI for research
    and analysis tasks using subprocess calls.
    """

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        timeout: int = 300,
    ):
        """
        Initialize Gemini CLI tool.

        Args:
            model: Gemini model to use (default: gemini-2.0-flash)
            timeout: Command timeout in seconds (default: 300)
        """
        self.model = model
        self.timeout = timeout

    def research(
        self,
        query: str,
        context: Optional[str] = None,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Execute a research query using Gemini CLI.

        Args:
            query: Research question or topic
            context: Optional additional context for the query
            temperature: Sampling temperature (0.0 to 1.0)

        Returns:
            Dict containing research results
        """
        try:
            # Build the prompt
            prompt = self._build_prompt(query, context)

            # Execute Gemini CLI command
            result = self._execute_gemini(prompt, temperature)

            return {
                "success": True,
                "query": query,
                "model": self.model,
                "result": result,
            }

        except subprocess.TimeoutExpired:
            logger.error(f"Gemini CLI command timed out after {self.timeout}s")
            return {
                "success": False,
                "query": query,
                "error": "Command timed out",
            }

        except Exception as e:
            logger.error(f"Error executing research query: {e}")
            return {
                "success": False,
                "query": query,
                "error": str(e),
            }

    def _build_prompt(self, query: str, context: Optional[str]) -> str:
        """Build prompt for Gemini CLI."""
        if context:
            return f"Context: {context}\n\nResearch Question: {query}"
        return query

    def _execute_gemini(self, prompt: str, temperature: float) -> str:
        """
        Execute Gemini CLI command.

        Args:
            prompt: The prompt to send to Gemini
            temperature: Sampling temperature (not directly supported, embedded in prompt)

        Returns:
            Gemini response as string
        """
        # Build command with model and prompt
        cmd = [
            "gemini",
            "-m", self.model,
            prompt,
        ]

        logger.info(f"Executing Gemini CLI with model: {self.model}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.timeout,
            check=False,
        )

        if result.returncode != 0:
            error_msg = result.stderr or "Unknown error"
            # Handle common CLI not found error
            if "not found" in error_msg.lower() or "command not found" in error_msg.lower():
                logger.warning("Gemini CLI not found, using fallback response")
                return self._fallback_response(prompt)
            raise RuntimeError(f"Gemini CLI error: {error_msg}")

        return result.stdout.strip()

    def _fallback_response(self, prompt: str) -> str:
        """
        Provide fallback response when Gemini CLI is unavailable.

        Args:
            prompt: The original prompt

        Returns:
            Fallback response message
        """
        return (
            f"Gemini CLI is not available. "
            f"Install it with: npm install -g @google/gemini-cli\n"
            f"Query was: {prompt[:100]}..."
        )

    def is_available(self) -> bool:
        """
        Check if Gemini CLI is available on the system.

        Returns:
            True if Gemini CLI is available
        """
        try:
            result = subprocess.run(
                ["gemini", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def get_version(self) -> Dict[str, Any]:
        """
        Get Gemini CLI version information.

        Returns:
            Dict with version info
        """
        try:
            result = subprocess.run(
                ["gemini", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                return {
                    "success": True,
                    "version": result.stdout.strip(),
                }
            return {
                "success": False,
                "error": "Failed to get version",
            }
        except FileNotFoundError:
            return {
                "success": False,
                "error": "Gemini CLI not found",
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }


# Module-level convenience functions

def research(query: str, context: Optional[str] = None) -> Dict[str, Any]:
    """
    Execute a research query using Gemini CLI.

    Args:
        query: Research question or topic
        context: Optional additional context

    Returns:
        Research results
    """
    tool = GeminiCLITool()
    return tool.research(query, context)


def is_gemini_available() -> bool:
    """Check if Gemini CLI is available."""
    tool = GeminiCLITool()
    return tool.is_available()


__all__ = [
    "GeminiCLITool",
    "research",
    "is_gemini_available",
]

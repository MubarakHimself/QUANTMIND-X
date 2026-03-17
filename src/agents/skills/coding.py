"""
Coding Skill
Provides File System Access and Shell Execution capabilities.

NOTE: LangChain imports removed - pending migration to Anthropic Agent SDK (Epic 7).
FileManagementToolkit and ShellTool from langchain_community are replaced with stubs.
"""

import os
from typing import List, Any, Callable
from src.agents.skills.base import AgentSkill


# Stub file tools - pending migration to Anthropic Agent SDK
class _StubFileTool:
    """Stub for langchain FileManagementToolkit tools."""
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self._is_tool = True

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(
            f"File tool '{self.name}' is a stub pending migration to Anthropic Agent SDK (Epic 7)"
        )


class CodingSkill(AgentSkill):
    def __init__(self, root_dir: str = "."):
        super().__init__(
            name="CodingSkill",
            description="Ability to read/write files and execute shell commands."
        )

        # Stub file tools - previously used FileManagementToolkit
        # file_toolkit = FileManagementToolkit(...)
        # self.tools.extend(file_toolkit.get_tools())

        # Add stub file tools
        self.tools.extend([
            _StubFileTool("read_file", "Read contents of a file"),
            _StubFileTool("write_file", "Write contents to a file"),
            _StubFileTool("list_directory", "List files in a directory"),
            _StubFileTool("file_delete", "Delete a file"),
        ])

        # Shell Tool is disabled (was: ShellTool)

        self.system_prompt_addition = """
        You are an expert Software Engineer.
        - Always verify file existence before reading.
        - Write clean, documented code.
        - Use type hints.

        NOTE: File tools are currently stubs pending migration to Anthropic Agent SDK (Epic 7).
        """

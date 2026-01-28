"""
Coding Skill
Provides File System Access and Shell Execution capabilities.
"""

from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_community.tools import ShellTool
from src.agents.skills.base import AgentSkill

class CodingSkill(AgentSkill):
    def __init__(self, root_dir: str = "."):
        super().__init__(
            name="CodingSkill",
            description="Ability to read/write files and execute shell commands."
        )
        
        # 1. File Tools
        file_toolkit = FileManagementToolkit(
            root_dir=root_dir,
            selected_tools=["read_file", "write_file", "list_directory", "file_delete"]
        )
        self.tools.extend(file_toolkit.get_tools())
        
        # 2. Shell Tool (Restricted)
        # self.tools.append(ShellTool()) # Disabled for safety until verified
        
        self.system_prompt_addition = """
        You are an expert Software Engineer. 
        - Always verify file existence before reading.
        - Write clean, documented code.
        - Use type hints.
        """

"""
QuantMind Agent Skill System
Defines the structure for modular capabilities (Skills) that can be attached to Agents.
"""

from typing import List, Optional
from langchain_core.tools import BaseTool

class AgentSkill:
    """
    A Skill is a collection of Tools and System Prompt Logic.
    Example: CodingSkill contains [read_file, write_file] and a "You are a coder" prompt.
    """
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.tools: List[BaseTool] = []
        self.system_prompt_addition: str = ""

    def get_tools(self) -> List[BaseTool]:
        return self.tools

    def get_system_prompt(self) -> str:
        return self.system_prompt_addition

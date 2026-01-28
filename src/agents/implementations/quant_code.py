"""
QuantCode Agent (The Coding Specialist)
Specializes in Python and MQL5 development.
"""

from src.agents.core.base_agent import BaseAgent
from src.agents.skills.coding import CodingSkill

def create_quant_code_agent() -> BaseAgent:
    """
    Factory to create the QuantCode Agent.
    """
    return BaseAgent(
        name="QuantCode",
        role="Senior MQL5 and Python Developer",
        model_name="gpt-4-turbo-preview", # or claude-3-5-sonnet
        skills=[
            CodingSkill(root_dir=".")
        ],
        enable_long_term_memory=True,
        user_id="developer_1"
    )

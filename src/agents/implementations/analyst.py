"""
Analyst Agent (The Research Specialist)
Specializes in market research, TRD analysis, and strategy evaluation.
"""

from src.agents.core.base_agent import BaseAgent
from src.agents.skills.research import ResearchSkill

def create_analyst_agent() -> BaseAgent:
    """
    Factory to create the Analyst Agent.
    """
    return BaseAgent(
        name="Analyst",
        role="Expert Financial Market Researcher and Strategy Architect",
        model_name="gpt-4-turbo-preview",
        skills=[
            ResearchSkill()
        ],
        enable_long_term_memory=True,
        user_id="researcher_1"
    )

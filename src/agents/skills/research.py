"""
Research Skill
Equips agents with the ability to query the internal Knowledge base.
"""

from src.agents.skills.base import AgentSkill
from src.agents.knowledge.retriever import get_retrieval_tool

class ResearchSkill(AgentSkill):
    def __init__(self):
        super().__init__(
            name="ResearchSkill",
            description="Access to internal articles, TRDs, and trading documentation."
        )
        
        self.tools.append(get_retrieval_tool())
        
        self.system_prompt_addition = """
        You have access to the QuantMindX Knowledge Base.
        - Always cite specific articles or TRDs when providing info.
        - If the KB doesn't have the answer, state that clearly.
        - Focus on extracting actionable trading rules.
        """

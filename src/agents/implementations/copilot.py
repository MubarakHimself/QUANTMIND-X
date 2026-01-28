"""
QuantMind Co-pilot Agent
The highest hierarchical level orchestrator.
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any

from src.agents.core.base_agent import BaseAgent
from src.agents.skills.queuing import TaskQueueSkill
from src.agents.implementations.analyst import create_analyst_agent
from src.agents.implementations.quant_code import create_quant_code_agent

logger = logging.getLogger(__name__)

class CopilotAgent(BaseAgent):
    """
    The Master Orchestrator. 
    Can delegate tasks to specialized agents (Analyst, QuantCode) 
    and monitor the global Task Queue.
    """
    def __init__(self, **kwargs):
        # Adding TaskQueueSkill as the primary communication channel
        if 'skills' not in kwargs:
            kwargs['skills'] = []
        kwargs['skills'].append(TaskQueueSkill())
        
        super().__init__(**kwargs)
        
        # Initialize internal references to specialized agents (lazy-loaded if needed)
        self._analyst = None
        self._quant_code = None

    def get_analyst(self):
        if not self._analyst:
            self._analyst = create_analyst_agent()
        return self._analyst

    def get_quant_code(self):
        if not self._quant_code:
            self._quant_code = create_quant_code_agent()
        return self._quant_code

    async def execute_mission(self, mission_statement: str):
        """
        High-level entry point for "Project Mode".
        Example: "Build a Trend Follower based on the transcript rsi_reversion.txt"
        """
        # 1. Ask Analyst to build TRD
        # (In a real implementation, this would be a node in a Co-pilot graph)
        logger.info(f"Co-pilot mission start: {mission_statement}")
        # Delegation logic goes here...
        pass

def create_copilot_agent() -> CopilotAgent:
    """Factory to create the Master Co-pilot."""
    return CopilotAgent(
        name="QuantMind_Co-pilot",
        role="Master Project Orchestrator & Quantitative System Manager",
        model_name="openai/gpt-4-turbo", # OpenRouter naming convention
        enable_long_term_memory=True,
        user_id="supervisor_1",
        kb_namespace="copilot_kb"
    )

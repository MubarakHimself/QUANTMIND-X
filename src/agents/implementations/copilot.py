from typing import TypedDict, List, Annotated
import operator
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

from src.agents.core.base_agent import BaseAgent
from src.agents.skills.queuing import TaskQueueSkill
from src.agents.implementations.analyst import create_analyst_agent
from src.agents.implementations.quant_code import create_quant_code_agent

import logging
import asyncio
from typing import TypedDict, List, Annotated, Literal
import operator
import json
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END

from src.agents.core.base_agent import BaseAgent
from src.agents.core.database import sys_db
from src.agents.core.hooks import hook_manager
from src.agents.knowledge.router import kb_router
from src.agents.implementations.analyst import create_analyst_agent
from src.agents.implementations.quant_code import create_quant_code_agent

logger = logging.getLogger(__name__)

class CopilotState(TypedDict):
    """The state for the master orchestrator."""
    messages: Annotated[List[BaseMessage], operator.add]
    mission_id: str
    mode: Literal["PLAN", "ASK", "BUILD"]
    next_action: str
    current_trd: str
    current_code: str

class CopilotAgent(BaseAgent):
    """
    The Mission Control Orchestrator (V2).
    Operates in PLAN, ASK, BUILD modes.
    Delegates via the HookManager.
    State persisted in QuantMindSysDB.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.graph = self._build_orchestration_graph()
        
        # Register local hooks (connecting the workers)
        self._register_worker_hooks()

    def _register_worker_hooks(self):
        """Define how local L1 agents respond to hooks."""
        
        async def analyst_handler(payload):
            # Instantiate ephemeral agent for the task
            agent = create_analyst_agent()
            response = await agent.ainvoke(f"Task: {payload.get('type')}\nContext: {payload.get('context')}")
            return {"output": response, "status": "success"}
            
        async def quant_code_handler(payload):
            # Instantiate ephemeral agent for the task
            agent = create_quant_code_agent()
            trd = payload.get('trd_content')
            response = await agent.ainvoke(f"Task: BUILD\nTRD: {trd}")
            return {"output": response, "status": "success"}

        hook_manager.register_agent("analyst", analyst_handler)
        hook_manager.register_agent("quant_code", quant_code_handler)

    def _build_orchestration_graph(self):
        builder = StateGraph(CopilotState)
        
        builder.add_node("router", self.router_node)
        builder.add_node("planner", self.plan_mode_node)
        builder.add_node("asker", self.ask_mode_node)
        builder.add_node("builder", self.build_mode_node)
        
        builder.set_entry_point("router")
        
        builder.add_conditional_edges(
            "router",
            lambda state: state["mode"].lower(),
            {
                "plan": "planner",
                "ask": "asker",
                "build": "builder"
            }
        )
        
        builder.add_edge("planner", END)
        builder.add_edge("asker", END)
        builder.add_edge("builder", END)
        
        return builder.compile(checkpointer=self.checkpointer)

    async def router_node(self, state: CopilotState):
        """Determine the current mode based on user input or state."""
        # Simple heuristic for V2: Default to PLAN if new, else inherit
        current_mode = state.get("mode", "PLAN")
        
        # If user explicitly asks for "build" or "deploy", switch mode
        last_msg = state["messages"][-1].content.lower() if state["messages"] else ""
        if "build" in last_msg or "deploy" in last_msg:
            current_mode = "BUILD"
        elif "question" in last_msg or "?" in last_msg:
            current_mode = "ASK"
            
        return {"mode": current_mode}

    async def plan_mode_node(self, state: CopilotState):
        """High-level mission decomposition."""
        logger.info("Co-pilot: Entering PLAN Mode")
        
        # 1. Create Mission in DB
        mission_id = sys_db.create_mission(self.user_id, state["messages"][-1].content)
        
        # 2. Consult Knowledge Router for similar past missions
        # (Example usage)
        # context = kb_router.search("copilot_kb", vector_embedding_of_query)
        
        prompt = "You are the Architect. Decompose this request into a Strategy Design Plan."
        response = await self.llm.ainvoke([SystemMessage(content=prompt)] + state["messages"])
        
        return {"messages": [response], "mission_id": mission_id}

    async def ask_mode_node(self, state: CopilotState):
        """Collaborative chat."""
        logger.info("Co-pilot: Entering ASK Mode")
        response = await self.llm.ainvoke(state["messages"])
        return {"messages": [response]}

    async def build_mode_node(self, state: CopilotState):
        """Execution pipeline: Analyst -> QuantCode."""
        logger.info("Co-pilot: Entering BUILD Mode")
        
        # 1. Trigger Analyst Hook
        logger.info("Submitting DESIGN job to Analyst...")
        analyst_result = await hook_manager.submit_job(
            "analyst", 
            "DESIGN", 
            {"context": state["messages"][-1].content},
            mission_id=state.get("mission_id", "orphan")
        )
        
        trd_content = analyst_result["output"].content # Mock extraction
        
        # 2. Trigger QuantCode Hook
        logger.info("Submitting BUILD job to QuantCode...")
        code_result = await hook_manager.submit_job(
            "quant_code",
            "BUILD",
            {"trd_content": trd_content},
            mission_id=state.get("mission_id", "orphan")
        )
        
        final_msg = f"BUILD COMPLETE.\nAnalyst: {analyst_result.get('status')}\nQuantCode: {code_result.get('status')}"
        return {"messages": [AIMessage(content=final_msg)], "current_trd": trd_content}

def create_copilot_agent() -> CopilotAgent:
    """Factory to create the Master Co-pilot V2."""
    # Fetch system prompt from DB if available
    db_prompt = sys_db.get_agent_prompt("QuantMind_Co-pilot")
    
    return CopilotAgent(
        name="QuantMind_Co-pilot",
        role="Master Project Orchestrator & Quantitative System Manager",
        model_name="openai/gpt-4-turbo",
        enable_long_term_memory=True,
        user_id="supervisor_1",
        kb_namespace="copilot_kb",
        system_prompt=db_prompt # Override default if DB has entry
    )


from typing import TypedDict, List, Annotated
import operator
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

from src.agents.core.base_agent import BaseAgent
from src.agents.skills.research import ResearchSkill
from src.agents.skills.queuing import TaskQueueSkill

class AnalystState(TypedDict):
    """The state maintained throughout the Analyst's reasoning chain."""
    messages: Annotated[List[BaseMessage], operator.add]
    mechanics: dict               # Extracted from NPRD
    augmented_data: str           # KB Research results
    compliance_report: str        # Risk/Router alignment
    final_trd_path: str           # Resulting file path
    source_context: str           # Link to original NPRD

class AnalystAgent(BaseAgent):
    """
    Subclass of BaseAgent that implements a specialized synthesis graph.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.graph = self._build_synthesis_graph()

    def _build_synthesis_graph(self):
        """Constructs the multi-node workflow."""
        builder = StateGraph(AnalystState)
        
        # 1. Add Nodes
        builder.add_node("nprd_miner", self.nprd_miner_node)
        builder.add_node("kb_augmenter", self.kb_augmenter_node)
        builder.add_node("compliance_checker", self.compliance_check_node)
        builder.add_node("synthesizer", self.synthesis_node)
        
        # 2. Define Edges
        builder.set_entry_point("nprd_miner")
        builder.add_edge("nprd_miner", "kb_augmenter")
        builder.add_edge("kb_augmenter", "compliance_checker")
        builder.add_edge("compliance_checker", "synthesizer")
        builder.add_edge("synthesizer", END)
        
        return builder.compile(checkpointer=self.checkpointer)

    # --- Node Implementations ---

    async def nprd_miner_node(self, state: AnalystState):
        """Ingests strategy mechanics from local NPRD/Opal synced files."""
        # For V1, we simulate reading from data/nprd/
        import os
        nprd_dir = "data/nprd"
        os.makedirs(nprd_dir, exist_ok=True)
        
        # Logic: Find the latest transcript
        files = os.listdir(nprd_dir)
        if not files:
            return {"messages": [SystemMessage(content="Waiting for NPRD strategy documents in data/nprd/")]}
            
        source_context = os.path.join(nprd_dir, files[0])
        with open(source_context, "r") as f:
            content = f.read()

        messages = [
            SystemMessage(content="You are the Strategy Extractor. Extract Entry, Exit, and Risk logic from this trading transcript. Output a structured summary."),
            HumanMessage(content=content)
        ]
        response = await self.llm.ainvoke(messages)
        return {
            "mechanics": {"logic": response.content},
            "source_context": source_context
        }

    async def kb_augmenter_node(self, state: AnalystState):
        """Searches KB for technical implementation details."""
        query = f"MQL5 implementation for {state['mechanics'].get('strategy_type')}"
        # We manually call the tool logic (simulated for now)
        from src.agents.knowledge.retriever import search_knowledge_base
        kb_results = search_knowledge_base.invoke({"query": query, "collection": "articles"})
        return {"augmented_data": kb_results}

    async def compliance_check_node(self, state: AnalystState):
        """Ensures logic follows QuantMindX Risk & Router standards."""
        prompt = f"""
        Review the following trading logic for compliance with QuantMindX V1 Standards:
        Standards:
        1. Must use 'BaseBot' as parent class.
        2. Must implement 'calculate_entry_size' using EnhancedKelly.
        3. Must utilize 'NativeBridge' for ZMQ communication.
        4. Must provide a performance rank for the 'Strategy Auction'.
        
        Logic to Review:
        {state['mechanics'].get('logic')}
        """
        messages = [SystemMessage(content="You are a Technical Compliance Officer."), HumanMessage(content=prompt)]
        response = await self.llm.ainvoke(messages)
        return {"compliance_report": response.content}

    async def synthesis_node(self, state: AnalystState):
        """Generates and saves the final TRD document."""
        trd_content = f"""# Strategy TRD: {state['mechanics'].get('strategy_type')}
## Context Links
- Source: {state.get('source_context', 'N/A')}

## Mechanics
{state['mechanics'].get('logic')}

## Technical Implementation (KB)
{state['augmented_data']}

## Ecosystem Alignment
{state['compliance_report']}
"""
        # Actual file write
        import os
        path = "docs/trds/generated_strategy.md"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(trd_content)
            
        return {"final_trd_path": path, "messages": [SystemMessage(content=f"TRD Generated successfully at {path}")]}

def create_analyst_agent() -> AnalystAgent:
    """Factory to create the Deep Analyst Agent."""
    return AnalystAgent(
        name="Analyst",
        role="Expert Financial Market Researcher and Strategy Architect",
        model_name="gpt-4-turbo-preview",
        skills=[ResearchSkill(), TaskQueueSkill()],
        enable_long_term_memory=True,
        user_id="researcher_1",
        kb_namespace="analyst_kb"
    )
    )

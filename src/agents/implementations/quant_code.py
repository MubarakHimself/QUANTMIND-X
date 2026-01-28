from typing import TypedDict, List, Annotated, Optional
import operator
import json
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from mcp import ClientSession # Implicit dependency managed by BaseAgent

from src.agents.core.base_agent import BaseAgent
from src.agents.skills.coding import CodingSkill
from src.agents.skills.queuing import TaskQueueSkill

class CodeState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    trd_content: str
    current_code: str
    backtest_results: Optional[dict]
    compilation_error: str
    retry_count: int

class QuantCodeAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.graph = self._build_trial_graph()

    def _build_trial_graph(self):
        builder = StateGraph(CodeState)
        
        # Nodes (using 'node' terminology for LangGraph)
        builder.add_node("planner", self.planning_node)
        builder.add_node("coder", self.coding_node)
        builder.add_node("validator", self.validation_node)
        builder.add_node("reflector", self.reflection_node)
        
        builder.set_entry_point("planner")
        builder.add_edge("planner", "coder")
        builder.add_edge("coder", "validator")
        
        builder.add_conditional_edges(
            "validator",
            self.should_continue,
            {
                "retry": "reflector",
                "done": END
            }
        )
        builder.add_edge("reflector", "coder")
        
        return builder.compile(checkpointer=self.checkpointer)

    async def planning_node(self, state: CodeState):
        """
        Analyze TRD and plan file structure. 
        Fetches CODING STANDARDS from Assets Hub first.
        """
        standards = "Standard: Use Python 3.10+, Type Hints."
        
        # Try to fetch real standards via MCP
        # Assuming self.call_mcp_tool is available in BaseAgent or we use a direct client
        # For simplicity in this scaffold, we simulate the retrieval if the client isn't ready
        try:
             # This would be: self.call_tool("quantmindx-kb-chroma", "get_coding_standards", {})
             pass 
        except Exception:
             pass

        messages = [
            SystemMessage(content=f"You are a Lead Software Architect. Create a file structure and class plan based on this TRD.\n\nSTANDARDS:\n{standards}"),
            HumanMessage(content=state.get("trd_content", "No TRD provided."))
        ]
        response = await self.llm.ainvoke(messages)
        return {"messages": [response]}

    async def coding_node(self, state: CodeState):
        """Generate the actual MQL5 or Python code."""
        prompt = "Generate the code based on the plan. Ensure 'BaseBot' integration."
        if state.get("compilation_error"):
            error_msg = state['compilation_error']
            # If it's a backtest failure, include the sharpe/drawdown info
            if state.get("backtest_results"):
                bt = state['backtest_results']
                error_msg += f"\nBacktest Metrics: Sharpe={bt.get('sharpe')}, Drawdown={bt.get('drawdown')}"
            
            prompt = f"Fix the following error/performance issue:\n{error_msg}\n\nCurrent code:\n{state['current_code']}"
        
        messages = state["messages"] + [HumanMessage(content=prompt)]
        response = await self.llm.ainvoke(messages)
        return {"current_code": response.content, "messages": [response]}

    async def validation_node(self, state: CodeState):
        """
        Validate code by running a BACKTEST via MCP.
        """
        code = state["current_code"]
        
        # 1. Static Checks
        if "class" not in code and "def" not in code:
             return {"compilation_error": "No logic detected.", "retry_count": state.get("retry_count", 0) + 1}
             
        # 2. Run Backtest (via MCP)
        # Note: In a real run, we would call the 'backtest-server' tool 'run_backtest'
        # result = await self.call_tool("backtest-server", "run_backtest", {"code_content": code})
        
        # Mocking the MCP call for this file updates to ensure validity without live server
        # In production this line is: result_json = await self.mcp_client.call("run_backtest", code_content=code)
        
        # Simulating a check for 'QuantMindBacktester' in code to determine pass/fail
        if "backtrader" not in code:
             return {
                 "compilation_error": "Code missing 'backtrader' import.", 
                 "retry_count": state.get("retry_count", 0) + 1
             }

        # Mock Success for now
        return {
            "compilation_error": None, 
            "backtest_results": {"sharpe": 1.5, "drawdown": 5.0} # Mock data
        }

    def should_continue(self, state: CodeState):
        if state.get("compilation_error") and state.get("retry_count", 0) < 3:
            return "retry"
        return "done"

    async def reflection_node(self, state: CodeState):
        """Analyze the failure and prepare a fix plan."""
        return {"messages": [SystemMessage(content=f"Reflecting on error: {state['compilation_error']}. Attempting fix.")]}

def create_quant_code_agent() -> QuantCodeAgent:
    """Factory to create the QuantCode Agent."""
    return QuantCodeAgent(
        name="QuantCode",
        role="Senior MQL5 and Python Developer",
        model_name="gpt-4-turbo-preview",
        skills=[CodingSkill(root_dir="."), TaskQueueSkill()],
        enable_long_term_memory=True,
        user_id="developer_1",
        kb_namespace="quantcode_kb"
    )

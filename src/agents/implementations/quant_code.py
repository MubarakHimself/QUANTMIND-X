from typing import TypedDict, List, Annotated
import operator
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

from src.agents.core.base_agent import BaseAgent
from src.agents.skills.coding import CodingSkill
from src.agents.skills.queuing import TaskQueueSkill

class CodeState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    trd_content: str
    current_code: str
    compilation_error: str
    retry_count: int

class QuantCodeAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.graph = self._build_trial_graph()

    def _build_trial_graph(self):
        builder = StateGraph(CodeState)
        
        builder.add_node("planner", self.planning_node)
        builder.add_node("coder", self.coding_node)
        builder.add_node("validator", self.validation_node)
        builder.add_node("reflector", self.reflection_node)
        
        builder.set_entry_point("planner")
        builder.add_edge("planner", "coder")
        builder.add_edge("coder", "validator")
        
        # Trial Loop: If error, go to reflector -> coder, else END
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
        """Analyze TRD and plan file structure."""
        messages = [
            SystemMessage(content="You are a Lead Software Architect. Create a file structure and class plan based on this TRD."),
            HumanMessage(content=state.get("trd_content", "No TRD provided."))
        ]
        response = await self.llm.ainvoke(messages)
        return {"messages": [response]}

    async def coding_node(self, state: CodeState):
        """Generate the actual MQL5 or Python code."""
        prompt = "Generate the code based on the plan. Ensure 'BaseBot' integration."
        if state.get("compilation_error"):
            prompt = f"Fix the following error:\n{state['compilation_error']}\n\nCurrent code:\n{state['current_code']}"
        
        messages = state["messages"] + [HumanMessage(content=prompt)]
        response = await self.llm.ainvoke(messages)
        return {"current_code": response.content, "messages": [response]}

    async def validation_node(self, state: CodeState):
        """Simulate compilation or linting."""
        # V1: Mock success or check for syntax errors
        code = state["current_code"]
        if "class" not in code and "def" not in code:
             return {"compilation_error": "No logic detected.", "retry_count": state.get("retry_count", 0) + 1}
        return {"compilation_error": None}

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

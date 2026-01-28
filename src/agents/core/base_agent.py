"""
QuantMind Base Agent
The foundational class for all autonomous agents in the ecosystem.
Wraps LangGraph and LangChain to provide a unified, reusable interface.
"""

import logging
import asyncio
from typing import List, Optional, Dict, Any, Union, Callable

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool, BaseTool
from langgraph.prebuilt import create_react_agent
from langmem import create_manage_memory_tool, create_search_memory_tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from src.agents.skills.base import AgentSkill

logger = logging.getLogger(__name__)

class BaseAgent:
    """
    A reusable Agent wrapper using LangGraph ReAct pattern with "Deep" features.
    """
    def __init__(
        self,
        name: str,
        role: str,
        model_name: str = "gpt-4-turbo-preview",
        skills: List[AgentSkill] = [],
        enable_long_term_memory: bool = False,
        enable_planning: bool = True,
        enable_subagents: bool = True,
        user_id: str = "default_user",
        mcp_servers: List[Dict[str, Any]] = []
    ):
        self.name = name
        self.role = role
        self.user_id = user_id
        self.skills = skills
        self.checkpointer = MemorySaver()
        self.mcp_clients: List[ClientSession] = []
        
        # 1. Initialize LLM
        self.llm = self._init_llm(model_name)
        
        # 2. Base Tools & System Prompt
        self.tools: List[Union[BaseTool, Callable]] = []
        self.system_prompt = f"You are {name}, {role}.\n"
        
        # 3. Add Planning (Deep Characteristic)
        if enable_planning:
            self._add_planning_tool()
            
        # 4. Add Sub-agent capability
        if enable_subagents:
            self._add_subagent_tool()
        
        # 5. Compile Tools from Skills
        for skill in skills:
            self.tools.extend(skill.get_tools())
            if skill.get_system_prompt():
                self.system_prompt += f"\n\n## Skill: {skill.name}\n{skill.get_system_prompt()}"

        # 6. Add MCP Servers
        for mcp_config in mcp_servers:
            # Note: MCP initialization is async, will be handled in a 'startup' method
            pass

        # 7. Add Memory Tools (LangMem)
        if enable_long_term_memory:
            self._add_memory_tools()

        # 8. Build Graph
        self.graph = create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=self.system_prompt,
            checkpointer=self.checkpointer
        )
        
        logger.info(f"Agent {name} initialized with {len(skills)} skills and {len(self.tools)} tools.")

    def _init_llm(self, model_name: str):
        """Factory for LLM initialization."""
        if "claude" in model_name:
            return ChatAnthropic(model=model_name)
        else:
            return ChatOpenAI(model=model_name)

    def _add_memory_tools(self):
        """Injects LangMem tools for long-term recall."""
        namespace = ("memories", self.user_id)
        self.tools.append(create_manage_memory_tool(namespace=namespace))
        self.tools.append(create_search_memory_tool(namespace=namespace))
        self.system_prompt += "\n\n## Memory Access\nYou have access to long-term memory. Use it to store preferences, user info, and procedural learnings."

    def _add_planning_tool(self):
        """Adds a No-Op Todo List tool for context engineering (Deep Agent pattern)."""
        @tool
        def update_todo_list(todos: List[str]) -> str:
            """
            Update your internal planning todo list. 
            Use this to break down complex tasks and track progress.
            """
            # In a stateless ReAct loop, this basically just puts the plan in the chat history
            return "Todo list updated. Continue with the next step."
        
        self.tools.append(update_todo_list)
        self.system_prompt += "\n\n## Planning\nYou have a 'todo_list' tool. Use it to decompose complex requests into smaller steps."

    def _add_subagent_tool(self):
        """Adds ability to spawn specialized subtasks."""
        @tool
        def delegate_task(agent_type: str, task_description: str) -> str:
            """
            Delegate a specialized sub-task to another agent.
            Supported types: 'researcher', 'coder', 'analyst'.
            """
            # Implementation of the Command pattern for multi-agent handoff
            # For V1, this could be a recursive call or a proper Command update
            return f"Delegating task to {agent_type}: {task_description}"

        self.tools.append(delegate_task)

    async def connect_mcp(self, name: str, command: str, args: List[str] = []):
        """Connect to an external MCP server and pull its tools."""
        server_params = StdioServerParameters(command=command, args=args, env=None)
        # This requires more complex integration with LangGraph's tool runtime
        # but represents the architectural intent.
        logger.info(f"Connecting to MCP server: {name}")

    def invoke(self, message: str, thread_id: str = "1") -> str:
        """
        Synchronous chat interface.
        """
        config = {"configurable": {"thread_id": thread_id}}
        
        # LangGraph inputs
        inputs = {"messages": [HumanMessage(content=message)]}
        
        # Stream response (simplified to get final message for now)
        final_state = self.graph.invoke(inputs, config=config)
        
        # Extract last AI message
        return final_state["messages"][-1].content

    def stream(self, message: str, thread_id: str = "1"):
        """
        Generator for streaming tokens/steps.
        """
        config = {"configurable": {"thread_id": thread_id}}
        inputs = {"messages": [HumanMessage(content=message)]}
        
        for event in self.graph.stream(inputs, config=config, stream_mode="values"):
            message = event["messages"][-1]
            if isinstance(message, BaseMessage):
                 yield message.content  # Simplified stream

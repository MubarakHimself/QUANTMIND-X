"""
Chat API Endpoints

Provides the backend for the QuantMind Copilot chat interface.
Routes messages to the appropriate agent or handles them via a general LLM orchestrator.
Integrates ToolNode-based agents (copilot_v2, analyst_v2, quantcode_v2).

**Validates: Requirements 8.1, 8.4**
"""

import os
import logging
from typing import List, Dict, Any, Optional
import json
import asyncio

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

# Pine Script agent imports
try:
    from src.agents.pinescript import (
        generate_pine_script_from_query,
        convert_mql5_to_pinescript
    )
    PINESCRIPT_AGENT_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Pine Script agent not available: {e}")
    PINESCRIPT_AGENT_AVAILABLE = False

# LangChain / LangGraph imports
try:
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    from langchain_openai import ChatOpenAI
    from langchain_anthropic import ChatAnthropic
    from langgraph.prebuilt import create_react_agent
except ImportError:
    # Fallback for environments without full AI dependencies
    ChatOpenAI = None
    ChatAnthropic = None
    create_react_agent = None

# Import Agent Workflows - v2 with ToolNode integration
try:
    from src.agents.copilot_v2 import compile_copilot_graph
    from src.agents.analyst_v2 import compile_analyst_graph
    from src.agents.quantcode_v2 import compile_quantcode_graph
    V2_AGENTS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"V2 agents not available: {e}")
    V2_AGENTS_AVAILABLE = False

# Legacy Agent Workflows
try:
    from src.agents.copilot import run_copilot_workflow
    from src.agents.analyst import run_analyst_workflow
    LEGACY_AGENTS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Legacy agents not available: {e}")
    LEGACY_AGENTS_AVAILABLE = False

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])

# =============================================================================
# Configuration Flag for ToolNode vs Legacy
# =============================================================================

# Can be overridden via environment variable
USE_TOOL_NODE_AGENTS = os.getenv("QUANTMIND_USE_TOOL_NODE", "true").lower() == "true"


# =============================================================================
# Models
# =============================================================================

class ChatMessage(BaseModel):
    role: str
    content: str
    agent: Optional[str] = None

class ChatRequest(BaseModel):
    message: str
    agent_id: str = "copilot"
    history: List[ChatMessage] = []
    model: str = "gemini-2.0-flash"  # Default model preference from UI
    api_keys: Dict[str, str] = {}
    use_tool_node: Optional[bool] = None  # Override global setting


class ChatResponse(BaseModel):
    reply: str
    agent_id: str
    action_taken: Optional[str] = None
    artifacts: List[Dict[str, Any]] = []


# =============================================================================
# Pine Script Request Models
# =============================================================================

class PineScriptGenerateRequest(BaseModel):
    """Request model for generating Pine Script from natural language."""
    query: str

class PineScriptConvertRequest(BaseModel):
    """Request model for converting MQL5 to Pine Script."""
    mql5_code: str


class PineScriptResponse(BaseModel):
    """Response model for Pine Script endpoints."""
    pine_script: Optional[str] = None
    status: str
    errors: List[str] = []


# =============================================================================
# LLM Setup
# =============================================================================

def get_llm(model_name: str, keys: Dict[str, str] = {}):
    """Get the appropriate LLM based on model name and available keys."""
    # Check for API Keys (Request > Env)
    openai_key = keys.get("openai") or os.getenv("OPENAI_API_KEY")
    anthropic_key = keys.get("anthropic") or os.getenv("ANTHROPIC_API_KEY")

    # Map UI model names to actual implementations
    if "claude" in model_name and anthropic_key:
        return ChatAnthropic(model="claude-3-5-sonnet-20240620", api_key=anthropic_key)
    elif "gpt" in model_name and openai_key:
        return ChatOpenAI(model="gpt-4o", api_key=openai_key)
    # Default fallback if keys exist
    if openai_key:
        return ChatOpenAI(model="gpt-4o", api_key=openai_key)
    if anthropic_key:
        return ChatAnthropic(model="claude-3-sonnet-20240229", api_key=anthropic_key)

    return None


# =============================================================================
# ToolNode Agent Invocation
# =============================================================================

async def invoke_toolnode_agent(
    agent_type: str,
    message: str,
    history: List[ChatMessage] = None
) -> Dict[str, Any]:
    """
    Invoke a ToolNode-based agent.

    Args:
        agent_type: One of 'copilot', 'analyst', 'quantcode'
        message: User message to process
        history: Optional conversation history

    Returns:
        Dict with reply and any artifacts
    """
    if not V2_AGENTS_AVAILABLE:
        raise RuntimeError("ToolNode agents not available")

    # Build messages list
    messages = []

    # Add history if provided
    if history:
        for msg in history:
            if msg.role == "user":
                messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                messages.append(AIMessage(content=msg.content))

    # Add current message
    messages.append(HumanMessage(content=message))

    # Select and compile the appropriate graph
    if agent_type == "copilot":
        graph = compile_copilot_graph(use_tool_node=True)
    elif agent_type == "analyst":
        graph = compile_analyst_graph(use_tool_node=True)
    elif agent_type == "quantcode":
        graph = compile_quantcode_graph(use_tool_node=True)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    # Invoke the graph
    config = {"configurable": {"thread_id": f"{agent_type}_session"}}
    result = graph.invoke({"messages": messages}, config)

    # Extract the final message
    if "messages" in result and result["messages"]:
        last_message = result["messages"][-1]
        reply = last_message.content if hasattr(last_message, 'content') else str(last_message)
    else:
        reply = "No response generated"

    return {
        "reply": reply,
        "agent_id": agent_type,
        "artifacts": [],
    }


def invoke_toolnode_agent_sync(
    agent_type: str,
    message: str,
    history: List[ChatMessage] = None
) -> Dict[str, Any]:
    """Synchronous wrapper for ToolNode agent invocation."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(invoke_toolnode_agent(agent_type, message, history))
    finally:
        loop.close()


# =============================================================================
# Legacy Tools & Agent Wrappers
# =============================================================================

def tool_deploy_strategy(strategy_name: str):
    """Deploys a trading strategy using the Copilot workflow."""
    logger.info(f"Tool invoked: deploy_strategy({strategy_name})")

    if LEGACY_AGENTS_AVAILABLE:
        result = run_copilot_workflow(f"Deploy strategy {strategy_name}")
        # Extract final message from result
        return f"Deployment initiated. Status: {result.get('monitoring_data', {}).get('status', 'Unknown')}"
    else:
        return f"Deployment simulation for strategy: {strategy_name}"


def tool_analyze_market(query: str):
    """Analyzes market conditions using the Analyst workflow."""
    logger.info(f"Tool invoked: analyze_market({query})")

    if LEGACY_AGENTS_AVAILABLE:
        result = run_analyst_workflow(query)
        return result.get("synthesis_result", "Analysis failed to produce result.")
    else:
        return f"Market analysis simulation for: {query}"


# =============================================================================
# Endpoints
# =============================================================================

@router.post("/send", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Handle chat messages from the UI.
    Routes to specific agent workflows or general LLM chat.
    Supports ToolNode-based agents (v2) or legacy workflows.
    """
    logger.info(f"Received chat request for agent: {request.agent_id} | Model: {request.model}")

    # Determine whether to use ToolNode agents
    use_tool_node = request.use_tool_node if request.use_tool_node is not None else USE_TOOL_NODE_AGENTS

    try:
        # 1. Check for specific slash commands (handled by UI mostly, but backup here)
        if request.message.startswith("/deploy"):
            # Direct invocation of deployment
            if use_tool_node and V2_AGENTS_AVAILABLE:
                result = await invoke_toolnode_agent("copilot", request.message)
                return ChatResponse(
                    reply=result["reply"],
                    agent_id="copilot",
                    action_taken="deployment"
                )
            else:
                reply = tool_deploy_strategy(request.message.replace("/deploy", "").strip())
                return ChatResponse(reply=reply, agent_id="copilot", action_taken="deployment")

        if request.message.startswith("/analyze"):
            # Direct invocation of analysis
            if use_tool_node and V2_AGENTS_AVAILABLE:
                result = await invoke_toolnode_agent("analyst", request.message)
                return ChatResponse(
                    reply=result["reply"],
                    agent_id="analyst",
                    action_taken="analysis"
                )
            else:
                reply = tool_analyze_market(request.message.replace("/analyze", "").strip())
                return ChatResponse(reply=reply, agent_id="analyst", action_taken="analysis")

        if request.message.startswith("/code") or request.message.startswith("/generate"):
            # Direct invocation of code generation
            if use_tool_node and V2_AGENTS_AVAILABLE:
                result = await invoke_toolnode_agent("quantcode", request.message)
                return ChatResponse(
                    reply=result["reply"],
                    agent_id="quantcode",
                    action_taken="code_generation"
                )

        # 2. Route to appropriate agent based on agent_id
        if use_tool_node and V2_AGENTS_AVAILABLE:
            # Use ToolNode-based agents
            if request.agent_id in ["copilot", "analyst", "quantcode"]:
                result = await invoke_toolnode_agent(
                    request.agent_id,
                    request.message,
                    request.history
                )
                return ChatResponse(
                    reply=result["reply"],
                    agent_id=request.agent_id,
                    artifacts=result.get("artifacts", [])
                )

        # 3. Fallback to General Chat / Orchestration
        llm = get_llm(request.model, request.api_keys)

        if not llm:
            # Fallback if no LLM/Keys available (Mock Mode)
            return mock_chat_response(request)

        # Create orchestrator agent
        tools = [tool_deploy_strategy, tool_analyze_market]
        if create_react_agent:
            agent_executor = create_react_agent(llm, tools)

            # Run agent
            inputs = {"messages": [HumanMessage(content=request.message)]}
            response = agent_executor.invoke(inputs)

            # Extract final message
            last_message = response["messages"][-1]
            content = last_message.content

            return ChatResponse(reply=content, agent_id=request.agent_id)
        else:
            # Fallback if langgraph prebuilt not available
            resp = llm.invoke(request.message)
            return ChatResponse(reply=resp.content, agent_id=request.agent_id)

    except Exception as e:
        logger.error(f"Chat processing error: {e}", exc_info=True)
        return ChatResponse(
            reply=f"I encountered an error processing your request: {str(e)}",
            agent_id="system"
        )


@router.post("/{agent_type}/invoke", response_model=ChatResponse)
async def invoke_agent_direct(
    agent_type: str,
    request: ChatRequest,
    background_tasks: BackgroundTasks
):
    """
    Directly invoke a specific agent type.

    Args:
        agent_type: One of 'copilot', 'analyst', 'quantcode'
        request: Chat request with message and optional history
    """
    if agent_type not in ["copilot", "analyst", "quantcode"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid agent type: {agent_type}. Must be one of: copilot, analyst, quantcode"
        )

    use_tool_node = request.use_tool_node if request.use_tool_node is not None else USE_TOOL_NODE_AGENTS

    try:
        if use_tool_node and V2_AGENTS_AVAILABLE:
            result = await invoke_toolnode_agent(agent_type, request.message, request.history)
            return ChatResponse(
                reply=result["reply"],
                agent_id=agent_type,
                artifacts=result.get("artifacts", [])
            )
        else:
            # Legacy fallback
            if agent_type == "copilot" and LEGACY_AGENTS_AVAILABLE:
                result = run_copilot_workflow(request.message)
                return ChatResponse(
                    reply=str(result.get("messages", ["Completed"])[-1]),
                    agent_id=agent_type
                )
            elif agent_type == "analyst" and LEGACY_AGENTS_AVAILABLE:
                result = run_analyst_workflow(request.message)
                return ChatResponse(
                    reply=result.get("synthesis_result", "Analysis complete"),
                    agent_id=agent_type
                )
            else:
                return ChatResponse(
                    reply=f"Agent {agent_type} processing: {request.message}",
                    agent_id=agent_type
                )

    except Exception as e:
        logger.error(f"Agent invocation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Pine Script Endpoints
# =============================================================================

@router.post("/pinescript", response_model=PineScriptResponse)
async def generate_pine_script_endpoint(request: PineScriptGenerateRequest):
    """
    Generate Pine Script v5 code from natural language query.
    
    Args:
        request: PineScriptGenerateRequest with query field
        
    Returns:
        PineScriptResponse with pine_script, status, and errors
    """
    logger.info(f"Received Pine Script generation request: {request.query[:100]}...")
    
    if not PINESCRIPT_AGENT_AVAILABLE:
        return PineScriptResponse(
            pine_script=None,
            status="error",
            errors=["Pine Script agent is not available. Please check server configuration."]
        )
    
    try:
        result = generate_pine_script_from_query(request.query)
        return PineScriptResponse(
            pine_script=result.get("pine_script"),
            status=result.get("status", "complete"),
            errors=result.get("errors", [])
        )
    except Exception as e:
        logger.error(f"Pine Script generation error: {e}", exc_info=True)
        return PineScriptResponse(
            pine_script=None,
            status="error",
            errors=[str(e)]
        )


@router.post("/pinescript/convert", response_model=PineScriptResponse)
async def convert_mql5_to_pinescript_endpoint(request: PineScriptConvertRequest):
    """
    Convert MQL5 code to Pine Script v5.
    
    Args:
        request: PineScriptConvertRequest with mql5_code field
        
    Returns:
        PineScriptResponse with pine_script, status, and errors
    """
    logger.info("Received MQL5 to Pine Script conversion request")
    
    if not PINESCRIPT_AGENT_AVAILABLE:
        return PineScriptResponse(
            pine_script=None,
            status="error",
            errors=["Pine Script agent is not available. Please check server configuration."]
        )
    
    try:
        result = convert_mql5_to_pinescript(request.mql5_code)
        return PineScriptResponse(
            pine_script=result.get("pine_script"),
            status=result.get("status", "complete"),
            errors=result.get("errors", [])
        )
    except Exception as e:
        logger.error(f"MQL5 to Pine Script conversion error: {e}", exc_info=True)
        return PineScriptResponse(
            pine_script=None,
            status="error",
            errors=[str(e)]
        )


def mock_chat_response(request: ChatRequest) -> ChatResponse:
    """Fallback response when no LLM is available."""
    msg = request.message.lower()

    if "deploy" in msg:
        return ChatResponse(
            reply="I can help with deployment. Since I'm in mock mode, I'll simulate a deployment check. Please verify your strategy parameters.",
            agent_id="copilot"
        )
    elif "analyze" in msg or "summary" in msg:
        return ChatResponse(
            reply="Market analysis requires a connection to live data sources. In this demo mode, I can confirm that the 'analyst' agent is registered and ready.",
            agent_id="analyst"
        )
    elif "code" in msg or "generate" in msg:
        return ChatResponse(
            reply="Code generation requires the QuantCode agent. In this demo mode, I can confirm the agent is registered and ready to generate MQL5 code.",
            agent_id="quantcode"
        )
    else:
        return ChatResponse(
            reply=f"I received your message: '{request.message}'. To get real responses, please configure your OPENAI_API_KEY or ANTHROPIC_API_KEY.",
            agent_id=request.agent_id
        )

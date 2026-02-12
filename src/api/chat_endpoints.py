"""
Chat API Endpoints

Provides the backend for the QuantMind Copilot chat interface.
Routes messages to the appropriate agent or handles them via a general LLM orchestrator.

**Validates: Requirements 8.1, 8.4**
"""

import os
import logging
from typing import List, Dict, Any, Optional
import json

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

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

# Import Agent Workflows
from src.agents.copilot import run_copilot_workflow
from src.agents.analyst import run_analyst_workflow

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])

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


class ChatResponse(BaseModel):
    reply: str
    agent_id: str
    action_taken: Optional[str] = None
    artifacts: List[Dict[str, Any]] = []

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
# Tools & Agent Wrappers
# =============================================================================

def tool_deploy_strategy(strategy_name: str):
    """Deploys a trading strategy using the Copilot workflow."""
    logger.info(f"Tool invoked: deploy_strategy({strategy_name})")
    result = run_copilot_workflow(f"Deploy strategy {strategy_name}")
    # Extract final message from result
    return f"Deployment initiated. Status: {result.get('monitoring_data', {}).get('status', 'Unknown')}"

def tool_analyze_market(query: str):
    """Analyzes market conditions using the Analyst workflow."""
    logger.info(f"Tool invoked: analyze_market({query})")
    result = run_analyst_workflow(query)
    return result.get("synthesis_result", "Analysis failed to produce result.")

# =============================================================================
# Endpoints
# =============================================================================

@router.post("/send", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Handle chat messages from the UI.
    Routes to specific agent workflows or general LLM chat.
    """
    logger.info(f"Received chat request for agent: {request.agent_id} | Model: {request.model}")
    
    try:
        # 1. Check for specific slash commands (handled by UI mostly, but backup here)
        if request.message.startswith("/deploy"):
            # Direct invocation of deployment
            reply = tool_deploy_strategy(request.message.replace("/deploy", "").strip())
            return ChatResponse(reply=reply, agent_id="copilot", action_taken="deployment")
            
        if request.message.startswith("/analyze"):
            # Direct invocation of analysis
            reply = tool_analyze_market(request.message.replace("/analyze", "").strip())
            return ChatResponse(reply=reply, agent_id="analyst", action_taken="analysis")

        # 2. General Chat / Orchestration
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
    else:
        return ChatResponse(
            reply=f"I received your message: '{request.message}'. To get real responses, please configure your OPENAI_API_KEY or ANTHROPIC_API_KEY.",
            agent_id=request.agent_id
        )

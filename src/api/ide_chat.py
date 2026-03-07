"""
QuantMind IDE Chat Endpoint

API endpoint for agent chat.
"""

import logging
from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat")
async def chat(request: dict):
    """Send message to agent and get response."""
    message = request.get("message", "")
    agent = request.get("agent", "copilot")
    model = request.get("model", "gemini-2.5-pro")
    context = request.get("context", [])

    # Import handlers for fallback responses
    from src.api.ide_handlers import LiveTradingAPIHandler
    trading_handler = LiveTradingAPIHandler()

    # Connect to actual LangGraph agent
    try:
        from src.agents.analyst_v2 import compile_analyst_graph
        from langgraph.checkpoint.memory import MemorySaver
        import uuid

        # Compile the graph with memory checkpointer
        graph = compile_analyst_graph(checkpointer=MemorySaver())

        # Build config for per-session conversation history
        thread_id = f"chat_{agent}_{uuid.uuid4().hex[:8]}"
        config = {"configurable": {"thread_id": thread_id}}

        # Invoke the graph
        result = graph.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config=config
        )

        # Extract the last AI message
        messages = result.get("messages", [])
        response = ""
        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type == "ai":
                response = msg.content
                break
            elif hasattr(msg, "role") and msg.role == "assistant":
                response = msg.content
                break

        if not response:
            response = f"Processed: {message[:50]}..."

    except Exception as e:
        logger.error(f"LangGraph agent invocation failed: {e}")
        # Fallback to keyword-based responses
        response = f"I understand you want to: {message[:50]}... I'll help you with that."

        if "backtest" in message.lower():
            response = "I can run backtests in 4 variants. Which strategy would you like to test?"
        elif "video" in message.lower() or "youtube" in message.lower() or "ingest" in message.lower():
            response = "To process video: 1. Click Video Ingest in EA Management 2. Paste YouTube URL 3. The system will transcribe and analyze."
        elif "bot" in message.lower() or "active" in message.lower():
            bots = trading_handler.get_active_bots()
            response = f"You have {len(bots)} active bots. Go to Live Trading to manage them."

    return {"response": response, "agent": agent, "model": model}

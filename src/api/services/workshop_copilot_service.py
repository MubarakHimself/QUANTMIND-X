"""
Workshop Copilot Service

Handles message routing for the Workshop Copilot with intent classification.
Routes messages to appropriate handlers based on detected intent:
- Simple queries: Direct responses
- Trading requests: Delegate to floor manager
- Workflow requests: Start department workflow
"""

import logging
import os
from typing import List, Dict, Any, Optional

import httpx

from src.config import get_internal_api_base_url

logger = logging.getLogger(__name__)

# =============================================================================
# Intent Classification Keywords
# =============================================================================

TRADING_KEYWORDS = [
    "deploy", "trade", "buy", "sell", "bot", "strategy",
    "backtest", "paper trade", "execute", "order"
]

WORKFLOW_KEYWORDS = [
    "create", "build", "generate", "develop",
    "analyze video", "from video", "youtube"
]


# =============================================================================
# Request/Response Models
# =============================================================================

class WorkshopCopilotRequest:
    """Request model for Workshop Copilot."""

    def __init__(
        self,
        message: str,
        history: Optional[List[Dict[str, Any]]] = None,
        session_id: Optional[str] = None
    ):
        self.message = message
        self.history = history or []
        self.session_id = session_id


class WorkshopCopilotResponse:
    """Response model for Workshop Copilot."""

    def __init__(
        self,
        reply: str,
        delegation: Optional[Dict[str, Any]] = None,
        action_taken: Optional[str] = None
    ):
        self.reply = reply
        self.delegation = delegation
        self.action_taken = action_taken

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reply": self.reply,
            "delegation": self.delegation,
            "action_taken": self.action_taken
        }


# =============================================================================
# Intent Classification
# =============================================================================

def _classify_intent(message: str) -> str:
    """
    Classify the intent of a message.

    Args:
        message: User message to classify

    Returns:
        Intent type: "simple", "trading", or "workflow"
    """
    msg_lower = message.lower()

    # Check for trading keywords first (higher priority)
    if any(keyword in msg_lower for keyword in TRADING_KEYWORDS):
        return "trading"

    # Check for workflow keywords
    if any(keyword in msg_lower for keyword in WORKFLOW_KEYWORDS):
        return "workflow"

    # Default to simple query
    return "simple"


# =============================================================================
# Workshop Copilot Service
# =============================================================================

class WorkshopCopilotService:
    """Service for handling Workshop Copilot messages."""

    def __init__(self, floor_manager_url: Optional[str] = None):
        if floor_manager_url is None:
            floor_manager_url = get_internal_api_base_url()
        self.floor_manager_url = floor_manager_url
        self.logger = logger

    async def handle_message(
        self,
        request: WorkshopCopilotRequest
    ) -> WorkshopCopilotResponse:
        """
        Main entry point for handling workshop copilot messages.

        Args:
            request: WorkshopCopilotRequest with message, history, session_id

        Returns:
            WorkshopCopilotResponse with reply, delegation, action_taken
        """
        # Classify intent
        intent = _classify_intent(request.message)
        self.logger.info(f"Classified intent: {intent} for message: {request.message[:50]}...")

        # Route to appropriate handler
        if intent == "simple":
            return await self._handle_simple_query(request)
        elif intent == "trading":
            return await self._handle_trading_request(request)
        elif intent == "workflow":
            return await self._handle_workflow_request(request)
        else:
            return WorkshopCopilotResponse(
                reply="I didn't understand that request. Try asking about trading, workflows, or general questions.",
                action_taken="unhandled"
            )

    async def _handle_simple_query(
        self,
        request: WorkshopCopilotRequest
    ) -> WorkshopCopilotResponse:
        """
        Handle simple queries - call LLM directly via ProviderRouter.

        Args:
            request: WorkshopCopilotRequest

        Returns:
            WorkshopCopilotResponse with reply from LLM
        """
        try:
            from src.agents.providers.router import get_router
            import httpx

            router = get_router()
            provider = router.primary

            if not provider:
                return WorkshopCopilotResponse(
                    reply="No LLM provider configured. Please set up an API key in Settings → Providers.",
                    action_taken="no_provider"
                )

            # Build messages
            messages = []
            if request.history:
                for h in request.history:
                    role = h.get("role", "user")
                    if role not in ("user", "assistant"):
                        role = "user"
                    messages.append({"role": role, "content": h.get("content", "")})
            messages.append({"role": "user", "content": request.message})

            model = (
                provider.model_list[0].get("id")
                or provider.model_list[0].get("model_id")
                or ""
            ) if provider.model_list else ""

            async with httpx.AsyncClient(timeout=60.0, trust_env=False) as client:
                response = await client.post(
                    f"{provider.base_url}/v1/messages",
                    headers={
                        "Authorization": f"Bearer {provider.api_key}",
                        "Content-Type": "application/json",
                        "Anthropic-Version": "2023-06-01",
                    },
                    json={
                        "model": model,
                        "max_tokens": 1024,
                        "messages": messages,
                    }
                )
                response.raise_for_status()
                data = response.json()
                content = data.get("content", [])
                reply = "No response"
                if isinstance(content, list):
                    for block in content:
                        if block.get("type") == "text":
                            reply = block.get("text", "No response")
                            break

            return WorkshopCopilotResponse(
                reply=reply,
                action_taken="direct_response",
                delegation=None
            )
        except Exception as e:
            logger.error(f"Direct LLM call failed: {e}")
            return WorkshopCopilotResponse(
                reply=f"I encountered an error: {str(e)[:100]}. Please try again.",
                action_taken="error"
            )

    async def _handle_trading_request(
        self,
        request: WorkshopCopilotRequest
    ) -> WorkshopCopilotResponse:
        """
        Handle trading requests - delegate to floor manager.

        Args:
            request: WorkshopCopilotRequest

        Returns:
            WorkshopCopilotResponse delegating to floor manager
        """
        return await self._delegate_to_floor_manager(request.message, history=request.history)

    async def _delegate_to_floor_manager(
        self,
        message: str,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> WorkshopCopilotResponse:
        """Delegate a trading task to Floor Manager."""
        try:
            async with httpx.AsyncClient(trust_env=False) as client:
                response = await client.post(
                    f"{self.floor_manager_url}/api/floor-manager/chat",
                    json={
                        "message": message,
                        "history": history or [],
                        "stream": False,
                    },
                    timeout=60.0
                )
                response.raise_for_status()
                data = response.json()
                # Handle both response formats: ChatResponse uses "content",
                # ChatMessageResponse uses "reply"
                content = data.get("content") or data.get("reply") or ""
                return WorkshopCopilotResponse(
                    reply=content or "Task delegated to Floor Manager",
                    action_taken="delegated_to_floor_manager",
                    delegation={
                        "type": "floor_manager",
                        "department": data.get("delegated_department"),
                        "task_id": data.get("task_id")
                    }
                )
        except Exception as e:
            logger.error(f"Failed to delegate to Floor Manager: {e}")
            return WorkshopCopilotResponse(
                reply=f"I understand you want to: {message}. The Floor Manager is currently unavailable.",
                action_taken="error_delegating"
            )

    async def _handle_workflow_request(
        self,
        request: WorkshopCopilotRequest
    ) -> WorkshopCopilotResponse:
        """
        Handle workflow requests - delegate to Floor Manager for real LLM response.

        Args:
            request: WorkshopCopilotRequest

        Returns:
            WorkshopCopilotResponse with reply from Floor Manager
        """
        self.logger.info(f"Processing workflow request: {request.message}")
        return await self._delegate_to_floor_manager(request.message, history=request.history)

    async def handle_message_stream(
        self,
        message: str,
        history: Optional[List[Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ):
        """
        Stream the copilot response token by token via SSE.

        The copilot always streams a direct LLM response — it is its own agent.
        Delegation to other agents happens via the mail system separately,
        not as part of the streamed chat response.

        Yields dicts:
          {"type": "tool",     "tool": "thinking", "status": "started"}
          {"type": "thinking", "content": "..."}     (Claude models only)
          {"type": "content",  "delta": "..."}
          {"type": "tool",     "tool": "thinking", "status": "completed"}
          {"type": "done"}
        """
        import json as _json
        import httpx as _httpx

        COPILOT_SYSTEM = (
            "You are the QuantMindX Workshop Copilot — a sharp, concise trading platform assistant. "
            "You help the user understand the system, answer questions about strategies and bots, "
            "explain risk metrics, and coordinate tasks with the trading departments. "
            "Keep responses focused and professional. Use markdown for structure when helpful."
        )

        yield {"type": "tool", "tool": "thinking", "status": "started"}

        try:
            from src.agents.providers.router import get_router
            router = get_router()
            runtime_config = router.resolve_runtime_config()
            if not runtime_config or not runtime_config.api_key:
                yield {
                    "type": "error",
                    "error": (
                        "No LLM runtime configured. Set up a provider in Settings "
                        "or define QMX_LLM_* environment variables."
                    ),
                }
                yield {"type": "done"}
                return

            api_key = runtime_config.api_key
            base_url = runtime_config.base_url
            model = runtime_config.model or ""

            # Build messages
            messages = []
            if history:
                for h in history:
                    role = h.get("role", "user")
                    if role not in ("user", "assistant"):
                        role = "user"
                    messages.append({"role": role, "content": h.get("content", "")})
            messages.append({"role": "user", "content": message})

            # Resolve URL
            base_stripped = base_url.rstrip("/")
            messages_url = (
                f"{base_stripped}/messages"
                if base_stripped.endswith("/v1")
                else f"{base_stripped}/v1/messages"
            )

            is_claude = model.lower().startswith("claude")
            is_anthropic_direct = "api.anthropic.com" in base_url
            auth_headers = (
                {"x-api-key": api_key}
                if is_anthropic_direct
                else {"Authorization": f"Bearer {api_key}"}
            )

            request_body: dict = {
                "model": model,
                "system": COPILOT_SYSTEM,
                "messages": messages,
                "max_tokens": 4096,
                "stream": True,
            }
            if is_claude:
                request_body["thinking"] = {"type": "enabled", "budget_tokens": 5000}

            async with _httpx.AsyncClient(timeout=120.0, trust_env=False) as client:
                async with client.stream(
                    "POST",
                    messages_url,
                    headers={
                        **auth_headers,
                        "Content-Type": "application/json",
                        "anthropic-version": "2023-06-01",
                    },
                    json=request_body,
                ) as response:
                    if response.status_code == 401:
                        yield {"type": "error", "error": "LLM auth failed — check API key in Settings → Providers."}
                        yield {"type": "done"}
                        return
                    if response.status_code != 200:
                        text = await response.aread()
                        yield {"type": "error", "error": f"LLM error ({response.status_code}): {text[:200]}"}
                        yield {"type": "done"}
                        return

                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data_str = line[6:].strip()
                        if not data_str or data_str == "[DONE]":
                            continue
                        try:
                            event = _json.loads(data_str)
                            event_type = event.get("type", "")
                            if event_type == "content_block_start":
                                block = event.get("content_block", {})
                                if block.get("type") == "thinking":
                                    yield {"type": "thinking", "content": ""}
                            elif event_type == "content_block_delta":
                                delta = event.get("delta", {})
                                dtype = delta.get("type", "")
                                if dtype == "thinking_delta":
                                    yield {"type": "thinking", "content": delta.get("thinking", "")}
                                elif dtype == "text_delta":
                                    yield {"type": "content", "delta": delta.get("text", "")}
                        except Exception:
                            continue

        except _httpx.ConnectError:
            yield {"type": "error", "error": f"Cannot connect to LLM provider at {base_url}."}
        except Exception as e:
            logger.error(f"Copilot stream failed: {e}")
            yield {"type": "error", "error": f"Streaming failed: {str(e)[:100]}"}

        yield {"type": "tool", "tool": "thinking", "status": "completed"}
        yield {"type": "done"}


# =============================================================================
# Singleton
# =============================================================================

_workshop_copilot_service: Optional[WorkshopCopilotService] = None


def get_workshop_copilot_service(floor_manager_url: Optional[str] = None) -> WorkshopCopilotService:
    """Get the Workshop Copilot service singleton."""
    global _workshop_copilot_service
    if _workshop_copilot_service is None:
        _workshop_copilot_service = WorkshopCopilotService(floor_manager_url)
    return _workshop_copilot_service

"""
SDK Orchestrator for Claude Agent SDK

Replaces ClaudeOrchestrator's subprocess spawning with direct SDK calls.
Eliminates nested session error and enables true streaming.
Supports multiple providers: Anthropic and Z.AI.
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional, AsyncGenerator, List

from src.agents.sdk_config import (
    get_sdk_agent_config,
    SDKAgentConfig,
    load_system_prompt,
    get_provider_config,
    get_model_for_tier,
)

logger = logging.getLogger(__name__)

# Check if SDK is available
SDK_AVAILABLE = False
try:
    from anthropic import Anthropic
    SDK_AVAILABLE = True
    logger.info("Anthropic SDK available for direct API calls")
except ImportError:
    logger.warning("Anthropic SDK not installed. Using fallback mode.")


class SDKOrchestrator:
    """
    Orchestrator using Claude Agent SDK directly.

    No subprocess spawning - uses in-process SDK calls.
    Supports both stateless queries and streaming.
    Supports multiple providers (Anthropic, Z.AI).
    """

    def __init__(self):
        """Initialize the SDK orchestrator."""
        self._client = None
        self._active_sessions: Dict[str, Any] = {}
        self._provider_config = get_provider_config()
        logger.info(f"SDKOrchestrator initialized with provider: {self._provider_config['provider']}")

    @property
    def client(self):
        """Lazy-initialize Anthropic-compatible client."""
        if self._client is None and SDK_AVAILABLE:
            client_kwargs = {"api_key": self._provider_config["api_key"]}

            # Add base_url for non-Anthropic providers (e.g., Z.AI)
            if self._provider_config["base_url"]:
                client_kwargs["base_url"] = self._provider_config["base_url"]

            self._client = Anthropic(**client_kwargs)
            logger.info(f"Client initialized for provider: {self._provider_config['provider']}")
        return self._client

    def get_model(self, tier: str = "sonnet") -> str:
        """Get model name for the current provider."""
        return get_model_for_tier(tier)

    async def invoke(
        self,
        agent_id: str,
        messages: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Invoke an agent using direct API call.

        Args:
            agent_id: Agent identifier (analyst, quantcode, etc.)
            messages: List of message dicts with role and content
            context: Optional context dictionary
            session_id: Optional session ID for continuity

        Returns:
            Result dictionary with output and metadata
        """
        config = get_sdk_agent_config(agent_id)
        system_prompt = load_system_prompt(config)

        try:
            if not SDK_AVAILABLE or not self.client:
                # Fallback mode - return informative response
                return {
                    "status": "completed",
                    "output": f"SDK mode not available. Agent {agent_id} received: {messages[-1].get('content', '') if messages else ''}",
                    "agent_id": agent_id,
                    "mode": "fallback",
                    "completed_at": datetime.utcnow().isoformat(),
                }

            # Build messages for API
            api_messages = self._build_api_messages(messages, context)

            # Get provider-aware model
            model = self.get_model("sonnet")  # Use sonnet tier by default

            # Make API call
            response = self.client.messages.create(
                model=model,
                max_tokens=4096,
                system=system_prompt[:100000] if system_prompt else None,  # Limit system prompt
                messages=api_messages,
            )

            # Extract text content
            output = ""
            for block in response.content:
                if hasattr(block, 'text'):
                    output += block.text

            return {
                "status": "completed",
                "output": output,
                "agent_id": agent_id,
                "model": model,
                "provider": self._provider_config["provider"],
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
                "completed_at": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"SDK invocation failed for {agent_id}: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "agent_id": agent_id,
                "completed_at": datetime.utcnow().isoformat(),
            }

    async def stream(
        self,
        agent_id: str,
        messages: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream agent output using SDK streaming.

        Yields events:
        - started: Session initialized
        - text: Text content delta
        - completed: Session finished
        - error: Error occurred

        Args:
            agent_id: Agent identifier (analyst, quantcode, etc.)
            messages: List of message dicts with role and content
            context: Optional context dictionary
            session_id: Optional session ID for continuity

        Yields:
            Event dictionaries with type, agent_id, and relevant data
        """
        config = get_sdk_agent_config(agent_id)
        system_prompt = load_system_prompt(config)

        session_key = session_id or f"{agent_id}_{datetime.utcnow().timestamp()}"

        # Yield started event
        yield {
            "type": "started",
            "agent_id": agent_id,
            "session_id": session_key,
            "timestamp": datetime.utcnow().isoformat(),
        }

        try:
            if not SDK_AVAILABLE or not self.client:
                # Fallback mode
                yield {
                    "type": "text",
                    "delta": f"SDK streaming mode. Processing request for {agent_id}...",
                    "agent_id": agent_id,
                }
                yield {
                    "type": "completed",
                    "agent_id": agent_id,
                    "output": "SDK not fully configured. Check ANTHROPIC_API_KEY.",
                    "timestamp": datetime.utcnow().isoformat(),
                }
                return

            # Build messages
            api_messages = self._build_api_messages(messages, context)

            # Get provider-aware model
            model = self.get_model("sonnet")  # Use sonnet tier by default

            # Stream from API
            full_output = ""
            with self.client.messages.stream(
                model=model,
                max_tokens=4096,
                system=system_prompt[:100000] if system_prompt else None,
                messages=api_messages,
            ) as stream:
                for text in stream.text_stream:
                    full_output += text
                    yield {
                        "type": "text",
                        "delta": text,
                        "agent_id": agent_id,
                    }

                # Get final message for usage stats
                final_message = stream.get_final_message()

            yield {
                "type": "completed",
                "agent_id": agent_id,
                "output": full_output,
                "model": model,
                "provider": self._provider_config["provider"],
                "usage": {
                    "input_tokens": final_message.usage.input_tokens,
                    "output_tokens": final_message.usage.output_tokens,
                },
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"SDK streaming failed for {agent_id}: {e}")
            yield {
                "type": "error",
                "error": str(e),
                "agent_id": agent_id,
                "timestamp": datetime.utcnow().isoformat(),
            }

    def _build_api_messages(
        self,
        messages: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        """
        Build messages list for Anthropic API.

        Args:
            messages: List of message dictionaries
            context: Optional context to prepend

        Returns:
            List of API-formatted message dictionaries
        """
        api_messages = []

        # Add context as system context if provided
        if context:
            context_str = json.dumps(context, indent=2)
            api_messages.append({
                "role": "user",
                "content": f"[Context]\n{context_str}\n\n[Request]"
            })

        # Add conversation messages
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Map roles (assistant -> assistant, everything else -> user)
            api_role = "assistant" if role == "assistant" else "user"
            api_messages.append({
                "role": api_role,
                "content": content,
            })

        return api_messages

    async def cancel(self, session_id: str) -> bool:
        """
        Cancel an active session.

        Args:
            session_id: The session identifier to cancel

        Returns:
            True if session was found and cancelled, False otherwise
        """
        if session_id in self._active_sessions:
            del self._active_sessions[session_id]
            logger.info(f"Cancelled session: {session_id}")
            return True
        return False

    def get_active_sessions(self) -> List[str]:
        """
        Get list of active session IDs.

        Returns:
            List of active session identifiers
        """
        return list(self._active_sessions.keys())


# Global instance
_sdk_orchestrator: Optional[SDKOrchestrator] = None


def get_sdk_orchestrator() -> SDKOrchestrator:
    """
    Get the global SDK orchestrator instance.

    Returns:
        Singleton SDKOrchestrator instance
    """
    global _sdk_orchestrator
    if _sdk_orchestrator is None:
        _sdk_orchestrator = SDKOrchestrator()
    return _sdk_orchestrator

"""
Department Head Base Class

Base class for all department heads. Provides:
- Isolated markdown-based memory per department
- Tool access control with permission filtering
- Mail inbox checking
- Cross-department messaging
- Worker spawning capability
- SDK Orchestrator integration

Model Tier: Sonnet (balanced reasoning)
"""
import json
import logging
import os
import uuid
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any

import anthropic

from src.agents.providers.router import get_router
from src.agents.departments.types import (
    Department,
    DepartmentHeadConfig,
    get_model_tier,
)
from src.agents.departments.department_mail import (
    DepartmentMailService,
    RedisDepartmentMailService,
    get_redis_mail_service,
    MessageType,
    Priority,
)
from src.agents.departments.memory_manager import DepartmentMemoryManager
from src.agents.departments.tool_access import ToolAccessController
from src.agents.departments.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


class DepartmentHead:
    """
    Base class for Department Heads.

    Each department head:
    - Uses Sonnet model tier for balanced reasoning
    - Checks mail inbox for dispatched tasks
    - Can spawn workers for complex sub-tasks
    - Sends results to other departments via mail

    Attributes:
        department: The department this head leads
        config: Department configuration
        mail_service: Mail service for communication
        spawner: Agent spawner for worker creation
        model_tier: Always "sonnet" for department heads
    """

    STANDARD_TOOLS = [
        {
            "name": "send_mail",
            "description": "Send a message to another department head or the floor manager",
            "input_schema": {
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "Department name: research, development, trading, or floor_manager",
                    },
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["to", "body"],
            },
        },
        {
            "name": "read_memory",
            "description": "Read relevant memory nodes for a given query",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for in memory",
                    }
                },
                "required": ["query"],
            },
        },
        {
            "name": "write_opinion",
            "description": "Write an opinion or insight to long-term memory",
            "input_schema": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The opinion or insight to save",
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Confidence 0.0-1.0",
                        "default": 0.7,
                    },
                },
                "required": ["content"],
            },
        },
    ]

    def __init__(
        self,
        config: DepartmentHeadConfig,
        mail_db_path: str = ".quantmind/department_mail.db",
        use_redis_mail: bool = True,
    ):
        """
        Initialize the Department Head.

        Args:
            config: Department configuration
            mail_db_path: Path to mail database (used if use_redis_mail=False)
            use_redis_mail: If True, use Redis Streams for mail (recommended)
        """
        self.config = config
        self.department = config.department
        self.agent_type = config.agent_type
        self.system_prompt = config.system_prompt
        self.sub_agents = config.sub_agents
        self.memory_namespace = config.memory_namespace
        self.model_tier = "sonnet"

        # Initialize isolated memory for this department
        self.memory_manager = DepartmentMemoryManager(
            department=self.department,
            base_path=".quantmind/departments"
        )

        # Initialize tool access controller
        self.tool_access = ToolAccessController(department=self.department)

        # Initialize tool registry
        self.tool_registry = ToolRegistry()
        self._tools = self.tool_registry.get_tools_for_department(self.department)

        # Initialize mail service (Redis Streams recommended)
        if use_redis_mail:
            self.mail_service = get_redis_mail_service(
                consumer_name=f"{self.department.value}-{uuid.uuid4().hex[:8]}"
            )
        else:
            self.mail_service = DepartmentMailService(db_path=mail_db_path)
        self._init_spawner()

        # Hooks system — mirrors BaseAgent hook lifecycle
        self._hooks: List[Dict[str, Any]] = []

        logger.info(f"DepartmentHead initialized: {self.department.value} with {len(self._tools)} tools")

    # ── Hook system (Agent SDK compliance) ────────────────────────────────────

    def register_hook(
        self,
        event: str,
        handler,
        tool_name_pattern: Optional[str] = None,
    ) -> None:
        """Register a lifecycle hook.

        Args:
            event: PRE_TOOL_USE | POST_TOOL_USE | POST_TOOL_USE_FAILURE | STOP
            handler: Callable or async callable
            tool_name_pattern: Optional regex pattern to match tool names
        """
        import re
        self._hooks.append({
            "event": event,
            "handler": handler,
            "pattern": re.compile(tool_name_pattern) if tool_name_pattern else None,
        })

    async def _fire_hook(self, event: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Fire all hooks matching the event. Returns first non-None result."""
        import asyncio
        tool_name = context.get("tool_name", "")
        for hook in self._hooks:
            if hook["event"] != event:
                continue
            if hook["pattern"] and not hook["pattern"].match(tool_name):
                continue
            try:
                result = hook["handler"](context)
                if asyncio.iscoroutine(result):
                    result = await result
                if result is not None:
                    return result
            except Exception as e:
                logger.debug(f"Hook {event} error: {e}")
        return None

    def _publish_thought(self, thought: str, thought_type: str = "reasoning") -> None:
        """Publish to SSE thought stream for UI display (Agent SDK §6.2)."""
        # Map internal types to UI-recognized types
        _map = {
            "reasoning": "reasoning", "tool_call": "action",
            "action": "action", "observation": "observation",
            "decision": "decision", "error": "observation",
        }
        try:
            from src.api.agent_thought_stream_endpoints import get_thought_publisher
            get_thought_publisher().publish(
                department=self.department.value,
                thought=thought,
                thought_type=_map.get(thought_type, "reasoning"),
            )
        except Exception:
            pass

    def get_available_tools(self) -> Dict[str, Any]:
        """
        Get all available tools for this department.

        Returns:
            Dictionary of tool_name to tool_instance
        """
        return self._tools

    def get_tool(self, tool_name: str) -> Optional[Any]:
        """
        Get a specific tool instance.

        Args:
            tool_name: Name of the tool

        Returns:
            Tool instance or None if not available
        """
        return self._tools.get(tool_name)

    def refresh_tools(self) -> None:
        """
        Refresh available tools from registry.

        Useful after permission changes.
        """
        self._tools = self.tool_registry.get_tools_for_department(self.department)
        logger.info(f"Refreshed tools for {self.department.value}: {len(self._tools)} available")

    def has_tool_access(self, tool_name: str, permission: str = "read") -> bool:
        """
        Check if department has access to a specific tool.

        Args:
            tool_name: Name of the tool
            permission: Permission level to check ("read" or "write")

        Returns:
            True if department has access, False otherwise
        """
        from src.agents.departments.tool_access import ToolPermission

        perm = ToolPermission.READ if permission == "read" else ToolPermission.WRITE
        return self.tool_access.can_access(tool_name, perm)

    def add_memory(self, content: str, memory_type: str = "note") -> None:
        """
        Add a memory to this department's isolated memory.

        Args:
            content: Memory content
            memory_type: Type of memory (note, observation, decision, etc.)
        """
        self.memory_manager.add_memory(content, memory_type)

    def read_memory(self) -> str:
        """
        Read this department's memory.

        Returns:
            Department memory content
        """
        return self.memory_manager.read_memory()

    def search_memory(self, query: str, limit: int = 10) -> List:
        """
        Search this department's memory.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of memory results
        """
        return self.memory_manager.search(query, limit)

    def _init_spawner(self):
        """Initialize the agent spawner."""
        try:
            from src.agents.subagent.spawner import get_spawner
            self.spawner = get_spawner()
        except ImportError:
            logger.warning("Agent spawner not available")
            self.spawner = None

    def check_mail(self, unread_only: bool = True) -> List[Any]:
        """
        Check inbox for messages.

        Args:
            unread_only: Only return unread messages

        Returns:
            List of messages
        """
        return self.mail_service.check_inbox(
            dept=self.department.value,
            unread_only=unread_only,
        )

    def send_result(
        self,
        to_dept: Department,
        subject: str,
        body: str,
        priority: str = "normal",
    ) -> Dict[str, Any]:
        """
        Send a result message to another department.

        Args:
            to_dept: Target department
            subject: Message subject
            body: Message body
            priority: Message priority

        Returns:
            Send result
        """
        priority_map = {
            "low": Priority.LOW,
            "normal": Priority.NORMAL,
            "high": Priority.HIGH,
            "urgent": Priority.URGENT,
        }

        message = self.mail_service.send(
            from_dept=self.department.value,
            to_dept=to_dept.value,
            type=MessageType.RESULT,
            subject=subject,
            body=body,
            priority=priority_map.get(priority.lower(), Priority.NORMAL),
        )

        logger.info(f"Sent result to {to_dept.value}: {message.id}")

        return {
            "status": "sent",
            "message_id": message.id,
            "to_dept": to_dept.value,
        }

    def send_question(
        self,
        to_dept: Department,
        subject: str,
        body: str,
    ) -> Dict[str, Any]:
        """
        Send a question to another department.

        Args:
            to_dept: Target department
            subject: Message subject
            body: Message body

        Returns:
            Send result
        """
        message = self.mail_service.send(
            from_dept=self.department.value,
            to_dept=to_dept.value,
            type=MessageType.QUESTION,
            subject=subject,
            body=body,
        )

        return {
            "status": "sent",
            "message_id": message.id,
            "to_dept": to_dept.value,
        }

    def spawn_worker(
        self,
        worker_type: str,
        task: str,
        input_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Spawn a worker agent.

        Args:
            worker_type: Type of worker to spawn
            task: Task description
            input_data: Optional input data

        Returns:
            Spawn result
        """
        # Validate worker type
        if worker_type not in self.sub_agents:
            return {
                "status": "invalid_worker_type",
                "worker_type": worker_type,
                "available_workers": self.sub_agents,
            }

        if not self.spawner:
            return {
                "status": "spawner_unavailable",
                "worker_type": worker_type,
            }

        try:
            from src.agents.subagent.spawner import SubAgentConfig

            # Prepare input data with tools
            worker_input = input_data or {"task": task}
            worker_input["available_tools"] = list(self._tools.keys())
            worker_input["department"] = self.department.value

            config = SubAgentConfig(
                agent_type=worker_type,
                name=f"{self.department.value}_{worker_type}",
                parent_agent_id=f"dept_head_{self.department.value}",
                input_data=worker_input,
                pool_key=self.department.value,
            )

            agent = self.spawner.spawn(config)
            logger.info(f"Spawned worker {agent.id} for {self.department.value} with {len(self._tools)} tools")

            return {
                "status": "spawned",
                "agent_id": agent.id,
                "worker_type": worker_type,
                "tools_available": len(self._tools),
            }

        except Exception as e:
            logger.error(f"Failed to spawn worker: {e}")
            return {
                "status": "spawn_failed",
                "error": str(e),
            }

    # ------------------------------------------------------------------
    # LLM invocation infrastructure
    # ------------------------------------------------------------------

    def _build_system_prompt(
        self,
        canvas_context: Optional[dict] = None,
        memory_nodes: Optional[list] = None,
    ) -> str:
        """
        Construct the full system prompt for Claude.

        Combines the department system prompt with relevant memory nodes
        and canvas context.

        Args:
            canvas_context: Optional canvas context dict to include.
            memory_nodes: Optional list of memory node dicts to include.

        Returns:
            Assembled system prompt string.
        """
        # Load user-saved system prompt override from settings (if any)
        try:
            from src.api.settings_endpoints import load_settings
            saved_prompts = load_settings().get("agents", {}).get("system_prompts", {})
            # Check both "research" and "research_head" key formats (UI saves as "_head")
            saved = (
                saved_prompts.get(f"{self.department.value}_head", "")
                or saved_prompts.get(self.department.value, "")
            )
            base_prompt = saved if saved.strip() else self.system_prompt
        except Exception:
            base_prompt = self.system_prompt
        parts = [base_prompt]

        if memory_nodes:
            memory_section = "## Relevant Memory\n"
            memory_section += "\n".join(
                node.get("content", "") for node in memory_nodes if node.get("content")
            )
            parts.append(memory_section)

        if canvas_context:
            parts.append(
                "## Current Canvas Context\n"
                + json.dumps(canvas_context, indent=2)
            )

        parts.append(
            "## Context Guard\n"
            "Do NOT include raw file contents or code blobs in your reasoning. "
            "Reference files by path only."
        )

        return "\n\n".join(parts)

    async def _read_relevant_memory(self, query: str, limit: int = 5) -> list:
        """
        Read relevant OPINION and FACT nodes from graph memory.

        Args:
            query: Search query string.
            limit: Maximum number of nodes to return.

        Returns:
            List of dicts with keys: content, type, created_at.
        """
        try:
            from src.memory.graph.facade import get_graph_memory

            memory = get_graph_memory()
            nodes = await memory.search_nodes(
                query=query,
                node_types=["OPINION", "FACT"],
                limit=limit,
            )
            return [
                {
                    "content": n.content,
                    "type": n.node_type,
                    "created_at": n.created_at,
                }
                for n in nodes
            ]
        except Exception:
            return []

    async def send_mail(self, to: str, subject: str, body: str) -> str:
        """
        Send mail to another department or the floor manager.

        Args:
            to: Destination department name string.
            subject: Message subject.
            body: Message body.

        Returns:
            Confirmation string.
        """
        try:
            message = self.mail_service.send(
                from_dept=self.department.value,
                to_dept=to,
                type=MessageType.RESULT,
                subject=subject,
                body=body,
            )
            logger.info(f"Mail sent to {to}: {message.id}")
            return f"Mail sent to {to} (id={message.id})"
        except Exception as e:
            logger.error(f"send_mail failed: {e}")
            return f"Mail send failed: {e}"

    async def _execute_tool(self, tool_name: str, tool_input: dict) -> str:
        """
        Dispatch a tool call to the appropriate handler.

        Args:
            tool_name: Name of the tool to invoke.
            tool_input: Parsed input dict from the model's tool_use block.

        Returns:
            String result to feed back as a tool_result message.
        """
        if tool_name == "send_mail":
            return await self.send_mail(
                to=tool_input["to"],
                subject=tool_input.get("subject", "(no subject)"),
                body=tool_input.get("body", ""),
            )

        if tool_name == "read_memory":
            nodes = await self._read_relevant_memory(
                query=tool_input.get("query", "")
            )
            if not nodes:
                return "No relevant memory found."
            lines = []
            for n in nodes:
                lines.append(f"[{n.get('type', 'UNKNOWN')}] {n.get('content', '')}")
            return "\n".join(lines)

        if tool_name in ("write_memory", "write_opinion"):
            await self._write_opinion_node(
                content=tool_input["content"],
                confidence=float(tool_input.get("confidence", 0.7)),
            )
            return "Memory saved"

        # Check tool registry for non-standard tools
        registry_tool = self._tools.get(tool_name)
        if registry_tool:
            # If the registry tool requires approval, request HITL
            if getattr(registry_tool, "requires_approval", False):
                try:
                    from src.agents.approval_manager import (
                        get_approval_manager, ApprovalType, ApprovalUrgency,
                    )
                    self._publish_thought(
                        f"Tool {tool_name} requires human approval — waiting...",
                        thought_type="observation",
                    )
                    req = get_approval_manager().request_approval(
                        approval_type=ApprovalType.TOOL_EXECUTION,
                        title=f"Tool: {tool_name}",
                        description=(
                            f"Department {self.department.value} wants to execute "
                            f"'{tool_name}': {json.dumps(tool_input)[:300]}"
                        ),
                        department=self.department.value,
                        agent_id=f"dept_head_{self.department.value}",
                        tool_name=tool_name,
                        tool_input=tool_input,
                        urgency=ApprovalUrgency.HIGH,
                    )
                    import asyncio
                    resolved = await get_approval_manager().wait_for_approval(
                        req.id, timeout=600,
                    )
                    if resolved.status.value != "approved":
                        reason = resolved.rejection_reason or "Rejected"
                        return f"Tool {tool_name} denied by human: {reason}"
                except Exception as e:
                    logger.warning(f"HITL tool approval failed: {e}")

            # Execute the registry tool
            if callable(getattr(registry_tool, "execute", None)):
                import asyncio
                if asyncio.iscoroutinefunction(registry_tool.execute):
                    result = await registry_tool.execute(tool_input)
                else:
                    result = registry_tool.execute(tool_input)
                return str(result)

        return f"Tool {tool_name} not found"

    async def _invoke_claude(
        self,
        task: str,
        canvas_context: Optional[dict] = None,
        tools: Optional[list] = None,
    ) -> dict:
        """
        Make an Anthropic SDK call with an agentic loop that handles tool use.

        Runs up to 10 iterations.  Each iteration:
        1. Calls client.messages.create().
        2. If stop_reason == "tool_use", executes every tool_use block and
           appends the results as a user tool_result turn, then loops.
        3. Stops when stop_reason != "tool_use" (i.e. "end_turn").

        When ``tools`` is not provided the loop defaults to STANDARD_TOOLS so
        all department heads always have mail + memory tools available.

        Args:
            task: The user-facing task string sent as the human turn.
            canvas_context: Optional canvas context to embed in the prompt.
            tools: Optional list of Anthropic tool definitions to pass.
                   Defaults to STANDARD_TOOLS.

        Returns:
            Dict with keys: content, tool_calls, model, usage, stop_reason.
            On error includes an "error" key.
        """
        if tools is None:
            tools = self.STANDARD_TOOLS

        memory_nodes = await self._read_relevant_memory(task)
        system = self._build_system_prompt(
            canvas_context=canvas_context,
            memory_nodes=memory_nodes,
        )

        # Resolve provider via ProviderRouter; fall back to env vars if none configured
        provider = get_router().primary
        if provider:
            client = anthropic.AsyncAnthropic(
                api_key=provider.api_key,
                base_url=provider.base_url,
            )
            # Pick model by tier from provider's model list
            ml = provider.model_list
            if ml:
                model = ml[0].get("id") or ml[0].get("model_id", "claude-sonnet-4-6")
            else:
                model = "claude-sonnet-4-6"
        else:
            client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            model = os.getenv("ANTHROPIC_MODEL_SONNET", "claude-sonnet-4-6")
            if self.model_tier == "opus":
                model = os.getenv("ANTHROPIC_MODEL_OPUS", "claude-opus-4-6")

        messages: list = [{"role": "user", "content": task}]

        base_kwargs: dict = {
            "model": model,
            "max_tokens": 4096,
            "system": system,
            "tools": tools,
        }

        MAX_ITERATIONS = 10
        total_input_tokens = 0
        total_output_tokens = 0
        final_text = ""
        all_tool_calls: list = []
        response = None

        self._publish_thought(
            f"Starting task: {task[:100]}...",
            thought_type="reasoning",
        )

        try:
            for iteration in range(MAX_ITERATIONS):
                response = await client.messages.create(
                    **base_kwargs,
                    messages=messages,
                )

                total_input_tokens += response.usage.input_tokens
                total_output_tokens += response.usage.output_tokens

                # Publish thinking blocks if present (extended thinking)
                for block in response.content:
                    if hasattr(block, "type") and block.type == "thinking":
                        self._publish_thought(
                            getattr(block, "thinking", "")[:500],
                            thought_type="reasoning",
                        )

                if response.stop_reason != "tool_use":
                    # Collect any final text blocks and exit the loop
                    for block in response.content:
                        if block.type == "text":
                            final_text += block.text
                    # Fire STOP hook (Agent SDK lifecycle)
                    await self._fire_hook("STOP", {
                        "department": self.department.value,
                        "iterations": iteration + 1,
                        "stop_reason": response.stop_reason,
                    })
                    self._publish_thought(
                        f"Turn {iteration + 1}: Final response ({len(final_text)} chars)",
                        thought_type="decision",
                    )
                    break

                # --- tool_use turn ---
                tool_use_blocks = [
                    b for b in response.content if b.type == "tool_use"
                ]

                # Also capture any text emitted alongside tool calls
                for block in response.content:
                    if block.type == "text":
                        final_text += block.text

                # Append the assistant message (with tool_use blocks) to history
                messages.append({"role": "assistant", "content": response.content})

                # Execute each tool and build the tool_result user message
                tool_result_content = []
                for block in tool_use_blocks:
                    all_tool_calls.append(
                        {"name": block.name, "input": block.input, "id": block.id}
                    )

                    # Fire PRE_TOOL_USE hook (Agent SDK lifecycle)
                    pre_result = await self._fire_hook("PRE_TOOL_USE", {
                        "tool_name": block.name,
                        "tool_input": block.input,
                        "tool_use_id": block.id,
                        "department": self.department.value,
                    })
                    # If hook returns a decision to deny, skip tool execution
                    if isinstance(pre_result, dict) and pre_result.get("deny"):
                        tool_result_content.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": pre_result.get("reason", "Tool denied by hook"),
                            "is_error": True,
                        })
                        continue

                    self._publish_thought(
                        f"Turn {iteration + 1}: Calling {block.name}",
                        thought_type="tool_call",
                    )

                    logger.info(
                        f"Executing tool '{block.name}' (id={block.id}) "
                        f"for {self.department.value}"
                    )

                    try:
                        result_str = await self._execute_tool(block.name, block.input)
                        # Fire POST_TOOL_USE hook
                        await self._fire_hook("POST_TOOL_USE", {
                            "tool_name": block.name,
                            "tool_input": block.input,
                            "tool_use_id": block.id,
                            "result": result_str,
                            "department": self.department.value,
                        })
                    except Exception as tool_err:
                        # Fire POST_TOOL_USE_FAILURE hook
                        await self._fire_hook("POST_TOOL_USE_FAILURE", {
                            "tool_name": block.name,
                            "tool_input": block.input,
                            "tool_use_id": block.id,
                            "error": str(tool_err),
                            "department": self.department.value,
                        })
                        self._publish_thought(
                            f"Tool {block.name} failed: {str(tool_err)[:100]}",
                            thought_type="error",
                        )
                        result_str = f"Error: {tool_err}"

                    tool_result_content.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result_str,
                        }
                    )

                messages.append({"role": "user", "content": tool_result_content})

                if iteration == MAX_ITERATIONS - 1:
                    logger.warning(
                        f"Agentic loop hit max iterations ({MAX_ITERATIONS}) "
                        f"for {self.department.value}"
                    )

            self._publish_thought(
                f"Task complete: {iteration + 1} turns, "
                f"{total_input_tokens + total_output_tokens} tokens",
                thought_type="decision",
            )

            return {
                "content": final_text,
                "tool_calls": all_tool_calls,
                "model": response.model if response else model,
                "usage": {
                    "input_tokens": total_input_tokens,
                    "output_tokens": total_output_tokens,
                },
                "stop_reason": response.stop_reason if response else "unknown",
            }

        except anthropic.AuthenticationError:
            logger.error(
                "Anthropic authentication failed — check ANTHROPIC_API_KEY"
            )
            self._publish_thought("Authentication failed", thought_type="error")
            return {"content": "", "tool_calls": [], "error": "auth_failed"}
        except Exception as e:
            logger.error(f"Claude invocation failed: {e}")
            self._publish_thought(f"Error: {str(e)[:100]}", thought_type="error")
            return {"content": "", "tool_calls": [], "error": str(e)}

    async def _write_opinion_node(
        self,
        content: str,
        confidence: float = 0.7,
        tags: Optional[list] = None,
    ) -> None:
        """
        Write an OPINION node to graph memory after task completion.

        Args:
            content: The opinion content to store.
            confidence: Confidence score (0.0–1.0).
            tags: Optional list of string tags.
        """
        try:
            from src.memory.graph.facade import get_graph_memory

            memory = get_graph_memory()
            await memory.add_node(
                node_type="OPINION",
                content=content,
                metadata={
                    "department": self.department.value,
                    "confidence": confidence,
                    "tags": tags or [],
                    "source": "department_head",
                },
            )
        except Exception as e:
            logger.warning(f"Failed to write opinion node: {e}")

    def close(self):
        """Clean up resources."""
        if self.mail_service:
            self.mail_service.close()
        logger.info(f"DepartmentHead closed: {self.department.value}")

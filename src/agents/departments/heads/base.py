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
import re
import uuid
from abc import ABC, abstractmethod
from types import SimpleNamespace
from typing import Dict, List, Optional, Any

import anthropic
import httpx

from src.agents.providers.router import get_router
from src.agents.core.base_agent import HookEvent
from src.agents.departments.types import (
    Department,
    DepartmentHeadConfig,
    get_model_tier,
)
from src.agents.departments.department_mail import (
    DepartmentMailService,
    create_mail_service,
    MessageType,
    Priority,
)
from src.agents.departments.memory_manager import DepartmentMemoryManager
from src.agents.departments.tool_access import ToolAccessController
from src.agents.departments.tool_registry import ToolRegistry
from src.agents.prompts.department_contracts import compose_department_head_prompt

logger = logging.getLogger(__name__)


LEGACY_HOOK_EVENT_ALIASES = {
    "PRE_TOOL_USE": HookEvent.PRE_TOOL_USE,
    "POST_TOOL_USE": HookEvent.POST_TOOL_USE,
    "POST_TOOL_USE_FAILURE": HookEvent.POST_TOOL_USE_FAILURE,
    "STOP": HookEvent.STOP,
    "SUBAGENT_START": HookEvent.SUBAGENT_START,
    "SUBAGENT_STOP": HookEvent.SUBAGENT_STOP,
}


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
        {
            "name": "list_resources",
            "description": "List workspace resources available to this department",
            "input_schema": {
                "type": "object",
                "properties": {
                    "canvases": {"type": "array", "items": {"type": "string"}},
                    "tabs": {"type": "array", "items": {"type": "string"}},
                    "types": {"type": "array", "items": {"type": "string"}},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 500, "default": 100},
                },
            },
        },
        {
            "name": "search_resources",
            "description": "Search workspace resources naturally by query",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "canvases": {"type": "array", "items": {"type": "string"}},
                    "tabs": {"type": "array", "items": {"type": "string"}},
                    "types": {"type": "array", "items": {"type": "string"}},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 200, "default": 20},
                },
                "required": ["query"],
            },
        },
        {
            "name": "read_resource",
            "description": "Read one workspace resource by resource_id",
            "input_schema": {
                "type": "object",
                "properties": {
                    "resource_id": {"type": "string"},
                    "max_chars": {"type": "integer", "minimum": 1024, "maximum": 500000, "default": 120000},
                },
                "required": ["resource_id"],
            },
        },
        {
            "name": "create_task",
            "description": "Create a department Kanban task",
            "input_schema": {
                "type": "object",
                "properties": {
                    "department": {"type": "string"},
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "priority": {"type": "string", "default": "normal"},
                    "workflow_id": {"type": "string"},
                    "strategy_id": {"type": "string"},
                },
                "required": ["title"],
            },
        },
        {
            "name": "update_task",
            "description": "Update an existing task status",
            "input_schema": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "status": {"type": "string"},
                    "result": {"type": "string"},
                },
                "required": ["task_id", "status"],
            },
        },
        {
            "name": "spawn_subagent",
            "description": "Spawn a department sub-agent worker for a bounded task",
            "input_schema": {
                "type": "object",
                "properties": {
                    "worker_type": {"type": "string"},
                    "task": {"type": "string"},
                    "input_data": {"type": "object"},
                },
                "required": ["worker_type", "task"],
            },
        },
        {
            "name": "update_workflow_state",
            "description": "Emit a workflow-state update event for UI projection",
            "input_schema": {
                "type": "object",
                "properties": {
                    "workflow_id": {"type": "string"},
                    "state": {"type": "string"},
                    "step": {"type": "string"},
                    "note": {"type": "string"},
                },
                "required": ["workflow_id", "state"],
            },
        },
    ]

    _MAX_PROMPT_HINTS = 24
    _MAX_PROMPT_ATTACHMENTS = 12

    _MINIMAX_TOOL_CALL_PATTERN = re.compile(
        r"<minimax:tool_call>\s*(.*?)\s*</minimax:tool_call>",
        re.DOTALL | re.IGNORECASE,
    )
    _MINIMAX_INVOKE_PATTERN = re.compile(
        r"<invoke\s+name=\"(?P<name>[^\"]+)\">\s*(?P<body>.*?)\s*</invoke>",
        re.DOTALL | re.IGNORECASE,
    )
    _MINIMAX_PARAMETER_PATTERN = re.compile(
        r"<parameter\s+name=\"(?P<name>[^\"]+)\">(?P<value>.*?)</parameter>",
        re.DOTALL | re.IGNORECASE,
    )

    def _summarize_canvas_context_for_prompt(self, canvas_context: dict) -> dict:
        """
        Build a bounded prompt-safe summary from canvas context.

        We intentionally avoid injecting raw payloads or large blobs into the
        system prompt. The model receives references (resource ids/paths) and
        can fetch details via tools when needed.
        """
        if not isinstance(canvas_context, dict):
            return {}

        summary: Dict[str, Any] = {}

        for key in (
            "canvas",
            "department",
            "session_type",
            "workflow_id",
            "workflow_step",
            "workflow_run_id",
            "provider",
            "llm_provider",
            "model",
            "llm_model",
        ):
            value = canvas_context.get(key)
            if isinstance(value, str) and value.strip():
                summary[key] = value.strip()

        workspace_contract = canvas_context.get("workspace_contract")
        if isinstance(workspace_contract, dict):
            summary["workspace_contract"] = {
                "version": str(workspace_contract.get("version") or "manifest-v1"),
                "strategy": str(workspace_contract.get("strategy") or "manifest-first"),
                "natural_resource_search": bool(
                    workspace_contract.get("natural_resource_search", True)
                ),
            }

        workspace_hints = canvas_context.get("workspace_resource_hints")
        if isinstance(workspace_hints, list):
            hints: List[Dict[str, Any]] = []
            for item in workspace_hints[: self._MAX_PROMPT_HINTS]:
                if not isinstance(item, dict):
                    continue
                hint = {
                    "id": str(item.get("id", ""))[:256],
                    "canvas": str(item.get("canvas", ""))[:64],
                    "path": str(item.get("path", ""))[:384],
                    "label": str(item.get("label", ""))[:160],
                    "type": str(item.get("type", ""))[:48],
                }
                hint = {k: v for k, v in hint.items() if v}
                if hint:
                    hints.append(hint)
            if hints:
                summary["workspace_resource_hints"] = hints

        attached_contexts = canvas_context.get("attached_contexts")
        if isinstance(attached_contexts, list):
            attachments: List[Dict[str, Any]] = []
            for item in attached_contexts[: self._MAX_PROMPT_ATTACHMENTS]:
                if not isinstance(item, dict):
                    continue
                entry = {
                    "canvas": str(item.get("canvas", ""))[:64],
                    "strategy": str(item.get("strategy", ""))[:64],
                }
                memory_identifiers = item.get("memory_identifiers")
                if isinstance(memory_identifiers, list):
                    entry["memory_identifiers"] = [
                        str(mid)[:256] for mid in memory_identifiers[:32] if mid
                    ]
                resources = item.get("resources")
                if isinstance(resources, list):
                    resource_refs = []
                    for resource in resources[:16]:
                        if not isinstance(resource, dict):
                            continue
                        ref = {
                            "id": str(resource.get("id", ""))[:256],
                            "path": str(resource.get("path", ""))[:384],
                        }
                        ref = {k: v for k, v in ref.items() if v}
                        if ref:
                            resource_refs.append(ref)
                    if resource_refs:
                        entry["resources"] = resource_refs
                entry = {k: v for k, v in entry.items() if v not in ("", None, [], {})}
                if entry:
                    attachments.append(entry)
            if attachments:
                summary["attached_contexts"] = attachments

        # Keep minimal preloaded canvas metadata from session service
        preloaded = canvas_context.get("canvas_context")
        if isinstance(preloaded, dict):
            preloaded_summary = {
                "canvas": str(preloaded.get("canvas", ""))[:64],
                "display_name": str(preloaded.get("display_name", ""))[:128],
                "department_head": str(preloaded.get("department_head", ""))[:96],
            }
            preloaded_summary = {
                k: v for k, v in preloaded_summary.items() if v
            }
            if preloaded_summary:
                summary["canvas_context"] = preloaded_summary

        return summary

    @classmethod
    def _strip_minimax_tool_markup(cls, text: str) -> str:
        if not text:
            return ""
        stripped = cls._MINIMAX_TOOL_CALL_PATTERN.sub("", text)
        return stripped.strip()

    @classmethod
    def _extract_minimax_tool_calls_from_text(cls, text: str) -> List[SimpleNamespace]:
        """
        MiniMax's Anthropic-compatible API can sometimes emit tool calls as
        XML-like text instead of native tool_use blocks. Parse those into a
        synthetic tool_use representation so the normal agentic loop can
        execute them.
        """
        if not text or "<minimax:tool_call>" not in text:
            return []

        tool_calls: List[SimpleNamespace] = []
        for wrapper_match in cls._MINIMAX_TOOL_CALL_PATTERN.finditer(text):
            wrapper_body = wrapper_match.group(1) or ""
            for invoke_match in cls._MINIMAX_INVOKE_PATTERN.finditer(wrapper_body):
                tool_name = (invoke_match.group("name") or "").strip()
                if not tool_name:
                    continue
                body = invoke_match.group("body") or ""
                tool_input: Dict[str, Any] = {}
                for parameter_match in cls._MINIMAX_PARAMETER_PATTERN.finditer(body):
                    param_name = (parameter_match.group("name") or "").strip()
                    if not param_name:
                        continue
                    param_value = (parameter_match.group("value") or "").strip()
                    tool_input[param_name] = param_value
                tool_calls.append(
                    SimpleNamespace(
                        id=f"minimax-tool-{uuid.uuid4().hex}",
                        name=tool_name,
                        input=tool_input,
                        type="tool_use",
                    )
                )
        return tool_calls

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

        # Initialize mail service with Redis preferred and SQLite fallback.
        self.mail_service = create_mail_service(
            db_path=mail_db_path,
            use_redis=use_redis_mail,
            consumer_name=f"{self.department.value}-{uuid.uuid4().hex[:8]}",
        )
        self._init_spawner()

        # Hooks system — mirrors BaseAgent hook lifecycle
        self._hooks: List[Dict[str, Any]] = []

        # Wire department skills as tools (Issue #17)
        self._register_skill_tools()

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
            event: PreToolUse | PostToolUse | PostToolUseFailure | Stop
            handler: Callable or async callable
            tool_name_pattern: Optional regex pattern to match tool names
        """
        import re
        self._hooks.append({
            "event": self._normalize_hook_event(event),
            "handler": handler,
            "pattern": re.compile(tool_name_pattern) if tool_name_pattern else None,
        })

    async def _fire_hook(self, event: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Fire all hooks matching the event. Returns first non-None result."""
        import asyncio
        event = self._normalize_hook_event(event)
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

    @staticmethod
    def _normalize_hook_event(event: str) -> str:
        """Accept both SDK and legacy hook event names."""
        return LEGACY_HOOK_EVENT_ALIASES.get(event, event)

    @staticmethod
    def _hook_denied(result: Optional[Dict[str, Any]]) -> tuple[bool, str]:
        """Interpret SDK-style or legacy pre-tool hook denial responses."""
        if not isinstance(result, dict):
            return False, ""
        if result.get("deny"):
            return True, str(result.get("reason", "Tool denied by hook"))
        hook_output = result.get("hookSpecificOutput")
        if isinstance(hook_output, dict) and hook_output.get("permissionDecision") == "deny":
            reason = hook_output.get("permissionDecisionReason") or "Tool denied by hook"
            return True, str(reason)
        return False, ""

    @staticmethod
    def _hook_updated_input(result: Optional[Dict[str, Any]], original_input: Dict[str, Any]) -> Dict[str, Any]:
        """Apply SDK-style updatedInput only when the hook explicitly allows execution."""
        if not isinstance(result, dict):
            return original_input
        hook_output = result.get("hookSpecificOutput")
        if not isinstance(hook_output, dict):
            return original_input
        if hook_output.get("permissionDecision") != "allow":
            return original_input
        updated = hook_output.get("updatedInput")
        return updated if isinstance(updated, dict) else original_input

    def _hook_context(self, event: str, **extra: Any) -> Dict[str, Any]:
        """Populate a hook context shape closer to the Agent SDK callback contract."""
        context = {
            "hook_event_name": self._normalize_hook_event(event),
            "cwd": os.getcwd(),
            "session_id": getattr(self, "_current_session_id", None),
            "agent_id": f"{self.department.value}_head",
            "agent_type": "department_head",
            "department": self.department.value,
        }
        context.update(extra)
        return context

    def _register_skill_tools(self) -> None:
        """
        Register department skills as callable tools (Issue #17).

        Converts skill definitions from department_skills.py into tool
        entries in self._tools so agents can invoke them.
        """
        try:
            from src.agents.tools.skill_tools import get_department_skill_tools
            skill_tools = get_department_skill_tools(self.department.value)
            for tool_def in skill_tools:
                tool_name = tool_def.get("name", "")
                if tool_name and tool_name not in self._tools:
                    self._tools[tool_name] = tool_def
            if skill_tools:
                logger.debug(
                    f"Registered {len(skill_tools)} skill tools for "
                    f"{self.department.value}"
                )
        except Exception as e:
            logger.debug(f"Skill tool registration skipped: {e}")

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

    def _format_tools_for_anthropic(self) -> list:
        """
        Format the active tool surface for Anthropic-compatible providers.

        This must include the standard department tools as well as registry and
        skill-backed tools. Several heads had drifted into exposing only
        registry-backed tools, which made prompts mention actions like mail or
        spawning while the runtime silently omitted those tools.
        """
        tools: list[dict[str, Any]] = []
        seen: set[str] = set()

        for tool_def in self.STANDARD_TOOLS:
            name = tool_def.get("name")
            input_schema = tool_def.get("input_schema")
            if not name or not input_schema or name in seen:
                continue
            tools.append(
                {
                    "name": name,
                    "description": tool_def.get("description", f"{name} tool"),
                    "input_schema": input_schema,
                }
            )
            seen.add(name)

        for tool_name, tool_obj in (self._tools or {}).items():
            if tool_name in seen:
                continue
            input_schema = getattr(tool_obj, "input_schema", None) or getattr(
                tool_obj, "parameters", None
            )
            if not input_schema:
                continue
            tools.append(
                {
                    "name": tool_name,
                    "description": getattr(tool_obj, "description", f"{tool_name} tool"),
                    "input_schema": input_schema,
                }
            )
            seen.add(tool_name)

        return tools

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

        if self.department.value in {
            Department.RESEARCH.value,
            Department.DEVELOPMENT.value,
            Department.TRADING.value,
            Department.RISK.value,
            Department.PORTFOLIO.value,
        }:
            base_prompt = compose_department_head_prompt(
                self.department.value,
                base_prompt,
                sub_agents=self.sub_agents,
            )

        parts = [base_prompt]

        if memory_nodes:
            memory_section = "## Relevant Memory\n"
            memory_section += "\n".join(
                node.get("content", "") for node in memory_nodes if node.get("content")
            )
            parts.append(memory_section)

        if canvas_context:
            prompt_context = self._summarize_canvas_context_for_prompt(canvas_context)
            if prompt_context:
                parts.append(
                    "## Current Canvas Context (manifest-first summary)\n"
                    + json.dumps(prompt_context, indent=2)
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

    def _tool_result(
        self,
        *,
        status: str,
        message: str,
        payload: Optional[Dict[str, Any]] = None,
        ui_projection_event: Optional[Dict[str, Any]] = None,
    ) -> str:
        envelope: Dict[str, Any] = {
            "status": status,
            "message": message,
            "department": self.department.value,
        }
        if payload:
            envelope["payload"] = payload
        if ui_projection_event:
            envelope["ui_projection_event"] = ui_projection_event
        return json.dumps(envelope, ensure_ascii=False)

    async def _list_workspace_resources(self, tool_input: Dict[str, Any]) -> str:
        from src.api.services.workspace_resource_service import get_workspace_resource_service

        canvases = tool_input.get("canvases")
        if not isinstance(canvases, list):
            canvases = [self.department.value]
        tabs = tool_input.get("tabs") if isinstance(tool_input.get("tabs"), list) else None
        types = tool_input.get("types") if isinstance(tool_input.get("types"), list) else None
        limit = int(tool_input.get("limit", 100))
        svc = get_workspace_resource_service()
        resources = svc.list_resources(
            canvases=[str(canvas) for canvas in canvases if canvas],
            tabs=[str(tab) for tab in tabs] if tabs else None,
            types=[str(t) for t in types] if types else None,
            limit=max(1, min(limit, 500)),
        )
        return self._tool_result(
            status="ok",
            message=f"Listed {len(resources)} workspace resources",
            payload={"count": len(resources), "resources": resources},
        )

    async def _search_workspace_resources(self, tool_input: Dict[str, Any]) -> str:
        from src.api.services.workspace_resource_service import get_workspace_resource_service

        query = str(tool_input.get("query", "")).strip()
        if not query:
            return self._tool_result(status="error", message="query is required")
        canvases = tool_input.get("canvases")
        tabs = tool_input.get("tabs")
        types = tool_input.get("types")
        limit = int(tool_input.get("limit", 20))
        svc = get_workspace_resource_service()
        resources = svc.search_resources(
            query=query,
            canvases=[str(canvas) for canvas in canvases] if isinstance(canvases, list) else [self.department.value],
            tabs=[str(tab) for tab in tabs] if isinstance(tabs, list) else None,
            types=[str(t) for t in types] if isinstance(types, list) else None,
            limit=max(1, min(limit, 200)),
        )
        return self._tool_result(
            status="ok",
            message=f"Found {len(resources)} resources for query",
            payload={"query": query, "count": len(resources), "resources": resources},
        )

    async def _read_workspace_resource(self, tool_input: Dict[str, Any]) -> str:
        from src.api.services.workspace_resource_service import get_workspace_resource_service

        resource_id = str(tool_input.get("resource_id", "")).strip()
        if not resource_id:
            return self._tool_result(status="error", message="resource_id is required")
        max_chars = int(tool_input.get("max_chars", 120000))
        svc = get_workspace_resource_service()
        try:
            payload = svc.read_resource(resource_id, max_chars=max(1024, min(max_chars, 500000)))
        except Exception as exc:
            return self._tool_result(status="error", message=f"Failed to read resource: {exc}")
        return self._tool_result(
            status="ok",
            message=f"Read workspace resource {resource_id}",
            payload=payload,
        )

    async def _create_task(self, tool_input: Dict[str, Any]) -> str:
        from src.agents.departments.mail_consumer import get_task_manager

        title = str(tool_input.get("title", "")).strip()
        if not title:
            return self._tool_result(status="error", message="title is required")
        department = str(tool_input.get("department") or self.department.value).strip().lower()
        mgr = get_task_manager()
        card = mgr.create_kanban_card(
            department=department,
            title=title,
            description=str(tool_input.get("description", "")).strip(),
            priority=str(tool_input.get("priority", "normal")).strip().lower() or "normal",
            workflow_id=str(tool_input.get("workflow_id", "")).strip() or None,
            strategy_id=str(tool_input.get("strategy_id", "")).strip() or None,
        )
        payload = card.to_dict()
        projection = {
            "type": "kanban.card.created",
            "canvas": department,
            "department": department,
            "payload": payload,
        }
        return self._tool_result(
            status="ok",
            message=f"Created task card {payload.get('id')}",
            payload=payload,
            ui_projection_event=projection,
        )

    async def _update_task(self, tool_input: Dict[str, Any]) -> str:
        from src.agents.departments.mail_consumer import (
            KanbanStatus,
            TodoStatus,
            get_task_manager,
        )

        task_id = str(tool_input.get("task_id", "")).strip()
        status = str(tool_input.get("status", "")).strip().lower()
        if not task_id or not status:
            return self._tool_result(status="error", message="task_id and status are required")

        mgr = get_task_manager()
        result_note = str(tool_input.get("result", "")).strip() or None

        card_status_map = {
            "inbox": KanbanStatus.INBOX,
            "processing": KanbanStatus.PROCESSING,
            "review": KanbanStatus.REVIEW,
            "pending_approval": KanbanStatus.PENDING_APPROVAL,
            "completed": KanbanStatus.COMPLETED,
            "failed": KanbanStatus.FAILED,
        }
        todo_status_map = {
            "pending": TodoStatus.PENDING,
            "in_progress": TodoStatus.IN_PROGRESS,
            "blocked": TodoStatus.BLOCKED,
            "completed": TodoStatus.COMPLETED,
            "cancelled": TodoStatus.CANCELLED,
        }

        if status in card_status_map:
            updated = mgr.update_kanban_status(task_id, card_status_map[status], result=result_note)
            if updated is None:
                return self._tool_result(status="error", message=f"Kanban card not found: {task_id}")
            payload = updated.to_dict()
            projection = {
                "type": "kanban.card.updated",
                "canvas": payload.get("department", self.department.value),
                "department": payload.get("department", self.department.value),
                "payload": payload,
            }
            return self._tool_result(
                status="ok",
                message=f"Updated card {task_id} to {status}",
                payload=payload,
                ui_projection_event=projection,
            )

        if status in todo_status_map:
            updated_todo = mgr.update_todo_status(task_id, todo_status_map[status])
            if updated_todo is None:
                return self._tool_result(status="error", message=f"Todo not found: {task_id}")
            payload = updated_todo.to_dict()
            projection = {
                "type": "todo.updated",
                "canvas": payload.get("department", self.department.value),
                "department": payload.get("department", self.department.value),
                "payload": payload,
            }
            return self._tool_result(
                status="ok",
                message=f"Updated todo {task_id} to {status}",
                payload=payload,
                ui_projection_event=projection,
            )

        return self._tool_result(status="error", message=f"Unsupported task status: {status}")

    async def _spawn_subagent_tool(self, tool_input: Dict[str, Any]) -> str:
        worker_type = str(tool_input.get("worker_type", "")).strip()
        task = str(tool_input.get("task", "")).strip()
        if not worker_type or not task:
            return self._tool_result(status="error", message="worker_type and task are required")
        input_data = tool_input.get("input_data") if isinstance(tool_input.get("input_data"), dict) else None
        result = self.spawn_worker(worker_type=worker_type, task=task, input_data=input_data)
        projection = {
            "type": "subagent.spawned",
            "canvas": self.department.value,
            "department": self.department.value,
            "payload": result,
        }
        status = "ok" if result.get("status") == "spawned" else "error"
        message = f"Spawned subagent {result.get('agent_id')}" if status == "ok" else f"Failed to spawn subagent: {result.get('status')}"
        return self._tool_result(
            status=status,
            message=message,
            payload=result,
            ui_projection_event=projection if status == "ok" else None,
        )

    async def _update_workflow_state(self, tool_input: Dict[str, Any]) -> str:
        workflow_id = str(tool_input.get("workflow_id", "")).strip()
        state = str(tool_input.get("state", "")).strip()
        step = str(tool_input.get("step", "")).strip()
        note = str(tool_input.get("note", "")).strip()
        if not workflow_id or not state:
            return self._tool_result(status="error", message="workflow_id and state are required")

        payload = {
            "workflow_id": workflow_id,
            "state": state,
            "step": step or None,
            "note": note or None,
            "department": self.department.value,
        }
        projection = {
            "type": "workflow.state.updated",
            "canvas": "flowforge",
            "department": self.department.value,
            "payload": payload,
        }
        self._publish_thought(
            f"Workflow {workflow_id} -> {state}{f' ({step})' if step else ''}",
            thought_type="decision",
        )
        return self._tool_result(
            status="ok",
            message=f"Workflow {workflow_id} state updated to {state}",
            payload=payload,
            ui_projection_event=projection,
        )

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
            projection = {
                "type": "mail.sent",
                "canvas": self.department.value,
                "department": self.department.value,
                "payload": {
                    "message_id": message.id,
                    "to": to,
                    "subject": subject,
                },
            }
            return self._tool_result(
                status="ok",
                message=f"Mail sent to {to}",
                payload={"message_id": message.id, "to": to},
                ui_projection_event=projection,
            )
        except Exception as e:
            logger.error(f"send_mail failed: {e}")
            return self._tool_result(status="error", message=f"Mail send failed: {e}")

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
            projection = {
                "type": "memory.opinion.created",
                "canvas": self.department.value,
                "department": self.department.value,
                "payload": {
                    "content": str(tool_input.get("content", ""))[:500],
                    "confidence": float(tool_input.get("confidence", 0.7)),
                },
            }
            return self._tool_result(
                status="ok",
                message="Memory saved",
                payload={"saved": True},
                ui_projection_event=projection,
            )

        if tool_name == "list_resources":
            return await self._list_workspace_resources(tool_input)

        if tool_name == "search_resources":
            return await self._search_workspace_resources(tool_input)

        if tool_name == "read_resource":
            return await self._read_workspace_resource(tool_input)

        if tool_name == "create_task":
            return await self._create_task(tool_input)

        if tool_name == "update_task":
            return await self._update_task(tool_input)

        if tool_name == "spawn_subagent":
            return await self._spawn_subagent_tool(tool_input)

        if tool_name == "update_workflow_state":
            return await self._update_workflow_state(tool_input)

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

        # Resolve provider via ProviderRouter; allow context override
        preferred_provider = None
        preferred_model = None
        if canvas_context and isinstance(canvas_context, dict):
            preferred_provider = canvas_context.get("provider") or canvas_context.get("llm_provider")
            preferred_model = canvas_context.get("model") or canvas_context.get("llm_model")

        router = get_router()
        runtime_config = router.resolve_runtime_config(
            preferred_provider=preferred_provider,
            preferred_model=preferred_model,
            tier=self.model_tier,
        )
        if not runtime_config or not runtime_config.api_key:
            raise RuntimeError(
                "No LLM runtime configured. Configure a provider in Settings or set QMX_LLM_* environment variables."
            )
        model = runtime_config.model

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
            timeout = httpx.Timeout(120.0, connect=30.0)
            async with httpx.AsyncClient(timeout=timeout, trust_env=False) as http_client:
                client = anthropic.AsyncAnthropic(
                    api_key=runtime_config.api_key,
                    base_url=runtime_config.base_url,
                    http_client=http_client,
                )
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

                    tool_use_blocks = [
                        b for b in response.content if getattr(b, "type", None) == "tool_use"
                    ]
                    text_blocks = [
                        b.text for b in response.content if getattr(b, "type", None) == "text"
                    ]
                    synthetic_tool_use_blocks: list = []
                    if not tool_use_blocks and text_blocks:
                        minimax_tool_calls = self._extract_minimax_tool_calls_from_text(
                            "\n".join(text_blocks)
                        )
                        if minimax_tool_calls:
                            synthetic_tool_use_blocks = minimax_tool_calls

                    if response.stop_reason != "tool_use" and not synthetic_tool_use_blocks:
                        # Collect any final text blocks and exit the loop
                        for block in response.content:
                            if block.type == "text":
                                final_text += block.text
                        # Fire STOP hook (Agent SDK lifecycle)
                        await self._fire_hook(
                            HookEvent.STOP,
                            self._hook_context(
                                HookEvent.STOP,
                                iterations=iteration + 1,
                                stop_reason=response.stop_reason,
                            ),
                        )
                        self._publish_thought(
                            f"Turn {iteration + 1}: Final response ({len(final_text)} chars)",
                            thought_type="decision",
                        )
                        break

                    # --- tool_use turn ---
                    tool_use_blocks = tool_use_blocks or synthetic_tool_use_blocks

                    # Also capture any text emitted alongside tool calls
                    for block in response.content:
                        if block.type == "text":
                            if synthetic_tool_use_blocks:
                                visible_text = self._strip_minimax_tool_markup(block.text)
                                if visible_text:
                                    final_text += visible_text
                            else:
                                final_text += block.text

                    # Append the assistant message (with tool_use blocks) to history
                    if synthetic_tool_use_blocks:
                        messages.append(
                            {
                                "role": "assistant",
                                "content": [
                                    {
                                        "type": "tool_use",
                                        "id": block.id,
                                        "name": block.name,
                                        "input": block.input,
                                    }
                                    for block in synthetic_tool_use_blocks
                                ],
                            }
                        )
                    else:
                        messages.append({"role": "assistant", "content": response.content})

                    # Execute each tool and build the tool_result user message
                    tool_result_content = []
                    for block in tool_use_blocks:
                        all_tool_calls.append(
                            {"name": block.name, "input": block.input, "id": block.id}
                        )

                        # Fire PRE_TOOL_USE hook (Agent SDK lifecycle)
                        pre_result = await self._fire_hook(
                            HookEvent.PRE_TOOL_USE,
                            self._hook_context(
                                HookEvent.PRE_TOOL_USE,
                                tool_name=block.name,
                                tool_input=block.input,
                                tool_use_id=block.id,
                            ),
                        )
                        # If hook returns a decision to deny, skip tool execution
                        denied, deny_reason = self._hook_denied(pre_result)
                        if denied:
                            tool_result_content.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": deny_reason,
                                "is_error": True,
                            })
                            continue
                        tool_input = self._hook_updated_input(pre_result, block.input)

                        self._publish_thought(
                            f"Turn {iteration + 1}: Calling {block.name}",
                            thought_type="tool_call",
                        )

                        logger.info(
                            f"Executing tool '{block.name}' (id={block.id}) "
                            f"for {self.department.value}"
                        )

                        try:
                            result_str = await self._execute_tool(block.name, tool_input)
                            # Fire POST_TOOL_USE hook
                            await self._fire_hook(
                                HookEvent.POST_TOOL_USE,
                                self._hook_context(
                                    HookEvent.POST_TOOL_USE,
                                    tool_name=block.name,
                                    tool_input=tool_input,
                                    tool_use_id=block.id,
                                    result=result_str,
                                ),
                            )
                        except Exception as tool_err:
                            # Fire POST_TOOL_USE_FAILURE hook
                            await self._fire_hook(
                                HookEvent.POST_TOOL_USE_FAILURE,
                                self._hook_context(
                                    HookEvent.POST_TOOL_USE_FAILURE,
                                    tool_name=block.name,
                                    tool_input=tool_input,
                                    tool_use_id=block.id,
                                    error=str(tool_err),
                                ),
                            )
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

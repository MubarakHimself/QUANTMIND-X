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

        logger.info(f"DepartmentHead initialized: {self.department.value} with {len(self._tools)} tools")

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
        parts = [self.system_prompt]

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

    async def _invoke_claude(
        self,
        task: str,
        canvas_context: Optional[dict] = None,
        tools: Optional[list] = None,
    ) -> dict:
        """
        Make an Anthropic SDK call with the assembled system prompt.

        Args:
            task: The user-facing task string sent as the human turn.
            canvas_context: Optional canvas context to embed in the prompt.
            tools: Optional list of Anthropic tool definitions to pass.

        Returns:
            Dict with keys: content, tool_calls, model, usage, stop_reason.
            On error includes an "error" key.
        """
        memory_nodes = await self._read_relevant_memory(task)
        system = self._build_system_prompt(
            canvas_context=canvas_context,
            memory_nodes=memory_nodes,
        )

        model = os.getenv("ANTHROPIC_MODEL_SONNET", "claude-sonnet-4-6")
        if self.model_tier == "opus":
            model = os.getenv("ANTHROPIC_MODEL_OPUS", "claude-opus-4-6")

        client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        kwargs: dict = {
            "model": model,
            "max_tokens": 4096,
            "system": system,
            "messages": [{"role": "user", "content": task}],
        }
        if tools:
            kwargs["tools"] = tools

        try:
            response = await client.messages.create(**kwargs)

            text_content = ""
            tool_calls = []
            for block in response.content:
                if block.type == "text":
                    text_content += block.text
                elif block.type == "tool_use":
                    tool_calls.append(
                        {"name": block.name, "input": block.input, "id": block.id}
                    )

            return {
                "content": text_content,
                "tool_calls": tool_calls,
                "model": response.model,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
                "stop_reason": response.stop_reason,
            }
        except anthropic.AuthenticationError:
            logger.error(
                "Anthropic authentication failed — check ANTHROPIC_API_KEY"
            )
            return {"content": "", "tool_calls": [], "error": "auth_failed"}
        except Exception as e:
            logger.error(f"Claude invocation failed: {e}")
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

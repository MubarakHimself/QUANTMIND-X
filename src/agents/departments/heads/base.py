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
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any

from src.agents.departments.types import (
    Department,
    DepartmentHeadConfig,
    get_model_tier,
)
from src.agents.departments.department_mail import (
    DepartmentMailService,
    MessageType,
    Priority,
)
from src.agents.departments.memory_manager import DepartmentMemoryManager
from src.agents.departments.tool_access import ToolAccessController

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
    ):
        """
        Initialize the Department Head.

        Args:
            config: Department configuration
            mail_db_path: Path to mail database
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

        self.mail_service = DepartmentMailService(db_path=mail_db_path)
        self._init_spawner()

        logger.info(f"DepartmentHead initialized: {self.department.value}")

    def get_available_tools(self) -> List[str]:
        """
        Get list of available tools for this department based on permissions.

        Returns:
            List of tool names this department can access
        """
        return self.tool_access.get_available_tools()

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

            config = SubAgentConfig(
                agent_type=worker_type,
                name=f"{self.department.value}_{worker_type}",
                parent_agent_id=f"dept_head_{self.department.value}",
                input_data=input_data or {"task": task},
                pool_key=self.department.value,
            )

            agent = self.spawner.spawn(config)
            logger.info(f"Spawned worker {agent.id} for {self.department.value}")

            return {
                "status": "spawned",
                "agent_id": agent.id,
                "worker_type": worker_type,
            }

        except Exception as e:
            logger.error(f"Failed to spawn worker: {e}")
            return {
                "status": "spawn_failed",
                "error": str(e),
            }

    def close(self):
        """Clean up resources."""
        if self.mail_service:
            self.mail_service.close()
        logger.info(f"DepartmentHead closed: {self.department.value}")

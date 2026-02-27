"""
Floor Manager

The Floor Manager is the top-level orchestrator for the Trading Floor.
It routes tasks to appropriate Department Heads and manages cross-department communication.

Model Tier: Opus (highest reasoning capability)
"""
import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

from src.agents.departments.types import (
    Department,
    DepartmentHeadConfig,
    get_department_configs,
    get_model_tier,
)
from src.agents.departments.department_mail import (
    DepartmentMailService,
    MessageType,
    Priority,
)

logger = logging.getLogger(__name__)


class FloorManager:
    """
    Floor Manager for the Trading Floor Model.

    The Floor Manager:
    - Routes incoming tasks to appropriate Department Heads
    - Manages cross-department communication via mail service
    - Coordinates with Agent Spawner for worker spawning
    - Uses Opus tier for highest reasoning capability

    Attributes:
        mail_service: SQLite mail service for cross-department messaging
        spawner: Agent spawner for dynamic worker creation
        departments: Dictionary of department configurations
        model_tier: Model tier (always "opus" for Floor Manager)
    """

    # Keyword-based task classification
    DEPARTMENT_KEYWORDS = {
        Department.ANALYSIS: [
            "analyze", "analysis", "market", "sentiment", "news", "scan",
            "technical", "indicator", "signal", "pattern", "chart",
            "trend", "support", "resistance", "forecast",
        ],
        Department.RESEARCH: [
            "research", "strategy", "backtest", "develop", "create",
            "alpha", "factor", "optimize", "validate", "test",
            "pinescript", "code", "implement",
        ],
        Department.RISK: [
            "risk", "position size", "drawdown", "var", "exposure",
            "limit", "stop loss", "take profit", "margin", "leverage",
        ],
        Department.EXECUTION: [
            "execute", "order", "buy", "sell", "trade", "fill",
            "route", "slippage", "broker", "venue",
        ],
        Department.PORTFOLIO: [
            "portfolio", "allocation", "rebalance", "performance",
            "diversify", "asset", "balance", "attribut",
        ],
    }

    def __init__(
        self,
        mail_db_path: str = ".quantmind/department_mail.db",
        max_workers_per_dept: int = 5,
    ):
        """
        Initialize the Floor Manager.

        Args:
            mail_db_path: Path to SQLite mail database
            max_workers_per_dept: Maximum workers per department
        """
        self.mail_service = DepartmentMailService(db_path=mail_db_path)
        self._init_spawner()
        self.departments = self._init_departments()
        self.model_tier = "opus"
        self._max_workers = max_workers_per_dept

        logger.info(f"FloorManager initialized with {len(self.departments)} departments")

    def _init_spawner(self):
        """Initialize the agent spawner."""
        try:
            from src.agents.subagent.spawner import get_spawner
            self.spawner = get_spawner()
        except ImportError:
            logger.warning("Agent spawner not available, using mock")
            self.spawner = None

    def _init_departments(self) -> Dict[Department, DepartmentHeadConfig]:
        """Initialize department configurations."""
        configs = get_department_configs()
        return {
            Department(dept_name): config
            for dept_name, config in configs.items()
        }

    def classify_task(self, task: str) -> Department:
        """
        Classify a task to determine which department should handle it.

        Uses keyword matching for simple classification.
        Can be upgraded to LLM-based classification for complex tasks.

        Args:
            task: The task description

        Returns:
            The department that should handle this task
        """
        task_lower = task.lower()

        # Score each department based on keyword matches
        scores: Dict[Department, int] = {}
        for dept, keywords in self.DEPARTMENT_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in task_lower)
            if score > 0:
                scores[dept] = score

        # Return highest scoring department, default to ANALYSIS
        if scores:
            return max(scores, key=scores.get)
        return Department.ANALYSIS

    def dispatch(
        self,
        to_dept: Department,
        task: str,
        priority: str = "normal",
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Dispatch a task to a department via mail.

        Args:
            to_dept: Target department
            task: Task description
            priority: Message priority (low, normal, high, urgent)
            context: Optional context dictionary

        Returns:
            Dispatch result with status and message ID
        """
        # Map priority string to enum
        priority_map = {
            "low": Priority.LOW,
            "normal": Priority.NORMAL,
            "high": Priority.HIGH,
            "urgent": Priority.URGENT,
        }
        msg_priority = priority_map.get(priority.lower(), Priority.NORMAL)

        # Build message body
        body = task
        if context:
            body = json.dumps({
                "task": task,
                "context": context,
            })

        # Send mail message
        message = self.mail_service.send(
            from_dept="floor_manager",
            to_dept=to_dept.value,
            type=MessageType.DISPATCH,
            subject=f"Task: {task[:50]}...",
            body=body,
            priority=msg_priority,
        )

        logger.info(f"Dispatched task to {to_dept.value}: {message.id}")

        return {
            "status": "dispatched",
            "message_id": message.id,
            "to_dept": to_dept.value,
            "priority": priority,
        }

    def process(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Process an incoming task.

        1. Classify the task to determine department
        2. Dispatch to the appropriate department

        Args:
            task: Task description
            context: Optional context dictionary

        Returns:
            Processing result with classification and dispatch info
        """
        # Classify task
        dept = self.classify_task(task)
        logger.info(f"Classified task to {dept.value}: {task[:50]}...")

        # Dispatch to department
        dispatch_result = self.dispatch(
            to_dept=dept,
            task=task,
            priority="normal",
            context=context,
        )

        return {
            "status": "processed",
            "classified_dept": dept.value,
            "dispatch": dispatch_result,
        }

    def handle_dispatch(
        self,
        from_department: str,
        task: str,
        suggested_department: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Handle a dispatch request from Copilot or another department.

        Processes incoming delegation requests and routes them to the appropriate
        department. If a suggested department is provided and valid, uses it.
        Otherwise, classifies the task automatically.

        Args:
            from_department: Department sending the dispatch (e.g., "copilot")
            task: Task description to dispatch
            suggested_department: Optional target department suggestion
            context: Optional context dictionary

        Returns:
            Dispatch result with status, message ID, and routing info
        """
        # Determine target department
        target_dept = None

        if suggested_department:
            # Validate suggested department
            try:
                target_dept = Department(suggested_department.lower())
                logger.info(f"Using suggested department: {suggested_department}")
            except ValueError:
                logger.warning(
                    f"Invalid suggested department: {suggested_department}, "
                    "falling back to classification"
                )
                target_dept = self.classify_task(task)
        else:
            # Auto-classify if no suggestion
            target_dept = self.classify_task(task)

        # Delegate to the determined department
        message = self.delegate_to_department(
            from_dept=from_department,
            to_dept=target_dept.value,
            task=task,
            priority="normal",
            context=context,
        )

        return {
            "status": "dispatched",
            "message_id": message.id,
            "from_department": from_department,
            "to_department": target_dept.value,
            "priority": "normal",
        }

    def delegate_to_department(
        self,
        from_dept: str,
        to_dept: str,
        task: str,
        priority: str = "normal",
        context: Optional[Dict[str, Any]] = None,
    ) -> "DepartmentMessage":
        """
        Delegate a task to a department via mail.

        Sends a dispatch message through the mail service to the target department.

        Args:
            from_dept: Sending department identifier
            to_dept: Target department identifier
            task: Task description
            priority: Message priority (low, normal, high, urgent)
            context: Optional context dictionary

        Returns:
            The created DepartmentMessage
        """
        # Map priority string to enum
        priority_map = {
            "low": Priority.LOW,
            "normal": Priority.NORMAL,
            "high": Priority.HIGH,
            "urgent": Priority.URGENT,
        }
        msg_priority = priority_map.get(priority.lower(), Priority.NORMAL)

        # Build message body with optional context
        body = task
        if context:
            body = json.dumps({
                "task": task,
                "context": context,
            })

        # Create subject line
        subject = f"Task from {from_dept}: {task[:50]}..."
        if len(task) > 50:
            subject += "..."

        # Send mail message
        message = self.mail_service.send(
            from_dept=from_dept,
            to_dept=to_dept,
            type=MessageType.DISPATCH,
            subject=subject,
            body=body,
            priority=msg_priority,
        )

        logger.info(
            f"Delegated task from {from_dept} to {to_dept}: "
            f"message_id={message.id}"
        )

        return message

    def close(self):
        """Clean up resources."""
        if self.mail_service:
            self.mail_service.close()
        logger.info("FloorManager closed")

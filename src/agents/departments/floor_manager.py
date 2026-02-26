"""
Floor Manager

The Floor Manager is the top-level orchestrator for the Trading Floor.
It routes tasks to appropriate Department Heads and manages cross-department communication.

Model Tier: Opus (highest reasoning capability)
"""
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

from src.agents.departments.types import (
    Department,
    DepartmentHeadConfig,
    get_department_configs,
    get_model_tier,
)
from src.agents.departments.department_mail import DepartmentMailService

logger = logging.getLogger(__name__)


class FloorManager:
    """
    Floor Manager for the Trading Floor Model.

    The Floor Manager is the top-level orchestrator that:
    - Routes incoming tasks to appropriate departments
    - Manages cross-department communication via mail
    - Coordinates multi-department workflows
    - Spawns department heads when needed
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
            mail_db_path: Path to the mail database
            max_workers_per_dept: Maximum workers per department
        """
        self.mail_service = DepartmentMailService(db_path=mail_db_path)
        self._init_spawner()
        self.departments = self._init_departments()
        self.model_tier = "opus"
        self._max_workers = max_workers_per_dept
        logger.info(f"FloorManager initialized with {len(self.departments)} departments")

    def _init_spawner(self):
        """
        Initialize the agent spawner.

        Tries to import the spawner from subagent module.
        Falls back to None if not available.
        """
        try:
            from src.agents.subagent.spawner import get_spawner
            self.spawner = get_spawner()
        except ImportError:
            logger.warning("Agent spawner not available, using mock")
            self.spawner = None

    def _init_departments(self) -> Dict[Department, DepartmentHeadConfig]:
        """
        Initialize department configurations.

        Returns:
            Dictionary mapping departments to their configs
        """
        configs = get_department_configs()
        return {
            Department(dept_name): config
            for dept_name, config in configs.items()
        }

    def classify_task(self, task: str) -> Department:
        """
        Classify a task to determine which department should handle it.

        Uses keyword matching for simple classification.
        Each keyword match adds a score, and the department with the
        highest score is selected.

        Args:
            task: The task description

        Returns:
            The department that should handle this task
        """
        task_lower = task.lower()
        scores: Dict[Department, int] = {}

        # Score each department by keyword matches
        for dept, keywords in self.DEPARTMENT_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in task_lower)
            if score > 0:
                scores[dept] = score

        # Return highest scoring department, or default to ANALYSIS
        if scores:
            return max(scores, key=scores.get)
        return Department.ANALYSIS

    def close(self):
        """
        Close the Floor Manager and cleanup resources.

        Closes the mail service database connection.
        """
        if self.mail_service:
            self.mail_service.close()
        logger.info("FloorManager closed")

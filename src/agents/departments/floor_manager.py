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

    def close(self):
        """Clean up resources."""
        if self.mail_service:
            self.mail_service.close()
        logger.info("FloorManager closed")

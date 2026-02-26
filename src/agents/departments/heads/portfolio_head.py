"""
Portfolio Management Department Head

Responsible for:
- Portfolio allocation and optimization
- Rebalancing decisions
- Performance tracking and attribution
"""
import logging
from typing import Dict, List, Any

from src.agents.departments.heads.base import DepartmentHead
from src.agents.departments.types import Department, get_department_config

logger = logging.getLogger(__name__)


class PortfolioHead(DepartmentHead):
    """Portfolio Management Department Head."""

    def __init__(self, mail_db_path: str = ".quantmind/department_mail.db"):
        config = get_department_config(Department.PORTFOLIO)
        super().__init__(config=config, mail_db_path=mail_db_path)

    def get_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "optimize_allocation",
                "description": "Optimize portfolio allocation",
                "parameters": {
                    "assets": "List of assets",
                    "target_return": "Target return",
                    "max_risk": "Maximum risk tolerance",
                },
            },
            {
                "name": "rebalance_portfolio",
                "description": "Rebalance portfolio to target allocation",
                "parameters": {
                    "target_allocation": "Target allocation weights",
                    "threshold": "Rebalance threshold (%)",
                },
            },
            {
                "name": "track_performance",
                "description": "Track portfolio performance",
                "parameters": {
                    "period": "Performance period",
                    "benchmark": "Benchmark for comparison",
                },
            },
        ]

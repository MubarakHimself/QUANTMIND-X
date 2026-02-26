"""
Research Department Head

Responsible for:
- Strategy research and development
- Backtesting and validation
- Alpha research and data science
"""
import logging
from typing import Dict, List, Any

from src.agents.departments.heads.base import DepartmentHead
from src.agents.departments.types import Department, get_department_config

logger = logging.getLogger(__name__)


class ResearchHead(DepartmentHead):
    """Research Department Head for strategy development."""

    def __init__(self, mail_db_path: str = ".quantmind/department_mail.db"):
        config = get_department_config(Department.RESEARCH)
        super().__init__(config=config, mail_db_path=mail_db_path)

    def get_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "develop_strategy",
                "description": "Develop a new trading strategy",
                "parameters": {
                    "type": "Strategy type (trend, mean_reversion, breakout)",
                    "symbols": "Target symbols",
                },
            },
            {
                "name": "backtest_strategy",
                "description": "Backtest a strategy on historical data",
                "parameters": {
                    "strategy_id": "Strategy identifier",
                    "start_date": "Backtest start date",
                    "end_date": "Backtest end date",
                },
            },
            {
                "name": "research_alpha",
                "description": "Research alpha factors",
                "parameters": {
                    "universe": "Asset universe",
                    "factors": "Factors to research",
                },
            },
        ]

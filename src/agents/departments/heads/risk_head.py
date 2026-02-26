"""
Risk Department Head

Responsible for:
- Position sizing and exposure management
- Drawdown monitoring and limits
- Value at Risk (VaR) calculations
"""
import logging
from typing import Dict, List, Any

from src.agents.departments.heads.base import DepartmentHead
from src.agents.departments.types import Department, get_department_config

logger = logging.getLogger(__name__)


class RiskHead(DepartmentHead):
    """Risk Department Head for risk management."""

    def __init__(self, mail_db_path: str = ".quantmind/department_mail.db"):
        config = get_department_config(Department.RISK)
        super().__init__(config=config, mail_db_path=mail_db_path)

    def get_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "calculate_position_size",
                "description": "Calculate optimal position size",
                "parameters": {
                    "symbol": "Trading symbol",
                    "account_balance": "Account balance",
                    "risk_percent": "Risk per trade (%)",
                },
            },
            {
                "name": "check_drawdown",
                "description": "Check drawdown limits",
                "parameters": {
                    "account_id": "Account identifier",
                },
            },
            {
                "name": "calculate_var",
                "description": "Calculate Value at Risk",
                "parameters": {
                    "portfolio": "Portfolio holdings",
                    "confidence": "Confidence level (e.g., 0.95)",
                    "timeframe": "Timeframe in days",
                },
            },
        ]

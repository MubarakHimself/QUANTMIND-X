"""
Development Department Head

Responsible for:
- Building and maintaining trading bots (Python, PineScript, MQL5)
- EA lifecycle management (create, test, deploy)
- Strategy implementation and optimization

Workers:
- python_dev: Python-based strategy implementation
- pinescript_dev: TradingView PineScript development
- mql5_dev: MetaTrader 5 EA development
"""
import logging
from typing import Dict, List, Any

from src.agents.departments.heads.base import DepartmentHead
from src.agents.departments.types import Department, get_department_config
from src.agents.departments.tool_registry import ToolRegistry
from src.agents.departments.tool_access import ToolPermission

logger = logging.getLogger(__name__)


class DevelopmentHead(DepartmentHead):
    """Development Department Head for EA/Bot building."""

    def __init__(self, mail_db_path: str = ".quantmind/department_mail.db"):
        config = get_department_config(Department.DEVELOPMENT)
        super().__init__(config=config, mail_db_path=mail_db_path)

    def get_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "create_ea",
                "description": "Create new Expert Advisor (EA)",
                "parameters": {
                    "name": "EA name",
                    "strategy_type": "Strategy type (trend, range, breakout)",
                    "language": "mql5, pinescript, or python",
                },
            },
            {
                "name": "write_code",
                "description": "Write trading bot code",
                "parameters": {
                    "file_type": "File type (.mq5, .py, .pine)",
                    "content": "Trading logic code",
                },
            },
            {
                "name": "test_ea",
                "description": "Test EA on historical data",
                "parameters": {
                    "ea_name": "EA to test",
                    "symbol": "Trading symbol",
                    "timeframe": "Timeframe (M1, M5, H1, etc.)",
                },
            },
            {
                "name": "deploy_ea",
                "description": "Deploy EA to paper trading",
                "parameters": {
                    "ea_name": "EA to deploy",
                    "symbol": "Trading symbol",
                    "parameters": "EA parameters",
                },
            },
        ]

    def get_tool_instances(self) -> Dict[str, Any]:
        """Get actual tool instances for Development department."""
        tools = {}
        dept = Department.DEVELOPMENT

        # Development gets full access to coding tools
        tool_names = [
            "mql5_tools",
            "pinescript_tools",
            "backtest_tools",
            "ea_lifecycle",
            "memory_tools",
            "knowledge_tools",
        ]

        for tool_name in tool_names:
            tool = ToolRegistry.get_tool(tool_name, dept)
            if tool:
                tools[tool_name] = tool

        return tools

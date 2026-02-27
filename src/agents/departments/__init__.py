"""
Department-Based Agent Framework

Trading Floor Model with 5 departments:
- Analysis: Market analysis, sentiment, news monitoring
- Research: Strategy development, backtesting, alpha research
- Risk: Position sizing, drawdown monitoring, VaR
- Execution: Order routing, fill tracking, slippage
- Portfolio: Allocation, rebalancing, performance
"""
from src.agents.departments.cold_storage import ColdStorageManager
from src.agents.departments.department_mail import DepartmentMailService
from src.agents.departments.memory_access import FloorManagerMemoryAccess
from src.agents.departments.memory_manager import DepartmentMemoryManager

__all__ = [
    "DepartmentMailService",
    "DepartmentMemoryManager",
    "ColdStorageManager",
    "FloorManagerMemoryAccess",
]

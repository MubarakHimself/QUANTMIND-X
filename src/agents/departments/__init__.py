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
from src.agents.departments.department_mail import (
    DepartmentMailService,
    RedisDepartmentMailService,
    get_mail_service,
    get_redis_mail_service,
)
from src.agents.departments.memory_access import FloorManagerMemoryAccess
from src.agents.departments.memory_manager import DepartmentMemoryManager
from src.agents.departments.task_router import (
    TaskRouter,
    TaskPriority,
    TaskStatus,
    Task,
    TaskResult,
    get_task_router,
    reset_task_router,
)

__all__ = [
    "DepartmentMailService",
    "RedisDepartmentMailService",
    "get_mail_service",
    "get_redis_mail_service",
    "DepartmentMemoryManager",
    "ColdStorageManager",
    "FloorManagerMemoryAccess",
    # Task Router (Story 7.7)
    "TaskRouter",
    "TaskPriority",
    "TaskStatus",
    "Task",
    "TaskResult",
    "get_task_router",
    "reset_task_router",
]

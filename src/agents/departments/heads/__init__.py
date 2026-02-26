"""
Department Heads

Each department has a head that:
- Receives tasks from Floor Manager via mail
- Can spawn workers for complex tasks
- Sends results to other departments
"""
from src.agents.departments.heads.base import DepartmentHead

__all__ = ["DepartmentHead"]

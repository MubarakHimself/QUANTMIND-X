"""
Floor Manager read-only access to all department memories.
"""

from typing import Dict, List

from .memory_manager import DepartmentMemoryManager, MemoryResult
from .types import Department


class FloorManagerMemoryAccess:
    """Floor Manager has read-only access to all departments."""

    def __init__(self, base_path: str = ".quantmind/departments"):
        self._dept_managers: Dict[Department, DepartmentMemoryManager] = {
            dept: DepartmentMemoryManager(dept, base_path=base_path)
            for dept in Department
        }

    def read_department_memory(self, department: Department) -> str:
        """Read-only access to any department's MEMORY.md."""
        return self._dept_managers[department].read_memory()

    def search_department(
        self,
        department: Department,
        query: str,
        limit: int = 10
    ) -> List[MemoryResult]:
        """Search within a specific department."""
        return self._dept_managers[department].search(query, limit)

    def search_all_departments(self, query: str, limit_per_dept: int = 5) -> Dict[str, List[MemoryResult]]:
        """Search across all departments (monitoring)."""
        results = {}
        for dept in Department:
            results[dept.value] = self.search_department(dept, query, limit_per_dept)
        return results

    def get_all_recent_activity(self, limit_per_dept: int = 3) -> Dict[str, List[str]]:
        """Get recent daily log entries from all departments."""
        activity = {}
        for dept in Department:
            manager = self._dept_managers[dept]
            log_content = manager.read_daily_logs()
            if log_content:
                # Extract recent entries (lines starting with ###)
                lines = log_content.split('\n')
                entries = [l.strip() for l in lines if l.strip() and not l.startswith('#')]
                activity[dept.value] = entries[:limit_per_dept]
        return activity

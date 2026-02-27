# src/agents/departments/memory_access.py
"""
Floor Manager Memory Access - Read-only cross-department memory access.

Provides the Floor Manager with read-only access to all department memories
for oversight, coordination, and decision-making purposes.
"""
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Any

from src.agents.departments.memory_manager import DepartmentMemoryManager
from src.agents.departments.types import Department


class FloorManagerMemoryAccess:
    """
    Read-only access to all department memories.

    Allows the Floor Manager to:
    - Read memories from any department
    - Search across all departments
    - Get consolidated views
    - Monitor department activity

    This is intentionally read-only - the Floor Manager coordinates but
    should not directly modify department memories.

    Args:
        base_path: Base directory for all departments
    """

    def __init__(self, base_path: Optional[Path] = None):
        """
        Initialize the memory access layer.

        Args:
            base_path: Base directory for departments (default: Path("data/departments"))
        """
        self.base_path = base_path or Path("data/departments")

    def _get_memory_manager(self, department: Department) -> DepartmentMemoryManager:
        """
        Get a memory manager for a specific department.

        Note: Uses auto_initialize=False to avoid creating files when just reading.

        Args:
            department: The department

        Returns:
            DepartmentMemoryManager instance
        """
        return DepartmentMemoryManager(
            department=department,
            base_path=self.base_path,
            auto_initialize=False  # Don't create files when just reading
        )

    def read_department_memory(self, department: Department) -> Optional[str]:
        """
        Read the full MEMORY.md content from a department.

        Args:
            department: The department to read from

        Returns:
            The memory content or None if file doesn't exist
        """
        try:
            manager = self._get_memory_manager(department)
            return manager.read_memory()
        except (FileNotFoundError, IOError):
            return None

    def read_all_departments(self) -> Dict[str, Optional[str]]:
        """
        Read memories from all departments.

        Returns:
            Dictionary mapping department names to their memory contents
        """
        all_memories = {}

        for dept in Department:
            content = self.read_department_memory(dept)
            if content is not None:
                all_memories[dept.value] = content

        return all_memories

    def read_daily_log(
        self,
        department: Department,
        log_date: date
    ) -> Optional[str]:
        """
        Read a specific daily log from a department.

        Args:
            department: The department
            log_date: The date of the log

        Returns:
            The log content or None if not found
        """
        try:
            manager = self._get_memory_manager(department)
            return manager.get_daily_log(log_date)
        except (FileNotFoundError, IOError):
            return None

    def read_recent_logs(
        self,
        department: Department,
        days: int = 7
    ) -> Dict[str, str]:
        """
        Read recent daily logs from a department.

        Args:
            department: The department
            days: Number of recent days

        Returns:
            Dictionary mapping dates to log contents
        """
        try:
            manager = self._get_memory_manager(department)
            return manager.get_recent_logs(days=days)
        except (FileNotFoundError, IOError):
            return {}

    def search_all(
        self,
        query: str,
        department: Optional[str] = None,
        category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for content across all or specific departments.

        Args:
            query: The search term
            department: Optional department filter
            category: Optional category filter

        Returns:
            List of matching entries with department context
        """
        results = []

        departments_to_search = (
            [Department(department)] if department
            else list(Department)
        )

        for dept in departments_to_search:
            try:
                manager = self._get_memory_manager(dept)

                # Use the manager's search method
                matches = manager.search(query, category=category)

                # Add department context to each match
                for match in matches:
                    results.append({
                        "department": dept.value,
                        "content": match,
                        "match_type": "memory"
                    })

            except (FileNotFoundError, IOError):
                continue

        return results

    def get_department_summary(self, department: Department) -> Optional[Dict[str, Any]]:
        """
        Get a summary of a department's memory state.

        Args:
            department: The department

        Returns:
            Dictionary with summary information or None
        """
        try:
            manager = self._get_memory_manager(department)
            stats = manager.get_stats()

            return {
                "department": department.value,
                "stats": stats,
                "memory_file": str(manager.get_memory_file_path()),
                "memory_dir": str(manager.get_memory_dir_path())
            }

        except (FileNotFoundError, IOError):
            return None

    def get_all_summaries(self) -> Dict[str, Dict[str, Any]]:
        """
        Get summaries for all departments.

        Returns:
            Dictionary mapping department names to summaries
        """
        summaries = {}

        for dept in Department:
            summary = self.get_department_summary(dept)
            if summary:
                summaries[dept.value] = summary

        return summaries

    def get_consolidated_view(
        self,
        category_filter: Optional[str] = None,
        max_lines_per_dept: Optional[int] = None
    ) -> str:
        """
        Get a consolidated view of all department memories.

        Args:
            category_filter: Optional category to filter by
            max_lines_per_dept: Optional limit per department

        Returns:
            Consolidated markdown string
        """
        sections = []

        for dept in sorted(Department, key=lambda d: d.value):
            try:
                manager = self._get_memory_manager(dept)

                if category_filter:
                    # Search for category-specific content
                    matches = manager.search(category_filter, category=category_filter)
                    if matches:
                        sections.append(f"\n## {dept.value.title()} Department\n")
                        for match in matches:
                            sections.append(f"- {dept.value}: {match}\n")
                else:
                    # Get full memory
                    content = manager.read_memory(max_lines=max_lines_per_dept)
                    if content:
                        sections.append(f"\n## {dept.value.title()} Department\n")
                        sections.append(content)

            except (FileNotFoundError, IOError):
                continue

        return "\n".join(sections)

    def get_memory_timeline(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get a timeline of recent memories across all departments.

        Args:
            limit: Maximum number of entries

        Returns:
            List of memory entries with timestamp and department
        """
        timeline = []

        for dept in Department:
            try:
                manager = self._get_memory_manager(dept)

                # Read memory and parse for recent entries
                content = manager.read_memory()
                if content:
                    # Parse memory entries (simplified - just get recent content)
                    lines = content.split("\n")
                    recent_lines = lines[-limit:] if len(lines) > limit else lines

                    for line in recent_lines:
                        if line.strip() and not line.startswith("#"):
                            timeline.append({
                                "department": dept.value,
                                "content": line.strip(),
                                "source": "memory"
                            })

                            if len(timeline) >= limit:
                                break

                if len(timeline) >= limit:
                    break

            except (FileNotFoundError, IOError):
                continue

        return timeline[:limit]

    def get_cross_department_insights(
        self,
        query: str
    ) -> Dict[str, Any]:
        """
        Get insights about a topic across all departments.

        Args:
            query: The topic to analyze

        Returns:
            Dictionary with cross-department analysis
        """
        results = self.search_all(query)

        # Group by department
        by_department: Dict[str, List[str]] = {}
        for result in results:
            dept = result["department"]
            if dept not in by_department:
                by_department[dept] = []
            by_department[dept].append(result["content"])

        return {
            "query": query,
            "total_matches": len(results),
            "departments_involved": list(by_department.keys()),
            "by_department": by_department
        }

    def check_department_health(self) -> Dict[str, Any]:
        """
        Check the health of all department memory systems.

        Returns:
            Dictionary with health status for each department
        """
        health = {}

        for dept in Department:
            dept_path = self.base_path / dept.value
            memory_file = dept_path / "MEMORY.md"
            memory_dir = dept_path / "memory"

            health[dept.value] = {
                "directory_exists": dept_path.exists(),
                "memory_file_exists": memory_file.exists(),
                "memory_dir_exists": memory_dir.exists(),
                "daily_log_count": len(list(memory_dir.glob("*.md"))) if memory_dir.exists() else 0
            }

        return health

    def get_active_departments(self) -> List[str]:
        """
        Get list of departments with active memory systems.

        Returns:
            List of department names that have memory files
        """
        active = []

        for dept in Department:
            dept_path = self.base_path / dept.value
            memory_file = dept_path / "MEMORY.md"

            if memory_file.exists():
                active.append(dept.value)

        return active

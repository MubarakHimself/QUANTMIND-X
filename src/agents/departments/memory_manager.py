# src/agents/departments/memory_manager.py
"""
Markdown-based Memory Management for Department-Based Agent Framework.

Implements a simple markdown-based memory system with:
- MEMORY.md: Permanent memories organized by category
- memory/YYYY-MM-DD.md: Daily logs for temporal tracking

Pattern inspired by openclaw's markdown memory approach.
"""
import os
import re
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

from src.agents.departments.types import Department


class DepartmentMemoryManager:
    """
    Markdown-based memory manager for department agents.

    Provides persistent storage without requiring embeddings or vector databases.
    Uses markdown files for human-readable and AI-parseable memory storage.

    Memory structure:
    - {dept_path}/MEMORY.md - Permanent categorized memories
    - {dept_path}/memory/YYYY-MM-DD.md - Daily logs

    Args:
        department: The department enum
        base_path: Base directory for all departments
    """

    def __init__(
        self,
        department: Department,
        base_path: Optional[Path] = None,
        auto_initialize: bool = True
    ):
        """
        Initialize the memory manager.

        Args:
            department: The department enum
            base_path: Base directory for all departments (default: Path("data/departments"))
            auto_initialize: Whether to auto-create directories and files (default: True)
        """
        self.department = department
        self.base_path = Path(base_path) if base_path else Path("data/departments")
        self.dept_path = self.base_path / department.value
        self.memory_dir = self.dept_path / "memory"
        self.memory_file = self.dept_path / "MEMORY.md"

        # Create directories and initialize memory file
        if auto_initialize:
            self._initialize()

    def _initialize(self) -> None:
        """Create directory structure and initialize MEMORY.md if needed."""
        # Create department directory
        self.dept_path.mkdir(parents=True, exist_ok=True)

        # Create memory subdirectory
        self.memory_dir.mkdir(exist_ok=True)

        # Initialize MEMORY.md if it doesn't exist
        if not self.memory_file.exists():
            self._initialize_memory_file()

    def _initialize_memory_file(self) -> None:
        """Create MEMORY.md with department header."""
        header = f"""# {self.department.value.title()} Department Memory

*Last updated: {datetime.now().isoformat()}*

## Categories

Use this file to store important memories, decisions, and knowledge.

"""

        self.memory_file.write_text(header)

    def add_memory(
        self,
        category: str,
        content: str,
        tags: Optional[List[str]] = None
    ) -> None:
        """
        Add a memory entry to MEMORY.md.

        Args:
            category: Category/section for the memory
            content: The memory content
            tags: Optional list of tags for organization
        """
        timestamp = datetime.now().isoformat()

        # Format tags if provided
        tag_line = ""
        if tags:
            tag_line = f"\n**Tags:** {', '.join(f'`{tag}`' for tag in tags)}"

        # Create memory entry
        entry = f"""

## {category.title()} - {timestamp}

{content}{tag_line}

---

"""

        # Append to MEMORY.md
        with open(self.memory_file, "a") as f:
            f.write(entry)

    def add_daily_log(
        self,
        date: date,
        content: str,
        category: Optional[str] = None
    ) -> None:
        """
        Add an entry to a daily log.

        Args:
            date: The date for the log entry
            content: The log content
            category: Optional category section within the daily log
        """
        # Create daily log file path
        log_file = self.memory_dir / f"{date.isoformat()}.md"

        # Get current time
        now = datetime.now()
        time_str = now.strftime("%H:%M:%S")

        # Format the entry
        if category:
            entry = f"""

### {category.title()} - {time_str}

{content}

"""
        else:
            entry = f"""

## {time_str}

{content}

"""

        # Append to daily log
        if log_file.exists():
            with open(log_file, "a") as f:
                f.write(entry)
        else:
            # Create new daily log with header
            header = f"""# Daily Log - {date.strftime("%B %d, %Y")}

*Department: {self.department.value.title()}*

"""
            log_file.write_text(header + entry)

    def read_memory(self, max_lines: Optional[int] = None) -> str:
        """
        Read the full MEMORY.md content.

        Args:
            max_lines: Optional limit on number of lines to return

        Returns:
            The content of MEMORY.md
        """
        content = self.memory_file.read_text()

        if max_lines:
            lines = content.split("\n")[:max_lines]
            content = "\n".join(lines)

        return content

    def search(
        self,
        query: str,
        category: Optional[str] = None
    ) -> List[str]:
        """
        Search MEMORY.md for matching content.

        Args:
            query: The search term
            category: Optional category filter

        Returns:
            List of matching excerpts with context
        """
        content = self.memory_file.read_text()

        # Filter by category if specified
        if category:
            # Extract category section
            pattern = rf"## {category.title()}.*?(?=\n## |\Z)"
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                content = match.group(0)
            else:
                return []

        # Case-insensitive search for query
        pattern = rf".{{0,100}}{re.escape(query)}.{{0,100}}"
        matches = re.findall(pattern, content, re.IGNORECASE)

        return matches

    def get_daily_log(self, date: date) -> Optional[str]:
        """
        Get a specific daily log by date.

        Args:
            date: The date to retrieve

        Returns:
            The daily log content or None if not found
        """
        log_file = self.memory_dir / f"{date.isoformat()}.md"

        if log_file.exists():
            return log_file.read_text()

        return None

    def get_recent_logs(self, days: int = 7) -> Dict[str, str]:
        """
        Get recent daily logs.

        Args:
            days: Number of recent days to retrieve

        Returns:
            Dictionary mapping date strings to log contents
        """
        recent_logs = {}

        # Get all daily log files
        log_files = list(self.memory_dir.glob("*.md"))

        # Sort by date (filename is YYYY-MM-DD.md)
        log_files.sort(reverse=True)

        # Take the most recent ones
        for log_file in log_files[:days]:
            date_str = log_file.stem  # Removes .md suffix
            recent_logs[date_str] = log_file.read_text()

        return recent_logs

    def list_categories(self) -> Set[str]:
        """
        Extract unique categories from MEMORY.md.

        Returns:
            Set of category names
        """
        content = self.memory_file.read_text()

        # Find all category headers (## Category - timestamp)
        pattern = r"## ([^-]+) - "
        matches = re.findall(pattern, content)

        # Normalize and deduplicate
        categories = {m.strip().lower() for m in matches}

        return categories

    def get_stats(self) -> Dict[str, any]:
        """
        Get memory statistics.

        Returns:
            Dictionary with stats including total entries and category breakdown
        """
        content = self.memory_file.read_text()

        # Count memory entries (## headers)
        total_entries = len(re.findall(r"^## ", content, re.MULTILINE))

        # Get categories
        categories = self.list_categories()

        # Count daily logs
        daily_logs = list(self.memory_dir.glob("*.md"))
        daily_log_count = len(daily_logs)

        return {
            "total_entries": total_entries,
            "categories": sorted(categories),
            "daily_log_count": daily_log_count,
            "memory_file": str(self.memory_file),
            "memory_dir": str(self.memory_dir),
        }

    def get_memory_file_path(self) -> Path:
        """Get the path to MEMORY.md."""
        return self.memory_file

    def get_memory_dir_path(self) -> Path:
        """Get the path to the memory directory."""
        return self.memory_dir

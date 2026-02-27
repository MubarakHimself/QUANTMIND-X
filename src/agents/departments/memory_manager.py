"""
Markdown-based department memory system.

Based on openclaw pattern: MEMORY.md + memory/YYYY-MM-DD.md
No embeddings - plain markdown files as source of truth.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, List

from .types import Department


class MemoryResult:
    """Result from memory search."""
    def __init__(self, path: str, line: int, snippet: str, score: float = 1.0):
        self.path = path
        self.line = line
        self.snippet = snippet
        self.score = score


class DepartmentMemoryManager:
    """Markdown-based memory for each department (no embeddings)."""

    def __init__(self, department: Department, base_path: str = ".quantmind/departments"):
        self.department = department
        base = Path(base_path)
        self.dept_path = base / department.value
        self.memory_file = self.dept_path / "MEMORY.md"
        self.memory_dir = self.dept_path / "memory"
        self._ensure_directories()

    def _ensure_directories(self):
        """Create memory directories if they don't exist."""
        self.dept_path.mkdir(parents=True, exist_ok=True)
        self.memory_dir.mkdir(parents=True, exist_ok=True)

    def add_memory(self, content: str, memory_type: str = "note") -> str:
        """Add memory to department's MEMORY.md file."""
        self._ensure_directories()
        timestamp = datetime.now().isoformat()

        # Format the memory entry
        entry = f"\n## {memory_type.capitalize()} - {timestamp}\n\n{content}\n"

        # Append to MEMORY.md
        with open(self.memory_file, "a") as f:
            f.write(entry)

        return str(self.memory_file)

    def add_daily_log(self, content: str) -> str:
        """Append to today's daily log file."""
        self._ensure_directories()
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = self.memory_dir / f"{today}.md"

        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"\n### {timestamp}\n\n{content}\n"

        with open(log_file, "a") as f:
            f.write(entry)

        return str(log_file)

    def read_memory(self) -> str:
        """Read entire MEMORY.md content."""
        if not self.memory_file.exists():
            return ""
        with open(self.memory_file, "r") as f:
            return f.read()

    def read_daily_logs(self, date: Optional[str] = None) -> str:
        """Read daily log for specific date or today."""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        log_file = self.memory_dir / f"{date}.md"
        if not log_file.exists():
            return ""
        with open(log_file, "r") as f:
            return f.read()

    def search(self, query: str, limit: int = 10) -> List[MemoryResult]:
        """Full-text search within department's markdown files."""
        results = []
        query_lower = query.lower()

        # Search MEMORY.md
        if self.memory_file.exists():
            results.extend(self._search_file(str(self.memory_file), query_lower, limit))

        # Search daily logs
        if self.memory_dir.exists():
            for log_file in sorted(self.memory_dir.glob("*.md"), reverse=True):
                if len(results) >= limit:
                    break
                results.extend(self._search_file(str(log_file), query_lower, limit - len(results)))

        return results[:limit]

    def _search_file(self, file_path: str, query: str, limit: int) -> List[MemoryResult]:
        """Search a single markdown file for query."""
        results = []
        try:
            with open(file_path, "r") as f:
                for line_no, line in enumerate(f, 1):
                    if query in line.lower():
                        results.append(MemoryResult(
                            path=file_path,
                            line=line_no,
                            snippet=line.strip()[:200],
                            score=1.0
                        ))
                        if len(results) >= limit:
                            break
        except FileNotFoundError:
            pass
        return results

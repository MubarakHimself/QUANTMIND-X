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
from typing import Any, Dict, List, Optional, Set

from src.agents.departments.types import Department


def _get_default_graph_store():
    """Get default graph store, using patched class if available."""
    try:
        from src.memory.graph.store import GraphMemoryStore
        return GraphMemoryStore()
    except Exception:
        return None


class SessionWorkspace:
    """
    Session-scoped workspace for graph memory operations.

    Provides isolated read/write access to graph memory nodes with:
    - Session ID filtering for cross-session contamination prevention
    - Draft-to-committed state transitions
    - Session-scoped queries

    Used by department agents to maintain isolated session state in graph memory.
    """

    def __init__(self, session_id: str, entity_id: str, graph_store=None):
        """
        Initialize a session workspace.

        Args:
            session_id: Unique session identifier
            entity_id: Entity this session is working on (e.g., strategy_id)
            graph_store: Optional GraphMemoryStore instance (uses default if not provided)
        """
        self.session_id = session_id
        self.entity_id = entity_id
        self._committed = False
        self._graph_store = graph_store
        self._draft_nodes = []

    def _get_graph_store(self):
        """Get the graph store, lazily creating if needed."""
        if self._graph_store is None:
            self._graph_store = _get_default_graph_store()
        return self._graph_store

    def write_node(
        self,
        node_type: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        status: str = "draft",
    ) -> Dict[str, Any]:
        """
        Write a node to the session workspace.

        Args:
            node_type: Type of node (e.g., "opinion", "fact")
            content: Node content
            metadata: Additional metadata
            status: Node status ("draft" or "committed")

        Returns:
            Created node dict with session_id captured
        """
        node_data = {
            "session_id": self.session_id,
            "entity_id": self.entity_id,
            "node_type": node_type,
            "content": content,
            "metadata": metadata or {},
            "status": status if not self._committed else "committed",
        }

        store = self._get_graph_store()
        if store is not None:
            result = store.create_node(node_data)
            node_data["node_id"] = result.get("node_id") if result else None
        else:
            node_data["node_id"] = f"node_{self.session_id}_{node_type}"

        self._draft_nodes.append(node_data)
        return node_data

    def query_nodes(
        self,
        node_type: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query nodes with session filtering.

        Args:
            node_type: Optional filter by node type
            session_id: Optional explicit session_id (defaults to self.session_id)

        Returns:
            List of matching nodes
        """
        filter_session_id = session_id or self.session_id

        store = self._get_graph_store()
        if store is not None:
            return store.query_nodes(
                node_type=node_type,
                session_id=filter_session_id,
                entity_id=self.entity_id,
            )

        # Fallback: return draft nodes matching filter
        results = [
            n for n in self._draft_nodes
            if n.get("session_id") == filter_session_id
            and (node_type is None or n.get("node_type") == node_type)
        ]
        return results

    def commit(self) -> None:
        """
        Commit all draft nodes in this session.

        After commit:
        - All draft nodes are marked as committed
        - Nodes become visible to other sessions
        """
        self._committed = True
        for node in self._draft_nodes:
            node["status"] = "committed"

    def is_committed(self) -> bool:
        """Return True if this session has been committed."""
        return self._committed


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

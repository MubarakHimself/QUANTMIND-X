# src/agents/departments/cold_storage.py
"""
SQLite-based Cold Storage for Department-Based Agent Framework.

Provides archival and retrieval of old memories that are no longer actively used
but need to be preserved for historical analysis and compliance.
"""
import os
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

from src.agents.departments.types import Department


class ColdStorageManager:
    """
    SQLite-based cold storage manager for archived department memories.

    Stores old memories, daily logs, and historical data in a SQLite database
    for efficient querying and long-term retention.

    Table schema:
    - archived_memories: Individual memory entries from MEMORY.md
    - archived_logs: Daily log entries from memory/YYYY-MM-DD.md files

    Args:
        db_path: Path to SQLite database file
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the cold storage manager.

        Args:
            db_path: Path to SQLite database (default: data/departments/cold_storage.db)
        """
        if db_path is None:
            base_path = Path("data/departments")
            base_path.mkdir(parents=True, exist_ok=True)
            db_path = str(base_path / "cold_storage.db")

        self.db_path = db_path
        self._initialize_database()

    def _initialize_database(self) -> None:
        """Create database tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create archived_memories table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS archived_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                department TEXT NOT NULL,
                category TEXT NOT NULL,
                content TEXT NOT NULL,
                tags TEXT,
                timestamp TEXT NOT NULL,
                source_agent TEXT,
                priority TEXT,
                archived_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create archived_logs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS archived_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                department TEXT NOT NULL,
                log_date TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                archived_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indexes for efficient querying
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_department
            ON archived_memories(department)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_category
            ON archived_memories(category)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_timestamp
            ON archived_memories(timestamp)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_logs_department
            ON archived_logs(department)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_logs_date
            ON archived_logs(log_date)
        """)

        conn.commit()
        conn.close()

    def archive_memory(self, memory_data: Dict[str, Any]) -> Optional[int]:
        """
        Archive a memory entry to cold storage.

        Args:
            memory_data: Dictionary containing:
                - department: str
                - category: str
                - content: str
                - tags: Optional[List[str]]
                - timestamp: str (ISO format)
                - source_agent: Optional[str]
                - priority: Optional[str]

        Returns:
            The ID of the archived entry or None on failure
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Convert tags list to comma-separated string
            tags = memory_data.get("tags")
            tags_str = ",".join(tags) if tags else None

            cursor.execute("""
                INSERT INTO archived_memories
                (department, category, content, tags, timestamp, source_agent, priority)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                memory_data["department"],
                memory_data["category"],
                memory_data["content"],
                tags_str,
                memory_data["timestamp"],
                memory_data.get("source_agent"),
                memory_data.get("priority")
            ))

            conn.commit()
            return cursor.lastrowid

        except sqlite3.Error as e:
            print(f"Error archiving memory: {e}")
            conn.rollback()
            return None

        finally:
            conn.close()

    def archive_daily_log(self, log_data: Dict[str, Any]) -> Optional[int]:
        """
        Archive a daily log entry to cold storage.

        Args:
            log_data: Dictionary containing:
                - department: str
                - log_date: str (YYYY-MM-DD format)
                - content: str
                - timestamp: str (ISO format)

        Returns:
            The ID of the archived entry or None on failure
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT INTO archived_logs
                (department, log_date, content, timestamp)
                VALUES (?, ?, ?, ?)
            """, (
                log_data["department"],
                log_data["log_date"],
                log_data["content"],
                log_data["timestamp"]
            ))

            conn.commit()
            return cursor.lastrowid

        except sqlite3.Error as e:
            print(f"Error archiving log: {e}")
            conn.rollback()
            return None

        finally:
            conn.close()

    def retrieve_by_id(self, entry_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve an archived memory by ID.

        Args:
            entry_id: The ID of the entry

        Returns:
            Dictionary with entry data or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT id, department, category, content, tags, timestamp,
                       source_agent, priority, archived_at
                FROM archived_memories
                WHERE id = ?
            """, (entry_id,))

            row = cursor.fetchone()

            if row:
                return {
                    "id": row[0],
                    "department": row[1],
                    "category": row[2],
                    "content": row[3],
                    "tags": row[4].split(",") if row[4] else [],
                    "timestamp": row[5],
                    "source_agent": row[6],
                    "priority": row[7],
                    "archived_at": row[8]
                }

            return None

        finally:
            conn.close()

    def retrieve_by_department(self, department: str) -> List[Dict[str, Any]]:
        """
        Retrieve all memories for a specific department.

        Args:
            department: The department name

        Returns:
            List of memory dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT id, department, category, content, tags, timestamp,
                       source_agent, priority, archived_at
                FROM archived_memories
                WHERE department = ?
                ORDER BY timestamp DESC
            """, (department,))

            rows = cursor.fetchall()

            results = []
            for row in rows:
                results.append({
                    "id": row[0],
                    "department": row[1],
                    "category": row[2],
                    "content": row[3],
                    "tags": row[4].split(",") if row[4] else [],
                    "timestamp": row[5],
                    "source_agent": row[6],
                    "priority": row[7],
                    "archived_at": row[8]
                })

            return results

        finally:
            conn.close()

    def retrieve_by_category(
        self,
        department: str,
        category: str
    ) -> List[Dict[str, Any]]:
        """
        Retrieve memories for a specific department and category.

        Args:
            department: The department name
            category: The category name

        Returns:
            List of memory dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT id, department, category, content, tags, timestamp,
                       source_agent, priority, archived_at
                FROM archived_memories
                WHERE department = ? AND category = ?
                ORDER BY timestamp DESC
            """, (department, category))

            rows = cursor.fetchall()

            results = []
            for row in rows:
                results.append({
                    "id": row[0],
                    "department": row[1],
                    "category": row[2],
                    "content": row[3],
                    "tags": row[4].split(",") if row[4] else [],
                    "timestamp": row[5],
                    "source_agent": row[6],
                    "priority": row[7],
                    "archived_at": row[8]
                })

            return results

        finally:
            conn.close()

    def retrieve_by_date_range(
        self,
        start_date: date,
        end_date: date,
        department: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve memories within a date range.

        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            department: Optional department filter

        Returns:
            List of memory dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            start_iso = start_date.isoformat()
            end_iso = (end_date + timedelta(days=1)).isoformat()  # Make end date inclusive

            if department:
                cursor.execute("""
                    SELECT id, department, category, content, tags, timestamp,
                           source_agent, priority, archived_at
                    FROM archived_memories
                    WHERE department = ? AND timestamp >= ? AND timestamp < ?
                    ORDER BY timestamp DESC
                """, (department, start_iso, end_iso))
            else:
                cursor.execute("""
                    SELECT id, department, category, content, tags, timestamp,
                           source_agent, priority, archived_at
                    FROM archived_memories
                    WHERE timestamp >= ? AND timestamp < ?
                    ORDER BY timestamp DESC
                """, (start_iso, end_iso))

            rows = cursor.fetchall()

            results = []
            for row in rows:
                results.append({
                    "id": row[0],
                    "department": row[1],
                    "category": row[2],
                    "content": row[3],
                    "tags": row[4].split(",") if row[4] else [],
                    "timestamp": row[5],
                    "source_agent": row[6],
                    "priority": row[7],
                    "archived_at": row[8]
                })

            return results

        finally:
            conn.close()

    def retrieve_recent(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve most recently archived memories.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of memory dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT id, department, category, content, tags, timestamp,
                       source_agent, priority, archived_at
                FROM archived_memories
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))

            rows = cursor.fetchall()

            results = []
            for row in rows:
                results.append({
                    "id": row[0],
                    "department": row[1],
                    "category": row[2],
                    "content": row[3],
                    "tags": row[4].split(",") if row[4] else [],
                    "timestamp": row[5],
                    "source_agent": row[6],
                    "priority": row[7],
                    "archived_at": row[8]
                })

            return results

        finally:
            conn.close()

    def retrieve_all(self) -> List[Dict[str, Any]]:
        """
        Retrieve all archived memories.

        Returns:
            List of all memory dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT id, department, category, content, tags, timestamp,
                       source_agent, priority, archived_at
                FROM archived_memories
                ORDER BY timestamp DESC
            """)

            rows = cursor.fetchall()

            results = []
            for row in rows:
                results.append({
                    "id": row[0],
                    "department": row[1],
                    "category": row[2],
                    "content": row[3],
                    "tags": row[4].split(",") if row[4] else [],
                    "timestamp": row[5],
                    "source_agent": row[6],
                    "priority": row[7],
                    "archived_at": row[8]
                })

            return results

        finally:
            conn.close()

    def search(
        self,
        query: str,
        department: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Search archived memories by content.

        Args:
            query: Search term to find in content
            department: Optional department filter
            limit: Maximum number of results

        Returns:
            List of matching memory dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            if department:
                cursor.execute("""
                    SELECT id, department, category, content, tags, timestamp,
                           source_agent, priority, archived_at
                    FROM archived_memories
                    WHERE department = ? AND content LIKE ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (department, f"%{query}%", limit))
            else:
                cursor.execute("""
                    SELECT id, department, category, content, tags, timestamp,
                           source_agent, priority, archived_at
                    FROM archived_memories
                    WHERE content LIKE ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (f"%{query}%", limit))

            rows = cursor.fetchall()

            results = []
            for row in rows:
                results.append({
                    "id": row[0],
                    "department": row[1],
                    "category": row[2],
                    "content": row[3],
                    "tags": row[4].split(",") if row[4] else [],
                    "timestamp": row[5],
                    "source_agent": row[6],
                    "priority": row[7],
                    "archived_at": row[8]
                })

            return results

        finally:
            conn.close()

    def delete_by_id(self, entry_id: int) -> bool:
        """
        Delete an archived memory by ID.

        Args:
            entry_id: The ID of the entry to delete

        Returns:
            True if deleted, False otherwise
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                DELETE FROM archived_memories
                WHERE id = ?
            """, (entry_id,))

            conn.commit()
            return cursor.rowcount > 0

        except sqlite3.Error:
            conn.rollback()
            return False

        finally:
            conn.close()

    def purge_old(self, days: int = 90) -> int:
        """
        Delete archived memories older than specified days.

        Args:
            days: Number of days to keep (older entries are deleted)

        Returns:
            Number of entries deleted
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

            cursor.execute("""
                DELETE FROM archived_memories
                WHERE timestamp < ?
            """, (cutoff_date,))

            conn.commit()
            return cursor.rowcount

        except sqlite3.Error:
            conn.rollback()
            return 0

        finally:
            conn.close()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about archived data.

        Returns:
            Dictionary with statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Total memories
            cursor.execute("SELECT COUNT(*) FROM archived_memories")
            total_memories = cursor.fetchone()[0]

            # Total logs
            cursor.execute("SELECT COUNT(*) FROM archived_logs")
            total_logs = cursor.fetchone()[0]

            # By department
            cursor.execute("""
                SELECT department, COUNT(*) as count
                FROM archived_memories
                GROUP BY department
            """)
            by_department = {row[0]: row[1] for row in cursor.fetchall()}

            # Oldest entry
            cursor.execute("""
                SELECT MIN(timestamp) FROM archived_memories
            """)
            oldest = cursor.fetchone()[0]

            # Newest entry
            cursor.execute("""
                SELECT MAX(timestamp) FROM archived_memories
            """)
            newest = cursor.fetchone()[0]

            return {
                "total_memories": total_memories,
                "total_logs": total_logs,
                "by_department": by_department,
                "oldest_entry": oldest,
                "newest_entry": newest,
                "db_path": self.db_path
            }

        finally:
            conn.close()

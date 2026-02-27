"""
SQLite cold storage for archived department memories.

Long-term archival of old daily logs and summarized memories.
"""

import sqlite3
from datetime import datetime, timedelta
from typing import Optional, List, Dict

from .types import Department


class ColdStorageManager:
    """SQLite database for archived memories."""

    def __init__(self, db_path: str = ".quantmind/cold_storage.db"):
        self.db_path = db_path
        self.db = sqlite3.connect(db_path)
        self.db.row_factory = sqlite3.Row
        self._init_tables()

    def _init_tables(self):
        """Initialize database tables."""
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS archived_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                department TEXT NOT NULL,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                archived_at TEXT NOT NULL
            )
        """)
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_department ON archived_memories(department)")
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON archived_memories(timestamp)")
        self.db.commit()

    def archive_memories(
        self,
        department: Department,
        content: str,
        memory_type: str,
        timestamp: datetime
    ) -> int:
        """Archive a memory to cold storage."""
        cursor = self.db.execute(
            """
            INSERT INTO archived_memories (department, content, memory_type, timestamp, archived_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (department.value, content, memory_type, timestamp.isoformat(), datetime.now().isoformat())
        )
        self.db.commit()
        return cursor.lastrowid

    def retrieve_archived(
        self,
        query: str,
        department: Optional[Department] = None,
        limit: int = 10
    ) -> List[Dict]:
        """Search archived memories."""
        sql = "SELECT * FROM archived_memories WHERE content LIKE ?"
        params = [f"%{query}%"]

        if department:
            sql += " AND department = ?"
            params.append(department.value)

        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        rows = self.db.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def get_old_memories(self, department: Department, older_than_days: int = 30) -> List[Dict]:
        """Get memories older than specified days."""
        cutoff = datetime.now() - timedelta(days=older_than_days)

        rows = self.db.execute(
            """
            SELECT * FROM archived_memories
            WHERE department = ? AND timestamp < ?
            ORDER BY timestamp ASC
            """,
            (department.value, cutoff.isoformat())
        ).fetchall()

        return [dict(row) for row in rows]

    def close(self):
        """Close database connection."""
        self.db.close()

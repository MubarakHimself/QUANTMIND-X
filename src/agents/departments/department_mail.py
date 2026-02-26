"""
Department Mail Service

SQLite-based mail system for cross-department communication.
Inspired by Overstory's mail system with WAL mode for concurrent access.
"""
import sqlite3
import uuid
import json
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any


class MessageType(str, Enum):
    """Types of department messages."""
    STATUS = "status"
    QUESTION = "question"
    RESULT = "result"
    ERROR = "error"
    DISPATCH = "dispatch"  # Cross-department task dispatch


class Priority(str, Enum):
    """Message priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class DepartmentMessage:
    """
    Message for cross-department communication.

    Attributes:
        id: Unique message identifier
        from_dept: Sending department
        to_dept: Receiving department
        type: Message type (status, question, result, error, dispatch)
        subject: Brief subject line
        body: Message content (can be JSON string)
        priority: Message priority
        timestamp: When message was created
        read: Whether message has been read
    """
    id: str
    from_dept: str
    to_dept: str
    type: MessageType
    subject: str
    body: str
    priority: Priority
    timestamp: datetime
    read: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "from_dept": self.from_dept,
            "to_dept": self.to_dept,
            "type": self.type.value,
            "subject": self.subject,
            "body": self.body,
            "priority": self.priority.value,
            "timestamp": self.timestamp.isoformat(),
            "read": self.read,
        }


class DepartmentMailService:
    """
    SQLite-based mail service for cross-department communication.

    Uses WAL mode for concurrent access from multiple departments.
    Based on Overstory's mail system patterns.
    """

    def __init__(self, db_path: str = ".quantmind/department_mail.db"):
        """
        Initialize the mail service.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.db = sqlite3.connect(str(self.db_path))
        self._init_schema()

    def _init_schema(self):
        """Initialize database schema with WAL mode."""
        self.db.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA busy_timeout=5000;

            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                from_dept TEXT NOT NULL,
                to_dept TEXT NOT NULL,
                type TEXT NOT NULL,
                subject TEXT,
                body TEXT,
                priority TEXT DEFAULT 'normal',
                timestamp TEXT NOT NULL,
                read INTEGER DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_to_dept_read
                ON messages(to_dept, read);
            CREATE INDEX IF NOT EXISTS idx_from_dept
                ON messages(from_dept);
            CREATE INDEX IF NOT EXISTS idx_timestamp
                ON messages(timestamp);
        """)
        self.db.commit()

    def close(self):
        """Close the database connection."""
        if self.db:
            self.db.close()

    def send(
        self,
        from_dept: str,
        to_dept: str,
        type: MessageType,
        subject: str,
        body: str,
        priority: Priority = Priority.NORMAL,
    ) -> DepartmentMessage:
        """
        Send a message to a department inbox.

        Args:
            from_dept: Sending department
            to_dept: Receiving department
            type: Message type
            subject: Brief subject line
            body: Message content
            priority: Message priority

        Returns:
            The created message
        """
        message = DepartmentMessage(
            id=str(uuid.uuid4()),
            from_dept=from_dept,
            to_dept=to_dept,
            type=type,
            subject=subject,
            body=body,
            priority=priority,
            timestamp=datetime.now(timezone.utc),
            read=False,
        )

        cursor = self.db.cursor()
        cursor.execute("""
            INSERT INTO messages (id, from_dept, to_dept, type, subject, body, priority, timestamp, read)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            message.id,
            message.from_dept,
            message.to_dept,
            message.type.value,
            message.subject,
            message.body,
            message.priority.value,
            message.timestamp.isoformat(),
            0,
        ))
        self.db.commit()

        return message

    def check_inbox(
        self,
        dept: str,
        unread_only: bool = True,
        limit: int = 100,
    ) -> List[DepartmentMessage]:
        """
        Check inbox for a department.

        Args:
            dept: Department to check
            unread_only: Only return unread messages
            limit: Maximum messages to return

        Returns:
            List of messages
        """
        cursor = self.db.cursor()

        if unread_only:
            cursor.execute("""
                SELECT id, from_dept, to_dept, type, subject, body, priority, timestamp, read
                FROM messages
                WHERE to_dept = ? AND read = 0
                ORDER BY timestamp DESC
                LIMIT ?
            """, (dept, limit))
        else:
            cursor.execute("""
                SELECT id, from_dept, to_dept, type, subject, body, priority, timestamp, read
                FROM messages
                WHERE to_dept = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (dept, limit))

        messages = []
        for row in cursor.fetchall():
            messages.append(DepartmentMessage(
                id=row[0],
                from_dept=row[1],
                to_dept=row[2],
                type=MessageType(row[3]),
                subject=row[4] or "",
                body=row[5] or "",
                priority=Priority(row[6]),
                timestamp=datetime.fromisoformat(row[7]),
                read=bool(row[8]),
            ))

        return messages

    def mark_read(self, message_id: str) -> bool:
        """
        Mark a message as read.

        Args:
            message_id: ID of message to mark read

        Returns:
            True if message was found and updated
        """
        cursor = self.db.cursor()
        cursor.execute("""
            UPDATE messages SET read = 1 WHERE id = ?
        """, (message_id,))
        self.db.commit()
        return cursor.rowcount > 0

    def get_message(self, message_id: str) -> Optional[DepartmentMessage]:
        """
        Get a specific message by ID.

        Args:
            message_id: Message ID

        Returns:
            Message or None if not found
        """
        cursor = self.db.cursor()
        cursor.execute("""
            SELECT id, from_dept, to_dept, type, subject, body, priority, timestamp, read
            FROM messages WHERE id = ?
        """, (message_id,))

        row = cursor.fetchone()
        if row is None:
            return None

        return DepartmentMessage(
            id=row[0],
            from_dept=row[1],
            to_dept=row[2],
            type=MessageType(row[3]),
            subject=row[4] or "",
            body=row[5] or "",
            priority=Priority(row[6]),
            timestamp=datetime.fromisoformat(row[7]),
            read=bool(row[8]),
        )

    def purge_old_messages(self, days: int = 30) -> int:
        """
        Delete messages older than specified days.

        Args:
            days: Number of days to keep

        Returns:
            Number of messages deleted
        """
        cursor = self.db.cursor()
        cursor.execute("""
            DELETE FROM messages
            WHERE timestamp < datetime('now', ?)
        """, (f"-{days} days",))
        self.db.commit()
        return cursor.rowcount

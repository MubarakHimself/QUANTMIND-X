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
    # Workflow message types
    STRATEGY_DISPATCH = "strategy_dispatch"  # Video → Research
    TRD_GENERATED = "trd_generated"          # Research → Dev
    CODE_READY = "code_ready"                 # Dev → Risk
    RISK_CLEARED = "risk_cleared"             # Risk → Execution
    BACKTEST_REQUEST = "backtest_request"     # Any → Research
    # Approval gate message types
    APPROVAL_REQUEST = "approval_request"    # Approval gate created
    APPROVAL_APPROVED = "approval_approved"  # Approval granted
    APPROVAL_REJECTED = "approval_rejected"  # Approval rejected


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
        # Approval gate related fields
        gate_id: Optional approval gate ID for approval messages
        workflow_id: Optional workflow ID for context
        from_stage: Optional source stage for approval transitions
        to_stage: Optional target stage for approval transitions
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
    gate_id: Optional[str] = None
    workflow_id: Optional[str] = None
    from_stage: Optional[str] = None
    to_stage: Optional[str] = None

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
            "gate_id": self.gate_id,
            "workflow_id": self.workflow_id,
            "from_stage": self.from_stage,
            "to_stage": self.to_stage,
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
                read INTEGER DEFAULT 0,
                gate_id TEXT,
                workflow_id TEXT,
                from_stage TEXT,
                to_stage TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_to_dept_read
                ON messages(to_dept, read);
            CREATE INDEX IF NOT EXISTS idx_from_dept
                ON messages(from_dept);
            CREATE INDEX IF NOT EXISTS idx_timestamp
                ON messages(timestamp);
            CREATE INDEX IF NOT EXISTS idx_gate_id
                ON messages(gate_id);
            CREATE INDEX IF NOT EXISTS idx_workflow_id
                ON messages(workflow_id);
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
        gate_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        from_stage: Optional[str] = None,
        to_stage: Optional[str] = None,
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
            gate_id: Optional approval gate ID for approval messages
            workflow_id: Optional workflow ID for context
            from_stage: Optional source stage for approval transitions
            to_stage: Optional target stage for approval transitions

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
            gate_id=gate_id,
            workflow_id=workflow_id,
            from_stage=from_stage,
            to_stage=to_stage,
        )

        cursor = self.db.cursor()
        cursor.execute("""
            INSERT INTO messages (id, from_dept, to_dept, type, subject, body, priority, timestamp, read, gate_id, workflow_id, from_stage, to_stage)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            message.gate_id,
            message.workflow_id,
            message.from_stage,
            message.to_stage,
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
                SELECT id, from_dept, to_dept, type, subject, body, priority, timestamp, read, gate_id, workflow_id, from_stage, to_stage
                FROM messages
                WHERE to_dept = ? AND read = 0
                ORDER BY timestamp DESC
                LIMIT ?
            """, (dept, limit))
        else:
            cursor.execute("""
                SELECT id, from_dept, to_dept, type, subject, body, priority, timestamp, read, gate_id, workflow_id, from_stage, to_stage
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
                gate_id=row[9],
                workflow_id=row[10],
                from_stage=row[11],
                to_stage=row[12],
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
            SELECT id, from_dept, to_dept, type, subject, body, priority, timestamp, read, gate_id, workflow_id, from_stage, to_stage
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
            gate_id=row[9],
            workflow_id=row[10],
            from_stage=row[11],
            to_stage=row[12],
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

    def get_messages_by_gate(self, gate_id: str) -> List[DepartmentMessage]:
        """
        Get all messages related to a specific approval gate.

        Args:
            gate_id: The approval gate ID

        Returns:
            List of messages for this gate
        """
        cursor = self.db.cursor()
        cursor.execute("""
            SELECT id, from_dept, to_dept, type, subject, body, priority, timestamp, read, gate_id, workflow_id, from_stage, to_stage
            FROM messages
            WHERE gate_id = ?
            ORDER BY timestamp DESC
        """, (gate_id,))

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
                gate_id=row[9],
                workflow_id=row[10],
                from_stage=row[11],
                to_stage=row[12],
            ))

        return messages

    def get_messages_by_workflow(self, workflow_id: str) -> List[DepartmentMessage]:
        """
        Get all messages related to a specific workflow.

        Args:
            workflow_id: The workflow ID

        Returns:
            List of messages for this workflow
        """
        cursor = self.db.cursor()
        cursor.execute("""
            SELECT id, from_dept, to_dept, type, subject, body, priority, timestamp, read, gate_id, workflow_id, from_stage, to_stage
            FROM messages
            WHERE workflow_id = ?
            ORDER BY timestamp DESC
        """, (workflow_id,))

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
                gate_id=row[9],
                workflow_id=row[10],
                from_stage=row[11],
                to_stage=row[12],
            ))

        return messages

    def send_approval_notification(
        self,
        from_dept: str,
        to_dept: str,
        gate_id: str,
        workflow_id: str,
        from_stage: str,
        to_stage: str,
        action: str,
        requester: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> DepartmentMessage:
        """
        Send an approval-related notification message.

        Args:
            from_dept: Sending department (system)
            to_dept: Receiving department
            gate_id: The approval gate ID
            workflow_id: The workflow ID
            from_stage: Current stage
            to_stage: Target stage
            action: The approval action (created, approved, rejected)
            requester: Who requested the approval
            reason: Reason for the request

        Returns:
            The created message
        """
        if action == "created":
            msg_type = MessageType.APPROVAL_REQUEST
            subject = f"Approval Required: {from_stage} → {to_stage}"
            body = f"An approval is required to transition from '{from_stage}' to '{to_stage}'.\n\n"
            if reason:
                body += f"Reason: {reason}\n\n"
            if requester:
                body += f"Requested by: {requester}\n"
            body += f"Workflow ID: {workflow_id}\n"
            body += f"Gate ID: {gate_id}\n"
            priority = Priority.HIGH
        elif action == "approved":
            msg_type = MessageType.APPROVAL_APPROVED
            subject = f"Approved: {from_stage} → {to_stage}"
            body = f"The transition from '{from_stage}' to '{to_stage}' has been approved.\n\n"
            if requester:
                body += f"Approved by: {requester}\n"
            body += f"Workflow ID: {workflow_id}\n"
            body += f"Gate ID: {gate_id}\n"
            priority = Priority.NORMAL
        else:  # rejected
            msg_type = MessageType.APPROVAL_REJECTED
            subject = f"Rejected: {from_stage} → {to_stage}"
            body = f"The transition from '{from_stage}' to '{to_stage}' has been rejected.\n\n"
            if reason:
                body += f"Reason: {reason}\n"
            if requester:
                body += f"Rejected by: {requester}\n"
            body += f"Workflow ID: {workflow_id}\n"
            body += f"Gate ID: {gate_id}\n"
            priority = Priority.HIGH

        return self.send(
            from_dept=from_dept,
            to_dept=to_dept,
            type=msg_type,
            subject=subject,
            body=body,
            priority=priority,
            gate_id=gate_id,
            workflow_id=workflow_id,
            from_stage=from_stage,
            to_stage=to_stage,
        )


# Singleton instance
_mail_service: Optional[DepartmentMailService] = None


def get_mail_service(db_path: str = ".quantmind/department_mail.db") -> DepartmentMailService:
    """
    Get or create the singleton DepartmentMailService instance.

    Args:
        db_path: Path to the SQLite database file

    Returns:
        The DepartmentMailService singleton instance
    """
    global _mail_service
    if _mail_service is None:
        _mail_service = DepartmentMailService(db_path)
    return _mail_service


def reset_mail_service() -> None:
    """
    Reset the mail service singleton. Useful for testing.
    """
    global _mail_service
    if _mail_service is not None:
        _mail_service.close()
        _mail_service = None

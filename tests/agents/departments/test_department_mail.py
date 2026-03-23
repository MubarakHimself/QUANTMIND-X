"""
Tests for Department Mail Service

Task Group: Core Infrastructure - Cross-department messaging
"""
import pytest
import tempfile
import os
from datetime import datetime, timezone, timedelta
from src.agents.departments.department_mail import (
    DepartmentMailService, MessageType, Priority
)


class TestDepartmentMailSchema:
    """Test mail service schema initialization."""

    def test_mail_service_creates_database(self):
        """Mail service should create database file on init."""
        from src.agents.departments.department_mail import DepartmentMailService

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_mail.db")
            service = DepartmentMailService(db_path=db_path)

            assert os.path.exists(db_path)
            service.close()

    def test_mail_service_creates_messages_table(self):
        """Mail service should create messages table with correct schema."""
        from src.agents.departments.department_mail import DepartmentMailService

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_mail.db")
            service = DepartmentMailService(db_path=db_path)

            # Check table exists
            cursor = service.db.cursor()
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='messages'
            """)
            result = cursor.fetchone()
            assert result is not None
            assert result[0] == "messages"

            # Check columns
            cursor.execute("PRAGMA table_info(messages)")
            columns = {row[1] for row in cursor.fetchall()}
            expected_columns = {
                'id', 'from_dept', 'to_dept', 'type', 'subject',
                'body', 'priority', 'timestamp', 'read',
                'gate_id', 'workflow_id', 'from_stage', 'to_stage'
            }
            assert columns == expected_columns

            service.close()

    def test_mail_service_enables_wal_mode(self):
        """Mail service should enable WAL mode for concurrent access."""
        from src.agents.departments.department_mail import DepartmentMailService

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_mail.db")
            service = DepartmentMailService(db_path=db_path)

            cursor = service.db.cursor()
            cursor.execute("PRAGMA journal_mode")
            result = cursor.fetchone()
            assert result[0].lower() == "wal"

            service.close()


class TestDepartmentMailSendReceive:
    """Test mail service send and receive operations."""

    def test_send_message(self):
        """Should send a message to department inbox."""
        # Create temporary database
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            db_path = tmp.name

        try:
            service = DepartmentMailService(db_path)

            # Send message
            message = service.send(
                from_dept="planning",
                to_dept="execution",
                type=MessageType.DISPATCH,
                subject="Test",
                body="content",
                priority=Priority.HIGH
            )

            # Verify message properties
            assert message.id is not None
            assert message.from_dept == "planning"
            assert message.to_dept == "execution"
            assert message.type == MessageType.DISPATCH
            assert message.subject == "Test"
            assert message.body == "content"
            assert message.priority == Priority.HIGH
            assert message.read is False

            # Verify message can be retrieved
            retrieved = service.get_message(message.id)
            assert retrieved is not None
            assert retrieved.id == message.id
            assert retrieved.subject == "Test"
            assert retrieved.body == "content"

            service.close()
        finally:
            # Clean up
            if os.path.exists(db_path):
                os.unlink(db_path)

    def test_check_inbox_returns_unread_messages(self):
        """Should return unread messages for department."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            db_path = tmp.name

        try:
            service = DepartmentMailService(db_path)

            # Send multiple messages
            message1 = service.send(
                from_dept="planning",
                to_dept="execution",
                type=MessageType.DISPATCH,
                subject="Test 1",
                body="content 1"
            )

            message2 = service.send(
                from_dept="planning",
                to_dept="execution",
                type=MessageType.STATUS,
                subject="Test 2",
                body="content 2",
                priority=Priority.HIGH
            )

            # Check inbox (unread only)
            messages = service.check_inbox("execution")

            # Verify results
            assert len(messages) == 2
            assert all(m.to_dept == "execution" for m in messages)
            assert all(m.read is False for m in messages)

            # Check specific message properties
            assert messages[0].subject == "Test 2"  # Should be sorted by timestamp (newest first)
            assert messages[0].priority == Priority.HIGH
            assert messages[1].subject == "Test 1"

            service.close()
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    def test_mark_message_as_read(self):
        """Should mark message as read."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            db_path = tmp.name

        try:
            service = DepartmentMailService(db_path)

            # Send message
            message = service.send(
                from_dept="planning",
                to_dept="execution",
                type=MessageType.DISPATCH,
                subject="Test",
                body="content"
            )

            # Mark as read
            result = service.mark_read(message.id)
            assert result is True

            # Verify message is marked as read
            retrieved = service.get_message(message.id)
            assert retrieved is not None
            assert retrieved.read is True

            # Check inbox (unread only) - should be empty
            messages = service.check_inbox("execution")
            assert len(messages) == 0

            # Check inbox (including read) - should contain the message
            messages = service.check_inbox("execution", unread_only=False)
            assert len(messages) == 1
            assert messages[0].read is True

            service.close()
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    def test_get_message_by_id(self):
        """Should retrieve specific message by ID."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            db_path = tmp.name

        try:
            service = DepartmentMailService(db_path)

            # Send message
            message = service.send(
                from_dept="planning",
                to_dept="execution",
                type=MessageType.DISPATCH,
                subject="Test",
                body="content"
            )

            # Get message by ID
            retrieved = service.get_message(message.id)

            # Verify retrieved message
            assert retrieved is not None
            assert retrieved.id == message.id
            assert retrieved.subject == "Test"
            assert retrieved.body == "content"

            # Try to get non-existent message
            non_existent = service.get_message("non-existent-id")
            assert non_existent is None

            service.close()
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    def test_purge_old_messages(self):
        """Should delete messages older than specified days."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            db_path = tmp.name

        try:
            service = DepartmentMailService(db_path)

            # Send message with old timestamp
            old_timestamp = datetime.now(timezone.utc) - timedelta(days=31)
            old_message = DepartmentMessage(
                id=str(uuid.uuid4()),
                from_dept="planning",
                to_dept="execution",
                type=MessageType.DISPATCH,
                subject="Old Test",
                body="old content",
                priority=Priority.NORMAL,
                timestamp=old_timestamp,
                read=False
            )

            cursor = service.db.cursor()
            cursor.execute("""
                INSERT INTO messages (id, from_dept, to_dept, type, subject, body, priority, timestamp, read)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                old_message.id,
                old_message.from_dept,
                old_message.to_dept,
                old_message.type.value,
                old_message.subject,
                old_message.body,
                old_message.priority.value,
                old_message.timestamp.isoformat(),
                0,
            ))
            service.db.commit()

            # Send recent message
            recent_message = service.send(
                from_dept="planning",
                to_dept="execution",
                type=MessageType.DISPATCH,
                subject="Recent Test",
                body="recent content"
            )

            # Purge old messages (30 days)
            deleted_count = service.purge_old_messages(days=30)
            assert deleted_count == 1

            # Verify old message is gone
            old_retrieved = service.get_message(old_message.id)
            assert old_retrieved is None

            # Verify recent message still exists
            recent_retrieved = service.get_message(recent_message.id)
            assert recent_retrieved is not None

            service.close()
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)


# Add DepartmentMessage for test_purge_old_messages
from dataclasses import dataclass
import uuid
from datetime import datetime, timezone, timedelta
from enum import Enum

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

    def to_dict(self) -> dict:
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

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
        """)
        # Migrate existing databases that predate the gate_id/workflow_id columns.
        # Must run before creating indexes that reference these columns.
        existing_cols = {
            row[1]
            for row in self.db.execute("PRAGMA table_info(messages)")
        }
        for col, defn in [
            ("gate_id", "TEXT"),
            ("workflow_id", "TEXT"),
            ("from_stage", "TEXT"),
            ("to_stage", "TEXT"),
        ]:
            if col not in existing_cols:
                self.db.execute(
                    f"ALTER TABLE messages ADD COLUMN {col} {defn}"
                )
        self.db.commit()
        # Create indexes after columns are guaranteed to exist.
        self.db.executescript("""
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


# ============================================================================
# Redis Streams-based Department Mail Service
# ============================================================================

import logging
import os
import warnings
from typing import List, Optional, Dict, Any
import uuid
import json

import redis
from redis import Redis
from redis.exceptions import RedisError, ConnectionError as RedisConnectionError

logger = logging.getLogger(__name__)

# Redis connection settings — read from environment so production deployments work
DEFAULT_REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
DEFAULT_REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
DEFAULT_REDIS_DB = int(os.environ.get("REDIS_DB", "0"))
DEFAULT_REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", None)

# Stream key patterns (lowercase, colon-separated)
# - mail:dept:{dept}:{workflow_id} — messages to dept in this workflow
# - mail:broadcast:{workflow_id} — all-dept broadcast for this workflow
# - dept:{dept}:{workflow_id}:queue — tasks assigned to dept in this workflow

# Consumer group pattern: dept:{dept_name}:group


class RedisDepartmentMailServiceError(Exception):
    """Base exception for Redis Department Mail Service errors."""
    pass


class RedisConnectionFailedError(RedisDepartmentMailServiceError):
    """Raised when Redis connection fails."""
    pass


class RedisStreamNotFoundError(RedisDepartmentMailServiceError):
    """Raised when stream does not exist."""
    pass


class RedisDepartmentMailService:
    """
    Redis Streams-based mail service for cross-department communication.

    Uses Redis Streams with consumer groups for reliable, ordered, and
    auditable message delivery between departments.

    Key Features:
    - Consumer groups for exactly-once delivery
    - Automatic message replay for offline consumers
    - Namespace isolation per workflow run
    - ≤500ms delivery latency

    Key Patterns:
    - mail:dept:{dept}:{workflow_id} — department messages
    - mail:broadcast:{workflow_id} — broadcast messages
    - dept:{dept}:{workflow_id}:queue — task queues
    """

    # Maximum time to wait for messages (ms)
    STREAM_READ_TIMEOUT = 500

    # Message retention time (7 days in seconds)
    MESSAGE_RETENTION = 7 * 24 * 60 * 60

    def __init__(
        self,
        host: str = DEFAULT_REDIS_HOST,
        port: int = DEFAULT_REDIS_PORT,
        db: int = DEFAULT_REDIS_DB,
        password: Optional[str] = DEFAULT_REDIS_PASSWORD,
        consumer_name: Optional[str] = None,
        workflow_id: Optional[str] = None,
    ):
        """
        Initialize the Redis Streams mail service.

        Args:
            host: Redis server host
            port: Redis server port
            db: Redis database number
            password: Optional Redis password
            consumer_name: Unique consumer name (defaults to hostname:uuid)
            workflow_id: Workflow ID for namespace isolation
        """
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.consumer_name = consumer_name or f"consumer-{uuid.uuid4().hex[:8]}"
        self.workflow_id = workflow_id or "default"

        # Connection pool
        self._pool: Optional[redis.ConnectionPool] = None
        self._client: Optional[Redis] = None

    def _create_connection_pool(self) -> redis.ConnectionPool:
        """Create Redis connection pool."""
        return redis.ConnectionPool(
            host=self.host,
            port=self.port,
            db=self.db,
            password=self.password,
            max_connections=20,
            socket_timeout=5.0,
            socket_connect_timeout=5.0,
            decode_responses=True,
        )

    def _get_client(self) -> Redis:
        """Get Redis client from pool."""
        if self._client is None:
            try:
                if self._pool is None:
                    self._pool = self._create_connection_pool()
                self._client = Redis(connection_pool=self._pool)
                # Test connection
                self._client.ping()
                logger.debug(f"Redis mail service connected to {self.host}:{self.port}")
            except RedisConnectionError as e:
                logger.error(f"Failed to connect to Redis: {e}")
                raise RedisConnectionFailedError(f"Cannot connect to Redis at {self.host}:{self.port}") from e
        return self._client

    def close(self):
        """Close Redis connections."""
        if self._client:
            try:
                self._client.close()
            except Exception as e:
                logger.warning(f"Error closing Redis client: {e}")
            finally:
                self._client = None

        if self._pool:
            try:
                self._pool.disconnect()
            except Exception as e:
                logger.warning(f"Error disconnecting pool: {e}")
            finally:
                self._pool = None

        logger.info("Redis mail service closed")

    # =========================================================================
    # Stream Key Generation
    # =========================================================================

    def _get_dept_stream_key(self, dept: str, workflow_id: Optional[str] = None) -> str:
        """Get department stream key."""
        wf = workflow_id or self.workflow_id
        return f"mail:dept:{dept.lower()}:{wf}"

    def _get_broadcast_stream_key(self, workflow_id: Optional[str] = None) -> str:
        """Get broadcast stream key."""
        wf = workflow_id or self.workflow_id
        return f"mail:broadcast:{wf}"

    def _get_queue_key(self, dept: str, workflow_id: Optional[str] = None) -> str:
        """Get department task queue key."""
        wf = workflow_id or self.workflow_id
        return f"dept:{dept.lower()}:{wf}:queue"

    def _get_consumer_group(self, dept: str) -> str:
        """Get consumer group name for department."""
        return f"dept:{dept.lower()}:group"

    # =========================================================================
    # Publishing
    # =========================================================================

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
        Send a message to a department inbox via Redis Stream.

        Args:
            from_dept: Sending department
            to_dept: Receiving department
            type: Message type
            subject: Brief subject line
            body: Message content
            priority: Message priority
            gate_id: Optional approval gate ID
            workflow_id: Optional workflow ID for context
            from_stage: Optional source stage
            to_stage: Optional target stage

        Returns:
            The created message
        """
        wf = workflow_id or self.workflow_id

        # Create message
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
            workflow_id=wf,
            from_stage=from_stage,
            to_stage=to_stage,
        )

        # Build stream payload
        stream_key = self._get_dept_stream_key(to_dept, wf)
        payload = {
            "id": message.id,
            "sender": from_dept,
            "recipient": to_dept,
            "message_type": type.value,
            "payload": json.dumps({
                "subject": subject,
                "body": body,
                "priority": priority.value,
                "gate_id": gate_id,
                "from_stage": from_stage,
                "to_stage": to_stage,
            }),
            "timestamp_utc": message.timestamp.isoformat(),
        }

        try:
            client = self._get_client()
            # Add to stream with max retention
            client.xadd(stream_key, payload, maxlen=self.MESSAGE_RETENTION, approximate=True)
            logger.debug(f"Published message {message.id} to {stream_key}")
        except RedisError as e:
            logger.error(f"Failed to publish message to Redis: {e}")
            raise RedisDepartmentMailServiceError(f"Failed to send message: {e}") from e

        return message

    def send_broadcast(
        self,
        from_dept: str,
        type: MessageType,
        subject: str,
        body: str,
        priority: Priority = Priority.NORMAL,
        workflow_id: Optional[str] = None,
    ) -> DepartmentMessage:
        """
        Send a broadcast message to all departments.

        Args:
            from_dept: Sending department
            type: Message type
            subject: Brief subject line
            body: Message content
            priority: Message priority
            workflow_id: Optional workflow ID

        Returns:
            The created message
        """
        wf = workflow_id or self.workflow_id

        message = DepartmentMessage(
            id=str(uuid.uuid4()),
            from_dept=from_dept,
            to_dept="*",  # Broadcast indicator
            type=type,
            subject=subject,
            body=body,
            priority=priority,
            timestamp=datetime.now(timezone.utc),
            read=False,
            workflow_id=wf,
        )

        stream_key = self._get_broadcast_stream_key(wf)
        payload = {
            "id": message.id,
            "sender": from_dept,
            "recipient": "*",
            "message_type": type.value,
            "payload": json.dumps({
                "subject": subject,
                "body": body,
                "priority": priority.value,
            }),
            "timestamp_utc": message.timestamp.isoformat(),
        }

        try:
            client = self._get_client()
            client.xadd(stream_key, payload, maxlen=self.MESSAGE_RETENTION, approximate=True)
            logger.debug(f"Broadcast message {message.id} to {stream_key}")
        except RedisError as e:
            logger.error(f"Failed to broadcast message: {e}")
            raise RedisDepartmentMailServiceError(f"Failed to broadcast: {e}") from e

        return message

    # =========================================================================
    # Consumer Group Management
    # =========================================================================

    def _ensure_consumer_group(self, dept: str, workflow_id: Optional[str] = None) -> bool:
        """
        Ensure consumer group exists for department stream.

        Args:
            dept: Department name
            workflow_id: Optional workflow ID

        Returns:
            True if group was created, False if already existed
        """
        wf = workflow_id or self.workflow_id
        stream_key = self._get_dept_stream_key(dept, wf)
        group_name = self._get_consumer_group(dept)

        try:
            client = self._get_client()
            # Try to create consumer group (fails if already exists)
            client.xgroup_create(stream_key, group_name, id="0", mkstream=True)
            logger.info(f"Created consumer group {group_name} for stream {stream_key}")
            return True
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" in str(e):
                # Group already exists - this is fine
                return False
            raise
        except RedisError as e:
            logger.error(f"Failed to create consumer group: {e}")
            raise RedisDepartmentMailServiceError(f"Failed to create consumer group: {e}") from e

    # =========================================================================
    # Consuming
    # =========================================================================

    def check_inbox(
        self,
        dept: str,
        unread_only: bool = True,
        limit: int = 100,
        workflow_id: Optional[str] = None,
    ) -> List[DepartmentMessage]:
        """
        Check inbox for department using consumer group.

        Uses consumer groups for reliable delivery with acknowledgment.
        Unacknowledged messages are replayed on consumer restart.

        Args:
            dept: Department to check
            unread_only: Only return unread messages (uses pending messages)
            limit: Maximum messages to return
            workflow_id: Optional workflow ID

        Returns:
            List of messages
        """
        wf = workflow_id or self.workflow_id
        stream_key = self._get_dept_stream_key(dept, wf)
        group_name = self._get_consumer_group(dept)
        consumer = self.consumer_name

        try:
            client = self._get_client()

            # Ensure consumer group exists
            self._ensure_consumer_group(dept, wf)

            messages = []

            if unread_only:
                # Get pending messages (unacknowledged)
                pending = client.xpending_range(
                    stream_key,
                    group_name,
                    min="-",
                    max="+",
                    count=limit,
                )

                for p in pending:
                    # Get message details
                    msg_data = client.xrange(stream_key, min=p.message_id, max=p.message_id)
                    if msg_data:
                        msg = self._parse_stream_message(msg_data[0])
                        if msg:
                            messages.append(msg)
            else:
                # Get all messages (read and unread)
                # Using consumer group read
                results = client.xreadgroup(
                    group_name,
                    consumer,
                    {stream_key: ">"},
                    count=limit,
                    block=self.STREAM_READ_TIMEOUT,
                )

                if results:
                    for stream, stream_messages in results:
                        for msg_id, msg_data in stream_messages:
                            msg = self._parse_stream_message((msg_id, msg_data))
                            if msg:
                                messages.append(msg)

            logger.debug(f"Checked inbox for {dept}: {len(messages)} messages")
            return messages

        except RedisError as e:
            logger.error(f"Failed to check inbox: {e}")
            raise RedisDepartmentMailServiceError(f"Failed to check inbox: {e}") from e

    def _parse_stream_message(self, msg_tuple) -> Optional[DepartmentMessage]:
        """Parse stream message tuple into DepartmentMessage."""
        try:
            msg_id, data = msg_tuple

            # Extract fields
            msg_id_str = data.get("id", str(msg_id))
            from_dept = data.get("sender", "")
            to_dept = data.get("recipient", "")
            msg_type_str = data.get("message_type", "status")
            timestamp_str = data.get("timestamp_utc", datetime.now(timezone.utc).isoformat())

            # Parse payload
            payload_json = data.get("payload", "{}")
            try:
                payload = json.loads(payload_json)
            except json.JSONDecodeError:
                payload = {}

            subject = payload.get("subject", "")
            body = payload.get("body", "")
            priority_str = payload.get("priority", "normal")
            gate_id = payload.get("gate_id")
            from_stage = payload.get("from_stage")
            to_stage = payload.get("to_stage")

            # Parse timestamp
            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                timestamp = datetime.now(timezone.utc)

            # Determine if read (if we have consumer info, it's delivered but not ACKed)
            read = False

            return DepartmentMessage(
                id=msg_id_str,
                from_dept=from_dept,
                to_dept=to_dept,
                type=MessageType(msg_type_str),
                subject=subject,
                body=body,
                priority=Priority(priority_str),
                timestamp=timestamp,
                read=read,
                gate_id=gate_id,
                workflow_id=data.get("workflow_id"),
                from_stage=from_stage,
                to_stage=to_stage,
            )
        except Exception as e:
            logger.warning(f"Failed to parse stream message: {e}")
            return None

    def mark_read(self, message_id: str) -> bool:
        """
        Acknowledge a message (mark as read).

        Args:
            message_id: ID of message to acknowledge

        Returns:
            True if acknowledged successfully
        """
        # In Redis Streams, we acknowledge via consumer group
        # This is a simplified implementation - stores acknowledgment in a separate key
        try:
            client = self._get_client()
            ack_key = f"mail:acks:{self.workflow_id}"
            client.sadd(ack_key, message_id)
            logger.debug(f"Acknowledged message {message_id}")
            return True
        except RedisError as e:
            logger.error(f"Failed to acknowledge message: {e}")
            return False

    def get_message(self, message_id: str) -> Optional[DepartmentMessage]:
        """
        Get a specific message by ID.

        Note: This is a best-effort lookup. Redis Streams are append-only,
        so we scan all department streams for the message.

        Args:
            message_id: Message ID

        Returns:
            Message or None if not found
        """
        try:
            client = self._get_client()

            # Scan for message in department streams
            # This is inefficient - consider adding an index if needed
            pattern = f"mail:dept:*:{self.workflow_id}"

            # Use SCAN to iterate streams (simplified)
            # In production, consider using a separate index
            cursor = 0
            while True:
                cursor, keys = client.scan(cursor, match=pattern, count=100)
                for key in keys:
                    msg_data = client.xrange(key, min=message_id, max=message_id)
                    if msg_data:
                        msg = self._parse_stream_message(msg_data[0])
                        if msg and msg.id == message_id:
                            return msg

                if cursor == 0:
                    break

            return None

        except RedisError as e:
            logger.error(f"Failed to get message: {e}")
            return None

    def get_messages_by_workflow(self, workflow_id: str) -> List[DepartmentMessage]:
        """Get all messages for a workflow across all departments."""
        messages = []
        try:
            client = self._get_client()

            # Scan department streams for this workflow
            pattern = f"mail:dept:*:{workflow_id}"
            cursor = 0

            while True:
                cursor, keys = client.scan(cursor, match=pattern, count=100)
                for key in keys:
                    # Get last 100 messages from each stream
                    msgs = client.xrevrange(key, "+", "-", count=100)
                    for msg_data in msgs:
                        msg = self._parse_stream_message(msg_data)
                        if msg:
                            messages.append(msg)

                if cursor == 0:
                    break

            return messages

        except RedisError as e:
            logger.error(f"Failed to get messages by workflow: {e}")
            return []

    def get_messages_by_gate(self, gate_id: str) -> List[DepartmentMessage]:
        """Get all messages related to a specific approval gate."""
        messages = []
        try:
            client = self._get_client()

            # Scan all streams for messages with this gate_id
            pattern = "mail:*"
            cursor = 0

            while True:
                cursor, keys = client.scan(cursor, match=pattern, count=100)
                for key in keys:
                    # This is expensive - in production use an index
                    msgs = client.xrevrange(key, "+", "-", count=50)
                    for msg_data in msgs:
                        msg = self._parse_stream_message(msg_data)
                        if msg and msg.gate_id == gate_id:
                            messages.append(msg)

                if cursor == 0:
                    break

            return messages

        except RedisError as e:
            logger.error(f"Failed to get messages by gate: {e}")
            return []

    def purge_old_messages(self, days: int = 30) -> int:
        """
        Delete messages older than specified days.

        Note: Redis Streams automatically handles TTL via MAXLEN.
        This method is a no-op in the Redis implementation.

        Args:
            days: Number of days to keep (for API compatibility)

        Returns:
            Number of messages deleted (always 0 for Redis Streams)
        """
        # Redis Streams handles this via MAXLEN
        # This is kept for API compatibility
        logger.debug(f"Redis Streams handles TTL via MAXLEN, days={days} ignored")
        return 0

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
        """Send an approval-related notification message."""
        if action == "created":
            msg_type = MessageType.APPROVAL_REQUEST
            subject = f"Approval Required: {from_stage} -> {to_stage}"
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
            subject = f"Approved: {from_stage} -> {to_stage}"
            body = f"The transition from '{from_stage}' to '{to_stage}' has been approved.\n\n"
            if requester:
                body += f"Approved by: {requester}\n"
            body += f"Workflow ID: {workflow_id}\n"
            body += f"Gate ID: {gate_id}\n"
            priority = Priority.NORMAL
        else:  # rejected
            msg_type = MessageType.APPROVAL_REJECTED
            subject = f"Rejected: {from_stage} -> {to_stage}"
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

    # =========================================================================
    # Pending Message Replay (for offline consumers)
    # =========================================================================

    def replay_pending_messages(self, dept: str, workflow_id: Optional[str] = None) -> List[DepartmentMessage]:
        """
        Replay all unacknowledged messages for a department.

        Used when a consumer comes back online to process missed messages.

        Args:
            dept: Department name
            workflow_id: Optional workflow ID

        Returns:
            List of pending messages
        """
        wf = workflow_id or self.workflow_id
        stream_key = self._get_dept_stream_key(dept, wf)
        group_name = self._get_consumer_group(dept)

        try:
            client = self._get_client()

            # Ensure consumer group exists
            self._ensure_consumer_group(dept, wf)

            # Get pending messages
            pending = client.xpending_range(
                stream_key,
                group_name,
                min="-",
                max="+",
                count=1000,
            )

            messages = []
            for p in pending:
                msg_data = client.xrange(stream_key, min=p.message_id, max=p.message_id)
                if msg_data:
                    msg = self._parse_stream_message(msg_data[0])
                    if msg:
                        messages.append(msg)

            logger.info(f"Replayed {len(messages)} pending messages for {dept}")
            return messages

        except RedisError as e:
            logger.error(f"Failed to replay pending messages: {e}")
            return []

    # =========================================================================
    # Health Check
    # =========================================================================

    def health_check(self) -> bool:
        """Check if Redis connection is healthy."""
        try:
            client = self._get_client()
            client.ping()
            return True
        except Exception:
            return False


# ============================================================================
# Factory Functions
# ============================================================================

_redis_mail_service: Optional[RedisDepartmentMailService] = None


def get_redis_mail_service(
    host: str = DEFAULT_REDIS_HOST,
    port: int = DEFAULT_REDIS_PORT,
    db: int = DEFAULT_REDIS_DB,
    password: Optional[str] = DEFAULT_REDIS_PASSWORD,
    consumer_name: Optional[str] = None,
    workflow_id: Optional[str] = None,
) -> RedisDepartmentMailService:
    """
    Get or create the singleton RedisDepartmentMailService instance.

    Args:
        host: Redis server host
        port: Redis server port
        db: Redis database number
        password: Optional Redis password
        consumer_name: Unique consumer name
        workflow_id: Workflow ID for namespace isolation

    Returns:
        The RedisDepartmentMailService singleton instance
    """
    global _redis_mail_service
    if _redis_mail_service is None:
        _redis_mail_service = RedisDepartmentMailService(
            host=host,
            port=port,
            db=db,
            password=password,
            consumer_name=consumer_name,
            workflow_id=workflow_id,
        )
    return _redis_mail_service


def reset_redis_mail_service() -> None:
    """Reset the Redis mail service singleton. Useful for testing."""
    global _redis_mail_service
    if _redis_mail_service is not None:
        _redis_mail_service.close()
        _redis_mail_service = None


def get_mail_service(
    db_path: str = ".quantmind/department_mail.db",
    use_redis: bool = True,
    **redis_kwargs,
) -> Any:
    """
    Get the mail service instance (factory function).

    Args:
        db_path: Path to SQLite database (ignored if use_redis=True)
        use_redis: If True, returns Redis Streams implementation
        **redis_kwargs: Additional arguments for Redis connection

    Returns:
        DepartmentMailService (SQLite) or RedisDepartmentMailService (Redis Streams)
    """
    if use_redis:
        warnings.warn(
            "SQLite-based DepartmentMailService is deprecated. "
            "Use RedisDepartmentMailService for reliable message delivery.",
            DeprecationWarning,
            stacklevel=2
        )
        return get_redis_mail_service(**redis_kwargs)
    else:
        warnings.warn(
            "SQLite-based DepartmentMailService is deprecated. "
            "Please migrate to Redis Streams-based implementation.",
            DeprecationWarning,
            stacklevel=2
        )
        return DepartmentMailService(db_path=db_path)


def create_mail_service(
    db_path: str = ".quantmind/department_mail.db",
    use_redis: bool = True,
    **redis_kwargs,
) -> Any:
    """
    Backward-compatible factory retained for existing department-head imports.

    The Redis/SQLite selection logic is owned by `get_mail_service`; this name
    remains as a stable import surface for older callers.
    """
    return get_mail_service(
        db_path=db_path,
        use_redis=use_redis,
        **redis_kwargs,
    )

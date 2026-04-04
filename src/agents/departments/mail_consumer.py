"""
Background Mail Consumer — Polls department mailboxes and processes tasks.

Issues addressed: #11, #12, #13, #14

Architecture:
- Each department gets a background consumer that polls its mailbox
- New mail → creates a to-do item → routes to Kanban board
- Department head processes the task (or delegates to sub-agent)
- Response mail sent back to sender
- Kanban card updated through lifecycle

Flow:
    Mail arrives (via delegate/dispatch)
    → Mail Consumer picks it up (background polling)
    → Creates TodoItem (unread mail → to-do list)
    → Creates/Updates Kanban card (task tracking)
    → Department head processes
    → Response mail sent to sender
    → Kanban card marked complete
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# To-Do Item (Issue #12)
# =============================================================================

class TodoStatus(str, Enum):
    """Status of a to-do item derived from mail."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"        # Waiting on approval
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class TodoItem:
    """A to-do item derived from an unread mail message."""
    id: str
    department: str
    mail_message_id: str
    title: str
    description: str
    priority: str
    status: TodoStatus = TodoStatus.PENDING
    source_dept: str = ""
    message_type: str = ""
    workflow_id: Optional[str] = None
    kanban_card_id: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self):
        now = datetime.now(timezone.utc).isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "department": self.department,
            "mail_message_id": self.mail_message_id,
            "title": self.title,
            "description": self.description,
            "priority": self.priority,
            "status": self.status.value,
            "source_dept": self.source_dept,
            "message_type": self.message_type,
            "workflow_id": self.workflow_id,
            "kanban_card_id": self.kanban_card_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


# =============================================================================
# Kanban Card (Issue #13)
# =============================================================================

class KanbanStatus(str, Enum):
    """Kanban card status — maps to board columns."""
    INBOX = "inbox"
    PROCESSING = "processing"
    REVIEW = "review"
    PENDING_APPROVAL = "pending_approval"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class KanbanCard:
    """A Kanban card representing a task on the department board."""
    id: str
    department: str
    title: str
    description: str
    status: KanbanStatus = KanbanStatus.INBOX
    priority: str = "normal"
    source_dept: str = ""
    mail_message_id: Optional[str] = None
    todo_id: Optional[str] = None
    workflow_id: Optional[str] = None
    strategy_id: Optional[str] = None
    approval_id: Optional[str] = None
    assigned_agent: Optional[str] = None
    result: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self):
        now = datetime.now(timezone.utc).isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "department": self.department,
            "title": self.title,
            "description": self.description,
            "status": self.status.value,
            "priority": self.priority,
            "source_dept": self.source_dept,
            "mail_message_id": self.mail_message_id,
            "todo_id": self.todo_id,
            "workflow_id": self.workflow_id,
            "strategy_id": self.strategy_id,
            "approval_id": self.approval_id,
            "assigned_agent": self.assigned_agent,
            "result": self.result,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


# =============================================================================
# Department Mail Consumer (Issue #11)
# =============================================================================

class DepartmentMailConsumer:
    """
    Background mail consumer for a single department.

    Polls the department mailbox at a configurable interval,
    creates to-do items from unread mail, routes to Kanban board,
    and optionally triggers department head processing.

    Architecture (from issues doc):
        Mail arrives → consumer picks up → TodoItem created →
        KanbanCard created → department head processes →
        response mail sent → Kanban updated
    """

    def __init__(
        self,
        department: str,
        poll_interval: float = 10.0,
        on_new_task: Optional[Callable] = None,
    ):
        self.department = department
        self.poll_interval = poll_interval
        self.on_new_task = on_new_task
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._processed_ids: set = set()

    async def start(self) -> None:
        """Start the background polling loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._poll_loop())
        logger.info(
            f"Mail consumer started for {self.department} "
            f"(interval={self.poll_interval}s)"
        )

    async def stop(self) -> None:
        """Stop the background polling loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info(f"Mail consumer stopped for {self.department}")

    async def _poll_loop(self) -> None:
        """Main polling loop — runs until stopped."""
        while self._running:
            try:
                await self._process_new_mail()
            except Exception as e:
                logger.error(
                    f"Mail consumer error ({self.department}): {e}"
                )
            await asyncio.sleep(self.poll_interval)

    async def _process_new_mail(self) -> None:
        """Check for new unread mail and create to-do + Kanban items."""
        try:
            from src.agents.departments.department_mail import (
                get_mail_service,
            )
            mail_svc = get_mail_service()
            messages = mail_svc.check_inbox(
                dept=self.department, unread_only=True, limit=50,
            )
        except Exception as e:
            logger.debug(f"Mail check failed ({self.department}): {e}")
            return

        mgr = get_task_manager()

        for msg in messages:
            if msg.id in self._processed_ids:
                continue

            # Create to-do item from mail
            todo = mgr.create_todo_from_mail(
                department=self.department,
                message=msg,
            )

            # Create Kanban card from to-do
            card = mgr.create_kanban_from_todo(todo)

            # Mark mail as read (XACK equivalent)
            try:
                from src.agents.departments.department_mail import (
                    get_mail_service,
                )
                stream_or_message_id = getattr(msg, "_stream_msg_id", msg.id)
                get_mail_service().mark_read(
                    stream_or_message_id,
                    dept=self.department,
                )
            except Exception:
                pass

            self._processed_ids.add(msg.id)

            # Notify if callback registered
            if self.on_new_task:
                try:
                    self.on_new_task(todo, card)
                except Exception as e:
                    logger.debug(f"Task callback error: {e}")

            # Publish SSE event for real-time UI
            self._publish_sse(todo, card)

            logger.info(
                f"Mail processed: {msg.id} → todo={todo.id}, "
                f"kanban={card.id} (dept={self.department})"
            )

    def _publish_sse(self, todo: TodoItem, card: KanbanCard) -> None:
        """Publish new task events to SSE thought stream."""
        try:
            from src.api.agent_thought_stream_endpoints import (
                get_thought_publisher,
            )
            get_thought_publisher().publish(
                department=self.department,
                thought=json.dumps({
                    "type": "new_task",
                    "todo": todo.to_dict(),
                    "kanban": card.to_dict(),
                }),
                thought_type="action",
            )
        except Exception:
            pass


# =============================================================================
# Task Manager — Central registry for To-Do + Kanban (Issues #12, #13)
# =============================================================================

class TaskManager:
    """
    Central registry for department to-do lists and Kanban boards.

    Manages:
    - Per-department to-do lists (derived from unread mail)
    - Per-department Kanban boards (task lifecycle tracking)
    - Mail consumer instances per department
    """

    DEPARTMENTS = [
        "research", "development", "risk", "trading", "portfolio",
    ]

    def __init__(self):
        self._todos: Dict[str, Dict[str, TodoItem]] = {
            dept: {} for dept in self.DEPARTMENTS
        }
        self._kanban: Dict[str, Dict[str, KanbanCard]] = {
            dept: {} for dept in self.DEPARTMENTS
        }
        self._consumers: Dict[str, DepartmentMailConsumer] = {}

    # ── To-Do Management (Issue #12) ─────────────────────────────────

    def create_todo_from_mail(self, department: str, message) -> TodoItem:
        """Create a to-do item from an unread mail message."""
        todo = TodoItem(
            id=f"todo_{uuid.uuid4().hex[:10]}",
            department=department,
            mail_message_id=message.id,
            title=message.subject or f"Task from {message.from_dept}",
            description=message.body or "",
            priority=message.priority.value if hasattr(message.priority, 'value') else str(message.priority),
            source_dept=message.from_dept,
            message_type=message.type.value if hasattr(message.type, 'value') else str(message.type),
            workflow_id=getattr(message, 'workflow_id', None),
        )
        self._todos.setdefault(department, {})[todo.id] = todo
        return todo

    def get_todos(
        self,
        department: str,
        status: Optional[TodoStatus] = None,
    ) -> List[TodoItem]:
        """Get to-do list for a department, optionally filtered by status."""
        dept_todos = self._todos.get(department, {})
        items = list(dept_todos.values())
        if status:
            items = [t for t in items if t.status == status]
        return sorted(items, key=lambda t: t.created_at, reverse=True)

    def update_todo_status(
        self, todo_id: str, status: TodoStatus,
    ) -> Optional[TodoItem]:
        """Update a to-do item's status."""
        for dept_todos in self._todos.values():
            if todo_id in dept_todos:
                todo = dept_todos[todo_id]
                todo.status = status
                todo.updated_at = datetime.now(timezone.utc).isoformat()
                return todo
        return None

    def get_todo_counts(self) -> Dict[str, Dict[str, int]]:
        """Get to-do counts per department per status."""
        counts = {}
        for dept, todos in self._todos.items():
            dept_counts: Dict[str, int] = {}
            for todo in todos.values():
                dept_counts[todo.status.value] = (
                    dept_counts.get(todo.status.value, 0) + 1
                )
            counts[dept] = dept_counts
        return counts

    # ── Kanban Management (Issue #13) ────────────────────────────────

    def create_kanban_from_todo(self, todo: TodoItem) -> KanbanCard:
        """Create a Kanban card from a to-do item."""
        card = KanbanCard(
            id=f"kb_{uuid.uuid4().hex[:10]}",
            department=todo.department,
            title=todo.title,
            description=todo.description,
            priority=todo.priority,
            source_dept=todo.source_dept,
            mail_message_id=todo.mail_message_id,
            todo_id=todo.id,
            workflow_id=todo.workflow_id,
        )
        todo.kanban_card_id = card.id
        self._kanban.setdefault(todo.department, {})[card.id] = card
        return card

    def create_kanban_card(
        self,
        department: str,
        title: str,
        description: str = "",
        priority: str = "normal",
        workflow_id: Optional[str] = None,
        strategy_id: Optional[str] = None,
        status: KanbanStatus = KanbanStatus.INBOX,
    ) -> KanbanCard:
        """Create a standalone Kanban card (not from mail)."""
        card = KanbanCard(
            id=f"kb_{uuid.uuid4().hex[:10]}",
            department=department,
            title=title,
            description=description,
            priority=priority,
            workflow_id=workflow_id,
            strategy_id=strategy_id,
            status=status,
        )
        self._kanban.setdefault(department, {})[card.id] = card
        return card

    def update_kanban_status(
        self, card_id: str, status: KanbanStatus,
        result: Optional[str] = None,
    ) -> Optional[KanbanCard]:
        """Update a Kanban card's status."""
        for dept_cards in self._kanban.values():
            if card_id in dept_cards:
                card = dept_cards[card_id]
                card.status = status
                card.updated_at = datetime.now(timezone.utc).isoformat()
                if result:
                    card.result = result
                return card
        return None

    def get_kanban_cards(
        self,
        department: str,
        status: Optional[KanbanStatus] = None,
    ) -> List[KanbanCard]:
        """Get Kanban cards for a department, optionally filtered."""
        dept_cards = self._kanban.get(department, {})
        cards = list(dept_cards.values())
        if status:
            cards = [c for c in cards if c.status == status]
        return sorted(cards, key=lambda c: c.created_at, reverse=True)

    def get_kanban_summary(self) -> Dict[str, Dict[str, int]]:
        """Get Kanban summary counts per department per status."""
        summary = {}
        for dept, cards in self._kanban.items():
            dept_counts: Dict[str, int] = {}
            for card in cards.values():
                dept_counts[card.status.value] = (
                    dept_counts.get(card.status.value, 0) + 1
                )
            summary[dept] = dept_counts
        return summary

    # ── HITL → Kanban (Issue #15) ────────────────────────────────────

    def create_approval_kanban(
        self,
        department: str,
        approval_id: str,
        title: str,
        workflow_id: Optional[str] = None,
    ) -> KanbanCard:
        """Create a Kanban card for an approval request (Issue #15)."""
        card = self.create_kanban_card(
            department=department,
            title=f"Approval: {title}",
            description=f"Waiting for human approval. ID: {approval_id}",
            priority="high",
            workflow_id=workflow_id,
            status=KanbanStatus.PENDING_APPROVAL,
        )
        card.approval_id = approval_id
        return card

    def resolve_approval_kanban(
        self, approval_id: str, approved: bool,
    ) -> Optional[KanbanCard]:
        """Update Kanban card when approval is resolved (Issue #15)."""
        for dept_cards in self._kanban.values():
            for card in dept_cards.values():
                if card.approval_id == approval_id:
                    if approved:
                        card.status = KanbanStatus.PROCESSING
                        card.result = "Approved — resuming"
                    else:
                        card.status = KanbanStatus.FAILED
                        card.result = "Rejected"
                    card.updated_at = (
                        datetime.now(timezone.utc).isoformat()
                    )
                    return card
        return None

    # ── Response Mail (Issue #14) ────────────────────────────────────

    def send_response_mail(
        self,
        department: str,
        original_message_id: str,
        result: str,
        next_steps: Optional[str] = None,
    ) -> None:
        """Send response mail back to the original sender (Issue #14)."""
        try:
            from src.agents.departments.department_mail import (
                get_mail_service, MessageType, Priority,
            )
            mail_svc = get_mail_service()
            original = mail_svc.get_message(original_message_id)
            if not original:
                return

            body = f"Task Result:\n{result}"
            if next_steps:
                body += f"\n\nNext Steps:\n{next_steps}"

            mail_svc.send(
                from_dept=department,
                to_dept=original.from_dept,
                type=MessageType.RESULT,
                subject=f"Re: {original.subject}",
                body=body,
                priority=original.priority,
                workflow_id=original.workflow_id,
            )
        except Exception as e:
            logger.debug(f"Response mail failed: {e}")

    # ── Mail Consumer Management (Issue #11) ─────────────────────────

    async def start_consumers(self, poll_interval: float = 10.0) -> None:
        """Start background mail consumers for all departments."""
        for dept in self.DEPARTMENTS:
            if dept not in self._consumers:
                consumer = DepartmentMailConsumer(
                    department=dept,
                    poll_interval=poll_interval,
                )
                self._consumers[dept] = consumer
                await consumer.start()
        logger.info(
            f"Started mail consumers for {len(self._consumers)} departments"
        )

    async def stop_consumers(self) -> None:
        """Stop all background mail consumers."""
        for consumer in self._consumers.values():
            await consumer.stop()
        self._consumers.clear()


# =============================================================================
# Singleton
# =============================================================================

_task_manager: Optional[TaskManager] = None


def get_task_manager() -> TaskManager:
    """Return the singleton TaskManager instance."""
    global _task_manager
    if _task_manager is None:
        _task_manager = TaskManager()
    return _task_manager

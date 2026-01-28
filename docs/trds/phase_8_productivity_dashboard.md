# Phase 8: Productivity Dashboard

> **Focus:** Transform Kanban from agent orchestration to personal productivity hub
> **Dependencies:** Phase 6 complete (DeepAgents), Phase 7 complete (Voice)
> **Deliverable:** Personal to-do lists, calendar integration, morning agenda, department status

---

## Overview

This phase transforms the existing Kanban board into a **personal productivity dashboard**. The major shift:

| Before (D.O.E. Kanban) | After (Productivity Dashboard) |
|------------------------|--------------------------------|
| Cards for agent orchestration | Cards for personal tasks |
| Department-assigned work | User's to-do list |
| Complex dependency tracking | Simple due dates and priorities |
| Cowork mode creates cards | ROI creates cards from conversation |
| Worker processing | Manual completion |

**What We Keep:**
- Card model (title, description, status, due_date)
- HITL system (for department approvals)
- SSE events (real-time updates)
- Storage layer

**What We Add:**
- Personal task management
- Calendar integration (Google Calendar MCP)
- Meeting → action item extraction
- Morning agenda greeting
- Department status aggregation

---

## Reference Links

| Resource | URL |
|----------|-----|
| Google Calendar MCP | https://github.com/modelcontextprotocol/servers/tree/main/src/google-calendar |
| Notion-like To-do | (Design inspiration) |

---

## Sub-Phases

### 8.1 Transform Kanban to Personal Tasks

**Goal:** Simplify card model for personal productivity

**Tasks:**

1. Create simplified task model:
```python
# /backend/productivity/models.py

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional, List
from enum import Enum
from uuid import UUID, uuid4

class TaskPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

class TaskStatus(str, Enum):
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    CANCELLED = "cancelled"

@dataclass
class Task:
    """Personal task (simplified from Kanban card)."""
    
    id: UUID = field(default_factory=uuid4)
    title: str = ""
    description: Optional[str] = None
    status: TaskStatus = TaskStatus.TODO
    priority: TaskPriority = TaskPriority.MEDIUM
    
    # Timing
    due_date: Optional[date] = None
    due_time: Optional[str] = None  # HH:MM format
    reminder: Optional[datetime] = None
    
    # Organization
    tags: List[str] = field(default_factory=list)
    project: Optional[str] = None
    
    # Source tracking
    created_from: Optional[str] = None  # "chat", "meeting", "manual"
    meeting_id: Optional[str] = None  # If extracted from meeting
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    
    def mark_done(self):
        """Mark task as complete."""
        self.status = TaskStatus.DONE
        self.completed_at = datetime.now()
        self.updated_at = datetime.now()
    
    def to_dict(self) -> dict:
        """Convert to dictionary for API/storage."""
        return {
            "id": str(self.id),
            "title": self.title,
            "description": self.description,
            "status": self.status.value,
            "priority": self.priority.value,
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "due_time": self.due_time,
            "tags": self.tags,
            "project": self.project,
            "created_from": self.created_from,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


@dataclass
class DepartmentStatus:
    """Status snapshot of a department."""
    
    department_id: str
    department_name: str
    state: str  # "idle", "working", "waiting_hitl", "error"
    current_task: Optional[str] = None
    progress: int = 0  # 0-100
    eta_seconds: Optional[int] = None
    last_activity: Optional[datetime] = None
    
    # Queue
    pending_tasks: int = 0
    completed_today: int = 0
    
    @property
    def eta_formatted(self) -> Optional[str]:
        """Format ETA as human-readable string."""
        if not self.eta_seconds:
            return None
        
        if self.eta_seconds < 60:
            return f"{self.eta_seconds}s"
        elif self.eta_seconds < 3600:
            return f"{self.eta_seconds // 60}m"
        else:
            hours = self.eta_seconds // 3600
            mins = (self.eta_seconds % 3600) // 60
            return f"{hours}h {mins}m"
```

2. Create productivity service:
```python
# /backend/productivity/service.py

from typing import List, Optional
from datetime import date, datetime
from uuid import UUID
from backend.productivity.models import Task, TaskStatus, TaskPriority, DepartmentStatus

class ProductivityService:
    """Manages personal tasks and productivity features."""
    
    def __init__(self, db_session):
        self.db = db_session
    
    # ========== TASKS ==========
    
    async def create_task(
        self,
        title: str,
        description: Optional[str] = None,
        priority: TaskPriority = TaskPriority.MEDIUM,
        due_date: Optional[date] = None,
        tags: Optional[List[str]] = None,
        created_from: str = "manual",
    ) -> Task:
        """Create a new task.
        
        Args:
            title: Task title
            description: Optional details
            priority: Task priority
            due_date: When task is due
            tags: Organizational tags
            created_from: Source ("chat", "meeting", "manual")
        
        Returns:
            Created Task
        """
        task = Task(
            title=title,
            description=description,
            priority=priority,
            due_date=due_date,
            tags=tags or [],
            created_from=created_from,
        )
        
        await self._save_task(task)
        return task
    
    async def list_tasks(
        self,
        status: Optional[TaskStatus] = None,
        due_date: Optional[date] = None,
        project: Optional[str] = None,
    ) -> List[Task]:
        """List tasks with optional filtering.
        
        Args:
            status: Filter by status (None = all non-cancelled)
            due_date: Filter by due date
            project: Filter by project
        
        Returns:
            List of matching tasks
        """
        # Query database (pseudo-code)
        query = self.db.query(Task)
        
        if status:
            query = query.filter(Task.status == status)
        else:
            query = query.filter(Task.status != TaskStatus.CANCELLED)
        
        if due_date:
            query = query.filter(Task.due_date == due_date)
        
        if project:
            query = query.filter(Task.project == project)
        
        return await query.order_by(Task.priority.desc(), Task.due_date).all()
    
    async def get_today_tasks(self) -> List[Task]:
        """Get tasks due today or overdue."""
        today = date.today()
        
        tasks = await self.list_tasks()
        
        return [
            t for t in tasks
            if t.status not in [TaskStatus.DONE, TaskStatus.CANCELLED]
            and (t.due_date is None or t.due_date <= today)
        ]
    
    async def update_task(self, task_id: UUID, **updates) -> Task:
        """Update a task."""
        task = await self._get_task(task_id)
        
        for key, value in updates.items():
            if hasattr(task, key):
                setattr(task, key, value)
        
        task.updated_at = datetime.now()
        await self._save_task(task)
        
        return task
    
    async def complete_task(self, task_id: UUID) -> Task:
        """Mark task as complete."""
        task = await self._get_task(task_id)
        task.mark_done()
        await self._save_task(task)
        return task
    
    # ========== QUICK ADD ==========
    
    async def quick_add_from_text(self, text: str) -> Task:
        """Parse natural language and create task.
        
        Examples:
            "Review docs tomorrow" -> Task(title="Review docs", due_date=tomorrow)
            "Call John #work" -> Task(title="Call John", tags=["work"])
            "!!! Fix critical bug" -> Task(title="Fix critical bug", priority=URGENT)
        
        Args:
            text: Natural language task description
        
        Returns:
            Created Task
        """
        import re
        from dateparser import parse as parse_date
        
        # Extract priority
        priority = TaskPriority.MEDIUM
        if text.startswith("!!!"):
            priority = TaskPriority.URGENT
            text = text[3:].strip()
        elif text.startswith("!!"):
            priority = TaskPriority.HIGH
            text = text[2:].strip()
        elif text.startswith("!"):
            priority = TaskPriority.LOW
            text = text[1:].strip()
        
        # Extract tags (#tag)
        tags = re.findall(r"#(\w+)", text)
        text = re.sub(r"#\w+", "", text).strip()
        
        # Extract date (basic parsing)
        due_date = None
        date_words = ["tomorrow", "today", "monday", "tuesday", "wednesday", 
                      "thursday", "friday", "saturday", "sunday", "next week"]
        
        for word in date_words:
            if word in text.lower():
                parsed = parse_date(word)
                if parsed:
                    due_date = parsed.date()
                    text = re.sub(rf"\b{word}\b", "", text, flags=re.IGNORECASE).strip()
                    break
        
        # Create task
        return await self.create_task(
            title=text,
            priority=priority,
            due_date=due_date,
            tags=tags,
            created_from="chat",
        )
    
    # ========== DEPARTMENT STATUS ==========
    
    async def get_all_department_status(self) -> List[DepartmentStatus]:
        """Get status of all departments."""
        from backend.departments.status import get_all_department_status
        return await get_all_department_status()
    
    async def get_department_status(self, department_name: str) -> DepartmentStatus:
        """Get status of a specific department."""
        from backend.departments.status import get_status
        return await get_status(department_name)
    
    # ========== INTERNAL ==========
    
    async def _get_task(self, task_id: UUID) -> Task:
        """Get task by ID."""
        # Database query
        pass
    
    async def _save_task(self, task: Task):
        """Save task to database."""
        # Database save
        pass
```

**Success Criteria:**
- [ ] Simplified Task model
- [ ] CRUD operations working
- [ ] Quick add with natural language
- [ ] Department status aggregation

---

### 8.2 Calendar Integration

**Goal:** Integrate Google Calendar via MCP for meeting management

**Tasks:**

1. Create calendar service:
```python
# /backend/productivity/calendar.py

from datetime import datetime, date, timedelta
from typing import List, Optional
from dataclasses import dataclass

@dataclass
class CalendarEvent:
    """Calendar event representation."""
    
    id: str
    title: str
    start: datetime
    end: datetime
    description: Optional[str] = None
    location: Optional[str] = None
    attendees: List[str] = None
    is_all_day: bool = False
    
    # Meeting with ROI flag
    is_roi_meeting: bool = False
    roi_meeting_topic: Optional[str] = None


class CalendarService:
    """Calendar integration via Google Calendar MCP."""
    
    def __init__(self):
        self._mcp_client = None
    
    async def _get_mcp_client(self):
        """Get or create MCP client for Google Calendar."""
        if self._mcp_client is None:
            from langchain_mcp_adapters import MCPClient
            
            self._mcp_client = MCPClient(
                server="google-calendar",
                # Config from environment
            )
        
        return self._mcp_client
    
    async def get_today_events(self) -> List[CalendarEvent]:
        """Get today's calendar events."""
        today = date.today()
        return await self.get_events(
            start=datetime.combine(today, datetime.min.time()),
            end=datetime.combine(today, datetime.max.time()),
        )
    
    async def get_week_events(self) -> List[CalendarEvent]:
        """Get this week's calendar events."""
        today = date.today()
        week_end = today + timedelta(days=7)
        
        return await self.get_events(
            start=datetime.combine(today, datetime.min.time()),
            end=datetime.combine(week_end, datetime.max.time()),
        )
    
    async def get_events(
        self,
        start: datetime,
        end: datetime,
    ) -> List[CalendarEvent]:
        """Get events in date range.
        
        Uses Google Calendar MCP server.
        """
        client = await self._get_mcp_client()
        
        # Call MCP tool
        result = await client.call_tool(
            "list_events",
            {
                "time_min": start.isoformat(),
                "time_max": end.isoformat(),
            }
        )
        
        # Parse response
        events = []
        for event_data in result.get("events", []):
            events.append(CalendarEvent(
                id=event_data["id"],
                title=event_data["summary"],
                start=datetime.fromisoformat(event_data["start"]["dateTime"]),
                end=datetime.fromisoformat(event_data["end"]["dateTime"]),
                description=event_data.get("description"),
                location=event_data.get("location"),
                attendees=event_data.get("attendees", []),
            ))
        
        return events
    
    async def create_event(
        self,
        title: str,
        start: datetime,
        end: datetime,
        description: Optional[str] = None,
        is_roi_meeting: bool = False,
    ) -> CalendarEvent:
        """Create a calendar event.
        
        If is_roi_meeting=True, also schedules a ROI voice check-in.
        """
        client = await self._get_mcp_client()
        
        # Create via MCP
        result = await client.call_tool(
            "create_event",
            {
                "summary": title,
                "start": {"dateTime": start.isoformat()},
                "end": {"dateTime": end.isoformat()},
                "description": description,
            }
        )
        
        event = CalendarEvent(
            id=result["id"],
            title=title,
            start=start,
            end=end,
            description=description,
            is_roi_meeting=is_roi_meeting,
        )
        
        # If ROI meeting, schedule voice trigger
        if is_roi_meeting:
            await self._schedule_roi_meeting_trigger(event)
        
        return event
    
    async def _schedule_roi_meeting_trigger(self, event: CalendarEvent):
        """Schedule ROI to start meeting mode at event time."""
        from backend.agents.agent_tools.scheduler import TaskScheduler
        
        scheduler = TaskScheduler()
        
        # Calculate cron for event start time
        # Example: "30 14 19 1 *" for Jan 19 at 14:30
        cron = f"{event.start.minute} {event.start.hour} {event.start.day} {event.start.month} *"
        
        await scheduler.create_task(
            name=f"ROI Meeting: {event.title}",
            prompt=f"Time for your scheduled meeting: {event.title}. Switch to meeting mode and start discussing: {event.description or 'the agenda'}",
            schedule=cron,
        )


# Singleton
_calendar_service: Optional[CalendarService] = None

def get_calendar_service() -> CalendarService:
    global _calendar_service
    if _calendar_service is None:
        _calendar_service = CalendarService()
    return _calendar_service
```

2. Add calendar API endpoints:
```python
# /backend/api/productivity.py

from fastapi import APIRouter
from backend.productivity.calendar import get_calendar_service
from backend.productivity.service import ProductivityService

router = APIRouter(prefix="/productivity", tags=["productivity"])

@router.get("/calendar/today")
async def get_today_events():
    """Get today's calendar events."""
    service = get_calendar_service()
    events = await service.get_today_events()
    return {"events": [e.__dict__ for e in events]}

@router.get("/calendar/week")
async def get_week_events():
    """Get this week's calendar events."""
    service = get_calendar_service()
    events = await service.get_week_events()
    return {"events": [e.__dict__ for e in events]}

@router.post("/calendar/meeting")
async def schedule_roi_meeting(
    title: str,
    start: datetime,
    duration_minutes: int = 30,
    topic: Optional[str] = None,
):
    """Schedule a meeting with ROI.
    
    ROI will automatically switch to meeting mode at the scheduled time.
    """
    service = get_calendar_service()
    
    end = start + timedelta(minutes=duration_minutes)
    
    event = await service.create_event(
        title=f"Meeting with ROI: {title}",
        start=start,
        end=end,
        description=topic,
        is_roi_meeting=True,
    )
    
    return {"event": event.__dict__, "message": "Meeting scheduled. ROI will remind you."}
```

**Success Criteria:**
- [ ] Google Calendar MCP integrated
- [ ] Today's events retrievable
- [ ] Week view working
- [ ] ROI meetings schedulable with triggers

---

### 8.3 Morning Agenda Greeting

**Goal:** ROI greets user with daily agenda on startup

**Tasks:**

1. Create morning greeting service:
```python
# /backend/productivity/morning_greeting.py

from datetime import datetime, date
from typing import Dict, List
from backend.productivity.service import ProductivityService
from backend.productivity.calendar import get_calendar_service
from backend.productivity.models import Task, DepartmentStatus

class MorningGreetingService:
    """Generates morning greeting with agenda."""
    
    def __init__(self):
        self.productivity = ProductivityService(db_session=None)  # Inject properly
        self.calendar = get_calendar_service()
    
    async def generate_greeting(self) -> Dict:
        """Generate complete morning greeting data.
        
        Returns dict with:
        - greeting_message: Natural greeting
        - tasks: Today's tasks
        - events: Today's events
        - departments: Department status summary
        """
        
        today = date.today()
        hour = datetime.now().hour
        
        # Time-based greeting
        if hour < 12:
            time_greeting = "Good morning"
        elif hour < 17:
            time_greeting = "Good afternoon"
        else:
            time_greeting = "Good evening"
        
        # Gather data
        tasks = await self.productivity.get_today_tasks()
        events = await self.calendar.get_today_events()
        departments = await self.productivity.get_all_department_status()
        
        # Build greeting message
        message = self._build_greeting_message(
            time_greeting, tasks, events, departments
        )
        
        return {
            "greeting_message": message,
            "tasks": [t.to_dict() for t in tasks],
            "events": [e.__dict__ for e in events],
            "departments": [d.__dict__ for d in departments],
            "date": today.isoformat(),
        }
    
    def _build_greeting_message(
        self,
        time_greeting: str,
        tasks: List[Task],
        events: List,
        departments: List[DepartmentStatus],
    ) -> str:
        """Build natural language greeting."""
        
        lines = [f"{time_greeting}! Here's your agenda for today:"]
        lines.append("")
        
        # Events section
        if events:
            lines.append("📅 **Today's Meetings:**")
            for event in events[:3]:  # Top 3
                time_str = event.start.strftime("%I:%M %p")
                lines.append(f"  • {time_str} - {event.title}")
            if len(events) > 3:
                lines.append(f"  ...and {len(events) - 3} more")
            lines.append("")
        
        # Tasks section
        if tasks:
            high_priority = [t for t in tasks if t.priority.value in ["high", "urgent"]]
            other = [t for t in tasks if t.priority.value not in ["high", "urgent"]]
            
            lines.append("📋 **Today's Tasks:**")
            
            if high_priority:
                lines.append("  🔴 High Priority:")
                for task in high_priority[:3]:
                    lines.append(f"    • {task.title}")
            
            if other:
                lines.append("  Other:")
                for task in other[:3]:
                    lines.append(f"    • {task.title}")
            
            lines.append("")
        
        # Department status
        active_depts = [d for d in departments if d.state == "working"]
        if active_depts:
            lines.append("🏢 **Department Activity:**")
            for dept in active_depts:
                eta = f" (ETA: {dept.eta_formatted})" if dept.eta_formatted else ""
                lines.append(f"  • {dept.department_name}: {dept.current_task}{eta}")
            lines.append("")
        
        # Closing
        remaining_tasks = len([t for t in tasks if t.status.value == "todo"])
        if remaining_tasks > 0:
            lines.append(f"You have {remaining_tasks} tasks to complete today.")
        
        lines.append("")
        lines.append("What would you like to focus on?")
        
        return "\n".join(lines)
    
    async def check_and_greet(self) -> str:
        """Check if greeting needed and return message.
        
        Only greets once per day (tracks in preferences).
        """
        from backend.intents.roi_preferences import ROIPreferenceService
        
        prefs = ROIPreferenceService()
        
        # Check last greeting date
        last_greeting = await prefs.get_preferences(
            user_id="default",
            category="system",
        )
        
        last_date = None
        for pref in last_greeting:
            if pref.get("key") == "last_morning_greeting":
                last_date = pref.get("value")
                break
        
        today = date.today().isoformat()
        
        if last_date == today:
            return None  # Already greeted today
        
        # Record greeting
        await prefs.record_delegation_preference(
            user_id="default",
            task_type="last_morning_greeting",
            preferred_department=today,  # Store date here
        )
        
        # Generate and return greeting
        data = await self.generate_greeting()
        return data["greeting_message"]


# API endpoint
@router.get("/productivity/morning")
async def get_morning_greeting():
    """Get morning greeting with agenda.
    
    Returns greeting message and structured data.
    """
    service = MorningGreetingService()
    data = await service.generate_greeting()
    return data

@router.post("/productivity/morning/trigger")
async def trigger_morning_greeting():
    """Trigger morning greeting check.
    
    Call this on app startup to potentially greet user.
    """
    service = MorningGreetingService()
    message = await service.check_and_greet()
    
    if message:
        return {"should_greet": True, "message": message}
    return {"should_greet": False}
```

**Success Criteria:**
- [ ] Greeting generated with tasks/events/departments
- [ ] Time-based greeting (morning/afternoon/evening)
- [ ] Only greets once per day
- [ ] Triggers on app startup

---

### 8.4 Meeting Action Item Extraction

**Goal:** Extract tasks from meeting minutes automatically

**Tasks:**

1. Create action item extractor:
```python
# /backend/productivity/action_items.py

from typing import List
from backend.productivity.models import Task, TaskPriority
from backend.productivity.service import ProductivityService

class ActionItemExtractor:
    """Extracts action items from meeting content."""
    
    def __init__(self, productivity_service: ProductivityService):
        self.productivity = productivity_service
    
    async def extract_from_meeting(
        self,
        meeting_id: str,
        transcript: str,
        minutes: str,
    ) -> List[Task]:
        """Extract action items from meeting content.
        
        Uses LLM to identify action items, then creates tasks.
        
        Args:
            meeting_id: ID of the meeting
            transcript: Full meeting transcript
            minutes: Generated meeting minutes
        
        Returns:
            List of created tasks
        """
        from backend.agents.roi_agent import create_roi_agent
        
        agent = create_roi_agent()
        
        prompt = f"""Analyze this meeting content and extract action items.

MEETING MINUTES:
{minutes}

TRANSCRIPT (for context):
{transcript[:2000]}...

For each action item, identify:
1. Task description (clear, actionable)
2. Owner (if mentioned, else "Unassigned")
3. Due date (if mentioned, else null)
4. Priority (based on urgency in discussion)

Return JSON array:
[
    {{"task": "...", "owner": "...", "due": "YYYY-MM-DD or null", "priority": "low|medium|high|urgent"}}
]
"""
        
        result = agent.invoke({
            "messages": [{"role": "user", "content": prompt}]
        })
        
        # Parse response
        import json
        response_text = result["messages"][-1].content
        
        # Extract JSON from response
        try:
            # Find JSON array in response
            start = response_text.find("[")
            end = response_text.rfind("]") + 1
            json_str = response_text[start:end]
            items = json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            return []
        
        # Create tasks
        tasks = []
        for item in items:
            task = await self.productivity.create_task(
                title=item["task"],
                description=f"From meeting: {meeting_id}\nOwner: {item.get('owner', 'Unassigned')}",
                priority=TaskPriority(item.get("priority", "medium")),
                due_date=item.get("due"),
                created_from="meeting",
            )
            task.meeting_id = meeting_id
            tasks.append(task)
        
        return tasks
    
    async def process_new_meeting(self, meeting_id: str):
        """Process a newly completed meeting.
        
        Called after meeting mode ends.
        """
        from pathlib import Path
        
        # Load meeting files
        base_path = Path(f"/home/mubarkahimself/Desktop/ROI/backend/departments/roi-orchestrator/memories/meetings")
        
        transcript_path = base_path / f"{meeting_id}_transcript.md"
        minutes_path = base_path / f"{meeting_id}_minutes.md"
        
        if not transcript_path.exists() or not minutes_path.exists():
            return []
        
        transcript = transcript_path.read_text()
        minutes = minutes_path.read_text()
        
        # Extract action items
        tasks = await self.extract_from_meeting(meeting_id, transcript, minutes)
        
        return tasks
```

2. Wire into voice mode:
```python
# /backend/voice/mode_controller.py (update)

async def _save_meeting(self):
    """Save meeting and extract action items."""
    # ... existing save logic ...
    
    # Extract action items
    from backend.productivity.action_items import ActionItemExtractor
    from backend.productivity.service import ProductivityService
    
    extractor = ActionItemExtractor(ProductivityService(db_session=None))
    tasks = await extractor.process_new_meeting(meeting_id)
    
    if tasks:
        # Notify user
        task_summary = "\n".join([f"• {t.title}" for t in tasks])
        await send_notification(
            title="Action Items Extracted",
            message=f"Created {len(tasks)} tasks from your meeting:\n{task_summary}",
        )
```

**Success Criteria:**
- [ ] Action items extracted from minutes
- [ ] Tasks created automatically
- [ ] Meeting ID linked to tasks
- [ ] User notified of extracted items

---

### 8.5 Productivity API Endpoints

**Goal:** Complete API for productivity features

**Tasks:**

1. Create comprehensive API:
```python
# /backend/api/productivity.py (complete)

from fastapi import APIRouter, HTTPException
from typing import Optional, List
from datetime import date
from uuid import UUID
from backend.productivity.service import ProductivityService
from backend.productivity.models import TaskStatus, TaskPriority

router = APIRouter(prefix="/productivity", tags=["productivity"])

# ========== TASKS ==========

@router.get("/tasks")
async def list_tasks(
    status: Optional[TaskStatus] = None,
    due_date: Optional[date] = None,
    project: Optional[str] = None,
):
    """List tasks with optional filters."""
    service = ProductivityService(db_session=None)  # Inject properly
    tasks = await service.list_tasks(status=status, due_date=due_date, project=project)
    return {"tasks": [t.to_dict() for t in tasks]}

@router.get("/tasks/today")
async def get_today_tasks():
    """Get today's tasks (due today or overdue)."""
    service = ProductivityService(db_session=None)
    tasks = await service.get_today_tasks()
    return {"tasks": [t.to_dict() for t in tasks]}

@router.post("/tasks")
async def create_task(
    title: str,
    description: Optional[str] = None,
    priority: TaskPriority = TaskPriority.MEDIUM,
    due_date: Optional[date] = None,
    tags: Optional[List[str]] = None,
):
    """Create a new task."""
    service = ProductivityService(db_session=None)
    task = await service.create_task(
        title=title,
        description=description,
        priority=priority,
        due_date=due_date,
        tags=tags,
        created_from="api",
    )
    return task.to_dict()

@router.post("/tasks/quick")
async def quick_add_task(text: str):
    """Quick add task from natural language.
    
    Examples:
        "Review docs tomorrow"
        "!!! Fix critical bug"
        "Call John #work"
    """
    service = ProductivityService(db_session=None)
    task = await service.quick_add_from_text(text)
    return task.to_dict()

@router.patch("/tasks/{task_id}")
async def update_task(
    task_id: UUID,
    title: Optional[str] = None,
    status: Optional[TaskStatus] = None,
    priority: Optional[TaskPriority] = None,
    due_date: Optional[date] = None,
):
    """Update a task."""
    service = ProductivityService(db_session=None)
    
    updates = {}
    if title: updates["title"] = title
    if status: updates["status"] = status
    if priority: updates["priority"] = priority
    if due_date: updates["due_date"] = due_date
    
    task = await service.update_task(task_id, **updates)
    return task.to_dict()

@router.post("/tasks/{task_id}/complete")
async def complete_task(task_id: UUID):
    """Mark task as complete."""
    service = ProductivityService(db_session=None)
    task = await service.complete_task(task_id)
    return task.to_dict()

# ========== DEPARTMENTS ==========

@router.get("/departments")
async def get_all_department_status():
    """Get status of all departments."""
    service = ProductivityService(db_session=None)
    statuses = await service.get_all_department_status()
    return {"departments": [s.__dict__ for s in statuses]}

@router.get("/departments/{name}/status")
async def get_department_status(name: str):
    """Get status of a specific department."""
    service = ProductivityService(db_session=None)
    status = await service.get_department_status(name)
    return status.__dict__

# ========== AGENDA ==========

@router.get("/agenda")
async def get_daily_agenda():
    """Get complete daily agenda (tasks + events + departments)."""
    from backend.productivity.morning_greeting import MorningGreetingService
    
    service = MorningGreetingService()
    return await service.generate_greeting()
```

**Success Criteria:**
- [ ] All CRUD endpoints for tasks
- [ ] Quick add endpoint
- [ ] Department status endpoints
- [ ] Agenda endpoint

---

## Phase 8 Overall Success Criteria

- [ ] Tasks model simplified from Kanban cards
- [ ] Personal task CRUD working
- [ ] Quick add with natural language parsing
- [ ] Google Calendar integration
- [ ] ROI meetings schedulable with triggers
- [ ] Morning greeting on startup
- [ ] Action items extracted from meetings
- [ ] Department status aggregation
- [ ] Complete REST API

## Phase 8 Deliverable

**Demo:**
1. Start app → Morning greeting appears
2. "Add task: Review budget tomorrow" → Task created
3. Check calendar → Today's events shown
4. "Schedule meeting with ROI tomorrow at 2pm about trading" → Event + trigger created
5. Have meeting → End meeting → Action items extracted
6. "What are the departments doing?" → Status shown

**Verification:**
- [ ] Tasks persist across sessions
- [ ] Calendar shows real Google Calendar data
- [ ] Morning greeting includes all sections
- [ ] Meeting action items become tasks
- [ ] Department status accurate

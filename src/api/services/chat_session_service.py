import uuid
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from src.database.engine import Session
from src.database.models import ChatSession, ChatMessage


class ChatSessionService:
    """Service for managing chat sessions and messages."""

    def __init__(self):
        pass

    @staticmethod
    def _normalize_session_type(
        raw_value: Optional[str],
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Normalize to contract session types."""
        aliases = {
            "interactive": "interactive_session",
            "interactive_session": "interactive_session",
            "chat": "interactive_session",
            "workflow": "workflow_session",
            "workflow_session": "workflow_session",
            "harness": "workflow_session",
            "autonomous": "workflow_session",
            "background": "workflow_session",
        }
        value = str(raw_value or "").strip().lower()
        if value in aliases:
            return aliases[value]
        ctx = context or {}
        if any(ctx.get(key) for key in ("workflow_id", "workflow_run_id", "workflow_step")):
            return "workflow_session"
        return "interactive_session"

    async def create_session(
        self,
        agent_type: str,
        agent_id: str,
        user_id: str,
        title: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ChatSession:
        """
        Create a new chat session with cross-session memory pre-loading.

        On creation:
        1. Saves session to SQLite
        2. Pre-loads committed memory nodes from prior sessions (graph memory)
        3. Loads canvas context template if canvas specified in context
        4. Publishes session creation to thought stream for UI awareness

        The committed memory from prior sessions enables continuity —
        agents can recall decisions, opinions, and observations from
        earlier sessions without needing to re-discover them.
        """
        session_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        # Build session context with memory pre-load
        session_context = dict(context or {})
        session_context["session_type"] = self._normalize_session_type(
            session_context.get("session_type"),
            context=session_context,
        )

        # Pre-load committed memory from prior sessions
        memory_summary = await self._load_cross_session_memory(
            agent_type=agent_type,
            agent_id=agent_id,
            canvas=session_context.get("canvas", ""),
        )
        if memory_summary:
            session_context["prior_session_memory"] = memory_summary

        # Load canvas context template if canvas is specified
        canvas_name = session_context.get("canvas", "")
        if canvas_name:
            canvas_ctx = self._load_canvas_context(canvas_name)
            if canvas_ctx:
                session_context["canvas_context"] = canvas_ctx

        session = ChatSession(
            id=session_id,
            agent_type=agent_type,
            agent_id=agent_id,
            title=title or f"Chat {now.strftime('%Y-%m-%d %H:%M')}",
            user_id=user_id,
            context=session_context,
            session_metadata=metadata or {},
            last_message_at=now
        )

        db = Session()
        try:
            db.add(session)
            db.commit()
            db.refresh(session)
            db.expunge(session)

            # Publish session creation to thought stream
            self._publish_session_created(session_id, agent_type, agent_id, canvas_name)

            return session
        finally:
            db.close()

    async def _load_cross_session_memory(
        self,
        agent_type: str,
        agent_id: str,
        canvas: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Load committed memory nodes from prior sessions.

        Retrieves HOT/WARM tier nodes relevant to this agent/canvas,
        filtered to committed-only (not draft) so agents see validated knowledge.

        Returns a list of memory node summaries (max 15 for token budget).
        """
        try:
            from src.memory.graph.facade import GraphMemoryFacade
            facade = GraphMemoryFacade()

            # Load committed state for this department/agent
            committed = facade.load_committed_state(
                department=agent_id if agent_id else agent_type,
            )
            if not committed:
                return []

            # Summarize for session context (max 15 nodes)
            summaries = []
            for node in committed[:15]:
                summaries.append({
                    "id": node.get("id", ""),
                    "type": node.get("node_type", ""),
                    "content": str(node.get("content", ""))[:200],
                    "tier": node.get("tier", ""),
                    "created_at": node.get("created_at", ""),
                })
            return summaries
        except Exception:
            return []

    def _load_canvas_context(self, canvas_name: str) -> Optional[Dict[str, Any]]:
        """Load canvas context template for session initialization."""
        try:
            from src.canvas_context.loader import load_template
            template = load_template(canvas_name)
            if template:
                return {
                    "canvas": template.canvas,
                    "display_name": template.canvas_display_name,
                    "department_head": template.department_head,
                    "memory_scope": template.memory_scope,
                    "workflow_namespaces": template.workflow_namespaces,
                    "skills": [s.id for s in (template.skill_index or [])],
                }
        except Exception:
            pass
        return None

    def _publish_session_created(
        self, session_id: str, agent_type: str, agent_id: str, canvas: str,
    ) -> None:
        """Publish session creation event to thought stream for UI."""
        try:
            from src.api.agent_thought_stream_endpoints import get_thought_publisher
            get_thought_publisher().publish(
                department=agent_id or agent_type,
                thought=f"New session created: {session_id} on canvas={canvas or 'workshop'}",
                thought_type="session",
                session_id=session_id,
            )
        except Exception:
            pass

    async def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Get a session by ID."""
        db = Session()
        try:
            session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
            if session:
                db.expunge(session)
            return session
        finally:
            db.close()

    async def update_session_title(self, session_id: str, title: str) -> Optional[ChatSession]:
        """Update a session title."""
        db = Session()
        try:
            session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
            if not session:
                return None

            session.title = title
            db.commit()
            db.refresh(session)
            db.expunge(session)
            return session
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        artifacts: Optional[List[Dict]] = None,
        tool_calls: Optional[List[Dict]] = None,
        token_count: Optional[int] = None
    ) -> ChatMessage:
        """Add a message to a session."""
        message_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        db = Session()
        try:
            # Validate session exists
            session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
            if not session:
                raise ValueError(f"Session {session_id} not found")

            # Update session's last_message_at
            session.last_message_at = now

            message = ChatMessage(
                id=message_id,
                session_id=session_id,
                role=role,
                content=content,
                artifacts=artifacts or [],
                tool_calls=tool_calls or [],
                token_count=token_count,
                created_at=now
            )
            db.add(message)
            db.commit()
            db.refresh(message)
            db.expunge(message)
            return message
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    async def get_messages(self, session_id: str, limit: int = 100) -> List[ChatMessage]:
        """Get messages for a session."""
        db = Session()
        try:
            messages = db.query(ChatMessage).filter(
                ChatMessage.session_id == session_id
            ).order_by(ChatMessage.created_at.asc()).limit(limit).all()
            for msg in messages:
                db.expunge(msg)
            return messages
        finally:
            db.close()

    async def list_sessions(
        self,
        user_id: Optional[str] = None,
        agent_type: Optional[str] = None,
        limit: int = 100
    ) -> List[ChatSession]:
        """List sessions with optional filtering."""
        db = Session()
        try:
            query = db.query(ChatSession)

            if user_id:
                query = query.filter(ChatSession.user_id == user_id)
            if agent_type:
                query = query.filter(ChatSession.agent_type == agent_type)

            sessions = query.order_by(ChatSession.updated_at.desc()).limit(limit).all()

            for session in sessions:
                db.expunge(session)
            return sessions
        finally:
            db.close()

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its messages."""
        db = Session()
        try:
            session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
            if not session:
                return False
            db.delete(session)
            db.commit()
            return True
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    async def build_session_context(self, session_id: str) -> Dict[str, Any]:
        """
        Build context for a chat session from history and memory.

        HOT: Recent messages from SQLite
        WARM: Search memory facade for relevant context (if available)
        """
        # Get recent messages (HOT tier)
        messages = await self.get_messages(session_id, limit=20)

        # Format history
        history = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        # Get session for context
        session = await self.get_session(session_id)

        # Try to get memory context (WARM tier via UnifiedMemoryFacade)
        memory_context = []
        try:
            from src.agents.memory.unified_memory_facade import get_memory_facade
            memory = get_memory_facade()

            if history:
                last_message = history[-1]["content"]
                # Search memory for relevant context
                results = await memory.search(last_message, limit=3)
                memory_context = results
        except Exception:
            pass  # Memory not available

        return {
            "session_id": session_id,
            "agent_type": session.agent_type if session else None,
            "agent_id": session.agent_id if session else None,
            "context": session.context if session else {},
            "history": history,
            "memory": memory_context
        }

    async def archive_old_sessions(self, days: int = 30) -> int:
        """
        Archive sessions older than specified days to cold storage.

        Moves session data to cold_storage.db for long-term retention.
        Returns count of archived sessions.
        """
        from datetime import timedelta
        import json
        from pathlib import Path
        import sqlite3

        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        db = Session()
        try:
            old_sessions = db.query(ChatSession).filter(
                ChatSession.last_message_at < cutoff
            ).all()

            archived_count = 0
            for session in old_sessions:
                # Get all messages for this session
                messages = db.query(ChatMessage).filter(
                    ChatMessage.session_id == session.id
                ).all()

                # Archive to cold storage
                self._archive_to_cold_storage(session, messages)
                archived_count += 1

                # Delete from hot storage
                db.delete(session)

            db.commit()
            return archived_count

        finally:
            db.close()

    def _archive_to_cold_storage(self, session: ChatSession, messages: List[ChatMessage]):
        """Archive session to cold storage (SQLite)."""
        import json
        from pathlib import Path
        import sqlite3

        # Use existing cold storage
        cold_db_path = Path("data/departments/cold_storage.db")
        cold_db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(cold_db_path))
        cursor = conn.cursor()

        # Create table if not exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS archived_chat_sessions (
                id TEXT PRIMARY KEY,
                agent_type TEXT,
                agent_id TEXT,
                title TEXT,
                user_id TEXT,
                context TEXT,
                metadata TEXT,
                created_at TEXT,
                updated_at TEXT,
                archived_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS archived_chat_messages (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                role TEXT,
                content TEXT,
                artifacts TEXT,
                tool_calls TEXT,
                created_at TEXT,
                archived_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Insert session
        cursor.execute("""
            INSERT OR REPLACE INTO archived_chat_sessions
            (id, agent_type, agent_id, title, user_id, context, metadata, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session.id,
            session.agent_type,
            session.agent_id,
            session.title,
            session.user_id,
            json.dumps(session.context) if session.context else '{}',
            json.dumps(session.session_metadata) if session.session_metadata else '{}',
            session.created_at.isoformat() if session.created_at else '',
            session.last_message_at.isoformat() if session.last_message_at else ''
        ))

        # Insert messages
        for msg in messages:
            cursor.execute("""
                INSERT OR REPLACE INTO archived_chat_messages
                (id, session_id, role, content, artifacts, tool_calls, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                msg.id,
                msg.session_id,
                msg.role,
                msg.content,
                json.dumps(msg.artifacts) if msg.artifacts else '[]',
                json.dumps(msg.tool_calls) if msg.tool_calls else '[]',
                msg.created_at.isoformat() if msg.created_at else ''
            ))

        conn.commit()
        conn.close()

    async def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        agent_id: str
    ) -> Dict[str, Any]:
        """
        Execute a tool with permission check.

        Integrates with Tool Registry for permission-based access.
        """
        # Check tool registry for permissions
        try:
            from src.agents.departments.tool_registry import get_tool_registry

            registry = get_tool_registry()
            allowed_tools = registry.get_tools_for_agent(agent_id)

            if tool_name not in allowed_tools and '*' not in allowed_tools:
                return {
                    "success": False,
                    "error": f"Tool {tool_name} not permitted for agent {agent_id}"
                }
        except Exception:
            pass  # Registry not available

        # Execute tool
        try:
            # Import tool executor - this is a placeholder
            # In actual implementation, this would call the tool executor
            return {"success": True, "result": "Tool execution not implemented"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def execute_skill(
        self,
        skill_name: str,
        parameters: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a skill via Skill Manager."""
        try:
            from src.agents.skills.skill_manager import get_skill_manager

            manager = get_skill_manager()
            result = await manager.execute(skill_name, parameters, context)
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def trigger_workflow(
        self,
        workflow_name: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Trigger a workflow via Workflow Coordinator."""
        try:
            from src.agents.departments.workflow_coordinator import get_workflow_coordinator

            coordinator = get_workflow_coordinator()
            task_id = await coordinator.start_workflow(workflow_name, parameters)
            return {"success": True, "task_id": task_id}
        except Exception as e:
            return {"success": False, "error": str(e)}

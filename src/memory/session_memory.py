"""
QuantMindX Session Memory

Session-based memory tracking with:
- Conversation history management
- Session file persistence
- Delta tracking for incremental updates
- Message role support (system, user, assistant)
"""

import asyncio
import json
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class MessageRole(str, Enum):
    """Message roles in conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    """
    A single message in a conversation.
    
    Attributes:
        role: Message role (system, user, assistant, tool)
        content: Message content
        timestamp: Message timestamp
        metadata: Additional metadata
        tool_calls: Tool calls made (for assistant/tool messages)
        tool_call_id: ID of tool call (for tool messages)
    """
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_call_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "tool_calls": self.tool_calls,
            "tool_call_id": self.tool_call_id,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create instance from dictionary."""
        return cls(
            role=MessageRole(data["role"]),
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
            tool_calls=data.get("tool_calls", []),
            tool_call_id=data.get("tool_call_id"),
        )
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI chat format."""
        msg = {
            "role": self.role.value,
            "content": self.content,
        }
        
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        
        return msg


@dataclass
class SessionDelta:
    """
    Delta tracking for incremental session updates.
    
    Tracks changes to a session since last sync.
    """
    added_messages: List[Message] = field(default_factory=list)
    updated_metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def is_empty(self) -> bool:
        """Check if delta has any changes."""
        return not self.added_messages and not self.updated_metadata


@dataclass
class SessionConfig:
    """Configuration for session behavior."""
    max_messages: int = 1000
    max_tokens: int = 128000
    persist_path: Optional[Path] = None
    auto_persist: bool = True
    compress_threshold: int = 500  # Messages before compression


class SessionMemory:
    """
    Session-based memory for conversation tracking.
    
    Features:
    - Full conversation history with metadata
    - Delta tracking for incremental updates
    - File persistence with automatic save/load
    - Token counting for context window management
    - Message compression for long sessions
    
    Example:
        >>> session = SessionMemory(
        ...     session_id="chat_001",
        ...     persist_path=Path("sessions/chat_001.json")
        ... )
        >>> 
        >>> # Add messages
        >>> await session.add_message(
        ...     role=MessageRole.USER,
        ...     content="Hello!"
        ... )
        >>> 
        >>> # Get conversation history
        >>> messages = session.get_messages()
        >>> 
        >>> # Get OpenAI format
        >>> openai_msgs = session.to_openai_format()
        >>> 
        >>> # Persist to disk
        >>> await session.persist()
    """
    
    def __init__(
        self,
        session_id: str,
        config: Optional[SessionConfig] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize session memory.
        
        Args:
            session_id: Unique session identifier
            config: Session configuration
            metadata: Session metadata
        """
        self.session_id = session_id
        self.config = config or SessionConfig()
        self.metadata = metadata or {}
        
        self.messages: List[Message] = []
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)
        
        # Delta tracking
        self._delta = SessionDelta()
        self._dirty = False
        
        # Load from disk if exists
        if self.config.persist_path and self.config.persist_path.exists():
            asyncio.create_task(self.load())
        
        logger.info(f"SessionMemory initialized: {session_id}")
    
    def add_message(
        self,
        role: MessageRole,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        tool_call_id: Optional[str] = None,
    ) -> Message:
        """
        Add a message to the session.
        
        Args:
            role: Message role
            content: Message content
            metadata: Additional metadata
            tool_calls: Tool calls (for assistant messages)
            tool_call_id: Tool call ID (for tool messages)
            
        Returns:
            Created Message
        """
        message = Message(
            role=role,
            content=content,
            metadata=metadata or {},
            tool_calls=tool_calls or [],
            tool_call_id=tool_call_id,
        )
        
        self.messages.append(message)
        self.updated_at = datetime.now(timezone.utc)
        
        # Track delta
        self._delta.added_messages.append(message)
        self._dirty = True
        
        # Auto-persist if enabled
        if self.config.auto_persist:
            asyncio.create_task(self.persist())
        
        # Check compression threshold
        if len(self.messages) >= self.config.compress_threshold:
            asyncio.create_task(self.compress_old_messages())
        
        logger.debug(f"Added message to {self.session_id}: {role}")
        
        return message
    
    def get_messages(
        self,
        limit: Optional[int] = None,
        roles: Optional[List[MessageRole]] = None,
        since: Optional[datetime] = None,
    ) -> List[Message]:
        """
        Get messages from the session.
        
        Args:
            limit: Maximum messages to return
            roles: Filter by roles
            since: Only messages after this timestamp
            
        Returns:
            List of messages
        """
        messages = self.messages
        
        # Filter by role
        if roles:
            messages = [m for m in messages if m.role in roles]
        
        # Filter by time
        if since:
            messages = [m for m in messages if m.timestamp >= since]
        
        # Limit
        if limit:
            messages = messages[-limit:]
        
        return messages
    
    def get_last_message(self, role: Optional[MessageRole] = None) -> Optional[Message]:
        """Get the last message, optionally filtered by role."""
        messages = self.messages
        
        if role:
            messages = [m for m in messages if m.role == role]
        
        return messages[-1] if messages else None
    
    def get_delta(self) -> SessionDelta:
        """Get pending delta and reset."""
        delta = self._delta
        self._delta = SessionDelta()
        return delta
    
    def update_metadata(self, metadata: Dict[str, Any]) -> None:
        """Update session metadata."""
        self.metadata.update(metadata)
        self._delta.updated_metadata.update(metadata)
        self._dirty = True
    
    async def persist(self) -> None:
        """Persist session to disk."""
        if not self.config.persist_path:
            return
        
        if not self._dirty and not self._delta.is_empty():
            return
        
        # Ensure parent directory exists
        self.config.persist_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
            "messages": [m.to_dict() for m in self.messages],
        }
        
        # Write to file
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self._write_file(self.config.persist_path, data)
        )
        
        self._dirty = False
        logger.debug(f"Persisted session: {self.session_id}")
    
    async def load(self) -> None:
        """Load session from disk."""
        if not self.config.persist_path:
            return
        
        if not self.config.persist_path.exists():
            return
        
        # Read from file
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(
            None,
            lambda: self._read_file(self.config.persist_path)
        )
        
        if not data:
            return
        
        # Load data
        self.session_id = data["session_id"]
        self.created_at = datetime.fromisoformat(data["created_at"])
        self.updated_at = datetime.fromisoformat(data["updated_at"])
        self.metadata = data.get("metadata", {})
        self.messages = [
            Message.from_dict(m) for m in data.get("messages", [])
        ]
        
        self._dirty = False
        logger.info(f"Loaded session: {self.session_id} ({len(self.messages)} messages)")
    
    async def compress_old_messages(self) -> int:
        """
        Compress old messages to save space.
        
        Summarizes old messages and keeps recent ones.
        
        Returns:
            Number of messages compressed
        """
        if len(self.messages) < self.config.compress_threshold:
            return 0
        
        # Keep recent messages
        keep_count = self.config.compress_threshold // 2
        old_messages = self.messages[:-keep_count]
        self.messages = self.messages[-keep_count:]
        
        # Create summary message
        summary = self._create_summary(old_messages)
        
        # Insert summary at beginning
        summary_msg = Message(
            role=MessageRole.SYSTEM,
            content=f"[Previous conversation summary]: {summary}",
            metadata={"compressed": True, "original_count": len(old_messages)},
        )
        
        self.messages.insert(0, summary_msg)
        self._dirty = True
        
        logger.info(
            f"Compressed {len(old_messages)} messages for {self.session_id}"
        )
        
        return len(old_messages)
    
    def _create_summary(self, messages: List[Message]) -> str:
        """Create summary of messages."""
        # Simple summary: count messages by role
        role_counts = {}
        for msg in messages:
            role_counts[msg.role.value] = role_counts.get(msg.role.value, 0) + 1
        
        parts = [f"{count} {role} messages" for role, count in role_counts.items()]
        return ", ".join(parts)
    
    def estimate_tokens(self) -> int:
        """
        Estimate total tokens in session.
        
        Uses rough estimation: ~4 characters per token.
        """
        total_chars = sum(len(m.content) for m in self.messages)
        return total_chars // 4
    
    def to_openai_format(
        self,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Convert messages to OpenAI chat format.
        
        Args:
            limit: Maximum messages to include
            
        Returns:
            List of message dictionaries in OpenAI format
        """
        messages = self.get_messages(limit=limit)
        return [m.to_openai_format() for m in messages]
    
    def clear(self) -> None:
        """Clear all messages from session."""
        self.messages.clear()
        self._dirty = True
        logger.info(f"Cleared session: {self.session_id}")
    
    def _write_file(self, path: Path, data: Dict[str, Any]) -> None:
        """Write data to file (synchronous)."""
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _read_file(self, path: Path) -> Optional[Dict[str, Any]]:
        """Read data from file (synchronous)."""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read session file: {e}")
            return None


class SessionManager:
    """
    Manager for multiple sessions.
    
    Provides centralized session management with:
    - Session creation and retrieval
    - Automatic persistence
    - Session cleanup
    """
    
    def __init__(
        self,
        persist_dir: Path = Path("sessions"),
        default_config: Optional[SessionConfig] = None,
    ):
        """
        Initialize session manager.
        
        Args:
            persist_dir: Directory for session files
            default_config: Default configuration for new sessions
        """
        self.persist_dir = persist_dir
        self.default_config = default_config or SessionConfig(
            persist_path=persist_dir / "{session_id}.json"
        )
        
        self.sessions: Dict[str, SessionMemory] = {}
        
        # Ensure directory exists
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"SessionManager initialized: {persist_dir}")
    
    async def get_session(
        self,
        session_id: str,
        create: bool = True,
    ) -> Optional[SessionMemory]:
        """
        Get or create a session.
        
        Args:
            session_id: Session identifier
            create: Create if doesn't exist
            
        Returns:
            SessionMemory or None
        """
        if session_id in self.sessions:
            return self.sessions[session_id]
        
        if not create:
            return None
        
        # Create new session
        persist_path = self.default_config.persist_path
        if persist_path and "{session_id}" in str(persist_path):
            persist_path = Path(str(persist_path).format(session_id=session_id))
        
        config = SessionConfig(
            max_messages=self.default_config.max_messages,
            max_tokens=self.default_config.max_tokens,
            persist_path=persist_path,
            auto_persist=self.default_config.auto_persist,
            compress_threshold=self.default_config.compress_threshold,
        )
        
        session = SessionMemory(
            session_id=session_id,
            config=config,
        )
        
        self.sessions[session_id] = session
        return session
    
    async def close_session(self, session_id: str) -> bool:
        """
        Close and persist a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if closed
        """
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        await session.persist()
        
        del self.sessions[session_id]
        logger.info(f"Closed session: {session_id}")
        
        return True
    
    async def close_all(self) -> None:
        """Close all sessions."""
        for session_id in list(self.sessions.keys()):
            await self.close_session(session_id)
    
    def list_sessions(self) -> List[str]:
        """List all active session IDs."""
        return list(self.sessions.keys())
    
    async def cleanup_old_sessions(
        self,
        older_than_hours: int = 24,
    ) -> List[str]:
        """
        Clean up old inactive sessions.
        
        Args:
            older_than_hours: Remove sessions inactive longer than this
            
        Returns:
            List of removed session IDs
        """
        cutoff = datetime.now(timezone.utc).replace(
            tzinfo=None
        ) - __import__('datetime').timedelta(hours=older_than_hours)
        
        removed = []
        
        for session_id, session in list(self.sessions.items()):
            if session.updated_at.replace(tzinfo=None) < cutoff:
                await self.close_session(session_id)
                removed.append(session_id)
        
        return removed

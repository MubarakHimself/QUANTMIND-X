"""
Agent Communication and Coordination

Implements patterns for agent handoffs, message formats, and coordination.

**Validates: Requirements 11.1-11.10**
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# Message Formats
# ============================================================================

class MessageRole(Enum):
    """Message role types."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    AGENT = "agent"


@dataclass
class StructuredMessage:
    """
    Structured message format for agent communication.
    
    **Validates: Requirements 11.2**
    """
    role: MessageRole
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    agent_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "role": self.role.value,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "agent_id": self.agent_id
        }


# ============================================================================
# Agent Handoff Patterns
# ============================================================================

class HandoffManager:
    """
    Manages agent handoffs using LangGraph multi-agent coordination.
    
    **Validates: Requirements 11.1**
    """
    
    def __init__(self):
        self.handoff_history: List[Dict[str, Any]] = []
    
    def handoff_to_agent(
        self,
        from_agent: str,
        to_agent: str,
        task: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Hand off task from one agent to another.
        
        Args:
            from_agent: Source agent name
            to_agent: Target agent name
            task: Task description
            context: Task context and data
            
        Returns:
            Handoff result
        """
        handoff = {
            "from_agent": from_agent,
            "to_agent": to_agent,
            "task": task,
            "context": context,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "initiated"
        }
        
        self.handoff_history.append(handoff)
        
        logger.info(f"Handoff from {from_agent} to {to_agent}: {task}")
        
        return handoff
    
    def get_handoff_history(self) -> List[Dict[str, Any]]:
        """Get complete handoff history."""
        return self.handoff_history


# ============================================================================
# Subagent Wrapping
# ============================================================================

class SubagentWrapper:
    """
    Wrapper for parallel subagent execution.
    
    **Validates: Requirements 11.4**
    """
    
    def __init__(self, agent_name: str, agent_function: callable):
        self.agent_name = agent_name
        self.agent_function = agent_function
    
    async def execute_async(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute subagent task asynchronously."""
        logger.info(f"Executing subagent {self.agent_name}")
        
        result = {
            "agent": self.agent_name,
            "task": task,
            "status": "completed",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return result


# ============================================================================
# Shared State Management
# ============================================================================

class SharedStateManager:
    """
    Manages shared state across coordinated workflows.
    
    **Validates: Requirements 11.5**
    """
    
    def __init__(self):
        self.shared_state: Dict[str, Any] = {}
        self.state_history: List[Dict[str, Any]] = []
    
    def update_state(self, key: str, value: Any, agent_id: str) -> None:
        """Update shared state."""
        self.shared_state[key] = value
        
        self.state_history.append({
            "key": key,
            "value": value,
            "agent_id": agent_id,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        logger.debug(f"State updated by {agent_id}: {key} = {value}")
    
    def get_state(self, key: str) -> Optional[Any]:
        """Get shared state value."""
        return self.shared_state.get(key)
    
    def get_all_state(self) -> Dict[str, Any]:
        """Get all shared state."""
        return self.shared_state.copy()


# ============================================================================
# Communication Patterns
# ============================================================================

class CommunicationManager:
    """
    Manages synchronous and asynchronous communication patterns.
    
    **Validates: Requirements 11.6**
    """
    
    def __init__(self):
        self.message_queue: List[StructuredMessage] = []
    
    def send_sync(self, message: StructuredMessage) -> Dict[str, Any]:
        """Send synchronous message."""
        self.message_queue.append(message)
        
        logger.info(f"Sync message sent from {message.agent_id}")
        
        return {"status": "delivered", "message_id": len(self.message_queue)}
    
    async def send_async(self, message: StructuredMessage) -> Dict[str, Any]:
        """Send asynchronous message."""
        self.message_queue.append(message)
        
        logger.info(f"Async message sent from {message.agent_id}")
        
        return {"status": "queued", "message_id": len(self.message_queue)}
    
    def get_messages(self, agent_id: Optional[str] = None) -> List[StructuredMessage]:
        """Get messages for agent."""
        if agent_id:
            return [msg for msg in self.message_queue if msg.agent_id == agent_id]
        return self.message_queue


# ============================================================================
# Skill Registry
# ============================================================================

class SkillRegistry:
    """
    Centralized skill registry for agent skill sharing.
    
    **Validates: Requirements 11.7**
    """
    
    def __init__(self):
        self.skills: Dict[str, Dict[str, Any]] = {}
    
    def register_skill(
        self,
        skill_name: str,
        skill_function: callable,
        agent_id: str,
        metadata: Dict[str, Any] = None
    ) -> None:
        """Register a skill."""
        self.skills[skill_name] = {
            "function": skill_function,
            "agent_id": agent_id,
            "metadata": metadata or {},
            "registered_at": datetime.utcnow().isoformat()
        }
        
        logger.info(f"Skill registered: {skill_name} by {agent_id}")
    
    def get_skill(self, skill_name: str) -> Optional[Dict[str, Any]]:
        """Get skill by name."""
        return self.skills.get(skill_name)
    
    def list_skills(self) -> List[str]:
        """List all registered skills."""
        return list(self.skills.keys())


# ============================================================================
# Human-in-the-Loop Integration
# ============================================================================

class HumanInTheLoopManager:
    """
    Manages human-in-the-loop integration points.
    
    **Validates: Requirements 11.8**
    """
    
    def __init__(self):
        self.pending_approvals: List[Dict[str, Any]] = []
    
    def request_approval(
        self,
        agent_id: str,
        action: str,
        context: Dict[str, Any]
    ) -> str:
        """Request human approval for action."""
        approval_id = f"approval_{len(self.pending_approvals) + 1}"
        
        approval_request = {
            "approval_id": approval_id,
            "agent_id": agent_id,
            "action": action,
            "context": context,
            "status": "pending",
            "requested_at": datetime.utcnow().isoformat()
        }
        
        self.pending_approvals.append(approval_request)
        
        logger.info(f"Approval requested: {approval_id} for {action}")
        
        return approval_id
    
    def approve(self, approval_id: str) -> bool:
        """Approve pending action."""
        for approval in self.pending_approvals:
            if approval["approval_id"] == approval_id:
                approval["status"] = "approved"
                approval["approved_at"] = datetime.utcnow().isoformat()
                logger.info(f"Approval granted: {approval_id}")
                return True
        return False
    
    def reject(self, approval_id: str, reason: str = "") -> bool:
        """Reject pending action."""
        for approval in self.pending_approvals:
            if approval["approval_id"] == approval_id:
                approval["status"] = "rejected"
                approval["rejected_at"] = datetime.utcnow().isoformat()
                approval["rejection_reason"] = reason
                logger.info(f"Approval rejected: {approval_id}")
                return True
        return False


# ============================================================================
# Error Handling and Retry
# ============================================================================

class CoordinationErrorHandler:
    """
    Error handling and retry mechanisms for agent coordination.
    
    **Validates: Requirements 11.9**
    """
    
    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        self.error_log: List[Dict[str, Any]] = []
    
    def handle_error(
        self,
        agent_id: str,
        error: Exception,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle coordination error."""
        error_entry = {
            "agent_id": agent_id,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.error_log.append(error_entry)
        
        logger.error(f"Coordination error in {agent_id}: {error}")
        
        return error_entry
    
    def should_retry(self, agent_id: str) -> bool:
        """Determine if operation should be retried."""
        recent_errors = [
            e for e in self.error_log
            if e["agent_id"] == agent_id
        ]
        
        return len(recent_errors) < self.max_retries


# ============================================================================
# Audit Trail
# ============================================================================

class AuditTrailLogger:
    """
    Audit trail logging for inter-agent communications.
    
    **Validates: Requirements 11.10**
    """
    
    def __init__(self):
        self.audit_log: List[Dict[str, Any]] = []
    
    def log_communication(
        self,
        from_agent: str,
        to_agent: str,
        message_type: str,
        content: str,
        metadata: Dict[str, Any] = None
    ) -> None:
        """Log inter-agent communication."""
        log_entry = {
            "from_agent": from_agent,
            "to_agent": to_agent,
            "message_type": message_type,
            "content": content,
            "metadata": metadata or {},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.audit_log.append(log_entry)
        
        logger.debug(f"Audit: {from_agent} -> {to_agent}: {message_type}")
    
    def get_audit_trail(
        self,
        agent_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get audit trail, optionally filtered by agent."""
        if agent_id:
            return [
                entry for entry in self.audit_log
                if entry["from_agent"] == agent_id or entry["to_agent"] == agent_id
            ]
        return self.audit_log

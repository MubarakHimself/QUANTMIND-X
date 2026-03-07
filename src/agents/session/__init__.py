"""
Session Management Module.

Provides checkpointing and session routine functionality for long-running agents.
Based on Anthropic's effective-harnesses-for-long-running-agents patterns.
"""

from .checkpoint import GitCheckpointManager, get_checkpoint_manager
from .routine import SessionRoutine, get_session_routine

__all__ = [
    "GitCheckpointManager",
    "get_checkpoint_manager",
    "SessionRoutine",
    "get_session_routine",
]

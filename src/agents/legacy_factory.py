"""Legacy compatibility exports for the deprecated agent factory path.

The canonical runtime path is the ClaudeOrchestrator-based stack. This module
keeps historical imports working while making the legacy status explicit.
"""

from src.agents.factory import AgentFactory, END, SimpleWorkflow, STATE_CLASS_MAP

__all__ = ["AgentFactory", "END", "SimpleWorkflow", "STATE_CLASS_MAP"]

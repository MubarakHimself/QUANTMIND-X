"""
Claude Agent Configuration for v2 Agent Stack

Defines configuration for all six agents that replace the LangGraph-based system.
Each agent uses Claude CLI with MCP servers and bash tools.

**Phase 2.2 - Claude Agent Configuration**
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Callable, Optional, List
from pathlib import Path

# Base paths - default to local development path
WORKSPACES_DIR = Path(os.getenv("WORKSPACES_DIR", str(Path(__file__).parent.parent.parent / "workspaces")))
CONFIG_DIR = Path(os.getenv("CONFIG_DIR", str(Path(__file__).parent.parent.parent / "config")))
MCP_CONFIG_DIR = CONFIG_DIR / "mcp"


@dataclass
class ClaudeAgentConfig:
    """
    Configuration for a Claude-powered agent.
    
    Attributes:
        agent_id: Unique identifier (analyst, quantcode, copilot, etc.)
        workspace: Path to agent's workspace directory
        system_prompt_path: Path to CLAUDE.md system prompt
        mcp_config_path: Path to MCP configuration JSON
        timeout_seconds: Maximum execution time per task
        max_retries: Number of retries on failure
        pre_hooks: Callable(s) to run before task execution
        post_hooks: Callable(s) to run after task completion
        env_vars: Additional environment variables for the subprocess
    """
    agent_id: str
    workspace: Path
    system_prompt_path: Path
    mcp_config_path: Path
    timeout_seconds: int = 300
    max_retries: int = 3
    pre_hooks: List[Callable] = field(default_factory=list)
    post_hooks: List[Callable] = field(default_factory=list)
    env_vars: Dict[str, str] = field(default_factory=dict)
    
    @property
    def tasks_dir(self) -> Path:
        """Path to tasks directory."""
        return self.workspace / "tasks"
    
    @property
    def results_dir(self) -> Path:
        """Path to results directory."""
        return self.workspace / "results"
    
    @property
    def context_dir(self) -> Path:
        """Path to context directory."""
        return self.workspace / "context"
    
    @property
    def scratch_dir(self) -> Path:
        """Path to scratch directory."""
        return self.workspace / "scratch"
    
    def validate(self) -> bool:
        """Validate that required paths exist."""
        return (
            self.workspace.exists() and
            self.tasks_dir.exists() and
            self.results_dir.exists() and
            self.context_dir.exists()
        )
    
    def ensure_directories(self) -> None:
        """Create workspace directories if they don't exist."""
        for dir_path in [
            self.workspace,
            self.tasks_dir,
            self.results_dir,
            self.context_dir,
            self.scratch_dir
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)


# Legacy agent hooks - these agents are deprecated, return empty hooks
def _get_analyst_hooks():
    """DEPRECATED: Legacy analyst hooks removed."""
    return [], []


def _get_quantcode_hooks():
    """DEPRECATED: Legacy quantcode hooks removed."""
    return [], []


def _get_copilot_hooks():
    """DEPRECATED: Legacy copilot hooks removed."""
    return [], []


def _get_default_hooks():
    """No hooks for lightweight agents."""
    return [], []


# Agent configuration registry
AGENT_CONFIGS: Dict[str, ClaudeAgentConfig] = {
    "analyst": ClaudeAgentConfig(
        agent_id="analyst",
        workspace=WORKSPACES_DIR / "analyst",
        system_prompt_path=WORKSPACES_DIR / "analyst" / "context" / "CLAUDE.md",
        mcp_config_path=MCP_CONFIG_DIR / "analyst-mcp.json",
        timeout_seconds=600,  # 10 minutes for complex Video Ingest analysis
        max_retries=3,
        env_vars={
            "AGENT_ROLE": "analyst",
            "AGENT_MODE": "production",
        },
    ),
    "quantcode": ClaudeAgentConfig(
        agent_id="quantcode",
        workspace=WORKSPACES_DIR / "quantcode",
        system_prompt_path=WORKSPACES_DIR / "quantcode" / "context" / "CLAUDE.md",
        mcp_config_path=MCP_CONFIG_DIR / "quantcode-mcp.json",
        timeout_seconds=900,  # 15 minutes for code generation + backtest
        max_retries=2,
        env_vars={
            "AGENT_ROLE": "quantcode",
            "AGENT_MODE": "production",
        },
    ),
    "copilot": ClaudeAgentConfig(
        agent_id="copilot",
        workspace=WORKSPACES_DIR / "copilot",
        system_prompt_path=WORKSPACES_DIR / "copilot" / "context" / "CLAUDE.md",
        mcp_config_path=MCP_CONFIG_DIR / "copilot-mcp.json",
        timeout_seconds=300,  # 5 minutes for orchestration tasks
        max_retries=3,
        env_vars={
            "AGENT_ROLE": "copilot",
            "AGENT_MODE": "production",
        },
    ),
    "pinescript": ClaudeAgentConfig(
        agent_id="pinescript",
        workspace=WORKSPACES_DIR / "pinescript",
        system_prompt_path=WORKSPACES_DIR / "pinescript" / "context" / "CLAUDE.md",
        mcp_config_path=MCP_CONFIG_DIR / "pinescript-mcp.json",
        timeout_seconds=600,  # 10 minutes for Pine Script generation
        max_retries=2,
        env_vars={
            "AGENT_ROLE": "pinescript",
            "AGENT_MODE": "production",
        },
    ),
    "router": ClaudeAgentConfig(
        agent_id="router",
        workspace=WORKSPACES_DIR / "router",
        system_prompt_path=WORKSPACES_DIR / "router" / "context" / "CLAUDE.md",
        mcp_config_path=MCP_CONFIG_DIR / "router-mcp.json",
        timeout_seconds=60,  # 1 minute for lightweight routing
        max_retries=1,
        env_vars={
            "AGENT_ROLE": "router",
            "AGENT_MODE": "production",
        },
    ),
    "executor": ClaudeAgentConfig(
        agent_id="executor",
        workspace=WORKSPACES_DIR / "executor",
        system_prompt_path=WORKSPACES_DIR / "executor" / "context" / "CLAUDE.md",
        mcp_config_path=MCP_CONFIG_DIR / "executor-mcp.json",
        timeout_seconds=120,  # 2 minutes for execution tasks
        max_retries=2,
        env_vars={
            "AGENT_ROLE": "executor",
            "AGENT_MODE": "production",
        },
    ),
}


def get_agent_config(agent_id: str) -> Optional[ClaudeAgentConfig]:
    """
    Get configuration for a specific agent.
    
    Args:
        agent_id: Agent identifier
        
    Returns:
        ClaudeAgentConfig or None if not found
    """
    return AGENT_CONFIGS.get(agent_id)


def get_all_agent_ids() -> List[str]:
    """Get list of all configured agent IDs."""
    return list(AGENT_CONFIGS.keys())


def initialize_hooks():
    """
    Initialize hooks for all agents after module load.
    
    This must be called after the hooks module is fully loaded.
    """
    # Analyst hooks
    pre_hooks, post_hooks = _get_analyst_hooks()
    AGENT_CONFIGS["analyst"].pre_hooks = pre_hooks
    AGENT_CONFIGS["analyst"].post_hooks = post_hooks
    
    # QuantCode hooks
    pre_hooks, post_hooks = _get_quantcode_hooks()
    AGENT_CONFIGS["quantcode"].pre_hooks = pre_hooks
    AGENT_CONFIGS["quantcode"].post_hooks = post_hooks
    
    # Copilot hooks
    pre_hooks, post_hooks = _get_copilot_hooks()
    AGENT_CONFIGS["copilot"].pre_hooks = pre_hooks
    AGENT_CONFIGS["copilot"].post_hooks = post_hooks
    
    # Router and executor have no hooks
    pre_hooks, post_hooks = _get_default_hooks()
    AGENT_CONFIGS["router"].pre_hooks = pre_hooks
    AGENT_CONFIGS["router"].post_hooks = post_hooks
    AGENT_CONFIGS["executor"].pre_hooks = pre_hooks
    AGENT_CONFIGS["executor"].post_hooks = post_hooks
    AGENT_CONFIGS["pinescript"].pre_hooks = pre_hooks
    AGENT_CONFIGS["pinescript"].post_hooks = post_hooks
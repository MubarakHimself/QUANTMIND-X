"""
Agent implementations for QuantMindX Unified Backend.

Simplified version - uses Claude Code native agents.
Legacy LangGraph agents have been removed.

Includes comprehensive hooks, cron, and subagent systems.
"""

from src.agents.state import (
    AgentState,
    AnalystState,
    QuantCodeState,
    CopilotState,
    RouterState
)

from src.agents.router import (
    create_router_graph,
    compile_router_graph,
    run_router_workflow,
    create_router_workflow,
    create_router_agent,
    create_router_from_config,
    RouterGraphWrapper,
    MemorySaver,
)

# Hooks system (inspired by OpenClaw)
from src.agents.hooks import (
    HookType,
    HookContext,
    HookCondition,
    HookResult,
    HookRegistry,
    get_global_registry as get_hook_registry,
    register_hook,
    execute_hooks,
)

# Cron scheduler
from src.agents.cron import (
    CronJob,
    JobSchedule,
    Scheduler,
    get_scheduler,
    initialize_scheduler,
    MemoryConsolidationJob,
    SessionCleanupJob,
    EmbeddingSyncJob,
    HealthCheckJob,
)

# Sub-agent spawner
from src.agents.subagent import (
    SubAgentConfig,
    SubAgentStatus,
    SubAgent,
    AgentSpawner,
    AgentPoolConfig,
    get_spawner,
)

# Placeholder functions for backward compatibility
# These will use Claude Code native backend
def create_analyst_agent(*args, **kwargs):
    """Placeholder - use Claude Code native backend."""
    pass

def create_quantcode_agent(*args, **kwargs):
    """Placeholder - use Claude Code native backend."""
    pass

def create_copilot_agent(*args, **kwargs):
    """Placeholder - use Claude Code native backend."""
    pass

def compile_copilot_graph(*args, **kwargs):
    """Placeholder - use Claude Code native backend."""
    pass

# Aliases for backward compatibility
create_analyst_graph = create_analyst_agent
compile_analyst_graph = create_analyst_agent
run_analyst_workflow = create_analyst_agent

create_quantcode_graph = create_quantcode_agent
run_quantcode_workflow = create_quantcode_agent

create_copilot_graph = compile_copilot_graph
run_copilot_workflow = lambda *args, **kwargs: compile_copilot_graph()

__all__ = [
    # State definitions
    "AgentState",
    "AnalystState",
    "QuantCodeState",
    "CopilotState",
    "RouterState",

    # Placeholder functions
    "create_analyst_agent",
    "create_analyst_graph",
    "compile_analyst_graph",
    "run_analyst_workflow",

    "create_quantcode_agent",
    "create_quantcode_graph",
    "run_quantcode_workflow",

    "compile_copilot_graph",
    "create_copilot_graph",
    "run_copilot_workflow",
    "create_copilot_agent",

    # Router agent (still available)
    "create_router_graph",
    "compile_router_graph",
    "run_router_workflow",
    "create_router_workflow",
    "create_router_agent",
    "create_router_from_config",
    "RouterGraphWrapper",
    "MemorySaver",

    # Hooks system
    "HookType",
    "HookContext",
    "HookCondition",
    "HookResult",
    "HookRegistry",
    "get_hook_registry",
    "register_hook",
    "execute_hooks",

    # Cron scheduler
    "CronJob",
    "JobSchedule",
    "Scheduler",
    "get_scheduler",
    "initialize_scheduler",
    "MemoryConsolidationJob",
    "SessionCleanupJob",
    "EmbeddingSyncJob",
    "HealthCheckJob",

    # Sub-agent spawner
    "SubAgentConfig",
    "SubAgentStatus",
    "SubAgent",
    "AgentSpawner",
    "AgentPoolConfig",
    "get_spawner",
]

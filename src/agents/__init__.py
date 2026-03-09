"""
Agent implementations for QuantMindX Unified Backend.

Simplified version - uses Claude Code native agents.
Legacy LangGraph agents have been removed.

Includes comprehensive hooks, cron, and subagent systems.
"""

# Hooks system (inspired by OpenClaw) - optional
try:
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
except ImportError:
    HookType = None
    HookContext = None
    HookCondition = None
    HookResult = None
    HookRegistry = None
    get_hook_registry = None
    register_hook = None
    execute_hooks = None

# Cron scheduler - optional
try:
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
except ImportError:
    CronJob = None
    JobSchedule = None
    Scheduler = None
    get_scheduler = None
    initialize_scheduler = None
    MemoryConsolidationJob = None
    SessionCleanupJob = None
    EmbeddingSyncJob = None
    HealthCheckJob = None

# Sub-agent spawner - optional
try:
    from src.agents.subagent import (
        SubAgentConfig,
        SubAgentStatus,
        SubAgent,
        AgentSpawner,
        AgentPoolConfig,
        get_spawner,
    )
except ImportError:
    SubAgentConfig = None
    SubAgentStatus = None
    SubAgent = None
    AgentSpawner = None
    AgentPoolConfig = None
    get_spawner = None

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
    # Placeholder functions (legacy agents removed)
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

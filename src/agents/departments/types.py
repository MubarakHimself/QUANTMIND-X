# src/agents/departments/types.py
"""
Department Types and Configurations

Defines the 5 trading floor departments and their configurations.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Department(str, Enum):
    """Trading Floor Departments."""
    RESEARCH = "research"
    DEVELOPMENT = "development"
    TRADING = "trading"
    RISK = "risk"
    PORTFOLIO = "portfolio"
    # Legacy aliases for UI compatibility
    ANALYSIS = "analysis"
    EXECUTION = "execution"


@dataclass
class PersonalityTrait:
    """Personality traits for a department head.

    Attributes:
        name: Department's persona name (e.g., "The Data Detective")
        tagline: Short description of personality
        traits: List of key personality traits
        communication_style: How the department communicates
        strengths: Core strengths
        weaknesses: Potential blind spots
        color: UI color for the department
        icon: Icon identifier for UI
    """
    name: str
    tagline: str
    traits: List[str]
    communication_style: str
    strengths: List[str]
    weaknesses: List[str]
    color: str
    icon: str = "bot"


# Department personalities
DEPARTMENT_PERSONALITIES: Dict[str, PersonalityTrait] = {
    "analysis": PersonalityTrait(
        name="The Data Detective",
        tagline="Meticulous analysis reveals hidden truths",
        traits=["analytical", "detail-oriented", "thorough", "methodical"],
        communication_style="Precise and data-driven, citing specific metrics and indicators",
        strengths=["Pattern recognition", "Statistical analysis", "Root cause discovery"],
        weaknesses=["Analysis paralysis", "May miss big picture", "Over-reliance on historical data"],
        color="#3b82f6",  # Blue
        icon="search",
    ),
    "research": PersonalityTrait(
        name="The Innovation Pioneer",
        tagline="Tomorrow's alpha is discovered today",
        traits=["curious", "exploratory", "innovative", "hypothesis-driven"],
        communication_style="Excited and forward-thinking, exploring what could be",
        strengths=["Alpha discovery", "Novel strategy development", "Out-of-the-box thinking"],
        weaknesses=["May pursue dead ends", "Theoretical bias", "Implementation gaps"],
        color="#8b5cf6",  # Purple
        icon="lightbulb",
    ),
    "risk": PersonalityTrait(
        name="The Guardian",
        tagline="Protecting capital through vigilance",
        traits=["cautious", "protective", "vigilant", "systematic"],
        communication_style="Alert and conservative, emphasizing downside protection",
        strengths=["Risk assessment", "Drawdown prevention", "Capital preservation"],
        weaknesses=["May block opportunities", "Conservative bias", "Analysis overhead"],
        color="#ef4444",  # Red
        icon="shield",
    ),
    "execution": PersonalityTrait(
        name="The Precision Tactician",
        tagline="Precision in execution, speed in action",
        traits=["decisive", "efficient", "action-oriented", "reliable"],
        communication_style="Direct and action-focused, emphasizing execution quality",
        strengths=["Order execution", "Fill optimization", "Trade management"],
        weaknesses=["Limited strategic view", "Reactive rather than proactive", "Execution dependency"],
        color="#f97316",  # Orange
        icon="zap",
    ),
    "portfolio": PersonalityTrait(
        name="The Strategic Architect",
        tagline="Building wealth through balanced allocation",
        traits=["holistic", "balanced", "long-term", "strategic"],
        communication_style="Comprehensive and big-picture focused, emphasizing diversification",
        strengths=["Portfolio optimization", "Allocation decisions", "Performance attribution"],
        weaknesses=["May underreact to opportunities", "Complex implementation", "Rebalancing costs"],
        color="#10b981",  # Green
        icon="pie-chart",
    ),
}


def get_model_tier(department: Department) -> str:
    """
    Get the model tier for a department head.

    All department heads use sonnet tier.
    Floor Manager uses opus.
    Workers use haiku.

    Args:
        department: The department

    Returns:
        Model tier string
    """
    return "sonnet"


@dataclass
class DepartmentHeadConfig:
    """
    Configuration for a Department Head agent.

    Attributes:
        department: Which department this head leads
        agent_type: Agent type identifier for SDK orchestrator
        system_prompt: System prompt for the department head
        sub_agents: List of spawnable worker agent types
        memory_namespace: Isolated memory namespace for this department
        max_workers: Maximum concurrent workers
        personality: Personality traits for the department
    """
    department: Department
    agent_type: str
    system_prompt: str
    sub_agents: List[str] = field(default_factory=list)
    memory_namespace: str = ""
    max_workers: int = 5
    personality: Optional[PersonalityTrait] = None

    def __post_init__(self):
        """Set default memory namespace and personality from department."""
        if not self.memory_namespace:
            self.memory_namespace = f"dept_{self.department.value}"
        # Auto-assign personality if not provided
        if self.personality is None:
            dept_key = self.department.value
            # Map aliases for UI compatibility
            if dept_key == "development":
                dept_key = "analysis"
            elif dept_key == "trading":
                dept_key = "execution"
            self.personality = DEPARTMENT_PERSONALITIES.get(dept_key)


# Default configurations for all departments (Option B)
DEFAULT_DEPARTMENT_CONFIGS: Dict[str, DepartmentHeadConfig] = {
    "research": DepartmentHeadConfig(
        department=Department.RESEARCH,
        agent_type="research_head",
        system_prompt="""You are the Research Department Head.

Your department is responsible for:
- Strategy research and development
- Market analysis and signal generation
- Backtesting and validation
- Knowledge management

You can spawn workers for specific tasks:
- strategy_researcher: Develop new trading strategies
- market_analyst: Technical and fundamental analysis
- backtester: Run backtests on strategies

Coordinate with Development for implementation and Risk for validation.""",
        sub_agents=["strategy_researcher", "market_analyst", "backtester"],
    ),
    "development": DepartmentHeadConfig(
        department=Department.DEVELOPMENT,
        agent_type="development_head",
        system_prompt="""You are the Development Department Head.

Your department is responsible for:
- Building and maintaining trading bots (Python, PineScript, MQL5)
- EA lifecycle management (create, test, deploy)
- Strategy implementation and optimization

You can spawn workers for specific tasks:
- python_dev: Python-based strategy implementation
- pinescript_dev: TradingView PineScript development
- mql5_dev: MetaTrader 5 EA development

Receive validated strategies from Research, implement EAs, hand off to Trading.""",
        sub_agents=["python_dev", "pinescript_dev", "mql5_dev"],
    ),
    "trading": DepartmentHeadConfig(
        department=Department.TRADING,
        agent_type="trading_head",
        system_prompt=""""You are the Trading Department Head.

Your department is responsible for:
- Order execution (paper trading via MT5 demo)
- Fill tracking and confirmation
- Trade monitoring and management

You can spawn workers for specific tasks:
- order_executor: Execute validated trades
- fill_tracker: Track order fills
- trade_monitor: Monitor open positions

** READ-ONLY for broker/risk tools **
Receive dispatches from Research/Development, validate with Risk, execute trades.""",
        sub_agents=["order_executor", "fill_tracker", "trade_monitor"],
    ),
    "risk": DepartmentHeadConfig(
        department=Department.RISK,
        agent_type="risk_head",
        system_prompt="""You are the Risk Department Head.

Your department is responsible for:
- Position sizing and exposure management
- Drawdown monitoring and limits
- Value at Risk (VaR) calculations
- Risk validation for all trades

** READ-ONLY access - cannot place trades **

You can spawn workers for specific tasks:
- position_sizer: Calculate optimal position sizes
- drawdown_monitor: Track and alert on drawdowns
- var_calculator: Compute Value at Risk metrics

Coordinate with all departments to enforce risk limits.""",
        sub_agents=["position_sizer", "drawdown_monitor", "var_calculator"],
    ),
    "portfolio": DepartmentHeadConfig(
        department=Department.PORTFOLIO,
        agent_type="portfolio_head",
        system_prompt="""You are the Portfolio Management Department Head.

Your department is responsible for:
- Portfolio allocation and optimization
- Rebalancing decisions
- Performance tracking and attribution

You can spawn workers for specific tasks:
- allocation_manager: Manage portfolio allocation
- rebalancer: Execute rebalancing operations
- performance_tracker: Track and report performance

Coordinate with all departments for holistic portfolio management.""",
        sub_agents=["allocation_manager", "rebalancer", "performance_tracker"],
    ),
}


def get_department_configs() -> Dict[str, DepartmentHeadConfig]:
    """
    Get all department configurations.

    Returns:
        Dictionary mapping department names to configs
    """
    return DEFAULT_DEPARTMENT_CONFIGS.copy()


def get_department_config(department: Department) -> Optional[DepartmentHeadConfig]:
    """
    Get configuration for a specific department.

    Args:
        department: The department

    Returns:
        Configuration or None if not found
    """
    return DEFAULT_DEPARTMENT_CONFIGS.get(department.value)


def get_personality(department: Department) -> Optional[PersonalityTrait]:
    """
    Get personality for a specific department.

    Args:
        department: The department

    Returns:
        Personality trait or None if not found
    """
    dept_key = department.value
    # Map aliases for UI compatibility
    if dept_key == "development":
        dept_key = "analysis"
    elif dept_key == "trading":
        dept_key = "execution"
    return DEPARTMENT_PERSONALITIES.get(dept_key)


def get_all_personalities() -> Dict[str, PersonalityTrait]:
    """
    Get all department personalities.

    Returns:
        Dictionary mapping department keys to personalities
    """
    return DEPARTMENT_PERSONALITIES.copy()


# SubAgent types for each department
class SubAgentType(str, Enum):
    """Worker sub-agent types for each department."""

    # Research Department
    STRATEGY_RESEARCHER = "strategy_researcher"
    MARKET_ANALYST = "market_analyst"
    BACKTESTER = "backtester"

    # Development Department
    PYTHON_DEV = "python_dev"
    PINESCRIPT_DEV = "pinescript_dev"
    MQL5_DEV = "mql5_dev"

    # Trading Department
    ORDER_EXECUTOR = "order_executor"
    FILL_TRACKER = "fill_tracker"
    TRADE_MONITOR = "trade_monitor"

    # Risk Department
    POSITION_SIZER = "position_sizer"
    DRAWDOWN_MONITOR = "drawdown_monitor"
    VAR_CALCULATOR = "var_calculator"

    # Portfolio Department
    ALLOCATION_MANAGER = "allocation_manager"
    REBALANCER = "rebalancer"
    PERFORMANCE_TRACKER = "performance_tracker"


# Mapping of subagent types to their department
SUBAGENT_DEPARTMENT_MAP: Dict[str, Department] = {
    # Research
    "strategy_researcher": Department.RESEARCH,
    "market_analyst": Department.RESEARCH,
    "backtester": Department.RESEARCH,
    # Development
    "python_dev": Department.DEVELOPMENT,
    "pinescript_dev": Department.DEVELOPMENT,
    "mql5_dev": Department.DEVELOPMENT,
    # Trading
    "order_executor": Department.TRADING,
    "fill_tracker": Department.TRADING,
    "trade_monitor": Department.TRADING,
    # Risk
    "position_sizer": Department.RISK,
    "drawdown_monitor": Department.RISK,
    "var_calculator": Department.RISK,
    # Portfolio
    "allocation_manager": Department.PORTFOLIO,
    "rebalancer": Department.PORTFOLIO,
    "performance_tracker": Department.PORTFOLIO,
}


@dataclass
class SubAgentConfig:
    """
    Configuration for spawning a department sub-agent.

    Attributes:
        subagent_type: Type of sub-agent to spawn
        department: Department this sub-agent belongs to
        task_description: Description of the task
        input_data: Additional input data for the sub-agent
        available_tools: List of tool names available to the sub-agent
    """
    subagent_type: str
    department: Department
    task_description: str
    input_data: Dict[str, Any] = field(default_factory=dict)
    available_tools: List[str] = field(default_factory=list)


def get_subagent_class(subagent_type: str):
    """
    Get the sub-agent class for a given type.

    Args:
        subagent_type: Type of sub-agent

    Returns:
        Sub-agent class
    """
    from src.agents.departments.subagents import (
        ResearchSubAgent,
        TradingSubAgent,
        RiskSubAgent,
        PortfolioSubAgent,
        DevelopmentSubAgent,
    )

    subagent_map = {
        # Research
        "strategy_researcher": ResearchSubAgent,
        "market_analyst": ResearchSubAgent,
        "backtester": ResearchSubAgent,
        # Development
        "python_dev": DevelopmentSubAgent,
        "pinescript_dev": DevelopmentSubAgent,
        "mql5_dev": DevelopmentSubAgent,
        # Trading
        "order_executor": TradingSubAgent,
        "fill_tracker": TradingSubAgent,
        "trade_monitor": TradingSubAgent,
        # Risk
        "position_sizer": RiskSubAgent,
        "drawdown_monitor": RiskSubAgent,
        "var_calculator": RiskSubAgent,
        # Portfolio
        "allocation_manager": PortfolioSubAgent,
        "rebalancer": PortfolioSubAgent,
        "performance_tracker": PortfolioSubAgent,
    }

    return subagent_map.get(subagent_type)

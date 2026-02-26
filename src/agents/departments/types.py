# src/agents/departments/types.py
"""
Department Types and Configurations

Defines the 5 trading floor departments and their configurations.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class Department(str, Enum):
    """Trading Floor Departments."""
    ANALYSIS = "analysis"
    RESEARCH = "research"
    RISK = "risk"
    EXECUTION = "execution"
    PORTFOLIO = "portfolio"


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
    """
    department: Department
    agent_type: str
    system_prompt: str
    sub_agents: List[str] = field(default_factory=list)
    memory_namespace: str = ""
    max_workers: int = 5

    def __post_init__(self):
        """Set default memory namespace from department."""
        if not self.memory_namespace:
            self.memory_namespace = f"dept_{self.department.value}"


# Default configurations for all departments
DEFAULT_DEPARTMENT_CONFIGS: Dict[str, DepartmentHeadConfig] = {
    "analysis": DepartmentHeadConfig(
        department=Department.ANALYSIS,
        agent_type="analyst_head",
        system_prompt="""You are the Analysis Department Head.

Your department is responsible for:
- Market analysis and technical analysis
- Sentiment analysis and news monitoring
- Signal generation for trading opportunities

You can spawn workers for specific tasks:
- market_analyst: Deep technical analysis
- sentiment_scanner: Social and news sentiment
- news_monitor: Real-time news monitoring

Coordinate with Research for strategy validation and Risk for exposure limits.
Dispatch trade signals to Execution department.""",
        sub_agents=["market_analyst", "sentiment_scanner", "news_monitor"],
    ),
    "research": DepartmentHeadConfig(
        department=Department.RESEARCH,
        agent_type="research_head",
        system_prompt="""You are the Research Department Head.

Your department is responsible for:
- Strategy research and development
- Backtesting and validation
- Alpha research and data science

You can spawn workers for specific tasks:
- strategy_researcher: Develop new trading strategies
- backtester: Run backtests on strategies
- data_scientist: Statistical analysis and feature engineering

Receive analysis from Analysis department, validate strategies, and hand off to Risk.""",
        sub_agents=["strategy_researcher", "backtester", "data_scientist"],
    ),
    "risk": DepartmentHeadConfig(
        department=Department.RISK,
        agent_type="risk_head",
        system_prompt="""You are the Risk Department Head.

Your department is responsible for:
- Position sizing and exposure management
- Drawdown monitoring and limits
- Value at Risk (VaR) calculations

You can spawn workers for specific tasks:
- position_sizer: Calculate optimal position sizes
- drawdown_monitor: Track and alert on drawdowns
- var_calculator: Compute Value at Risk metrics

Coordinate with all departments to enforce risk limits.""",
        sub_agents=["position_sizer", "drawdown_monitor", "var_calculator"],
    ),
    "execution": DepartmentHeadConfig(
        department=Department.EXECUTION,
        agent_type="executor_head",
        system_prompt="""You are the Execution Department Head.

Your department is responsible for:
- Order routing and execution
- Fill tracking and confirmation
- Slippage monitoring and optimization

You can spawn workers for specific tasks:
- order_router: Route orders to best venues
- fill_tracker: Track order fills
- slippage_monitor: Monitor and optimize slippage

Receive dispatches from Analysis, validate with Risk, execute trades.""",
        sub_agents=["order_router", "fill_tracker", "slippage_monitor"],
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

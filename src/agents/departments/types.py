# src/agents/departments/types.py
"""
Department Types and Configurations

Defines the 5 trading floor departments and their configurations.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class Department(str, Enum):
    """Trading Floor Departments (Option B)."""
    RESEARCH = "research"
    DEVELOPMENT = "development"
    TRADING = "trading"
    RISK = "risk"
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

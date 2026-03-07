"""
QuantMind Agent Skills Package

This package provides skill implementations for trading, system, and data operations.
Each skill is a callable function with defined input/output schemas.
"""

from .base import AgentSkill
from .skill_manager import (
    SkillManager,
    Skill,
    SkillResult,
    ChainMode,
    SkillCategory,
    SKILL_CATEGORY_METADATA,
    SkillError,
    SkillNotFoundError,
    SkillExecutionError,
    SkillValidationError,
    get_skill_manager,
    set_skill_manager,
)
from .builtin_skills import (
    register_builtin_skills,
    # Research
    knowledge_search,
    extract_trading_rules,
    research_summary,
    # Trading
    calculate_position_size,
    calculate_rsi,
    detect_support_resistance,
    calculate_pivot_points,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_moving_average,
    calculate_atr,
    calculate_stochastic,
    calculate_obv,
    # Risk
    validate_risk_parameters,
    calculate_portfolio_risk,
    calculate_correlation_risk,
    calculate_kelly_criterion,
    calculate_value_at_risk,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_sortino_ratio,
    # Coding
    analyze_code_complexity,
    suggest_code_improvements,
    generate_documentation,
    # Data
    fetch_historical_data,
    fetch_live_tick,
    resample_timeframe,
    clean_data_anomalies,
    calculate_returns,
    normalize_data,
)

__all__ = [
    # Base
    "AgentSkill",
    # Manager
    "SkillManager",
    "Skill",
    "SkillResult",
    "ChainMode",
    "SkillCategory",
    "SKILL_CATEGORY_METADATA",
    "SkillError",
    "SkillNotFoundError",
    "SkillExecutionError",
    "SkillValidationError",
    "get_skill_manager",
    "set_skill_manager",
    # Registration
    "register_builtin_skills",
    # Research Skills
    "knowledge_search",
    "extract_trading_rules",
    "research_summary",
    # Trading Skills
    "calculate_position_size",
    "calculate_rsi",
    "detect_support_resistance",
    "calculate_pivot_points",
    "calculate_macd",
    "calculate_bollinger_bands",
    "calculate_moving_average",
    "calculate_atr",
    "calculate_stochastic",
    "calculate_obv",
    # Risk Skills
    "validate_risk_parameters",
    "calculate_portfolio_risk",
    "calculate_correlation_risk",
    "calculate_kelly_criterion",
    "calculate_value_at_risk",
    "calculate_sharpe_ratio",
    "calculate_max_drawdown",
    "calculate_sortino_ratio",
    # Coding Skills
    "analyze_code_complexity",
    "suggest_code_improvements",
    "generate_documentation",
    # Data Skills
    "fetch_historical_data",
    "fetch_live_tick",
    "resample_timeframe",
    "clean_data_anomalies",
    "calculate_returns",
    "normalize_data",
]
